import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(__file__).parent.parent / "data" / "keirin.db"

# ---------------------------------------------------------------------------
# PostgreSQL 互換レイヤー
# KEIRIN_DB_URL 環境変数が設定されている場合は psycopg2 で VPS に接続する。
# 例: postgresql://user:pass@vps-host:5432/keiba
# ---------------------------------------------------------------------------

# INSERT OR REPLACE / INSERT OR IGNORE の ON CONFLICT 先
_PG_CONFLICT_COLS: dict[str, tuple[str, ...]] = {
    "wt_races":         ("race_key",),
    "wt_entries":       ("race_key", "frame_no"),
    "wt_odds":          ("race_key", "bet_type", "combination"),
    "wt_odds_snapshot": ("race_key", "bet_type", "combination", "snapshot_type"),
    "wt_weather":       ("venue_id", "dt_hour"),
    "venue_info":       ("venue_code",),
    "picks_history":    ("race_key",),
    "races":            ("race_key",),
    "odds":             ("race_key", "bet_type", "combination"),
    "race_entries":     ("race_key", "frame_no"),
    "race_results":     ("race_key", "player_id"),
    "venues":           ("code",),
    "players":          ("player_id",),
    "model_evaluation": ("model_name", "period_type"),
}

# スキップする SQLite 固有ステートメントのパターン
_PG_SKIP_RE = re.compile(
    r"^\s*(?:PRAGMA\b|CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+IF\s+NOT\s+EXISTS\b"
    r"|ALTER\s+TABLE\s+\S+\s+ADD\s+COLUMN\b)",
    re.IGNORECASE,
)

_INSERT_OR_RE = re.compile(
    r"INSERT\s+OR\s+(REPLACE|IGNORE)\s+INTO\s+(\w+)\s*\(([^)]+)\)(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _pg_translate(sql: str, params: tuple | list | dict) -> tuple[str | None, object]:
    """SQLite SQL を PostgreSQL 用に変換する。スキップすべき場合は (None, None) を返す。"""
    sql = sql.strip()
    if _PG_SKIP_RE.match(sql):
        return None, None

    m = _INSERT_OR_RE.match(sql)
    if m:
        action = m.group(1).upper()
        table = m.group(2).lower()
        cols_str = m.group(3)
        rest = m.group(4).strip()

        cols = [c.strip() for c in cols_str.split(",")]
        # ? → %s, :name → %(name)s
        rest = re.sub(r":(\w+)", r"%(\1)s", rest)
        rest = rest.replace("?", "%s")
        # datetime('now') → NOW()
        rest = re.sub(r"datetime\('now'\)", "NOW()", rest, flags=re.IGNORECASE)
        # INSERT ... SELECT の SELECT 本体にもスキーマを付与する
        # （snapshot_morning_odds_wt.py の INSERT OR IGNORE ... SELECT FROM wt_odds が
        #   未変換で relation "wt_odds" does not exist になっていた・2026-07-16 修正）
        rest = re.sub(r"(?<!\w)(?:keirin\.)?(wt_races|wt_entries|wt_odds_snapshot|wt_odds"
                      r"|wt_weather|venue_info|picks_history|model_evaluation)\b",
                      r"keirin.\1", rest, flags=re.IGNORECASE)

        if action == "IGNORE":
            sql = f"INSERT INTO keirin.{table} ({', '.join(cols)}) {rest} ON CONFLICT DO NOTHING"
        else:
            conflict = _PG_CONFLICT_COLS.get(table)
            if not conflict:
                sql = f"INSERT INTO keirin.{table} ({', '.join(cols)}) {rest}"
            else:
                lc_conflict = {c.lower() for c in conflict}
                non_conf = [c for c in cols if c.lower() not in lc_conflict]
                if non_conf:
                    upd = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_conf)
                    sql = (
                        f"INSERT INTO keirin.{table} ({', '.join(cols)}) {rest} "
                        f"ON CONFLICT ({', '.join(conflict)}) DO UPDATE SET {upd}"
                    )
                else:
                    sql = (
                        f"INSERT INTO keirin.{table} ({', '.join(cols)}) {rest} "
                        f"ON CONFLICT ({', '.join(conflict)}) DO NOTHING"
                    )
        return sql, params

    # 通常の SQL: keirin スキーマ付与 + パラメータ変換
    # テーブル名に keirin. プレフィックスを付ける（既についている場合はスキップ）
    sql = re.sub(r"(?<!\w)(?:keirin\.)?(wt_races|wt_entries|wt_odds_snapshot|wt_odds"
                 r"|wt_weather|venue_info|picks_history|model_evaluation)\b",
                 r"keirin.\1", sql, flags=re.IGNORECASE)
    # psycopg2 は % をフォーマット文字として扱う。
    # LIKE '7PLUS%' 等リテラル % を先に %% にエスケープしてから :name / ? を変換する。
    sql = sql.replace("%", "%%")
    sql = re.sub(r":(\w+)", r"%(\1)s", sql)
    sql = sql.replace("?", "%s")
    sql = re.sub(r"datetime\('now'\)", "NOW()", sql, flags=re.IGNORECASE)
    return sql, params


class _PgRow:
    """psycopg2 RealDictRow を sqlite3.Row 互換にラップする。

    PostgreSQL の集約関数（SUM, COUNT 等）は列名が重複する場合がある（例: SUM(*) が
    複数あると全て "sum" になる）。dict() 変換すると重複キーが上書きされるため、
    インデックスアクセス(r[0], r[1], ...)は元の値リストから取得する。
    """

    def __init__(self, mapping: dict, keys: list[str], values: list | None = None) -> None:
        self._m = mapping
        self._k = keys
        self._v = values  # 列順を保持した値リスト（重複キー対応）

    def __getitem__(self, key):
        if isinstance(key, int):
            # 重複キーがある場合でも正しい位置の値を返す
            if self._v is not None:
                return self._v[key]
            return self._m[self._k[key]]
        return self._m[key]

    def __contains__(self, key: str) -> bool:
        return key in self._m

    def keys(self):
        return self._k


class _PgCursor:
    """psycopg2 カーソルを sqlite3 カーソル互換にラップする。"""

    def __init__(self, cur):
        self._cur = cur

    def _make_row(self, raw_row) -> "_PgRow":
        keys = [d[0] for d in self._cur.description]
        values = list(raw_row.values()) if hasattr(raw_row, "values") else list(raw_row)
        return _PgRow(dict(raw_row), keys, values)

    def _rows(self, raw_rows):
        if self._cur is None or self._cur.description is None:
            return []
        return [self._make_row(r) for r in raw_rows]

    def fetchall(self):
        if self._cur is None:
            return []
        return self._rows(self._cur.fetchall())

    def fetchone(self):
        if self._cur is None:
            return None
        row = self._cur.fetchone()
        if row is None:
            return None
        return self._make_row(row)

    @property
    def rowcount(self) -> int:
        if self._cur is None:
            return 0
        return self._cur.rowcount

    def __iter__(self):
        return iter(self.fetchall())


class _PgRawCursor:
    """pandas.read_sql_query 向け DBAPI2 互換カーソル（クエリ変換付き）。"""

    def __init__(self, raw_cur) -> None:
        self._cur = raw_cur

    def execute(self, sql: str, params=None):
        translated, translated_params = _pg_translate(sql, params or ())
        if translated is not None:
            self._cur.execute(translated, translated_params or None)
        return self

    def fetchall(self):
        return self._cur.fetchall()

    def fetchmany(self, size=None):
        return self._cur.fetchmany(size) if size is not None else self._cur.fetchmany()

    def fetchone(self):
        return self._cur.fetchone()

    @property
    def description(self):
        return self._cur.description

    @property
    def rowcount(self):
        return self._cur.rowcount

    def close(self):
        self._cur.close()

    def __iter__(self):
        return iter(self.fetchall())


class _PgConn:
    """psycopg2 接続を sqlite3.Connection 互換にラップする。"""

    def __init__(self, conn) -> None:
        self._conn = conn

    def execute(self, sql: str, params=()) -> _PgCursor:
        translated, translated_params = _pg_translate(sql, params)
        if translated is None:
            return _PgCursor(None)
        cur = self._conn.cursor()
        cur.execute(translated, translated_params or None)
        return _PgCursor(cur)

    def executemany(self, sql: str, params_list) -> None:
        translated, _ = _pg_translate(sql, ())
        if translated is None or not params_list:
            return
        cur = self._conn.cursor()
        cur.executemany(translated, params_list)

    def executescript(self, sql: str) -> None:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    self.execute(stmt)
                    self._conn.commit()
                except Exception:
                    try:
                        self._conn.rollback()
                    except Exception:
                        pass

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def cursor(self) -> _PgRawCursor:
        """pandas.read_sql_query など DBAPI2 直アクセス向け。"""
        return _PgRawCursor(self._conn.cursor())

    # sqlite3 互換: row_factory 属性は無視
    @property
    def row_factory(self):
        return None

    @row_factory.setter
    def row_factory(self, _):
        pass

VENUE_STATIC = {
    "11": ("函館", 333, 0, "北海道"),
    "12": ("青森", 333, 0, "青森"),
    "13": ("いわき平", 333, 0, "福島"),
    "14": ("会津", 333, 0, "福島"),
    "15": ("八戸", 333, 0, "青森"),
    "16": ("六郷", 333, 0, "秋田"),
    "17": ("宮城", 333, 0, "宮城"),
    "21": ("弥彦", 400, 0, "新潟"),
    "22": ("前橋", 333, 0, "群馬"),
    "23": ("取手", 400, 0, "茨城"),
    "24": ("宇都宮", 333, 0, "栃木"),
    "25": ("大宮", 400, 0, "埼玉"),
    "26": ("西武園", 400, 0, "埼玉"),
    "27": ("京王閣", 400, 0, "東京"),
    "28": ("立川", 500, 0, "東京"),
    "31": ("松戸", 333, 0, "千葉"),
    "32": ("千葉", 250, 1, "千葉"),
    "34": ("川崎", 333, 0, "神奈川"),
    "35": ("平塚", 500, 0, "神奈川"),
    "36": ("小田原", 333, 0, "神奈川"),
    "37": ("伊東", 333, 0, "静岡"),
    "38": ("静岡", 500, 0, "静岡"),
    "41": ("一宮", 400, 0, "愛知"),
    "42": ("名古屋", 400, 0, "愛知"),
    "43": ("岐阜", 400, 0, "岐阜"),
    "44": ("大垣", 400, 0, "岐阜"),
    "45": ("豊橋", 400, 0, "愛知"),
    "46": ("富山", 400, 0, "富山"),
    "47": ("松阪", 333, 0, "三重"),
    "48": ("四日市", 400, 0, "三重"),
    "51": ("福井", 400, 0, "福井"),
    "52": ("大津", 333, 0, "滋賀"),
    "53": ("奈良", 400, 0, "奈良"),
    "54": ("向日町", 333, 0, "京都"),
    "55": ("和歌山", 400, 0, "和歌山"),
    "56": ("岸和田", 333, 0, "大阪"),
    "57": ("大阪", 333, 0, "大阪"),
    "61": ("玉野", 400, 0, "岡山"),
    "62": ("広島", 333, 0, "広島"),
    "63": ("防府", 333, 0, "山口"),
    "64": ("松江", 333, 0, "島根"),
    "71": ("高松", 333, 0, "香川"),
    "72": ("観音寺", 333, 0, "香川"),
    "73": ("小松島", 333, 0, "徳島"),
    "74": ("高知", 333, 0, "高知"),
    "75": ("松山", 333, 0, "愛媛"),
    "81": ("小倉", 400, 0, "福岡"),
    "82": ("門司", 400, 0, "福岡"),
    "83": ("久留米", 333, 0, "福岡"),
    "84": ("武雄", 400, 0, "佐賀"),
    "85": ("佐世保", 333, 0, "長崎"),
    "86": ("別府", 400, 0, "大分"),
    "87": ("熊本", 400, 0, "熊本"),
    "88": ("長崎", 333, 0, "長崎"),
    "89": ("福岡", 400, 0, "福岡"),
}


def init_db():
    """データベースの初期化（テーブル作成）"""
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS venues (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            prefecture TEXT
        );

        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            player_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            name_kana TEXT,
            prefecture TEXT,
            registration_grade TEXT,
            birth_year INTEGER,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS races (
            id INTEGER PRIMARY KEY,
            race_key TEXT NOT NULL UNIQUE,
            venue_code TEXT NOT NULL,
            race_date TEXT NOT NULL,
            race_no INTEGER NOT NULL,
            grade TEXT,
            distance INTEGER,
            weather TEXT,
            track_condition TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS race_entries (
            id INTEGER PRIMARY KEY,
            race_key TEXT NOT NULL,
            player_id TEXT NOT NULL,
            frame_no INTEGER NOT NULL,
            line_position TEXT,
            line_group INTEGER,
            gear_ratio REAL,
            racing_score REAL,
            power_rank INTEGER,
            recent_win_rate_3m REAL,
            recent_win_rate_6m REAL,
            recent_top3_rate_3m REAL,
            recent_top3_rate_6m REAL,
            venue_win_rate REAL,
            days_since_last_race INTEGER,
            quinella_rate REAL,
            period INTEGER,
            prefecture TEXT,
            player_class TEXT,
            UNIQUE(race_key, frame_no)
        );

        CREATE TABLE IF NOT EXISTS race_results (
            id INTEGER PRIMARY KEY,
            race_key TEXT NOT NULL,
            player_id TEXT NOT NULL,
            frame_no INTEGER NOT NULL,
            finish_position INTEGER,
            finish_time REAL,
            UNIQUE(race_key, player_id)
        );

        CREATE TABLE IF NOT EXISTS odds (
            id INTEGER PRIMARY KEY,
            race_key TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            combination TEXT NOT NULL,
            odds_value REAL,
            payout INTEGER,
            collected_at TEXT DEFAULT (datetime('now')),
            UNIQUE(race_key, bet_type, combination)
        );

        CREATE TABLE IF NOT EXISTS venue_info (
            venue_code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            bank_length INTEGER,
            is_indoor INTEGER DEFAULT 0,
            prefecture TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);
        CREATE INDEX IF NOT EXISTS idx_race_entries_race ON race_entries(race_key);
        CREATE INDEX IF NOT EXISTS idx_race_results_race ON race_results(race_key);
        CREATE INDEX IF NOT EXISTS idx_race_results_player ON race_results(player_id);
        CREATE INDEX IF NOT EXISTS idx_odds_race ON odds(race_key);
        """)
    migrate_db()


def migrate_db():
    """既存DBへの安全なスキーマ追加（ALTER TABLE / CREATE IF NOT EXISTS）"""
    new_columns = [
        ("quinella_rate", "REAL"),
        ("period", "INTEGER"),
        ("prefecture", "TEXT"),
        ("player_class", "TEXT"),
    ]
    with get_connection() as conn:
        for col_name, col_type in new_columns:
            try:
                conn.execute(
                    f"ALTER TABLE race_entries ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError:
                pass  # column already exists

        try:
            conn.execute("ALTER TABLE races ADD COLUMN start_time TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

        conn.execute("""
            CREATE TABLE IF NOT EXISTS picks_history (
                id INTEGER PRIMARY KEY,
                race_date TEXT NOT NULL,
                race_key TEXT NOT NULL UNIQUE,
                rank TEXT NOT NULL,
                pred_combo TEXT,
                n_combos INTEGER,
                hit INTEGER DEFAULT 0,
                payout INTEGER DEFAULT 0,
                bet_amount INTEGER,
                prerace_gami REAL
            )
        """)
        try:
            conn.execute("ALTER TABLE picks_history ADD COLUMN prerace_gami REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            conn.execute("ALTER TABLE picks_history ADD COLUMN trifecta_payout INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column already exists
        for _col in ("gap12 REAL", "gap34 REAL", "gap23 REAL"):
            # gap23 のみ pt(%ポイント)スケール。gap12/gap34 は 0-1 スケール（歴史的経緯・変更不可）
            try:
                conn.execute(f"ALTER TABLE picks_history ADD COLUMN {_col}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_picks_history_date ON picks_history(race_date)"
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS venue_info (
                venue_code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                bank_length INTEGER,
                is_indoor INTEGER DEFAULT 0,
                prefecture TEXT
            )
        """)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_race_results_player ON race_results(player_id)"
        )

        # winticket 専用テーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wt_races (
                race_key  TEXT PRIMARY KEY,
                venue_id  TEXT NOT NULL,
                race_date TEXT NOT NULL,
                race_no   INTEGER NOT NULL,
                cup_id    TEXT NOT NULL,
                day_index INTEGER NOT NULL,
                grade     TEXT,
                race_type TEXT,
                distance  INTEGER,
                n_entries INTEGER,
                start_at  TEXT,
                status    INTEGER DEFAULT 0,
                cancel    INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wt_entries (
                id              INTEGER PRIMARY KEY,
                race_key        TEXT NOT NULL,
                frame_no        INTEGER NOT NULL,
                player_id       INTEGER,
                name            TEXT,
                prefecture      TEXT,
                player_class    TEXT,
                term            INTEGER,
                gear_ratio      REAL,
                style           TEXT,
                race_point      REAL,
                comment         TEXT,
                prediction_mark INTEGER,
                s_count         INTEGER,
                h_count         INTEGER,
                b_count         INTEGER,
                front_runner    INTEGER,
                stalker         INTEGER,
                deep_closer     INTEGER,
                marker          INTEGER,
                first_rate      REAL,
                second_rate     REAL,
                third_rate      REAL,
                ex_spurt_pct    REAL,
                ex_thrust_pct   REAL,
                ex_left_behind_pct REAL,
                ex_split_line_pct  REAL,
                ex_snatch_pct   REAL,
                line_group      INTEGER,
                line_size       INTEGER,
                line_pos        INTEGER,
                is_line_leader  INTEGER,
                n_lines         INTEGER,
                finish_order    INTEGER,
                factor          TEXT,
                UNIQUE(race_key, frame_no)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wt_odds (
                id           INTEGER PRIMARY KEY,
                race_key     TEXT NOT NULL,
                bet_type     TEXT NOT NULL,
                combination  TEXT NOT NULL,
                odds_value   REAL,
                collected_at TEXT DEFAULT (datetime('now')),
                UNIQUE(race_key, bet_type, combination)
            )
        """)
        # 朝オッズ前向き計測用スナップショット。
        # wt_odds は INSERT OR REPLACE で最終オッズに上書きされるため、
        # 朝7:00時点のオッズを別テーブルへ退避し、後日 wt_odds(最終) と突合して
        # 朝→直前のドリフトを計測する。snapshot_type 単位で初回値を保持（OR IGNORE）。
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wt_odds_snapshot (
                id            INTEGER PRIMARY KEY,
                race_key      TEXT NOT NULL,
                bet_type      TEXT NOT NULL,
                combination   TEXT NOT NULL,
                odds_value    REAL,
                snapshot_type TEXT NOT NULL DEFAULT 'morning',
                snapshot_at   TEXT DEFAULT (datetime('now')),
                UNIQUE(race_key, bet_type, combination, snapshot_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wt_races_date   ON wt_races(race_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wt_entries_race ON wt_entries(race_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wt_odds_race    ON wt_odds(race_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wt_odds_snap_race ON wt_odds_snapshot(race_key)")

        for code, (name, bank_length, is_indoor, prefecture) in VENUE_STATIC.items():
            conn.execute(
                "INSERT OR IGNORE INTO venue_info "
                "(venue_code, name, bank_length, is_indoor, prefecture) "
                "VALUES (?, ?, ?, ?, ?)",
                (code, name, bank_length, is_indoor, prefecture),
            )


@contextmanager
def get_connection():
    """DB 接続を返す。

    KEIRIN_DB_URL 環境変数が設定されている場合は VPS PostgreSQL に接続する。
    未設定の場合は従来どおりローカル SQLite を使用する。

    例:
        export KEIRIN_DB_URL="postgresql://keirin_app:pass@vps-host:5432/keiba"
    """
    db_url = os.environ.get("KEIRIN_DB_URL", "")
    if db_url:
        import psycopg2
        import psycopg2.extras

        raw = psycopg2.connect(
            db_url,
            cursor_factory=psycopg2.extras.RealDictCursor,
            connect_timeout=10,
        )
        raw.autocommit = False
        conn = _PgConn(raw)
        try:
            yield conn
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(str(DB_PATH), timeout=30)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA journal_mode = WAL")
        raw.execute("PRAGMA synchronous = NORMAL")
        raw.execute("PRAGMA foreign_keys = ON")
        try:
            yield raw
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized: {DB_PATH}")
    migrate_db()
    print("Migration applied.")
