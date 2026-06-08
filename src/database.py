import sqlite3
from pathlib import Path
from contextlib import contextmanager


DB_PATH = Path(__file__).parent.parent / "data" / "keirin.db"

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
                bet_amount INTEGER
            )
        """)
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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")   # マルチスレッド書き込みを効率化
    conn.execute("PRAGMA synchronous = NORMAL") # WAL時に安全かつ高速
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized: {DB_PATH}")
    migrate_db()
    print("Migration applied.")
