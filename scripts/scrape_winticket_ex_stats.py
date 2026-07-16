"""WINTICKET 条件別成績スクレイパー（G44）

選手ごとに直近レースを1件フェッチし、条件別成績（天候別・バンク別・時間帯別・位置別）を
data/player_ex_stats.csv に保存する。

Usage:
  python3 scripts/scrape_winticket_ex_stats.py           # 全選手をスクレイプ（再開可能）
  python3 scripts/scrape_winticket_ex_stats.py --stats   # 取得済みの統計表示
  python3 scripts/scrape_winticket_ex_stats.py --limit 10  # テスト用

戦略: 選手ごとの直近レースをDBから取得し、同一レースをキャッシュして
1ページで複数選手のデータを同時取得する。
"""
import sys, csv, argparse, time
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if ".claude/worktrees" in str(_REPO_ROOT):
    _REPO_ROOT = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(_REPO_ROOT))

from src.database import get_connection
from src.scraper.winticket import WinticketScraper, VENUE_SLUGS, _extract_state, _get_query

OUTPUT = _REPO_ROOT / "data" / "player_ex_stats.csv"

# WINTICKET JSON キー → CSV 列名プレフィックスのマッピング
CONDITION_FIELDS = {
    "weather_sunny":   "weatherSunny",
    "weather_cloudy":  "weatherCloudy",
    "weather_rainy":   "weatherRainy",
    "track_333":       "trackDistance333",
    "track_400":       "trackDistance400",
    "track_500":       "trackDistance500",
    "hour_normal":     "hourTypeNormal",
    "hour_morning":    "hourTypeMorning",
    "hour_night":      "hourTypeNight",
    "hour_midnight":   "hourTypeMidnight",
    "pos_first":       "linePositionFirst",
    "pos_second":      "linePositionSecond",
    "pos_third":       "linePositionThird",
    "pos_single":      "lineSingleHorseman",
    "pos_compete":     "lineCompete",
}

COLS = ["player_id"] + [f"{k}_top3_pct" for k in CONDITION_FIELDS]


def _top3_pct(rec: dict, json_key: str) -> float | None:
    """records_raw の条件別成績エントリから top3 率(%) を計算する。"""
    entry = rec.get(json_key)
    if not entry or not isinstance(entry, dict):
        return None
    total = entry.get("total", 0)
    if not total:
        return None
    f = entry.get("first", 0) or 0
    s = entry.get("second", 0) or 0
    t = entry.get("third", 0) or 0
    return round((f + s + t) / total * 100, 2)


def get_player_race_map() -> dict[int, tuple[str, str, int, int]]:
    """選手ID → (venue_id, cup_id, day_index, race_no) の直近レース情報を返す。"""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT e.player_id, r.venue_id, r.cup_id, r.day_index, r.race_no,
                   r.race_date
            FROM wt_entries e
            JOIN wt_races r ON e.race_key = r.race_key
            WHERE r.race_date >= '2026-01-01'
            ORDER BY r.race_date DESC
        """).fetchall()

    # 選手ごとに最新レースだけ保持（最初に出てきたものが最新・ORDER BY DESC）
    seen: dict[int, tuple] = {}
    for row in rows:
        pid = row["player_id"]
        if pid not in seen:
            seen[pid] = (
                row["venue_id"], row["cup_id"], row["day_index"], row["race_no"]
            )
    return seen


def load_done() -> set[int]:
    """CSV 取得済みの player_id セットを返す。"""
    if not OUTPUT.exists():
        return set()
    done = set()
    with open(OUTPUT, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["player_id"]))
            except (ValueError, KeyError):
                pass
    return done


def scrape_race(scraper: WinticketScraper, venue_id: str,
                cup_id: str, day_index: int, race_no: int
                ) -> dict[int, dict] | None:
    """1レースページをフェッチし、{player_id: {col: value}} を返す。"""
    slug = VENUE_SLUGS.get(venue_id)
    if not slug:
        return None

    from src.scraper.winticket import _BASE
    url = f"{_BASE}/keirin/{slug}/racecard/{cup_id}/{day_index}/{race_no}"
    resp = scraper._get(url)
    if resp is None or resp.status_code != 200:
        return None

    state = _extract_state(resp.text)
    data = _get_query(state, "FETCH_KEIRIN_RACE")
    if not data:
        return None

    records_raw = {r["playerId"]: r for r in data.get("records", [])}
    result = {}
    for pid_str, rec in records_raw.items():
        try:
            pid = int(pid_str)
        except (ValueError, TypeError):
            continue
        row = {"player_id": pid}
        for col, json_key in CONDITION_FIELDS.items():
            row[f"{col}_top3_pct"] = _top3_pct(rec, json_key)
        result[pid] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="最大処理選手数（テスト用）")
    parser.add_argument("--stats", action="store_true", help="取得済みの統計を表示して終了")
    args = parser.parse_args()

    if args.stats:
        if not OUTPUT.exists():
            print("データファイルが存在しません。スクレイプを先に実行してください。")
            return
        import csv as _csv
        rows = []
        with open(OUTPUT, newline="") as f:
            rows = list(_csv.DictReader(f))
        print(f"取得済み選手数 : {len(rows)}")
        for col in COLS[1:]:
            cnt = sum(1 for r in rows if r.get(col) not in (None, "", "None"))
            print(f"  {col:<35}: {cnt:5d} / {len(rows)} ({cnt/max(1,len(rows))*100:.1f}%)")
        return

    player_race_map = get_player_race_map()
    done = load_done()

    targets = [pid for pid in player_race_map if pid not in done]
    if args.limit:
        targets = targets[: args.limit]

    if not targets:
        print("全選手取得済み。--stats で確認してください。")
        return

    print(f"Total: {len(player_race_map)}  Done: {len(done)}  Remaining: {len(targets)}")

    # レース単位でグループ化（1フェッチで複数選手を取得）
    race_to_players: dict[tuple, list[int]] = defaultdict(list)
    for pid in targets:
        key = player_race_map[pid]  # (venue_id, cup_id, day_index, race_no)
        race_to_players[key].append(pid)

    write_header = not OUTPUT.exists()
    scraper = WinticketScraper(request_interval=1.5)
    done_new: dict[int, dict] = {}
    fetched = 0

    for (venue_id, cup_id, day_index, race_no), pids in race_to_players.items():
        if all(p in done_new for p in pids):
            continue

        result = scrape_race(scraper, venue_id, cup_id, day_index, race_no)
        fetched += 1

        if result:
            for pid in pids:
                if pid in result:
                    done_new[pid] = result[pid]
                else:
                    # このレースのrecords_rawに含まれない場合（欠場等）
                    done_new[pid] = {"player_id": pid, **{c: None for c in COLS[1:]}}

        total_done = len(done) + len(done_new)
        total_target = len(done) + len(targets)
        print(f"\r  races={fetched}/{len(race_to_players)}  players={total_done}/{total_target}", end="", flush=True)

        if args.limit and len(done_new) >= args.limit:
            break

    print()

    # CSV に追記
    with open(OUTPUT, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if write_header and OUTPUT.stat().st_size == 0:
            writer.writeheader()
        for row in done_new.values():
            writer.writerow(row)

    print(f"Done. {len(done_new)} 選手を追記しました。 --stats で確認できます。")


if __name__ == "__main__":
    main()
