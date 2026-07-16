"""師匠情報スクレイパー（keirin.jp → data/player_mentors.csv）

Usage:
  python3 scripts/scrape_mentors_wt.py           # 全選手をスクレイプ（再開可能）
  python3 scripts/scrape_mentors_wt.py --stats   # 取得済みの統計表示
  python3 scripts/scrape_mentors_wt.py --limit 20  # テスト用（20人だけ）

keirin.jp の選手詳細ページ（静的HTML）から師匠の snum を取得。
snum は player_id をゼロ埋め6桁にしたもの（例: player_id=14258 → snum=014258）。
robots.txt は /pc/ を ALLOW。
"""
import sys, re, time, csv, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from src.database import get_connection

OUTPUT = Path("data/player_mentors.csv")
BASE_URL = "https://keirin.jp/pc/racerprofile?snum={snum}"
SLEEP = 0.5  # 2 req/sec


def get_player_ids() -> list[int]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT player_id FROM wt_entries ORDER BY player_id"
        ).fetchall()
    return [r[0] for r in rows]


def scrape_mentor(player_id: int, session: requests.Session) -> int | None:
    """師匠の player_id を返す。師匠なし or エラー時は None。"""
    snum = f"{player_id:06d}"
    try:
        r = session.get(BASE_URL.format(snum=snum), timeout=10)
        if r.status_code != 200:
            return None
        html = r.text
        idx = html.find("師匠")
        if idx == -1:
            return None
        # 師匠セクション直後の snum リンクを取得
        snippet = html[idx : idx + 600]
        m = re.search(r"snum=(\d{6})", snippet)
        if m:
            return int(m.group(1))
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        if not OUTPUT.exists():
            print("No data yet. Run without --stats first.")
            return
        rows = list(csv.DictReader(open(OUTPUT, encoding="utf-8")))
        n = len(rows)
        has = sum(1 for r in rows if r.get("mentor_id", "").strip())
        print(f"Scraped : {n}")
        print(f"HasMentor: {has}  ({100 * has / n:.1f}%)")
        return

    all_ids = get_player_ids()
    if args.limit:
        all_ids = all_ids[: args.limit]

    # 既取得の player_id をスキップ（再開可能）
    done: set[int] = set()
    if OUTPUT.exists():
        for row in csv.DictReader(open(OUTPUT, encoding="utf-8")):
            done.add(int(row["player_id"]))

    todo = [p for p in all_ids if p not in done]
    print(f"Total: {len(all_ids)}  Done: {len(done)}  Remaining: {len(todo)}")

    if not todo:
        print("All done. Run with --stats for coverage summary.")
        return

    write_header = not OUTPUT.exists()
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )

    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["player_id", "mentor_id"])

        for i, pid in enumerate(todo):
            mentor_id = scrape_mentor(pid, session)
            writer.writerow([pid, mentor_id if mentor_id is not None else ""])
            f.flush()
            if (i + 1) % 100 == 0 or (i + 1) == len(todo):
                print(f"  {i + 1}/{len(todo)}", flush=True)
            time.sleep(SLEEP)

    print("Done. Run with --stats for coverage summary.")


if __name__ == "__main__":
    main()
