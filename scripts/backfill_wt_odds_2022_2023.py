"""wt_odds の2022-12-01〜2023-12-31欠損分バックフィル（2026-07-29）。

[[keirin_s7_foundational_rethink_2026_07_29]]。外部データ取得監査で、この期間
（全レースの29.4%・29,444件）のwt_oddsが完全に欠落していることが判明した
（CLAUDE.mdの「2026-07-22に解消済み」という記載は誤り）。winticket.jpは
この期間のレースでも最終オッズページを提供し続けていることを確認済み
（2023-06-15で実データ検証済み）。

wt_races には cup_id/day_index が既にキャッシュされているため、find_cup_info
での再探索は不要。fetch_odds() を直接呼び出すだけで済む。

再開可能設計: 「wt_odds に1行も存在しないrace_key」を対象クエリの条件にしている
ため、中断後に再実行すると自動的に未処理分から再開する。

並列度は既存パイプライン(pipeline_wt.py)と同じ4（単一ドメインへの配慮）。
"""
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.scraper.winticket import WinticketScraper

DATE_FROM, DATE_TO = "2022-12-01", "2023-12-31"
MAX_WORKERS = 4
_ORDERED = {"exacta", "trifecta"}


def load_missing_races():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, venue_id, race_date, race_no, cup_id, day_index "
            "FROM wt_races WHERE race_date >= :d1 AND race_date <= :d2 AND cancel = 0 "
            "AND NOT EXISTS (SELECT 1 FROM wt_odds o WHERE o.race_key = wt_races.race_key) "
            "ORDER BY race_date, venue_id, race_no",
            {"d1": DATE_FROM, "d2": DATE_TO}).fetchall()
    return [dict(r) for r in rows]


def write_odds(race_key, odds):
    with get_connection() as c:
        for bet_type, items in (odds or {}).items():
            sep = "-" if bet_type in _ORDERED else "="
            for item in items:
                combo = item["combination"]
                if isinstance(combo, (list, tuple)):
                    combo = sep.join(str(x) for x in combo)
                c.execute(
                    "INSERT OR REPLACE INTO wt_odds (race_key, bet_type, combination, odds_value) "
                    "VALUES (?, ?, ?, ?)",
                    (race_key, bet_type, combo, item["odds_value"]))


def fetch_one(race, scraper):
    rk = race["race_key"]
    try:
        odds = scraper.fetch_odds(
            race["venue_id"], race["race_date"], race["race_no"],
            race["cup_id"], race["day_index"])
    except Exception as e:
        return rk, "error", str(e)
    if not odds or not any(odds.values()):
        return rk, "empty", None
    write_odds(rk, odds)
    return rk, "ok", None


def worker(races_chunk, worker_id):
    scraper = WinticketScraper(request_interval=2.0)
    n_ok = n_empty = n_error = 0
    for i, race in enumerate(races_chunk):
        rk, status, err = fetch_one(race, scraper)
        if status == "ok":
            n_ok += 1
        elif status == "empty":
            n_empty += 1
        else:
            n_error += 1
            print(f"[worker{worker_id}] error {rk}: {err}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"[worker{worker_id}] {i+1}/{len(races_chunk)} "
                  f"(ok={n_ok} empty={n_empty} error={n_error})", flush=True)
    return {"ok": n_ok, "empty": n_empty, "error": n_error}


def main():
    print(f"欠損レース読み込み中({DATE_FROM}〜{DATE_TO})...")
    races = load_missing_races()
    print(f"対象レース数: {len(races)}")
    if not races:
        print("欠損なし。終了。")
        return

    # venue_idでグループ化し、workerに均等分配（同一venue内は順次、venue間は並列）
    by_venue = defaultdict(list)
    for r in races:
        by_venue[r["venue_id"]].append(r)
    venues = sorted(by_venue.keys())
    chunks = [[] for _ in range(MAX_WORKERS)]
    for i, v in enumerate(venues):
        chunks[i % MAX_WORKERS].extend(by_venue[v])

    start = time.time()
    totals = {"ok": 0, "empty": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, chunk, i): i for i, chunk in enumerate(chunks) if chunk}
        for future in as_completed(futures):
            res = future.result()
            for k in totals:
                totals[k] += res[k]

    elapsed = time.time() - start
    print(f"\n完了: {totals} (所要時間 {elapsed/60:.1f}分)")

    remaining = load_missing_races()
    print(f"残存欠損: {len(remaining)}件")


if __name__ == "__main__":
    main()
