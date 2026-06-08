"""winticket収集の欠損チェック（ks races と突合）

収集期間内で、ks `races` に開催記録があるのに wt_races に無い「会場×日」を検出。
瞬断等で取りこぼした会場日を洗い出し、--recollect で再収集する。

使い方:
  PYTHONPATH=. .venv/bin/python3 scripts/gap_check_wt.py            # 検出のみ
  PYTHONPATH=. .venv/bin/python3 scripts/gap_check_wt.py --recollect # 欠損日を再収集
"""
import argparse
from src.database import get_connection
from src.scraper.winticket import VENUE_SLUGS

ap = argparse.ArgumentParser()
ap.add_argument("--recollect", action="store_true", help="欠損日を collect-wt で再収集")
args = ap.parse_args()

with get_connection() as conn:
    wt_min, wt_max = conn.execute("SELECT MIN(race_date), MAX(race_date) FROM wt_races").fetchone()
    # ks側: winticket対象会場かつ wt収集期間内の (date, venue)
    ks = conn.execute("""
        SELECT DISTINCT race_date, venue_code FROM races
        WHERE race_date >= ? AND race_date <= ?
    """, (wt_min, wt_max)).fetchall()
    wt = {(r[0], r[1]) for r in conn.execute("""
        SELECT DISTINCT race_date, venue_id FROM wt_races
        WHERE race_date >= ? AND race_date <= ?
    """, (wt_min, wt_max)).fetchall()}

print(f"wt収集期間: {wt_min} 〜 {wt_max}")
ks_target = [(d, v) for d, v in ks if v in VENUE_SLUGS]
missing = sorted({(d, v) for d, v in ks_target} - wt)
print(f"ks対象(会場×日): {len(ks_target)}  wt収集済: {len(wt)}  欠損: {len(missing)}")

if missing:
    # 欠損日ごとに会場をまとめて表示
    from collections import defaultdict
    by_date = defaultdict(list)
    for d, v in missing:
        by_date[d].append(v)
    print(f"\n欠損のある日数: {len(by_date)}")
    for d in sorted(by_date)[:40]:
        print(f"  {d}: {sorted(by_date[d])}")
    if len(by_date) > 40:
        print(f"  ... 他 {len(by_date)-40} 日")

    if args.recollect:
        from src.scraper.pipeline_wt import WinticketPipeline
        from src.database import init_db
        init_db()
        pipe = WinticketPipeline()
        print(f"\n=== 再収集開始（{len(by_date)}日）===")
        for i, d in enumerate(sorted(by_date), 1):
            stats = pipe.collect_date(d)
            print(f"[{i}/{len(by_date)}] {d}: {stats}")
        print("再収集完了。再度このスクリプトで残欠損を確認してください。")
else:
    print("\n✅ 欠損なし。収集は完全です。")
