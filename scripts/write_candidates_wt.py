#!/usr/bin/env python3
"""朝時点の候補レースを picks_history に即時書き込む。

gap12 条件を満たす全候補を picks_history に書き込むことで、
同日中から推奨ページに候補レース（miwokuri=False, bet_amount=0）を表示できる。
翌朝 notify_results_wt.py が購入済み/見送りを正確に上書きする。

実行:
    python3 scripts/write_candidates_wt.py [YYYY-MM-DD]

daily_picks_wt.sh および evening_picks_wt.sh から wave-picks-wt の直後に呼ばれる。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")

    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    candidates: list[dict] = []
    for fname in (
        f"wave_picks_wt_{target_date}_candidates.json",
        f"wave_picks_wt_{target_date}_night_candidates.json",
    ):
        p = picks_dir / fname
        if p.exists():
            try:
                candidates += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[write_candidates_wt] {fname} 読み込み失敗: {e}", flush=True)

    if not candidates:
        print(f"[write_candidates_wt] {target_date}: candidates なし", flush=True)
        return

    rows: list[tuple] = []
    for cand in candidates:
        rk = cand.get("race_key")
        if not rk:
            continue
        gap12 = cand.get("gap12", 0.0)
        rank = "7PLUS_S" if gap12 >= 0.10 else "7PLUS_A"
        p1 = cand.get("pivot1")
        p2 = cand.get("pivot2")
        thirds = cand.get("thirds", [])
        pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
        n_combos = len(thirds)
        store_key = f"{rk}#CAND"
        rows.append((target_date, store_key, rank, pred, n_combos))

    with get_connection() as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT race_key FROM picks_history WHERE race_date=? AND route='wt'",
                (target_date,),
            ).fetchall()
        }
        inserted = 0
        for target_date_v, store_key, rank, pred, n_combos in rows:
            base = store_key.rsplit("#", 1)[0]
            # 購入済み (#7SS/#7S/#7A) が既に存在すれば候補書き込みをスキップ
            if any(k.startswith(base + "#") and k != store_key for k in existing):
                continue
            if store_key in existing:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,'wt',False)",
                (target_date_v, store_key, rank, pred, n_combos),
            )
            existing.add(store_key)
            inserted += 1

    print(
        f"[write_candidates_wt] {target_date}: {inserted}/{len(rows)} 件書き込み完了",
        flush=True,
    )


if __name__ == "__main__":
    main()
