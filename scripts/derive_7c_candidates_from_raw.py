#!/usr/bin/env python3
"""既存の 7S 生候補JSONから 7C 候補を導出する（7C 導入日の当日リカバリ用）。

7C を本番投入した 2026-08-07 は、朝8:00のバッチが **7C 実装前のコードで走った**ため
`wave_picks_wt_{date}_s7c_candidates.json` が存在しない。
`wave-picks-wt` を丸ごと再実行すれば作られるが、それは他ランクの候補ファイルも
書き換えるため、**既に入稿・記録済みの 7SS/7S/7A/7B を日中に動かす危険**がある。

生候補（`_s7_raw_candidates.json`）は全7車レース分の `top3_probs` を持っており、
7C が必要とする軸・相手・選別値はすべてそこから決まる（`src/cli/main.py` の
7C ブロックと同一の式）。したがって**他のファイルに一切触れずに** 7C だけを
作れる。本スクリプトはそのためのもの。

⚠️ 恒常運用では使わない。翌日以降は朝バッチが 7C 候補を直接生成する。

    python3 scripts/derive_7c_candidates_from_raw.py 2026-08-07
    python3 scripts/derive_7c_candidates_from_raw.py 2026-08-07 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import (  # noqa: E402
    RANK_7C_LEGS_MIN, RANK_7C_P3_SUM_MIN, rank_7c_daily_select,
    rank_7c_select_axis, rank_7c_select_legs, rank_7c_unit_stake,
)

PICKS = Path(__file__).resolve().parent.parent / "data" / "picks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for suffix in ("", "_night"):
        src = PICKS / f"wave_picks_wt_{args.date}{suffix}_s7_raw_candidates.json"
        dst = PICKS / f"wave_picks_wt_{args.date}{suffix}_s7c_candidates.json"
        if not src.exists():
            print(f"[skip] {src.name} が無い")
            continue
        raw = json.loads(src.read_text(encoding="utf-8"))

        enriched = []
        for c in raw:
            probs = {int(k): float(v) for k, v in (c.get("top3_probs") or {}).items()}
            sel = rank_7c_select_axis(probs)
            if sel is None:
                continue
            a1, a2, p3sum = sel
            legs = rank_7c_select_legs(sorted(set(probs) - {a1, a2}), probs)
            d = dict(c)
            d["axis1_7c"], d["axis2_7c"] = a1, a2
            d["p3_sum_top2"] = round(p3sum, 6)
            d["legs_7c"] = legs
            enriched.append(d)

        picked = rank_7c_daily_select(enriched)
        n_sum_ok = sum(1 for c in enriched
                       if float(c["p3_sum_top2"]) >= RANK_7C_P3_SUM_MIN)
        print(f"{src.name}: 生候補{len(raw)}件 → 合計条件通過{n_sum_ok}件 "
              f"→ 相手{RANK_7C_LEGS_MIN}点以上 {len(picked)}件")
        for c in picked:
            k = len(c["legs_7c"])
            print(f"   {c['venue_name']}{c['race_no']}R {c.get('start_time','--:--')} "
                  f"軸{c['axis1_7c']}={c['axis2_7c']} 相手{c['legs_7c']} "
                  f"合計{100*c['p3_sum_top2']:.1f}% "
                  f"{k}点×{rank_7c_unit_stake(k):,}円={k*rank_7c_unit_stake(k):,}円")
        if args.dry_run:
            print("[dry-run] 書き込みなし")
            continue
        dst.write_text(json.dumps(picked, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"→ {dst}")


if __name__ == "__main__":
    main()
