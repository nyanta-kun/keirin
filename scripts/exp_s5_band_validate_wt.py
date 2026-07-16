"""S5(2車軸三連単) gap12>=0.25 × オッズ帯 の VALIDATE 窓検証（2026-07-15）。

exp_bet_structures_sweep_wt の EXPLORE で最良だった
「S5tri_g12_25 × 40-80倍帯 = 97.0%」がバンド選択の運か実効かを、
untouched の VALIDATE 窓（2026-04-01〜2026-07-10, M2=≤2026-03-31学習）で判定する。
S5系3条件 × 全16オッズ帯を出力（EXPLORE比較用）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
)
from exp_bet_structures_sweep_wt import (
    settle_window, summarize, boot_ci, PARAMS, M2_TO, VA_FROM, VA_TO,
)


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date="2022-12-01", max_date=VA_TO)
    df = build_features_wt(raw)
    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races").fetchall())
    df["_ne"] = df["race_key"].map(ne)

    sub = df[(df["finish_order"] >= 1) & (df["race_date"] <= M2_TO)]
    m2 = lgb.LGBMClassifier(**PARAMS)
    m2.fit(sub[FEATURE_COLS_WT].fillna(0).values, sub[TARGET_COL_WT].values)

    va = df[(df["race_date"] >= VA_FROM) & (df["race_date"] <= VA_TO) & (df["_ne"] == 7)].copy()
    va["pred_prob"] = m2.predict_proba(va[FEATURE_COLS_WT].fillna(0).values)[:, 1]
    print(f"VALIDATE {va['race_key'].nunique()}R (2026-04-01〜07-10)")

    cells = settle_window(va, "VALIDATE")
    rows = [r for r in summarize(cells) if r["cond"].startswith("S5")]
    rows.sort(key=lambda r: (r["cond"], r["lo"], r["hi"]))
    print(f"\n{'cond':<16}{'odds帯':>14}{'nR':>6}{'hits':>6}{'ROI':>8}{'  95%CI':>16}")
    for r in rows:
        hi = "∞" if r["hi"] >= 1e9 else f"{r['hi']:.0f}"
        key = (r["cond"], r["lo"], r["hi"])
        pays, bets = cells[key]
        lo_ci, hi_ci = boot_ci(pays, bets)
        mark = "  ←EXPLORE最良帯" if (r["cond"] == "S5tri_g12_25" and r["lo"] == 40
                                      and r["hi"] == 80) else ""
        print(f"{r['cond']:<16}{r['lo']:>6.0f}-{hi:>7}{r['n']:>6}{r['hits']:>6}"
              f"{r['roi']:>8.1%}  [{lo_ci:.0%},{hi_ci:.0%}]{mark}")


if __name__ == "__main__":
    main()
