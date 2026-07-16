"""条件総当たり探索: ROI>1 が CONFIRM で replicate する購入条件が存在するか（2026-07-15）

安定2車ワイド（1点=1レース）を対象に、
  top2_share閾値 × gap23閾値 × gap12閾値 × ワイドオッズ帯
の総当たりで、各条件「限定購入」の DISCOVER/CONFIRM ROI を出す。
DISC で ROI>=100% かつ CONF でも ROI>=100%（=replicate）する条件を探す。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from exp_stable_top2_wt import load_odds_maps, seg, DISC, CONF, CACHE_DIR  # noqa: E402


def main():
    df = pd.read_pickle(CACHE_DIR / "stable_top2_n7.pkl")
    wide, _ = load_odds_maps(df["race_key"].unique().tolist())
    df = df.copy()
    df["wide_od"] = [wide.get(rk, {}).get(frozenset({a, b}))
                     for rk, a, b in zip(df["race_key"], df["a1"], df["a2"])]
    df = df[df["wide_od"].notna()].copy()

    def roi(d):
        b = len(d) * 100
        p = int((d["hit2"] * d["wide_od"] * 100).sum())
        return len(d), (p / b if b else 0)

    disc, conf = seg(df, DISC), seg(df, CONF)

    ts_ths = [0.0, 0.40, 0.45, 0.50, 0.55]
    g23_ths = [0.0, 0.03, 0.05, 0.08]
    g12_ths = [0.0, 0.05, 0.10, 0.15]
    od_bands = [(0, 99), (1.3, 99), (1.5, 3.0), (1.5, 99), (2.0, 99), (1.5, 2.5)]

    results = []
    for ts in ts_ths:
        for g23 in g23_ths:
            for g12 in g12_ths:
                for lo, hi in od_bands:
                    def sel(d):
                        return d[(d.top2_share >= ts) & (d.gap23 >= g23)
                                 & (d.gap12 >= g12) & (d.wide_od >= lo) & (d.wide_od < hi)]
                    dn, dr = roi(sel(disc))
                    cn, cr = roi(sel(conf))
                    if cn >= 25 and dn >= 25:  # 最低サンプル
                        results.append((ts, g23, g12, lo, hi, dn, dr, cn, cr))

    res = pd.DataFrame(results, columns=["ts", "g23", "g12", "od_lo", "od_hi",
                                         "disc_n", "disc_roi", "conf_n", "conf_roi"])
    print(f"条件総数（n>=25両側）: {len(res)}")
    print("\n=== CONFIRM ROI 上位20（n>=25） ===")
    top = res.sort_values("conf_roi", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"  ts>={r.ts} g23>={r.g23} g12>={r.g12} od[{r.od_lo},{r.od_hi}) "
              f"| DISC n={int(r.disc_n):>4} ROI={r.disc_roi:6.1%} "
              f"| CONF n={int(r.conf_n):>4} ROI={r.conf_roi:6.1%}")

    both = res[(res.disc_roi >= 1.0) & (res.conf_roi >= 1.0)]
    print(f"\n=== DISC・CONF 両方 ROI>=100% の条件数: {len(both)} ===")
    for _, r in both.sort_values("conf_roi", ascending=False).iterrows():
        print(f"  ts>={r.ts} g23>={r.g23} g12>={r.g12} od[{r.od_lo},{r.od_hi}) "
              f"| DISC n={int(r.disc_n)} ROI={r.disc_roi:.1%} | CONF n={int(r.conf_n)} ROI={r.conf_roi:.1%}")

    print(f"\n参考: 全条件の CONF ROI 分布 — 中央値{res.conf_roi.median():.1%} "
          f"最大{res.conf_roi.max():.1%} 最小{res.conf_roi.min():.1%}")


if __name__ == "__main__":
    main()
