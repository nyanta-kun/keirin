"""検証B: モデル vs 市場の不一致（2026-07-15）

モデルの上位2車（安定2車）と、市場のワイド最有力ペア（quinellaPlace 最安オッズの2車）を
比較する。両者が一致するレース／不一致のレースで hit2・ROI を測り、
「モデルが市場より正しい（＝割高な安定2車）」不一致サブセットがあるか検証する。

前提: exp_stable_top2_wt.py のキャッシュ（クリーンレース・落車失格欠車除外済）
窓: DISCOVER 03-01〜05-31 / CONFIRM 06-01〜07-10

使い方: .venv/bin/python scripts/exp_value_B_wt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from exp_stable_top2_wt import load_odds_maps, seg, DISC, CONF, CACHE_DIR  # noqa: E402


def enrich(df):
    wide, _ = load_odds_maps(df["race_key"].unique().tolist())
    df = df.copy()
    model_od, mkt_od, agree, mkt_hit = [], [], [], []
    for rk, a1, a2, top3 in zip(df["race_key"], df["a1"], df["a2"], df["top3"]):
        w = wide.get(rk, {})
        model_pair = frozenset({a1, a2})
        mo = w.get(model_pair)
        model_od.append(mo)
        if w:
            mkt_pair = min(w, key=w.get)   # 最安ワイド = 市場最有力ペア
            mkt_od.append(w[mkt_pair])
            agree.append(int(mkt_pair == model_pair))
            t3 = frozenset(top3)
            mkt_hit.append(int(mkt_pair <= t3))  # 市場ペアが両方3着内
        else:
            mkt_od.append(None); agree.append(None); mkt_hit.append(None)
    df["model_od"] = model_od
    df["mkt_od"] = mkt_od
    df["agree"] = agree
    df["mkt_hit2"] = mkt_hit
    return df[df["model_od"].notna() & df["mkt_od"].notna()].copy()


def roi_wide(d, od_col, hit_col):
    b = len(d) * 100
    p = int((d[hit_col] * d[od_col] * 100).sum())
    return len(d), d[hit_col].mean() if len(d) else 0, p / b if b else 0


def main():
    df = enrich(pd.read_pickle(CACHE_DIR / "stable_top2_n7.pkl"))
    print(f"7車・ワイドオッズ有り: {len(df):,}レース")

    for wl, w in (("DISCOVER", DISC), ("CONFIRM", CONF)):
        s = seg(df, w)
        days = s["race_date"].nunique() or 1
        agree = s[s["agree"] == 1]
        disag = s[s["agree"] == 0]
        print(f"\n===== {wl} （{days}日） =====")
        n, h, r = roi_wide(s, "model_od", "hit2")
        print(f"[全体] モデル2車 wide: n={n}({n/days:.1f}/日) hit2={h:.1%} ROI={r:.1%}")
        # 一致 vs 不一致
        for lab, d in (("一致(model=市場)", agree), ("不一致(model≠市場)", disag)):
            n, h, r = roi_wide(d, "model_od", "hit2")
            _, mh, mr = roi_wide(d, "mkt_od", "mkt_hit2")
            print(f"[{lab}] n={n:>4}({n/days:4.1f}/日)  "
                  f"モデル2車: hit2={h:5.1%} ROI={r:6.1%}  | "
                  f"市場2車: hit2={mh:5.1%} ROI={mr:6.1%}")
        # 不一致 × モデル信頼度でさらに分解
        print("  -- 不一致を top2_share で分解（モデル2車wide購入）--")
        for cs in (0.45, 0.50):
            d = disag[disag["top2_share"] >= cs]
            n, h, r = roi_wide(d, "model_od", "hit2")
            print(f"     top2_share>={cs}: n={n:>4}({n/days:4.1f}/日) hit2={h:5.1%} ROI={r:6.1%}")


if __name__ == "__main__":
    main()
