"""検証A: 安定2車 × オッズ割高ゲート（2026-07-15）

安定2車（欠車前ランキング上位2車）のワイド/三連複を、モデル信頼度を固定した上で
「オッズが割高な帯」だけ買うと ROI>1 になるかを検証する。

前提データ: scripts/exp_stable_top2_wt.py が生成した車数別キャッシュ
  data/exp_cache/stable_top2_n{6,7,9}.pkl（落車/失格/欠車除外済・クリーンレース）
汚染防止窓: DISCOVER 2026-03-01〜05-31（探索）/ CONFIRM 2026-06-01〜07-10（確認）

使い方: .venv/bin/python scripts/exp_value_A_wt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from exp_stable_top2_wt import load_odds_maps, seg, DISC, CONF, CACHE_DIR  # noqa: E402


def attach_wide(df):
    wide, trio = load_odds_maps(df["race_key"].unique().tolist())
    df = df.copy()
    df["wide_od"] = [wide.get(rk, {}).get(frozenset({a, b}))
                     for rk, a, b in zip(df["race_key"], df["a1"], df["a2"])]
    # 三連複 2-全 の最安レグ・的中配当も持っておく
    trio_min, trio_pay = [], []
    for rk, a, b, thirds, top3, hit2 in zip(
            df["race_key"], df["a1"], df["a2"], df["thirds"], df["top3"], df["hit2"]):
        legs = {t: trio.get(rk, {}).get(frozenset({a, b, t})) for t in thirds}
        legs = {t: o for t, o in legs.items() if o}
        trio_min.append(min(legs.values()) if legs else None)
        pay = 0
        if legs:
            won = frozenset(top3)
            for t, o in legs.items():
                if frozenset({a, b, t}) == won:
                    pay = o
                    break
        trio_pay.append(pay)
    df["trio_min"] = trio_min
    df["trio_pay"] = trio_pay  # 的中レグの配当(倍)・外れは0
    df["trio_nlegs"] = df["thirds"].map(len)
    return df


def roi_wide(d):
    b = len(d) * 100
    p = int((d["hit2"] * d["wide_od"] * 100).sum())
    return len(d), d["hit2"].mean() if len(d) else 0, p / b if b else 0


def roi_trio(d):
    b = int((d["trio_nlegs"] * 100).sum())
    p = int((d["trio_pay"] * 100).sum())
    return len(d), (d["trio_pay"] > 0).mean() if len(d) else 0, p / b if b else 0


def bucket_report(df, size):
    df = df[df["wide_od"].notna()].copy()
    print("\n" + "=" * 84)
    print(f"===== {size}車 A: ワイドオッズ帯 × モデル信頼度 =====")
    edges = [0, 1.3, 1.6, 2.0, 3.0, 5.0, 99]
    for cs_label, cs_mask in (("全体", df["top2_share"] >= 0),
                              ("top2_share>=0.45", df["top2_share"] >= 0.45),
                              ("top2_share>=0.50", df["top2_share"] >= 0.50)):
        base = df[cs_mask]
        print(f"\n--- 信頼度: {cs_label}  (n={len(base)}) ---")
        print(f"{'ワイド帯':<12}{'│ DISC  n / hit2 / ROI':<30}{'│ CONF  n / hit2 / ROI'}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            dsel = seg(base, DISC); dsel = dsel[(dsel.wide_od >= lo) & (dsel.wide_od < hi)]
            csel = seg(base, CONF); csel = csel[(csel.wide_od >= lo) & (csel.wide_od < hi)]
            dn, dh, dr = roi_wide(dsel)
            cn, ch, cr = roi_wide(csel)
            print(f"[{lo:>4}-{hi:>4})   │ {dn:>4} / {dh:5.1%} / {dr:6.1%}       "
                  f"│ {cn:>4} / {ch:5.1%} / {cr:6.1%}")


def gate_test(df, size, cs_th, od_lo, od_hi):
    """value gate 候補: top2_share>=cs_th ∧ wide_od∈[od_lo,od_hi) の DISC/CONF ROI。"""
    d = df[(df.wide_od.notna()) & (df.top2_share >= cs_th)
           & (df.wide_od >= od_lo) & (df.wide_od < od_hi)]
    print(f"\n[GATE] {size}車 top2_share>={cs_th} ∧ wide∈[{od_lo},{od_hi})")
    for wl, w in (("DISC", DISC), ("CONF", CONF)):
        s = seg(d, w); days = s["race_date"].nunique() or 1
        n, h, r = roi_wide(s)
        print(f"  {wl}: n={n:>4} ({n/days:4.1f}/日) hit2={h:5.1%} ワイドROI={r:6.1%}")


def main():
    for size in (7, 6, 9):
        f = CACHE_DIR / f"stable_top2_n{size}.pkl"
        if not f.exists():
            continue
        df = attach_wide(pd.read_pickle(f))
        bucket_report(df, size)
    # 7車の代表的 value gate を数点
    df7 = attach_wide(pd.read_pickle(CACHE_DIR / "stable_top2_n7.pkl"))
    gate_test(df7, 7, 0.50, 1.6, 3.0)
    gate_test(df7, 7, 0.50, 2.0, 5.0)
    gate_test(df7, 7, 0.45, 2.0, 5.0)


if __name__ == "__main__":
    main()
