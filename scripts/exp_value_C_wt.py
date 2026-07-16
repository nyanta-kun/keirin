"""検証C: 安定2車を軸に3列目で稼ぐ（2026-07-15）

安定2車(a1,a2)を軸に、3列目候補 t ごとの三連複レグ(a1,a2,t)を
「個別の1点」として、①3列目のモデル順位別、②レグ・オッズ帯別に
hit・ROI を測る。人気/穴どちらに妙味(favorite-longshot bias)があるか検証。

窓: DISCOVER 03-01〜05-31 / CONFIRM 06-01〜07-10（7車クリーン）
使い方: .venv/bin/python scripts/exp_value_C_wt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from exp_stable_top2_wt import load_odds_maps, seg, DISC, CONF, CACHE_DIR  # noqa: E402


def legs_table(df):
    """各レース×3列目候補を1行に展開（1点=1レグ）。"""
    _, trio = load_odds_maps(df["race_key"].unique().tolist())
    rows = []
    for _, r in df.iterrows():
        rk, a1, a2 = r["race_key"], r["a1"], r["a2"]
        won = frozenset(r["top3"])
        tb = trio.get(rk, {})
        # 3列目候補は thirds（モデル3位以下）。モデル順位= thirds内の並び順が
        # pred降順（build時 frames[2:] は pred降順）なので index0=モデル3位。
        for rank3, t in enumerate(r["thirds"], start=3):
            od = tb.get(frozenset({a1, a2, t}))
            if od is None:
                continue
            rows.append({
                "race_key": rk, "race_date": r["race_date"],
                "top2_share": r["top2_share"], "hit2": r["hit2"],
                "rank3": rank3,                       # 3列目のモデル順位(3..7)
                "leg_od": od,
                "won": int(frozenset({a1, a2, t}) == won),
            })
    return pd.DataFrame(rows)


def roi(d):
    b = len(d) * 100
    p = int((d["won"] * d["leg_od"] * 100).sum())
    return len(d), d["won"].mean() if len(d) else 0, p / b if b else 0


def main():
    df = pd.read_pickle(CACHE_DIR / "stable_top2_n7.pkl")
    L = legs_table(df)
    print(f"7車 三連複レグ総数: {len(L):,}")

    for wl, w in (("DISCOVER", DISC), ("CONFIRM", CONF)):
        s = seg(L, w)
        print(f"\n===== {wl} =====")
        # ① 3列目のモデル順位別（1点=軸2車+モデルn位）
        print("① 3列目モデル順位別（軸2車+その1点）")
        for rk3 in range(3, 8):
            d = s[s["rank3"] == rk3]
            n, h, r = roi(d)
            print(f"   モデル{rk3}位を3列目: n={n:>5} hit={h:5.1%} ROI={r:6.1%}")
        # ② レグ・オッズ帯別
        print("② レグ・オッズ帯別（全3列目候補）")
        edges = [0, 5, 10, 20, 40, 80, 200, 9999]
        for lo, hi in zip(edges[:-1], edges[1:]):
            d = s[(s["leg_od"] >= lo) & (s["leg_od"] < hi)]
            n, h, r = roi(d)
            print(f"   [{lo:>4}-{hi:>4}) n={n:>5} hit={h:5.1%} ROI={r:6.1%}")
        # ③ 高信頼(top2_share>=0.5) × レグオッズ帯
        print("③ top2_share>=0.5 × レグオッズ帯")
        hi_conf = s[s["top2_share"] >= 0.5]
        for lo, hi in zip(edges[:-1], edges[1:]):
            d = hi_conf[(hi_conf["leg_od"] >= lo) & (hi_conf["leg_od"] < hi)]
            n, h, r = roi(d)
            print(f"   [{lo:>4}-{hi:>4}) n={n:>5} hit={h:5.1%} ROI={r:6.1%}")


if __name__ == "__main__":
    main()
