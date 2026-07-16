"""検証: 我々の頭予想 × WINTICKET◎ の一致/不一致 別 二車単・三連単 的中率/ROI（2026-07-15）

波乱傾向レース(rp_std下位=拮抗)で、我々のモデル1位を「頭(1着)」に据えた
二車単・三連単の的中率・ROIを、WINTICKET◎(prediction_mark==1)との一致/不一致で分ける。
「◎と重ならないレース限定」の販売価値を評価する。

前提: perhorse_n7.pkl + wt_entries.prediction_mark
窓: DISCOVER 03-01〜05-31 / CONFIRM 06-01〜07-10
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.database import get_connection  # noqa: E402
from exp_stable_top2_wt import seg, DISC, CONF, CACHE_DIR  # noqa: E402

PH = CACHE_DIR / "perhorse_n7.pkl"


def load_ordered(race_keys):
    exa, tri = {}, {}
    rks = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(rks), 900):
            chunk = rks[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('exacta','trifecta') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                if od is None or not (0 < float(od) < 90000):
                    continue
                try:
                    parts = tuple(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if bt == "exacta" and len(parts) == 2:
                    exa.setdefault(rk, {})[parts] = float(od)
                elif bt == "trifecta" and len(parts) == 3:
                    tri.setdefault(rk, {})[parts] = float(od)
    return exa, tri


def load_marks(race_keys):
    mark = defaultdict(dict)
    rks = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(rks), 900):
            chunk = rks[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fr, m in c.execute(q, chunk):
                mark[rk][int(fr)] = m
    return mark


def build(ph, mark):
    recs = []
    for rk, g in ph.groupby("race_key"):
        g = g.sort_values("model_rank")
        if len(g) != 7:
            continue
        frames = g["frame_no"].astype(int).tolist()
        rp = g["race_point"].astype(float).to_numpy()
        fo = {int(f): (int(o) if pd.notna(o) else 99)
              for f, o in zip(frames, g["finish_order"])}
        pos = {v: k for k, v in fo.items()}
        if not all(p in pos for p in (1, 2, 3)):
            continue
        mk = mark.get(rk, {})
        honmei = next((f for f in frames if mk.get(f) == 1), None)
        if honmei is None:
            continue
        our = frames[0]  # モデル1位
        recs.append({
            "race_key": rk, "race_date": g["race_date"].iloc[0],
            "rp_std": rp.std(),
            "our": our, "model_order": frames, "honmei": honmei,
            "overlap": int(our == honmei),
            "pos1": pos[1], "pos2": pos[2], "pos3": pos[3],
        })
    return pd.DataFrame(recs)


def evaluate(df, exa, tri, label):
    print(f"\n{'='*72}\n=== {label} ===")
    for wl, w in (("DISC", DISC), ("CONF", CONF)):
        s = seg(df, w)
        days = s["race_date"].nunique() or 1
        for ov_label, ov in (("◎一致", 1), ("◎不一致(=売価値)", 0)):
            d = s[s["overlap"] == ov]
            n = len(d)
            if not n:
                continue
            # 集計器
            agg = defaultdict(lambda: [0, 0, 0, 0])  # n,hit,bet,pay
            for _, r in d.iterrows():
                er, tr = exa.get(r["race_key"], {}), tri.get(r["race_key"], {})
                mo = r["model_order"]
                our = r["our"]
                actual2 = (r["pos1"], r["pos2"])
                actual3 = (r["pos1"], r["pos2"], r["pos3"])
                # E1 二車単 our→全(6点) : our が1着なら的中
                def acc(key, legs_dict, actual):
                    legs = {k: v for k, v in legs_dict.items() if v}
                    if not legs:
                        return
                    a = agg[key]
                    a[0] += 1; a[2] += len(legs) * 100
                    if actual in legs:
                        a[1] += 1; a[3] += int(legs[actual] * 100)
                acc("E1 二車単 our→全(6点)",
                    {(our, j): er.get((our, j)) for j in mo if j != our}, actual2)
                acc("E2 二車単 our→モデル2-4(3点)",
                    {(our, j): er.get((our, j)) for j in mo[1:4]}, actual2)
                acc("T1 三連単 our→2-4→2-5(F)",
                    {(our, j, k): tr.get((our, j, k)) for j in mo[1:4] for k in mo[1:5]
                     if k not in (our, j)}, actual3)
                acc("T2 三連単 our→2-3→全(F)",
                    {(our, j, k): tr.get((our, j, k)) for j in mo[1:3] for k in mo
                     if k not in (our, j)}, actual3)
            print(f"  --- {wl} [{ov_label}] n={n} ({n/days:.1f}R/日) ---")
            for k in ["E1 二車単 our→全(6点)", "E2 二車単 our→モデル2-4(3点)",
                      "T1 三連単 our→2-4→2-5(F)", "T2 三連単 our→2-3→全(F)"]:
                a = agg[k]
                if a[0] and a[2]:
                    print(f"     {k:<24} 的中率={a[1]/a[0]:5.1%} ROI={a[3]/a[2]:6.1%} (R={a[0]})")


def main():
    ph = pd.read_pickle(PH)
    mark = load_marks(ph["race_key"].unique().tolist())
    df = build(ph, mark)
    exa, tri = load_ordered(df["race_key"].unique().tolist())
    print(f"クリーン7車・◎有: {len(df):,}レース  全体◎一致率={df['overlap'].mean():.1%}")

    th = df["rp_std"].quantile(1 / 3)
    upset = df[df["rp_std"] <= th]
    print(f"波乱傾向(rp_std<={th:.2f}): {len(upset):,}  ◎一致率={upset['overlap'].mean():.1%}")
    evaluate(df, exa, tri, "全レース")
    evaluate(upset, exa, tri, "波乱傾向レース(拮抗)")


if __name__ == "__main__":
    main()
