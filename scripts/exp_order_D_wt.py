"""検証D: 固め2車の着順を構造で絞り三連単を狙う（2026-07-15）

固め2車(top2_share>=0.5)前提で、
  ①どちらの軸が先着するかを 脚質/得点/ライン で読めるか（二車単で診断）
  ②三連単フォーメーション（軸2車の並び×3列目）のROI
を検証する。着順市場(exacta/trifecta)は公衆資金が最も分散するため妙味の可能性。

前提: exp_third_structure_wt.py の per-horse キャッシュ（perhorse_n7.pkl）
窓: DISCOVER 03-01〜05-31 / CONFIRM 06-01〜07-10
使い方: .venv/bin/python scripts/exp_order_D_wt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.database import get_connection  # noqa: E402
from exp_stable_top2_wt import seg, DISC, CONF, CACHE_DIR  # noqa: E402

PH_CACHE = CACHE_DIR / "perhorse_n7.pkl"


def load_ordered_odds(race_keys):
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
                    parts = tuple(int(x) for x in comb.split("-"))
                except ValueError:
                    continue
                if bt == "exacta" and len(parts) == 2:
                    exa.setdefault(rk, {})[parts] = float(od)
                elif bt == "trifecta" and len(parts) == 3:
                    tri.setdefault(rk, {})[parts] = float(od)
    return exa, tri


def build_records(ph):
    recs = []
    for rk, g in ph.groupby("race_key"):
        g = g.sort_values("model_rank")
        if len(g) != 7:
            continue
        fr = g["frame_no"].astype(int).tolist()
        pred = g["pred"].tolist()
        stl = dict(zip(g["frame_no"].astype(int), g["style"].astype(str)))
        rp = dict(zip(g["frame_no"].astype(int), g["race_point"].astype(float)))
        lg = dict(zip(g["frame_no"].astype(int), g["line_group"].astype(int)))
        fo = {int(f): (int(o) if pd.notna(o) else 99)
              for f, o in zip(g["frame_no"].astype(int), g["finish_order"])}
        pos = {v: k for k, v in fo.items()}  # finish_order -> frame
        if not all(p in pos for p in (1, 2, 3)):
            continue
        a1, a2 = fr[0], fr[1]
        pn = np.array(pred) / sum(pred)
        recs.append({
            "race_key": rk, "race_date": g["race_date"].iloc[0],
            "top2_share": pn[0] + pn[1],
            "a1": a1, "a2": a2, "thirds": fr[2:],
            "pos1": pos[1], "pos2": pos[2], "pos3": pos[3],
            "a1_style": stl[a1], "a2_style": stl[a2],
            "a1_rp": rp[a1], "a2_rp": rp[a2],
            "a1_line": lg[a1], "a2_line": lg[a2],
            "same_line": int(lg[a1] == lg[a2]),
        })
    return pd.DataFrame(recs)


STYLE_LEAD = {"逃": 3, "両": 2, "追": 1, "マ": 1}  # 先行度（大=前）


def main():
    ph = pd.read_pickle(PH_CACHE)
    print(f"per-horse: {len(ph):,} rows")
    R = build_records(ph)
    exa, tri = load_ordered_odds(R["race_key"].unique().tolist())
    R = R[R["top2_share"] >= 0.5].copy()
    print(f"固め2車(top2_share>=0.5): {len(R):,}レース")

    # ===== ① 二車単: 軸2車の並びを読めるか =====
    for wl, w in (("DISCOVER", DISC), ("CONFIRM", CONF)):
        s = seg(R, w).copy()
        days = s["race_date"].nunique() or 1
        # 軸2車がワンツー独占する率
        s["box_hit"] = ((s["pos1"].isin([s["a1"], s["a2"]])) &
                        (s["pos2"].isin([s["a1"], s["a2"]])) &
                        ({*[0]} or True))  # placeholder
        box = s.apply(lambda r: {r["pos1"], r["pos2"]} == {r["a1"], r["a2"]}, axis=1)
        a1_first = s.apply(lambda r: r["pos1"] == r["a1"] and r["pos2"] == r["a2"], axis=1)
        # 脚質で並び予測: 先行度が高い方を1着に
        def style_order_hit(r):
            l1, l2 = STYLE_LEAD.get(r["a1_style"], 1), STYLE_LEAD.get(r["a2_style"], 1)
            lead = r["a1"] if l1 >= l2 else r["a2"]
            sub = r["a2"] if lead == r["a1"] else r["a1"]
            return r["pos1"] == lead and r["pos2"] == sub
        style_hit = s.apply(style_order_hit, axis=1)
        print(f"\n===== {wl} 固め2車 二車単診断 n={len(s)} ({days}日) =====")
        print(f"  軸2車ワンツー独占率: {box.mean():.1%}  "
              f"| うちモデル順(a1→a2)正解: {a1_first.sum()}/{box.sum()}={a1_first.sum()/max(box.sum(),1):.1%}  "
              f"脚質順正解: {style_hit.sum()}/{box.sum()}={style_hit.sum()/max(box.sum(),1):.1%}")
        # 二車単ROI（モデル順 / 両流し / 脚質順）
        def exa_roi(pick_fn):
            n = b = p = h = 0
            for _, r in s.iterrows():
                picks = pick_fn(r)
                od_map = exa.get(r["race_key"], {})
                legs = {pk: od_map.get(pk) for pk in picks}
                legs = {k: v for k, v in legs.items() if v}
                if not legs:
                    continue
                n += 1; b += len(legs) * 100
                actual = (r["pos1"], r["pos2"])
                if actual in legs:
                    p += int(legs[actual] * 100); h += 1
            return n, h / n if n else 0, p / b if b else 0
        for name, fn in (
            ("モデル順 a1→a2", lambda r: [(r["a1"], r["a2"])]),
            ("両流し(box2点)", lambda r: [(r["a1"], r["a2"]), (r["a2"], r["a1"])]),
            ("脚質順1点", lambda r: [((r["a1"], r["a2"]) if STYLE_LEAD.get(r["a1_style"],1) >= STYLE_LEAD.get(r["a2_style"],1) else (r["a2"], r["a1"]))]),
        ):
            n, h, roi = exa_roi(fn)
            print(f"    二車単 {name:<16} n={n:>4} 的中={h:5.1%} ROI={roi:6.1%}")

    # ===== ② 三連単フォーメーション =====
    print("\n" + "=" * 70)
    print("三連単フォーメーション（固め2車・最終オッズ100円/点）")
    forms = {
        "F0 1,2着=軸box × 3着=モデル3-5位": lambda r: [
            (i, j, k) for (i, j) in [(r["a1"], r["a2"]), (r["a2"], r["a1"])]
            for k in r["thirds"][:3]],
        "F1 1,2着=軸box × 3着=全5": lambda r: [
            (i, j, k) for (i, j) in [(r["a1"], r["a2"]), (r["a2"], r["a1"])]
            for k in r["thirds"]],
        "F2 1着=脚質先行軸 × 2着=他軸 × 3着=モデル3-5": lambda r: [
            ((r["a1"], r["a2"]) if STYLE_LEAD.get(r["a1_style"],1) >= STYLE_LEAD.get(r["a2_style"],1)
             else (r["a2"], r["a1"]))[:2] + (k,) for k in r["thirds"][:3]] if True else [],
        "F3 3着=軸 1,2着=モデル3-5位から(裏)": lambda r: [
            (i, j, k) for k in [r["a1"], r["a2"]]
            for (i, j) in [(r["thirds"][0], r["thirds"][1]), (r["thirds"][1], r["thirds"][0])]],
    }
    for wl, w in (("DISCOVER", DISC), ("CONFIRM", CONF)):
        s = seg(R, w)
        days = s["race_date"].nunique() or 1
        print(f"\n--- {wl} ({days}日) ---")
        for name, fn in forms.items():
            n = b = p = h = 0
            for _, r in s.iterrows():
                od_map = tri.get(r["race_key"], {})
                picks = fn(r)
                legs = {}
                for pk in picks:
                    if len(pk) == 3 and od_map.get(pk):
                        legs[pk] = od_map[pk]
                if not legs:
                    continue
                n += 1; b += len(legs) * 100
                actual = (r["pos1"], r["pos2"], r["pos3"])
                if actual in legs:
                    p += int(legs[actual] * 100); h += 1
            if n and b:
                print(f"  {name:<34} R={n:>4}({n/days:4.1f}/日) 的中={h/n:5.1%} 投{b:>8,} ROI={p/b:6.1%}")


if __name__ == "__main__":
    main()
