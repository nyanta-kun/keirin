"""波乱レースの構造プロファイル分析（2026-07-15）

欠車/落車/失格のないクリーン7車レースに限定し、「人気が飛んで高配当（波乱）」の
レースが、事前の構造特徴でどう特徴づけられるかを記述分析する:
  - 競争得点: 平均・分散(std)・レンジ・トップ2得点差
  - ライン: ライン数・最大ライン人数・単騎数・最強ライン得点シェア・ライン得点の分散
  - 脚質: 逃/両/追/マ の頭数分布（特に逃頭数=つぶし合い）

波乱指標 = 実際の的中三連単オッズ（=配当倍率）。tier分け。
（モデル指数は「人気が飛ぶ」の確認用にのみ併記。市場人気≒指数上位）

窓: 全クリーン期間で傾向を見る（記述統計・学習なし）。
使い方: .venv/bin/python scripts/exp_upset_profile_wt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.database import get_connection  # noqa: E402
from exp_stable_top2_wt import CACHE_DIR  # noqa: E402

PH_CACHE = CACHE_DIR / "perhorse_n7.pkl"


def load_tri_odds(race_keys):
    tri = {}
    rks = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(rks), 900):
            chunk = rks[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trifecta' AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is None or not (0 < float(od) < 90000):
                    continue
                try:
                    parts = tuple(int(x) for x in comb.split("-"))
                except ValueError:
                    continue
                if len(parts) == 3:
                    tri.setdefault(rk, {})[parts] = float(od)
    return tri


def build(ph, tri):
    recs = []
    for rk, g in ph.groupby("race_key"):
        g = g.sort_values("model_rank")
        if len(g) != 7:
            continue
        rp = g["race_point"].astype(float).to_numpy()
        styles = g["style"].astype(str).tolist()
        lg = g["line_group"].astype(int).to_numpy()
        fo = {int(f): (int(o) if pd.notna(o) else 99)
              for f, o in zip(g["frame_no"].astype(int), g["finish_order"])}
        pos = {v: k for k, v in fo.items()}
        if not all(p in pos for p in (1, 2, 3)):
            continue
        result = (pos[1], pos[2], pos[3])
        win_od = tri.get(rk, {}).get(result)
        if win_od is None:
            continue
        pred = g["pred"].to_numpy()
        fr = g["frame_no"].astype(int).tolist()
        a1, a2 = fr[0], fr[1]
        top3 = {pos[1], pos[2], pos[3]}
        # ライン得点
        line_tot = {}
        for lgi, r in zip(lg, rp):
            line_tot[lgi] = line_tot.get(lgi, 0) + r
        line_totals = np.array(list(line_tot.values()))
        sizes = pd.Series(lg).value_counts()
        # 脚質頭数
        n_nige = styles.count("逃")
        n_ryo = styles.count("両")
        n_oi = styles.count("追") + styles.count("マ")
        recs.append({
            "race_key": rk, "race_date": g["race_date"].iloc[0],
            "win_od": win_od,
            "favs_flew": int(not (a1 in top3 and a2 in top3)),  # 指数上位2が3着内でない
            # 競争得点
            "rp_mean": rp.mean(), "rp_std": rp.std(),
            "rp_range": rp.max() - rp.min(),
            "rp_top_gap": np.sort(rp)[-1] - np.sort(rp)[-2],
            # ライン
            "n_lines": len(line_tot),
            "max_line_size": int(sizes.max()),
            "n_solo": int((sizes == 1).sum()),
            "top_line_share": line_totals.max() / rp.sum(),
            "line_tot_std": line_totals.std(),
            # 脚質
            "n_nige": n_nige, "n_ryo": n_ryo, "n_oi": n_oi,
            # 指数
            "top2_share": (pred / pred.sum())[:2].sum(),
            "gap12": pred[0] - pred[1],
        })
    return pd.DataFrame(recs)


def upset_rate_by(df, col, bins=5):
    """特徴 col の分位ビンごとに 波乱率(win_od>=100)・平均配当 を出す。"""
    try:
        q = pd.qcut(df[col], bins, duplicates="drop")
    except ValueError:
        q = pd.cut(df[col], bins)
    g = df.groupby(q, observed=True).agg(
        n=("win_od", "size"),
        upset100=("win_od", lambda s: (s >= 100).mean()),
        upset300=("win_od", lambda s: (s >= 300).mean()),
        favs_flew=("favs_flew", "mean"),
        med_od=("win_od", "median"),
    )
    print(f"\n== {col} 分位別 ==")
    for idx, r in g.iterrows():
        print(f"  {str(idx):<22} n={int(r.n):>5} 波乱率(≥100倍)={r.upset100:5.1%} "
              f"(≥300倍)={r.upset300:5.1%} 本命飛び={r.favs_flew:5.1%} 中央配当={r.med_od:6.1f}")


def main():
    ph = pd.read_pickle(PH_CACHE)
    tri = load_tri_odds(ph["race_key"].unique().tolist())
    df = build(ph, tri)
    print(f"クリーン7車・三連単配当あり: {len(df):,}レース")

    # 全体の配当分布
    print("\n=== 的中三連単オッズ（配当倍率）分布 ===")
    for th in (10, 30, 50, 100, 200, 300, 500):
        print(f"  ≥{th:>4}倍: {(df.win_od>=th).mean():5.1%}")
    print(f"  中央値={df.win_od.median():.1f}倍  平均={df.win_od.mean():.1f}倍")

    # 波乱(≥100倍) vs 平穏(<30倍) の特徴平均比較
    up = df[df.win_od >= 100]
    calm = df[df.win_od < 30]
    print(f"\n=== 特徴平均: 波乱(≥100倍 n={len(up)}) vs 平穏(<30倍 n={len(calm)}) ===")
    feats = ["rp_mean", "rp_std", "rp_range", "rp_top_gap", "n_lines",
             "max_line_size", "n_solo", "top_line_share", "line_tot_std",
             "n_nige", "n_ryo", "n_oi", "top2_share", "gap12"]
    print(f"  {'特徴':<16}{'波乱':>10}{'平穏':>10}{'差(波乱-平穏)':>14}")
    for f in feats:
        print(f"  {f:<16}{up[f].mean():>10.3f}{calm[f].mean():>10.3f}{up[f].mean()-calm[f].mean():>14.3f}")

    # 主要特徴の分位別 波乱率
    for f in ["rp_std", "rp_top_gap", "top_line_share", "n_nige", "n_solo",
              "top2_share", "gap12"]:
        upset_rate_by(df, f)

    # === モデル信頼度を固定して構造特徴が波乱を"上乗せ"予測するか ===
    print("\n" + "=" * 70)
    print("モデルtop2_share(3分位) × 構造特徴 の波乱率(≥100倍)＝直交性の確認")
    df["ts_bin"] = pd.qcut(df["top2_share"], 3, labels=["低(不確実)", "中", "高(固い)"])
    for f, edges in (("n_nige", [-1, 1, 2, 9]),
                     ("rp_std", None),
                     ("top_line_share", None)):
        print(f"\n-- 行=top2_share3分位, 列={f} --")
        if edges:
            df["_fb"] = pd.cut(df[f], edges)
        else:
            df["_fb"] = pd.qcut(df[f], 3, duplicates="drop")
        piv = df.pivot_table(index="ts_bin", columns="_fb", values="win_od",
                             aggfunc=lambda s: (s >= 100).mean(), observed=True)
        print((piv * 100).round(1).to_string())


if __name__ == "__main__":
    main()
