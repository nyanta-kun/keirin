"""7車以上: 本命バストの「個別精度」検証（払戻と突き合わせる前段）

2問を分離して精度のみ確認:
 A. バスト検出精度: 「1位(pred最上位)が3着外」をr1_prob(モデル自己信頼度)以上に当てられるか。
    専用バスト分類器(race-level特徴)のAUCを baseline(1-r1_prob) と比較。
 B. 波乱時の軸選定精度: 1位が飛んだ条件下で実際に車券内(top3)に来るのは本当にpred2位か。
    pred順位 / 脚質(逃0/両1/追2) / 1位と同ライン否か 別の物差しで top3率を測り、
    「2位より良い second軸」候補があるか。pred非バスト時との構造差も見る。
払戻不使用・的中精度のみ。pooled lgbm_wt・7+・train(2023-07〜2026-02)→test(2026-03〜)。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import defaultdict, Counter
import pandas as pd
import lightgbm as lgb
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model

model = load_model("lgbm_wt")


def manual_auc(y, s):
    """rank法AUC（sklearn非依存）。"""
    pairs = sorted(zip(s, y))
    n = len(pairs)
    npos = sum(y); nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    # 同値タイは平均ランク
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (sum_pos - npos * (npos + 1) / 2) / (npos * nneg)


def collect(f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz >= 7].index)].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    races, riders = [], []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(g)
        if n < 7:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        win = set(int(x) for x in fin["frame_no"])
        r1 = g.iloc[0]
        r1_line = r1["line_group"]
        bust = int(int(r1["frame_no"]) not in win)
        rp = sorted(g["race_point"].tolist(), reverse=True)
        races.append({
            "rk": rk, "bust": bust, "n": n,
            "r1_prob": p[0], "gap12": p[0] - p[1], "gap13": p[0] - p[2],
            "gap_mean": p[0] - (sum(p[1:]) / (n - 1)), "pred_std": pd.Series(p).std(),
            "r1_style": int(r1["style_enc"]), "n_senko": int(r1["n_senko"]),
            "r1_score_z": float(r1["score_z"]), "r1_rp_gap": rp[0] - rp[1] if len(rp) > 1 else 0.0,
            "top3_sum": p[0] + p[1] + p[2],
        })
        # 非r1の各選手（条件付き分析用）
        for idx in range(1, n):
            r = g.iloc[idx]
            riders.append({
                "rk": rk, "bust": bust, "pred_rank": idx + 1,        # 2位,3位...
                "style": int(r["style_enc"]),
                "same_line": int(r["line_group"] == r1_line),
                "in_top3": int(int(r["frame_no"]) in win),
                "n": n,
            })
    return pd.DataFrame(races), pd.DataFrame(riders)


# ---------- A: バスト検出精度 ----------
def part_A(rtr, rte):
    print(f"\n{'='*88}\n  A. バスト検出精度（1位が3着外を当てる）  TRAIN {len(rtr)}R / TEST {len(rte)}R\n{'='*88}")
    print(f"  全体バスト率: TRAIN {rtr['bust'].mean():.1%} / TEST {rte['bust'].mean():.1%}")
    # baseline: 1 - r1_prob
    base_auc_tr = manual_auc(rtr["bust"].tolist(), (1 - rtr["r1_prob"]).tolist())
    base_auc_te = manual_auc(rte["bust"].tolist(), (1 - rte["r1_prob"]).tolist())
    print(f"\n  baseline(1-r1_prob)のバストAUC: TRAIN {base_auc_tr:.4f} / TEST {base_auc_te:.4f}")
    # 単一特徴のAUC（向き調整: バストと正相関になるよう符号）
    print("  単一特徴のバストAUC(TEST):")
    for col, sign in [("r1_prob", -1), ("gap12", -1), ("gap13", -1), ("gap_mean", -1),
                      ("r1_score_z", -1), ("r1_rp_gap", -1), ("n_senko", +1), ("pred_std", -1)]:
        a = manual_auc(rte["bust"].tolist(), (sign * rte[col]).tolist())
        print(f"    {col:<12}{a:.4f}")
    # 専用LGBMバスト分類器
    feats = ["r1_prob", "gap12", "gap13", "gap_mean", "pred_std", "r1_style",
             "n_senko", "r1_score_z", "r1_rp_gap", "n"]
    dtr = lgb.Dataset(rtr[feats], label=rtr["bust"])
    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 42}
    bm = lgb.train(params, dtr, num_boost_round=300)
    pred_tr = bm.predict(rtr[feats]); pred_te = bm.predict(rte[feats])
    print(f"\n  専用バスト分類器LGBM AUC: TRAIN {manual_auc(rtr['bust'].tolist(), pred_tr.tolist()):.4f}"
          f" / TEST {manual_auc(rte['bust'].tolist(), pred_te.tolist()):.4f}")
    imp = sorted(zip(feats, bm.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    print("  重要度(gain上位): " + ", ".join(f"{k}={v:.0f}" for k, v in imp[:6]))
    # バスト予測十分位ごとの実バスト率(TEST・キャリブレーション)
    rte = rte.copy(); rte["pb"] = pred_te
    rte["dec"] = pd.qcut(rte["pb"], 10, labels=False, duplicates="drop")
    print("  TEST: 予測バスト十分位ごとの実バスト率(下位→上位):")
    print("    " + " ".join(f"{rte[rte['dec']==d]['bust'].mean():.0%}" for d in sorted(rte["dec"].unique())))
    print("  ※ baselineとLGBMのAUC差が小さければ、バスト検出はr1_prob(自己信頼度)でほぼ尽き、専用化の余地小。")


# ---------- B: 波乱時の軸選定精度 ----------
def part_B(dtr, dte):
    bust = dtr[dtr["bust"] == 1]; nb = dtr[dtr["bust"] == 0]
    nbust_races = bust["rk"].nunique()
    print(f"\n{'='*88}\n  B. 波乱時(1位バスト)の軸選定精度  TRAIN バスト{nbust_races}R\n{'='*88}")

    # B1: pred順位ごとの top3率（バスト時 vs 非バスト時）
    print("\n  B1. pred順位ごとの top3率（その順位が存在するレース内）:")
    print(f"    {'pred順位':<8}{'バスト時':>10}{'非バスト時':>12}")
    for rk_ in [2, 3, 4, 5, 6, 7]:
        bsub = bust[bust["pred_rank"] == rk_]; nsub = nb[nb["pred_rank"] == rk_]
        bt = bsub["in_top3"].mean() if len(bsub) else float("nan")
        nt = nsub["in_top3"].mean() if len(nsub) else float("nan")
        print(f"    {rk_}位{'':<5}{bt:>9.1%}{nt:>11.1%}")
    print("   ※ バスト時に2位のtop3率が3位4位と大差なければ『2位を軸』の優位は薄い。")

    # B2: 脚質ごとの top3率（バスト時 vs 非バスト時）
    SMAP = {0: "逃", 1: "両", 2: "追", -1: "不明"}
    print("\n  B2. 脚質ごとの top3率（非r1選手・バスト時 vs 非バスト時）:")
    print(f"    {'脚質':<6}{'バスト時top3':>12}{'非バスト時':>12}{'バスト時シェア':>14}")
    for st in [0, 1, 2]:
        bsub = bust[bust["style"] == st]; nsub = nb[nb["style"] == st]
        bt = bsub["in_top3"].mean() if len(bsub) else float("nan")
        nt = nsub["in_top3"].mean() if len(nsub) else float("nan")
        share = len(bsub) / max(len(bust), 1)
        print(f"    {SMAP[st]:<6}{bt:>11.1%}{nt:>11.1%}{share:>13.1%}")
    print("   ※ バスト時に特定脚質(例:追=差し)のtop3率が跳ねれば、脚質で軸を選ぶ価値。")

    # B3: 1位と同ライン vs 別ライン の top3率（バスト時）
    print("\n  B3. 1位ライン関係ごとの top3率（バスト時）:")
    for sl, lab in [(1, "1位と同ライン(番手等)"), (0, "別ライン")]:
        sub = bust[bust["same_line"] == sl]
        bt = sub["in_top3"].mean() if len(sub) else float("nan")
        print(f"    {lab:<22}{bt:>8.1%}  (該当選手シェア{len(sub)/max(len(bust),1):>5.1%})")
    print("   ※ 1位が飛ぶ時その番手も崩れる(同ライン低)なら、相手は別ラインから採るべき。")

    # B4: second軸候補の比較（バスト時 1人選び その top3率）— レース単位で再構成
    print("\n  B4. 『相手第1候補』別の top3率（バスト時・1レース1候補）:")
    # rebuild per-race rider lists for bust races
    cand = defaultdict(lambda: [0, 0])  # name -> [hit, total]
    for rk_, g in bust.groupby("rk"):
        g = g.sort_values("pred_rank")
        rows = g.to_dict("records")
        def first(pred):
            for r in rows:
                if pred(r):
                    return r
            return None
        picks = {
            "pred2位(r2)": rows[0] if rows else None,
            "別ライン最上位": first(lambda r: r["same_line"] == 0),
            "追(差し)最上位": first(lambda r: r["style"] == 2),
            "両or追 最上位": first(lambda r: r["style"] >= 1),
            "1位ライン番手(同ライン最上位)": first(lambda r: r["same_line"] == 1),
            "逃 最上位(非r1)": first(lambda r: r["style"] == 0),
        }
        for name, pk in picks.items():
            if pk is None:
                continue
            cand[name][1] += 1
            cand[name][0] += pk["in_top3"]
    print(f"    {'相手第1候補':<28}{'top3率':>8}{'該当R':>8}")
    for name, (h, tot) in sorted(cand.items(), key=lambda x: -(x[1][0] / max(x[1][1], 1))):
        print(f"    {name:<28}{h/max(tot,1):>7.1%}{tot:>8}")
    print("   ※ pred2位(r2)を別候補が上回れば、バスト時は『2位以外』を軸にすべき＝軸選定の修正余地。")


if __name__ == "__main__":
    rtr, dtr = collect("2023-07-01", "2026-02-28")
    rte, dte = collect("2026-03-01", "2026-06-08")
    part_A(rtr, rte)
    part_B(dtr, dte)
