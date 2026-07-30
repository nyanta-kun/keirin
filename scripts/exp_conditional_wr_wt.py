"""周長別・時間帯別（デイ/ナイター）・グレード別の選手条件別勝率 特徴の追加検証
（netkeirin未活用データ調査の一環・2026-07-28）。

netkeirinのデータ分析タブ「周長」「時間帯別」「グレード別」（選手×条件の過去勝率）に
相当する軸。3つとも既存DBの生カラムだけで point-in-time 計算可能（スクレイピング不要）:
  - 周長  : venue_info.bank_length（既に raw に JOIN 済み。250/333/400/500の4値）
  - 時間帯: wt_races.start_at（UNIX秒）から JST 時刻を算出→17時以降を night とする単純区分
  - グレード: wt_races.grade（既に raw に JOIN 済み）

計算方式は build_features_wt() 内の venue_wr と同一（選手×条件のexpanding().mean().shift(1)、
完走のみ・当日を含まないpoint-in-time）。venue_wrは既存特徴なので、ここでは
「同じ考え方を別の条件軸に展開したら独立した情報になるか」を検証する。

検証は既存ハーネスと同一分割・指標。baseline / +bank_wr / +night(is_night+night_wr) /
+grade_wr / +all(3種+is_night) を複数seed・クリーンOOSで比較。
本番 FEATURE_COLS_WT / lgbm_wt.pkl は変更しない。

クリーン分割: TRAIN 2022-12-01〜2026-03-31 / TEST(未使用OOS) 2026-04-01〜2026-06-30 /
FWD 2026-07-01〜2026-07-10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.database import get_connection
from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
)

TR_FROM, TR_TO = "2022-12-01", "2026-03-31"
TE_FROM, TE_TO = "2026-04-01", "2026-06-30"
FW_FROM, FW_TO = "2026-07-01", "2026-07-10"
SEEDS = [42, 7, 123, 2024, 99]

PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              verbose=-1)

NEW_COLS = ["bank_wr", "is_night", "night_wr", "grade_wr"]


def compute_conditional_wr(raw: pd.DataFrame) -> pd.DataFrame:
    """venue_wr と同じ point-in-time 方式で 周長別/時間帯別/グレード別 勝率を付与する。"""
    H = raw[["race_key", "player_id", "finish_order", "race_date",
             "start_at", "bank_length", "grade"]].copy()
    H["_dt"] = pd.to_datetime(H["race_date"])
    H["win"] = (pd.to_numeric(H["finish_order"], errors="coerce") == 1).astype(float)
    # JST時刻→17時以降をnight（ナイター/ミッドナイト想定の単純区分）
    st = pd.to_numeric(H["start_at"], errors="coerce")
    hour_jst = pd.to_datetime(st, unit="s", utc=True).dt.tz_convert("Asia/Tokyo").dt.hour
    H["is_night"] = (hour_jst >= 17).astype(float)
    H = H[pd.to_numeric(H["finish_order"], errors="coerce") >= 1].copy()
    H = H.sort_values(["player_id", "_dt"]).reset_index(drop=True)

    def _cond_wr(group_col):
        return (H.sort_values(["player_id", group_col, "_dt"])
                 .groupby(["player_id", group_col])["win"]
                 .apply(lambda s: s.expanding().mean().shift(1))
                 .reset_index(level=[0, 1], drop=True))

    H["bank_wr"] = _cond_wr("bank_length")
    H["grade_wr"] = _cond_wr("grade")
    H["night_wr"] = _cond_wr("is_night")

    Hroll = H[["race_key", "player_id", "is_night", "bank_wr", "grade_wr", "night_wr"]]
    out = raw.merge(Hroll, on=["race_key", "player_id"], how="left")
    fill = {"bank_wr": 0.0, "grade_wr": 0.0, "night_wr": 0.0, "is_night": 0.0}
    out = out.fillna(value=fill)
    return out


def race_metrics(df_ev):
    both_top3, win1, top3_1, n = 0, 0, 0, 0
    for _, g in df_ev.groupby("race_key"):
        g = g[g["finish_order"] >= 1]
        if g["race_key"].isna().any() or len(g) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        fo = g["finish_order"].astype(float).tolist()
        n += 1
        if fo[0] == 1:
            win1 += 1
        if 1 <= fo[0] <= 3:
            top3_1 += 1
        if (1 <= fo[0] <= 3) and (1 <= fo[1] <= 3):
            both_top3 += 1
    if n == 0:
        return dict(n=0, ss=0.0, win1=0.0, top3_1=0.0)
    return dict(n=n, ss=both_top3 / n, win1=win1 / n, top3_1=top3_1 / n)


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date=TR_FROM, max_date=FW_TO)
    raw = compute_conditional_wr(raw)
    print(f"  night比率={raw['is_night'].mean():.2%} "
          f"bank_length分布={raw['bank_length'].value_counts().to_dict()}")
    df = build_features_wt(raw)
    df = df[df["finish_order"] >= 1].copy()

    with get_connection() as c:
        ne = dict(c.execute("SELECT race_key, n_entries FROM wt_races").fetchall())
    df["_ne"] = df["race_key"].map(ne)
    df7 = df[df["_ne"] == 7].copy()

    tr = df[df["race_date"] <= TR_TO].copy()
    te = df7[(df7["race_date"] >= TE_FROM) & (df7["race_date"] <= TE_TO)].copy()
    fw = df7[(df7["race_date"] >= FW_FROM) & (df7["race_date"] <= FW_TO)].copy()
    print(f"TRAIN {tr['race_key'].nunique()}R / TEST(7車) {te['race_key'].nunique()}R / "
          f"FWD(7車) {fw['race_key'].nunique()}R")

    variants = {
        "baseline": list(FEATURE_COLS_WT),
        "+bank_wr": list(FEATURE_COLS_WT) + ["bank_wr"],
        "+night": list(FEATURE_COLS_WT) + ["is_night", "night_wr"],
        "+grade_wr": list(FEATURE_COLS_WT) + ["grade_wr"],
        "+all": list(FEATURE_COLS_WT) + NEW_COLS,
    }

    agg = {v: {"auc_te": [], "auc_fw": [], "ss_te": [], "win1_te": [], "top3_te": [],
               "ss_fw": [], "win1_fw": [], "top3_fw": []} for v in variants}

    for seed in SEEDS:
        for vname, cols in variants.items():
            Xtr = tr[cols].fillna(0).values
            ytr = tr[TARGET_COL_WT].values
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(Xtr, ytr)
            for tag, ev in (("te", te), ("fw", fw)):
                ev = ev.copy()
                ev["pred_prob"] = m.predict_proba(ev[cols].fillna(0).values)[:, 1]
                auc = roc_auc_score(ev[TARGET_COL_WT], ev["pred_prob"])
                mt = race_metrics(ev)
                agg[vname][f"auc_{tag}"].append(auc)
                agg[vname][f"ss_{tag}"].append(mt["ss"])
                agg[vname][f"win1_{tag}"].append(mt["win1"])
                agg[vname][f"top3_{tag}"].append(mt["top3_1"])
        print(f"  seed {seed} done")

    def ms(a):
        return np.mean(a), np.std(a)

    print("\n================ 結果（seed平均 ± std, n_seeds=%d）================" % len(SEEDS))
    for tag, label in (("te", "TEST 2026-04〜06 (クリーンOOS)"), ("fw", "FWD 2026-07")):
        print(f"\n--- {label} ---")
        print(f"{'variant':<12}{'AUC':>16}{'SS的中(2車3着内)':>22}{'1位勝率':>14}{'1位複勝率':>14}")
        base = agg["baseline"]
        for v in variants:
            a = agg[v]
            auc_m, auc_s = ms(a[f"auc_{tag}"])
            ss_m, ss_s = ms(a[f"ss_{tag}"])
            w_m, w_s = ms(a[f"win1_{tag}"])
            t_m, t_s = ms(a[f"top3_{tag}"])
            dss = ss_m - np.mean(base[f"ss_{tag}"])
            mark = ""
            if v != "baseline":
                mark = "  ★" if dss > ss_s else ("  ×" if dss < -ss_s else "  ~")
            print(f"{v:<12}{auc_m:>7.4f}±{auc_s:.4f}{ss_m:>13.1%}±{ss_s:.1%}"
                  f"{w_m:>8.1%}±{w_s:.1%}{t_m:>8.1%}±{t_s:.1%}{mark}")
    print("\n判定: SSΔが seed std を超えれば ★(採用候補) / 範囲内 ~ / 悪化 ×。")
    print("採用は TEST・FWD 双方で非悪化かつ TEST で ★ が条件。")


if __name__ == "__main__":
    main()
