"""競輪の基本要素で未表現のもののA/B検証（2026-08-04）。

ユーザー要望:
  「今回のライン能力のように、競輪として基本で組み込まれる要素で考慮されて
    いないものがないか再度洗い直して。単騎、競り合い、まくりなどの展開的要素は
    予想ができるようであれば価値はありそう」

洗い出しの結果、DB にあって未使用・かつ train/serve skew リスクが無いものを検証する。
baseline は **60特徴**（2026-08-04 に race_type 7 + ライン実力 5 を追加した後）。

【① 決まり手（kimarite）の自前 point-in-time 集計】← ユーザーの「まくり」への回答
  `wt_entries.factor`（逃/捲/差/マ）は 1-2着に充足99.98%で付与済み。
  winticket 提供の `ex_spurt_pct`（捲り実行率）/`ex_thrust_pct`（差し実行率）は
  **開催期間中に値が更新される**ため 2026-07-31 に train/serve skew で除外されたが、
  **確定した過去レースの factor から自前で集計すれば point-in-time が保証でき、
  同じ情報をリスクなしで使える**。
      km_nige_90 / km_makuri_90 / km_sashi_90 / km_mark_90
      = 直近90日で「その決まり手で1-2着に入った」率

【② day_index（節の何日目・1〜4）】
  充足率100%なのに完全に未使用。連闘の疲労と勝ち上がりの位置づけを表す。

【③ ライン内の実力バランス】
  2026-08-04 に入れた line_rp_sum は**ライン合計**のみ。ライン内の構造
  （自分がライン内で何番目に強いか・先頭との差）は依然として未表現。

【④ 同ライン内の同県率】
  prefecture は is_home にしか使っていない。ラインの結束の強さ。

【⑤ 前走からのギア比変化】
  gear_ratio はあるが変化量は未使用。ギア上げ＝戦法・調子の変化サイン。

※「競り（番手争い）」は DB に実質存在しないため対象外
  （comment に「競」は 115,777件中36件=0.03%、line_pos 重複も1.4%）。

検証: exp_sb_dyn_ab.py / exp_racetype_field_ab.py と同一方法論
  （2窓 × 5seed × アーム・deterministic=True・AUC / 指数1位の勝率・3着内率）。

DB書き込みなし。

使い方:
    python scripts/exp_basic_elements_ab.py [--windows w1,w2] [--arms all]
"""
import argparse
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, TARGET_COL_WT, build_features_wt, load_raw_data_wt,
)

TRAIN_FROM = "2024-04-01"
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
}
SEEDS = [42, 101, 202, 303, 404]

KM_COLS = ["km_nige_90", "km_makuri_90", "km_sashi_90", "km_mark_90"]
MISC_COLS = ["day_index_n", "line_rp_rank_in_line", "line_rp_gap_leader",
             "line_same_pref_frac", "gear_delta"]

# factor の実値（2026-08-04 実測: 差 / 逃 / 捲 / マ の4種のみ）
_KM_MAP = {"逃": "km_nige_90", "捲": "km_makuri_90",
           "差": "km_sashi_90", "マ": "km_mark_90"}


def _engine():
    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        raise RuntimeError("KEIRIN_DB_URL が未設定です")
    from sqlalchemy import create_engine
    return create_engine(db_url)


def _read_sql(sql: str) -> pd.DataFrame:
    """⚠️ get_connection() を pandas に直接渡すと RealDictCursor のせいで
    全行が「列名の文字列」になる（PG環境で無言で壊れる）。必ず engine 経由。"""
    eng = _engine()
    df = pd.read_sql_query(sql, eng)
    eng.dispose()
    if len(df) and df.iloc[0, 0] == df.columns[0]:
        raise RuntimeError("SQL取得が壊れています（列名が値として入っている）")
    return df


def add_kimarite(df: pd.DataFrame) -> pd.DataFrame:
    """決まり手の point-in-time ローリング（直近90日・closed="left"）。

    sb_dyn と同じく「行は残し、集計値だけ NaN 化」する。行ごと drop すると
    発走前ライブ予測で対象レース自身が merge キーから消え、全選手0.0補完という
    分布外入力になる（2026-07-18〜28 に実際に起きた事故と同型）。
    """
    H = _read_sql(
        "SELECT e.race_key, e.player_id, e.factor, e.finish_order, r.race_date "
        "FROM keirin.wt_entries e JOIN keirin.wt_races r ON e.race_key=r.race_key")
    H["_dt"] = pd.to_datetime(H["race_date"])
    confirmed = pd.to_numeric(H["finish_order"], errors="coerce") >= 1
    fac = H["factor"].fillna("").astype(str).str.strip()
    for jp, col in _KM_MAP.items():
        v = (fac == jp).astype(float)
        H[f"_{col}"] = v.where(confirmed)     # 未確定・DNFは NaN（行は残す）

    H = H.sort_values(["player_id", "_dt"]).reset_index(drop=True)
    for col in KM_COLS:
        H[col] = (H.set_index("_dt").groupby("player_id")[f"_{col}"]
                  .rolling("90D", closed="left").mean()
                  .reset_index(level=0, drop=True).values)

    out = df.merge(H[["race_key", "player_id"] + KM_COLS],
                   on=["race_key", "player_id"], how="left")
    for c in KM_COLS:
        out[c] = out[c].fillna(0.0)
    return out


def add_misc(df: pd.DataFrame) -> pd.DataFrame:
    """day_index / ライン内実力バランス / 同県率 / ギア比変化。"""
    out = df.copy()

    # ② day_index（節の何日目）
    di = _read_sql("SELECT race_key, day_index FROM keirin.wt_races")
    out = out.merge(di, on="race_key", how="left")
    out["day_index_n"] = pd.to_numeric(out["day_index"], errors="coerce").fillna(0.0)

    # ③ ライン内の実力バランス（line_group 欠損は単騎として一意化）
    lg = out["line_group"]
    lg = lg.where(lg.notna(), -out["frame_no"].astype(float))
    key = out["race_key"].astype(str) + "#" + lg.astype(str)
    rp = out["race_point"].astype(float)
    out["line_rp_rank_in_line"] = rp.groupby(key).rank(ascending=False, method="min") - 1
    # ライン先頭（line_pos 最小）の得点との差
    pos = pd.to_numeric(out["line_pos"], errors="coerce").fillna(1)
    tmp = pd.DataFrame({"_k": key, "_pos": pos, "_rp": rp})
    leader = tmp.sort_values("_pos").groupby("_k")["_rp"].first()
    out["line_rp_gap_leader"] = key.map(leader).astype(float) - rp

    # ④ 同ライン内の同県率（自分を除く同ライン中の同県の割合）
    pref = out["player_prefecture"].fillna("")
    tmp2 = pd.DataFrame({"_k": key, "_p": pref})
    size = tmp2.groupby("_k")["_p"].transform("size")
    same = tmp2.groupby(["_k", "_p"])["_p"].transform("size")
    out["line_same_pref_frac"] = ((same - 1) / (size - 1).replace(0, np.nan)).fillna(0.0)

    # ⑤ 前走からのギア比変化（point-in-time・自分の直近の確定走との差）
    G = _read_sql(
        "SELECT e.race_key, e.player_id, e.gear_ratio, e.finish_order, r.race_date "
        "FROM keirin.wt_entries e JOIN keirin.wt_races r ON e.race_key=r.race_key")
    G["_dt"] = pd.to_datetime(G["race_date"])
    gr = pd.to_numeric(G["gear_ratio"], errors="coerce")
    G["_gr"] = gr.where(pd.to_numeric(G["finish_order"], errors="coerce") >= 1)
    G = G.sort_values(["player_id", "_dt"]).reset_index(drop=True)
    G["_prev"] = G.groupby("player_id")["_gr"].shift(1)
    out = out.merge(G[["race_key", "player_id", "_prev"]],
                    on=["race_key", "player_id"], how="left")
    out["gear_delta"] = (out["gear_ratio"].astype(float) - out["_prev"]).fillna(0.0)

    for c in MISC_COLS:
        out[c] = out[c].astype(float).fillna(0.0)
    return out


def race_metrics(test: pd.DataFrame, prob: np.ndarray, ne_map: dict) -> tuple:
    t = test.copy()
    t["p"] = prob
    win = top3 = n = 0
    for rk, g in t.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        fo = g["finish_order"]
        if not (fo.notna() & (fo >= 1)).sum() >= 3:
            continue
        top = g.loc[g["p"].idxmax()]
        f = top["finish_order"]
        if f is None or not f == f:
            f = 99
        n += 1
        win += 1 if f == 1 else 0
        top3 += 1 if 1 <= f <= 3 else 0
    return (win / n if n else 0.0, top3 / n if n else 0.0, n)


def run_window(df: pd.DataFrame, test_from: str, test_to: str, arms: list) -> None:
    from sklearn.metrics import roc_auc_score

    with get_connection() as conn:
        ne_map = dict(conn.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (test_from, test_to)))
    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < test_from)]
    test = df[(df["race_date"] >= test_from) & (df["race_date"] <= test_to)]
    print(f"\n######## 窓 test={test_from}〜{test_to}  "
          f"train {len(train):,}行 / test {len(test):,}行 ########", flush=True)
    print("== 新特徴の分布（test） ==")
    print(test[KM_COLS + MISC_COLS].describe()
          .loc[["mean", "std", "min", "max"]].round(4).to_string())

    results = {}
    n = 0
    for arm, cols in arms:
        aucs, wins, top3s = [], [], []
        m = None
        for seed in SEEDS:
            m = lgb.LGBMClassifier(
                objective="binary", n_estimators=500, learning_rate=0.05,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, random_state=seed,
                deterministic=True, force_row_wise=True, verbose=-1)
            m.fit(train[cols], train[TARGET_COL_WT])
            p = m.predict_proba(test[cols])[:, 1]
            aucs.append(roc_auc_score(test[TARGET_COL_WT], p))
            w, t3, n = race_metrics(test, p, ne_map)
            wins.append(w)
            top3s.append(t3)
        results[arm] = (aucs, wins, top3s)
        print(f"== {arm} ({len(cols)}特徴) ==")
        print(f"  AUC      : {np.mean(aucs):.5f} ± {np.std(aucs):.5f}")
        print(f"  1位勝率  : {np.mean(wins)*100:.2f}% ± {np.std(wins)*100:.2f} (n={n})")
        print(f"  1位3着内 : {np.mean(top3s)*100:.2f}% ± {np.std(top3s)*100:.2f}")
        new = [c for c in cols if c in KM_COLS + MISC_COLS]
        if new and m is not None:
            imp = pd.Series(m.feature_importances_, index=cols)
            ranks = imp.rank(ascending=False).astype(int)
            print("  新特徴の重要度（最終seed・順位/全特徴中）:")
            for c in new:
                print(f"    {c:<22} imp={imp[c]:4d}  順位 {ranks[c]}/{len(cols)}")

    base = results.get("baseline")
    if base:
        for arm, v in results.items():
            if arm == "baseline":
                continue
            print(f"== 差分（{arm} − baseline） ==")
            print(f"  ΔAUC      : {np.mean(v[0])-np.mean(base[0]):+.5f} "
                  f"(seed std baseline={np.std(base[0]):.5f})")
            print(f"  Δ1位勝率  : {(np.mean(v[1])-np.mean(base[1]))*100:+.2f}pt")
            print(f"  Δ1位3着内 : {(np.mean(v[2])-np.mean(base[2]))*100:+.2f}pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2")
    ap.add_argument("--arms", default="all", help="all | km | misc")
    args = ap.parse_args()

    print(f"データ読み込み ... (baseline={len(FEATURE_COLS_WT)}特徴)", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    print("決まり手ローリング ...", flush=True)
    df = add_kimarite(df)
    print("その他の基本要素 ...", flush=True)
    df = add_misc(df)

    arms = [("baseline", FEATURE_COLS_WT)]
    if args.arms in ("all", "km"):
        arms.append(("+kimarite", FEATURE_COLS_WT + KM_COLS))
    if args.arms in ("all", "misc"):
        arms.append(("+misc", FEATURE_COLS_WT + MISC_COLS))
    if args.arms == "all":
        arms.append(("+all", FEATURE_COLS_WT + KM_COLS + MISC_COLS))

    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt, arms)


if __name__ == "__main__":
    main()
