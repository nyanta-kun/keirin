"""レース種別（race_type）とレース内メンバー構成（拮抗度）のA/B検証（2026-08-04）。

仮説の出所（scripts/exp_axis1_miss_analysis.py の層別較正誤差）:
  全体の較正は±1pt以内で正確なのに、レース種別で見ると系統的にズレている。
      初特選 −6.7pt / 選抜 −6.3pt / 決勝 −5.6pt（いずれも過大評価・2SE超）
      ガールズ予選 +2.1〜+2.2pt（過小評価）
  `FEATURE_COLS_WT` を確認すると
    - `grade_enc`（S級/A級/L級）はあるが **race_type は存在しない**
    - レース内 race_point の**集約（水準・ばらつき）も存在しない**
      （`score_z` はレース内標準化なので「全員が強いレース」と
        「実力差が大きいレース」が同じ分布に潰れ、レース全体の
        レベルと拮抗度が構造的に消えている）
  初特選・選抜・決勝は「勝ち上がった実力者が揃って拮抗する」レースであり、
  この2つの欠落で説明がつく。どちらも全レースに効く未表現の構造で、
  隊列位置（ΔAUC +0.00303）が当たったのと同じ型。

新特徴（いずれも発走前に確定・リーク経路なし）:
  [race_type 系] 文字列キーワードのフラグ。100種以上のロングテールを
    序数化せず分解して表現するため、学習データに無い種別も破綻しない。
      rt_is_final / rt_is_semifinal / rt_is_heat / rt_is_senbatsu
      rt_is_tokusen / rt_is_hatsu / rt_is_ippan
  [メンバー構成系] レース内 race_point の集約。
      rp_mean          : レース内平均（レースのレベル）
      rp_std           : レース内標準偏差（拮抗度）
      rp_gap_top2      : レース内 1位−2位 の得点差（抜けた1車がいるか）
      rp_gap_top_self  : 最上位と自分の得点差（score_z の標準化で消える生の差）

検証: exp_sb_dyn_ab.py と同一方法論（2窓 × 5seed × {48特徴, +新特徴}・
  deterministic=True・AUC / 指数1位の勝率・3着内率で比較）。

DB書き込みなし。

使い方:
    python scripts/exp_racetype_field_ab.py [--windows w1,w2] [--arms all]
"""
import argparse
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

TRAIN_FROM = "2024-04-01"  # 既存48特徴（S/Bローリング）が充足する時点
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
}
SEEDS = [42, 101, 202, 303, 404]

RT_COLS = ["rt_is_final", "rt_is_semifinal", "rt_is_heat", "rt_is_senbatsu",
           "rt_is_tokusen", "rt_is_hatsu", "rt_is_ippan"]
FIELD_COLS = ["rp_mean", "rp_std", "rp_gap_top2", "rp_gap_top_self"]
# ライン単位の実力（2026-08-04追加）。
# 現行48特徴はラインの「構造」（line_size/line_pos/is_line_leader/n_lines/
# line_frac）しか持たず、「そのラインが強いか」を一切持っていない。競輪は
# ライン戦なので、ライン単位の実力集約は本来決定的な情報のはず。
# FIELD_COLS がレース単位（レース内で全車同値＝レース内順位に寄与しない）なのに対し、
# こちらは**ラインごとに値が変わる＝レース内で差がつく**ため軸選定に効きうる。
LINE_COLS = ["line_rp_sum", "line_rp_max", "line_rp_mean",
             "line_rank_by_rp", "line_rp_gap_top"]


def add_race_type(df: pd.DataFrame) -> pd.DataFrame:
    """wt_races.race_type からキーワードフラグを付与する。

    load_raw_data_wt は race_type を SELECT していないため DB から別途引く。
    """
    # ⚠️ get_connection() は RealDictCursor を返すため pandas.read_sql_query に
    #    直接渡すと「列名そのもの」が全行の値として読み込まれる（無言で壊れる）。
    #    load_raw_data_wt と同じく SQLAlchemy engine を経由すること。
    import os

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        raise RuntimeError("KEIRIN_DB_URL が未設定です")
    from sqlalchemy import create_engine

    engine = create_engine(db_url)
    # PG 側は keirin スキーマ。load_raw_data_wt と同じくテーブル名を明示修飾する。
    rt = pd.read_sql_query(
        f"SELECT race_key, race_type FROM keirin.wt_races "
        f"WHERE race_date >= '{TRAIN_FROM}'", engine)
    engine.dispose()
    if rt["race_key"].eq("race_key").any():
        raise RuntimeError("race_type の取得が壊れています（列名が値として入っている）")
    out = df.merge(rt, on="race_key", how="left")
    s = out["race_type"].fillna("")
    out["rt_is_semifinal"] = s.str.contains("準決").astype(int)
    out["rt_is_final"] = (s.str.contains("決勝") & ~s.str.contains("準決")).astype(int)
    out["rt_is_heat"] = s.str.contains("予選").astype(int)
    out["rt_is_senbatsu"] = s.str.contains("選抜").astype(int)
    out["rt_is_tokusen"] = s.str.contains("特選").astype(int)
    out["rt_is_hatsu"] = s.str.startswith("初").astype(int)
    out["rt_is_ippan"] = s.str.contains("一般").astype(int)
    return out


def add_field_strength(df: pd.DataFrame) -> pd.DataFrame:
    """レース内 race_point の集約（レベル・拮抗度）を付与する。

    score_z がレース内標準化で捨てている「レース全体の水準とばらつき」を戻す。
    自分自身を含む出走表の情報のみを使うのでリーク経路は無い。
    """
    out = df.copy()
    rp = out["race_point"].astype(float)
    g = rp.groupby(out["race_key"])
    out["rp_mean"] = g.transform("mean")
    out["rp_std"] = g.transform("std").fillna(0.0)
    mx = g.transform("max")
    out["rp_gap_top_self"] = mx - rp

    # 1位−2位差: レース内で降順2番目との差
    def _gap_top2(x: pd.Series) -> float:
        v = x.sort_values(ascending=False).values
        return float(v[0] - v[1]) if len(v) >= 2 else 0.0

    gap = rp.groupby(out["race_key"]).apply(_gap_top2)
    out["rp_gap_top2"] = out["race_key"].map(gap).astype(float)
    for c in FIELD_COLS:
        out[c] = out[c].fillna(0.0)
    return out


def add_line_strength(df: pd.DataFrame) -> pd.DataFrame:
    """ライン単位の実力集約を付与する（レース内で差がつく特徴）。

    ラインは `line_group`（winticket の予想並びのグループID）で識別する。
    ※ line_group の値そのものは隊列の前後を表さないが、所属の識別には使える。
    単騎は line_size=1 の1車ラインとして同じ土俵で扱う。
    race_point は開催中に更新されない安定値・line_group は出走表情報のため、
    ex_* 系のような train/serve skew リスクは持たない。
    """
    out = df.copy()
    rp = out["race_point"].astype(float).fillna(0.0)
    # line_group 欠損は「単騎扱い」として車番で一意なグループを与える
    lg = out["line_group"]
    lg = lg.where(lg.notna(), -out["frame_no"].astype(float))
    key = out["race_key"].astype(str) + "#" + lg.astype(str)

    grp = rp.groupby(key)
    out["line_rp_sum"] = grp.transform("sum")
    out["line_rp_max"] = grp.transform("max")
    out["line_rp_mean"] = grp.transform("mean")

    # レース内でのラインの強さ順位（0=最強）と、最強ラインとの差
    per_line = out.groupby(["race_key", key.rename("_lk")])["line_rp_sum"].first()
    rank = (per_line.groupby(level=0).rank(ascending=False, method="min") - 1)
    top = per_line.groupby(level=0).transform("max")
    lk = list(zip(out["race_key"], key))
    out["line_rank_by_rp"] = [rank.get(k, 0.0) for k in lk]
    out["line_rp_gap_top"] = [top.get(k, 0.0) for k in lk] - out["line_rp_sum"]

    for c in LINE_COLS:
        out[c] = out[c].astype(float).fillna(0.0)
    return out


def race_metrics(test: pd.DataFrame, prob: np.ndarray, ne_map: dict) -> tuple:
    """7車レースの指数1位の勝率・3着内率。"""
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
    print(test[RT_COLS + FIELD_COLS + LINE_COLS].describe()
          .loc[["mean", "std", "min", "max"]].round(3).to_string())

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
        new = [c for c in cols if c in RT_COLS + FIELD_COLS + LINE_COLS]
        if new and m is not None:
            imp = pd.Series(m.feature_importances_, index=cols)
            ranks = imp.rank(ascending=False).astype(int)
            print("  新特徴の重要度（最終seed・順位/全特徴中）:")
            for c in new:
                print(f"    {c:<18} imp={imp[c]:4d}  順位 {ranks[c]}/{len(cols)}")

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
    ap.add_argument("--arms", default="all",
                    help="all | rt | field （切り分け用）")
    args = ap.parse_args()

    print("データ読み込み ...", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    df = add_race_type(df)
    df = add_field_strength(df)
    df = add_line_strength(df)

    arms = [("baseline", FEATURE_COLS_WT)]
    if args.arms in ("all", "rt"):
        arms.append(("+race_type", FEATURE_COLS_WT + RT_COLS))
    if args.arms in ("all", "field"):
        arms.append(("+field", FEATURE_COLS_WT + FIELD_COLS))
    if args.arms in ("all", "line"):
        arms.append(("+line", FEATURE_COLS_WT + LINE_COLS))
    if args.arms in ("all", "line"):
        arms.append(("+rt_line", FEATURE_COLS_WT + RT_COLS + LINE_COLS))
    if args.arms == "all":
        arms.append(("+both", FEATURE_COLS_WT + RT_COLS + FIELD_COLS))
        arms.append(("+all3", FEATURE_COLS_WT + RT_COLS + FIELD_COLS + LINE_COLS))

    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt, arms)


if __name__ == "__main__":
    main()
