"""大敗（6着以下）を直接予測するヘッドの検証（2026-08-04）。

ユーザー要望:
  「軸2で惜敗は許容できるが大敗が多すぎる。ここの精度向上が必要」

背景（scripts/exp_low_mark_top3_rate.py・honest 36,831レース）:
  軸2の大敗率（6着以下）は WT印で大きく違う — ◎12.5% / ◯11.4% /
  △16.4% / ×19.8% / 無印24.6%。かつ△以下では p2 を 2.8〜4.4pt 過大評価している。
  一方で「△以下の中から走る車を選ぶ」選別能力は明確にある（+27.5pt）。

**本検証の中心論点**:
  「6着以下確率」は「3着内確率の裏返し」ではないのか。3着内モデルの較正は
  honest 全期間で ±1.2pt 以内と正確なので、単に `-pred_prob` で順位づけるのと
  変わらないなら**専用ヘッドを持つ意味はない**。したがって
    (A) 大敗ヘッド（bad_flag を直接学習）
    (B) ベースライン = 3着内モデルの `-pred_prob` をそのまま大敗スコアとする
  を **同じ土俵で比較**し、(A) が (B) を超えるかを測る。
  JRAでは同型の out_probability が足切りに有効だったが、地方競馬では
  「較正済みの指数差ルールと同等」で不採用になっている。競輪では未検証。

測定内容:
  ① 大敗の予測可能性: (A) vs (B) の AUC / PR-AUC
  ② 軸2に適用した効果: 大敗スコア上位を除外したときの
     「除外率 / 除外した軸2の実大敗率 / 取りこぼした3着内」
  ③ DNF（欠車・失格）を大敗に含めるか否かの比較
     （DNF率は印によらず0.9〜1.3%でほぼ一定＝予測不能。含めるとノイズになりうる）

検証: 2窓 × 5seed・exp_racetype_field_ab.py と同一方法論。DB書き込みなし。

使い方:
    python scripts/exp_bad_finish_head.py [--windows w1,w2]
"""
from __future__ import annotations

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
from src.strategy_wt import rank_7s_select_axis

TRAIN_FROM = "2024-04-01"
WINDOWS = {
    "w1": ("2026-04-13", "2026-07-15"),
    "w2": ("2026-01-01", "2026-04-12"),
}
SEEDS = [42, 101, 202, 303, 404]


def add_bad_targets(df: pd.DataFrame) -> pd.DataFrame:
    """大敗ラベルを2種類作る。

    bad6      : 完走して6着以下（DNFは除外＝NaN扱いにせず0にしない）
    bad6_dnf  : 6着以下 または 欠車・失格
    """
    out = df.copy()
    fo = pd.to_numeric(out["finish_order"], errors="coerce")
    finished = fo >= 1
    out["bad6"] = ((fo >= 6) & finished).astype(int)
    out["bad6_dnf"] = (((fo >= 6) & finished) | (~finished)).astype(int)
    out["_finished"] = finished.astype(int)
    return out


def race_axis_map(test: pd.DataFrame, prob_top3: np.ndarray,
                  prob_win: np.ndarray) -> dict[str, tuple[int, int]]:
    """本番と同じ rank_7s_select_axis で軸2車を決める。"""
    t = test.copy()
    t["pp3"] = prob_top3
    t["ppw"] = prob_win
    out: dict[str, tuple[int, int]] = {}
    for rk, g in t.groupby("race_key"):
        if len(g) != 7:
            continue
        win_probs = {int(r.frame_no): float(r.ppw) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pp3) for r in g.itertuples(index=False)}
        sel = rank_7s_select_axis(win_probs, top3_probs)
        if sel:
            out[rk] = (sel[0], sel[1])
    return out


def run_window(df: pd.DataFrame, test_from: str, test_to: str) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    train = df[(df["race_date"] >= TRAIN_FROM) & (df["race_date"] < test_from)]
    test = df[(df["race_date"] >= test_from) & (df["race_date"] <= test_to)]
    print(f"\n######## 窓 test={test_from}〜{test_to}  "
          f"train {len(train):,}行 / test {len(test):,}行 ########", flush=True)
    for lbl in ("bad6", "bad6_dnf"):
        print(f"  {lbl} ベースレート: train {train[lbl].mean()*100:.1f}% / "
              f"test {test[lbl].mean()*100:.1f}%")

    def fit(target: str, seed: int):
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=seed,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(train[FEATURE_COLS_WT], train[target])
        return m.predict_proba(test[FEATURE_COLS_WT])[:, 1]

    # 3着内モデル（ベースライン用・大敗スコアは -p3）
    p3_list, pw_list = [], []
    for seed in SEEDS:
        p3_list.append(fit(TARGET_COL_WT, seed))
        pw_list.append(fit("win_flag", seed))
    p3 = np.mean(p3_list, axis=0)
    pw = np.mean(pw_list, axis=0)

    print("\n【① 大敗の予測可能性: 専用ヘッド vs 3着内モデルの裏返し】")
    print(f"  {'ラベル':10} {'手法':28} {'AUC':>9} {'PR-AUC':>9}")
    heads: dict[str, np.ndarray] = {}
    for lbl in ("bad6", "bad6_dnf"):
        y = test[lbl].values
        # (B) ベースライン: 3着内確率の符号反転
        auc_b = roc_auc_score(y, -p3)
        ap_b = average_precision_score(y, -p3)
        # (A) 専用ヘッド
        preds = [fit(lbl, s) for s in SEEDS]
        pa = np.mean(preds, axis=0)
        heads[lbl] = pa
        auc_a = roc_auc_score(y, pa)
        ap_a = average_precision_score(y, pa)
        print(f"  {lbl:10} {'(B) -pred_prob（裏返し）':28} {auc_b:9.5f} {ap_b:9.5f}")
        print(f"  {lbl:10} {'(A) 専用ヘッド':28} {auc_a:9.5f} {ap_a:9.5f}")
        print(f"  {lbl:10} {'差 (A)-(B)':28} {auc_a-auc_b:+9.5f} {ap_a-ap_b:+9.5f}")
        # 相関（裏返しとどれだけ同じものを見ているか）
        r = np.corrcoef(pa, -p3)[0, 1]
        print(f"  {lbl:10} {'(A)と(B)の相関':28} {r:9.4f}")
    print()

    # ---------------------------------------------------------------- ②軸2適用
    print("【② 軸2に適用: 大敗スコア上位を除外したときの効果】")
    axes = race_axis_map(test, p3, pw)
    t = test.copy()
    t["pp3"] = p3
    t["pbad"] = heads["bad6"]
    idx = {(r.race_key, int(r.frame_no)): i for i, r in enumerate(t.itertuples(index=False))}
    fo_map = {(r.race_key, int(r.frame_no)): r.finish_order
              for r in t.itertuples(index=False)}

    rows = []
    for rk, (a1, a2) in axes.items():
        i2 = idx.get((rk, a2))
        i1 = idx.get((rk, a1))
        if i2 is None or i1 is None:
            continue
        f2 = fo_map.get((rk, a2))
        f1 = fo_map.get((rk, a1))
        if f2 is None or f2 != f2 or f1 is None or f1 != f1:
            continue
        rows.append({
            "rk": rk,
            "bad2": t["pbad"].values[i2], "p2": t["pp3"].values[i2],
            "in3_2": 1 <= int(f2) <= 3, "in3_1": 1 <= int(f1) <= 3,
            "bad_actual2": int(f2) >= 6,
        })
    if not rows:
        print("  対象レースなし")
        return
    R = pd.DataFrame(rows)
    n = len(R)
    base_both = (R["in3_1"] & R["in3_2"]).mean()
    print(f"  対象 {n} レース / 軸2の実大敗率 {R['bad_actual2'].mean()*100:.1f}% / "
          f"両方3着内 {base_both*100:.1f}%")
    print(f"  {'除外基準':26} {'除外率':>7} {'除外分の実大敗率':>16} "
          f"{'残りの両方3着内':>15} {'取りこぼした3着内':>16}")
    for q in (0.10, 0.20, 0.30):
        for key, name in (("bad2", "大敗ヘッド上位"), ("p2", "p2下位（裏返し）")):
            thr = R[key].quantile(1 - q if key == "bad2" else q)
            drop = R[key] >= thr if key == "bad2" else R[key] <= thr
            keep = ~drop
            if drop.sum() == 0 or keep.sum() == 0:
                continue
            lost = (R.loc[drop, "in3_1"] & R.loc[drop, "in3_2"]).sum()
            tot_hit = (R["in3_1"] & R["in3_2"]).sum()
            print(f"  {name+f' {int(q*100)}%':26} {drop.mean()*100:6.1f}% "
                  f"{R.loc[drop,'bad_actual2'].mean()*100:15.1f}% "
                  f"{(R.loc[keep,'in3_1'] & R.loc[keep,'in3_2']).mean()*100:14.1f}% "
                  f"{100*lost/max(tot_hit,1):15.1f}%")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2")
    args = ap.parse_args()

    print(f"データ読み込み ... ({len(FEATURE_COLS_WT)}特徴)", flush=True)
    max_to = max(t for _, t in WINDOWS.values())
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=max_to))
    df = add_bad_targets(df)
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()
    print(f"7車立てに限定: {len(df):,}行")

    for w in args.windows.split(","):
        tf, tt = WINDOWS[w.strip()]
        run_window(df, tf, tt)


if __name__ == "__main__":
    main()
