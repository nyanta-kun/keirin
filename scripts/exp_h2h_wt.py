"""対戦表（選手間 head-to-head 直接対戦成績）特徴の追加検証（netkeirin未活用データ調査・2026-07-28）。

発見: netkeirin(keirin.netkeiba.com)の各レースページに「対戦表」タブがあり、出走選手同士の
直近1年間の頭対頭勝敗（例: 福島栄一 vs 山田敦也 = 0-2）が表示される。この情報自体は
netkeirinからスクレイプせずとも、自前の wt_entries.finish_order 履歴から point-in-time で
再現可能（= 過去に同じレースへ出走し両者とも完走した際、どちらが先着したか、を全ペアで
累積するだけ）。

[[keirin_netkeirin_factor_import]]で検証済みの Elo（ペア比較を Bradley-Terry 的に集約した
グローバルレーティング）は AUC は改善したが SS的中率が動かず不採用だった。対戦表(H2H)は
Eloとは質的に異なる情報（グローバルな強さではなく「今日の対戦相手個々との相性・因縁」）
であり、Eloが吸収しきれない独立シグナルを持つ可能性がある一方、ペアごとのサンプル数が
非常に小さい（多くのペアは年に数回しか同走しない）ためノイズが支配的である懸念もある。

特徴量（point-in-time・全履歴累積、finish_order 1〜99 の完走者ペアのみ集計。DNF/欠車は除外）:
  h2h_win_rate  : 当該レース出走者のうち、過去に対戦履歴がある相手に対する勝率
                  （先着した回数 / 対戦回数、対戦履歴なしの相手は集計対象外・全体で0件ならNaN→0.5補完）
  h2h_n_total   : 当該レース出走者のうち、過去に対戦履歴がある相手との対戦回数の合計（カバレッジ）
  h2h_net_norm  : (先着数 - 後着数) の合計 / レース出走頭数（対戦履歴が無ければ0＝情報なしと同義）

検証は既存の exp_elo_linecoop_wt.py と同一ハーネス・分割・指標。
baseline / +h2h(3特徴) / +h2h_rate_only(win_rateのみ) を複数seed・クリーンOOSで比較。
本番 FEATURE_COLS_WT / lgbm_wt.pkl は変更しない。

クリーン分割: TRAIN 〜2026-03-31 / TEST(未使用OOS) 2026-04-01〜2026-06-30 / FWD 2026-07-01〜2026-07-10
"""
import sys
from collections import defaultdict
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

TR_TO = "2026-03-31"
TE_FROM, TE_TO = "2026-04-01", "2026-06-30"
FW_FROM, FW_TO = "2026-07-01", "2026-07-10"
SEEDS = [42, 7, 123, 2024, 99]
PARAMS = dict(objective="binary", metric="auc", n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              verbose=-1)

H2H_COLS = ["h2h_win_rate", "h2h_n_total", "h2h_net_norm"]


def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """レース時系列で point-in-time の対戦表(H2H)特徴を付与する。"""
    d = df[["race_key", "race_date", "start_at", "player_id", "finish_order"]].copy()
    d["fin"] = pd.to_numeric(d["finish_order"], errors="coerce")

    race_order = (d.groupby("race_key")
                  .agg(race_date=("race_date", "first"), start_at=("start_at", "first"))
                  .sort_values(["race_date", "start_at"])
                  .index.tolist())

    h2h_win: dict = defaultdict(int)   # (pid_a, pid_b) a<b → a が先着した回数
    h2h_n: dict = defaultdict(int)     # (pid_a, pid_b) a<b → 対戦回数（両者完走）

    groups = {rk: g for rk, g in d.groupby("race_key", sort=False)}
    out = {}

    for rk in race_order:
        g = groups[rk]
        pids = g["player_id"].tolist()
        fins = g["fin"].tolist()

        # --- 特徴（レース前の対戦履歴で） ---
        for p in pids:
            wins, matches, net = 0, 0, 0
            for q in pids:
                if q == p:
                    continue
                key = (p, q) if p < q else (q, p)
                n = h2h_n[key]
                if n == 0:
                    continue
                w_ab = h2h_win[key]  # min(p,q) が先着した回数
                w_p = w_ab if p < q else (n - w_ab)
                matches += n
                wins += w_p
                net += w_p - (n - w_p)
            out[(rk, p)] = (
                (wins / matches) if matches > 0 else np.nan,
                float(matches),
                float(net),
            )

        # --- 更新（レース後・完走者ペアのみ） ---
        finished = [(p, f) for p, f in zip(pids, fins) if f is not None and 1 <= f <= 99]
        for i in range(len(finished)):
            for j in range(i + 1, len(finished)):
                pa, fa = finished[i]
                pb, fb = finished[j]
                if fa == fb:
                    continue
                key = (pa, pb) if pa < pb else (pb, pa)
                h2h_n[key] += 1
                a_first = fa < fb
                a_is_min = pa < pb
                if a_first == a_is_min:
                    h2h_win[key] += 1

    df = df.copy()
    key = list(zip(df["race_key"], df["player_id"]))
    vals = [out.get(k, (np.nan, 0.0, 0.0)) for k in key]
    df["h2h_win_rate"] = [v[0] for v in vals]
    df["h2h_n_total"] = [v[1] for v in vals]
    df["h2h_net_norm"] = [v[2] for v in vals]
    ne = df.groupby("race_key")["player_id"].transform("count").replace(0, np.nan)
    df["h2h_net_norm"] = (df["h2h_net_norm"] / ne).fillna(0.0)
    df["h2h_win_rate"] = df["h2h_win_rate"].fillna(0.5)
    return df


def race_metrics(df_ev):
    both_top3, win1, top3_1, n = 0, 0, 0, 0
    for _, g in df_ev.groupby("race_key"):
        g = g[g["finish_order"] >= 1]
        if len(g) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        fo = g["finish_order"].astype(float).tolist()
        n += 1
        win1 += fo[0] == 1
        top3_1 += 1 <= fo[0] <= 3
        both_top3 += (1 <= fo[0] <= 3) and (1 <= fo[1] <= 3)
    if n == 0:
        return dict(ss=0.0, win1=0.0, top3_1=0.0)
    return dict(ss=both_top3 / n, win1=win1 / n, top3_1=top3_1 / n)


def main():
    print("データ構築中...")
    raw = load_raw_data_wt(min_date="2022-12-01", max_date=FW_TO)
    print(f"  raw rows={len(raw)}")
    raw = compute_h2h(raw)
    cov = (raw["h2h_n_total"] > 0).mean()
    print(f"  H2H計算完了 (対戦履歴カバレッジ={cov:.2%}, "
          f"h2h_n_total平均={raw['h2h_n_total'].mean():.2f}, "
          f"h2h_win_rate分布: min={raw['h2h_win_rate'].min():.2f} "
          f"max={raw['h2h_win_rate'].max():.2f})")
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
        "+h2h": list(FEATURE_COLS_WT) + H2H_COLS,
        "+h2h_rate_only": list(FEATURE_COLS_WT) + ["h2h_win_rate"],
    }
    agg = {v: defaultdict(list) for v in variants}

    for seed in SEEDS:
        for vname, cols in variants.items():
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(tr[cols].fillna(0).values, tr[TARGET_COL_WT].values)
            for tag, ev in (("te", te), ("fw", fw)):
                ev = ev.copy()
                ev["pred_prob"] = m.predict_proba(ev[cols].fillna(0).values)[:, 1]
                agg[vname][f"auc_{tag}"].append(
                    roc_auc_score(ev[TARGET_COL_WT], ev["pred_prob"]))
                mt = race_metrics(ev)
                agg[vname][f"ss_{tag}"].append(mt["ss"])
                agg[vname][f"win1_{tag}"].append(mt["win1"])
                agg[vname][f"top3_{tag}"].append(mt["top3_1"])
        print(f"  seed {seed} done")

    ms = lambda a: (np.mean(a), np.std(a))
    print("\n============= 結果（seed平均 ± std, n=%d）=============" % len(SEEDS))
    for tag, label in (("te", "TEST 2026-04〜06 (クリーンOOS)"), ("fw", "FWD 2026-07")):
        print(f"\n--- {label} ---")
        print(f"{'variant':<16}{'AUC':>16}{'SS的中(2車3着内)':>22}{'1位勝率':>13}{'1位複勝率':>13}")
        base = agg["baseline"]
        for v in variants:
            a = agg[v]
            am, asd = ms(a[f"auc_{tag}"]); sm, ssd = ms(a[f"ss_{tag}"])
            wm, wsd = ms(a[f"win1_{tag}"]); tm, tsd = ms(a[f"top3_{tag}"])
            dss = sm - np.mean(base[f"ss_{tag}"])
            mk = "" if v == "baseline" else ("  ★" if dss > ssd else ("  ×" if dss < -ssd else "  ~"))
            print(f"{v:<16}{am:>7.4f}±{asd:.4f}{sm:>13.1%}±{ssd:.1%}"
                  f"{wm:>7.1%}±{wsd:.1%}{tm:>7.1%}±{tsd:.1%}{mk}")
    print("\n判定: SSΔ>seed std で ★(採用候補)。採用は TEST・FWD 双方で非悪化かつ TEST で ★。")


if __name__ == "__main__":
    main()
