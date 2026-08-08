#!/usr/bin/env python3
"""RANK_9H1 の「波乱スコア」モデル（`lgbm_upset_screen`）を学習する。

## 何を当てるモデルか

**レース単位**で「そのレースが高配当で決着するか」を当てる二値分類。
特徴は `src/preprocessing/upset_features.py` の単一正本を通す（オッズ非依存）。

## 🔴 6/7/9車を統合して学習する

9車は 4,738R しか無く、9車だけで学習すると効果が検出できない
（Δratio +0.090・95%CI 下限 +0.001）。6/7/9車を統合すると同じ9車の評価が
**Δratio +0.131 [+0.028, +0.230]・月次一貫性 75%→90%** へ改善する。
6車は単独では成立しない（月次42%）が、**外すと9車が壊れる**（+0.131→+0.070）。

## 🔴 目的変数は「車数ごとの分位」で定義する

「決着オッズ >= 300倍」という絶対閾値のまま統合すると、基準率が
7車 9.8% / 9車 24.0% と違うため、**モデルは主に『これは9車か』を当てるだけ**に
なる。車数は既知なので、それを当てても9車の中の順位付けは1ミリも良くならない。

車数ごとに閾値を引き直すことで「**同じ車数の中で相対的に荒れやすいか**」という
**転移する構造**だけを学ばせる。

## 使い方

    # 本番モデル（全期間で学習）
    PYTHONPATH=. .venv/bin/python scripts/train_upset_screen.py

    # 過去分の再構築用に、ある時点までのデータだけで学習した vintage を作る
    PYTHONPATH=. .venv/bin/python scripts/train_upset_screen.py \\
        --train-end 2025-06-30 --name lgbm_upset_screen_m2507

⚠️ **本番モデルで過去を再スコアしてはいけない**（全期間学習を過去へ遡って適用する
   model-vintage look-ahead になる）。`backfill_9h1_rank_wt.py` は必ず vintage を使う。

⚠️ 学習後は**9車のスコア分布を必ず確認**し、`strategy_wt.RANK_9H1_SCORE_MIN` を
   上位20%点へ引き直すこと（本スクリプトが最後に出力する）。放置すると
   推奨件数が数倍に振れる。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.models.trainer import save_model  # noqa: E402
from src.preprocessing.upset_features import (  # noqa: E402
    UPSET_FEATURE_COLS, build_upset_row, feature_vector,
)

#: 学習に使う車数。9車だけでは標本が足りないので統合する（上記参照）。
POOL_N_ENTRIES = (6, 7, 9)

#: 「波乱」とみなす分位。車数ごとに引き直す（上位25% = 各車数の75%点以上）。
TARGET_Q = 0.25

ODDS_MAX = 9000.0   # winticket の未確定センチネル 9999.9 を捨てる


def _load(n_list: tuple[int, ...], train_end: str | None) -> list[dict]:
    """レース単位の (特徴, 決着した三連単オッズ) を返す。

    ⚠️ 絞り込みは `wt_races` への JOIN で書くこと。`race_key IN (SELECT ...)` に
    書き換えると `wt_odds`（2,200万行）のプランが崩れて15秒→15分以上になる。
    """
    ne_in = ",".join(str(x) for x in n_list)
    end = f"AND r.race_date <= '{train_end}'" if train_end else ""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT e.race_key, r.race_date, r.n_entries, r.grade, r.race_type,
                   r.day_index, r.distance, r.start_at,
                   e.frame_no, e.race_point, e.line_group, e.line_size,
                   e.style, e.player_class, e.s_count, e.b_count,
                   e.first_rate, e.third_rate, e.prediction_mark, e.finish_order,
                   v.bank_length, v.is_indoor
            FROM wt_entries e
            JOIN wt_races r USING(race_key)
            LEFT JOIN venue_info v ON v.venue_code = r.venue_id
            WHERE r.cancel=0 AND r.n_entries IN ({ne_in}) {end}
        """)
        by_race: dict[str, list[dict]] = defaultdict(list)
        for e in cur:
            by_race[e["race_key"]].append(dict(e))

        cur.execute(f"""
            WITH fin AS (
              SELECT e.race_key,
                     concat(max(CASE WHEN e.finish_order=1 THEN e.frame_no END), '-',
                            max(CASE WHEN e.finish_order=2 THEN e.frame_no END), '-',
                            max(CASE WHEN e.finish_order=3 THEN e.frame_no END)) combo
              FROM wt_entries e JOIN wt_races r USING(race_key)
              WHERE r.cancel=0 AND r.n_entries IN ({ne_in}) {end}
              GROUP BY e.race_key)
            SELECT o.race_key,
                   max(CASE WHEN o.combination=f.combo THEN o.odds_value END) win_odds
            FROM wt_odds o
            JOIN wt_races r USING(race_key)
            JOIN fin f ON f.race_key=o.race_key
            WHERE o.bet_type='trifecta' AND r.cancel=0 AND r.n_entries IN ({ne_in}) {end}
              AND o.odds_value>0 AND o.odds_value<{ODDS_MAX}
            GROUP BY o.race_key
        """)
        win = {o["race_key"]: o["win_odds"] for o in cur if o["win_odds"] is not None}

    rows = []
    for rk, ents in by_race.items():
        if rk not in win:
            continue
        race = {k: ents[0].get(k) for k in
                ("n_entries", "grade", "race_type", "day_index", "distance",
                 "start_at", "bank_length", "is_indoor")}
        feat = build_upset_row(ents, race)
        if feat is None:
            continue          # 事前欠車で行数が合わないレースは母集団外
        rows.append({"race_key": rk, "date": ents[0]["race_date"],
                     "win_odds": float(win[rk]), "feat": feat})
    return rows


def _target(rows: list[dict], q: float) -> np.ndarray:
    """車数ごとの分位で「波乱」を定義する（絶対閾値にしてはいけない・上記参照）。"""
    y = np.zeros(len(rows))
    for ne in {r["feat"]["n_entries"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["feat"]["n_entries"] == ne]
        thr = float(np.quantile([rows[i]["win_odds"] for i in idx], 1 - q))
        for i in idx:
            y[i] = 1.0 if rows[i]["win_odds"] >= thr else 0.0
        print(f"  {int(ne)}車 n={len(idx):>6}  波乱の閾値 {thr:>7.1f}倍")
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="lgbm_upset_screen", help="保存するモデル名")
    ap.add_argument("--train-end", default=None,
                    help="この日までのデータだけで学習する（vintage 用・YYYY-MM-DD）")
    ap.add_argument("--target-q", type=float, default=TARGET_Q)
    ap.add_argument("--force", action="store_true", help="既存モデルを上書きする")
    args = ap.parse_args()

    import lightgbm as lgb

    rows = _load(POOL_N_ENTRIES, args.train_end)
    if len(rows) < 5000:
        raise SystemExit(f"学習データが少なすぎます: {len(rows)}R")
    print(f"学習データ {len(rows)}R"
          f"（{min(r['date'] for r in rows)}〜{max(r['date'] for r in rows)}）")
    y = _target(rows, args.target_q)
    X = np.array([feature_vector(r["feat"]) for r in rows], dtype=float)
    print(f"特徴 {len(UPSET_FEATURE_COLS)}本 / 陽性率 {y.mean()*100:.1f}%")

    model = lgb.train(
        {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15,
         "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.8,
         "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1, "seed": 0},
        lgb.Dataset(X, label=y, feature_name=list(UPSET_FEATURE_COLS)),
        num_boost_round=350)
    save_model(model, args.name, force=args.force)
    print(f"保存: {args.name}")

    imp = sorted(zip(UPSET_FEATURE_COLS, model.feature_importance("gain")),
                 key=lambda kv: -kv[1])
    print("\n重要度 上位10:")
    for name, gain in imp[:10]:
        print(f"  {name:<24}{gain:>12,.0f}")

    # 🔴 閾値の再較正。RANK_9H1_SCORE_MIN はこの分布の上位20%点でなければならない
    n9 = [i for i, r in enumerate(rows) if r["feat"]["n_entries"] == 9]
    if n9:
        s9 = model.predict(X[n9])
        from src.strategy_wt import RANK_9H1_SCORE_MIN
        p80 = float(np.quantile(s9, 0.80))
        rate = float((s9 >= RANK_9H1_SCORE_MIN).mean()) * 100
        print(f"\n9車スコア分布 n={len(n9)}: "
              f"p50={np.quantile(s9, .5):.4f} **p80={p80:.4f}** p90={np.quantile(s9, .9):.4f}")
        print(f"現行 RANK_9H1_SCORE_MIN = {RANK_9H1_SCORE_MIN:.4f} → 該当率 {rate:.1f}%")
        # 🔴 判定は「スコアの差」ではなく**該当率のずれ**で行う。スコアの絶対差が
        #    0.02 でも該当率は7pt動くことがあり、件数はそちらに比例するため。
        if abs(rate - 20.0) > 3.0:
            print(f"⚠️ 該当率が目標20%から外れている。strategy_wt.RANK_9H1_SCORE_MIN を "
                  f"{p80:.4f} へ更新すること（放置すると推奨件数が {rate/20:.2f} 倍に振れる）")
        # vintage を作った場合は、その vintage 用の閾値としてこの値を控えておく
        if args.train_end:
            print(f"→ この vintage の閾値: rank_9h1_daily_select(..., score_min={p80:.4f})")


if __name__ == "__main__":
    main()
