"""波乱の解剖（探索分析）: 波乱の定義・頻度 / 発生の事前条件 / 発生時の指数×ライン

「軸1位ありき」を仮定せず、波乱レースの正体を多角的に記述する。
- 指数 = eval model(lgbm_wt_eval) の pred_prob（OOS寄り）。≤6車・実top3確定レース。
- 波乱の代理: ①line_break(1着と2着が別line_group) ②三連複払戻(高配当) ③指数top3との不一致。
期間は volume 確保のため広めに取る（指数は記述目的。最終オッズ=上限値）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.strategy_wt import upset_tier
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _filter_by_n_riders, _load_payouts_wt

FROM, TO = "2025-06-01", "2026-06-08"
STYLE = {0: "逃/先行", 1: "両", 2: "追/差し", -1: "?"}


def build():
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=FROM, max_date=TO))
    df = _apply_pred_prob_wt(model, df); df = _filter_by_n_riders(df, 6)
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(g)
        if n < 3:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 < 0.06:   # 賭け対象ユニバースに合わせる
            continue
        idx_frame = g["frame_no"].astype(int).tolist()          # 指数順の車番
        rank_of = {int(f): i for i, f in enumerate(idx_frame)}   # 車番→指数rank(0始まり)
        line_of = {int(r.frame_no): r.line_group for r in g.itertuples()}
        style_of = {int(r.frame_no): int(r.style_enc) for r in g.itertuples()}
        leader_of = {int(r.frame_no): int(r.is_line_leader) for r in g.itertuples()}
        fav_line = g.loc[0, "line_group"]
        fin = fin.sort_values("finish_order")
        order = [int(f) for f in fin["frame_no"]]                # 実1,2,3着の車番
        top3 = set(order)
        # 波乱代理
        line_break = line_of[order[0]] != line_of[order[1]]
        idx_in_top3 = sum(1 for f in top3 if rank_of[f] <= 2)    # 指数top3が実top3に何人
        n_fav_in_top3 = sum(1 for f in top3 if line_of[f] == fav_line)
        n_lines_in_top3 = len({line_of[f] for f in top3})
        # 三連複 実払戻
        trio = pm.get(rk, {}).get(("trio", frozenset(top3)))
        # 実top3の指数rank（0=1位）
        idx_ranks = sorted(rank_of[f] for f in top3)
        worst = order[int(np.argmax([rank_of[f] for f in order]))]  # 実top3で最も指数が低い"伏兵"
        rows.append({
            "ut": upset_tier(p[0]+p[1]+p[2]),
            "n": n, "gap12": gap12,
            "n_lines": int(g.loc[0, "n_lines"]) if "n_lines" in g else 0,
            "n_iso": int((g["line_size"] == 1).sum()) if "line_size" in g else 0,
            "n_senko": sum(1 for v in style_of.values() if v == 0),
            "line_break": line_break,
            "idx_in_top3": idx_in_top3,
            "n_fav_in_top3": n_fav_in_top3,
            "n_lines_in_top3": n_lines_in_top3,
            "trio": trio if trio else np.nan,
            "p1_in": rank_of[order[0]] == 0 or (0 in [rank_of[f] for f in top3]),
            "idx1_in_top3": 0 in [rank_of[f] for f in top3],
            "idx_rank_max": max(rank_of[f] for f in top3),     # 伏兵の指数rank
            "worst_style": STYLE.get(style_of[worst], "?"),
            "worst_isfav": line_of[worst] == fav_line,
            "worst_leader": leader_of[worst],
        })
    return pd.DataFrame(rows)


def main():
    d = build()
    n = len(d)
    print(f"\n{'#'*86}\n  波乱の解剖（{FROM}〜{TO}・≤6車・gap12≥0.06・{n}R・指数=eval/OOS寄り）\n{'#'*86}")

    # ── Part1: 波乱の定義と頻度 ──
    print(f"\n【Part1: 波乱代理指標の全体像】")
    print(f"  line_break(1-2着が別ライン)率: {d['line_break'].mean():.0%}")
    print(f"  指数top3が実top3を当てた人数(平均/3): {d['idx_in_top3'].mean():.2f}")
    print(f"  実top3のライン数(平均): {d['n_lines_in_top3'].mean():.2f}")
    print(f"  三連複払戻 中央値 {d['trio'].median():.0f}円 / 平均 {d['trio'].mean():.0f}円")
    print(f"  指数1位が3着内: {d['idx1_in_top3'].mean():.0%}")

    print(f"\n【Part1b: top3_sum四分位 × 波乱代理】")
    print(f"  {'帯':<9}{'R':>5}{'line_break':>11}{'指数的中/3':>11}{'実top3ライン数':>13}{'三連複中央':>11}{'指数1位3着内':>12}")
    for ut in ["Q1_loose", "Q2", "Q3", "Q4_chalk"]:
        s = d[d["ut"] == ut]
        if len(s) == 0: continue
        print(f"  {ut:<9}{len(s):>5}{s['line_break'].mean():>10.0%}{s['idx_in_top3'].mean():>11.2f}"
              f"{s['n_lines_in_top3'].mean():>13.2f}{s['trio'].median():>9.0f}円{s['idx1_in_top3'].mean():>11.0%}")

    # ── Part2: 発生の事前条件（line_break率で層別）──
    print(f"\n【Part2: 事前条件 × line_break率】（条件ごとの波乱発生率）")
    def by(col, bins=None, labels=None):
        print(f"\n  ▼ {col}")
        x = d[col]
        if bins is not None:
            x = pd.cut(x, bins=bins, labels=labels)
        for v, s in d.groupby(x):
            if len(s) < 10: continue
            print(f"    {str(v):<14} n={len(s):>4}  line_break={s['line_break'].mean():>4.0%}  "
                  f"指数的中={s['idx_in_top3'].mean():.2f}  三連複中央={s['trio'].median():>6.0f}円")
    by("n_lines")
    by("n_iso")
    by("n_senko")
    by("gap12", bins=[0.06, 0.10, 0.15, 0.25, 1.0], labels=["0.06-0.10","0.10-0.15","0.15-0.25","0.25+"])

    # ── Part3: 波乱(line_break)時の指数×ライン ──
    lb = d[d["line_break"]]
    print(f"\n{'='*86}\n【Part3: 波乱(line_break={len(lb)}R)時の指数×ライン】\n{'='*86}")
    print(f"  実top3に占める本命ライン人数(平均): {lb['n_fav_in_top3'].mean():.2f}/3"
          f"（非波乱: {d[~d['line_break']]['n_fav_in_top3'].mean():.2f}）")
    print(f"  指数1位が3着内: {lb['idx1_in_top3'].mean():.0%}")
    print(f"  伏兵(実top3で最低指数)の指数rank分布（0=1位,5=最下位）:")
    vc = lb["idx_rank_max"].value_counts().sort_index()
    for r, c in vc.items():
        print(f"    rank{r}: {c}R ({c/len(lb):.0%})")
    print(f"  伏兵の脚質: " + " / ".join(f"{k}:{v}" for k, v in lb["worst_style"].value_counts().items()))
    print(f"  伏兵が本命ライン所属: {lb['worst_isfav'].mean():.0%} / ライン先頭: {lb['worst_leader'].mean():.0%}")
    print(f"\n  ※ 示唆: line_breakでも指数1位は{lb['idx1_in_top3'].mean():.0%}来る一方、"
          f"top3の{3-lb['n_fav_in_top3'].mean():.1f}人が本命ライン外。伏兵の指数rank・脚質に偏りがあれば狙い目。")


if __name__ == "__main__":
    main()
