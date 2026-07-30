"""axis_sum(モデル選定軸2車の複勝確率合計) と mark_sum(◎◯複勝確率合計) の
乖離(delta)を検証（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

前段でmark_sumゲートをD構成にそのまま追加しても効果がなかった（axis_sumが既に
ほぼ同じ母集団を絞っていたため冗長）。ここでは仮説を変える: **axis_sumとmark_sumの
「値」ではなく「乖離幅」delta = mark_sum - axis_sum が追加情報を持つか**を検証する。

delta > 0: 市場印(◎◯)の複勝確率合計の方が、モデルが選んだ軸2車の複勝確率合計より
  高い＝モデルの軸選定が◎◯より"弱い"組み合わせを選んでいる（wt_overlap_n=1で
  axis2が◯ではない場合などに発生）。
delta <= 0: モデルの軸選定の方が◎◯自身より強い（または同等＝重なりが大きい）。

母集団はA構成（wt_overlap_n∈{0,1}のみ、最も広い母集団＝統計的検出力を優先）を使い、
TRAIN(2024-01-01〜2025-12-31)でdeltaと的中/ROIの関係を確認→有望な閾値があれば
TEST(2026-01-01〜2026-07-28)で一度だけ検証する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

S7_AXIS_SUM_MAX = 1.3  # 参考: D構成のaxis_sumゲート（本スクリプトのA構成では未適用）

TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20260728"


def a_filter(c):
    return c["wt_overlap_n"] in (0, 1)


def d_filter(c):
    return (c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX
            and c["entropy"] <= 1.8329
            and (c["wt_mark3_overlap_n"] is not None and c["wt_mark3_overlap_n"] <= 1))


def load_mark_sum(race_keys):
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, prediction_mark, pred_top3_pct FROM wt_entries "
                 "WHERE race_key IN (%s) AND prediction_mark IN (1, 2)"
                 % ",".join("?" * len(chunk)))
            per_race = {}
            for rk, pm, pct in c.execute(q, chunk):
                if pct is None:
                    continue
                per_race.setdefault(rk, {})[int(pm)] = float(pct)
            for rk, marks in per_race.items():
                if 1 in marks and 2 in marks:
                    # pred_top3_pct は0-100のパーセント表記。axis_sumは
                    # model.predict_proba()の0-1確率(top3_probs)の合計なので
                    # 100で割って単位を揃える（揃えないとdeltaが常に約100ずれる）。
                    out[rk] = (marks[1] + marks[2]) / 100.0
    return out


def summarize(cands):
    n = len(cands)
    hits = sum(c["hit"] for c in cands)
    bet = sum(c["bet_amount"] for c in cands)
    pay = sum(c["payout"] for c in cands)
    roi = pay / bet * 100 if bet else 0.0
    hitrate = hits / n * 100 if n else 0.0
    return n, hits, hitrate, bet, pay, roi


def main():
    import pickle
    cache_path = (Path(__file__).resolve().parent / ".." / "data" / "exp_cache"
                  / "s7_staged_audit_candidates.pkl").resolve()
    with open(cache_path, "rb") as f:
        all_cands = pickle.load(f)

    a_sel = [c for c in all_cands if a_filter(c)]
    print(f"A構成(wt_overlap_n in {{0,1}}) 全体: {len(a_sel)}件")

    # axis_sumはpred_top3_pctの生値と単位が違う可能性があるため確認
    # (s7_select_axisはtop3_probs=pred_top3_pctをそのまま使うので同一単位のはず)
    mark_sum_map = load_mark_sum([c["race_key"] for c in a_sel])
    for c in a_sel:
        ms = mark_sum_map.get(c["race_key"])
        c["mark_sum"] = ms
        c["delta"] = (ms - c["axis_sum"]) if ms is not None else None

    a_sel = [c for c in a_sel if c["delta"] is not None]
    print(f"mark_sum突合できた件数: {len(a_sel)}")
    print(f"axis_sum範囲: {min(c['axis_sum'] for c in a_sel):.1f}-{max(c['axis_sum'] for c in a_sel):.1f}"
          f"  mark_sum範囲: {min(c['mark_sum'] for c in a_sel):.1f}-{max(c['mark_sum'] for c in a_sel):.1f}")

    train = [c for c in a_sel if TRAIN_FROM <= c["race_key"][:8] <= TRAIN_TO]
    test = [c for c in a_sel if TEST_FROM <= c["race_key"][:8] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    n0, h0, hr0, bet0, pay0, roi0 = summarize(train)
    print(f"\n[TRAIN] A構成全体(deltaゲートなし): n={n0} hit={hr0:.1f}% ROI={roi0:.1f}%")

    # deltaの分布を十分位で分割
    train_sorted = sorted(train, key=lambda c: c["delta"])
    n = len(train_sorted)
    print(f"\n[TRAIN] delta十分位別ROI")
    print(f"{'decile':<10}{'delta範囲':<20}{'n':>8}{'hit%':>8}{'ROI':>10}")
    deciles_edges = []
    for i in range(10):
        lo_idx = n * i // 10
        hi_idx = n * (i + 1) // 10
        chunk = train_sorted[lo_idx:hi_idx]
        if not chunk:
            continue
        lo_d, hi_d = chunk[0]["delta"], chunk[-1]["delta"]
        deciles_edges.append((lo_d, hi_d))
        nn, hits, hitrate, bet, pay, roi = summarize(chunk)
        mark = " ★100%超" if roi > 100 else ""
        print(f"d{i+1:<9}{lo_d:>7.1f}〜{hi_d:>7.1f}    {nn:>8}{hitrate:>7.1f}%{roi:>9.1f}%{mark}")

    # 相関確認
    def spearman(data, key1, key2):
        n = len(data)
        r1 = {id(x): i for i, x in enumerate(sorted(data, key=lambda c: c[key1]))}
        r2 = {id(x): i for i, x in enumerate(sorted(data, key=lambda c: c[key2]))}
        xs = [r1[id(c)] for c in data]
        ys = [r2[id(c)] for c in data]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        return cov / (vx ** 0.5 * vy ** 0.5) if vx > 0 and vy > 0 else 0.0

    print(f"\ndelta vs hit の順位相関(TRAIN): {spearman(train, 'delta', 'hit'):.3f}")
    print(f"axis_sum vs hit の順位相関(TRAIN): {spearman(train, 'axis_sum', 'hit'):.3f}")
    print(f"mark_sum vs hit の順位相関(TRAIN): {spearman(train, 'mark_sum', 'hit'):.3f}")

    # TRAINで最良のdelta閾値（上限or下限）を探索しTESTで確認
    print("\n[TRAIN] delta上限ゲート探索（delta<=th）")
    best_th, best_roi, best_n = None, -1, 0
    for th in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 9.0]:
        sub = [c for c in train if c["delta"] <= th]
        nn, hits, hitrate, bet, pay, roi = summarize(sub)
        print(f"  delta<={th:<6} n={nn:>6} hit={hitrate:>5.1f}% ROI={roi:>7.1f}%")
        if nn >= 500 and roi > best_roi:
            best_roi, best_th, best_n = roi, th, nn

    print("\n[TRAIN] delta下限ゲート探索（delta>=th、乖離が大きい側を残す）")
    for th in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]:
        sub = [c for c in train if c["delta"] >= th]
        nn, hits, hitrate, bet, pay, roi = summarize(sub)
        print(f"  delta>={th:<6} n={nn:>6} hit={hitrate:>5.1f}% ROI={roi:>7.1f}%")

    if best_th is not None:
        print(f"\n[選定] TRAIN最良(n>=500制約): delta<={best_th} (ROI={best_roi:.1f}%, n={best_n})")
        sub_test = [c for c in test if c["delta"] <= best_th]
        nn, hits, hitrate, bet, pay, roi = summarize(sub_test)
        print(f"[TEST] delta<={best_th} 一度だけ適用: n={nn} hit={hitrate:.1f}% ROI={roi:.1f}%")


if __name__ == "__main__":
    main()
