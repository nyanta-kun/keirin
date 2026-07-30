"""mark_sum(◎◯合算複勝確率)ゲートをS7(D構成=現行本番定義)に追加した場合の
honest検証（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

前段(`exp_honmei_taikou_both_top3_predict.py`)でmark_sumが「◎◯両方3着内」の
発走前予測として強く機能する（TRAIN/TESTで再現するきれいな用量反応）ことを
確認した。本スクリプトはその先: **S7現行本番(D構成=527件)にmark_sum上限ゲートを
追加した場合、honest ROIが実際に改善するか**をTRAIN選定→TEST一度きり評価で検証する。

方法:
1. `s7_staged_audit_candidates.pkl`（既存honestキャッシュ、D構成前後の全段階を含む）
   からD構成(527件)を抽出
2. 各候補レースのmark_sum(=pred_top3_pct(◎)+pred_top3_pct(◯)、四半期walk-forward
   モデルによる発走前予測・wt_entries格納値をそのまま使用)を突き合わせる
3. TRAIN(2024-01-01〜2025-12-31)でmark_sum上限の閾値グリッドを振り、ROIが
   最良となる閾値を1つ選定
4. TEST(2026-01-01〜2026-07-28)で同一閾値を一度だけ適用し、honest ROIを確認

本番コード・モデルは一切変更しない。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

S7_AXIS_SUM_MAX = 1.3
S7_ENTROPY_MAX = 1.8329
S7_MARK3_OVERLAP_MAX = 1

TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20260728"

THRESHOLD_GRID = [200, 190, 180, 170, 160, 150, 140, 130, 120]


def d_filter(c):
    return (c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX
            and c["entropy"] <= S7_ENTROPY_MAX
            and (c["wt_mark3_overlap_n"] is not None
                 and c["wt_mark3_overlap_n"] <= S7_MARK3_OVERLAP_MAX))


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
                    out[rk] = marks[1] + marks[2]
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
    cache_path = Path(__file__).resolve().parent / ".." / "data" / "exp_cache" / "s7_staged_audit_candidates.pkl"
    cache_path = cache_path.resolve()
    with open(cache_path, "rb") as f:
        all_cands = pickle.load(f)

    d_sel = [c for c in all_cands if d_filter(c)]
    print(f"D構成(現行本番)全体: {len(d_sel)}件")

    mark_sum_map = load_mark_sum([c["race_key"] for c in d_sel])
    for c in d_sel:
        c["mark_sum"] = mark_sum_map.get(c["race_key"])
    n_with = sum(1 for c in d_sel if c["mark_sum"] is not None)
    print(f"mark_sum取得できた件数: {n_with}/{len(d_sel)}")

    d_sel = [c for c in d_sel if c["mark_sum"] is not None]
    train = [c for c in d_sel if TRAIN_FROM <= c["race_key"][:8] <= TRAIN_TO]
    test = [c for c in d_sel if TEST_FROM <= c["race_key"][:8] <= TEST_TO]

    n0, h0, hr0, bet0, pay0, roi0 = summarize(train)
    print(f"\n[TRAIN] ゲートなし全体: n={n0} hit={hr0:.1f}% ROI={roi0:.1f}%")
    n0t, h0t, hr0t, bet0t, pay0t, roi0t = summarize(test)
    print(f"[TEST]  ゲートなし全体: n={n0t} hit={hr0t:.1f}% ROI={roi0t:.1f}%")

    print(f"\n[TRAIN] mark_sum上限ゲート別ROI")
    print(f"{'上限':<10}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    best_th, best_roi = None, -1
    for th in THRESHOLD_GRID:
        sub = [c for c in train if c["mark_sum"] <= th]
        n, hits, hitrate, bet, pay, roi = summarize(sub)
        mark = " ★100%超" if roi > 100 else ""
        print(f"<={th:<8}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")
        if n >= 30 and roi > best_roi:
            best_roi, best_th = roi, th

    print(f"\n[選定] TRAINで最良（n>=30制約下）: mark_sum<={best_th} (ROI={best_roi:.1f}%)")

    sub_test = [c for c in test if c["mark_sum"] <= best_th]
    n, hits, hitrate, bet, pay, roi = summarize(sub_test)
    print(f"\n[TEST] mark_sum<={best_th} を一度だけ適用:")
    print(f"  n={n} hit={hitrate:.1f}% bet={bet:,} payout={pay:,} ROI={roi:.1f}%")
    mark = " ★100%超(再現)" if roi > 100 and n > 0 else ""
    print(f"  {mark if mark else '(100%超えず・再現せず)'}")

    # 参考: 除外された分＝mark_sum>best_th の候補 (TRAIN/TEST双方) のROIも見る
    print(f"\n--- 参考: 除外側(mark_sum>{best_th})のROI ---")
    for label, data in (("TRAIN", train), ("TEST", test)):
        excl = [c for c in data if c["mark_sum"] > best_th]
        n, hits, hitrate, bet, pay, roi = summarize(excl)
        print(f"{label} 除外側: n={n} hit={hitrate:.1f}% ROI={roi:.1f}%")


if __name__ == "__main__":
    main()
