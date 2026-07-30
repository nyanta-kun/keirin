"""race_type事後セグメントフィルタのブラインド検証（2026-07-29）。

[[keirin_roi_validation_crisis_2026_07_29]] で「race_type別ROI(D構成)で
準決勝151.4%/特予選123.8%/チャレンジ選抜107.2%(n=21-36)」が見つかったが、
これは全期間(2024-01-01〜2026-07-28)を通しで見てから良さそうなセグメントを
拾っただけの post-hoc 選定であり、多重比較バイアス（10区分中3区分が偶然
100%を超える程度は十分あり得る）を排除できていなかった。

本スクリプトは「TRAIN期間でセグメントを選定 → TEST期間で一度だけ評価」という
ブラインド分割で同じ問いを検証し直す。S7のD構成（現行本番定義）の
honest candidate pool（`s7_staged_audit_candidates.pkl`、既にvintageモデルの
walk-forwardで作成済み）をそのまま再利用し、モデル再学習は行わない。

TRAIN: 2024-01-01〜2025-12-31 (n>=20のセグメントのうちROI>100%のものを選定)
TEST : 2026-01-01〜2026-07-28 (TRAINで選定したセグメントのみ、一度だけ評価)
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "exp_cache" / "s7_staged_audit_candidates.pkl"

TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20260728"

S7_AXIS_SUM_MAX = 1.3
S7_ENTROPY_MAX = 1.8329
S7_MARK3_OVERLAP_MAX = 1
MIN_N = 20


def d_filter(c):
    return (c["wt_overlap_n"] in (0, 1) and c["axis_sum"] <= S7_AXIS_SUM_MAX
            and c["entropy"] <= S7_ENTROPY_MAX
            and (c["wt_mark3_overlap_n"] is not None
                 and c["wt_mark3_overlap_n"] <= S7_MARK3_OVERLAP_MAX))


def summarize(cands):
    n = len(cands)
    hits = sum(c["hit"] for c in cands)
    bet = sum(c["bet_amount"] for c in cands)
    pay = sum(c["payout"] for c in cands)
    roi = pay / bet * 100 if bet else 0.0
    hitrate = hits / n * 100 if n else 0.0
    return n, hits, hitrate, bet, pay, roi


def main():
    if not CACHE_PATH.exists():
        print(f"キャッシュが見つかりません: {CACHE_PATH}")
        print("先に exp_s7_gate_staged_audit.py を実行してキャッシュを作成してください。")
        return
    with open(CACHE_PATH, "rb") as f:
        all_cands = pickle.load(f)

    d_sel = [c for c in all_cands if d_filter(c)]
    train = [c for c in d_sel if TRAIN_FROM <= c["race_key"][:8] <= TRAIN_TO]
    test = [c for c in d_sel if TEST_FROM <= c["race_key"][:8] <= TEST_TO]
    print(f"D構成 全体: {len(d_sel)}件 / TRAIN({TRAIN_FROM}〜{TRAIN_TO}): {len(train)}件 "
          f"/ TEST({TEST_FROM}〜{TEST_TO}): {len(test)}件")

    n_all, hits_all, hr_all, bet_all, pay_all, roi_all = summarize(train)
    print(f"\n[TRAIN] D構成全体 ROI: n={n_all} hit={hr_all:.1f}% ROI={roi_all:.1f}%")

    # --- Step1: TRAINでrace_type別ROIを集計し、n>=MIN_NかつROI>100%のセグメントを選定 ---
    by_rt_train = defaultdict(list)
    for c in train:
        by_rt_train[c.get("race_type") or "(NULL)"].append(c)

    print(f"\n[TRAIN] race_type別ROI (n>={MIN_N}のみ表示)")
    print(f"{'race_type':<20}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    train_rows = []
    for rt, cs in by_rt_train.items():
        n, hits, hitrate, bet, pay, roi = summarize(cs)
        if n >= MIN_N:
            train_rows.append((rt, n, hitrate, bet, pay, roi))
    train_rows.sort(key=lambda r: -r[5])
    for rt, n, hitrate, bet, pay, roi in train_rows:
        mark = " ★100%超" if roi > 100 else ""
        print(f"{str(rt):<20}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")

    selected = [rt for rt, n, hitrate, bet, pay, roi in train_rows if roi > 100]
    print(f"\n[選定] TRAINでROI>100%だったセグメント: {selected if selected else '(なし)'}")

    if not selected:
        print("\n判定: TRAINの時点で n>=20 かつ ROI>100% のセグメントが一つもない。"
              "\n      race_type事後セグメントフィルタは、より厳しいブラインド条件下では"
              "\n      再現性のあるエッジを示せなかった。")
        return

    # --- Step2: TESTで選定セグメントのみを一度だけ評価 ---
    by_rt_test = defaultdict(list)
    for c in test:
        by_rt_test[c.get("race_type") or "(NULL)"].append(c)

    print(f"\n[TEST] TRAINで選定したセグメントのみ評価（一度きり・{TEST_FROM}〜{TEST_TO}）")
    print(f"{'race_type':<20}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    n_survive = 0
    for rt in selected:
        cs = by_rt_test.get(rt, [])
        n, hits, hitrate, bet, pay, roi = summarize(cs)
        mark = " ★100%超(再現)" if roi > 100 and n > 0 else ""
        if roi > 100 and n > 0:
            n_survive += 1
        print(f"{str(rt):<20}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")

    # 参考: TEST期間全体のD構成ROI（比較のベースライン）
    n_t, hits_t, hr_t, bet_t, pay_t, roi_t = summarize(test)
    print(f"\n[参考] TEST期間 D構成全体ROI: n={n_t} hit={hr_t:.1f}% ROI={roi_t:.1f}%")

    print(f"\n判定: TRAINで選定した{len(selected)}セグメント中、TESTでもROI>100%を"
          f"維持したのは{n_survive}件。")
    if n_survive == 0:
        print("      → race_type事後セグメントフィルタはブラインド検証で再現せず、"
              "\n        全期間内訳で見えていた3セグメントは多重比較ノイズだったと判断するのが妥当。")
    else:
        print("      → 再現したセグメントについては n の絶対数・1日あたりの発生頻度も確認の上、"
              "\n        実用性を判断すること。")


if __name__ == "__main__":
    main()
