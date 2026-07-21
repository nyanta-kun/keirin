"""S3(M)の絞り込み検討: 現行 gap12 OR win_rank OR ratio の3way ORゲートは、
honest全期間再構築（rebuild_s3_walkforward.py）でゲート別内訳を見ると
win_rankゲート単独が最強（ROI119.1%・736R）で、gap12単独(87.9%)・
ratio単独(88.2%)は共に赤字と判明している。母数確保のためのOR拡張が
実はROIを押し下げている構造なので、win_rank単独ゲートへの絞り込みと、
win_rank閾値そのものの引き上げ・買い目オッズ下限の引き上げを検証する。

正規プロトコル（exp_composite_prob_diff_wt.py と同一の学習/検証/テスト分割・
同一のcollect()を再利用）: 学習=〜2025-03-31・検証=2025-04-01〜2026-03-31
（条件選定）・テスト=2026-04-01〜07-15（選択条件のみ1回評価）。
選定基準: 検証 ROI>=95 ∧ n>=100 で的中率最大。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/Users/ysuzuki/GitHub/keirin")))

from scripts.exp_composite_prob_diff_wt import (
    GAP12_MIN, LEG_MIN_ODDS, TEST_FROM, TEST_TO, VAL_FROM, VAL_TO, WIN_RANK_MIN,
    collect, settle_trio, train_models,
)

VAL_DAYS = 365
TEST_DAYS = 106


def report(label, val, test, gate_fn, leg=LEG_MIN_ODDS):
    vn, vh, vroi = settle_trio(val, gate_fn, leg)
    tn, th, troi = settle_trio(test, gate_fn, leg)
    v_rate = vh / vn * 100 if vn else 0
    t_rate = th / tn * 100 if tn else 0
    flag = "*" if vn >= 100 and vroi >= 95 else " "
    print(f"{flag} {label:34s} | val n={vn:4d}({vn/VAL_DAYS:5.2f}/日) 的中={v_rate:5.1f}% ROI={vroi:6.1f}% | "
          f"test n={tn:4d}({tn/TEST_DAYS:5.2f}/日) 的中={t_rate:5.1f}% ROI={troi:6.1f}%")


def main():
    win_model, top3_model = train_models()
    print("\n検証データ構築...", flush=True)
    val = collect(VAL_FROM, VAL_TO, win_model, top3_model)
    print("テストデータ構築...", flush=True)
    test = collect(TEST_FROM, TEST_TO, win_model, top3_model)
    print(f"不一致7車レース 検証 {len(val)}R / テスト {len(test)}R", flush=True)

    print("\n===== ベースライン =====")
    report("現行(gap12>=.10 OR win_rank>=3)", val, test,
           lambda r: r["gap12"] >= GAP12_MIN or (r["win_rank"] is not None and r["win_rank"] >= WIN_RANK_MIN))
    report("gap12単独(参考・honest赤字87.9%)", val, test, lambda r: r["gap12"] >= GAP12_MIN)
    report("win_rank>=3単独", val, test,
           lambda r: r["win_rank"] is not None and r["win_rank"] >= WIN_RANK_MIN)

    print("\n===== win_rank閾値 単変量スイープ（win_rank単独ゲート） =====")
    for th in (2, 3, 4, 5, 6, 7):
        report(f"win_rank単独>={th}", val, test,
               lambda r, th=th: r["win_rank"] is not None and r["win_rank"] >= th)

    print("\n===== 買い目オッズ下限 スイープ（win_rank>=3単独固定） =====")
    for leg in (15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0):
        report(f"win_rank>=3単独×leg>={leg:.0f}倍", val, test,
               lambda r: r["win_rank"] is not None and r["win_rank"] >= WIN_RANK_MIN, leg)

    print("\n===== win_rank閾値×オッズ下限 有望組合せ =====")
    for th in (3, 4, 5):
        for leg in (15.0, 20.0, 25.0):
            report(f"win_rank>={th}×leg>={leg:.0f}", val, test,
                   lambda r, th=th: r["win_rank"] is not None and r["win_rank"] >= th, leg)


if __name__ == "__main__":
    main()
