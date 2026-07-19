"""S1(SEVEN_S1)の絞り込み検討: top3_gap閾値を現行0.15から引き上げ、
1日15レース以下・的中率向上を狙うスイープ（正規プロトコル）。

現行本番: 7車・軸=win model1位・相手=top3モデルで軸以外の上位2頭(p1,p2)・
top3_gap(p1-p2)>=0.15・三連単2点流し(軸→p1→p2, 軸→p2→p1)・目オッズ下限なし。
テスト実績(2026-04-01〜07-15): n=2851 (約26.9R/日)。

正規プロトコル: 学習=〜2025-03-31・検証=2025-04-01〜2026-03-31（閾値選定）・
テスト=2026-04-01〜07-15（選択閾値のみ1回評価）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/Users/ysuzuki/GitHub/keirin")))

from scripts.exp_s1_win_axis_trifecta import (
    TRAIN_FROM, TRAIN_TO, VAL_FROM, VAL_TO, TEST_FROM, TEST_TO,
    train_models, collect, settle_2pt,
)

VAL_DAYS = 365   # 2025-04-01〜2026-03-31
TEST_DAYS = 106  # 2026-04-01〜07-15

THRESHOLDS = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40, 0.45, 0.50]


def main():
    win_model, top3_model = train_models()
    print("\n検証データ構築(7車)...", flush=True)
    val = collect(VAL_FROM, VAL_TO, win_model, top3_model, 7)
    print("テストデータ構築(7車)...", flush=True)
    test = collect(TEST_FROM, TEST_TO, win_model, top3_model, 7)
    print(f"検証 {len(val)}R / テスト {len(test)}R", flush=True)

    print(f"\n{'th':>6} | {'val_n':>6} {'v/day':>6} {'v的中%':>7} {'v_ROI':>7} | "
          f"{'test_n':>7} {'t/day':>6} {'t的中%':>7} {'t_ROI':>7}")
    print("-" * 80)
    for th in THRESHOLDS:
        gate = lambda r, th=th: r["top3_gap"] >= th
        vn, vh, vroi = settle_2pt(val, gate, 0.0)
        tn, th_, troi = settle_2pt(test, gate, 0.0)
        v_rate = vh / vn * 100 if vn else 0
        t_rate = th_ / tn * 100 if tn else 0
        print(f"{th:6.2f} | {vn:6d} {vn/VAL_DAYS:6.1f} {v_rate:7.1f} {vroi:7.1f} | "
              f"{tn:7d} {tn/TEST_DAYS:6.1f} {t_rate:7.1f} {troi:7.1f}")


if __name__ == "__main__":
    main()
