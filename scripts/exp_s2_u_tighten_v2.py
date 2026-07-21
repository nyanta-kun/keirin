"""S2(U)絞り込みの再検証（本番モデル lgbm_wt_val25 使用・exp_ranks_valtest.py の
collect7/_u_pair/eval_pair_trio をそのまま再利用し、mto閾値のみ細かくスイープする）。

exp_s2_u_tighten.py（フレッシュ学習の簡易モデル）ではbaselineが公式値(検証127.8%/
テスト117.1%)と一致しなかったため、公式スクリプトと同一のモデル・パイプラインで
再検証する。

正規プロトコル: 検証=2025-04-01〜2026-03-31・テスト=2026-04-01〜07-15（モデルは
学習済みの lgbm_wt_val25 をロードするのみ・再学習なし）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/Users/ysuzuki/GitHub/keirin")))

from src.models.trainer import load_model
from scripts.exp_ranks_valtest import (
    MODEL, TEST, VAL, _u_pair, collect7, eval_pair_trio,
)

VAL_DAYS = 365
TEST_DAYS = 106

# 現行本番値
BASE_ENT, BASE_MTO, BASE_LEG = 1.84, 4.3, 15.0


def report(label, val, test, ent, mto, leg):
    vn, vh, vroi = eval_pair_trio(val, _u_pair, ent, mto, leg)
    tn, th, troi = eval_pair_trio(test, _u_pair, ent, mto, leg)
    v_rate = vh / vn * 100 if vn else 0
    t_rate = th / tn * 100 if tn else 0
    flag = "*" if vn >= 100 and vroi >= 95 else " "
    print(f"{flag} {label:26s} | val n={vn:4d}({vn/VAL_DAYS:5.2f}/日) 的中={v_rate:5.1f}% ROI={vroi:6.1f}% | "
          f"test n={tn:4d}({tn/TEST_DAYS:5.2f}/日) 的中={t_rate:5.1f}% ROI={troi:6.1f}%")


def main():
    model = load_model(MODEL)
    print("検証/テストデータ構築(公式モデル lgbm_wt_val25)...", flush=True)
    val = collect7(*VAL, model)
    test = collect7(*TEST, model)
    print(f"検証 {len(val)}R / テスト {len(test)}R", flush=True)

    print("\n===== ベースライン(現行本番値) =====")
    report("現行(1.84/4.3/15)", val, test, BASE_ENT, BASE_MTO, BASE_LEG)

    print("\n===== mto_min 単変量スイープ(entropy/leg固定) =====")
    for mto in (4.3, 4.5, 4.8, 5.0, 5.3, 5.6, 6.0, 6.5, 7.0):
        report(f"mto>={mto:.1f}", val, test, BASE_ENT, mto, BASE_LEG)

    print("\n===== entropy_min 単変量スイープ(mto/leg固定) =====")
    for ent in (1.84, 1.86, 1.88, 1.90, 1.92, 1.95):
        report(f"entropy>={ent:.2f}", val, test, ent, BASE_MTO, BASE_LEG)

    print("\n===== leg_min 単変量スイープ(entropy/mto固定) =====")
    for leg in (15.0, 18.0, 20.0, 25.0, 30.0):
        report(f"leg>={leg:.0f}倍", val, test, BASE_ENT, BASE_MTO, leg)

    print("\n===== mto×leg 有望組合せ =====")
    for mto in (4.3, 5.0, 5.5, 6.0):
        for leg in (15.0, 18.0, 20.0):
            report(f"mto>={mto:.1f}×leg>={leg:.0f}", val, test, BASE_ENT, mto, leg)


if __name__ == "__main__":
    main()
