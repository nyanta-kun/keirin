"""月次vintageモデル体系の一括構築（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

背景: 四半期vintageモデル18本が2026-07-28にアドホック実験で無断上書きされ、
honest ROI検証の再現性が失われる事故が発生した。ユーザー指示により、期間設定を
根本から統一する:
  - 2023-12-31以前（2022-12-01〜2023-12-31・約13ヶ月）は全モデル共通の
    「ベース学習データ」として常に含める
  - 2024-01以降は月単位でモデルを学習し、スライドさせる
    （月Mのモデルは「ベース+2024-01〜M-1月末」を学習データとし、M月のレースを
    スコアリングする用途に使う）

命名規則: lgbm_wt_eval_mYYMM / lgbm_wt_win_mYYMM（例: lgbm_wt_eval_m2401）。
`src/models/trainer.py::save_model()`の書き込み保護（_VINTAGE_NAME_RE）が
`_m\\d{6}$`にマッチするため、一度作成されたこれらのモデルは再度このスクリプトを
実行しても上書きされない（意図的な再構築時は個別に--force-overwrite-vintageで
train-wtを呼ぶか、事前にpklを削除する）。

対象月: 2024-01 〜 実行時点の当月（進行中の月も含む・現行のtail窓と同じ考え方）。
学習データ開始日は固定で2022-12-01（全モデル共通のベース学習データ起点）。

実行: .venv/bin/python scripts/train_monthly_vintage_models.py [--dry-run] [--only-missing]
"""
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wt_vintage_config import BASE_FROM, monthly_windows

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "data" / "models"


def run_train(test_from: str, test_to: str, save_as: str, target: str, dry_run: bool) -> bool:
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "src.cli.main", "train-wt",
        "--from", BASE_FROM, "--test-from", test_from, "--test-to", test_to,
        "--save-as", save_as, "--no-promote", "--target", target,
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0:
        print(f"[error] {save_as}: {result.stderr[-2000:]}", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-missing", action="store_true",
                     help="既にpklが存在する月はスキップ（中断後の再開用）")
    args = ap.parse_args()

    windows = monthly_windows()
    print(f"対象月数: {len(windows)}（{windows[0][0]}〜{windows[-1][1]}）")

    ok, skipped, failed = 0, 0, 0
    for test_from, test_to, eval_name, win_name in windows:
        tag = eval_name.replace("lgbm_wt_eval_", "")

        if args.only_missing and (MODEL_DIR / f"{eval_name}.pkl").exists() \
                and (MODEL_DIR / f"{win_name}.pkl").exists():
            print(f"[skip] {tag} ({test_from}~{test_to}) 既存")
            skipped += 1
            continue

        print(f"\n=== {tag}: test_from={test_from} test_to={test_to} ===", flush=True)
        ok1 = run_train(test_from, test_to, eval_name, "top3", args.dry_run)
        ok2 = run_train(test_from, test_to, win_name, "win", args.dry_run)
        if ok1 and ok2:
            ok += 1
        else:
            failed += 1

    print(f"\n完了: ok={ok} skipped={skipped} failed={failed} / 合計{len(windows)}ヶ月")


if __name__ == "__main__":
    main()
