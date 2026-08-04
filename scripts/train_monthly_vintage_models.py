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
実行しても上書きされない。2026-07-31にこの保護は「ファイル実体が無くても
vintage_manifest.jsonに登録済みなら拒否」へ強化された（rm後の再作成という
事故経路を塞ぐため・commit bd127b1）ので、pklを削除するだけでは再学習できない。

【--force-retrain-all】特徴量セット（FEATURE_COLS_WT）を変更した場合、既存の
vintageモデルは全て古い特徴量数で学習されており推論に使えなくなるため、
**全62本を再生成する必要がある**。従来このスクリプトには
--force-overwrite-vintage を渡す経路が無く、特徴量変更後にvintage体系を
再構築できないという構造的な欠落があった（2026-07-31・Phase3で発覚）。
--force-retrain-all を明示指定した場合のみ各train-wtに
--force-overwrite-vintage を渡す。**旧重みは失われる**（ハッシュはgit管理下の
vintage_manifest.jsonの履歴に残るため、どの重みが正しかったかの証跡は追える）。
このフラグは事故（2026-07-28の無断上書き）と同じ操作なので、
理由を必ずコミットメッセージ等に記録すること。

対象月: 2024-01 〜 実行時点の当月（進行中の月も含む・現行のtail窓と同じ考え方）。
学習データ開始日は固定で2022-12-01（全モデル共通のベース学習データ起点）。

【--months】特徴量セット変更などで一部の月だけ作り直したいときに範囲指定する
（例: --months 2509-2607）。--force-retrain-all と併用する。

【空回りの防止・2026-08-04】pklが無いのにvintage_manifest.jsonには登録済み、という
状態（バックアップへ退避した等）では save_model のガードが必ず保存を拒否する。
従来はこれを検出せず1モデル10〜15分学習してから捨てており、m2509〜m2603の
7ヶ月×2モデルで約2時間半を空転させた。現在は**学習を始める前に**検出して
非ゼロ終了し、必要なコマンド（--force-retrain-all --months ...）を提示する。

実行: .venv/bin/python scripts/train_monthly_vintage_models.py
        [--dry-run] [--only-missing] [--force-retrain-all] [--months YYMM-YYMM]
"""
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import vintage_manifest
from src.wt_vintage_config import BASE_FROM, monthly_windows

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "data" / "models"


def _registered_model_names() -> set[str]:
    """vintage_manifest.json に登録済みのモデル名を返す。"""
    return set(vintage_manifest.load_manifest().get("models", {}))


def run_train(test_from: str, test_to: str, save_as: str, target: str, dry_run: bool,
              force: bool = False) -> bool:
    """1モデルを学習する。

    force=True のとき --force-overwrite-vintage を渡し、凍結vintageの
    書き込み保護（既存ファイル・マニフェスト登録の両方）を突破する。
    特徴量セット変更に伴う全再生成でのみ使う。
    """
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "src.cli.main", "train-wt",
        "--from", BASE_FROM, "--test-from", test_from, "--test-to", test_to,
        "--save-as", save_as, "--no-promote", "--target", target,
    ]
    if force:
        cmd.append("--force-overwrite-vintage")
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
    ap.add_argument("--force-retrain-all", action="store_true",
                     help="凍結vintageの書き込み保護を突破して全月を再学習する。"
                          "特徴量セット変更時のみ使用。旧重みは失われる（ハッシュは"
                          "vintage_manifest.jsonのgit履歴に残る）")
    ap.add_argument("--months", default=None, metavar="YYMM-YYMM",
                     help="対象月を範囲で絞る（例: 2509-2607）。特徴量セット変更で"
                          "一部の月だけ作り直したいときに使う。省略時は全月。")
    args = ap.parse_args()

    if args.force_retrain_all and args.only_missing:
        ap.error("--force-retrain-all と --only-missing は同時指定できません"
                 "（前者は既存を上書きし、後者は既存をスキップするため矛盾する）")

    windows = monthly_windows()
    if args.months:
        try:
            lo, hi = (s.strip() for s in args.months.split("-", 1))
            if not (lo.isdigit() and hi.isdigit() and len(lo) == len(hi) == 4):
                raise ValueError
        except ValueError:
            ap.error(f"--months の形式が不正です: {args.months!r}（例: 2509-2607）")
        windows = [w for w in windows
                   if lo <= w[2].replace("lgbm_wt_eval_m", "") <= hi]
        if not windows:
            ap.error(f"--months {args.months} に該当する月がありません")
    print(f"対象月数: {len(windows)}（{windows[0][0]}〜{windows[-1][1]}）")
    if args.force_retrain_all:
        print("⚠️ --force-retrain-all: 凍結vintageの書き込み保護を突破して"
              f"全{len(windows)}ヶ月×2モデルを再学習します（旧重みは失われます）")

    # 「pklは無いがマニフェストには登録済み」は save_model のガードに必ず弾かれる。
    # 従来はこれを検出せず1モデルあたり10〜15分学習してから保存を拒否され、結果を
    # 捨てていた（2026-08-04に m2509〜m2603 の7ヶ月×2モデルで約2時間半を空転）。
    # 学習を始める前に落として、必要な操作を提示する。
    if not args.force_retrain_all and not args.dry_run:
        registered = _registered_model_names()
        orphaned = [n for _, _, e, w in windows for n in (e, w)
                    if n in registered and not (MODEL_DIR / f"{n}.pkl").exists()]
        if orphaned:
            months = sorted({n.rsplit("_m", 1)[1] for n in orphaned})
            print(f"[FATAL] マニフェスト登録済みなのにファイルが存在しないモデルが"
                  f"{len(orphaned)}件あります（月: {', '.join(months)}）。", file=sys.stderr)
            print("        save_model のガードにより保存が拒否されるため、学習しても"
                  "結果は捨てられます。", file=sys.stderr)
            print(f"        再作成するなら: --force-retrain-all --months "
                  f"{months[0]}-{months[-1]}", file=sys.stderr)
            print("        （凍結vintageの重みが変わるため honest walk-forward の"
                  "再現性は失われます）", file=sys.stderr)
            raise SystemExit(1)

    ok, skipped, failed = 0, 0, 0
    for test_from, test_to, eval_name, win_name in windows:
        tag = eval_name.replace("lgbm_wt_eval_", "")

        if args.only_missing and (MODEL_DIR / f"{eval_name}.pkl").exists() \
                and (MODEL_DIR / f"{win_name}.pkl").exists():
            print(f"[skip] {tag} ({test_from}~{test_to}) 既存")
            skipped += 1
            continue

        print(f"\n=== {tag}: test_from={test_from} test_to={test_to} ===", flush=True)
        ok1 = run_train(test_from, test_to, eval_name, "top3", args.dry_run,
                        force=args.force_retrain_all)
        ok2 = run_train(test_from, test_to, win_name, "win", args.dry_run,
                        force=args.force_retrain_all)
        if ok1 and ok2:
            ok += 1
        else:
            failed += 1

    print(f"\n完了: ok={ok} skipped={skipped} failed={failed} / 合計{len(windows)}ヶ月")

    # 2026-08-01 F-4対応: 従来は failed>0 でも常に exit 0 していたため、
    # このスクリプトを呼び出す自動化ラッパー（scripts/ensure_monthly_vintage.sh）が
    # 学習失敗を検知できず、不完全な状態のままVPS配布(sync_models_to_vps.sh)へ
    # 進んでしまう恐れがあった。failed>0 の場合は非ゼロで終了する。
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
