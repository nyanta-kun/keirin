#!/usr/bin/env python3
"""週次再学習(weekly_retrain_wt.sh)のモデル更新が届いているかをVPS側から検知する
（2026-07-31・D-6対応）。

背景:
    週次再学習はMacのcron(日曜23:30 JST)で実行され、成功時のみ
    `scripts/sync_models_to_vps.sh`（またはcrontab直書きのrsync）でVPSへ
    モデルファイルを配布する運用になっている。Macがスリープ/シャットダウン
    していると再学習自体が丸ごとスキップされ、モデル配布も発生しないが、
    それを検知する仕組みがVPS側に存在しなかった。`weekly_retrain_wt.sh`は
    AUCゲート（品質劣化時に本番昇格を中止）は持つが「そもそも今週実行された
    か」自体は検知できない。誰にも通知されないまま数週間気づかれない
    リスクがある。

    実機確認（2026-07-31 `pmset -g sched`）ではwake/poweronスケジュールが
    0件だったため、Macが夜間スリープすれば週次再学習は確実にスキップされる
    状態だった。

設計方針（VPS側から監視・方向A）:
    VPSはメモリ1.9GB・空き実測101MB程度と逼迫しているため、本スクリプトは
    `os.stat`によるファイル更新日時(mtime)比較のみを行い、pandas/lightgbm等
    重い依存は一切importしない（Discord通知が必要な場合のみ、依存が軽量な
    `src.notify.discord`を遅延importする）。

対象ファイルと閾値:
    - `lgbm_wt.pkl` / `lgbm_wt_eval.pkl` / `lgbm_wt_win.pkl` / `lgbm_wt_win_eval.pkl`
      （いずれも`weekly_retrain_wt.sh`が正常実行された週には必ず書き換えられる
      本番モデル。`lgbm_wt_train_only.pkl`は週次再学習では再生成されない
      静的アーティファクトのため対象外＝含めると常に「古い」誤検知になる）。
    - 既定の閾値は7日周期に1日の実行猶予を足した **8日**
      （`--stale-days`または環境変数`MODEL_STALE_DAYS`で上書き可）。
      日曜23:30開始・完了後直ちに配布される正常運用なら、月曜未明には
      全ファイルのmtimeが更新される。8日を超えて更新が無ければ、直近の
      日曜分の実行そのものが行われなかった（またはrsyncが届かなかった）
      可能性が高いと判断する。
    - `lgbm_wt_win.pkl`/`lgbm_wt_win_eval.pkl`は1着専用モデルの品質ゲート
      （`WIN_AUC_GATE_MIN`）不合格時には更新されないことがある。これは
      「今週実行されたが品質判定で見送った」正常系であり「実行されなかった」
      とは区別すべきだが、本スクリプトはmtimeのみで判定するため両者を
      厳密には切り分けられない。`lgbm_wt.pkl`/`lgbm_wt_eval.pkl`（無条件に
      毎週更新される）が新しいのに`_win`系だけ古い場合は品質ゲート見送りの
      可能性が高い、という判断材料としてDiscordメッセージに個別ファイル名と
      経過日数を列挙する。

使い方（VPS cron想定・cronは未登録、以下は提示のみ）:
    cd $KEIRIN_HOME && PYTHONPATH=. .venv/bin/python3 scripts/check_model_freshness.py

    # 推奨cron登録例（1日1回で十分。実際の登録はユーザー判断で実施すること）:
    0 9 * * * cd $KEIRIN_HOME && PYTHONPATH=. .venv/bin/python3 \
      scripts/check_model_freshness.py >> $KEIRIN_HOME/data/logs/cron.log 2>&1

終了コード: 全ファイルが閾値内なら0、1件でも古い/欠落していれば1
（cronのログ監視・メール等と組み合わせる場合の補助シグナルとして利用可能）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# 週次再学習で無条件に(品質ゲートに関わらず)毎回更新される本番モデル。
# lgbm_wt_train_only.pkl は再学習対象外の静的アーティファクトのため含めない。
TARGET_FILES: tuple[str, ...] = (
    "lgbm_wt.pkl",
    "lgbm_wt_eval.pkl",
    "lgbm_wt_win.pkl",
    "lgbm_wt_win_eval.pkl",
)

DEFAULT_STALE_DAYS = 8.0


def check_staleness(
    model_dir: Path, target_files: tuple[str, ...], stale_days: float
) -> list[tuple[str, float | None]]:
    """しきい値を超えて古い(または存在しない)ファイルを返す。

    戻り値は (ファイル名, 経過日数) のリスト。ファイルが存在しない場合は
    経過日数を None にする。
    """
    stale: list[tuple[str, float | None]] = []
    now = time.time()
    threshold_sec = stale_days * 86400
    for name in target_files:
        path = model_dir / name
        if not path.exists():
            stale.append((name, None))
            continue
        age_sec = now - path.stat().st_mtime
        if age_sec > threshold_sec:
            stale.append((name, age_sec / 86400))
    return stale


def build_message(stale: list[tuple[str, float | None]], stale_days: float) -> str:
    lines = [
        "⚠️ **[check_model_freshness] 週次再学習のモデル更新が確認できません"
        f"（閾値 {stale_days:.0f}日）**"
    ]
    for name, age_days in stale:
        if age_days is None:
            lines.append(f"- `{name}`: ファイルが存在しません")
        else:
            lines.append(f"- `{name}`: 最終更新 {age_days:.1f}日前")
    lines.append(
        "Mac側の週次再学習(weekly_retrain_wt.sh、日曜23:30 JST)が実行されて"
        "いない、またはモデル配布(sync_models_to_vps.sh/rsync)が失敗している"
        "可能性があります。Macのスリープ/電源状態を確認してください。"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="週次再学習モデルの更新日時をチェックし、古い場合はDiscordへ警告する。"
    )
    parser.add_argument(
        "--stale-days",
        type=float,
        default=float(os.environ.get("MODEL_STALE_DAYS", DEFAULT_STALE_DAYS)),
        help=f"この日数を超えて更新が無ければ警告する（既定: {DEFAULT_STALE_DAYS}日）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discordへは送信せず、結果をstdoutに表示するのみ",
    )
    args = parser.parse_args(argv)

    stale = check_staleness(MODEL_DIR, TARGET_FILES, args.stale_days)

    if not stale:
        print(
            f"[check_model_freshness] OK: 全{len(TARGET_FILES)}ファイルが"
            f"{args.stale_days:.0f}日以内に更新されています。"
        )
        return 0

    message = build_message(stale, args.stale_days)
    print(message)

    if args.dry_run:
        print("[check_model_freshness] --dry-run のためDiscord通知は送信していません。")
        return 1

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.notify.discord import send  # 遅延import（健全時は読み込まない）

        ok = send(message, channel="system")
        if not ok:
            print(
                "[check_model_freshness] Discord通知に失敗しました"
                "（DISCORD_WEBHOOK_URL_SYSTEM未設定などの可能性）。",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 - 通知失敗でcron全体を壊さないため広く捕捉
        print(f"[check_model_freshness] Discord通知中に例外: {exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
