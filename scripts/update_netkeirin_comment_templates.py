"""netkeirin の商品文面を傾斜配分の説明へ差し替える（2026-08-07）。

## なぜスクリプトが要るのか

文面は `keirin.netkeirin_settings.comment_template`（DB）が正で、**行があれば
コードの既定文（`_DEFAULT_COMMENT_TEMPLATE`）は絶対に使われない**。
全ランクに行があるので、コードだけ直しても商品説明は「5点均等」のまま残る。

## 🔴 実行タイミング

**コードをデプロイした直後に実行する。**
先に実行すると、まだ均等割りで入稿している本番が「オッズに応じて配分しています」と
説明することになり、商品説明が事実と食い違う。

    # 1. keirin の PR を master へマージ（= VPS へ自動デプロイ）
    # 2. その直後に
    PYTHONPATH=. .venv/bin/python scripts/update_netkeirin_comment_templates.py --apply

既定は dry-run（差分表示のみ・書き込みなし）。

## 対象

傾斜配分を入れたランクだけ（`tilt_stakes` が真のもの）。
**7B は対象外**（3点の均等買いのままなので現行文面が正しい）。
**7H1 も対象外**（別の買い方・別の文面）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.netkeirin_submit_wt import RANK_CONFIGS  # noqa: E402
from src.database import get_connection  # noqa: E402

# 差し替え対象の一文。**この文字列が見つからないランクは触らない**
# （文面をユーザーが編集済みかもしれないので、機械的に潰すほうが危険）。
OLD_SNIPPETS = (
    "買い目は三連複・軸2車流し（5点均等）でお届けします。",
    "買い目は三連複・軸2車流し（均等買い）でお届けします。",
)

NEW_SENTENCE = (
    "買い目は三連複・軸2車流しです。金額は均等ではなく、"
    "当方が想定する発走時オッズに応じて配分しています。"
    "配当が低くなりやすい買い目に厚く、高くなりやすい買い目に薄く置き、"
    "どの目で決まっても払戻が投資を上回ることを狙う組み立てです。"
)

# 配分の調整をお願いする節。**一文の置換とは分けて、節の直前に差し込む**。
# 7C の文面は 買い目の一文のうしろに「3着以内に届きそうにない車は相手から
# 外しています。」が続くため、一文ごと丸ごと置換すると
# その説明が【ご購入にあたって】の後ろへ飛んで意味が通らなくなる（実際にやった）。
NEW_ADVICE = (
    "【ご購入にあたって】\n"
    "入稿は朝の時点で行うため、この配分はあくまで想定オッズに基づくものです。"
    "レース直前の実際のオッズをご自身でご確認いただき、配分を調整いただくと"
    "精度が上がります。目安は「各買い目の 賭け金 × オッズ が投資総額を"
    "上回っていること」です。"
)
ADVICE_ANCHOR = "【参考データ】"


def targets() -> list[str]:
    return [k for k, cfg in RANK_CONFIGS.items() if cfg.get("tilt_stakes")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="実際に書き込む（既定はdry-run）")
    args = ap.parse_args()

    ranks = targets()
    print(f"対象ランク: {', '.join(ranks)}\n")
    updated, skipped = 0, []

    with get_connection() as conn:
        for rank in ranks:
            row = conn.execute(
                "SELECT comment_template FROM netkeirin_settings WHERE rank_key = ?",
                (rank,),
            ).fetchone()
            if row is None or not row["comment_template"]:
                skipped.append(f"{rank}: DBに行が無い（コード既定文が使われる）")
                continue
            cur = row["comment_template"]
            hit = next((s for s in OLD_SNIPPETS if s in cur), None)
            if hit is None:
                skipped.append(f"{rank}: 差し替え対象の一文が見つからない（手編集済み？）")
                continue
            if NEW_ADVICE[:20] in cur:
                skipped.append(f"{rank}: すでに差し替え済み")
                continue
            new = cur.replace(hit, NEW_SENTENCE)
            if ADVICE_ANCHOR in new:
                new = new.replace(ADVICE_ANCHOR, f"{NEW_ADVICE}\n\n{ADVICE_ANCHOR}", 1)
            else:
                new = f"{new}\n\n{NEW_ADVICE}"
            print(f"── {rank} ──\n[before]\n{cur}\n\n[after]\n{new}\n")
            if args.apply:
                conn.execute(
                    "UPDATE netkeirin_settings SET comment_template = ? WHERE rank_key = ?",
                    (new, rank),
                )
                conn.commit()
            updated += 1

    print(f"{'更新' if args.apply else 'dry-run'}: {updated} 件")
    for s in skipped:
        print(f"  ⚠️ スキップ {s}")
    if not args.apply:
        print("\n※ 書き込むには --apply を付けて実行してください（デプロイ直後に）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
