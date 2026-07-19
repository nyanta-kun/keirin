#!/usr/bin/env python3
"""S3(M)行を四半期対応のリークなしwin_rankモデルで全期間再構築する（2026-07-19）。

背景: Phase B/複合ゲート拡張で追加した win_rank/ratio ゲートは、本番モデル
lgbm_wt_win（full_refit=True・ホールドアウトなし・毎回全期間で再学習）で
計算されていた。gap12側の lgbm_wt_eval には q2401〜q2507/w2/w3 という
四半期ウォークフォワードモデル群が存在し過去バックフィルもそれで行われていたが、
win側には対応するモデルがなく、2024-01〜2026-07-19 の picks_history 再構築
（backfill_um_rank_wt.py 既定 --win-model lgbm_wt_win）は「今日学習した
全期間モデルで過去のレースを判定する」リーク状態だった。

本スクリプトは win 側にも同じ四半期ウォークフォワードモデル
（lgbm_wt_win_q2401 等・train-wt --target win で学習済み）を対応させ、
S3(M)行のみをリークなしで再構築する。S2(U)行は win_rank を使わないため対象外
（--wipe-m 相当・S2は変更しない）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/rebuild_s3_walkforward.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_um_rank_wt import build_rows, insert_rows, wipe_rank_rows

# get_connection() は KEIRIN_DB_URL が立っていると読み取りも VPS PG に向く。
# PG 側の wt_odds は直近分のみのミラー（2026-06〜）で全履歴を持たないため、
# 読み取り（build_rows）は必ずローカル SQLite の完全データを使う。
# 書き込み（wipe/insert）だけ最後に PG へも反映したいので、ここで環境変数を
# 一時的に退避し、最後の wipe/insert 直前に復元する。
_PG_URL = os.environ.pop("KEIRIN_DB_URL", None)

# (date_from, date_to, eval_model, win_model) — eval_model/win_model は同一 test 窓で
# 学習されたペア（train-wt --test-from/--test-to が一致）。2026-04-13〜04-19 のみ
# 対応する専用モデルがなく base(lgbm_wt_eval/lgbm_wt_win_eval, test_from=2026-04-20)を
# 転用する（既存のgap12バックフィルと同じ扱い・7日分のみ軽微な残存リーク）。
QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507"),
    ("2025-10-01", "2025-12-31", "lgbm_wt_eval_w3", "lgbm_wt_win_w3"),
    ("2026-01-01", "2026-04-12", "lgbm_wt_eval_w2", "lgbm_wt_win_w2"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="末尾窓の終了日（省略時は昨日）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.end:
        from datetime import date, timedelta
        args.end = (date.today() - timedelta(days=1)).isoformat()

    quarters = list(QUARTERS)
    quarters.append(("2026-04-13", args.end, "lgbm_wt_eval", "lgbm_wt_win_eval"))

    all_m_rows: list[dict] = []
    for date_from, date_to, eval_model, win_model in quarters:
        print(f"\n[rebuild-s3] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        rows = build_rows(eval_model, date_from, date_to, win_model_name=win_model)
        m_rows = [r for r in rows if r["rank"] == "7PLUS_M"]
        n_hit = sum(r["hit"] for r in m_rows)
        bet = sum(r["bet_amount"] for r in m_rows)
        pay = sum(r["payout"] for r in m_rows)
        gate_counts: dict[str, int] = {}
        for r in m_rows:
            gl = r.get("gate_label") or "?"
            gate_counts[gl] = gate_counts.get(gl, 0) + 1
        print(f"[rebuild-s3]   S3(M): {len(m_rows)}R 的中{n_hit} "
              f"({n_hit / len(m_rows) * 100 if m_rows else 0:.1f}%) "
              f"投資{bet:,} → 回収{pay:,} ROI {pay / bet * 100 if bet else 0:.1f}%  "
              f"gate内訳={gate_counts}", flush=True)
        all_m_rows.extend(m_rows)

    total_hit = sum(r["hit"] for r in all_m_rows)
    total_bet = sum(r["bet_amount"] for r in all_m_rows)
    total_pay = sum(r["payout"] for r in all_m_rows)
    print(f"\n[rebuild-s3] ===== 全期間合計 =====")
    print(f"[rebuild-s3] S3(M): {len(all_m_rows)}R 的中{total_hit} "
          f"({total_hit / len(all_m_rows) * 100 if all_m_rows else 0:.1f}%) "
          f"投資{total_bet:,} → 回収{total_pay:,} "
          f"ROI {total_pay / total_bet * 100 if total_bet else 0:.1f}%")

    if _PG_URL:
        os.environ["KEIRIN_DB_URL"] = _PG_URL
    wipe_rank_rows("7PLUS_M", "#7M", "2024-01-01", args.end, args.dry_run)
    insert_rows(all_m_rows, args.dry_run)
    if args.dry_run:
        print("[rebuild-s3] DRY RUN（書き込みなし）")


if __name__ == "__main__":
    main()
