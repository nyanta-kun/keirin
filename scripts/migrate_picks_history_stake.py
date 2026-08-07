#!/usr/bin/env python3
"""picks_history の投資・払戻を「1レース RACE_BUDGET 円」方式へ全期間移行する。

2026-08-07 に全ランクの賭け金を **1点100円固定 → 1レース10,000円を点数で均等割り**
へ統一した（ユーザー指示）。過去分は 100円/点 で記録されているため、
**新旧が混在すると Web の投資・回収グラフが日付で段差になる**。
[[feedback_full_period_migration]]「ロジック変更時は必ず全期間を再計算・新旧混在は禁止」
に従い、既存行を再計算する。

## 何をどう直すか

  bet_amount = n_combos × 旧単価        → n_combos × unit_stake(n_combos)
  payout     = 払戻(旧単価ベース)        → payout × 新単価 / 旧単価

旧単価は **bet_amount / n_combos** から復元する（ランク定数を参照しない）。
既に新方式で記録された行（旧単価 == 新単価）は触らない＝**冪等**。

**触らない列**:
  - `trio_payout` / `trifecta_payout` は「そのレースの100円あたり確定払戻」で
    賭け金に依存しない生の値。スケールしてはいけない
  - 見送り行（bet_amount = 0）はそのまま（払戻も0）
  - 🔴 **7H1 は対象外**。唯一の2券種ランクで、n_combos は三連単と三連複の
    合計点数だが**券種ごとに単価が違う**（予算枠 7,500円 / 2,500円 を別々に割る）。
    合計点数から単価を復元する本スクリプトの式は 7H1 には当てはまらない。
    そもそも最初から予算枠方式（平均9,388円/レース）なので移行の必要も無い。

## 使い方

    python3 scripts/migrate_picks_history_stake.py --dry-run     # 影響を確認
    python3 scripts/migrate_picks_history_stake.py               # 実行
    python3 scripts/migrate_picks_history_stake.py --rank RANK_7S

⚠️ 実行前に picks_history のバックアップCSVを取ること（--backup で自動保存）。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402

from src.strategy_wt import CURRENT_PAPER_RANKS, RACE_BUDGET, unit_stake  # noqa: E402

BACKUP_DIR = Path(__file__).resolve().parent.parent / "data" / "backup"

# 合計点数から単価を復元できないランク（券種ごとに単価が違う）。
EXCLUDED_RANKS = {"RANK_7H1"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rank", help="対象ランクを1つに限定（既定は現行全ランク）")
    ap.add_argument("--backup", action="store_true", default=True,
                    help="実行前に対象行をCSVへ退避（既定ON）")
    args = ap.parse_args()

    # 7H1 は2券種で券種ごとに単価が違うため、この式では扱えない（上記docstring）。
    ranks = ([args.rank] if args.rank
             else [s.rank for s in CURRENT_PAPER_RANKS if s.rank not in EXCLUDED_RANKS])
    if set(ranks) & EXCLUDED_RANKS:
        sys.exit(f"{sorted(EXCLUDED_RANKS)} は対象外です（2券種のため式が異なる）")
    url = os.environ.get("KEIRIN_DB_URL")
    if not url:
        sys.exit("KEIRIN_DB_URL が未設定です")

    with psycopg2.connect(url) as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT race_key, rank, race_date, n_combos, bet_amount, payout
            FROM keirin.picks_history
            WHERE rank = ANY(%s) AND route = 'wt'
              AND bet_amount > 0 AND n_combos > 0
            ORDER BY race_date, race_key
        """, (ranks,))
        rows = cur.fetchall()
        print(f"対象候補: {len(rows):,}行（{', '.join(ranks)}）")

        updates = []
        stats = defaultdict(lambda: [0, 0, 0, 0])  # rank → [件数, 旧投資, 新投資, 変更件数]
        for r in rows:
            n = int(r["n_combos"])
            old_bet = int(r["bet_amount"])
            old_unit = old_bet // n
            new_unit = unit_stake(n)
            st = stats[r["rank"]]
            st[0] += 1
            st[1] += old_bet
            st[2] += n * new_unit
            if old_unit == new_unit or old_unit <= 0:
                continue          # 既に新方式＝冪等
            new_bet = n * new_unit
            new_pay = int(r["payout"]) * new_unit // old_unit
            updates.append((new_bet, new_pay, r["race_key"]))
            st[3] += 1

        print(f"\n{'ランク':10s} {'行数':>8s} {'要更新':>8s} "
              f"{'旧 総投資':>14s} {'新 総投資':>14s} {'倍率':>6s}")
        for rank in sorted(stats):
            n, ob, nb, ch = stats[rank]
            print(f"{rank:10s} {n:8,d} {ch:8,d} {ob:14,d} {nb:14,d} "
                  f"{(nb/ob if ob else 0):6.1f}x")
        print(f"\n更新対象 {len(updates):,}行")

        if not updates:
            print("更新対象なし（既に移行済み）")
            return
        if args.dry_run:
            print("[dry-run] 書き込みは行いません")
            for new_bet, new_pay, rk in updates[:5]:
                print(f"   例: {rk} → bet={new_bet:,} payout={new_pay:,}")
            return

        if args.backup:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = BACKUP_DIR / f"picks_history_before_stake_migration_{ts}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"バックアップ: {path}")

        # 単一トランザクション。途中で落ちればDBは無傷。
        cur.execute("""
            UPDATE keirin.picks_history AS p
               SET bet_amount = v.bet, payout = v.pay
              FROM (VALUES %s) AS v(bet, pay, race_key)
             WHERE p.race_key = v.race_key
        """ % ",".join(cur.mogrify("(%s,%s,%s)", u).decode() for u in updates))
        print(f"更新 {cur.rowcount:,}行 → commit")
    print("完了")


if __name__ == "__main__":
    main()
