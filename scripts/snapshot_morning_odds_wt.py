"""朝オッズ前向き計測（winticket）

目的:
  wt_odds は INSERT OR REPLACE で「最終オッズ」に上書きされてしまうため、
  過去データからは「朝→直前のオッズ変動」を測定できない。
  そこで日次cron(7:00)の当日 collect-wt 直後に本スクリプトを呼び、
  その時点の wt_odds（=朝オッズ）を wt_odds_snapshot に退避して保全する。
  翌日以降の前日再収集で wt_odds が最終オッズに更新された後、
  --report で両者を突合し、朝→最終のドリフトを計測する。

使い方:
  # 取得（cron用・既定は当日）。初回値を保持するため INSERT OR IGNORE。
  PYTHONPATH=. .venv/bin/python3 scripts/snapshot_morning_odds_wt.py [YYYY-MM-DD]

  # 計測（朝snapshot vs wt_odds最終 の乖離レポート）
  PYTHONPATH=. .venv/bin/python3 scripts/snapshot_morning_odds_wt.py --report [--from YYYY-MM-DD] [--to YYYY-MM-DD]
"""
import sys
import argparse
from datetime import date, datetime

# プロジェクトルートをパスに追加（PYTHONPATH 未設定でも動くように）
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection, init_db


def snapshot(target_date: str, snapshot_type: str = "morning") -> None:
    """target_date のレースの現在の wt_odds を wt_odds_snapshot に退避する。

    INSERT OR IGNORE のため、同 (race_key,bet_type,combination,snapshot_type) は
    初回（=朝）の値を保持し、再実行しても上書きしない。
    """
    init_db()  # テーブル未作成でも安全
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO wt_odds_snapshot
                (race_key, bet_type, combination, odds_value, snapshot_type, snapshot_at)
            SELECT o.race_key, o.bet_type, o.combination, o.odds_value, ?, ?
            FROM wt_odds o
            JOIN wt_races r ON o.race_key = r.race_key
            WHERE r.race_date = ?
            """,
            (snapshot_type, datetime.now().isoformat(timespec="seconds"), target_date),
        )
        inserted = cur.rowcount
        n_races = conn.execute(
            """
            SELECT COUNT(DISTINCT o.race_key)
            FROM wt_odds_snapshot o
            JOIN wt_races r ON o.race_key = r.race_key
            WHERE r.race_date = ? AND o.snapshot_type = ?
            """,
            (target_date, snapshot_type),
        ).fetchone()[0]
    print(f"[snapshot] {target_date} ({snapshot_type}): "
          f"新規 {inserted:,} 行を退避 / 累計 {n_races:,} レース分を保全")


def report(date_from: str | None, date_to: str | None,
           snapshot_type: str = "morning") -> None:
    """朝snapshot と現在の wt_odds(=最終) を突合し、ドリフトを集計する。"""
    where = ["s.snapshot_type = ?"]
    params: list = [snapshot_type]
    if date_from:
        where.append("r.race_date >= ?"); params.append(date_from)
    if date_to:
        where.append("r.race_date <= ?"); params.append(date_to)
    where_sql = " AND ".join(where)

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT r.race_date, s.bet_type,
                   s.odds_value AS morning_odds,
                   o.odds_value AS final_odds
            FROM wt_odds_snapshot s
            JOIN wt_races r ON s.race_key = r.race_key
            JOIN wt_odds   o ON o.race_key = s.race_key
                            AND o.bet_type = s.bet_type
                            AND o.combination = s.combination
            WHERE {where_sql}
              AND s.odds_value IS NOT NULL AND o.odds_value IS NOT NULL
              AND s.odds_value > 0 AND o.odds_value > 0
            """,
            params,
        ).fetchall()

    if not rows:
        print("[report] 突合可能なデータがありません。")
        print("  朝snapshot蓄積後、前日再収集で wt_odds が最終オッズに更新されると突合可能になります。")
        return

    import statistics
    n = len(rows)
    n_races = len({(r[0], r[1]) for r in rows})  # ざっくり日付×市場
    ratios = [final / morning for _, _, morning, final in rows]  # 最終/朝（>1=朝より上昇=人気低下）
    abs_pct = [abs(x - 1.0) for x in ratios]

    def pct(p):
        return statistics.quantiles(abs_pct, n=100)[p - 1] if n >= 100 else max(abs_pct)

    within10 = sum(1 for x in abs_pct if x <= 0.10) / n
    within20 = sum(1 for x in abs_pct if x <= 0.20) / n
    med_ratio = statistics.median(ratios)

    print(f"\n{'='*72}")
    print(f"朝→最終オッズ ドリフト計測 (snapshot_type={snapshot_type})")
    print(f"{'='*72}")
    print(f"  突合組合せ数:        {n:,}")
    print(f"  中央値 最終/朝:      {med_ratio:.3f}  (1.0=不変, >1=朝より人気低下)")
    print(f"  |変動| 中央値:        {statistics.median(abs_pct):.1%}")
    print(f"  ±10%以内に収まる割合: {within10:.1%}")
    print(f"  ±20%以内に収まる割合: {within20:.1%}")

    # 市場別
    from collections import defaultdict
    by_market: dict = defaultdict(list)
    for _, bet_type, morning, final in rows:
        by_market[bet_type].append(abs(final / morning - 1.0))
    print(f"\n  {'市場':<14} {'件数':>8}  {'|変動|中央値':>12}  {'±20%以内':>9}")
    print(f"  {'-'*48}")
    for bt, vals in sorted(by_market.items(), key=lambda kv: -len(kv[1])):
        w20 = sum(1 for x in vals if x <= 0.20) / len(vals)
        print(f"  {bt:<14} {len(vals):>8,}  {statistics.median(vals):>11.1%}  {w20:>8.1%}")
    print(f"{'='*72}")
    print("  ※ ±20%以内割合が高いほど『朝オッズで張ったガミ回避フィルタが直前まで保つ』")
    print("    ことを意味する。低い市場は朝確定のリスクが大きい。")


def main():
    ap = argparse.ArgumentParser(description="朝オッズ前向き計測（winticket）")
    ap.add_argument("target_date", nargs="?", default=None,
                    help="取得対象日 YYYY-MM-DD（既定: 当日）")
    ap.add_argument("--type", default="morning", help="snapshot_type（既定: morning）")
    ap.add_argument("--report", action="store_true", help="ドリフト計測レポートを出力")
    ap.add_argument("--from", dest="date_from", default=None, help="--report時の開始日")
    ap.add_argument("--to", dest="date_to", default=None, help="--report時の終了日")
    args = ap.parse_args()

    if args.report:
        report(args.date_from, args.date_to, args.type)
    else:
        target = args.target_date or date.today().isoformat()
        snapshot(target, args.type)


if __name__ == "__main__":
    main()
