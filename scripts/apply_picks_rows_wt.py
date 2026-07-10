"""backfill_july_newrules_wt.py が生成した picks_history 行 JSON を PG に適用する（VPS用）。

指定期間の route='wt' 行を全削除して置き換える。

使い方（VPS上・KEIRIN_DB_URL 設定済み環境で）:
  .venv/bin/python3 scripts/apply_picks_rows_wt.py /tmp/july_newrules_rows.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    date_from, date_to = payload["date_from"], payload["date_to"]
    rows = payload["rows"]

    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM picks_history WHERE route='wt' AND race_date BETWEEN ? AND ?",
            (date_from, date_to))
        deleted = cur.rowcount
        for r in rows:
            conn.execute(
                "INSERT INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,trifecta_payout,"
                " bet_amount,route,miwokuri,prerace_gami,gap23,gap12,gap34) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,'wt',?,?,?,?,?)",
                (r["race_date"], r["race_key"], r["rank"], r["pred_combo"], r["n_combos"],
                 r["hit"], r["payout"], r["trio_payout"], r.get("trifecta_payout", 0), r["bet_amount"],
                 r["miwokuri"], r["prerace_gami"], r["gap23"], r.get("gap12"), r.get("gap34")))
    print(f"{date_from}〜{date_to}: 削除 {deleted} 行 → 挿入 {len(rows)} 行")


if __name__ == "__main__":
    main()
