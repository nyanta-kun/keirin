"""picks_history のペイアウトを VPS に同期する。

ローカル SQLite の wt_odds（三連複）で payout を算出し、
VPS PostgreSQL の picks_history の payout を UPDATE する。

Usage:
    python3 scripts/sync_payout_to_vps.py            # 全期間
    python3 scripts/sync_payout_to_vps.py 2025-07    # 月フィルタ
    python3 scripts/sync_payout_to_vps.py 2025-07-01 2025-07-31  # 日付範囲
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_connection  # VPS接続（KEIRIN_DB_URL必須）


def _parse_picks_full(target_date: str) -> dict:
    """公開買い目ファイルから (venue, race_no, slot) → (rank, combo_str) を返す。"""
    base = Path(__file__).parent.parent / "data" / "picks"
    picks = {}
    for fname in (
        f"wave_picks_wt_{target_date}.txt",
        f"wave_picks_wt_{target_date}_night.txt",
    ):
        p = base / fname
        if not p.exists():
            continue
        rank = None
        for line in p.read_text(encoding="utf-8").splitlines():
            if "【7+車 SSランク】" in line:
                rank = "7PLUS_SS"
            elif "【7+車 Sランク】" in line:
                rank = "7PLUS_S"
            elif "【7+車 Aランク】" in line:
                rank = None  # 廃止済み
            elif "【7+車】" in line:
                rank = "7PLUS_S"
            elif any(x in line for x in ["【SSランク】", "【Sランク】", "【Aランク】", "【Bランク】", "【ワイド1点】"]):
                rank = None
            elif rank:
                m = re.match(
                    r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[\d+車\]\s+(.+?)\s+\(\d+点",
                    line,
                )
                if m:
                    slot = "7plus_ss" if rank == "7PLUS_SS" else "7plus_s"
                    picks[(m.group(2), int(m.group(3)), slot)] = (rank, m.group(4))
    return picks


def _parse_combo(combo_str: str):
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-").replace("⇄", "-")
    parts = body.split("-")
    thirds = (
        [int(x) for x in parts[2].split(",")]
        if len(parts) >= 3
        else []
    )
    return int(parts[0]), int(parts[1]), thirds


def _load_trio_payouts_local(race_keys: list[str], local_db: sqlite3.Connection) -> dict:
    """ローカル SQLite wt_odds から trio payout を取得する。"""
    payout_map: dict[str, dict] = {}
    if not race_keys:
        return payout_map
    CHUNK = 900
    for i in range(0, len(race_keys), CHUNK):
        chunk = race_keys[i : i + CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = local_db.execute(
            f"SELECT race_key, combination, odds_value FROM wt_odds "
            f"WHERE bet_type='trio' AND race_key IN ({placeholders})",
            chunk,
        ).fetchall()
        for race_key, combo, odds_value in rows:
            if odds_value is None:
                continue
            parts = [p for p in re.split(r"[-=→]", str(combo)) if p != ""]
            try:
                nums = [int(p) for p in parts]
            except ValueError:
                continue
            key = frozenset(nums)
            payout_map.setdefault(race_key, {})[key] = int(round(float(odds_value) * 100))
    return payout_map


def process_date(
    target_date: str,
    local_db: sqlite3.Connection,
    name2code: dict,
) -> list[tuple]:
    """1日分のペイアウトを計算して (race_key, payout) リストを返す。"""
    picks = _parse_picks_full(target_date)
    if not picks:
        return []

    dc = target_date.replace("-", "")
    keys = list(
        {
            f"{dc}_{name2code[v]}_{int(rn):02d}"
            for (v, rn, _s) in picks
            if v in name2code
        }
    )
    pm = _load_trio_payouts_local(keys, local_db)

    results = []
    for (venue, race_no, _slot), (rank, combo_str) in picks.items():
        code = name2code.get(venue)
        if code is None:
            continue
        rk = f"{dc}_{code}_{int(race_no):02d}"

        # 着順はローカルSQLiteから
        rows = local_db.execute(
            "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
            "ORDER BY finish_order",
            (rk,),
        ).fetchall()
        order = [int(r[0]) for r in rows]
        if len(order) < 3:
            continue
        top3 = frozenset(order[:3])

        p1, p2, thirds = _parse_combo(combo_str)
        pay = 0
        hit = False
        for t in thirds:
            if frozenset((p1, p2, t)) == top3:
                pay = pm.get(rk, {}).get(frozenset((p1, p2, t)), 0)
                hit = True
                break

        if not hit:
            continue

        # race_key suffix: 7PLUS_SS→#7SS / 7PLUS_S→#7S / 7PLUS_A→#7A
        slot = _slot_suffix(rank)
        store_key = f"{rk}{slot}"
        if pay > 0:
            results.append((pay, store_key))

    return results


def _slot_suffix(rank: str) -> str:
    if rank == "7PLUS_SS":
        return "#7SS"
    if rank == "7PLUS_S":
        return "#7S"
    return "#7A"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    month_filter = args[0] if len(args) == 1 else None
    date_from = args[0] if len(args) >= 2 else None
    date_to = args[1] if len(args) >= 2 else None

    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    dates = sorted(
        p.stem.replace("wave_picks_wt_", "")
        for p in picks_dir.glob("wave_picks_wt_20??-??-??.txt")
    )

    from datetime import date as _date

    today = _date.today().strftime("%Y-%m-%d")
    dates = [d for d in dates if d < today]

    if month_filter:
        dates = [d for d in dates if d.startswith(month_filter)]
    elif date_from and date_to:
        dates = [d for d in dates if date_from <= d <= date_to]

    if not dates:
        print("対象日付なし")
        return

    local_db = sqlite3.connect(
        str(Path(__file__).parent.parent / "data" / "keirin.db")
    )
    name2code = dict(
        local_db.execute("SELECT name, venue_code FROM venue_info").fetchall()
    )

    total = len(dates)
    print(f"対象: {total} 日")

    ok = 0
    updated = 0
    with get_connection() as vps_conn:
        for idx, d in enumerate(dates, 1):
            results = process_date(d, local_db, name2code)
            if not results:
                print(f"[{idx}/{total}] {d} skip (ヒットなし or ファイル無)")
                continue
            # VPS の picks_history を payout で UPDATE
            for pay, store_key in results:
                vps_conn.execute(
                    "UPDATE picks_history SET payout=? WHERE race_key=? AND route='wt'",
                    (pay, store_key),
                )
            print(f"[{idx}/{total}] {d} OK ({len(results)} 件 UPDATE)")
            ok += 1
            updated += len(results)

    local_db.close()
    print(f"\n=== 完了: {ok} 日 / {updated} 件 UPDATE ===")


if __name__ == "__main__":
    main()
