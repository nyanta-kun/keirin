"""winticket 成績通知＋picks_history保存

wave_picks_wt_{date}.txt の公開買い目を、winticket の確定結果(wt_entries.finish_order)
と wt_odds(三連複/三連単) で採点し、Discord通知＋picks_history に保存する。
欠車(finish_order=0/NULL)は着外として除外。公開した買い目のみ採点（再導出しない）。
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.discord import send
from src.evaluation.backtest_wt import _load_payouts_wt
from src.database import get_connection


def _parse_picks_full(target_date: str) -> dict:
    """wave_picks_wt_{date}.txt から {(venue, race_no): (rank, time, combo_str)}"""
    p = Path(__file__).parent.parent / "data" / "picks" / f"wave_picks_wt_{target_date}.txt"
    if not p.exists():
        return {}
    picks, rank = {}, None
    for line in p.read_text(encoding="utf-8").splitlines():
        if "【SSランク】" in line: rank = "SS"
        elif "【Sランク】" in line: rank = "S"
        elif "【Aランク】" in line: rank = "A"
        elif rank:
            m = re.match(r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[\d+車\]\s+(.+?)\s+\(\d+点", line)
            if m:
                picks[(m.group(2), int(m.group(3)))] = (rank, m.group(1), m.group(4))
    return picks


def _parse_combo(combo_str: str):
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-")
    parts = body.split("-")
    return int(parts[0]), int(parts[1]), [int(x) for x in parts[2].split(",")][:3]


def _stats_line(label, s):
    if not s or s["bets"] == 0:
        return f"{label}: データなし"
    roi = s["returns"] / s["bets"] * 100
    return (f"{label}: {s['races']}R 的中{s['hits']}回 "
            f"{s['hits']/s['races']*100:.1f}%  投資{s['bets']:,}→回収{s['returns']:,}  ROI{roi:.1f}%")


def _query_stats(like):
    with get_connection() as conn:
        r = conn.execute(
            "SELECT COUNT(*), SUM(hit), SUM(payout), SUM(bet_amount) "
            "FROM picks_history WHERE route='wt' AND race_date LIKE ?", (like,)).fetchone()
    return {"races": r[0] or 0, "hits": r[1] or 0, "returns": r[2] or 0, "bets": r[3] or 0}


def main():
    from datetime import date
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m-%d")
    dc = target_date.replace("-", "")

    picks = _parse_picks_full(target_date)
    if not picks:
        send(f"⚠️ 競輪AI[wt] [{target_date}] 予想ファイルが見つかりません")
        return

    with get_connection() as conn:
        # picks_history に route 列が無ければ追加（後方互換）
        cols = [r[1] for r in conn.execute("PRAGMA table_info(picks_history)").fetchall()]
        if "route" not in cols:
            conn.execute("ALTER TABLE picks_history ADD COLUMN route TEXT DEFAULT 'ks'")
        name2code = {n: c for c, n in conn.execute("SELECT venue_code, name FROM venue_info").fetchall()}
        start_map = dict(conn.execute(
            "SELECT race_key, start_at FROM wt_races WHERE race_date=?", (target_date,)).fetchall())

    keys = [f"{dc}_{name2code[v]}_{int(rn):02d}" for (v, rn) in picks if v in name2code]
    pm = _load_payouts_wt(keys)

    results, history = [], []
    tb = tr = th = 0
    with get_connection() as conn:
        for (venue, race_no), (rank, ptime, combo_str) in sorted(picks.items(), key=lambda x: (x[0][0], x[0][1])):
            code = name2code.get(venue)
            if code is None:
                continue
            rk = f"{dc}_{code}_{int(race_no):02d}"
            rows = conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                "ORDER BY finish_order", (rk,)).fetchall()
            order = [int(r[0]) for r in rows]
            if len(order) < 3:
                continue
            top3 = frozenset(order[:3])
            p1, p2, thirds = _parse_combo(combo_str)
            bet = len(thirds) * 100
            tb += bet
            hit, pay = False, 0
            if rank == "SS":
                pred = f"{p1}→{p2}→" + ",".join(map(str, thirds))
                for t in thirds:
                    if order[:3] == [p1, p2, t]:
                        pay = pm.get(rk, {}).get(("trifecta", (p1, p2, t)), 0); hit = True; break
            else:
                pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
                for t in thirds:
                    if frozenset((p1, p2, t)) == top3:
                        pay = pm.get(rk, {}).get(("trio", frozenset((p1, p2, t))), 0); hit = True; break
            if hit:
                tr += pay; th += 1
            actual = "-".join(map(str, order[:3]))
            stt = start_map.get(rk)
            from datetime import datetime, timezone, timedelta
            tstr = ptime
            if stt:
                try:
                    tstr = datetime.fromtimestamp(int(stt), tz=timezone(timedelta(hours=9))).strftime("%H:%M")
                except (ValueError, TypeError):
                    pass
            mark = f"◎ ¥{pay:,}" if hit else "×"
            results.append(f"[{rank}] {venue} {race_no}R {tstr}  予:{pred}  実:{actual}  {mark}")
            history.append((target_date, rk, rank, pred, len(thirds), int(hit), pay, bet))

        if history:
            conn.execute("DELETE FROM picks_history WHERE route='wt' AND race_date=?", (target_date,))
            conn.executemany(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,bet_amount,route) "
                "VALUES (?,?,?,?,?,?,?,?,'wt')", history)

    if not results:
        send(f"📊 **競輪AI[wt]成績 {target_date}**\n確定レースなし")
        return

    roi = tr / tb * 100 if tb else 0
    month = _query_stats(target_date[:7] + "%")
    year = _query_stats(target_date[:4] + "%")
    header = (f"📊 **競輪AI[wt]成績 {target_date}**  [6車以下/SS:3連単・S+A:3連複]\n"
              f"確定 {len(results)}R　的中 {th}回 ({th/len(results)*100:.1f}%)\n"
              f"投資 {tb:,}円 → 回収 {tr:,}円　ROI {roi:.1f}%　損益 {tr-tb:+,}円")
    body = "\n".join(results)
    stats = f"\n{'─'*28}\n📅 {target_date[:7]}: {_stats_line('月', month)}\n🗓 {target_date[:4]}年: {_stats_line('年', year)}"
    msg = f"{header}\n```\n{body}\n```{stats}"
    send(msg[:1900])
    print(f"[notify_results_wt] {target_date} 確定{len(results)}R 的中{th} ROI{roi:.1f}%")


if __name__ == "__main__":
    main()
