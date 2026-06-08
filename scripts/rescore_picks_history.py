"""picks_history を「公開した予想ファイル」から正しく再採点する

不具合: notify_results.py が翌朝の再収集データでモデルを再実行し、
①公開ファイルに無いレースを自動算入 ②買い目を再導出 していた。
本スクリプトは wave_picks_{date}.txt の公開買い目をそのまま採点し、
picks_history を上書き修正する。

使い方:
  PYTHONPATH=. .venv/bin/python3 scripts/rescore_picks_history.py            # 全日付を再採点(dry-run表示)
  PYTHONPATH=. .venv/bin/python3 scripts/rescore_picks_history.py --apply    # picks_history を上書き
"""
import argparse, re, glob
from pathlib import Path
from src.database import get_connection
from src.evaluation.backtest import _load_payouts
import importlib.util
# notify_results の _parse_picks_full を再利用
spec = importlib.util.spec_from_file_location("nr", Path(__file__).parent / "notify_results.py")
nr = importlib.util.module_from_spec(spec); spec.loader.exec_module(nr)

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="picks_history を実際に上書き")
args = ap.parse_args()

with get_connection() as conn:
    name2code = {n: c for c, n in conn.execute("SELECT venue_code, name FROM venue_info").fetchall()}

def parse_combo(combo_str):
    """'3連単: 4→2→1,5,3' / '3連複: 2-4-5,1,3' → (pivot1,pivot2,[thirds])"""
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-")
    parts = body.split("-")          # ['4','2','1,5,3']
    p1, p2 = int(parts[0]), int(parts[1])
    thirds = [int(x) for x in parts[2].split(",")][:3]
    return p1, p2, thirds

def score_date(date):
    picks = nr._parse_picks_full(date)        # {(venue, race_no): (rank, time, combo_str)}
    if not picks:
        return None
    dc = date.replace("-", "")
    rows = []
    with get_connection() as conn:
        for (venue, race_no), (rank, _t, combo_str) in picks.items():
            code = name2code.get(venue)
            if code is None:
                continue
            rk = f"{dc}_{code}_{int(race_no):02d}"
            res = conn.execute(
                "SELECT frame_no FROM race_results WHERE race_key=? AND finish_position<=3 "
                "ORDER BY finish_position", (rk,)).fetchall()
            order = [r[0] for r in res]
            if len(order) < 3:
                continue  # 結果未確定はスキップ
            pm = _load_payouts([rk]).get(rk, {})
            p1, p2, thirds = parse_combo(combo_str)
            top3 = frozenset(order[:3]); bet = len(thirds) * 100; pay = 0; hit = False
            if rank == "SS":
                pred = f"{p1}→{p2}→" + ",".join(map(str, thirds))
                for t in thirds:
                    if order[:3] == [p1, p2, t]:
                        pay = pm.get(("trifecta", f"{p1}-{p2}-{t}"), 0); hit = True; break
            else:
                pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
                for t in thirds:
                    if frozenset([p1, p2, t]) == top3:
                        pay = pm.get(("trifecta_box", "=".join(map(str, sorted(top3)))), 0); hit = True; break
            rows.append(dict(race_date=date, race_key=rk, rank=rank, pred_combo=pred,
                             n_combos=len(thirds), hit=int(hit), payout=pay, bet_amount=bet))
    return rows

dates = sorted({Path(f).stem.replace("wave_picks_", "")
                for f in glob.glob("data/picks/wave_picks_2026-*.txt")
                if "_wt_" not in f and "wave_picks_wt" not in f})

print(f"{'日付':<12}{'旧R':>4}{'旧的中':>6}{'旧ROI':>8}   {'新R':>4}{'新的中':>6}{'新ROI':>8}")
grand_old = grand_new = None
tot_new_bet = tot_new_ret = tot_new_hit = tot_new_n = 0
all_new_rows = []
for d in dates:
    rows = score_date(d)
    if rows is None:
        continue
    with get_connection() as conn:
        old = conn.execute("SELECT COUNT(*), SUM(hit), SUM(payout), SUM(bet_amount) "
                           "FROM picks_history WHERE race_date=?", (d,)).fetchone()
    on, oh, op, ob = (old[0] or 0, old[1] or 0, old[2] or 0, old[3] or 0)
    nn = len(rows); nh = sum(r["hit"] for r in rows)
    npay = sum(r["payout"] for r in rows); nbet = sum(r["bet_amount"] for r in rows)
    if nn == 0:
        continue
    oroi = f"{op/ob*100:.0f}%" if ob else "-"
    nroi = f"{npay/nbet*100:.0f}%" if nbet else "-"
    print(f"{d:<12}{on:>4}{oh:>6}{oroi:>8}   {nn:>4}{nh:>6}{nroi:>8}")
    tot_new_bet += nbet; tot_new_ret += npay; tot_new_hit += nh; tot_new_n += nn
    all_new_rows += rows

print("-"*52)
with get_connection() as conn:
    o = conn.execute("SELECT COUNT(*), SUM(hit), SUM(payout), SUM(bet_amount) FROM picks_history").fetchone()
print(f"{'旧合計':<12}{o[0]:>4}{o[1]:>6}{o[2]/o[3]*100:>7.0f}%")
print(f"{'新合計':<12}{tot_new_n:>4}{tot_new_hit:>6}{tot_new_ret/tot_new_bet*100:>7.0f}%  "
      f"(投資{tot_new_bet:,} 回収{tot_new_ret:,} 損益{tot_new_ret-tot_new_bet:+,})")

if args.apply:
    with get_connection() as conn:
        for d in dates:
            conn.execute("DELETE FROM picks_history WHERE race_date=?", (d,))
        conn.executemany("""
            INSERT INTO picks_history (race_date,race_key,rank,pred_combo,n_combos,hit,payout,bet_amount)
            VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,:payout,:bet_amount)
        """, all_new_rows)
    print("\n✅ picks_history を公開予想ベースで上書きしました。")
else:
    print("\n(dry-run。--apply で picks_history を上書きします)")
