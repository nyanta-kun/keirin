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
    """公開買い目ファイルから {(venue, race_no, slot): (rank, time, combo_str)}

    2段階生成のため 昼〜夕 = wave_picks_wt_{date}.txt と
    夜 = wave_picks_wt_{date}_night.txt の両方を読み、採点対象を統合する
    （夜レースは start≥19時で昼と発走時刻が重ならず race_no 衝突なし）。
    slot は "wide"(ワイド1点)/"main"(SS/S/A)。同一レースで両プロダクトが並立するため分離。
    """
    base = Path(__file__).parent.parent / "data" / "picks"
    picks = {}
    for fname in (f"wave_picks_wt_{target_date}.txt", f"wave_picks_wt_{target_date}_night.txt"):
        p = base / fname
        if not p.exists():
            continue
        rank = None
        for line in p.read_text(encoding="utf-8").splitlines():
            if "【SSランク】" in line: rank = "SS"
            elif "【Sランク】" in line: rank = "S"
            elif "【Aランク】" in line: rank = "A"
            elif "【Bランク】" in line: rank = None  # B=各自判断＝公式成績には含めない
            elif "【ワイド1点】" in line: rank = "WIDE"  # 独立プロダクト・rank=WIDEで別集計
            elif rank:
                m = re.match(r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[\d+車\]\s+(.+?)\s+\(\d+点", line)
                if m:
                    slot = "wide" if rank == "WIDE" else "main"
                    picks[(m.group(2), int(m.group(3)), slot)] = (rank, m.group(1), m.group(4))
    return picks


def _parse_combo(combo_str: str):
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-").replace("⇄", "-")   # ⇄=SS 1-2着BOX(両順)
    parts = body.split("-")
    thirds = [int(x) for x in parts[2].split(",")][:3] if len(parts) >= 3 else []  # ワイド=2車で空
    return int(parts[0]), int(parts[1]), thirds


def _void_by_dns(p1, p2, thirds, runners, is_wide=False):
    """欠車(購入不可=返還)の無効化ルール。

    runners = そのレースで出走した車(finish_order>=1)の集合。
      軸(p1/p2)が欠車      → レース無効（返還）。 returns (True, [])
      相手(thirds)が欠車   → その目のみ除外。     returns (False, 有効thirds)
      相手が全員欠車       → 買える目なし→無効。  returns (True, [])
    ワイドは2車とも軸扱い（どちらか欠車で無効）。
    """
    if p1 not in runners or p2 not in runners:
        return True, []
    if is_wide:
        return False, []
    valid = [t for t in thirds if t in runners]
    return (not valid), valid


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
            "FROM picks_history WHERE route='wt' AND rank!='WIDE' AND race_date LIKE ?", (like,)).fetchone()
    return {"races": r[0] or 0, "hits": r[1] or 0, "returns": r[2] or 0, "bets": r[3] or 0}


def main():
    from datetime import date
    # 位置引数=日付 / --silent=Discord抑止(picks_history修復のみ・バックフィル用)
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")
    silent = "--silent" in sys.argv
    emit = (lambda m: None) if silent else send
    dc = target_date.replace("-", "")

    picks = _parse_picks_full(target_date)
    if not picks:
        # ファイル不在(真のエラー) と 推奨(SS/S/A)0件(=Bランクのみ/静かな日・正常) を区別する
        picks_file = Path(__file__).parent.parent / "data" / "picks" / f"wave_picks_wt_{target_date}.txt"
        if not picks_file.exists():
            emit(f"⚠️ 競輪AI[wt] [{target_date}] 予想ファイルが見つかりません")
        else:
            emit(f"📊 競輪AI[wt] [{target_date}] 推奨買い目(SS/S/A)なし＝採点対象なし"
                 f"（Bランク=各自判断のみ、または対象レースなしの日）")
        return

    with get_connection() as conn:
        # picks_history に route 列が無ければ追加（後方互換）
        cols = [r[1] for r in conn.execute("PRAGMA table_info(picks_history)").fetchall()]
        if "route" not in cols:
            conn.execute("ALTER TABLE picks_history ADD COLUMN route TEXT DEFAULT 'ks'")
        name2code = {n: c for c, n in conn.execute("SELECT venue_code, name FROM venue_info").fetchall()}
        start_map = dict(conn.execute(
            "SELECT race_key, start_at FROM wt_races WHERE race_date=?", (target_date,)).fetchall())

    keys = list({f"{dc}_{name2code[v]}_{int(rn):02d}" for (v, rn, _s) in picks if v in name2code})
    pm = _load_payouts_wt(keys)

    results, results_wide, history = [], [], []
    tb = tr = th = 0          # SS/S/A 合計
    wb = wr = wh = 0          # ワイド1点 合計（独立プロダクト・別集計）
    skipped_dns = 0           # 軸欠車/全相手欠車でレース無効（返還）→不計上
    with get_connection() as conn:
        for (venue, race_no, _slot), (rank, ptime, combo_str) in sorted(picks.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
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
            # 出走した車(=finish_order>=1)。これに無い車は欠車/失格=購入不可(返還)。
            runners = {int(r[0]) for r in conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order >= 1", (rk,)).fetchall()}
            p1, p2, thirds = _parse_combo(combo_str)
            # ── 欠車の無効化（返還＝損益に計上しない）──
            skip_race, thirds = _void_by_dns(p1, p2, thirds, runners, is_wide=(rank == "WIDE"))
            if skip_race:
                skipped_dns += 1
                continue
            is_box = "BOX" in combo_str          # SS 1-2着BOX(pred1,pred2 両順・6点)
            hit, pay = False, 0
            if rank == "SS" and is_box:
                n_combos = 2 * len(thirds)
                pred = f"{p1}⇄{p2}→" + ",".join(map(str, thirds))
                for t in thirds:
                    for a, b in ((p1, p2), (p2, p1)):
                        if order[:3] == [a, b, t]:
                            pay = pm.get(rk, {}).get(("trifecta", (a, b, t)), 0); hit = True; break
                    if hit:
                        break
            elif rank == "SS":
                n_combos = len(thirds)
                pred = f"{p1}→{p2}→" + ",".join(map(str, thirds))
                for t in thirds:
                    if order[:3] == [p1, p2, t]:
                        pay = pm.get(rk, {}).get(("trifecta", (p1, p2, t)), 0); hit = True; break
            elif rank == "WIDE":
                n_combos = 1
                pred = f"{p1}-{p2}"
                if frozenset((p1, p2)).issubset(top3):
                    pay = pm.get(rk, {}).get(("quinellaPlace", frozenset((p1, p2))), 0); hit = True
            else:
                n_combos = len(thirds)
                pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
                for t in thirds:
                    if frozenset((p1, p2, t)) == top3:
                        pay = pm.get(rk, {}).get(("trio", frozenset((p1, p2, t))), 0); hit = True; break
            bet = n_combos * 100
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
            row_str = f"[{rank}] {venue} {race_no}R {tstr}  予:{pred}  実:{actual}  {mark}"
            if rank == "WIDE":
                wb += bet
                if hit:
                    wr += pay; wh += 1
                results_wide.append(row_str)
            else:
                tb += bet
                if hit:
                    tr += pay; th += 1
                results.append(row_str)
            # race_key は UNIQUE。同一レースで SS/S/A(main) と WIDE が並立しうるため
            # WIDE は "#W" 接尾でキーを分離（main 行の上書き＝既存成績破壊を防ぐ）。
            store_key = f"{rk}#W" if rank == "WIDE" else rk
            history.append((target_date, store_key, rank, pred, n_combos, int(hit), pay, bet))

        if history:
            conn.execute("DELETE FROM picks_history WHERE route='wt' AND race_date=?", (target_date,))
            conn.executemany(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,bet_amount,route) "
                "VALUES (?,?,?,?,?,?,?,?,'wt')", history)

    if not results and not results_wide:
        emit(f"📊 **競輪AI[wt]成績 {target_date}**\n確定レースなし")
        return

    month = _query_stats(target_date[:7] + "%")
    year = _query_stats(target_date[:4] + "%")
    stats = f"\n{'─'*28}\n📅 {target_date[:7]}: {_stats_line('月', month)}\n🗓 {target_date[:4]}年: {_stats_line('年', year)}"

    if results:
        roi = tr / tb * 100 if tb else 0
        header = (f"📊 **競輪AI[wt]成績 {target_date}**  [6車以下/SS:3連単・S+A:3連複]\n"
                  f"確定 {len(results)}R　的中 {th}回 ({th/len(results)*100:.1f}%)\n"
                  f"投資 {tb:,}円 → 回収 {tr:,}円　ROI {roi:.1f}%　損益 {tr-tb:+,}円")
        msg = f"{header}\n```\n" + "\n".join(results) + "\n```"
    else:
        msg = f"📊 **競輪AI[wt]成績 {target_date}**  推奨(SS/S/A)の確定なし"

    # ワイド1点（独立プロダクト・別集計）
    if results_wide:
        wroi = wr / wb * 100 if wb else 0
        wide_header = (f"\n🎯 **ワイド1点(指数1-2位)**　確定 {len(results_wide)}R　"
                       f"的中 {wh}回 ({wh/len(results_wide)*100:.1f}%)\n"
                       f"投資 {wb:,}円 → 回収 {wr:,}円　ROI {wroi:.1f}%　損益 {wr-wb:+,}円")
        msg += f"{wide_header}\n```\n" + "\n".join(results_wide) + "\n```"

    if skipped_dns:
        msg += f"\n※欠車返還によりレース無効: {skipped_dns}件（軸欠車/全相手欠車・損益不計上）"
    msg += stats
    emit(msg[:1900])
    print(f"[notify_results_wt] {target_date} SS/S/A {len(results)}R 的中{th} / "
          f"ワイド {len(results_wide)}R 的中{wh} / 欠車無効{skipped_dns}件")


if __name__ == "__main__":
    main()
