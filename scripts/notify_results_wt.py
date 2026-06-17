"""winticket 成績通知＋picks_history保存（7+車 S/A ランク専用）

wave_picks_wt_{date}.txt の公開買い目を、winticket の確定結果(wt_entries.finish_order)
と wt_odds(三連複) で採点し、Discord通知＋picks_history に保存する。
欠車(finish_order=0/NULL)は着外として除外。公開した買い目のみ採点（再導出しない）。
7+車 Sランク(#7S) / Aランク(#7A) 別に集計。
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
            if "【7+車 SSランク】" in line: rank = "7PLUS_SS"
            elif "【7+車 Sランク】" in line: rank = "7PLUS_S"
            elif "【7+車 Aランク】" in line: rank = "7PLUS_A"
            elif "【7+車】" in line: rank = "7PLUS_S"  # 旧フォーマット後方互換
            elif "【SSランク】" in line: rank = None   # 旧SS/S/A/B/WIDEは採点対象外
            elif "【Sランク】" in line: rank = None
            elif "【Aランク】" in line: rank = None
            elif "【Bランク】" in line: rank = None
            elif "【ワイド1点】" in line: rank = None
            elif rank:
                m = re.match(r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[\d+車\]\s+(.+?)\s+\(\d+点", line)
                if m:
                    slot = "7plus_ss" if rank == "7PLUS_SS" else "7plus_s" if rank == "7PLUS_S" else "7plus_a"
                    picks[(m.group(2), int(m.group(3)), slot)] = (rank, m.group(1), m.group(4))
    return picks


def _parse_combo(combo_str: str):
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-").replace("⇄", "-")   # ⇄=SS 1-2着BOX(両順)
    parts = body.split("-")
    thirds = [int(x) for x in parts[2].split(",")] if len(parts) >= 3 else []  # ワイド=2車で空
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
            "FROM picks_history WHERE route='wt' AND rank IN ('7PLUS_SS','7PLUS_S','7PLUS_A') AND race_date LIKE ?", (like,)).fetchone()
    return {"races": r[0] or 0, "hits": r[1] or 0, "returns": r[2] or 0, "bets": r[3] or 0}


def _query_stats_rank(like, rank):
    """ランク別の統計を取得。"""
    with get_connection() as conn:
        r = conn.execute(
            "SELECT COUNT(*), SUM(hit), SUM(payout), SUM(bet_amount) "
            "FROM picks_history WHERE route='wt' AND rank=? AND race_date LIKE ?", (rank, like)).fetchone()
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
        # ファイル不在(真のエラー) と 7+車推奨0件(静かな日・正常) を区別する
        picks_file = Path(__file__).parent.parent / "data" / "picks" / f"wave_picks_wt_{target_date}.txt"
        if not picks_file.exists():
            emit(f"⚠️ 競輪AI[wt] [{target_date}] 予想ファイルが見つかりません")
        else:
            emit(f"📊 競輪AI[wt] [{target_date}] 7+車推奨なし＝採点対象なし"
                 f"（gami≥5.0倍+gap12≥0.07 の該当レースなし）")
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

    results_7plus_ss, results_7plus_s, results_7plus_a, history = [], [], [], []
    p7ssb = p7ssr = p7ssh = 0  # 7+車 SSランク 合計
    p7sb = p7sr = p7sh = 0    # 7+車 Sランク 合計
    p7ab = p7ar = p7ah = 0    # 7+車 Aランク 合計
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
            hit, pay = False, 0
            # 7+車は常に三連複（全相手流し）
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
            rank_label = "7SS" if rank == "7PLUS_SS" else "7S" if rank == "7PLUS_S" else "7A"
            row_str = f"[{rank_label}] {venue} {race_no}R {tstr}  予:{pred}  実:{actual}  {mark}"
            if rank == "7PLUS_SS":
                p7ssb += bet
                if hit:
                    p7ssr += pay; p7ssh += 1
                results_7plus_ss.append(row_str)
            elif rank == "7PLUS_S":
                p7sb += bet
                if hit:
                    p7sr += pay; p7sh += 1
                results_7plus_s.append(row_str)
            else:  # 7PLUS_A および旧 7PLUS
                p7ab += bet
                if hit:
                    p7ar += pay; p7ah += 1
                results_7plus_a.append(row_str)
            # race_key suffix: 7PLUS_SS → #7SS / 7PLUS_S → #7S / 7PLUS_A → #7A
            if rank == "7PLUS_SS":
                store_key = f"{rk}#7SS"
            elif rank == "7PLUS_S":
                store_key = f"{rk}#7S"
            else:
                store_key = f"{rk}#7A"
            history.append((target_date, store_key, rank, pred, n_combos, int(hit), pay, bet))

        if history:
            conn.execute("DELETE FROM picks_history WHERE route='wt' AND race_date=?", (target_date,))
            conn.executemany(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,bet_amount,route) "
                "VALUES (?,?,?,?,?,?,?,?,'wt')", history)

    total_7plus = results_7plus_ss + results_7plus_s + results_7plus_a
    if not total_7plus:
        emit(f"📊 **競輪AI[wt]成績 {target_date}**\n確定レースなし")
        return

    p7b = p7ssb + p7sb + p7ab
    p7r = p7ssr + p7sr + p7ar
    p7h = p7ssh + p7sh + p7ah
    p7roi = p7r / p7b * 100 if p7b else 0
    n7 = len(total_7plus)
    header = (
        f"📊 **競輪AI[wt]成績 {target_date}**  [7+車]\n"
        f"確定 {n7}R　的中 {p7h}回 ({p7h/n7*100:.1f}%)\n"
        f"投資 {p7b:,}円 → 回収 {p7r:,}円　ROI {p7roi:.1f}%　損益 {p7r-p7b:+,}円"
    )

    # ランク別サマリー
    def _rank_line(label, results_list, bet_total, ret_total, hit_count):
        if not results_list:
            return ""
        roi = ret_total / bet_total * 100 if bet_total else 0
        return (f"[7+車 {label}] {len(results_list)}R 的中{hit_count} "
                f"投資{bet_total:,}→回収{ret_total:,} ROI{roi:.1f}%")

    rank_lines = []
    ss_line = _rank_line("SS", results_7plus_ss, p7ssb, p7ssr, p7ssh)
    s_line  = _rank_line("S",  results_7plus_s,  p7sb,  p7sr,  p7sh)
    a_line  = _rank_line("A",  results_7plus_a,  p7ab,  p7ar,  p7ah)
    if ss_line: rank_lines.append(ss_line)
    if s_line:  rank_lines.append(s_line)
    if a_line:  rank_lines.append(a_line)

    msg = header
    if rank_lines:
        msg += "\n" + "\n".join(rank_lines)
    msg += "\n```\n" + "\n".join(total_7plus) + "\n```"

    if skipped_dns:
        msg += f"\n※欠車返還によりレース無効: {skipped_dns}件（軸欠車/全相手欠車・損益不計上）"

    month = _query_stats(target_date[:7] + "%")
    year = _query_stats(target_date[:4] + "%")
    msg += f"\n{'─'*28}\n📅 {target_date[:7]}: {_stats_line('月', month)}\n🗓 {target_date[:4]}年: {_stats_line('年', year)}"

    emit(msg[:1900])
    print(f"[notify_results_wt] {target_date} 7+車SS {len(results_7plus_ss)}R 的中{p7ssh} / "
          f"7+車S {len(results_7plus_s)}R 的中{p7sh} / 7+車A {len(results_7plus_a)}R 的中{p7ah} / "
          f"欠車無効{skipped_dns}件")


if __name__ == "__main__":
    main()
