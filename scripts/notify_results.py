#!/usr/bin/env python3
"""
当日の6車立て以下レース（jiku2_3）成績を集計して Discord へ通知する。
daily_picks.sh から呼び出す（前日の結果確認用）。

wave_picks_{target_date}.txt の A/B ランクレースについて
3連複2軸×3頭流し(jiku2_3)の的中・払戻を集計して picks_history に蓄積する。
"""
import re
import sys
import itertools
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.discord import send
from src.evaluation.backtest import _load_payouts
from src.database import get_connection


def _parse_picks_keys(target_date: str) -> dict[tuple, str]:
    """wave_picks_{date}.txt から {(venue_name, race_no): rank} を返す。"""
    return {k: v for k, (v, *_) in _parse_picks_full(target_date).items()}


def _parse_combo(combo_str: str) -> tuple[int, int, list[int]]:
    """'3連単: 4→2→1,5,3' / '3連複: 2-4-5,1,3' → (pivot1, pivot2, [thirds])"""
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-")
    parts = body.split("-")
    p1, p2 = int(parts[0]), int(parts[1])
    thirds = [int(x) for x in parts[2].split(",")][:3]
    return p1, p2, thirds


def _parse_picks_full(target_date: str) -> dict[tuple, tuple]:
    """wave_picks_{date}.txt から {(venue_name, race_no): (rank, time, combo_str)} を返す。"""
    picks_path = (
        Path(__file__).parent.parent / "data" / "picks"
        / f"wave_picks_{target_date}.txt"
    )
    if not picks_path.exists():
        return {}

    text = picks_path.read_text(encoding="utf-8")
    picks = {}
    current_rank = None

    for line in text.splitlines():
        if "【SSランク】" in line:
            current_rank = "SS"
        elif "【Sランク】" in line:
            current_rank = "S"
        elif "【Aランク】" in line:
            current_rank = "A"
        elif "【Bランク】" in line:
            current_rank = "A"
        elif current_rank:
            m = re.match(
                r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[\d+車\]\s+(.+?)\s+\(\d+点",
                line
            )
            if m:
                picks[(m.group(2), int(m.group(3)))] = (
                    current_rank, m.group(1), m.group(4)
                )

    return picks


def _save_picks_history(records: list[dict]):
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO picks_history
                (race_date, race_key, rank, pred_combo, n_combos, hit, payout, bet_amount)
            VALUES (:race_date, :race_key, :rank, :pred_combo, :n_combos, :hit, :payout, :bet_amount)
            ON CONFLICT(race_key) DO UPDATE SET
                hit        = excluded.hit,
                payout     = excluded.payout,
                bet_amount = excluded.bet_amount
        """, records)


def _query_stats(period_filter: str, params: tuple) -> dict:
    with get_connection() as conn:
        row = conn.execute(f"""
            SELECT COUNT(*), SUM(hit), SUM(bet_amount), SUM(payout)
            FROM picks_history
            WHERE {period_filter}
        """, params).fetchone()
    races = row[0] or 0; hits = row[1] or 0
    bets  = row[2] or 0; returns = row[3] or 0
    return {"races": races, "hits": hits, "bets": bets, "returns": returns,
            "roi": returns / bets if bets else 0}


def _stats_line(label: str, s: dict) -> str:
    if s["races"] == 0:
        return f"{label}: データなし"
    hit_pct = s["hits"] / s["races"] * 100
    return (
        f"{label}: {s['races']}R 的中{s['hits']}回 {hit_pct:.1f}%  "
        f"投資{s['bets']:,}円 回収{s['returns']:,}円 ROI {s['roi']:.1%}"
    )


def main():
    target_date  = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m-%d")
    year_prefix  = target_date[:4]
    month_prefix = target_date[:7]

    try:
        with get_connection() as conn:
            venue_map = dict(conn.execute("SELECT venue_code, name FROM venue_info").fetchall())
            start_time_map = dict(conn.execute(
                "SELECT race_key, start_time FROM races WHERE race_date = ?",
                (target_date,)
            ).fetchall())
    except Exception:
        venue_map = {}
        start_time_map = {}

    # 公開した予想ファイルを「正」として採点する。
    # （翌朝の再収集データでモデルを再実行すると、公開時と買い目が変わったり
    #   未公開レースが混入して成績が乖離するため。実際に配信した買い目のみ採点）
    picks_full = _parse_picks_full(target_date)
    if not picks_full:
        send(f"⚠️ 競輪AI [{target_date}] 予想ファイルが見つかりません")
        return

    name2code = {name: code for code, name in venue_map.items()}
    dc = target_date.replace("-", "")
    all_keys = [
        f"{dc}_{name2code[v]}_{int(rn):02d}"
        for (v, rn) in picks_full if v in name2code
    ]
    payout_map = _load_payouts(all_keys)

    results      = []
    history_rows = []
    total_bets = total_returns = total_hits = 0

    with get_connection() as conn:
        ordered_picks = sorted(picks_full.items(), key=lambda x: (x[0][0], x[0][1]))
        for (venue, race_no), (rank, pub_time, combo_str) in ordered_picks:
            code = name2code.get(venue)
            if code is None:
                continue
            race_key = f"{dc}_{code}_{int(race_no):02d}"

            rows = conn.execute(
                "SELECT frame_no FROM race_results WHERE race_key=? "
                "AND finish_position<=3 ORDER BY finish_position", (race_key,)
            ).fetchall()
            actual_order = [int(r[0]) for r in rows]
            if len(actual_order) < 3:
                continue  # 結果未確定レースはスキップ
            top3_set = frozenset(actual_order[:3])

            pivot1, pivot2, thirds = _parse_combo(combo_str)
            thirds_str = ",".join(str(t) for t in thirds)
            n_combos = len(thirds)
            bet_amt  = n_combos * 100
            total_bets += bet_amt

            hit    = False
            payout = 0

            if rank == "SS":
                pred_str = f"{pivot1}→{pivot2}→{thirds_str}"
                for t in thirds:
                    if actual_order[:3] == [pivot1, pivot2, t]:
                        payout = payout_map.get(race_key, {}).get(("trifecta", f"{pivot1}-{pivot2}-{t}"), 0)
                        hit = True; total_returns += payout; total_hits += 1
                        break
            else:
                pred_str = f"{pivot1}-{pivot2}-{thirds_str}"
                for t in thirds:
                    if frozenset([pivot1, pivot2, t]) == top3_set:
                        pk = "=".join(map(str, sorted(top3_set)))
                        payout = payout_map.get(race_key, {}).get(("trifecta_box", pk), 0)
                        hit = True; total_returns += payout; total_hits += 1
                        break

            start_time = start_time_map.get(race_key) or pub_time or "--:--"
            actual_str = "-".join(str(x) for x in actual_order[:3])

            if hit:
                block = (f"[{rank}] {venue} {race_no}R  {start_time}  "
                         f"予:{pred_str}  実:{actual_str}  ◎ ¥{payout:,}")
            else:
                # 外れ時: 実際の winning combo の配当をカッコ付きで表示
                if rank == "SS":
                    actual_key = "-".join(map(str, actual_order[:3]))
                    actual_payout = payout_map.get(race_key, {}).get(("trifecta", actual_key), 0)
                else:
                    pk = "=".join(map(str, sorted(top3_set)))
                    actual_payout = payout_map.get(race_key, {}).get(("trifecta_box", pk), 0)
                pay_str = f" (実¥{actual_payout:,})" if actual_payout else ""
                block = (f"[{rank}] {venue} {race_no}R  {start_time}  "
                         f"予:{pred_str}  実:{actual_str}  ×{pay_str}")
            results.append(block)

            history_rows.append({
                "race_date":  target_date,
                "race_key":   race_key,
                "rank":       rank,
                "pred_combo": pred_str,
                "n_combos":   n_combos,
                "hit":        int(hit),
                "payout":     payout,
                "bet_amount": bet_amt,
            })

    if history_rows:
        _save_picks_history(history_rows)

    confirmed_n = len(results)
    if confirmed_n == 0:
        send(f"📊 **競輪AI成績 {target_date}**\n6車立て以下の確定レースなし")
        return

    roi    = total_returns / total_bets if total_bets else 0
    profit = total_returns - total_bets
    hit_pct = total_hits / confirmed_n * 100

    monthly = _query_stats("race_date LIKE ?", (month_prefix + "%",))
    yearly  = _query_stats("race_date LIKE ?", (year_prefix  + "%",))

    SEP = "─" * 28
    header = (
        f"📊 **競輪AI成績 {target_date}**  [6車立て以下/SS:3連単・S+A:3連複]\n"
        f"確定 {confirmed_n}R　的中 {total_hits}回 ({hit_pct:.1f}%)\n"
        f"投資 {total_bets:,}円　→　回収 {total_returns:,}円\n"
        f"ROI {roi:.1%}　損益 {profit:+,}円"
    )

    body = "\n".join(results)
    stats_block = (
        f"\n{SEP}\n"
        f"{_stats_line(f'📅 {month_prefix}', monthly)}\n"
        f"{_stats_line(f'🗓 {year_prefix}年', yearly)}"
    )

    msg = f"{header}\n```\n{body}\n```{stats_block}"
    if len(msg) > 1900:
        msg = msg[:1900] + "\n…(省略)"
    send(msg)

    # Xポスト用（的中のみ）
    md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"
    hit_blocks = [r for r in results if "◎" in r]

    if not hit_blocks:
        tw_none = (
            f"📊 穴車AI結果 {md}\n\n"
            f"的中なし ({confirmed_n}R中0回)\n\n"
            f"#競輪 #穴車AI #AI予想"
        )
        send(f"**--- Xポスト用（コピペ）---**\n```\n{tw_none}\n```")
    else:
        max_chars = 270
        tw_header  = f"📊 穴車AI結果 {md}\n\n"
        tw_footer  = "\n\n#競輪 #穴車AI #AI予想"
        tw_cont_hd = f"📊 穴車AI結果 {md}（続き）\n\n"
        tw_cont_ft = "\n\n#競輪 #穴車AI"

        tw_tweets = []; tw_body = ""
        cur_hd, cur_ft = tw_header, tw_footer

        for block in hit_blocks:
            # "[SS] 会場 NR HH:MM  予:A→B→C,D,E  実:X-Y-Z  ◎ ¥1,234"
            m = re.match(
                r"\[(SS|S|A)\]\s+(\S+)\s+(\d+)R\s+(\d+:\d+)\s+予:(\S+)\s+実:(\S+)\s+◎\s+¥([\d,]+)",
                block
            )
            if not m:
                continue
            rank_str = m.group(1)
            bet_type = "3連単" if rank_str == "SS" else "3連複"
            star = "⭐" if rank_str == "SS" else ""
            line = (
                f"◇ [{rank_str}] {m.group(2)} {m.group(3)}R  発走{m.group(4)}\n"
                f"  {bet_type} {m.group(5)}  実:{m.group(6)}  ◎ ¥{m.group(7)}{star}\n\n"
            )
            if len(cur_hd + tw_body + line + cur_ft) > max_chars and tw_body:
                tw_tweets.append(cur_hd + tw_body.rstrip() + cur_ft)
                tw_body = line; cur_hd, cur_ft = tw_cont_hd, tw_cont_ft
            else:
                tw_body += line

        if tw_body:
            tw_tweets.append(cur_hd + tw_body.rstrip() + cur_ft)

        for i, tw in enumerate(tw_tweets, 1):
            label = (f"**--- Xポスト用 {i}/{len(tw_tweets)}（コピペ）---**"
                     if len(tw_tweets) > 1 else "**--- Xポスト用（コピペ）---**")
            send(f"{label}\n```\n{tw}\n```")

    # 未確定レース通知
    picks_full = _parse_picks_full(target_date)
    confirmed_keys = set()
    for race_key in sorted(all_keys):
        grp = df[df["race_key"] == race_key].copy()
        if grp[grp["finish_position"].notna()].empty:
            continue
        grp_fin = grp[grp["finish_position"].notna()]
        n_riders = len(grp_fin)
        if n_riders > 6:
            continue
        venue_code = grp_fin["venue_code"].iloc[0]
        venue = venue_map.get(venue_code, venue_code)
        race_no = int(race_key.split("_")[-1])
        if (venue, race_no) in picks_full:
            confirmed_keys.add((venue, race_no))

    pending_lines = []
    for (venue, race_no), (rank, time, combo_str) in sorted(
        picks_full.items(), key=lambda x: x[1][1]
    ):
        if (venue, race_no) not in confirmed_keys:
            pending_lines.append(f"[{rank}] {venue} {race_no}R  {time}  {combo_str}")

    if pending_lines:
        pending_body = "\n".join(pending_lines)
        pending_msg = (
            f"⏳ **競輪AI未確定 {target_date}**  [{len(pending_lines)}件]\n"
            f"```\n{pending_body}\n```"
        )
        send(pending_msg)

    print(f"[notify_results] Discord 送信完了 ({target_date}, {confirmed_n}R, 的中{total_hits}回, 未確定{len(pending_lines)}件)")


if __name__ == "__main__":
    main()
