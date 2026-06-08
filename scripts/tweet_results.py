#!/usr/bin/env python3
"""
Aランクの成績を X (Twitter) にポストする。
daily_picks.sh から呼び出す（23:00）。
"""
import sys
import itertools
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.x_post import post_thread
from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
from src.models.trainer import load_model
from src.evaluation.backtest import _load_payouts
from src.database import get_connection

MAX_CHARS = 270


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m-%d")
    md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"

    try:
        with get_connection() as conn:
            venue_map = dict(conn.execute("SELECT venue_code, name FROM venue_info").fetchall())
    except Exception:
        venue_map = {}

    df_raw = load_raw_data(min_date=target_date, max_date=target_date)
    if df_raw.empty:
        print(f"[tweet_results] {target_date} DBにデータなし、ツイートなし")
        return

    model = load_model("lgbm")
    df = build_features(df_raw)
    df = df.dropna(subset=FEATURE_COLS)
    df["pred_prob"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

    # Aランクのみ抽出
    a_races = {}
    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(grp)
        top1 = grp["pred_prob"].iloc[0]
        top2 = grp["pred_prob"].iloc[1] if n > 1 else 0.0
        gap12 = top1 - top2
        if 0.60 <= top1 < 0.70 and gap12 > 0.12:
            a_races[race_key] = grp

    if not a_races:
        print(f"[tweet_results] {target_date} はAランク対象なし、ツイートなし")
        return

    payout_map = _load_payouts(list(a_races.keys()))

    results = []
    total_bets = total_returns = total_hits = 0

    for race_key in sorted(a_races.keys()):
        grp = a_races[race_key]
        r = grp["frame_no"].tolist()
        p1, p2 = int(r[0]), int(r[1])
        thirds = [int(r[i]) for i in range(2, 5) if i < len(r)]
        combos = [frozenset([p1, p2, t]) for t in thirds]
        pred_str = f"{p1}-{p2}-{','.join(str(t) for t in thirds)}"

        top3_set = frozenset(grp[grp["finish_position"] <= 3]["frame_no"].tolist())
        has_result = len(top3_set) == 3
        actual_order = []
        if has_result:
            actual_order = list(
                grp[grp["finish_position"].isin([1, 2, 3])]
                .sort_values("finish_position")["frame_no"]
                .astype(int)
            )

        bet_amt = len(combos) * 100
        total_bets += bet_amt

        hit = False
        payout = 0
        if has_result:
            pk = "=".join(map(str, sorted(top3_set)))
            pay = payout_map.get(race_key, {}).get(("trifecta_box", pk), 0)
            for c in combos:
                if c == top3_set:
                    hit = True
                    payout = pay
                    total_returns += pay
                    total_hits += 1
                    break

        venue_code = grp["venue_code"].iloc[0]
        venue = venue_map.get(venue_code, venue_code)
        race_no = int(race_key.split("_")[-1])

        if not has_result:
            line = f"◇ {venue} {race_no}R  {pred_str}\n  結果待ち"
        elif hit:
            actual_str = "-".join(str(x) for x in actual_order)
            line = f"◇ {venue} {race_no}R  {pred_str}\n  実:{actual_str}  ◎ ¥{payout:,}"
        else:
            actual_str = "-".join(str(x) for x in actual_order)
            line = f"◇ {venue} {race_no}R  {pred_str}\n  実:{actual_str}  ×"

        results.append(line)

    roi = total_returns / total_bets if total_bets > 0 else 0
    profit = total_returns - total_bets

    summary_line = (
        f"投資{total_bets:,}円 → 回収{total_returns:,}円\n"
        f"ROI {roi:.0%}  損益 {profit:+,}円"
    )

    header  = f"📊 穴車AI結果 {md}\n\n"
    footer  = f"\n{summary_line}\n#競輪 #穴車AI #AI予想"
    cont_hd = f"📊 穴車AI結果 {md}（続き）\n\n"
    cont_ft = "\n#競輪 #穴車AI"

    tweets = []
    current_body = ""
    cur_header = header
    cur_footer  = footer

    for line in results:
        block = line + "\n\n"
        if len(cur_header + current_body + block + cur_footer) > MAX_CHARS and current_body:
            tweets.append(cur_header + current_body.rstrip() + cur_footer)
            current_body = block
            cur_header = cont_hd
            cur_footer  = cont_ft
        else:
            current_body += block

    if current_body:
        tweets.append(cur_header + current_body.rstrip() + cur_footer)

    ok = post_thread(tweets)
    if ok:
        print(f"[tweet_results] X投稿完了 ({target_date}, {len(results)}件, 的中{total_hits}回, {len(tweets)}ツイート)")
    else:
        print(f"[tweet_results] X投稿失敗")


if __name__ == "__main__":
    main()
