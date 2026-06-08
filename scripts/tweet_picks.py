#!/usr/bin/env python3
"""
Aランク予想を X (Twitter) にポストする。
daily_picks.sh から呼び出す（8:00）。
280字制限を超える場合はスレッド投稿。
"""
import sys
import re
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.x_post import post_thread

MAX_CHARS = 270


def _build_tweets(target_date: str, a_entries: list[dict]) -> list[str]:
    md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"

    header  = f"🎯 穴車AI予想 {md}\n\n"
    footer  = "\n\n#競輪 #穴車AI #AI予想"
    cont_hd = f"🎯 穴車AI予想 {md}（続き）\n\n"
    cont_ft = "\n\n#競輪 #穴車AI"

    tweets = []
    current_body = ""
    cur_header = header
    cur_footer  = footer

    for e in a_entries:
        block = (
            f"◇ {e['venue']} {e['race_no']}R  発走{e['start_time']}\n"
            f"  3連複 {e['combo']}（{e['n_combos']}点）\n\n"
        )
        if len(cur_header + current_body + block + cur_footer) > MAX_CHARS and current_body:
            tweets.append(cur_header + current_body.rstrip() + cur_footer)
            current_body = block
            cur_header = cont_hd
            cur_footer  = cont_ft
        else:
            current_body += block

    if current_body:
        tweets.append(cur_header + current_body.rstrip() + cur_footer)

    return tweets


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m-%d")
    picks_path = Path(__file__).parent.parent / "data" / "picks" / f"wave_picks_{target_date}.txt"

    if not picks_path.exists():
        print(f"[tweet_picks] picks ファイルが見つかりません: {picks_path}")
        return

    text = picks_path.read_text(encoding="utf-8")

    # Aランクエントリを抽出
    a_entries = []
    in_a = False
    current_entry: dict = {}

    for line in text.splitlines():
        if "【Aランク】" in line:
            in_a = True
            continue
        if in_a and line.startswith("┌"):
            break
        if not in_a:
            continue

        m_hd = re.match(r"\s+◇\s+(\S+)\s+(\d+)R.*発走:(\d{1,2}:\d{2})", line)
        if m_hd:
            current_entry = {
                "venue":      m_hd.group(1),
                "race_no":    m_hd.group(2),
                "start_time": m_hd.group(3),
            }
            continue

        m_cb = re.match(r"\s+→\s+3連複:\s+(\S+)\s+\((\d+)点", line)
        if m_cb and current_entry:
            current_entry["combo"]    = m_cb.group(1)
            current_entry["n_combos"] = int(m_cb.group(2))
            a_entries.append(current_entry)
            current_entry = {}

    if not a_entries:
        md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"
        text = f"🎯 穴車AI予想 {md}\n\n本日の対象レースはありません\n\n#競輪 #穴車AI #AI予想"
        post_thread([text])
        print(f"[tweet_picks] {target_date} はAランク対象なし、対象なしツイートを投稿")
        return

    tweets = _build_tweets(target_date, a_entries)
    ok = post_thread(tweets)
    if ok:
        print(f"[tweet_picks] X投稿完了 ({target_date}, {len(a_entries)}件, {len(tweets)}ツイート)")
    else:
        print(f"[tweet_picks] X投稿失敗")


if __name__ == "__main__":
    main()
