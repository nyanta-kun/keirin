#!/usr/bin/env python3
"""
wave-picks の結果を Discord へ通知する。
daily_picks.sh から呼び出す。

ランク別フォーマット:
  SS: "  HH:MM  会場   NR  [N車]  3連単: A→B→C,D,E  (3点/300円)"
  S/A: "  HH:MM  会場   NR  [N車]  3連複: A-B-C,D,E  (3点/300円)"
"""
import json
import re
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.discord import send, send_file


def _parse_picks(text: str) -> dict[str, list[dict]]:
    """wave-picks テキストから SS/S/A ランク別にエントリを抽出する。"""
    result = {"SS": [], "S": [], "A": []}
    current_rank = None

    for line in text.splitlines():
        if "【SSランク】" in line:
            current_rank = "SS"
            continue
        if "【Sランク】" in line:
            current_rank = "S"
            continue
        if "【Aランク】" in line:
            current_rank = "A"
            continue
        if current_rank is None:
            continue

        # SS: "  HH:MM  会場   NR  [N車]  3連単: A→B→C,D,E  (3点/300円)"
        # S/A: "  HH:MM  会場   NR  [N車]  3連複: A-B-C,D,E  (3点/300円)"
        m = re.match(
            r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[\d+車\]\s+(3連単|3連複):\s+(\S+)\s+\(",
            line
        )
        if m:
            result[current_rank].append({
                "start_time": m.group(1),
                "venue":      m.group(2),
                "race_no":    m.group(3),
                "bet_type":   m.group(4),
                "combo":      m.group(5),
            })

    return result


def _build_tweet_texts(target_date: str, picks_by_rank: dict) -> list[str]:
    """Xへのコピペ用テキストを280字制限で分割して返す。"""
    md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"
    max_chars = 270

    header  = f"🎯 穴車AI予想 {md}\n\n"
    footer  = "\n\n#競輪 #穴車AI #AI予想"
    cont_hd = f"🎯 穴車AI予想 {md}（続き）\n\n"
    cont_ft = "\n\n#競輪 #穴車AI"

    # SS→S→A の順でまとめる（SSを先頭に）
    all_picks = [("SS", p) for p in picks_by_rank.get("SS", [])] \
              + [("S",  p) for p in picks_by_rank.get("S",  [])] \
              + [("A",  p) for p in picks_by_rank.get("A",  [])]

    tweets = []
    body = ""
    cur_hd, cur_ft = header, footer

    for rank, p in all_picks:
        if rank == "SS":
            line2 = f"  3連単 {p['combo']}（3点）⭐"
        else:
            line2 = f"  3連複 {p['combo']}（3点）"

        block = (
            f"◇ [{rank}] {p['venue']} {p['race_no']}R  発走{p['start_time']}\n"
            f"{line2}\n\n"
        )
        if len(cur_hd + body + block + cur_ft) > max_chars and body:
            tweets.append(cur_hd + body.rstrip() + cur_ft)
            body = block
            cur_hd, cur_ft = cont_hd, cont_ft
        else:
            body += block

    if body:
        tweets.append(cur_hd + body.rstrip() + cur_ft)

    return tweets


def _generate_picks_pdf(detail_json_path: str, output_path: str) -> bool:
    """全車指数PDFを生成してoutput_pathに保存する。
    matplotlib でページごとにPNG→Pillowで1本のPDFに結合。
    """
    import tempfile

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PDF] matplotlib が必要です: pip install matplotlib")
        return False

    try:
        from PIL import Image
    except ImportError:
        print("[PDF] Pillow が必要です: pip install Pillow")
        return False

    path = Path(detail_json_path)
    if not path.exists():
        print(f"[PDF] detail JSON が見つかりません: {detail_json_path}")
        return False

    races = json.loads(path.read_text(encoding="utf-8"))
    if not races:
        return False

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    rank_colors = {"SS": "#FFD700", "S": "#AED6F1", "A": "#ABEBC6"}
    role_bg = {"軸1": "#AED6F1", "軸2": "#D6EAF8", "流し": "#EBF5FB", "-": "#FFFFFF"}
    col_labels = ["AI順", "車番", "クラス", "期", "得点", "勝率3m%", "脚質", "AI確率%", "役割"]

    png_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, race in enumerate(races):
            rank = race["rank"]
            title = (
                f"[{rank}]  {race['venue_name']}  {race['race_no']}R  {race['start_time']}"
                f"    gap12={race['gap12']:.3f}  ratio={race['ratio']:.2f}"
            )
            subtitle = f"{race['bet_type']}: {race['combo_str']}"
            riders = sorted(race.get("riders", []), key=lambda r: r["frame_no"])

            table_data = []
            row_colors = []
            for r in riders:
                table_data.append([
                    str(r["ai_rank"]),
                    str(r["frame_no"]),
                    r["player_class"],
                    str(r["period"]) if r["period"] else "-",
                    f"{r['racing_score']:.1f}",
                    f"{r['win_rate_3m']:.1f}",
                    r["line_position"],
                    f"{r['pred_prob_pct']:.1f}",
                    r["role"],
                ])
                row_colors.append([role_bg.get(r["role"], "#FFFFFF")] * len(col_labels))

            n = len(riders)
            fig_h = max(3.5, 1.8 + n * 0.45)
            fig, ax = plt.subplots(figsize=(9, fig_h))
            ax.axis("off")

            title_bg = rank_colors.get(rank, "#EEEEEE")
            ax.text(
                0.5, 1.0, title,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(facecolor=title_bg, edgecolor="gray", boxstyle="round,pad=0.3"),
            )
            ax.text(
                0.5, 0.88, subtitle,
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
            )

            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc="center",
                bbox=[0.0, 0.0, 1.0, 0.78],
                cellColours=row_colors,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.auto_set_column_width(list(range(len(col_labels))))

            hdr_color = rank_colors.get(rank, "#CCCCCC")
            for (row_idx, _col_idx), cell in table.get_celld().items():
                if row_idx == 0:
                    cell.set_facecolor(hdr_color)
                    cell.get_text().set_fontweight("bold")

            png_path = f"{tmpdir}/race_{i:03d}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            png_paths.append(png_path)

        if not png_paths:
            return False

        images = [Image.open(p).convert("RGB") for p in png_paths]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=150,
        )

    return True


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m-%d")
    # 第2引数でファイルプレフィックス指定（ks="wave_picks" / winticket="wave_picks_wt"）
    prefix = sys.argv[2] if len(sys.argv) > 2 else "wave_picks"
    picks_path = Path(__file__).parent.parent / "data" / "picks" / f"{prefix}_{target_date}.txt"

    if not picks_path.exists():
        send(f"⚠️ 競輪AI [{target_date}] picks ファイルが見つかりません")
        return

    text = picks_path.read_text(encoding="utf-8")
    picks_by_rank = _parse_picks(text)
    ss_n = len(picks_by_rank["SS"])
    s_n  = len(picks_by_rank["S"])
    a_n  = len(picks_by_rank["A"])
    total = ss_n + s_n + a_n

    m_cost = re.search(r"合計投資額:\s*([\d,]+)円", text)
    total_cost = m_cost.group(1) if m_cost else f"{total * 300:,}"

    if total == 0:
        md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"
        send(f"🏁 **競輪AI予想 {target_date}**\n本日の対象レースはありません（6車立て以下 gap12≥0.06 なし）")
        tweet_none = f"🎯 穴車AI予想 {md}\n\n本日の対象レースはありません\n\n#競輪 #穴車AI #AI予想"
        send(f"**--- Xポスト用（コピペ）---**\n```\n{tweet_none}\n```")
        return

    def fmt_section(rank_label, picks):
        if not picks:
            return ""
        lines = [f"**【{rank_label}ランク】{len(picks)}件**"]
        for p in picks:
            arrow = "⭐" if rank_label == "SS" else "  "
            lines.append(f"{arrow} {p['start_time']}  {p['venue']:<6} {int(p['race_no']):>2}R  {p['combo']}")
        return "\n".join(lines)

    sections = []
    for rank in ("SS", "S", "A"):
        s = fmt_section(rank, picks_by_rank[rank])
        if s:
            sections.append(s)

    detail = "\n\n".join(sections)
    msg = (
        f"🏁 **競輪AI予想 {target_date}**\n"
        f"SS:{ss_n}件 / S:{s_n}件 / A:{a_n}件　計{total}件\n"
        f"投資: {total_cost}円  (6車立て以下)\n"
        f"```\n{detail}\n```"
    )
    if len(msg) > 1900:
        msg = msg[:1900] + "\n…(省略)```"
    send(msg)

    tweets = _build_tweet_texts(target_date, picks_by_rank)
    for i, tw in enumerate(tweets, 1):
        label = (
            f"**--- Xポスト用 {i}/{len(tweets)}（コピペ）---**"
            if len(tweets) > 1
            else "**--- Xポスト用（コピペ）---**"
        )
        send(f"{label}\n```\n{tw}\n```")

    print(f"[notify_picks] Discord 送信完了 ({target_date}, SS:{ss_n}/S:{s_n}/A:{a_n})")

    # 全車指数PDF
    detail_json = picks_path.parent / f"{prefix}_{target_date}_detail.json"
    if detail_json.exists():
        pdf_path = str(picks_path.parent / f"{prefix}_{target_date}_detail.pdf")
        if _generate_picks_pdf(str(detail_json), pdf_path):
            md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"
            send_file(pdf_path, caption=f"📊 全車指数 {md}  SS:{ss_n}/S:{s_n}/A:{a_n}")
            print(f"[notify_picks] PDF 送信完了: {pdf_path}")
        else:
            print("[notify_picks] PDF 生成失敗")
    else:
        print(f"[notify_picks] detail JSON なし（wave-picks を先に実行してください）")


if __name__ == "__main__":
    main()
