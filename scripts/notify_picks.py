#!/usr/bin/env python3
"""
wave-picks の結果を Discord へ通知する（7+車専用版）。
daily_picks_wt.sh から呼び出す。

朝の通知: 全候補レース（gap12≥0.07）をガミ判定付きで一覧表示 + 推奨ランク詳細 + 全指数PDF
夜の通知: 夜の部候補レース同様
"""
import json
import re
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.discord import send, send_file


def _parse_7plus_ranked(text: str) -> dict[str, list[dict]]:
    """wave-picks テキストから 7+車 SSランク・Sランク を抽出する。"""
    result: dict[str, list[dict]] = {"SS": [], "S": []}
    current_rank = None

    for line in text.splitlines():
        if "【7+車 SSランク】" in line:
            current_rank = "SS"
            continue
        if "【7+車 Sランク】" in line:
            current_rank = "S"
            continue
        if "【7+車 Aランク】" in line:
            current_rank = None  # 廃止済み
            continue
        if current_rank is None:
            continue
        if line.startswith("【"):
            current_rank = None
            continue

        # "  HH:MM  会場   NR  [N車]  3連複: A-B-C,D,E  (N点/M円)  [X.X倍]"
        # Sランクは "3連単F: 1→2,3→全  (10点/1,000円)  [minX.X倍]" 形式（2026-07-10〜）
        m = re.match(
            r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[(\d+)車\]\s+(?:3連複|3連単F):\s+(\S+)\s+\((\d+)点/([\d,]+)円\)(?:\s+\[(.+?)\])?",
            line
        )
        if m:
            result[current_rank].append({
                "start_time": m.group(1),
                "venue":      m.group(2),
                "race_no":    m.group(3),
                "n_riders":   m.group(4),
                "combo":      m.group(5),
                "n_points":   m.group(6),
                "stake":      m.group(7),
                "odds_label": m.group(8) or "",
            })

    return result


def _load_candidates(today: str, prefix: str, night: bool) -> list[dict]:
    """candidates.json から候補一覧を取得する。"""
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    suffix = "_night_candidates.json" if night else "_candidates.json"
    p = picks_dir / f"{prefix}_{today}{suffix}"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _fmt_candidates_block(cands: list[dict]) -> str:
    """全候補をガミ判定付きで整形した文字列を返す。"""
    lines = []
    sorted_cands = sorted(
        cands,
        key=lambda x: (x.get("start_time", ""), x.get("venue_name", ""), x.get("race_no", 0)),
    )
    for c in sorted_cands:
        venue   = c.get("venue_name", "?")
        race_no = c.get("race_no", "?")
        n       = c.get("n_riders", "?")
        gap12   = c.get("gap12", 0.0)
        start   = c.get("start_time", "--:--")
        min_odds  = c.get("min_trio_odds")
        gami_rank = c.get("gami_rank")

        if gami_rank:
            status = f"✅{gami_rank}"
        elif min_odds is not None and min_odds > 0:
            status = f"❌{min_odds:.1f}倍"
        else:
            status = "❌オッズなし"

        lines.append(
            f"{start} {venue:<5} {int(race_no):>2}R [{n}車] gap={gap12:.3f} {status}"
        )
    return "\n".join(lines)


def _send_candidates(cands: list[dict], title: str) -> None:
    """候補一覧を Discord に送信する（2000文字制限で分割）。"""
    if not cands:
        return
    block = _fmt_candidates_block(cands)
    lines = block.split("\n")

    chunk_lines: list[str] = []
    chunk_len = len(title) + 10  # コードブロック記号分

    for line in lines:
        if chunk_len + len(line) + 1 > 1800:
            send(f"{title}\n```\n" + "\n".join(chunk_lines) + "\n```")
            chunk_lines = []
            chunk_len = 10
            title = ""  # 2枚目以降はタイトルなし
        chunk_lines.append(line)
        chunk_len += len(line) + 1

    if chunk_lines:
        prefix = f"{title}\n" if title else ""
        send(f"{prefix}```\n" + "\n".join(chunk_lines) + "\n```")


def _generate_picks_pdf(detail_json_path: str, output_path: str, dpi: int = 150) -> bool:
    """全車指数PDFを生成してoutput_pathに保存する。"""
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

    # macOS: Hiragino Sans / Linux: IPAGothic
    import platform
    if platform.system() == "Darwin":
        plt.rcParams["font.family"] = "Hiragino Sans"
    else:
        plt.rcParams["font.family"] = "IPAGothic"
    plt.rcParams["axes.unicode_minus"] = False

    rank_colors = {
        "7PLUS_R": "#FFD700",
        "SS": "#FFD700", "S": "#AED6F1", "A": "#ABEBC6", "B": "#F5B7B1",
    }
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
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
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
    prefix = sys.argv[2] if len(sys.argv) > 2 else "wave_picks"
    night = len(sys.argv) > 3 and sys.argv[3] == "night"
    fname = f"{prefix}_{target_date}_night.txt" if night else f"{prefix}_{target_date}.txt"
    title_label = "競輪AI予想（夜の部）" if night else "競輪AI予想"
    picks_path = Path(__file__).parent.parent / "data" / "picks" / fname

    # 候補レース読み込み（朝: _candidates.json / 夜: _night_candidates.json）
    cands = _load_candidates(target_date, prefix, night)

    # txt ファイルから推奨ピック（SS/S/A）を抽出
    picks_by_rank: dict[str, list[dict]] = {"SS": [], "S": [], "A": []}
    if picks_path.exists():
        try:
            text = picks_path.read_text(encoding="utf-8")
            picks_by_rank = _parse_7plus_ranked(text)
        except Exception:
            text = ""
    else:
        text = ""

    ss_n = len(picks_by_rank["SS"])
    total = ss_n

    md = f"{int(target_date[5:7])}/{int(target_date[8:10])}"

    m_cost = re.search(r"推奨合計投資額:\s*([\d,]+)円", text) if text else None
    total_cost = m_cost.group(1) if m_cost else "—"

    n_cands = len(cands)

    def _send_index_pdf():
        """全レース指数PDF（allindex.json優先・無ければ推奨のみdetail.json）を送信。"""
        allidx = picks_path.parent / f"{prefix}_{target_date}_allindex.json"
        detail = picks_path.parent / f"{prefix}_{target_date}_detail.json"
        if allidx.exists():
            src, pdf, dpi, label = allidx, picks_path.parent / f"{prefix}_{target_date}_allindex.pdf", 100, "全レース指数"
        elif detail.exists():
            src, pdf, dpi, label = detail, picks_path.parent / f"{prefix}_{target_date}_detail.pdf", 150, "全車指数(推奨のみ)"
        else:
            print("[notify_picks] 指数JSONなし（wave-picks を先に実行）"); return
        if _generate_picks_pdf(str(src), str(pdf), dpi=dpi):
            send_file(str(pdf), caption=f"📊 {label} {md}  S1:{ss_n}")
            print(f"[notify_picks] PDF 送信完了: {pdf}")
        else:
            print("[notify_picks] PDF 生成失敗")

    # ── ヘッダー送信 ──────────────────────────────────────────────────────────
    if total == 0:
        gami_skip = n_cands  # 全候補が条件不成立 or 候補なし
        header = (
            f"🚲 **{title_label} {target_date}**  [7+車]\n"
            f"推奨なし（S1条件不成立）　候補{gami_skip}件（gap12≥0.07）"
        )
    else:
        header = (
            f"🚲 **{title_label} {target_date}**  [7+車]\n"
            f"S1:{ss_n}件　投資:{total_cost}円\n"
            f"候補{n_cands}件（gap12≥0.07）"
        )
    send(header)

    # ── 推奨ランク詳細ブロック（SS/S/Aある場合のみ） ──────────────────────────
    def _fmt_rank_block(rank_label: str, picks: list[dict], desc: str) -> str:
        lines = [f"**【{rank_label}ランク】{len(picks)}件** （{desc}）"]
        for p in picks:
            odds_part = f"  [{p['odds_label']}]" if p.get("odds_label") else ""
            lines.append(
                f"  {p['start_time']}  {p['venue']:<6} {int(p['race_no']):>2}R"
                f"  [{p['n_riders']}車]  {p['combo']}  ({p['n_points']}点/{p['stake']}円){odds_part}"
            )
        return "\n".join(lines)

    if total > 0:
        sections = []
        if picks_by_rank.get("SS"):
            # 表示名は S1（2026-07-16 名称整理。内部キー"SS"はtxtパース互換のため不変）
            sections.append(_fmt_rank_block("S1", picks_by_rank["SS"], "全目min≥7倍+gap12≥0.10+gap23≥1pt  的中29%/ROI148%(2025)"))
        for section in sections:
            msg = f"```\n{section}\n```"
            if len(msg) > 1900:
                msg = msg[:1900] + "\n…(省略)```"
            send(msg)

    # ── 全候補一覧（ガミ判定付き） ─────────────────────────────────────────────
    if cands:
        scope = "夜の部" if night else "本日"
        _send_candidates(cands, f"📋 {scope}候補レース一覧（{n_cands}件 / gap12≥0.07 / 朝オッズ）")
    else:
        # candidates JSON がない場合（古い形式やエラー）はスキップ
        if not picks_path.exists():
            send(f"⚠️ 競輪AI [{target_date}] picks ファイルが見つかりません")

    # ── 全レース指数PDF ────────────────────────────────────────────────────────
    _send_index_pdf()

    print(f"[notify_picks] Discord 送信完了 ({target_date}{'/夜' if night else ''}, SS:{ss_n}, 候補:{n_cands})")


if __name__ == "__main__":
    main()
