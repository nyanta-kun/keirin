"""朝の入稿分をレース単位でまとめた確認用ページを生成する（2026-08-04 新設）。

ユーザー要望:
  「朝の入稿通知の際に WINTICKET との印比較、推奨買い目のオッズ、合成オッズも
    レース単位でまとめて送る、もしくは一覧ページ作成の上リンクを送って」

各推奨レースについて次を1ページにまとめる:
  - 出走表（車番・選手・級班・脚質・得点・ライン）
  - **WINTICKET 公式印（◎◯△）とモデル評価の対比** … 軸2車がどの印と一致したか
  - 推奨買い目と**朝オッズ**（wt_odds_snapshot の snapshot_type='morning'）
  - **合成オッズ** 1/Σ(1/oᵢ) と、均等買い時にガミになる目の明示

オッズは入稿時点の朝スナップショットを使う（発走直前の変動は反映しない）。
朝スナップショットが無い目は最終オッズ（wt_odds）へフォールバックし、その旨を表示する。

DB書き込みなし。

使い方:
    python scripts/morning_summary_wt.py 2026-08-04 --out /tmp/keirin_morning.html
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 競輪の車番色（実際のユニフォーム/枠の配色）。視認性のため文字色も併記する。
FRAME_COLORS = {
    1: ("#f5f5f5", "#1a1a1a"), 2: ("#1a1a1a", "#f5f5f5"), 3: ("#d92b2b", "#ffffff"),
    4: ("#1f5fd0", "#ffffff"), 5: ("#f0c419", "#1a1a1a"), 6: ("#2c9e4b", "#ffffff"),
    7: ("#e8730c", "#ffffff"), 8: ("#e87fa8", "#1a1a1a"), 9: ("#7d4bc3", "#ffffff"),
}
MARK_LABEL = {1: "◎", 2: "◯", 3: "△", 4: "×"}
RANK_LABEL = {"RANK_7S": "7S", "RANK_7A": "7A", "RANK_7B": "7B",
              "RANK_9S": "9S", "RANK_9A": "9A"}


def _engine():
    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        raise RuntimeError("KEIRIN_DB_URL が未設定です")
    from sqlalchemy import create_engine
    return create_engine(db_url)


def _q(sql: str) -> list[dict]:
    """⚠️ get_connection() を pandas/DBAPI 経由で使うと RealDictCursor のせいで
    全行が列名文字列になる事故があるため、SQLAlchemy engine で明示的に読む。"""
    from sqlalchemy import text
    eng = _engine()
    with eng.connect() as c:
        rows = [dict(r._mapping) for r in c.execute(text(sql))]
    eng.dispose()
    return rows


def parse_combo(pred_combo: str) -> tuple[list[int], list[int]]:
    """'5=3-1,2,4,6,7' → 軸[5,3] / 相手[1,2,4,6,7]。"""
    if not pred_combo or "-" not in pred_combo:
        return [], []
    axis_part, legs_part = pred_combo.split("-", 1)
    axes = [int(x) for x in axis_part.replace("=", ",").split(",") if x.strip().isdigit()]
    legs = [int(x) for x in legs_part.split(",") if x.strip().isdigit()]
    return axes, legs


def fetch(date: str) -> list[dict]:
    d8 = date.replace("-", "")
    picks = _q(f"""
        SELECT p.race_key, p.rank, p.gate_label, p.pred_combo,
               r.venue_id, r.race_no, r.start_at, r.grade, r.race_type, r.n_entries,
               COALESCE(v.name, r.venue_id) AS venue_name
        FROM keirin.picks_history p
        JOIN keirin.wt_races r ON r.race_key = split_part(p.race_key, '#', 1)
        LEFT JOIN keirin.venue_info v ON v.venue_code = r.venue_id
        WHERE p.race_date = '{date}'
        ORDER BY r.start_at, r.venue_id, r.race_no
    """)
    if not picks:
        return []

    base_keys = sorted({p["race_key"].split("#")[0] for p in picks})
    in_keys = ",".join(f"'{k}'" for k in base_keys)

    ent = defaultdict(list)
    for e in _q(f"""
        SELECT race_key, frame_no, name, player_class, style, race_point,
               prediction_mark, pred_win_pct, pred_top3_pct, line_group, line_pos
        FROM keirin.wt_entries WHERE race_key IN ({in_keys}) ORDER BY frame_no
    """):
        ent[e["race_key"]].append(e)

    # ⚠️ 9999.9 は Winticket 側の「未確定プレースホルダ」で実オッズではない。
    #    朝スナップショットには相応の割合で混入する（2026-08-04 実測: trio 全体の
    #    17.1%）。これを実値として扱うと合成オッズ 1/Σ(1/oᵢ) が過大になる
    #    （1/9999.9 の寄与がほぼゼロのため）。kiseki backend の _calc_synth_odds が
    #    2026-07-20 に踏んだ罠と同型。未確定として除外し、件数を明示する。
    PLACEHOLDER = 9999.0

    def _valid(v) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if 0 < f < PLACEHOLDER else None

    morning: dict[tuple[str, str], float] = {}
    for o in _q(f"""
        SELECT race_key, combination, odds_value FROM keirin.wt_odds_snapshot
        WHERE bet_type='trio' AND snapshot_type='morning'
          AND race_key IN ({in_keys})
    """):
        v = _valid(o["odds_value"])
        if v is not None:
            morning[(o["race_key"], o["combination"])] = v
    latest: dict[tuple[str, str], float] = {}
    for o in _q(f"""
        SELECT race_key, combination, odds_value FROM keirin.wt_odds
        WHERE bet_type='trio' AND race_key IN ({in_keys})
    """):
        v = _valid(o["odds_value"])
        if v is not None:
            latest[(o["race_key"], o["combination"])] = v

    out = []
    for p in picks:
        base = p["race_key"].split("#")[0]
        axes, legs = parse_combo(p["pred_combo"])
        rows = ent.get(base, [])
        marks = {int(e["frame_no"]): e["prediction_mark"] for e in rows}
        bets = []
        for x in legs:
            # combination の表記は収集経路で混在する（VPS収集='1-2-3' /
            # 旧Mac収集='1=2=3'）。両方を照合しないと過去日で無言の欠損になる
            # （kiseki backend の _parse_combinations が同じ対策を持つ）。
            nums = sorted(axes + [x])
            keys = ["-".join(str(n) for n in nums), "=".join(str(n) for n in nums)]
            od = next((morning[(base, k)] for k in keys if (base, k) in morning), None)
            src = "朝"
            if od is None:
                od = next((latest[(base, k)] for k in keys if (base, k) in latest), None)
                src = "最終" if od is not None else "未確定"
            bets.append({"combo": keys[0], "leg": x, "odds": od, "src": src})
        valid = [b["odds"] for b in bets if b["odds"]]
        syn = (1.0 / sum(1.0 / o for o in valid)) if valid else None
        n_pts = len(bets)
        for b in bets:
            # 均等買い（1点100円）で的中しても賭け金割れになる目＝ガミ
            b["gami"] = bool(b["odds"] and b["odds"] < n_pts)
        out.append({**p, "axes": axes, "legs": legs, "entries": rows,
                    "marks": marks, "bets": bets, "syn": syn,
                    "n_pts": n_pts, "n_unknown": n_pts - len(valid),
                    "min_odds": min(valid) if valid else None,
                    "max_odds": max(valid) if valid else None,
                    "wt_honmei": next((f for f, m in marks.items() if m == 1), None),
                    "wt_taikou": next((f for f, m in marks.items() if m == 2), None),
                    "wt_ana": next((f for f, m in marks.items() if m == 3), None)})
    return out


def frame_badge(n: int) -> str:
    bg, fg = FRAME_COLORS.get(n, ("#888", "#fff"))
    return (f'<span class="fno" style="background:{bg};color:{fg}">{n}</span>')


def render(date: str, races: list[dict]) -> str:
    by_rank: dict[str, int] = defaultdict(int)
    for r in races:
        by_rank[RANK_LABEL.get(r["rank"], r["rank"])] += 1
    rank_chips = "".join(
        f'<span class="chip chip-{k}">{k} <b>{v}</b></span>'
        for k, v in sorted(by_rank.items()))

    cards = []
    for r in races:
        lbl = RANK_LABEL.get(r["rank"], r["rank"])
        axes = r["axes"]
        # 印比較: 軸2車がWT◎◯とどれだけ重なるか
        ov = len(set(axes) & {r["wt_honmei"], r["wt_taikou"]} - {None})
        ov_txt = {0: "◎◯と不一致", 1: "片方一致", 2: "◎◯完全一致"}.get(ov, "-")

        trs = []
        for e in r["entries"]:
            f = int(e["frame_no"])
            role = "軸" if f in axes else ("相手" if f in r["legs"] else "")
            mk = MARK_LABEL.get(e["prediction_mark"], "")
            p3 = e["pred_top3_pct"]
            pw = e["pred_win_pct"]
            trs.append(
                f'<tr class="{"is-axis" if f in axes else ("is-leg" if f in r["legs"] else "is-off")}">'
                f'<td>{frame_badge(f)}</td>'
                f'<td class="nm">{html.escape(e["name"] or "")}'
                f'<span class="sub">{html.escape(e["player_class"] or "")}'
                f' / {html.escape(e["style"] or "")}</span></td>'
                f'<td class="num">{e["race_point"] or ""}</td>'
                f'<td class="mark">{mk}</td>'
                f'<td class="num">{"" if pw is None else f"{float(pw):.1f}"}</td>'
                f'<td class="num strong">{"" if p3 is None else f"{float(p3):.1f}"}</td>'
                f'<td class="role">{role}</td></tr>')

        bet_rows = []
        for b in r["bets"]:
            od = b["odds"]
            bet_rows.append(
                f'<tr class="{"gami" if b["gami"] else ""}">'
                f'<td class="combo">{b["combo"]}</td>'
                f'<td class="num">{"—" if od is None else f"{od:.1f}"}</td>'
                f'<td class="src">{b["src"]}</td>'
                f'<td class="flag">{"ガミ" if b["gami"] else ""}</td></tr>')

        n_gami = sum(1 for b in r["bets"] if b["gami"])
        nk = r["n_unknown"]
        syn_txt = "—" if r["syn"] is None else f"{r['syn']:.2f}"
        syn_note = (f'<span class="note">確定 {r["n_pts"] - nk}/{r["n_pts"]}点で算出</span>'
                    if nk else "")
        rng = ("—" if r["min_odds"] is None
               else f"{r['min_odds']:.1f} 〜 {r['max_odds']:.1f}")
        start = (r["start_at"] or "")[-5:] if r["start_at"] else ""

        cards.append(f"""
<article class="race">
  <header class="race-hd">
    <div class="hd-l">
      <span class="rk rk-{lbl}">{lbl}</span>
      <h2>{html.escape(r["venue_name"] or "")} <span class="rno">{r["race_no"]}R</span></h2>
      <span class="meta">{html.escape(r["grade"] or "")} · {html.escape(r["race_type"] or "")} · {r["n_entries"]}車</span>
    </div>
    <div class="hd-r"><span class="time">{start}</span></div>
  </header>

  <div class="cmp">
    <div class="cmp-i"><span class="k">モデル軸</span>
      <span class="v">{"".join(frame_badge(a) for a in axes)}</span></div>
    <div class="cmp-i"><span class="k">WT ◎◯△</span>
      <span class="v">{"".join(frame_badge(x) for x in [r["wt_honmei"], r["wt_taikou"], r["wt_ana"]] if x)}</span></div>
    <div class="cmp-i"><span class="k">印比較</span>
      <span class="v tag ov{ov}">{ov_txt}</span></div>
  </div>

  <div class="grid">
    <div class="col">
      <table class="ent">
        <thead><tr><th>車</th><th>選手</th><th>得点</th><th>WT</th>
          <th>単%</th><th>3着%</th><th></th></tr></thead>
        <tbody>{"".join(trs)}</tbody>
      </table>
    </div>
    <div class="col">
      <table class="bet">
        <thead><tr><th>買い目（三連複）</th><th>オッズ</th><th>時点</th><th></th></tr></thead>
        <tbody>{"".join(bet_rows)}</tbody>
      </table>
      <dl class="sum">
        <div><dt>合成オッズ</dt><dd class="big">{syn_txt}</dd>{syn_note}</div>
        <div><dt>オッズ範囲</dt><dd>{rng}</dd></div>
        <div><dt>点数 / 投資</dt><dd>{r["n_pts"]}点 / {r["n_pts"] * 100}円</dd></div>
        <div><dt>ガミ目</dt><dd class="{"warn" if n_gami else ""}">{n_gami} / {r["n_pts"]}</dd></div>
        {f'<div><dt>オッズ未確定</dt><dd class="warn">{nk} 点</dd></div>' if nk else ""}
      </dl>
    </div>
  </div>
</article>""")

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<title>朝の推奨まとめ {date}</title>
<style>
:root {{
  --bg:#f7f8fa; --panel:#ffffff; --line:#dfe3ea; --ink:#171b22; --dim:#5c6675;
  --accent:#0f6f5c; --warn:#b0452c; --axis:#eef6f3; --leg:#f6f8fb;
  --mono:ui-monospace,"SFMono-Regular",Menlo,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#12151a; --panel:#191d24; --line:#2a3039; --ink:#e6e9ee; --dim:#93a0b1;
    --accent:#5fd0b4; --warn:#ff8a6a; --axis:#16302b; --leg:#1c2129; }}
}}
:root[data-theme="dark"] {{ --bg:#12151a; --panel:#191d24; --line:#2a3039; --ink:#e6e9ee;
  --dim:#93a0b1; --accent:#5fd0b4; --warn:#ff8a6a; --axis:#16302b; --leg:#1c2129; }}
:root[data-theme="light"] {{ --bg:#f7f8fa; --panel:#ffffff; --line:#dfe3ea; --ink:#171b22;
  --dim:#5c6675; --accent:#0f6f5c; --warn:#b0452c; --axis:#eef6f3; --leg:#f6f8fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.55; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:28px 18px 64px; }}
.page-hd {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:14px;
  padding-bottom:14px; border-bottom:2px solid var(--ink); margin-bottom:22px; }}
.page-hd h1 {{ margin:0; font-size:1.5rem; letter-spacing:.01em; }}
.page-hd .d {{ font-family:var(--mono); font-size:1.05rem; color:var(--dim); }}
.chips {{ display:flex; gap:8px; margin-left:auto; flex-wrap:wrap; }}
.chip {{ font-size:.8rem; padding:3px 10px; border:1px solid var(--line);
  border-radius:999px; font-family:var(--mono); }}
.chip b {{ color:var(--accent); }}
.races {{ display:flex; flex-direction:column; gap:20px; }}
.race {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  overflow:hidden; }}
.race-hd {{ display:flex; align-items:center; gap:12px; padding:12px 16px;
  border-bottom:1px solid var(--line); flex-wrap:wrap; }}
.hd-l {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.hd-r {{ margin-left:auto; }}
.race-hd h2 {{ margin:0; font-size:1.05rem; }}
.rno {{ font-family:var(--mono); color:var(--dim); }}
.meta {{ font-size:.8rem; color:var(--dim); }}
.time {{ font-family:var(--mono); font-size:1.05rem; }}
.rk {{ font-family:var(--mono); font-weight:700; font-size:.85rem; padding:2px 9px;
  border-radius:4px; border:1px solid var(--accent); color:var(--accent); }}
.rk-7S {{ background:var(--accent); color:var(--panel); }}
.cmp {{ display:flex; gap:22px; padding:10px 16px; background:var(--leg);
  border-bottom:1px solid var(--line); flex-wrap:wrap; align-items:center; }}
.cmp-i {{ display:flex; align-items:center; gap:8px; }}
.cmp-i .k {{ font-size:.75rem; color:var(--dim); letter-spacing:.06em; }}
.cmp-i .v {{ display:flex; gap:4px; align-items:center; }}
.tag {{ font-size:.78rem; padding:2px 8px; border-radius:4px; border:1px solid var(--line); }}
.tag.ov0 {{ border-color:var(--accent); color:var(--accent); }}
.tag.ov2 {{ border-color:var(--warn); color:var(--warn); }}
.grid {{ display:grid; grid-template-columns:1.35fr 1fr; gap:0; }}
.col {{ padding:12px 16px; min-width:0; overflow-x:auto; }}
.col+.col {{ border-left:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
th {{ text-align:left; font-weight:600; font-size:.72rem; color:var(--dim);
  letter-spacing:.06em; padding:4px 6px; border-bottom:1px solid var(--line); }}
td {{ padding:5px 6px; border-bottom:1px solid var(--line); }}
tbody tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.strong {{ font-weight:700; }}
.nm {{ min-width:7em; }}
.nm .sub {{ display:block; font-size:.7rem; color:var(--dim); }}
.mark {{ text-align:center; font-size:1rem; }}
.role {{ font-size:.72rem; color:var(--accent); text-align:right; white-space:nowrap; }}
tr.is-axis {{ background:var(--axis); }}
tr.is-axis td {{ font-weight:600; }}
tr.is-leg {{ background:var(--leg); }}
tr.is-off {{ opacity:.5; }}
.fno {{ display:inline-flex; align-items:center; justify-content:center;
  width:22px; height:22px; border-radius:4px; font-family:var(--mono);
  font-weight:700; font-size:.8rem; border:1px solid rgba(128,128,128,.45); }}
.combo {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.src {{ font-size:.7rem; color:var(--dim); }}
.flag {{ font-size:.72rem; color:var(--warn); text-align:right; }}
tr.gami td {{ color:var(--warn); }}
.sum {{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px 14px;
  margin:12px 0 0; padding-top:10px; border-top:1px solid var(--line); }}
.sum div {{ display:flex; flex-direction:column; gap:1px; }}
dt {{ font-size:.7rem; color:var(--dim); letter-spacing:.06em; }}
dd {{ margin:0; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
dd.big {{ font-size:1.3rem; font-weight:700; color:var(--accent); }}
dd.warn {{ color:var(--warn); font-weight:700; }}
.note {{ font-size:.65rem; color:var(--dim); }}
footer {{ margin-top:28px; font-size:.75rem; color:var(--dim); }}
@media (max-width:820px) {{
  .grid {{ grid-template-columns:1fr; }}
  .col+.col {{ border-left:none; border-top:1px solid var(--line); }}
}}
</style>
<div class="wrap">
  <div class="page-hd">
    <h1>朝の推奨まとめ</h1>
    <span class="d">{date}</span>
    <div class="chips"><span class="chip">計 <b>{len(races)}</b> R</span>{rank_chips}</div>
  </div>
  <div class="races">{"".join(cards)}</div>
  <footer>
    オッズは入稿時点の朝スナップショット（無い目は最終オッズ・「時点」列に表示）。
    合成オッズ = 1/Σ(1/oᵢ)。ガミ = 均等買い（1点100円）で的中しても投資額を割る目。
    生成 {gen}
  </footer>
</div>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    races = fetch(args.date)
    if not races:
        print(f"{args.date} の推奨レースがありません")
        return
    out = Path(args.out or f"/tmp/keirin_morning_{args.date.replace('-', '')}.html")
    out.write_text(render(args.date, races), encoding="utf-8")
    print(f"{len(races)}レース → {out}")


if __name__ == "__main__":
    main()
