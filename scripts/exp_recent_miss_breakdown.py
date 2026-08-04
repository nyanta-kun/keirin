"""直近の推奨が「どこで外しているか」を軸1/軸2/相手に分解する（2026-08-04 新設）。

ユーザー要望:
  「昨日、一昨日も合わせて分析して。軸1、軸2、相手のどこで外しているケースが
    多いか。基本的に総流しのため、軸のどちらかが外れている」

7S/7A は相手が残り全車（総流し）なので、外れ＝軸2車のいずれかが3着内を外した
ケースに限られる。一方 7B は相手を3点（WT△を除外した pred_prob 上位）に絞るため、
**軸2車がともに3着内でも相手を外して不的中になりうる**。この違いを分けて数える。

外れパターン:
  E   的中
  A   軸1のみ外し（軸2は3着内）
  B   軸2のみ外し（軸1は3着内）
  C   軸2車とも外し
  D   軸2車とも3着内だが相手で外し（7Bのみ発生・総流しなら的中していた）

DB書き込みなし。

使い方:
    python scripts/exp_recent_miss_breakdown.py 2026-08-02 2026-08-04
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MARK = {1: "◎", 2: "◯", 3: "△", 4: "×", 0: "—", None: "—"}


def _q(sql: str) -> list[dict]:
    from sqlalchemy import create_engine, text
    url = os.environ.get("KEIRIN_DB_URL")
    if not url:
        raise RuntimeError("KEIRIN_DB_URL が未設定です")
    eng = create_engine(url)
    with eng.connect() as c:
        rows = [dict(r._mapping) for r in c.execute(text(sql))]
    eng.dispose()
    return rows


def parse_combo(s: str) -> tuple[list[int], list[int]]:
    if not s or "-" not in s:
        return [], []
    a, l = s.split("-", 1)
    return ([int(x) for x in a.replace("=", ",").split(",") if x.strip().isdigit()],
            [int(x) for x in l.split(",") if x.strip().isdigit()])


def main() -> None:
    d_from, d_to = sys.argv[1], sys.argv[2]
    picks = _q(f"""
        SELECT p.race_date, p.race_key, split_part(p.race_key,'#',1) AS base,
               p.rank, p.pred_combo, r.race_no, r.start_at,
               COALESCE(v.name, r.venue_id) AS venue
        FROM keirin.picks_history p
        JOIN keirin.wt_races r ON r.race_key = split_part(p.race_key,'#',1)
        LEFT JOIN keirin.venue_info v ON v.venue_code = r.venue_id
        WHERE p.race_date BETWEEN '{d_from}' AND '{d_to}'
        ORDER BY p.race_date, r.start_at
    """)
    if not picks:
        print("対象なし")
        return
    keys = ",".join(f"'{b}'" for b in sorted({p["base"] for p in picks}))
    ent = defaultdict(dict)
    for e in _q(f"""
        SELECT race_key, frame_no, name, prediction_mark, pred_top3_pct, finish_order
        FROM keirin.wt_entries WHERE race_key IN ({keys})
    """):
        ent[e["race_key"]][int(e["frame_no"])] = e

    pat_cnt: dict[str, int] = defaultdict(int)
    by_rank: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_date: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    miss_axis: list[dict] = []       # 外した軸の明細
    n_conf = 0
    rows_out = []

    for p in picks:
        rows = ent.get(p["base"], {})
        fins = {f: e["finish_order"] for f, e in rows.items()}
        if not any(v is not None and v >= 1 for v in fins.values()):
            continue                  # 未確定
        n_conf += 1
        axes, legs = parse_combo(p["pred_combo"])
        if len(axes) < 2:
            continue
        a1, a2 = axes[0], axes[1]
        top3 = {f for f, v in fins.items() if v is not None and 1 <= v <= 3}
        in1, in2 = a1 in top3, a2 in top3
        third = list(top3 - {a1, a2})
        if in1 and in2:
            pat = "E" if (third and third[0] in legs) else "D"
        elif in1:
            pat = "B"
        elif in2:
            pat = "A"
        else:
            pat = "C"
        pat_cnt[pat] += 1
        by_rank[p["rank"]][pat] += 1
        by_date[p["race_date"]][pat] += 1

        for tag, f in (("軸1", a1), ("軸2", a2)):
            if f not in top3:
                e = rows.get(f, {})
                miss_axis.append({
                    "tag": tag, "fin": fins.get(f), "p3": e.get("pred_top3_pct"),
                    "mark": e.get("prediction_mark"), "rank": p["rank"]})
        rows_out.append({**p, "a1": a1, "a2": a2, "pat": pat, "in1": in1, "in2": in2,
                         "top3": sorted(top3, key=lambda f: fins[f]), "legs": legs,
                         "rows": rows})

    print(f"対象 {d_from}〜{d_to}: 推奨 {len(picks)}件 / 確定 {n_conf}件\n")

    print("【外れパターンの内訳】")
    names = {"E": "的中", "A": "軸1のみ外し", "B": "軸2のみ外し",
             "C": "軸2車とも外し", "D": "軸は的中だが相手で外し"}
    for k in ("E", "A", "B", "C", "D"):
        v = pat_cnt.get(k, 0)
        bar = "█" * round(30 * v / max(n_conf, 1))
        print(f"  {k} {names[k]:22} {v:3d}件 ({100*v/max(n_conf,1):5.1f}%) {bar}")
    miss = n_conf - pat_cnt.get("E", 0)
    if miss:
        print(f"\n  不的中 {miss}件の内訳:")
        for k in ("A", "B", "C", "D"):
            v = pat_cnt.get(k, 0)
            if v:
                print(f"    {names[k]:22} {v:3d}件 ({100*v/miss:5.1f}%)")

    print("\n【軸1 / 軸2 の3着内率】")
    n1 = sum(1 for r in rows_out if r["in1"])
    n2 = sum(1 for r in rows_out if r["in2"])
    nb = sum(1 for r in rows_out if r["in1"] and r["in2"])
    t = len(rows_out)
    print(f"  軸1: {n1}/{t} = {100*n1/max(t,1):.1f}%   （honest全期間 79.3%）")
    print(f"  軸2: {n2}/{t} = {100*n2/max(t,1):.1f}%   （honest全期間 66.9%）")
    print(f"  両方: {nb}/{t} = {100*nb/max(t,1):.1f}%  （honest全期間 52.5%）")

    print("\n【外した軸の着順分布】（4着=惜敗 / 7着=大敗）")
    for tag in ("軸1", "軸2"):
        sub = [m for m in miss_axis if m["tag"] == tag]
        if not sub:
            continue
        dist = defaultdict(int)
        for m in sub:
            dist[m["fin"] if m["fin"] else 0] += 1
        s = "  ".join(f"{k if k else 'DNF'}着 {v}" for k, v in sorted(dist.items()))
        near = sum(v for k, v in dist.items() if k == 4)
        print(f"  {tag}（{len(sub)}件）: {s}")
        print(f"       うち4着（惜敗）= {near}件 ({100*near/len(sub):.0f}%)")

    print("\n【外した軸の WT印】")
    for tag in ("軸1", "軸2"):
        sub = [m for m in miss_axis if m["tag"] == tag]
        if not sub:
            continue
        d = defaultdict(int)
        for m in sub:
            d[MARK.get(m["mark"], "—")] += 1
        print(f"  {tag}: " + "  ".join(f"{k} {v}件" for k, v in
                                        sorted(d.items(), key=lambda kv: -kv[1])))

    print("\n【日別】")
    for d in sorted(by_date):
        c = by_date[d]
        tot = sum(c.values())
        print(f"  {d}: 確定{tot:2d}件  的中{c.get('E',0)}  "
              f"軸1外し{c.get('A',0)}  軸2外し{c.get('B',0)}  "
              f"両方外し{c.get('C',0)}  相手外し{c.get('D',0)}")

    print("\n【ランク別】")
    for r in sorted(by_rank):
        c = by_rank[r]
        tot = sum(c.values())
        print(f"  {r}: 確定{tot:2d}件  的中{c.get('E',0)}  "
              f"軸1外し{c.get('A',0)}  軸2外し{c.get('B',0)}  "
              f"両方外し{c.get('C',0)}  相手外し{c.get('D',0)}")

    print("\n【明細】")
    for r in rows_out:
        rows = r["rows"]
        def _d(f: int) -> str:
            e = rows.get(f, {})
            fin = e.get("finish_order")
            return (f"{f}({MARK.get(e.get('prediction_mark'),'—')}"
                    f"/{e.get('pred_top3_pct')}/"
                    f"{fin if fin else 'DNF'}着)")
        print(f"  {r['race_date']} {r['venue']}{r['race_no']}R "
              f"{r['rank'].replace('RANK_','')} [{r['pat']}] "
              f"軸1={_d(r['a1'])} 軸2={_d(r['a2'])} → 着順 "
              f"{'-'.join(str(f) for f in r['top3'])}")


if __name__ == "__main__":
    main()
