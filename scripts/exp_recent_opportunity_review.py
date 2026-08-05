"""直近の推奨を「見逃し・難易度・軸の代替」の3観点で検証する（2026-08-04 新設）。

ユーザー要望:
  「8/1以降のレースについて結果の分析を行い
    ① 購入・的中可能だったレースを見逃していないか
    ② 的中難易度が高いレースを推奨していないか
    ③ 推奨レースで軸とし外してしまったものは別の観点で的中することはできなかったか」

データ源は本番が実際に書き込んだ `wt_entries.pred_win_pct / pred_top3_pct`
（＝当日その時点の本番モデルの出力）。honest なバックテストではなく
**実際に出した予想の事後レビュー**である点に注意。

⚠️ 期間が4日・推奨50件と小標本なので、統計的な結論は出せない。
   個別レースの傾向を掴み、次に検証すべき仮説を出すための分析。

DB書き込みなし。

使い方:
    python scripts/exp_recent_opportunity_review.py [--from 2026-08-01] [--to 2026-08-04]
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import (
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
    rank_7s_field_entropy, rank_7s_select_axis, rank_7s_wt_overlap_n,
)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="2026-08-01")
    ap.add_argument("--to", dest="d_to", default="2026-08-04")
    args = ap.parse_args()

    ent = defaultdict(dict)
    for e in _q(f"""
        SELECT e.race_key, e.frame_no, e.name, e.prediction_mark,
               e.pred_win_pct, e.pred_top3_pct, e.finish_order, e.line_pos, e.line_size
        FROM keirin.wt_entries e JOIN keirin.wt_races r ON r.race_key = e.race_key
        WHERE r.race_date BETWEEN '{args.d_from}' AND '{args.d_to}'
          AND r.n_entries = 7
    """):
        ent[e["race_key"]][int(e["frame_no"])] = e

    races = {r["race_key"]: r for r in _q(f"""
        SELECT r.race_key, r.race_date, r.race_no, r.race_type,
               COALESCE(v.name, r.venue_id) AS venue
        FROM keirin.wt_races r LEFT JOIN keirin.venue_info v ON v.venue_code = r.venue_id
        WHERE r.race_date BETWEEN '{args.d_from}' AND '{args.d_to}' AND r.n_entries = 7
    """)}

    picked = {}
    for p in _q(f"""
        SELECT split_part(race_key,'#',1) AS base, rank, pred_combo
        FROM keirin.picks_history
        WHERE race_date BETWEEN '{args.d_from}' AND '{args.d_to}'
    """):
        picked[p["base"]] = p

    trio = defaultdict(dict)
    keys = ",".join(f"'{k}'" for k in races)
    if keys:
        for o in _q(f"SELECT race_key, combination, odds_value FROM keirin.wt_odds "
                    f"WHERE bet_type='trio' AND race_key IN ({keys})"):
            try:
                v = float(o["odds_value"])
            except (TypeError, ValueError):
                continue
            if 0 < v < 9999:
                trio[o["race_key"]][frozenset(
                    int(x) for x in re.split(r"[-=→]", str(o["combination"])))] = v

    rows = []
    for rk, meta in races.items():
        rows_e = ent.get(rk, {})
        if len(rows_e) != 7:
            continue
        top3 = {f for f, e in rows_e.items()
                if e["finish_order"] and 1 <= int(e["finish_order"]) <= 3}
        if len(top3) != 3:
            continue                       # 未確定
        pw = {f: float(e["pred_win_pct"] or 0) for f, e in rows_e.items()}
        p3 = {f: float(e["pred_top3_pct"] or 0) / 100.0 for f, e in rows_e.items()}
        sel = rank_7s_select_axis(pw, p3)
        if not sel:
            continue
        a1, a2, asum = sel
        hon = next((f for f, e in rows_e.items() if e["prediction_mark"] == 1), None)
        tai = next((f for f, e in rows_e.items() if e["prediction_mark"] == 2), None)
        ov = rank_7s_wt_overlap_n(a1, a2, hon, tai)
        ent_v = rank_7s_field_entropy(p3)
        n_fail = (asum > RANK_7S_AXIS_SUM_MAX) + (ent_v > RANK_7S_ENTROPY_MAX)
        both = {a1, a2} <= top3
        third = list(top3 - {a1, a2})
        pay = trio[rk].get(frozenset(top3))
        rows.append({
            "rk": rk, **meta, "a1": a1, "a2": a2, "asum": asum, "ent": ent_v,
            "ov": ov, "n_fail": n_fail, "top3": top3, "both": both,
            "in1": a1 in top3, "in2": a2 in top3,
            "third": third[0] if len(third) == 1 else None,
            "pay": pay, "picked": rk in picked,
            "rank": picked.get(rk, {}).get("rank", ""),
            "e": rows_e,
        })

    n = len(rows)
    pk = [r for r in rows if r["picked"]]
    nk = [r for r in rows if not r["picked"]]
    print(f"対象 {args.d_from}〜{args.d_to}: 7車立て・結果確定 {n}レース "
          f"（推奨 {len(pk)} / 非推奨 {len(nk)}）\n")

    # ------------------------------------------------------------ ①見逃し
    print("【① 見逃し: 推奨しなかったが軸2車がともに3着内だったレース】")
    miss = [r for r in nk if r["both"]]
    print(f"  非推奨 {len(nk)}件中 {len(miss)}件（{100*len(miss)/max(len(nk),1):.1f}%）で"
          f"軸2車が3着内\n  ※ 推奨レースの実績は {sum(1 for r in pk if r['both'])}/{len(pk)}"
          f"（{100*sum(1 for r in pk if r['both'])/max(len(pk),1):.1f}%）")
    why = defaultdict(list)
    for r in miss:
        if r["ov"] == 2:
            why["◎◯完全一致(overlap2)で対象外"].append(r)
        elif r["ov"] is None:
            why["WT印欠損"].append(r)
        elif r["n_fail"] > 1:
            why["axis_sum/entropy 両方不合格"].append(r)
        else:
            why["その他"].append(r)
    for k, v in sorted(why.items(), key=lambda kv: -len(kv[1])):
        pays = [x["pay"] for x in v if x["pay"]]
        med = statistics.median(pays) if pays else 0
        print(f"    {k:32} {len(v):3d}件  三連複配当中央値 {med:6.1f}倍")
    print()

    # ------------------------------------------------------------ ②難易度
    print("【② 難易度: 推奨レースは当てにくいレースに偏っていないか】")
    print(f"  {'区分':22} {'n':>4} {'両方3着内':>9} {'entropy':>9} {'axis_sum':>9} "
          f"{'配当中央値':>10}")

    def _stat(label, lst):
        if not lst:
            return
        pays = [x["pay"] for x in lst if x["pay"]]
        print(f"  {label:22} {len(lst):4d} "
              f"{100*sum(1 for x in lst if x['both'])/len(lst):8.1f}% "
              f"{statistics.mean(x['ent'] for x in lst):9.4f} "
              f"{statistics.mean(x['asum'] for x in lst):9.3f} "
              f"{statistics.median(pays) if pays else 0:9.1f}倍")

    _stat("推奨した", pk)
    _stat("推奨しなかった", nk)
    _stat("  └ うち overlap2", [r for r in nk if r["ov"] == 2])
    _stat("  └ うち 2ゲート不合格", [r for r in nk if r["ov"] in (0, 1) and r["n_fail"] > 1])
    print()

    # ------------------------------------------------------------ ③代替観点
    print("【③ 軸を外したレース: 別の観点なら当たったか】")
    lost = [r for r in pk if not r["both"]]
    print(f"  推奨 {len(pk)}件中 {len(lost)}件が軸で外し")
    alts = defaultdict(int)
    for r in lost:
        e = r["e"]
        cand = {
            "pred_win 上位2": sorted(e, key=lambda f: -float(e[f]["pred_win_pct"] or 0))[:2],
            "pred_top3 上位2": sorted(e, key=lambda f: -float(e[f]["pred_top3_pct"] or 0))[:2],
            "WT ◎◯": [f for f in e if e[f]["prediction_mark"] in (1, 2)],
            "WT ◎△": [f for f in e if e[f]["prediction_mark"] in (1, 3)],
            "ライン先頭2車": sorted(
                [f for f in e if (e[f]["line_pos"] or 9) == 1],
                key=lambda f: -float(e[f]["pred_top3_pct"] or 0))[:2],
        }
        for k, v in cand.items():
            if len(v) == 2 and set(v) <= r["top3"]:
                alts[k] += 1
    print(f"  {'代替の軸選び':22} {'当たっていた数':>12}")
    for k, v in sorted(alts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22} {v:8d} / {len(lost)} ({100*v/max(len(lost),1):.0f}%)")
    print()

    print("  ── 外した推奨の明細 ──")
    for r in lost:
        e = r["e"]
        def _d(f):
            x = e[f]
            return (f"{f}({MARK.get(x['prediction_mark'],'—')}/"
                    f"{x['pred_top3_pct']}/{x['finish_order'] or 'DNF'}着)")
        order = sorted(r["top3"], key=lambda f: int(e[f]["finish_order"]))
        print(f"    {r['race_date']} {r['venue']}{r['race_no']}R "
              f"{r['rank'].replace('RANK_','')} 軸1={_d(r['a1'])} 軸2={_d(r['a2'])} "
              f"→ {'-'.join(str(f) for f in order)} "
              f"({r['pay'] or '—'}倍)")


if __name__ == "__main__":
    main()
