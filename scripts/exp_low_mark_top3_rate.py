"""WT印が△以下の車の3着内率と、軸2に選ばれたときの挙動を測る（2026-08-04）。

ユーザー要望:
  「軸2で惜敗は許容できるが大敗が多すぎる。ここの精度向上が必要。
    △以下ではあるが、該当レースで△以下が3着以内に入っている確率も確認して」

背景（scripts/exp_recent_miss_breakdown.py・8/02〜8/04 確定24件）:
  不的中22件の**50%が「軸2のみ外し」**。軸1は外しても4着（惜敗）が62%なのに対し、
  軸2は**6着以下の大敗が67%**。外した軸2のWT印は △7件・×2件で6割が市場低評価。

7S/7A は「軸2車がWT◎◯と完全一致しない」レースを買う設計上、**必ず軸のどちらかが
△以下**になる。その△以下がどれだけ走るのか（＝モデルが軸に据える妥当性があるのか）を
honest データで測る。

測定内容:
  ① WT印別の着順分布（3着内率・1着率・大敗率）
  ② 3着内3枠の印構成（誰が実際に3着内を占めているか）
  ③ 軸2に選ばれた△以下 vs 同一レースで選ばれなかった△以下
     → モデルに「△以下の中から走る車を選ぶ」能力があるか
  ④ 軸2の印別の較正（予測 p2 と実測3着内率の乖離）

honest: 月次凍結vintageモデルのキャッシュ（scripts/exp_7c_cache.py）+ DBの印。
DB書き込みなし。

使い方:
    python scripts/exp_low_mark_top3_rate.py data/exp_7c_cache
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MARK_NAME = {1: "◎ 本命", 2: "◯ 対抗", 3: "△ 単穴", 4: "× 連下", 0: "無印"}
LOW_MARKS = (3, 4, 0)   # △以下


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


def load_cache(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def main() -> None:
    cands = load_cache(Path(sys.argv[1]))
    days = sorted({c["race_date"] for c in cands})
    keys = {c["race_key"] for c in cands}
    print(f"母集団: 7車立て・軸選定成功 {len(cands)}件 / {len(days)}日 "
          f"({days[0]}〜{days[-1]})")
    print("DB から印・着順を取得中 ...", flush=True)

    ent: dict[str, dict[int, dict]] = defaultdict(dict)
    for e in _q(f"""
        SELECT e.race_key, e.frame_no, e.prediction_mark, e.finish_order
        FROM keirin.wt_entries e
        JOIN keirin.wt_races r ON r.race_key = e.race_key
        WHERE r.race_date BETWEEN '{days[0]}' AND '{days[-1]}'
          AND r.n_entries = 7
    """):
        ent[e["race_key"]][int(e["frame_no"])] = e
    print("完了\n")

    # ---------------------------------------------------------------- ①印別
    stat: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in cands:
        rows = ent.get(c["race_key"])
        if not rows:
            continue
        for f, e in rows.items():
            fo = e["finish_order"]
            mk = e["prediction_mark"] if e["prediction_mark"] is not None else 0
            s = stat[mk]
            s["n"] += 1
            if fo is None or fo < 1:
                s["dnf"] += 1
                continue
            fo = int(fo)
            if fo == 1:
                s["win"] += 1
            if fo <= 3:
                s["top3"] += 1
            if fo == 4:
                s["fourth"] += 1
            if fo >= 6:
                s["bad"] += 1

    print("【① WT印別の着順分布（出走全車）】")
    print(f"  {'印':10} {'n':>7} {'3着内':>8} {'1着':>7} {'4着':>7} "
          f"{'6着以下':>8} {'欠車失格':>8}")
    for mk in (1, 2, 3, 4, 0):
        s = stat.get(mk)
        if not s or not s["n"]:
            continue
        n = s["n"]
        print(f"  {MARK_NAME[mk]:10} {n:7d} {100*s['top3']/n:7.1f}% "
              f"{100*s['win']/n:6.1f}% {100*s['fourth']/n:6.1f}% "
              f"{100*s['bad']/n:7.1f}% {100*s['dnf']/n:7.1f}%")
    low_n = sum(stat[m]["n"] for m in LOW_MARKS if m in stat)
    low_t3 = sum(stat[m]["top3"] for m in LOW_MARKS if m in stat)
    print(f"  {'△以下 計':10} {low_n:7d} {100*low_t3/max(low_n,1):7.1f}%")
    print()

    # ---------------------------------------------------------------- ②3着内の構成
    comp: dict[int, int] = defaultdict(int)
    tot3 = 0
    for c in cands:
        rows = ent.get(c["race_key"])
        if not rows:
            continue
        for f, e in rows.items():
            fo = e["finish_order"]
            if fo is not None and 1 <= int(fo) <= 3:
                comp[e["prediction_mark"] or 0] += 1
                tot3 += 1
    print("【② 3着内3枠を誰が占めているか】")
    for mk in (1, 2, 3, 4, 0):
        if comp.get(mk):
            print(f"  {MARK_NAME[mk]:10} {comp[mk]:7d} 枠 ({100*comp[mk]/tot3:5.1f}%)")
    low_c = sum(comp[m] for m in LOW_MARKS)
    print(f"  {'△以下 計':10} {low_c:7d} 枠 ({100*low_c/tot3:5.1f}%)")
    print(f"  → 3着内の {100*low_c/tot3:.1f}% は△以下が占める"
          f"（1レース3枠のうち約{3*low_c/tot3:.2f}枠）")
    print()

    # ---------------------------------------------------------------- ③選別能力
    print("【③ 軸2に選ばれた△以下 vs 同レースで選ばれなかった△以下】")
    print("  ＝モデルに「△以下の中から走る車を選ぶ」能力があるか")
    sel = defaultdict(int)
    unsel = defaultdict(int)
    for c in cands:
        rows = ent.get(c["race_key"])
        if not rows:
            continue
        a2 = c["axis2"]
        e2 = rows.get(a2)
        if not e2:
            continue
        mk2 = e2["prediction_mark"] or 0
        if mk2 not in LOW_MARKS:
            continue                       # 軸2が△以下のレースだけを対象
        for f, e in rows.items():
            mk = e["prediction_mark"] or 0
            if mk not in LOW_MARKS:
                continue
            fo = e["finish_order"]
            d = sel if f == a2 else unsel
            d["n"] += 1
            if fo is None or int(fo) < 1:
                d["dnf"] += 1
                continue
            fo = int(fo)
            if fo <= 3:
                d["top3"] += 1
            if fo >= 6:
                d["bad"] += 1
    for label, d in (("軸2に選ばれた△以下", sel), ("選ばれなかった△以下", unsel)):
        n = max(d["n"], 1)
        print(f"  {label:22} n={d['n']:6d}  3着内 {100*d['top3']/n:5.1f}%  "
              f"6着以下 {100*d['bad']/n:5.1f}%  欠車失格 {100*d['dnf']/n:4.1f}%")
    if sel["n"] and unsel["n"]:
        lift = 100*sel["top3"]/sel["n"] - 100*unsel["top3"]/unsel["n"]
        print(f"  → 選別リフト {lift:+.1f}pt"
              f"（正なら「△以下の中から走る車を選べている」）")
    print()

    # ---------------------------------------------------------------- ④較正
    print("【④ 軸2の印別 較正（モデル予測 p2 と実測3着内率）】")
    print(f"  {'軸2の印':10} {'n':>7} {'予測p2':>8} {'実測':>8} {'乖離':>8} "
          f"{'6着以下':>8}")
    cal: dict[int, list] = defaultdict(list)
    for c in cands:
        rows = ent.get(c["race_key"])
        if not rows:
            continue
        a2 = c["axis2"]
        e2 = rows.get(a2)
        if not e2:
            continue
        mk = e2["prediction_mark"] or 0
        fo = e2["finish_order"]
        in3 = fo is not None and 1 <= int(fo) <= 3
        bad = fo is not None and int(fo) >= 6
        cal[mk].append((c["top3_probs"].get(a2, 0.0), in3, bad))
    for mk in (1, 2, 3, 4, 0):
        v = cal.get(mk)
        if not v or len(v) < 100:
            continue
        n = len(v)
        pm = sum(p for p, _, _ in v) / n
        am = sum(1 for _, h, _ in v if h) / n
        bm = sum(1 for _, _, b in v if b) / n
        print(f"  {MARK_NAME[mk]:10} {n:7d} {100*pm:7.1f}% {100*am:7.1f}% "
              f"{100*(am-pm):+7.1f}pt {100*bm:7.1f}%")


if __name__ == "__main__":
    main()
