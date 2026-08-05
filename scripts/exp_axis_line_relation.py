"""軸1と軸2のライン関係が「両方3着内」に与える影響の検証（2026-08-04）。

競輪はライン戦なのに、軸2を選ぶとき**軸1と同ラインか別ラインか**を一切見ていない。
現行（および3ヘッド版）は個々の車の確率だけで軸2を決めており、
**2車が同時に3着内へ入る確率（共分散）を無視**している。

  - 同ライン（先頭＋番手）: 連れ込みで両方来やすい／ライン全体が沈むと共倒れ
  - 別ライン: 互いに独立に近い

「両方3着内」を最大化したいなら個々の確率の積ではなく同時確率を見るべきで、
これは 2026-08-04 に追加した line_rp_gap_top（ラインの"強さ"）とは別の観点。

本スクリプトはまず**記述統計**で効果の有無を確かめる:
  ① 同ライン / 別ライン別の両方3着内率（素の比較）
  ② **予測確率を揃えた上での比較**（p1×p2 の帯で層別）
     ← これが本題。素の比較は「同ラインの方が強い車が集まる」等の交絡を含む
  ③ ライン内位置の組み合わせ別（先頭+番手 / 番手+3番手 など）
  ④ 軸1のライン規模別

⚠️ キャッシュは48特徴時代の vintage モデル生成。傾向を見る目的では十分。

DB書き込みなし。

使い方:
    python scripts/exp_axis_line_relation.py data/exp_7c_cache
"""
from __future__ import annotations

import os
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX


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


def load(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def rate(sub: list[dict]) -> tuple[int, float, float]:
    """(n, 両方3着内率, 三連複配当中央値)"""
    if not sub:
        return 0, 0.0, 0.0
    both = sum(1 for c in sub if {c["axis1"], c["axis2"]} <= set(c["actual_top3"]))
    pays = [c["trio_pay"] for c in sub
            if {c["axis1"], c["axis2"]} <= set(c["actual_top3"]) and c["trio_pay"]]
    return len(sub), 100.0 * both / len(sub), (statistics.median(pays) / 100 if pays else 0.0)


def main() -> None:
    cands = load(Path(sys.argv[1]))
    days = sorted({c["race_date"] for c in cands})
    print(f"読み込み {len(cands)}件 / {len(days)}日 ({days[0]}〜{days[-1]})")
    print("DB からライン情報を取得中 ...", flush=True)

    line = defaultdict(dict)
    for e in _q(f"""
        SELECT e.race_key, e.frame_no, e.line_group, e.line_pos, e.line_size
        FROM keirin.wt_entries e JOIN keirin.wt_races r ON r.race_key = e.race_key
        WHERE r.race_date BETWEEN '{days[0]}' AND '{days[-1]}' AND r.n_entries = 7
    """):
        line[e["race_key"]][int(e["frame_no"])] = e
    print("完了\n")

    # 現行 7S+7A 相当に絞る
    pool = []
    for c in cands:
        if c["wt_overlap_n"] not in (0, 1):
            continue
        if ((c["axis_sum"] > RANK_7S_AXIS_SUM_MAX)
                + (c["entropy"] > RANK_7S_ENTROPY_MAX)) > 1:
            continue
        L = line.get(c["race_key"])
        if not L:
            continue
        l1, l2 = L.get(c["axis1"]), L.get(c["axis2"])
        if not l1 or not l2:
            continue
        g1, g2 = l1["line_group"], l2["line_group"]
        # line_group が欠損 = 単騎扱い（車番で一意化して別ライン扱いにする）
        k1 = g1 if g1 is not None else -c["axis1"]
        k2 = g2 if g2 is not None else -c["axis2"]
        c["_same_line"] = (k1 == k2)
        c["_pos1"] = l1["line_pos"] or 1
        c["_pos2"] = l2["line_pos"] or 1
        c["_size1"] = l1["line_size"] or 1
        c["_size2"] = l2["line_size"] or 1
        c["_p1"] = c["top3_probs"][c["axis1"]]
        c["_p2"] = c["top3_probs"][c["axis2"]]
        pool.append(c)

    print(f"評価対象（現行7S+7A相当・ライン情報あり）: {len(pool)}件\n")

    # ------------------------------------------------------------ ①素の比較
    print("【① 軸1と軸2のライン関係（素の比較）】")
    print(f"  {'区分':16} {'n':>6} {'割合':>7} {'両方3着内':>10} {'配当中央値':>10}")
    for lbl, sub in (("同ライン", [c for c in pool if c["_same_line"]]),
                     ("別ライン", [c for c in pool if not c["_same_line"]])):
        n, r, med = rate(sub)
        print(f"  {lbl:16} {n:6d} {100*n/len(pool):6.1f}% {r:9.1f}% {med:9.1f}倍")
    print()

    # ------------------------------------------------------------ ②確率を揃える
    print("【② 予測確率を揃えた比較（p1×p2 の帯で層別）】")
    print("  ※ 素の比較は「同ラインには強い車が集まる」等の交絡を含むため、")
    print("     同じ予測確率どうしで比べる。ここで差が残れば**モデルが未表現の情報**。")
    print(f"  {'p1×p2 帯':14} {'同ライン n':>10} {'両方3着内':>10} "
          f"{'別ライン n':>10} {'両方3着内':>10} {'差':>8}")
    edges = [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]
    tot_d, tot_w = 0.0, 0
    for lo, hi in edges:
        s = [c for c in pool if c["_same_line"] and lo <= c["_p1"] * c["_p2"] < hi]
        d = [c for c in pool if not c["_same_line"] and lo <= c["_p1"] * c["_p2"] < hi]
        if len(s) < 50 or len(d) < 50:
            continue
        ns, rs, _ = rate(s)
        nd, rd, _ = rate(d)
        tot_d += (rs - rd) * (ns + nd)
        tot_w += ns + nd
        print(f"  {lo:.2f}〜{hi:.2f}    {ns:10d} {rs:9.1f}% {nd:10d} {rd:9.1f}% "
              f"{rs - rd:+7.1f}pt")
    if tot_w:
        print(f"  → 確率を揃えた上での加重平均差: {tot_d / tot_w:+.2f}pt")
    print()

    # ------------------------------------------------------------ ③位置の組合せ
    print("【③ ライン内位置の組み合わせ別（同ラインのみ）】")
    print(f"  {'組み合わせ':20} {'n':>6} {'両方3着内':>10} {'配当中央値':>10}")
    combo = defaultdict(list)
    for c in pool:
        if c["_same_line"]:
            a, b = sorted([int(c["_pos1"]), int(c["_pos2"])])
            combo[f"{a}番手 + {b}番手"].append(c)
    for k, v in sorted(combo.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 30:
            continue
        n, r, med = rate(v)
        print(f"  {k:20} {n:6d} {r:9.1f}% {med:9.1f}倍")
    print()

    # ------------------------------------------------------------ ④ライン規模
    print("【④ 軸1のライン規模別】")
    print(f"  {'区分':24} {'n':>6} {'両方3着内':>10} {'配当中央値':>10}")
    for sz in (1, 2, 3, 4):
        for same in (True, False):
            sub = [c for c in pool if int(c["_size1"]) == sz and c["_same_line"] == same]
            if len(sub) < 50:
                continue
            n, r, med = rate(sub)
            lbl = f"軸1が{sz}車ライン・{'同ライン' if same else '別ライン'}"
            print(f"  {lbl:24} {n:6d} {r:9.1f}% {med:9.1f}倍")


if __name__ == "__main__":
    main()
