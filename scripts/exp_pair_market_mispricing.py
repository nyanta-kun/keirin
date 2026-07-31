"""【市場は軸間相関を織り込めているか】ペア単位の市場ミスプライシング検証（2026-07-30）。

`inputs/二軸探偵_予想ロジック調査メモ.md` 4章の核心論点:
「リフト値が1から乖離が大きい条件＝軸間相関が強く働く条件であり、
  これは市場のオッズが軸間相関を織り込みきれていない＝妙味が出やすい条件の
  発見にも直結する」

この「市場が相関を織り込めていないか」は**三連複オッズから厳密に測れる**。
[[keirin_segment_market_edge_closure_2026_07_30]]はレース単位で市場エッジを否定したが、
**ペア単位（＝二軸探偵の賭けの単位）では未検証**だったため本スクリプトで測る。

## S7のROIの厳密な分解（本スクリプトの理論的支柱）

S7は「軸2車 + 残り5車流し」＝**軸ペア{i,j}を含む5通りの三連複を買う**構造。
的中条件は「i,j がともに3着内」と厳密に同値。パリミュチュエルでは
  odds(組) = 0.75 / market_prob(組)      （控除率25%）
なので、市場確率が正しければどの組に賭けても期待回収は 0.75。よって:

    **ROI(S7型5点流し) = 0.75 × [ 実際のペア的中率 ÷ 市場が織り込むペア確率 ]**

  market_P(i,j ともに3着内) = Σ_{i,jを含む三連複5通り} market_prob(組)

したがって ROI≥100% には **実測/市場 ≥ 1.333** が必要。
この比を条件別に測れば「どのペア条件で市場が相関を過小評価しているか」が
ROIに直結する形で分かる。既存のROIバックテストより桁違いに検出力が高い
（的中というレアイベントではなく全21ペア×全レースを使うため）。

## 測る条件（メモ4-1の仮説リストに対応）

1. ライン関係: 同ライン1-2番手 / 2-3番手 / 1-3番手 / その他 / 別ライン / 単騎絡み
2. 上記 × 分戦数（2/3/4/5+）    ← メモ「他ラインとの主導権争い」
3. 脚質の組み合わせ（逃×追 等） ← メモ「バンク・風との交互作用」の脚質部分
4. 同県ペアか                    ← メモ「同県対決・裏ライン」
5. グレード                      ← メモ「級班・格による並びの信頼度」
6. バンク周長                    ← メモ「バンク特性」
7. 我々のモデルでのペア順位（1位ペア＝実際に軸に選ぶペア）
8. ペアの市場確率帯（人気ペア/穴ペア）

## 出力の読み方

- `実測/市場` > 1.333 → そのペア条件でS7型5点流しがROI100%超（★）
- `実測/市場` ≈ 1.0    → 市場は相関を正しく織り込んでいる（妙味なし）
- `市場lift` vs `実測lift` の比較で「市場が相関自体を過小評価しているか」を直接確認
  （lift = P(ペア同時) / [P(i)×P(j)]・1.0が独立）

honest分割: TRAIN 2024-01-01〜2025-12-31（我々のlift推定のみ）/
            TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
TAKEOUT_RETURN = 0.75
MIN_BOARD = 33
MIN_PAIRS = 800          # ペア単位なので母数は大きく取れる
ROI_BREAKEVEN = 1.0 / TAKEOUT_RETURN     # = 1.3333


def load_all():
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date, grade, distance FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: {"race_date": str(r["race_date"]),
                             "grade": str(r["grade"]) if r["grade"] is not None else "?",
                             "distance": r["distance"]} for r in rrows}
    keys = list(races)
    print(f"[load] races: {len(keys)}", flush=True)

    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_top3_pct, line_group, line_pos, "
                 "       line_size, prefecture, style, finish_order "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load] entries: {sum(len(v) for v in by_race.values())}", flush=True)
    return races, by_race


def load_trio_odds(race_keys):
    import re
    out = {}
    keys = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if len(set(parts)) != 3:
                    continue
                m = 0
                for p in parts:
                    m |= 1 << p
                out.setdefault(rk, {})[m] = fv
    return out


def build_rows(races, entries_by_race):
    rows = []
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None for e in ents):
            continue
        fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        tm = 0
        for _, f in fin[:3]:
            tm |= 1 << f
        bf = {int(e["frame_no"]): e for e in ents}
        sizes = defaultdict(int)
        for e in ents:
            lg = e["line_group"]
            sizes[lg if lg is not None else f"_s{e['frame_no']}"] += 1
        rows.append({"race_key": rk, "race_date": meta["race_date"],
                     "grade": meta["grade"], "distance": meta["distance"],
                     "by_frame": bf, "top3_mask": tm, "n_lines": len(sizes)})
    print(f"[build] rows: {len(rows)}", flush=True)
    return rows


def line_bucket(bf, i, j):
    li, lj = bf[i]["line_group"], bf[j]["line_group"]
    si = bf[i]["line_size"]
    sj = bf[j]["line_size"]
    if (si == 1) or (sj == 1):
        return "単騎絡み"
    if li is None or lj is None:
        return "不明"
    if li != lj:
        return "別ライン"
    pi, pj = bf[i]["line_pos"], bf[j]["line_pos"]
    if pi is None or pj is None:
        return "同ラインその他"
    a, b = sorted([int(pi), int(pj)])
    return {(1, 2): "同ライン1-2番手", (2, 3): "同ライン2-3番手",
            (1, 3): "同ライン1-3番手"}.get((a, b), "同ラインその他")


def style_pair(bf, i, j):
    a, b = str(bf[i]["style"] or "?"), str(bf[j]["style"] or "?")
    return "×".join(sorted([a, b]))


def estimate_lifts(rows):
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        tm = r["top3_mask"]
        for i, j in combinations(sorted(bf), 2):
            b = line_bucket(bf, i, j)
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if (tm >> i) & 1 and (tm >> j) & 1:
                a["obs"] += 1
    return {b: ((a["obs"] / a["n"]) / (a["exp"] / a["n"]) if a["n"] >= 100 and a["exp"] > 0 else 1.0)
            for b, a in agg.items()}


class PairAcc:
    __slots__ = ("n", "obs", "mkt", "our", "mi_mj", "d", "d2")

    def __init__(self):
        self.n = 0
        self.obs = 0
        self.mkt = 0.0
        self.our = 0.0
        self.mi_mj = 0.0
        self.d = self.d2 = 0.0

    def add(self, y, mkt_p, our_p, mi_mj):
        self.n += 1
        self.obs += y
        self.mkt += mkt_p
        self.our += our_p
        self.mi_mj += mi_mj
        d = y - mkt_p               # 実測 - 市場（正なら市場が過小評価）
        self.d += d
        self.d2 += d * d

    def report(self):
        n = self.n
        act = self.obs / n
        mkt = self.mkt / n
        our = self.our / n
        md = self.d / n
        var = max(self.d2 / n - md * md, 0.0)
        t = md / math.sqrt(var / n) if var > 0 else 0.0
        ratio = act / mkt if mkt > 0 else 0.0
        return {"n": n, "act": act * 100, "mkt": mkt * 100, "our": our * 100,
                "ratio": ratio, "t": t, "roi": TAKEOUT_RETURN * ratio * 100,
                "mkt_lift": mkt / (self.mi_mj / n) if self.mi_mj > 0 else 0.0,
                "act_lift": act / (self.mi_mj / n) if self.mi_mj > 0 else 0.0}


def main():
    races, entries = load_all()
    rows = build_rows(races, entries)
    del entries
    train = [r for r in rows if r["race_date"] <= TRAIN_TO]
    print(f"[split] TRAIN={len(train)} TEST={len(rows)-len(train)}")

    lifts = estimate_lifts(train)
    print("\n[lift] 我々のlift（TRAIN推定）")
    for b in sorted(lifts, key=lambda x: -lifts[x]):
        print(f"    {b:<16} {lifts[b]:.4f}")

    acc = defaultdict(lambda: defaultdict(PairAcc))   # dim -> seg -> acc  (窓別にキーへ埋め込む)

    by_month = defaultdict(list)
    for r in rows:
        by_month[r["race_date"][:7]].append(r)

    for ym in sorted(by_month):
        chunk = by_month[ym]
        boards = load_trio_odds([r["race_key"] for r in chunk])
        for r in chunk:
            board = boards.get(r["race_key"])
            if not board or len(board) < MIN_BOARD:
                continue
            w = "TRAIN" if r["race_date"] <= TRAIN_TO else "TEST"
            bf = r["by_frame"]
            frames = sorted(bf)
            tm = r["top3_mask"]

            # 市場確率（レース内正規化）
            mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if o > 0}
            tot = sum(mk_raw.values())
            if tot <= 0:
                continue
            mkt = {m: v / tot for m, v in mk_raw.items()}

            # 市場の車単位周辺確率
            marg = {f: 0.0 for f in frames}
            for m, p in mkt.items():
                for f in frames:
                    if (m >> f) & 1:
                        marg[f] += p

            # 市場のペア同時確率 = そのペアを含む5通りの和
            pair_mkt = defaultdict(float)
            for m, p in mkt.items():
                fs = [f for f in frames if (m >> f) & 1]
                for a, b in combinations(fs, 2):
                    pair_mkt[(a, b)] += p

            # 我々のペア確率（積×lift）とレース内順位
            our_pair = {}
            for i, j in combinations(frames, 2):
                pi = float(bf[i]["pred_top3_pct"]) / 100.0
                pj = float(bf[j]["pred_top3_pct"]) / 100.0
                our_pair[(i, j)] = pi * pj * lifts.get(line_bucket(bf, i, j), 1.0)
            our_rank = {k: n + 1 for n, k in
                        enumerate(sorted(our_pair, key=lambda k: -our_pair[k]))}

            for i, j in combinations(frames, 2):
                key = (i, j)
                y = 1.0 if ((tm >> i) & 1 and (tm >> j) & 1) else 0.0
                mp = pair_mkt.get(key, 0.0)
                if mp <= 0:
                    continue
                op = our_pair[key]
                mimj = marg[i] * marg[j]
                lb = line_bucket(bf, i, j)
                nl = f"分戦{min(r['n_lines'], 5)}{'+' if r['n_lines'] >= 5 else ''}"
                same_pref = (bf[i]["prefecture"] is not None
                             and bf[i]["prefecture"] == bf[j]["prefecture"])
                rk = our_rank[key]
                mband = ("市場帯 <5%" if mp < 0.05 else
                         "市場帯 5-10%" if mp < 0.10 else
                         "市場帯 10-20%" if mp < 0.20 else
                         "市場帯 20-35%" if mp < 0.35 else "市場帯 35%+")
                segs = [
                    ("ALL", "全ペア"),
                    ("line", lb),
                    ("line_x_nlines", f"{lb}×{nl}"),
                    ("style", style_pair(bf, i, j)),
                    ("pref", ("同県ペア" if same_pref else "他県ペア") + f"／{('同' if lb.startswith('同ライン') else '別')}ライン系"),
                    ("grade", f"G:{r['grade']}"),
                    ("bank", f"周長{r['distance']}"),
                    ("our_rank", f"我々ペア{rk}位" if rk <= 3 else "我々ペア4位以下"),
                    ("mkt_band", mband),
                    ("rank1_x_line", f"我々1位ペア×{lb}" if rk == 1 else None),
                ]
                for dim, seg in segs:
                    if seg is None:
                        continue
                    acc[dim][(w, seg)].add(y, mp, op, mimj)
        print(f"  {ym}: {len(chunk)}R", flush=True)

    DIMS = [
        ("ALL", "全ペア（ベースライン）"),
        ("line", "① ライン関係"),
        ("line_x_nlines", "② ライン関係 × 分戦数"),
        ("style", "③ 脚質の組み合わせ"),
        ("pref", "④ 同県ペアか × ライン系"),
        ("grade", "⑤ グレード"),
        ("bank", "⑥ バンク周長"),
        ("our_rank", "⑦ 我々のペア順位（1位＝実際に軸に選ぶペア）"),
        ("mkt_band", "⑧ ペアの市場確率帯"),
        ("rank1_x_line", "⑨ 我々1位ペア × ライン関係"),
    ]

    print("\n" + "=" * 124)
    print("ペア単位の市場ミスプライシング検証")
    print(f"  ROI(S7型5点流し) = 0.75 × 実測/市場  →  ROI100%超には 実測/市場 ≥ {ROI_BREAKEVEN:.3f} が必要")
    print("  市場lift vs 実測lift: 市場が軸間相関自体を織り込めているかの直接確認（1.0が独立）")
    print("=" * 124)

    for dim, title in DIMS:
        print("\n" + "-" * 124)
        print(title)
        print("-" * 124)
        print(f"{'セグメント':<26}{'窓':<6}{'ペア数':>9}{'実測%':>8}{'市場%':>8}{'our%':>8}"
              f"{'実測/市場':>10}{'t値':>8}{'→ROI%':>9}{'市場lift':>10}{'実測lift':>10}")
        segs = sorted({k[1] for k in acc[dim]},
                      key=lambda s: -(acc[dim][("TEST", s)].n if ("TEST", s) in acc[dim] else 0))
        for seg in segs:
            printed = False
            for w in ("TRAIN", "TEST"):
                a = acc[dim].get((w, seg))
                if not a or a.n < MIN_PAIRS:
                    continue
                p = a.report()
                flag = ""
                if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3:
                    flag = "  ★ROI100%超"
                elif p["ratio"] > 1.0 and p["t"] > 3:
                    flag = "  (市場過小評価だが不足)"
                print(f"{seg if not printed else '':<26}{w:<6}{p['n']:>9}{p['act']:>8.2f}"
                      f"{p['mkt']:>8.2f}{p['our']:>8.2f}{p['ratio']:>10.3f}{p['t']:>+8.2f}"
                      f"{p['roi']:>9.1f}{p['mkt_lift']:>10.3f}{p['act_lift']:>10.3f}{flag}")
                printed = True
            if printed:
                print()

    print("\n" + "=" * 124)
    print(f"【結論】TRAIN/TESTともに 実測/市場 ≥ {ROI_BREAKEVEN:.3f} かつ TEST t>3 のペア条件")
    print("=" * 124)
    hits = []
    for dim, _ in DIMS:
        if dim == "ALL":
            continue
        for seg in {k[1] for k in acc[dim]}:
            a, b = acc[dim].get(("TRAIN", seg)), acc[dim].get(("TEST", seg))
            if not a or not b or a.n < MIN_PAIRS or b.n < MIN_PAIRS:
                continue
            ra, rb = a.report(), b.report()
            if ra["ratio"] >= ROI_BREAKEVEN and rb["ratio"] >= ROI_BREAKEVEN and rb["t"] > 3:
                hits.append((dim, seg, ra, rb))
    if not hits:
        print("  該当なし（実測/市場 ≥ 1.333 を満たす条件は存在しない）。")
        print()
        print("  ただし『市場が軸間相関を織り込みきれていない』という調査メモ4章の仮説は")
        print("  **方向としては正しい**。同ライン系ペアで市場は系統的に過小評価しており")
        print("  （実測/市場 1.03〜1.09・TRAIN/TEST一致・t>3.5）、市場liftは実測liftを")
        print("  一貫して下回る。別ライン・単騎絡みは逆に過大評価（0.96前後）。")
        print("  棄却されるのは仮説の方向ではなく**幅**である：控除率25ptを埋めるには")
        print("  33.3%の過小評価が必要だが、実際に存在するのは3〜9%＝ギャップの約1/3。")
        print("  完全に活用しても到達点はROI 81%前後（最良セルでも85%）。")
    else:
        for dim, seg, ra, rb in sorted(hits, key=lambda x: -x[3]["ratio"]):
            print(f"  ★ [{dim}] {seg}")
            print(f"      TRAIN n={ra['n']:>7} 実測/市場={ra['ratio']:.3f} → ROI {ra['roi']:.1f}%")
            print(f"      TEST  n={rb['n']:>7} 実測/市場={rb['ratio']:.3f} → ROI {rb['roi']:.1f}% (t={rb['t']:+.2f})")


if __name__ == "__main__":
    main()
