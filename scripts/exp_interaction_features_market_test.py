"""【交互作用特徴量の市場ミスプライシング測定】（2026-07-30）。

`inputs/外部データ源と交互作用特徴量_深掘り調査.md` 第II部のうち
**外部収集ゼロで即日着手できる項目（実行順序1〜5）**を実装・測定する。

## 測定する項目（メモ対応）

| 項目 | メモ | 内容 |
|---|---|---|
| A | II-7 | **級班混走 × elo残差** ★最優先（C-5の最優先セグメント） |
| B | II-8 / I-3 | 開催時間帯（ミッドナイト等）× 車番 / × 脚質 |
| C | II-2 | 隊列の力学: バック回数の分布 × 当該選手の脚質 |
| D | II-4 | S/B 比率（好位は取るが先行はしない＝番手狙い型の分離） |
| E | II-3 | ライン内非対称性（番手−三番手の得点差＝三番手ジャンプ） |

## ⚠️ 構造的制約への対応（メモ「測定時の追加注意」項目4と同一の問題）

[[keirin_c_candidates_market_test_2026_07_30]]で実証済み:
**レース単位の区分（全選手に同じ値が入る）は比が構造的に 1.000 に固定される。**
市場の周辺確率は Σ_i P_i = 3 に正規化され、実測も1レース必ず3車が3着内であるため、
分子と分母が恒等的に一致する。

メモが「レース単位変数は必ず『レース単位変数 × 選手単位変数』の交互作用として
入れること」と正しく指摘しているのと同じ制約であり、**測定側でも同じ扱いが必要**。
よって本スクリプトでは:

- `n_senko_in_race` / `B_top2_gap` / `B_gini` / `is_mixed_class_race` /
  `race_time_category` 等のレース単位変数は **単体では測定しない**（無意味）
- 必ず **当該選手の属性（脚質・車番・ライン位置・elo残差）との交互作用**として測る

## 測定プロトコル

    比 = 実測3着内率 ÷ 市場の3着内含意確率（三連複35通りから周辺化・Σ=3.0 検算）
    ROI = 0.75 × 比 → ROI100%超には 比 ≥ 1.333

- TRAIN/TEST 両窓で符号・効果量が一致することを要求
- 交互作用は多重比較の温床なので**メモに明記された仮説の方向のみ**を測る
  （総当たり探索はしない）
- 主要候補は人気帯で層別して交絡を確認

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
Elo は 2023-01 から chronological に warm-up。DB書き込みなし・読み取り専用。
"""
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

from exp_c_candidates_market_test import (  # noqa: E402
    ELO_INIT, ELO_K, MEAS_FROM, MEAS_TO, MIN_BOARD, ROI_BREAKEVEN,
    TAKEOUT_RETURN, TRAIN_TO, Acc, load_trio_odds,
)

HIST_FROM = "2023-01-01"
MIN_SEG = 1200
MIN_STRAT = 700
JST = timezone(timedelta(hours=9))


def load_all():
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date, race_no, grade, cup_id, day_index, "
            "       venue_id, n_entries, start_at FROM wt_races "
            "WHERE cancel = 0 AND race_date BETWEEN ? AND ?",
            (HIST_FROM, MEAS_TO)).fetchall()
    races = {}
    for r in rrows:
        races[r["race_key"]] = {
            "date": str(r["race_date"]), "race_no": r["race_no"] or 0,
            "grade": str(r["grade"] or "?"), "cup_id": r["cup_id"],
            "n_entries": r["n_entries"], "start_at": r["start_at"],
        }
    print(f"[load] races: {len(races)}", flush=True)

    keys = list(races)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, player_id, pred_top3_pct, player_class, "
                 "       style, race_point, s_count, b_count, h_count, "
                 "       line_group, line_pos, line_size, finish_order, final_half "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load] entries: {sum(len(v) for v in by_race.values())}", flush=True)
    return races, by_race


def time_category(start_at):
    """start_at は Unix epoch 秒の文字列。JST の発走時刻から開催形態を判定する。"""
    try:
        ts = int(str(start_at))
    except (TypeError, ValueError):
        return "時刻不明"
    h = datetime.fromtimestamp(ts, tz=JST).hour
    if h < 10:
        return "モーニング"
    if h < 15:
        return "デイ"
    if h < 20:
        return "ナイター"
    return "ミッドナイト"


def build_elo(races, entries_by_race):
    order = sorted(races, key=lambda rk: (races[rk]["date"], races[rk]["race_no"]))
    elo = defaultdict(lambda: ELO_INIT)
    snap = {}
    for rk in order:
        ents = entries_by_race.get(rk)
        if not ents:
            continue
        for e in ents:
            snap[(rk, e["player_id"])] = elo[e["player_id"]]
        fin = [(int(e["finish_order"]), e["player_id"]) for e in ents
               if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
        if len(fin) < 2:
            continue
        fin.sort()
        pids = [p for _, p in fin]
        n = len(pids)
        delta = defaultdict(float)
        for a in range(n):
            for b in range(a + 1, n):
                pa, pb = pids[a], pids[b]
                ea = 1.0 / (1.0 + 10 ** ((elo[pb] - elo[pa]) / 400.0))
                g = ELO_K * (1.0 - ea) / (n - 1)
                delta[pa] += g
                delta[pb] -= g
        for p, g in delta.items():
            elo[p] += g
    print(f"[elo] snapshots: {len(snap)}  players: {len(elo)}", flush=True)
    return snap


def zstats(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    sd = statistics.pstdev(v)
    return (m, sd) if sd > 0 else None


def gini(vals):
    v = sorted(x for x in vals if x is not None)
    n = len(v)
    if n < 2:
        return None
    s = sum(v)
    if s <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2 * cum) / (n * s) - (n + 1) / n


def main():
    races, entries = load_all()
    elo_snap = build_elo(races, entries)

    targets = [rk for rk, m in races.items()
               if m["n_entries"] == 7 and MEAS_FROM <= m["date"] <= MEAS_TO]
    print(f"[meas] 対象(7車立て): {len(targets)}", flush=True)

    acc = defaultdict(lambda: defaultdict(Acc))
    strat = defaultdict(lambda: defaultdict(Acc))
    sum_check = []

    by_month = defaultdict(list)
    for rk in targets:
        by_month[races[rk]["date"][:7]].append(rk)

    for ym in sorted(by_month):
        rks = by_month[ym]
        boards = load_trio_odds(rks)
        for rk in rks:
            meta = races[rk]
            ents = entries.get(rk)
            board = boards.get(rk)
            if not ents or len(ents) != 7 or not board or len(board) < MIN_BOARD:
                continue
            if any(e["pred_top3_pct"] is None for e in ents):
                continue
            fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
                   if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
            if len(fin) < 3:
                continue
            fin.sort()
            tm = 0
            for _, fr in fin[:3]:
                tm |= 1 << fr

            mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if o > 0}
            tot = sum(mk_raw.values())
            if tot <= 0:
                continue
            frames = sorted(int(e["frame_no"]) for e in ents)
            marg = {fr: 0.0 for fr in frames}
            for m, v in mk_raw.items():
                p = v / tot
                for fr in frames:
                    if (m >> fr) & 1:
                        marg[fr] += p
            if len(sum_check) < 3000:
                sum_check.append(sum(marg.values()))

            w = "TRAIN" if meta["date"] <= TRAIN_TO else "TEST"
            tcat = time_category(meta["start_at"])

            # ---- レース単位の構成変数（単体では測らない・交互作用専用）----
            classes = {str(e["player_class"] or "?") for e in ents}
            mixed = len(classes) > 1
            bcs = [float(e["b_count"]) if e["b_count"] is not None else None for e in ents]
            bs = [x for x in bcs if x is not None]
            b_sorted = sorted(bs, reverse=True)
            b_top2_gap = (b_sorted[0] - b_sorted[1]) if len(b_sorted) >= 2 else None
            b_gini = gini(bs)
            n_senko = sum(1 for e in ents if str(e["style"] or "") == "逃")

            elos = [elo_snap.get((rk, e["player_id"])) for e in ents]
            rps = [float(e["race_point"]) if e["race_point"] is not None else None
                   for e in ents]
            ze, zr = zstats(elos), zstats(rps)

            # S/B のレース内順位（1 = 最多）
            def rank_of(key):
                vals = [(float(e[key]) if e[key] is not None else -1.0, int(e["frame_no"]))
                        for e in ents]
                vals.sort(reverse=True)
                return {fr: i + 1 for i, (_v, fr) in enumerate(vals)}
            s_rank = rank_of("s_count")
            b_rank = rank_of("b_count")

            # ライン内の番手/三番手の得点差（三番手ジャンプの検出）
            lines = defaultdict(dict)
            for e in ents:
                lg = e["line_group"]
                lp = e["line_pos"]
                if lg is None or lp is None:
                    continue
                lines[lg][int(lp)] = e
            jump = {}          # frame_no -> ラベル
            for lg, pos in lines.items():
                if 2 in pos and 3 in pos:
                    r2 = pos[2]["race_point"]
                    r3 = pos[3]["race_point"]
                    if r2 is None or r3 is None:
                        continue
                    gap = float(r2) - float(r3)
                    lab = ("番手が三番手より弱い(差<-1)" if gap < -1.0 else
                           "番手≒三番手(-1〜1)" if gap <= 1.0 else "番手が三番手より強い(差>1)")
                    jump[int(pos[2]["frame_no"])] = ("番手視点/" + lab)
                    jump[int(pos[3]["frame_no"])] = ("三番手視点/" + lab)

            for idx, e in enumerate(ents):
                fr = int(e["frame_no"])
                mp = marg[fr]
                if mp <= 0:
                    continue
                y = 1.0 if (tm >> fr) & 1 else 0.0
                sty = str(e["style"] or "?")

                segs = []

                # ---- A: II-7 級班混走 × elo残差（★最優先）----
                resid = None
                if ze and zr and elos[idx] is not None and rps[idx] is not None:
                    resid = ((elos[idx] - ze[0]) / ze[1]) - ((rps[idx] - zr[0]) / zr[1])
                    rl = "elo残差>1.0" if resid > 1.0 else ("elo残差<-1.0" if resid < -1.0
                                                          else "elo残差±1.0内")
                    segs.append(("A_mixed_x_elo",
                                 f"{'級班混走' if mixed else '単一級班'}×{rl}"))

                # ---- B: II-8/I-3 開催時間帯 × 車番 / × 脚質 ----
                segs.append(("B_time_x_car", f"{tcat}×{fr}番車"))
                segs.append(("B_time_x_style", f"{tcat}×{sty}"))

                # ---- C: II-2 隊列の力学 × 脚質 ----
                if b_top2_gap is not None:
                    gl = "主導権争い(B差≤2)" if b_top2_gap <= 2 else "主導権明確(B差>2)"
                    segs.append(("C_bgap_x_style", f"{gl}×{sty}"))
                if b_gini is not None:
                    gl2 = "B分散(gini低)" if b_gini < 0.35 else "B集中(gini高)"
                    segs.append(("C_bgini_x_style", f"{gl2}×{sty}"))
                segs.append(("C_nsenko_x_style", f"先行{min(n_senko,3)}人×{sty}"))

                # ---- D: II-4 S/B 比率 ----
                sr, br = s_rank.get(fr), b_rank.get(fr)
                if sr and br:
                    if sr <= 2 and br >= 5:
                        dl = "S上位×B下位(番手狙い型)"
                    elif sr <= 2 and br <= 2:
                        dl = "S上位×B上位(主導権型)"
                    elif sr >= 5 and br >= 5:
                        dl = "S下位×B下位(後方待機型)"
                    else:
                        dl = "S/B 中間"
                    segs.append(("D_sb_type", dl))
                    segs.append(("D_sb_x_car", f"{dl}×{'内枠(1-3)' if fr <= 3 else '外枠(4-7)'}"))

                # ---- E: II-3 ライン内非対称性（三番手ジャンプ）----
                if fr in jump:
                    segs.append(("E_line_asym", jump[fr]))

                for dim, seg in segs:
                    acc[dim][(w, seg)].add(y, mp)

                # 交絡確認（人気帯層別）: A と D
                band = ("人気帯1(低)" if mp < 0.25 else "人気帯2" if mp < 0.40 else
                        "人気帯3" if mp < 0.55 else "人気帯4(高)")
                if resid is not None:
                    strat[("A_mixed_x_elo", band)][
                        (w, f"{'混走' if mixed else '単一'}×"
                            f"{'elo強' if resid > 1.0 else 'elo並'}")].add(y, mp)
                if sr and br:
                    strat[("D_sb_type", band)][(w, dl)].add(y, mp)

        print(f"  {ym}: {len(rks)}R", flush=True)

    print("\n" + "=" * 116)
    print("[検算] Σ_i market_P(3着内) = 3.0: "
          f"平均 {statistics.mean(sum_check):.4f} (n={len(sum_check)})")
    print("=" * 116)

    TITLES = [
        ("A_mixed_x_elo", "A【最優先・II-7】級班混走 × elo残差"
                          "（仮説: 混走では表示得点の歪みが最大化しeloの優位が出る）"),
        ("B_time_x_car", "B【II-8/I-3】開催時間帯 × 車番"
                         "（仮説: ミッドナイトは得点順配置なので車番の意味が異なる）"),
        ("B_time_x_style", "B【II-8】開催時間帯 × 脚質（仮説: ミッドナイトは先行有利）"),
        ("C_bgap_x_style", "C【II-2】主導権争い × 脚質"
                           "（仮説: B差が小さい＝叩き合い→先行不利・追込有利）"),
        ("C_bgini_x_style", "C【II-2】バック回数の集中度 × 脚質"),
        ("C_nsenko_x_style", "C【II-2】先行型の人数 × 脚質"),
        ("D_sb_type", "D【II-4】S/B 比率による行動タイプ"
                      "（仮説: S上位×B下位＝好位は取るが先行しない番手狙い型）"),
        ("D_sb_x_car", "D【II-4】行動タイプ × 枠（仮説: 内枠×S意志＝実際に前を取れる）"),
        ("E_line_asym", "E【II-3】ライン内の番手/三番手の力関係"
                        "（仮説: 番手が弱いと三番手が差してくる＝三番手ジャンプ）"),
    ]
    for dim, title in TITLES:
        if dim not in acc:
            continue
        print("\n" + "-" * 116)
        print(title)
        print("-" * 116)
        print(f"{'セグメント':<38}{'窓':<6}{'車数':>9}{'実測%':>8}{'市場%':>8}"
              f"{'比':>8}{'t値':>8}{'→ROI%':>9}{'判定':>16}")
        for seg in sorted({k[1] for k in acc[dim]}):
            printed = False
            for w in ("TRAIN", "TEST"):
                a = acc[dim].get((w, seg))
                if not a or a.n < MIN_SEG:
                    continue
                r = a.report()
                v = ""
                if r["ratio"] >= ROI_BREAKEVEN and r["t"] > 3:
                    v = "★ROI100%超"
                elif r["ratio"] > 1.0 and r["t"] > 3:
                    v = "市場が過小評価"
                elif r["ratio"] < 1.0 and r["t"] < -3:
                    v = "除外候補"
                print(f"{seg if not printed else '':<38}{w:<6}{r['n']:>9}{r['act']:>8.2f}"
                      f"{r['mkt']:>8.2f}{r['ratio']:>8.3f}{r['t']:>+8.2f}{r['roi']:>9.1f}{v:>16}")
                printed = True
            if printed:
                print()

    print("\n" + "-" * 116)
    print("【交絡確認】人気帯で層別")
    print("-" * 116)
    for dim, label in (("A_mixed_x_elo", "級班混走×elo残差"), ("D_sb_type", "S/B行動タイプ")):
        print(f"\n  ===== {label} =====")
        for band in ("人気帯1(低)", "人気帯2", "人気帯3", "人気帯4(高)"):
            key = (dim, band)
            if key not in strat:
                continue
            print(f"  [{band}]")
            for seg in sorted({k[1] for k in strat[key]}):
                for w in ("TRAIN", "TEST"):
                    a = strat[key].get((w, seg))
                    if not a or a.n < MIN_STRAT:
                        continue
                    r = a.report()
                    print(f"    {seg:<28}{w:<6}n={r['n']:>7} 実測{r['act']:>6.2f}% "
                          f"市場{r['mkt']:>6.2f}% 比{r['ratio']:>7.3f} t{r['t']:>+7.2f}")

    print("\n" + "=" * 116)
    print(f"【結論】両窓で 比 ≥ {ROI_BREAKEVEN:.3f} かつ TEST t>3")
    print("=" * 116)
    hits, excl, near = [], [], []
    for dim in acc:
        for seg in {k[1] for k in acc[dim]}:
            a, b = acc[dim].get(("TRAIN", seg)), acc[dim].get(("TEST", seg))
            if not a or not b or a.n < MIN_SEG or b.n < MIN_SEG:
                continue
            ra, rb = a.report(), b.report()
            if ra["ratio"] >= ROI_BREAKEVEN and rb["ratio"] >= ROI_BREAKEVEN and rb["t"] > 3:
                hits.append((dim, seg, ra, rb))
            elif ra["ratio"] > 1.0 and rb["ratio"] > 1.0 and ra["t"] > 2 and rb["t"] > 2:
                near.append((dim, seg, ra, rb))
            if ra["ratio"] < 1.0 and rb["ratio"] < 1.0 and ra["t"] < -3 and rb["t"] < -3:
                excl.append((dim, seg, ra, rb))
    print("  " + ("該当なし。" if not hits else ""))
    for dim, seg, ra, rb in sorted(hits, key=lambda x: -x[3]["ratio"]):
        print(f"  ★[{dim}] {seg}: TRAIN {ra['ratio']:.3f} / TEST {rb['ratio']:.3f} ROI={rb['roi']:.1f}%")

    print("\n【両窓で市場過小評価（比>1・t>2）＝方向は本物だが1.333未満】")
    print("  " + ("該当なし。" if not near else ""))
    for dim, seg, ra, rb in sorted(near, key=lambda x: -x[3]["ratio"]):
        print(f"  ○[{dim}] {seg}: TRAIN {ra['ratio']:.3f}(t{ra['t']:+.1f}) / "
              f"TEST {rb['ratio']:.3f}(t{rb['t']:+.1f}) ROI={rb['roi']:.1f}%")

    print("\n【除外ルール候補（両窓で比<1・|t|>3）】")
    print("  " + ("該当なし。" if not excl else ""))
    for dim, seg, ra, rb in sorted(excl, key=lambda x: x[3]["ratio"]):
        print(f"  ▼[{dim}] {seg}: TRAIN {ra['ratio']:.3f}(t{ra['t']:+.1f}) / "
              f"TEST {rb['ratio']:.3f}(t{rb['t']:+.1f}) ROI={rb['roi']:.1f}%")


if __name__ == "__main__":
    main()
