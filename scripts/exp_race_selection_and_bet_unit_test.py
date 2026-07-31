"""【netkeirinメモ タスク1・2・3の検証】レース選択・測定単位・軸2選定方式（2026-07-30）。

対象: `inputs/netkeirin予想家_ROI100超の構造分析.md`
ユーザーの直接の問い「1軸目選定後、相関を考慮した2軸目選定は行われているか」への
回答も兼ねる。

## 【最重要】測定単位の変更（メモ タスク2）

これまでの検証（[[keirin_pair_correlation_mispricing_2026_07_30]] 等）は
「軸ペアが三連複3着内に入るか」（trioオッズから合成した市場確率）で測っていた。
本スクリプトは Mr.T の実際の買い目構造に合わせ、**quinella（二車複）・
exacta（二車単）の実オッズ**を直接使う。

    quinella事象 = 軸ペア{a,b}が1-2着を占める（順序不問）
    exacta事象   = 軸ペア{a,b}が指定順序で1-2着を占める

wt_odds は quinella/exacta とも 2024-01〜2026-07 で100%カバレッジ確認済み
（trio/trifectaと同じ）。控除率もいずれも約25%（sum(1/odds)≈1.34、検算済み）。

## 軸選定3方式の比較（ユーザーの問いへの直接回答）

1. **S_marginal**（現行の実装方式）: pred_win_pct 上位2車を**独立に**選ぶ。
   軸2選定は軸1と無関係＝周辺確率の掛け算。[[keirin_pair_correlation_mispricing_2026_07_30]]
   で使ったのもこの方式（`our_pair = pi × pj × lift`のlift自体は補正だが、
   選定は依然として「上位2車」の独立ソート）
2. **S_jointpair**（相関考慮）: 全21ペアについて `pred_top3_pct(a)×pred_top3_pct(b)×
   同ラインlift` を計算し、**このjoint scoreが最大のペア**を選ぶ。ライン相関を
   選定時点で使うため、二軸探偵メモ4章が求めた「P(軸2|軸1)」の近似になっている
3. **S_market**（参考・上界）: quinella市場が最も高く評価しているペア
   （市場自身の二軸予想。これを超えられるかが本質的な問い）

## Task1: レース選択（分位別ROI）

⚠️ [[keirin_c_candidates_market_test_2026_07_30]]で「レース単位の区分を全車で
比を測ると構造的に1.000固定される」ことが判明済み。これはレース内の**全車**を
分母分子に含めるために起きる恒等式（Σ実測=Σ市場=3/レース）。
**本スクリプトはこの罠を回避する**: 各分位について「その分位のレースで実際に
軸として選んだ2車（1レースにつき2車のみ）」の比を測る。全車を含めないため
恒等式は成立せず、意味のある差が出せる。

レース単位の「読みやすさ」指標（発走前・オッズ非依存）:
  g12（win 1-2位差）/ top3_entropy / top3_sum_top2（≒ top2_prob_sum）
はTRAINカットで五分位化し、TESTで各分位内の軸ペア比を評価する。

## Task3: 券種別の比較

quinella / exacta / trio の市場確率帯別に比を測る。「組み合わせ数が多い券種は
市場価格形成が粗い」という仮説を、確率帯を揃えて比較することで検証する。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import re
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

from exp_segment_market_edge import (  # noqa: E402
    MIN_BOARD, TAKEOUT_RETURN, TEST_FROM, TEST_TO, TRAIN_FROM, TRAIN_TO,
    build_rows, load_entries, load_races, load_trio_odds, pair_bucket,
)

ROI_BREAKEVEN = 1.0 / TAKEOUT_RETURN
MIN_SEG = 300


def quintile_cuts(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return [0.0] * 4
    return [v[len(v) * i // 5] for i in range(1, 5)]


def load_pair_odds(race_keys, bet_type, sep):
    """quinella('=')/exacta('-') オッズを {race_key: {(a,b) or (a,b)ordered: odds}} で返す。"""
    out = {}
    keys = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = ? AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, [bet_type] + chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = [int(x) for x in str(comb).split(sep)]
                except ValueError:
                    continue
                if len(parts) != 2 or parts[0] == parts[1]:
                    continue
                key = tuple(sorted(parts)) if sep == "=" else tuple(parts)
                out.setdefault(rk, {})[key] = fv
    return out


def normalize(odds_map):
    """オッズ辞書 → 正規化された市場含意確率辞書（控除率を戻し合計1.0）。"""
    raw = {k: TAKEOUT_RETURN / o for k, o in odds_map.items() if o > 0}
    tot = sum(raw.values())
    if tot <= 0:
        return None
    return {k: v / tot for k, v in raw.items()}


def estimate_lifts(rows):
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        tm = r["top3_mask"]
        for i, j in combinations(sorted(bf), 2):
            b = pair_bucket(bf, i, j)
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if (tm >> i) & 1 and (tm >> j) & 1:
                a["obs"] += 1
    return {b: ((a["obs"] / a["n"]) / (a["exp"] / a["n"]) if a["n"] >= 100 and a["exp"] > 0 else 1.0)
            for b, a in agg.items()}


class Acc:
    __slots__ = ("n", "hit", "mkt", "d", "d2")

    def __init__(self):
        self.n = 0
        self.hit = self.mkt = self.d = self.d2 = 0.0

    def add(self, y, mp):
        self.n += 1
        self.hit += y
        self.mkt += mp
        d = y - mp
        self.d += d
        self.d2 += d * d

    def report(self):
        n = self.n
        act, mkt = self.hit / n, self.mkt / n
        md = self.d / n
        var = max(self.d2 / n - md * md, 0.0)
        t = md / math.sqrt(var / n) if var > 0 else 0.0
        ratio = act / mkt if mkt > 0 else 0.0
        return {"n": n, "act": act * 100, "mkt": mkt * 100, "ratio": ratio,
                "t": t, "roi": TAKEOUT_RETURN * ratio * 100}


def main():
    races = load_races()
    entries = load_entries(races.keys())
    rows = build_rows(races, entries)
    del entries

    train_rows = [r for r in rows if r["race_date"] <= TRAIN_TO]
    print(f"[split] TRAIN={len(train_rows)}R TEST={len(rows)-len(train_rows)}R")

    lifts = estimate_lifts(train_rows)
    print("\n[lift] ライン相関lift(TRAIN):", {k: round(v, 3) for k, v in lifts.items()})

    # レース選択指標のTRAINカット
    cuts = {
        "g12": quintile_cuts([r["g12"] for r in train_rows]),
        "entropy": quintile_cuts([r["top3_entropy"] for r in train_rows]),
        "sum_top2": quintile_cuts([r["top3_sum_top2"] for r in train_rows]),
    }
    sum_top2_vals = sorted(r["top3_sum_top2"] for r in train_rows)
    decile_cuts_sum_top2 = [sum_top2_vals[len(sum_top2_vals) * i // 10] for i in range(1, 10)]
    print("[cut] レース選択指標の五分位カット:", {k: [round(x, 3) for x in v] for k, v in cuts.items()})

    # 集計器
    axis_acc = defaultdict(lambda: defaultdict(Acc))     # strategy -> (window) -> Acc（quinella事象）
    order_acc = defaultdict(lambda: defaultdict(Acc))    # exacta: favorite/other/quinella比較
    seldim_acc = defaultdict(lambda: defaultdict(Acc))   # レース選択分位 -> (window,quantile) -> Acc
    bettype_acc = defaultdict(lambda: defaultdict(Acc))  # 券種×確率帯
    combo_acc = defaultdict(lambda: defaultdict(Acc))    # 戦略×sum_top2十分位（積み上げ確認）
    decile_acc = defaultdict(lambda: defaultdict(Acc))   # sum_top2十分位（分解能確認）

    by_month = defaultdict(list)
    for r in rows:
        by_month[r["race_date"][:7]].append(r)

    for ym in sorted(by_month):
        chunk = by_month[ym]
        rks = [r["race_key"] for r in chunk]
        trio_boards = load_trio_odds(rks)
        qn_boards = load_pair_odds(rks, "quinella", "=")
        ex_boards = load_pair_odds(rks, "exacta", "-")

        for r in chunk:
            rk = r["race_key"]
            trio = trio_boards.get(rk)
            qn = qn_boards.get(rk)
            ex = ex_boards.get(rk)
            if not trio or len(trio) < MIN_BOARD or not qn or not ex:
                continue
            qn_mkt = normalize(qn)
            ex_mkt = normalize(ex)
            if qn_mkt is None or ex_mkt is None:
                continue

            w = "TRAIN" if r["race_date"] <= TRAIN_TO else "TEST"
            bf = r["by_frame"]
            frames = sorted(bf)
            fin = [(int(bf[f]["finish_order"]), f) for f in frames
                   if bf[f]["finish_order"] is not None and int(bf[f]["finish_order"]) >= 1]
            fin.sort()
            if len(fin) < 2:
                continue
            actual_top2 = (fin[0][1], fin[1][1])          # (1着車, 2着車) 実際の順序
            actual_pair = tuple(sorted(actual_top2))

            # ---- 3つの軸選定方式でペアを決める ----
            # S_marginal: pred_win_pct 独立上位2車
            a1, a2 = r["win_order"][0], r["win_order"][1]
            pair_marginal = tuple(sorted([a1, a2]))

            # S_jointpair: 21ペア中 joint score 最大（相関考慮）
            best_pair, best_score = None, -1.0
            for i, j in combinations(frames, 2):
                pi = float(bf[i]["pred_top3_pct"]) / 100.0
                pj = float(bf[j]["pred_top3_pct"]) / 100.0
                s = pi * pj * lifts.get(pair_bucket(bf, i, j), 1.0)
                if s > best_score:
                    best_score, best_pair = s, (i, j)
            pair_joint = tuple(sorted(best_pair))

            # S_market: quinella市場が最も高く評価するペア
            pair_market = max(qn_mkt, key=lambda k: qn_mkt[k])

            st2 = r["top3_sum_top2"]
            decile = sum(1 for x in decile_cuts_sum_top2 if st2 >= x) + 1   # 1(最低)〜10(最高)
            for strat, pair in (("S_marginal(現行:独立2車)", pair_marginal),
                               ("S_jointpair(相関考慮)", pair_joint),
                               ("S_market(市場参考)", pair_market)):
                mp = qn_mkt.get(pair)
                if mp is None or mp <= 0:
                    continue
                y = 1.0 if pair == actual_pair else 0.0
                axis_acc[strat][w].add(y, mp)
                combo_label = "上位10%以内" if decile == 10 else ("上位20%以内" if decile >= 9 else "下位80%")
                combo_acc[strat][(w, combo_label)].add(y, mp)
                if decile == 10:   # 上位10%は上位20%にも含める（累積バケット）
                    combo_acc[strat][(w, "上位20%以内")].add(y, mp)
                if strat == "S_jointpair(相関考慮)":
                    decile_acc["sum_top2"][(w, f"D{decile}")].add(y, mp)

            # ---- Task2: 表裏(exacta) vs 順不同(quinella) ----
            # S_jointpair のペアで、pred_win_pct が高い方を「本命の順序」とする
            fav, other = ((pair_joint[0], pair_joint[1])
                         if float(bf[pair_joint[0]]["pred_win_pct"]) >= float(bf[pair_joint[1]]["pred_win_pct"])
                         else (pair_joint[1], pair_joint[0]))
            mp_qn = qn_mkt.get(pair_joint)
            mp_ex_fav = ex_mkt.get((fav, other))
            mp_ex_oth = ex_mkt.get((other, fav))
            if mp_qn and mp_ex_fav and mp_ex_oth:
                y_qn = 1.0 if pair_joint == actual_pair else 0.0
                y_fav = 1.0 if actual_top2 == (fav, other) else 0.0
                y_oth = 1.0 if actual_top2 == (other, fav) else 0.0
                order_acc["quinella(順不同=表裏両建てと同値)"][w].add(y_qn, mp_qn)
                order_acc["exacta(本命順のみ=片張り)"][w].add(y_fav, mp_ex_fav)
                order_acc["exacta(逆順のみ)"][w].add(y_oth, mp_ex_oth)

            # ---- Task1: レース選択分位（S_jointpairの軸ペアで測る・全車を含めない）----
            mp_joint = qn_mkt.get(pair_joint)
            if mp_joint and mp_joint > 0:
                y_joint = 1.0 if pair_joint == actual_pair else 0.0
                for dim, val in (("g12", r["g12"]), ("entropy", r["top3_entropy"]),
                                 ("sum_top2", r["top3_sum_top2"])):
                    c_ = cuts[dim]
                    q = sum(1 for x in c_ if val >= x) + 1     # 1=値が最小の帯・5=値が最大の帯
                    label = f"Q{q}/5"
                    seldim_acc[dim][(w, label)].add(y_joint, mp_joint)

            # ---- Task3: 券種別・確率帯別 ----
            for label, mkt_map, actual_check in (
                ("quinella", qn_mkt, lambda k: tuple(sorted(k)) == actual_pair),
                ("exacta", ex_mkt, lambda k: k == actual_top2),
            ):
                for k, mp in mkt_map.items():
                    band = ("<5%" if mp < 0.05 else "5-10%" if mp < 0.10 else
                            "10-20%" if mp < 0.20 else "20-35%" if mp < 0.35 else "35%+")
                    y = 1.0 if actual_check(k) else 0.0
                    bettype_acc[(label, band)][w].add(y, mp)
            # trio（3着内・既存の枠組みとの整合確認用）
            trio_mkt = normalize(trio)
            if trio_mkt:
                fin3 = fin[:3] if len(fin) >= 3 else None
                if fin3:
                    tm3 = 0
                    for _, fno in fin3:
                        tm3 |= 1 << fno
                    for m, mp in trio_mkt.items():   # m はビットマスク整数（load_trio_odds仕様）
                        band = ("<5%" if mp < 0.05 else "5-10%" if mp < 0.10 else
                                "10-20%" if mp < 0.20 else "20-35%" if mp < 0.35 else "35%+")
                        y = 1.0 if m == tm3 else 0.0
                        bettype_acc[("trio", band)][w].add(y, mp)

        print(f"  {ym}: {len(chunk)}R", flush=True)

    # ================= 出力 =================
    print("\n" + "=" * 118)
    print("軸選定3方式の比較（quinella市場=軸ペアが1-2着を占める事象）")
    print("  ★ユーザーの問い「1軸目選定後、相関考慮した2軸目選定は行われているか」への直接回答")
    print("=" * 118)
    print(f"{'方式':<28}{'窓':<6}{'n':>7}{'実測%':>8}{'市場%':>8}{'比':>8}{'t値':>8}{'→ROI%':>9}")
    for strat in ("S_marginal(現行:独立2車)", "S_jointpair(相関考慮)", "S_market(市場参考)"):
        for w in ("TRAIN", "TEST"):
            a = axis_acc[strat].get(w)
            if not a or a.n < MIN_SEG:
                continue
            p = a.report()
            print(f"{strat:<28}{w:<6}{p['n']:>7}{p['act']:>8.2f}{p['mkt']:>8.2f}"
                  f"{p['ratio']:>8.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}")
        print()

    print("\n" + "=" * 118)
    print("追加: sum_top2 十分位（分解能確認・S_jointpair軸）")
    print("=" * 118)
    print(f"{'分位':<8}{'窓':<6}{'n':>7}{'実測%':>8}{'市場%':>8}{'比':>8}{'t値':>8}{'→ROI%':>9}")
    for d in range(1, 11):
        seg = f"D{d}"
        for w in ("TRAIN", "TEST"):
            a = decile_acc["sum_top2"].get((w, seg))
            if not a or a.n < 100:
                continue
            p = a.report()
            flag = "  ★" if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3 else ""
            print(f"{seg:<8}{w:<6}{p['n']:>7}{p['act']:>8.2f}{p['mkt']:>8.2f}"
                  f"{p['ratio']:>8.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}{flag}")

    print("\n" + "=" * 118)
    print("追加: レース選択(sum_top2上位%) × 軸選定方式 の積み上げ")
    print("=" * 118)
    print(f"{'方式':<28}{'絞り込み':<12}{'窓':<6}{'n':>7}{'実測%':>8}{'市場%':>8}{'比':>8}{'t値':>8}{'→ROI%':>9}")
    for strat in ("S_marginal(現行:独立2車)", "S_jointpair(相関考慮)", "S_market(市場参考)"):
        for cond in ("上位10%以内", "上位20%以内", "下位80%"):
            for w in ("TRAIN", "TEST"):
                a = combo_acc[strat].get((w, cond))
                if not a or a.n < 100:
                    continue
                p = a.report()
                flag = "  ★" if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3 else ""
                print(f"{strat:<28}{cond:<12}{w:<6}{p['n']:>7}{p['act']:>8.2f}{p['mkt']:>8.2f}"
                      f"{p['ratio']:>8.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}{flag}")
        print()

    print("\n" + "=" * 118)
    print("Task2: 順不同(quinella=表裏両建てと同値) vs 順序特定(exacta)")
    print("  quinella比 > exacta本命順比 なら「両建てが正解」（Mr.Tの実装と一致）")
    print("=" * 118)
    print(f"{'構成':<32}{'窓':<6}{'n':>7}{'実測%':>8}{'市場%':>8}{'比':>8}{'t値':>8}{'→ROI%':>9}")
    for label in ("quinella(順不同=表裏両建てと同値)", "exacta(本命順のみ=片張り)", "exacta(逆順のみ)"):
        for w in ("TRAIN", "TEST"):
            a = order_acc[label].get(w)
            if not a or a.n < MIN_SEG:
                continue
            p = a.report()
            print(f"{label:<32}{w:<6}{p['n']:>7}{p['act']:>8.2f}{p['mkt']:>8.2f}"
                  f"{p['ratio']:>8.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}")
        print()

    print("\n" + "=" * 118)
    print("Task1: レース選択（軸ペアのみで測定・全車を含めないため恒等式1.000は回避される）")
    print("=" * 118)
    for dim, title in (("g12", "① win 1-2位差（Q5=差が大きい＝主導権明確・最も読みやすいはず）"),
                       ("entropy", "② top3エントロピー（Q1=エントロピー小＝最も読みやすいはず）"),
                       ("sum_top2", "③ top2確率合計＝Mr.T的『二軸の堅さ』（Q5=合計が大きい＝堅い）")):
        print(f"\n--- {title} ---")
        print(f"{'分位':<12}{'窓':<6}{'n':>7}{'実測%':>8}{'市場%':>8}{'比':>8}{'t値':>8}{'→ROI%':>9}")
        segs = sorted({k[1] for k in seldim_acc[dim]})
        for seg in segs:
            for w in ("TRAIN", "TEST"):
                a = seldim_acc[dim].get((w, seg))
                if not a or a.n < MIN_SEG:
                    continue
                p = a.report()
                flag = "  ★" if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3 else ""
                print(f"{seg:<12}{w:<6}{p['n']:>7}{p['act']:>8.2f}{p['mkt']:>8.2f}"
                      f"{p['ratio']:>8.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}{flag}")

    print("\n" + "=" * 118)
    print("Task3: 券種別・確率帯別の比（組み合わせ数が多い券種ほど価格形成が粗いか）")
    print("=" * 118)
    print(f"{'券種':<12}{'確率帯':<10}{'窓':<6}{'n':>8}{'実測%':>8}{'市場%':>8}{'比':>8}{'t値':>8}{'→ROI%':>9}")
    for bt in ("quinella", "exacta", "trio"):
        for band in ("<5%", "5-10%", "10-20%", "20-35%", "35%+"):
            key = (bt, band)
            if key not in bettype_acc:
                continue
            for w in ("TRAIN", "TEST"):
                a = bettype_acc[key].get(w)
                if not a or a.n < MIN_SEG:
                    continue
                p = a.report()
                print(f"{bt:<12}{band:<10}{w:<6}{p['n']:>8}{p['act']:>8.2f}{p['mkt']:>8.2f}"
                      f"{p['ratio']:>8.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}")
        print()


if __name__ == "__main__":
    main()
