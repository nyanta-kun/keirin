"""【市場エッジの有無を診断する】我々のモデルは市場が知らない情報を持っているか
（2026-07-30・ユーザー方針「モデルBは市場とのギャップで高配当を狙う」の前提検証）。

## 背景（ここまでの確定事実）

競輪の控除率は25%＝ランダムに買っても期待ROIは75%。我々のモデルは全構成で
ROI 73〜76%に張り付いており、**ROIにおいてランダムと同等**。一方で的中率は
明確に機能している（5点で52%的中・ランダムなら5/35=14%）。
つまり「どの組が来るか」は当てられているが**オッズが正確に織り込んでいる**ため
ROIが控除率の限界に張り付く＝教科書的な効率市場の結果。

これを破るには**市場が持っていない情報**が必要。本スクリプトはその有無を診断する。

## 決定的なテスト（較正の比較）

各3頭の組について:
  our_prob    = 相関補正済み結合確率（TRAINで推定したliftを適用・正規化）
  market_prob = 0.75 / trio_odds   （控除率25%を戻した市場の含意確率）

`our_prob > market_prob` の組（我々が「市場は過小評価」と見た組）を抽出し、
**実際の的中率が our_prob 側か market_prob 側のどちらに一致するか**を測る。
  - our_prob側に一致 → 真のエッジあり（モデルBが成立しうる）
  - market_prob側に一致 → 我々が過信しているだけ（モデルBは成立しない）

これは較正（calibration）の直接比較であり、ROIより先に確認すべき本質的な問い。

## 副次的に測るもの
- 我々の確率とオッズ含意確率のズレ幅別の実的中率（較正曲線）
- WINTICKET印とのギャップ別の較正（朝イチで使える代理指標の候補）
- Brierスコア/対数損失でのモデル vs 市場の直接比較

honest分割: TRAIN 2024-01-01〜2025-12-31（lift推定のみ）/ TEST 2026-01-01〜2026-07-30（評価）
"""
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
TAKEOUT_RETURN = 0.75   # 控除率25% → 払戻率75%


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: str(r["race_date"]) for r in rrows}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_top3_pct, prediction_mark, "
                 "       line_group, line_pos, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))

    print("[load] trio boards (全35通り) ...", flush=True)
    boards = {}
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
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    boards.setdefault(rk, {})[parts] = fv
            if (i // 600) % 25 == 0:
                print(f"[load]   progress: {i}/{len(keys)}", flush=True)
    print(f"[load]   boards: {len(boards)}", flush=True)
    return races, by_race, boards


def build(races, entries_by_race, boards):
    out = []
    for rk, rdate in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None for e in ents):
            continue
        board = boards.get(rk)
        if not board or len(board) < 30:   # 35通りほぼ揃っているレースのみ
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        by_frame = {int(e["frame_no"]): e for e in ents}
        out.append({"race_key": rk, "race_date": rdate, "by_frame": by_frame,
                    "board": board,
                    "top3": frozenset(fno for _, fno in fin[:3])})
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


def pair_bucket(bf, i, j):
    li, lj = bf[i]["line_group"], bf[j]["line_group"]
    if li is None or lj is None:
        return "unknown"
    if li != lj:
        return "diff"
    pi, pj = bf[i]["line_pos"], bf[j]["line_pos"]
    if pi is None or pj is None:
        return "same_other"
    a, b = sorted([int(pi), int(pj)])
    if (a, b) == (1, 2):
        return "same_12"
    if (a, b) == (2, 3):
        return "same_23"
    if (a, b) == (1, 3):
        return "same_13"
    return "same_other"


def estimate_lifts(rows):
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        for i, j in combinations(bf.keys(), 2):
            b = pair_bucket(bf, i, j)
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if i in r["top3"] and j in r["top3"]:
                a["obs"] += 1
    return {b: ((a["obs"] / a["n"]) / (a["exp"] / a["n"]) if a["n"] >= 100 and a["exp"] > 0 else 1.0)
            for b, a in agg.items()}


def race_probs(r, lifts):
    """35通りの組について our_prob（正規化済み）と market_prob を返す。"""
    bf = r["by_frame"]
    frames = list(bf.keys())
    p = {f: float(bf[f]["pred_top3_pct"]) / 100.0 for f in frames}
    raw = {}
    for tri in combinations(frames, 3):
        s = p[tri[0]] * p[tri[1]] * p[tri[2]]
        for x, y in combinations(tri, 2):
            s *= lifts.get(pair_bucket(bf, x, y), 1.0)
        raw[frozenset(tri)] = s
    tot = sum(raw.values())
    if tot <= 0:
        return None
    our = {k: v / tot for k, v in raw.items()}
    # 市場含意確率（控除率を戻し、レース内で正規化）
    mk_raw = {}
    for k, o in r["board"].items():
        if k in our and o > 0:
            mk_raw[k] = TAKEOUT_RETURN / o
    if not mk_raw:
        return None
    mtot = sum(mk_raw.values())
    market = {k: v / mtot for k, v in mk_raw.items()}
    common = set(our) & set(market)
    if len(common) < 30:
        return None
    return {k: (our[k], market[k], r["board"][k]) for k in common}


def main():
    races, entries_by_race, boards = load_all()
    rows = build(races, entries_by_race, boards)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")

    print("[lift] TRAINでlift推定 ...", flush=True)
    lifts = estimate_lifts(train)
    for b in sorted(lifts, key=lambda x: -lifts[x]):
        print(f"    {b:<12} {lifts[b]:.4f}")

    # ===== 1. our_prob > market_prob の組で、実際の的中率はどちら側か =====
    print("\n" + "=" * 100)
    print("1.【決定的テスト】our_prob > market_prob の組の実的中率")
    print("   我々の確率側に一致→真のエッジあり / 市場側に一致→我々が過信しているだけ")
    print("=" * 100)
    for label, data in (("TRAIN", train), ("TEST", test)):
        # 乖離率でビン分割
        bins = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 999.0)]
        agg = defaultdict(lambda: {"n": 0, "hit": 0, "our": 0.0, "mkt": 0.0})
        for r in data:
            pr = race_probs(r, lifts)
            if pr is None:
                continue
            for k, (op, mp, odds) in pr.items():
                if mp <= 0:
                    continue
                ratio = op / mp
                for lo, hi in bins:
                    if lo <= ratio < hi:
                        a = agg[(lo, hi)]
                        a["n"] += 1
                        a["our"] += op
                        a["mkt"] += mp
                        if k == r["top3"]:
                            a["hit"] += 1
                        break
        print(f"\n  [{label}]")
        print(f"    {'our/market':<14}{'組数':>10}{'実的中率':>10}"
              f"{'our平均':>10}{'市場平均':>10}{'判定':>20}")
        for lo, hi in bins:
            a = agg.get((lo, hi))
            if not a or a["n"] < 500:
                continue
            act = a["hit"] / a["n"]
            our = a["our"] / a["n"]
            mkt = a["mkt"] / a["n"]
            # 実測がourとmarketのどちらに近いか
            d_our = abs(act - our)
            d_mkt = abs(act - mkt)
            verdict = "★我々側(エッジ有)" if d_our < d_mkt * 0.8 else (
                "市場側(過信)" if d_mkt < d_our * 0.8 else "中間")
            tag = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
            print(f"    {tag:<14}{a['n']:>10}{act*100:>9.2f}%"
                  f"{our*100:>9.2f}%{mkt*100:>9.2f}%{verdict:>20}")

    # ===== 2. Brier / logloss でモデル vs 市場を直接比較 =====
    print("\n" + "=" * 100)
    print("2. 予測精度の直接比較（Brierスコア・対数損失。低い方が優秀）")
    print("=" * 100)
    for label, data in (("TRAIN", train), ("TEST", test)):
        n = 0
        b_our = b_mkt = ll_our = ll_mkt = 0.0
        for r in data:
            pr = race_probs(r, lifts)
            if pr is None:
                continue
            for k, (op, mp, odds) in pr.items():
                y = 1.0 if k == r["top3"] else 0.0
                n += 1
                b_our += (op - y) ** 2
                b_mkt += (mp - y) ** 2
                ll_our -= y * math.log(max(op, 1e-12)) + (1 - y) * math.log(max(1 - op, 1e-12))
                ll_mkt -= y * math.log(max(mp, 1e-12)) + (1 - y) * math.log(max(1 - mp, 1e-12))
        if n:
            print(f"  [{label}] 評価組数={n}")
            print(f"    Brier  : モデル={b_our/n:.6f}  市場={b_mkt/n:.6f}  "
                  f"{'★モデル優位' if b_our < b_mkt else '市場優位'}")
            print(f"    logloss: モデル={ll_our/n:.6f}  市場={ll_mkt/n:.6f}  "
                  f"{'★モデル優位' if ll_our < ll_mkt else '市場優位'}")

    # ===== 3. WINTICKET印とのギャップ別（朝イチで使える代理指標） =====
    print("\n" + "=" * 100)
    print("3. WINTICKET印とのギャップ別の較正（朝イチで使える代理指標の候補）")
    print("   我々の複勝確率トップ3にWT印(◎◯△)が何個含まれるかで層別")
    print("=" * 100)
    for label, data in (("TRAIN", train), ("TEST", test)):
        agg = defaultdict(lambda: {"n": 0, "hit": 0, "our": 0.0, "mkt": 0.0, "odds": []})
        for r in data:
            bf = r["by_frame"]
            marks = {int(f) for f in bf if bf[f]["prediction_mark"] in (1, 2, 3)}
            tsorted = sorted(bf.keys(), key=lambda f: -float(bf[f]["pred_top3_pct"]))
            our_top3 = set(tsorted[:3])
            overlap = len(our_top3 & marks)
            pr = race_probs(r, lifts)
            if pr is None:
                continue
            k = frozenset(our_top3)
            if k not in pr:
                continue
            op, mp, odds = pr[k]
            a = agg[overlap]
            a["n"] += 1
            a["our"] += op
            a["mkt"] += mp
            a["odds"].append(odds)
            if k == r["top3"]:
                a["hit"] += 1
        print(f"\n  [{label}] （我々の上位3頭をそのまま買った場合）")
        print(f"    {'WT印との重なり':<16}{'n':>8}{'実的中率':>10}{'our':>9}{'市場':>9}"
              f"{'オッズ中央値':>13}")
        for ov in sorted(agg):
            a = agg[ov]
            if a["n"] < 100:
                continue
            od = sorted(a["odds"])
            print(f"    {ov}個一致{'':<10}{a['n']:>8}{a['hit']/a['n']*100:>9.2f}%"
                  f"{a['our']/a['n']*100:>8.2f}%{a['mkt']/a['n']*100:>8.2f}%"
                  f"{od[len(od)//2]:>12.1f}倍")


if __name__ == "__main__":
    main()
