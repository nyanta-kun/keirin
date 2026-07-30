"""【結合確率モデル】3頭の組を直接スコアリングするモデルを構築し、市場と比較する
（2026-07-30・「相関補正込みの結合確率モデルの構築」）。

## 背景

現行は各選手の周辺確率(pred_top3_pct)の積で組を評価している。しかしペア相関の
測定で「同ライン1.12x / 別ライン0.67x（同一強度帯でも1.4-2.0倍差）」という巨大な
構造が判明した。手作業のlift補正を入れた版では:
  - 検出精度: 組ランクが低〜中配当帯で+0.5〜0.9改善（100倍+帯では悪化）
  - 市場との比較: Brier 0.02535 vs 市場 0.02450 で**依然として市場に負け**

本スクリプトは手作業のlift補正ではなく、**組(3頭)単位の特徴量でLightGBMを学習**して
「この3頭の組が3着内を占めるか」を直接予測する本格的な結合モデルを作る。

## 学習設計
- 1レース = 35通りの組。うち1つが正例（実際の3着内）、34が負例
- 特徴量は組レベル（3頭の周辺確率統計 + 3ペアのライン関係 + ライン構成 + レース属性）
- TRAIN 2024-01-01〜2025-12-31 で学習、TEST 2026-01-01〜2026-07-30 で評価
- **レース内で正規化**してから確率として扱う（35通りの和が1になるように）

## 比較対象（同一TESTで）
1. naive: 周辺確率の単純積（現行方式）
2. lift補正: TRAIN推定のペアliftを掛ける（手作業補正・既に測定済み）
3. **本モデル: 組単位LightGBM**
4. market: オッズ含意確率（控除率25%を戻す）

## 評価指標
- Brier / logloss（低いほど良い・市場に勝てるか）
- 検出精度: 実際の3着内の組が予測順で何位か（35通り中）・上位1/3/5に入る率
- 配当帯別の内訳（どの帯で市場に勝てるか／負けるか）
"""
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
TAKEOUT_RETURN = 0.75
BANDS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, float("inf"))]


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT r.race_key, r.race_date, r.grade, v.bank_length "
            "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
            "WHERE r.n_entries = 7 AND r.cancel = 0 AND r.race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: {"race_date": str(r["race_date"]), "grade": r["grade"],
                              "bank_length": r["bank_length"]} for r in rrows}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_top3_pct, pred_win_pct, prediction_mark, "
                 "       line_group, line_pos, line_size, is_line_leader, n_lines, style, "
                 "       race_point, prefecture, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))

    print("[load] trio boards ...", flush=True)
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
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None or e["pred_win_pct"] is None for e in ents):
            continue
        board = boards.get(rk)
        if not board or len(board) < 30:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        by_frame = {int(e["frame_no"]): e for e in ents}
        out.append({"race_key": rk, "race_date": meta["race_date"], "meta": meta,
                    "by_frame": by_frame, "board": board,
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


TRIPLE_FEATS = [
    "p_sum", "p_prod_log", "p_min", "p_max", "p_std",
    "w_sum", "w_max", "w_min",
    "n_same_line_pairs", "n_diff_line_pairs",
    "n_same12", "n_same23", "n_same13",
    "all_same_line", "n_distinct_lines",
    "n_line_leaders", "n_solo", "n_senko", "n_oikomi",
    "n_marks", "has_honmei", "has_taikou",
    "rp_sum", "rp_max", "rp_min",
    "same_pref_pairs",
    "field_top3_sum2", "field_entropy", "n_lines_race",
    "grade_enc", "bank_length",
    "rank_sum", "rank_max",
]
GRADE_MAP = {"L級": 0, "A級": 1, "S級": 2, "SA混合": 3}


def triple_features(r, tri, tsorted_rank):
    bf = r["by_frame"]
    ps = [float(bf[f]["pred_top3_pct"]) / 100.0 for f in tri]
    ws = [float(bf[f]["pred_win_pct"]) / 100.0 for f in tri]
    buckets = [pair_bucket(bf, x, y) for x, y in combinations(tri, 2)]
    lines = [bf[f]["line_group"] for f in tri]
    styles = [bf[f]["style"] for f in tri]
    marks = [bf[f]["prediction_mark"] for f in tri]
    rps = [float(bf[f]["race_point"]) if bf[f]["race_point"] is not None else 0.0 for f in tri]
    prefs = [bf[f]["prefecture"] for f in tri]

    all_p = sorted((float(bf[f]["pred_top3_pct"]) / 100.0 for f in bf), reverse=True)
    tot = sum(all_p)
    ent = 0.0
    if tot > 0:
        for v in all_p:
            s = max(v / tot, 1e-9)
            ent -= s * math.log(s)

    return [
        sum(ps), sum(math.log(max(p, 1e-9)) for p in ps), min(ps), max(ps), float(np.std(ps)),
        sum(ws), max(ws), min(ws),
        sum(1 for b in buckets if b.startswith("same")),
        sum(1 for b in buckets if b == "diff"),
        sum(1 for b in buckets if b == "same_12"),
        sum(1 for b in buckets if b == "same_23"),
        sum(1 for b in buckets if b == "same_13"),
        1.0 if len(set(l for l in lines if l is not None)) == 1 and None not in lines else 0.0,
        float(len(set(l for l in lines if l is not None))),
        float(sum(1 for f in tri if bf[f]["is_line_leader"] == 1)),
        float(sum(1 for f in tri if bf[f]["line_size"] == 1)),
        float(sum(1 for s in styles if s == "逃")),
        float(sum(1 for s in styles if s == "追")),
        float(sum(1 for m in marks if m in (1, 2, 3))),
        1.0 if 1 in marks else 0.0,
        1.0 if 2 in marks else 0.0,
        sum(rps), max(rps), min(rps),
        float(sum(1 for a, b in combinations(prefs, 2) if a and b and a == b)),
        all_p[0] + all_p[1], ent,
        float(next(iter(bf.values()))["n_lines"] or 0),
        float(GRADE_MAP.get(r["meta"]["grade"], -1)),
        float(r["meta"]["bank_length"] or 0),
        float(sum(tsorted_rank[f] for f in tri)),
        float(max(tsorted_rank[f] for f in tri)),
    ]


def make_dataset(rows):
    X, y, groups = [], [], []
    for r in rows:
        bf = r["by_frame"]
        frames = list(bf.keys())
        tsorted = sorted(frames, key=lambda f: -float(bf[f]["pred_top3_pct"]))
        rank = {f: i + 1 for i, f in enumerate(tsorted)}
        cnt = 0
        for tri in combinations(frames, 3):
            key = frozenset(tri)
            if key not in r["board"]:
                continue
            X.append(triple_features(r, tri, rank))
            y.append(1 if key == r["top3"] else 0)
            cnt += 1
        groups.append(cnt)
    return np.array(X, dtype=float), np.array(y), groups


def race_scores(r, lifts, model):
    """1レースの各組について naive/lift/model/market の確率を返す。"""
    bf = r["by_frame"]
    frames = list(bf.keys())
    tsorted = sorted(frames, key=lambda f: -float(bf[f]["pred_top3_pct"]))
    rank = {f: i + 1 for i, f in enumerate(tsorted)}
    p = {f: float(bf[f]["pred_top3_pct"]) / 100.0 for f in frames}

    keys, naive, lift, feats, mkt_raw = [], [], [], [], []
    for tri in combinations(frames, 3):
        k = frozenset(tri)
        if k not in r["board"]:
            continue
        base = p[tri[0]] * p[tri[1]] * p[tri[2]]
        mult = 1.0
        for x, y in combinations(tri, 2):
            mult *= lifts.get(pair_bucket(bf, x, y), 1.0)
        keys.append(k)
        naive.append(base)
        lift.append(base * mult)
        feats.append(triple_features(r, tri, rank))
        mkt_raw.append(TAKEOUT_RETURN / r["board"][k])
    if len(keys) < 30:
        return None
    mdl = model.predict_proba(np.array(feats, dtype=float))[:, 1]

    def norm(v):
        v = np.asarray(v, dtype=float)
        s = v.sum()
        return v / s if s > 0 else np.full(len(v), 1.0 / len(v))

    return keys, norm(naive), norm(lift), norm(mdl), norm(mkt_raw)


def band_of(p):
    for lo, hi in BANDS:
        if lo <= p < hi:
            return f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
    return "?"


def main():
    races, entries_by_race, boards = load_all()
    rows = build(races, entries_by_race, boards)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")

    print("[lift] TRAINでlift推定 ...", flush=True)
    lifts = estimate_lifts(train)

    print("[dataset] 組単位データセット構築 ...", flush=True)
    Xtr, ytr, _ = make_dataset(train)
    print(f"[dataset]   TRAIN: {Xtr.shape[0]}組 (正例{int(ytr.sum())})", flush=True)

    print("[train] 組単位LightGBM学習 ...", flush=True)
    import lightgbm as lgb
    model = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=500,
                                learning_rate=0.05, num_leaves=63, min_child_samples=100,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                verbose=-1)
    model.fit(Xtr, ytr)
    imp = sorted(zip(TRIPLE_FEATS, model.feature_importances_), key=lambda t: -t[1])
    print("[train] 重要度上位12: " + ", ".join(f"{f}({v})" for f, v in imp[:12]))

    # ===== 評価 =====
    for label, data in (("TRAIN", train), ("TEST", test)):
        print("\n" + "=" * 100)
        print(f"[{label}] 4方式の比較")
        print("=" * 100)
        stats = {m: {"b": 0.0, "ll": 0.0, "n": 0, "rank": 0, "top1": 0, "top3": 0, "top5": 0,
                     "rn": 0}
                 for m in ("naive", "lift", "model", "market")}
        band_stats = defaultdict(lambda: {m: {"b": 0.0, "n": 0, "rank": 0, "rn": 0}
                                           for m in ("naive", "lift", "model", "market")})
        for r in data:
            sc = race_scores(r, lifts, model)
            if sc is None:
                continue
            keys, pn, pl, pm, pk = sc
            band = band_of(r["board"].get(r["top3"], 0))
            for name, probs in (("naive", pn), ("lift", pl), ("model", pm), ("market", pk)):
                s = stats[name]
                bs = band_stats[band][name]
                for k, pr in zip(keys, probs):
                    yy = 1.0 if k == r["top3"] else 0.0
                    s["b"] += (pr - yy) ** 2
                    s["ll"] -= yy * math.log(max(pr, 1e-12)) + (1 - yy) * math.log(max(1 - pr, 1e-12))
                    s["n"] += 1
                    bs["b"] += (pr - yy) ** 2
                    bs["n"] += 1
                order = np.argsort(-probs)
                pos = next((idx + 1 for idx, oi in enumerate(order) if keys[oi] == r["top3"]), None)
                if pos:
                    s["rank"] += pos
                    s["rn"] += 1
                    s["top1"] += 1 if pos == 1 else 0
                    s["top3"] += 1 if pos <= 3 else 0
                    s["top5"] += 1 if pos <= 5 else 0
                    bs["rank"] += pos
                    bs["rn"] += 1
        print(f"  {'方式':<10}{'Brier':>12}{'logloss':>12}{'平均組rank':>12}"
              f"{'top1率':>9}{'top3率':>9}{'top5率':>9}")
        for name in ("naive", "lift", "model", "market"):
            s = stats[name]
            if s["n"] == 0:
                continue
            print(f"  {name:<10}{s['b']/s['n']:>12.6f}{s['ll']/s['n']:>12.6f}"
                  f"{s['rank']/max(s['rn'],1):>12.2f}"
                  f"{s['top1']/max(s['rn'],1)*100:>8.1f}%"
                  f"{s['top3']/max(s['rn'],1)*100:>8.1f}%"
                  f"{s['top5']/max(s['rn'],1)*100:>8.1f}%")

        print(f"\n  配当帯別の平均組rank（低いほど良い検出）")
        print(f"    {'配当帯':<10}{'n':>7}{'naive':>9}{'lift':>9}{'model':>9}{'market':>9}"
              f"{'model-市場':>12}")
        order_bands = [f"{lo}-{hi}" if hi != float("inf") else f"{lo}+" for lo, hi in BANDS]
        for b in order_bands:
            bs = band_stats.get(b)
            if not bs or bs["model"]["rn"] < 50:
                continue
            vals = {m: bs[m]["rank"] / bs[m]["rn"] for m in ("naive", "lift", "model", "market")}
            diff = vals["model"] - vals["market"]
            mark = " ★モデル優位" if diff < 0 else ""
            print(f"    {b:<10}{bs['model']['rn']:>7}{vals['naive']:>9.2f}{vals['lift']:>9.2f}"
                  f"{vals['model']:>9.2f}{vals['market']:>9.2f}{diff:>+12.2f}{mark}")


if __name__ == "__main__":
    main()
