"""【高配当帯のエッジがROIとして成立するかの検証】（2026-07-30）。

## 背景（`exp_joint_triple_model.py`の発見）

組(3頭)単位のLightGBM結合モデルを構築したところ、配当帯別の検出精度で
**50倍以上の帯では市場より正確**という結果が出た（平均組rank・低いほど良い）:

| 配当帯 | モデル | 市場 | 差 |
|---|---|---|---|
| 0-5倍 | 1.52 | 1.25 | +0.26 (市場優位) |
| 10-20倍 | 6.67 | 5.59 | +1.08 (市場優位) |
| 30-50倍 | 13.38 | 12.70 | +0.67 (市場優位) |
| **50-100倍** | **18.24** | 18.72 | **-0.48 (モデル優位)** |
| **100倍+** | **24.45** | 26.10 | **-1.65 (モデル優位)** |

低〜中配当帯は多くの人が買うので市場が精緻だが、極端な高配当帯は市場の関心が
薄くオッズが荒いため、モデルの体系的計算が優位に立てる、という解釈。

## 本スクリプトの検証内容

「検出精度で勝つ」ことと「ROIで勝つ」ことは別問題。控除率25%を超える優位性が
あるかを直接検証する。

### 実装可能性について
朝イチ入稿では最終オッズが使えないが、既存の`notify_prerace_wt.py`が発走15分前に
実オッズで判定する仕組みを持つため、**オッズを条件に使う戦略は発走前判定として
実装可能**。よってオッズ閾値による銘柄選択は現実的な設計として検証してよい。

### 検証手順
1. 組単位LightGBMをTRAINで学習（TESTには一切触れない）
2. 各組について model_prob（レース内正規化）と market_prob(=0.75/odds、正規化)を算出
3. 選択ルール「odds >= X かつ model_prob/market_prob >= Y」でベットを選び、
   TRAIN/TESTそれぞれでROI・的中率・件数/日を算出
4. **較正の確認**: 高オッズ帯で model_prob と実的中率が一致するか
   （前回の全帯域診断では実測が市場側に一致＝モデルの過信だった。
     高オッズ帯だけは違うのかを確認する）
5. 月次安定性（特定月の1本の大穴で見かけ上黒字になっていないか）
6. 1点100円固定・買い目は「モデルが選んだ個別の組」（軸2車構造とは独立の検証）
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
STAKE = 100

GRADE_MAP = {"L級": 0, "A級": 1, "S級": 2, "SA混合": 3}
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
    print(f"[load]   entries: {len(by_race)}", flush=True)

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
                print(f"[load]   odds progress: {i}/{len(keys)}", flush=True)
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
        out.append({"race_key": rk, "race_date": meta["race_date"], "meta": meta,
                    "by_frame": {int(e["frame_no"]): e for e in ents}, "board": board,
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
    return {(1, 2): "same_12", (2, 3): "same_23", (1, 3): "same_13"}.get((a, b), "same_other")


def triple_features(r, tri, rank):
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
        1.0 if 1 in marks else 0.0, 1.0 if 2 in marks else 0.0,
        sum(rps), max(rps), min(rps),
        float(sum(1 for a, b in combinations(prefs, 2) if a and b and a == b)),
        all_p[0] + all_p[1], ent,
        float(next(iter(bf.values()))["n_lines"] or 0),
        float(GRADE_MAP.get(r["meta"]["grade"], -1)),
        float(r["meta"]["bank_length"] or 0),
        float(sum(rank[f] for f in tri)), float(max(rank[f] for f in tri)),
    ]


def race_rows(r):
    bf = r["by_frame"]
    frames = list(bf.keys())
    tsorted = sorted(frames, key=lambda f: -float(bf[f]["pred_top3_pct"]))
    rank = {f: i + 1 for i, f in enumerate(tsorted)}
    keys, feats, mkt_raw = [], [], []
    for tri in combinations(frames, 3):
        k = frozenset(tri)
        if k not in r["board"]:
            continue
        keys.append(k)
        feats.append(triple_features(r, tri, rank))
        mkt_raw.append(TAKEOUT_RETURN / r["board"][k])
    return keys, feats, mkt_raw


def main():
    races, entries_by_race, boards = load_all()
    rows = build(races, entries_by_race, boards)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")

    print("[train] 組単位データセット構築+学習 ...", flush=True)
    X, y = [], []
    for r in train:
        keys, feats, _ = race_rows(r)
        for k, f in zip(keys, feats):
            X.append(f)
            y.append(1 if k == r["top3"] else 0)
    X = np.array(X, dtype=float)
    y = np.array(y)
    print(f"[train]   {X.shape[0]}組 正例{int(y.sum())}", flush=True)
    import lightgbm as lgb
    model = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=500,
                                learning_rate=0.05, num_leaves=63, min_child_samples=100,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(X, y)

    def collect(data):
        """各組の (odds, model_prob, market_prob, hit, race_date, race_key) を返す"""
        out = []
        for r in data:
            keys, feats, mkt_raw = race_rows(r)
            if len(keys) < 30:
                continue
            mp = model.predict_proba(np.array(feats, dtype=float))[:, 1]
            mp = mp / mp.sum() if mp.sum() > 0 else np.full(len(mp), 1.0 / len(mp))
            mk = np.array(mkt_raw, dtype=float)
            mk = mk / mk.sum() if mk.sum() > 0 else np.full(len(mk), 1.0 / len(mk))
            for k, o_m, o_k, feat_i in zip(keys, mp, mk, range(len(keys))):
                out.append({
                    "race_key": r["race_key"], "race_date": r["race_date"],
                    "odds": r["board"][k], "model": float(o_m), "market": float(o_k),
                    "hit": 1 if k == r["top3"] else 0,
                })
        return out

    print("[collect] TRAIN ...", flush=True)
    ctr = collect(train)
    print("[collect] TEST ...", flush=True)
    cte = collect(test)
    print(f"[collect]   TRAIN {len(ctr)}組 / TEST {len(cte)}組")

    # ===== 1. 較正の確認（オッズ帯別） =====
    print("\n" + "=" * 100)
    print("1. オッズ帯別の較正: 実的中率は model 側か market 側か")
    print("   （前回の全帯域診断では実測が市場側に一致＝モデルの過信だった。")
    print("     高オッズ帯だけは違うのかを確認する）")
    print("=" * 100)
    OBANDS = [(0, 10), (10, 30), (30, 50), (50, 100), (100, 300), (300, 1e9)]
    for label, data in (("TRAIN", ctr), ("TEST", cte)):
        print(f"\n  [{label}]")
        print(f"    {'オッズ帯':<12}{'組数':>10}{'実的中率':>11}{'model':>10}{'market':>10}{'判定':>18}")
        for lo, hi in OBANDS:
            sub = [d for d in data if lo <= d["odds"] < hi]
            if len(sub) < 500:
                continue
            n = len(sub)
            act = sum(d["hit"] for d in sub) / n
            mo = sum(d["model"] for d in sub) / n
            mk = sum(d["market"] for d in sub) / n
            d_mo, d_mk = abs(act - mo), abs(act - mk)
            v = "★モデル側" if d_mo < d_mk * 0.9 else ("市場側" if d_mk < d_mo * 0.9 else "中間")
            tag = f"{lo}-{hi}" if hi < 1e9 else f"{lo}+"
            print(f"    {tag:<12}{n:>10}{act*100:>10.3f}%{mo*100:>9.3f}%{mk*100:>9.3f}%{v:>18}")

    # ===== 2. 選択ルール別ROI =====
    print("\n" + "=" * 100)
    print("2. 選択ルール『odds >= X かつ model/market >= Y』のROI")
    print("=" * 100)
    n_days = {"TRAIN": len({d['race_date'] for d in ctr}),
              "TEST": len({d['race_date'] for d in cte})}
    print(f"  {'odds>=':>8}{'ratio>=':>9}"
          f"{'TRAIN件数':>10}{'/日':>7}{'的中%':>8}{'ROI%':>9}"
          f"{'TEST件数':>10}{'/日':>7}{'的中%':>8}{'ROI%':>9}")
    best = []
    for omin in (30, 50, 70, 100, 150):
        for rmin in (1.0, 1.2, 1.5, 2.0):
            res = []
            for label, data in (("TRAIN", ctr), ("TEST", cte)):
                sel = [d for d in data
                       if d["odds"] >= omin and d["market"] > 0
                       and d["model"] / d["market"] >= rmin]
                n = len(sel)
                if n == 0:
                    res.append((0, 0, 0, 0))
                    continue
                hits = sum(d["hit"] for d in sel)
                bet = n * STAKE
                pay = sum(int(d["odds"] * STAKE) for d in sel if d["hit"])
                res.append((n, n / n_days[label], hits / n * 100, pay / bet * 100))
            (n1, pd1, h1, r1), (n2, pd2, h2, r2) = res
            if n1 < 100:
                continue
            flag = ""
            if r1 >= 100 and r2 >= 100:
                flag = " ★★両窓100%超"
            elif r2 >= 100:
                flag = " ★TEST100%超"
            elif r1 >= 100:
                flag = " ★TRAIN100%超"
            print(f"  {omin:>8}{rmin:>9}"
                  f"{n1:>10}{pd1:>7.1f}{h1:>7.2f}%{r1:>8.1f}%"
                  f"{n2:>10}{pd2:>7.1f}{h2:>7.2f}%{r2:>8.1f}%{flag}")
            if r1 >= 100 and r2 >= 100:
                best.append((omin, rmin, n1, r1, n2, r2))

    # ===== 3. 両窓100%超があれば月次安定性 =====
    if best:
        print("\n" + "=" * 100)
        print("3. 両窓100%超の条件の月次安定性")
        print("=" * 100)
        for omin, rmin, n1, r1, n2, r2 in best[:3]:
            print(f"\n  --- odds>={omin} & ratio>={rmin} ---")
            by_m = defaultdict(lambda: {"n": 0, "hit": 0, "bet": 0, "pay": 0})
            for d in ctr + cte:
                if d["odds"] >= omin and d["market"] > 0 and d["model"] / d["market"] >= rmin:
                    m = d["race_date"][:7]
                    a = by_m[m]
                    a["n"] += 1
                    a["bet"] += STAKE
                    if d["hit"]:
                        a["hit"] += 1
                        a["pay"] += int(d["odds"] * STAKE)
            print(f"    {'年月':<10}{'件数':>7}{'的中':>6}{'ROI%':>9}")
            pos = 0
            for m in sorted(by_m):
                a = by_m[m]
                roi = a["pay"] / a["bet"] * 100 if a["bet"] else 0
                if roi >= 100:
                    pos += 1
                print(f"    {m:<10}{a['n']:>7}{a['hit']:>6}{roi:>8.1f}%")
            print(f"    → ROI100%超の月: {pos}/{len(by_m)}ヶ月")
    else:
        print("\n" + "=" * 100)
        print("3. 両窓でROI100%超となる条件は存在しなかった")
        print("=" * 100)


if __name__ == "__main__":
    main()
