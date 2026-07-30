"""【1車選択の精度向上】t1固定をやめ、メンバー構成から切替判断できるか
（2026-07-30・ユーザー提案「最上位固定ではなくレース出走メンバの構成を考慮し、
最上位の3着以内が困難な可能性を検出し別の1車を選択できれば90%程度の精度」）。

## ベースラインと理論上限（測定済み・TRAIN/TEST完全一致）

| | TRAIN | TEST |
|---|---|---|
| 現行（t1=pred_top3_pct最上位を固定） | 79.33% | 79.23% |
| オラクル上限（t1/t2から最適選択） | 93.86% | 93.43% |
| オラクル上限（t1〜t3から最適選択） | 98.27% | 98.03% |

t1固定79.2%に対し、t1/t2の2択を完璧に切り替えられれば93.4%。**14ptの伸び代**が
理論的に存在する。ユーザー想定の90%は射程内。

## 検証する3つのアプローチ

### A. t1失敗検出器 → 切替
「t1が3着内を外す」を予測する分類器をメンバー構成特徴量で学習し、
失敗確率が閾値超のレースでt2に切り替える。閾値を振って精度を測る。

### B. 直接ランキングモデル
「この選手が3着内に入るか」を**メンバー構成を明示的に含む特徴量**で学習し直し、
その最上位1車の精度を現行t1と比較する。現行の`pred_top3_pct`は選手単位の
周辺確率モデルで、フィールド相対情報が限定的（score_rank/wr_rank等のみ）。
ライン構成・脚質構成・相対的な力関係をより明示的に入れる。

### C. ペアワイズ比較モデル
「t1 vs t2 のどちらが3着内に入りやすいか」を直接学習する二値分類。
Aの間接的な失敗検出より、2択の判断としては素直な定式化。

## 重要な注意（この検証の落とし穴）
「t1が失敗する」と検出できても、**その状況でt2が成功するとは限らない**。
t1が失敗するのはレースが荒れているからで、その場合t2も失敗しやすい。
よって「失敗検出のAUC」だけでなく**切替後の実精度**を必ず測る。

条件付き確率の事実確認も出力する:
  P(t2 in top3 | t1 not in top3) vs P(t2 in top3 | t1 in top3)

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
"""
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
JST = timezone(timedelta(hours=9))
GRADE_MAP = {"L級": 0, "A級": 1, "S級": 2, "SA混合": 3}


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT r.race_key, r.race_date, r.grade, r.day_index, r.start_at, r.distance, "
            "       v.bank_length, v.is_indoor "
            "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
            "WHERE r.n_entries = 7 AND r.cancel = 0 AND r.race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {}
    for r in rrows:
        st = r["start_at"]
        night = 0
        if st is not None:
            try:
                night = 1 if datetime.fromtimestamp(int(st), tz=timezone.utc).astimezone(JST).hour >= 17 else 0
            except (TypeError, ValueError):
                pass
        races[r["race_key"]] = {"race_date": str(r["race_date"]), "grade": r["grade"],
                                 "day_index": r["day_index"], "bank_length": r["bank_length"],
                                 "is_indoor": r["is_indoor"], "distance": r["distance"],
                                 "is_night": night}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_top3_pct, pred_win_pct, prediction_mark, "
                 "       line_group, line_pos, line_size, is_line_leader, n_lines, style, "
                 "       race_point, prefecture, player_class, gear_ratio, term, "
                 "       first_rate, second_rate, third_rate, s_count, b_count, "
                 "       finish_order FROM wt_entries WHERE race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load]   entries: {len(by_race)}", flush=True)
    return races, by_race


def build(races, entries_by_race):
    out = []
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None or e["pred_win_pct"] is None for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        out.append({"race_key": rk, "race_date": meta["race_date"], "meta": meta,
                    "ents": ents, "by_frame": {int(e["frame_no"]): e for e in ents},
                    "top3": frozenset(f for _, f in fin[:3])})
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


# ========== 特徴量 ==========
# 「対象選手 target」 + 「フィールド構成」 の組み合わせ
RIDER_FEATS = [
    # 対象選手自身
    "t_top3", "t_win", "t_rank_top3", "t_rank_win",
    "t_is_leader", "t_line_size", "t_line_pos", "t_is_solo",
    "t_style_nige", "t_style_oi", "t_style_ryo",
    "t_mark", "t_rp", "t_rp_rank", "t_class_enc",
    "t_first_rate", "t_third_rate", "t_s_count", "t_b_count",
    # 対象選手のフィールド内での相対位置
    "t_top3_minus_2nd", "t_top3_over_max", "t_top3_minus_mean",
    "t_rp_minus_max", "t_rp_z",
    # 対象選手のライン内の状況
    "t_line_mate_best_top3", "t_line_mate_count", "t_line_has_nige",
    "t_line_leader_top3",
    # フィールド全体の構成（メンバー構成）
    "f_top3_max", "f_top3_2nd", "f_top3_3rd", "f_top3_gap12", "f_top3_gap23",
    "f_top3_sum2", "f_top3_sum3", "f_entropy",
    "f_win_max", "f_win_gap12", "f_win_entropy",
    "f_n_lines", "f_max_line_size", "f_n_solo", "f_n_nige", "f_n_oi",
    "f_rp_std", "f_rp_gap12", "f_rp_max",
    "f_n_marks_in_top3pred",
    # 対象選手を脅かす存在の数
    "n_stronger_diff_line", "n_stronger_same_line",
    "n_nige_diff_line", "n_solo_stronger",
    # レース属性
    "grade_enc", "bank_length", "is_indoor", "is_night", "distance", "day_index",
]


def rider_features(r, target):
    bf = r["by_frame"]
    ents = r["ents"]
    e = bf[target]
    frames = list(bf.keys())
    tp = {f: float(bf[f]["pred_top3_pct"]) for f in frames}
    wp = {f: float(bf[f]["pred_win_pct"]) for f in frames}
    tsorted = sorted(frames, key=lambda f: -tp[f])
    wsorted = sorted(frames, key=lambda f: -wp[f])
    trank = {f: i + 1 for i, f in enumerate(tsorted)}
    wrank = {f: i + 1 for i, f in enumerate(wsorted)}
    tvals = [tp[f] for f in tsorted]
    wvals = [wp[f] for f in wsorted]

    rps = {f: (float(bf[f]["race_point"]) if bf[f]["race_point"] is not None else 0.0)
           for f in frames}
    rp_sorted = sorted(rps.values(), reverse=True)
    rp_rank = {f: i + 1 for i, f in enumerate(sorted(frames, key=lambda x: -rps[x]))}
    rp_mean = sum(rp_sorted) / len(rp_sorted)
    rp_std = float(np.std(rp_sorted)) or 1.0

    def ent(vals):
        tot = sum(vals)
        if tot <= 0:
            return 0.0
        s = 0.0
        for v in vals:
            q = max(v / tot, 1e-9)
            s -= q * math.log(q)
        return s

    lg = e["line_group"]
    mates = [f for f in frames if f != target and bf[f]["line_group"] is not None
             and bf[f]["line_group"] == lg] if lg is not None else []
    mate_best = max((tp[f] for f in mates), default=0.0)
    line_has_nige = 1.0 if any(bf[f]["style"] == "逃" for f in mates + [target]) else 0.0
    leader = next((f for f in ([target] + mates) if bf[f]["is_line_leader"] == 1), None)
    leader_top3 = tp[leader] if leader is not None else 0.0

    line_sizes = defaultdict(int)
    for f in frames:
        if bf[f]["line_group"] is not None:
            line_sizes[bf[f]["line_group"]] += 1

    stronger = [f for f in frames if tp[f] > tp[target]]
    n_str_diff = sum(1 for f in stronger
                     if bf[f]["line_group"] != lg or lg is None or bf[f]["line_group"] is None)
    n_str_same = len(stronger) - n_str_diff
    n_nige_diff = sum(1 for f in frames
                      if bf[f]["style"] == "逃" and f != target
                      and (lg is None or bf[f]["line_group"] != lg))
    n_solo_str = sum(1 for f in stronger if bf[f]["line_size"] == 1)

    marks_in_pred_top3 = sum(1 for f in tsorted[:3] if bf[f]["prediction_mark"] in (1, 2, 3))

    return [
        tp[target], wp[target], float(trank[target]), float(wrank[target]),
        float(e["is_line_leader"] or 0), float(e["line_size"] or 0), float(e["line_pos"] or 0),
        1.0 if e["line_size"] == 1 else 0.0,
        1.0 if e["style"] == "逃" else 0.0, 1.0 if e["style"] == "追" else 0.0,
        1.0 if e["style"] == "両" else 0.0,
        float(e["prediction_mark"] or 0), rps[target], float(rp_rank[target]),
        float({"S1": 3, "S2": 2, "A1": 1, "A2": 0, "A3": -1, "L1": 0}.get(e["player_class"], 0)),
        float(e["first_rate"] or 0), float(e["third_rate"] or 0),
        float(e["s_count"] or 0), float(e["b_count"] or 0),
        tp[target] - (tvals[1] if trank[target] == 1 else tvals[0]),
        tp[target] / max(tvals[0], 1e-9),
        tp[target] - sum(tvals) / len(tvals),
        rps[target] - rp_sorted[0], (rps[target] - rp_mean) / rp_std,
        mate_best, float(len(mates)), line_has_nige, leader_top3,
        tvals[0], tvals[1], tvals[2], tvals[0] - tvals[1], tvals[1] - tvals[2],
        tvals[0] + tvals[1], tvals[0] + tvals[1] + tvals[2], ent(tvals),
        wvals[0], wvals[0] - wvals[1], ent(wvals),
        float(e["n_lines"] or 0), float(max(line_sizes.values()) if line_sizes else 0),
        float(sum(1 for v in line_sizes.values() if v == 1)),
        float(sum(1 for f in frames if bf[f]["style"] == "逃")),
        float(sum(1 for f in frames if bf[f]["style"] == "追")),
        rp_std, rp_sorted[0] - rp_sorted[1], rp_sorted[0],
        float(marks_in_pred_top3),
        float(n_str_diff), float(n_str_same), float(n_nige_diff), float(n_solo_str),
        float(GRADE_MAP.get(r["meta"]["grade"], -1)),
        float(r["meta"]["bank_length"] or 0), float(r["meta"]["is_indoor"] or 0),
        float(r["meta"]["is_night"]), float(r["meta"]["distance"] or 0),
        float(r["meta"]["day_index"] or 0),
    ]


def main():
    races, entries_by_race = load_all()
    rows = build(races, entries_by_race)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")

    def tk(r, k):
        bf = r["by_frame"]
        return sorted(bf.keys(), key=lambda f: -float(bf[f]["pred_top3_pct"]))[k]

    # ===== 0. 条件付き確率の事実確認（この検証の要） =====
    print("\n" + "=" * 96)
    print("0. 条件付き確率: t1が外れたときt2は入るのか（切替が原理的に有効かの確認）")
    print("=" * 96)
    for label, data in (("TRAIN", train), ("TEST", test)):
        a = defaultdict(int)
        for r in data:
            t1, t2, t3 = tk(r, 0), tk(r, 1), tk(r, 2)
            h1, h2, h3 = t1 in r["top3"], t2 in r["top3"], t3 in r["top3"]
            a["n"] += 1
            if h1:
                a["t1_hit"] += 1
                a["t2_given_t1hit"] += 1 if h2 else 0
            else:
                a["t1_miss"] += 1
                a["t2_given_t1miss"] += 1 if h2 else 0
                a["t3_given_t1miss"] += 1 if h3 else 0
        print(f"  [{label}] n={a['n']}")
        print(f"    P(t2 in top3 | t1 in top3)     = {a['t2_given_t1hit']/max(a['t1_hit'],1)*100:.2f}%")
        print(f"    P(t2 in top3 | t1 NOT in top3) = {a['t2_given_t1miss']/max(a['t1_miss'],1)*100:.2f}%")
        print(f"    P(t3 in top3 | t1 NOT in top3) = {a['t3_given_t1miss']/max(a['t1_miss'],1)*100:.2f}%")

    # ===== A. t1失敗検出器 =====
    print("\n" + "=" * 96)
    print("A. t1失敗検出器（t1が3着内を外すかを予測）→ 失敗予測時にt2へ切替")
    print("=" * 96)
    import lightgbm as lgb
    Xtr = np.array([rider_features(r, tk(r, 0)) for r in train], dtype=float)
    ytr = np.array([0 if tk(r, 0) in r["top3"] else 1 for r in train])
    Xte = np.array([rider_features(r, tk(r, 0)) for r in test], dtype=float)
    yte = np.array([0 if tk(r, 0) in r["top3"] else 1 for r in test])
    mf = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=400,
                             learning_rate=0.05, num_leaves=31, min_child_samples=60,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    mf.fit(Xtr, ytr)
    ptr, pte = mf.predict_proba(Xtr)[:, 1], mf.predict_proba(Xte)[:, 1]
    print(f"  失敗検出AUC: TRAIN={roc_auc_score(ytr, ptr):.4f}  **TEST={roc_auc_score(yte, pte):.4f}**")
    imp = sorted(zip(RIDER_FEATS, mf.feature_importances_), key=lambda t: -t[1])
    print(f"  重要度上位10: " + ", ".join(f"{f}({v})" for f, v in imp[:10]))

    print(f"\n  切替閾値別の1車精度（失敗確率>閾値ならt2に切替）")
    print(f"    {'閾値':<10}{'TRAIN切替数':>12}{'TRAIN精度':>11}{'TEST切替数':>12}{'TEST精度':>11}")
    base_tr = sum(1 for r in train if tk(r, 0) in r["top3"]) / len(train) * 100
    base_te = sum(1 for r in test if tk(r, 0) in r["top3"]) / len(test) * 100
    print(f"    {'切替なし':<10}{0:>12}{base_tr:>10.2f}%{0:>12}{base_te:>10.2f}%")
    for th in (0.30, 0.40, 0.50, 0.60, 0.70):
        res = []
        for data, probs in ((train, ptr), (test, pte)):
            sw = ok = 0
            for r, p in zip(data, probs):
                pick = tk(r, 1) if p > th else tk(r, 0)
                if p > th:
                    sw += 1
                if pick in r["top3"]:
                    ok += 1
            res.append((sw, ok / len(data) * 100))
        (s1, a1), (s2, a2) = res
        flag = " ★改善" if a2 > base_te else ""
        print(f"    {th:<10}{s1:>12}{a1:>10.2f}%{s2:>12}{a2:>10.2f}%{flag}")

    # ===== B. 直接ランキングモデル =====
    print("\n" + "=" * 96)
    print("B. 直接ランキングモデル（全7車をメンバー構成込み特徴量でスコアし最上位1車を選ぶ）")
    print("=" * 96)
    Xtr2, ytr2 = [], []
    for r in train:
        for f in r["by_frame"]:
            Xtr2.append(rider_features(r, f))
            ytr2.append(1 if f in r["top3"] else 0)
    Xtr2 = np.array(Xtr2, dtype=float)
    ytr2 = np.array(ytr2)
    print(f"  学習データ: {Xtr2.shape[0]}行（正例{int(ytr2.sum())}）", flush=True)
    mr = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=500,
                             learning_rate=0.05, num_leaves=63, min_child_samples=80,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    mr.fit(Xtr2, ytr2)
    imp2 = sorted(zip(RIDER_FEATS, mr.feature_importances_), key=lambda t: -t[1])
    print(f"  重要度上位10: " + ", ".join(f"{f}({v})" for f, v in imp2[:10]))

    print(f"\n    {'手法':<28}{'TRAIN精度':>11}{'TEST精度':>11}")
    print(f"    {'現行t1固定':<28}{base_tr:>10.2f}%{base_te:>10.2f}%")
    for label, data in (("", None),):
        pass
    res = []
    for data in (train, test):
        ok = 0
        for r in data:
            frames = list(r["by_frame"].keys())
            F = np.array([rider_features(r, f) for f in frames], dtype=float)
            s = mr.predict_proba(F)[:, 1]
            pick = frames[int(np.argmax(s))]
            if pick in r["top3"]:
                ok += 1
        res.append(ok / len(data) * 100)
    flag = " ★改善" if res[1] > base_te else ""
    print(f"    {'新ランキングモデル最上位':<28}{res[0]:>10.2f}%{res[1]:>10.2f}%{flag}")

    # ===== C. t1 vs t2 ペアワイズ =====
    print("\n" + "=" * 96)
    print("C. ペアワイズ比較モデル（t1とt2のどちらが3着内に入りやすいかを直接学習）")
    print("   ※ 両方入る/両方外れるケースは学習から除外し、優劣が決まるケースのみで学習")
    print("=" * 96)
    Xp, yp = [], []
    for r in train:
        t1, t2 = tk(r, 0), tk(r, 1)
        h1, h2 = t1 in r["top3"], t2 in r["top3"]
        if h1 == h2:
            continue
        f1 = rider_features(r, t1)
        f2 = rider_features(r, t2)
        Xp.append([a - b for a, b in zip(f1, f2)])
        yp.append(1 if h1 else 0)   # 1 = t1の方が良い
    Xp = np.array(Xp, dtype=float)
    yp = np.array(yp)
    print(f"  学習データ: {Xp.shape[0]}レース（t1優位{int(yp.sum())} / t2優位{len(yp)-int(yp.sum())}）")
    mp = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=400,
                             learning_rate=0.05, num_leaves=31, min_child_samples=60,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    mp.fit(Xp, yp)

    print(f"\n    {'閾値(t1採用確率)':<20}{'TRAIN精度':>11}{'TEST精度':>11}{'TEST切替数':>12}")
    for th in (0.30, 0.40, 0.50, 0.60):
        res = []
        for data in (train, test):
            ok = sw = 0
            for r in data:
                t1, t2 = tk(r, 0), tk(r, 1)
                d = np.array([[a - b for a, b in zip(rider_features(r, t1),
                                                      rider_features(r, t2))]], dtype=float)
                p1 = float(mp.predict_proba(d)[0, 1])
                pick = t1 if p1 >= th else t2
                if pick == t2:
                    sw += 1
                if pick in r["top3"]:
                    ok += 1
            res.append((ok / len(data) * 100, sw))
        (a1, _), (a2, s2) = res
        flag = " ★改善" if a2 > base_te else ""
        print(f"    {th:<20}{a1:>10.2f}%{a2:>10.2f}%{s2:>12}{flag}")


if __name__ == "__main__":
    main()
