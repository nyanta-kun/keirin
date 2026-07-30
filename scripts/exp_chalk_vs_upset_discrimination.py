"""【硬いレース vs 波乱レース】の事前読み分け可否検証（2026-07-30）。

[[keirin_dominance_pattern_verification_2026_07_30]]の続き。ユーザー方針:
「ROI検証は時期尚早。まずは硬いレース・波乱レースの読み分けができるかの検証」。

評価指標はROIではなく**判別力(AUC)とリフト**で測る:
  - AUC: 発走前情報だけで「配当が高い/低い」をランク付けできるか
  - リフト: 上位X%を選んだとき実際の30倍以上率がベースラインからどれだけ上がるか
    （これが実務上の「絞り込み効果」の直接的な尺度）

前提（クリーン）: `wt_entries.pred_win_pct`/`pred_top3_pct`は2026-07-30に月次凍結
vintageモデルで全期間クリーン再計算済み（[[keirin_wt_foundational_audit_2026_07_29]]）。
DB格納値をそのまま使う。

**オッズは特徴量に使わない**（朝イチ入稿時点では最終オッズが確定していないため、
実運用可能な条件で検証する。オッズを使えば当然当たるが実用にならない）。

honest分割: TRAIN 2024-01-01〜2025-12-31 で学習・TEST 2026-01-01〜2026-07-30 で評価。
LightGBM分類器はTRAINのみで学習しTESTには一切触れない。
"""
import math
import re
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

CHALK_MAX = 10.0    # これ未満を「硬い」
UPSET_MIN = 30.0    # これ以上を「波乱」


def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def load_races():
    print("[load] races/venue ...", flush=True)
    with get_connection() as c:
        rows = c.execute(
            "SELECT r.race_key, r.race_date, r.grade, r.start_at, r.distance, "
            "       v.bank_length, v.is_indoor "
            "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
            "WHERE r.n_entries = 7 AND r.cancel = 0 "
            "  AND r.race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    out = {}
    for r in rows:
        st = r["start_at"]
        is_night = 0
        if st is not None:
            try:
                dt = datetime.fromtimestamp(int(st), tz=timezone.utc).astimezone(JST)
                is_night = 1 if dt.hour >= 17 else 0
            except (TypeError, ValueError):
                pass
        out[r["race_key"]] = {
            "race_date": str(r["race_date"]), "grade": r["grade"],
            "bank_length": r["bank_length"], "is_indoor": r["is_indoor"],
            "distance": r["distance"], "is_night": is_night,
        }
    print(f"[load]   races: {len(out)}", flush=True)
    return out


def load_entries(race_keys):
    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_win_pct, pred_top3_pct, prediction_mark, "
                 "       race_point, line_group, line_size, n_lines, finish_order, style "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load]   races with entries: {len(by_race)}", flush=True)
    return by_race


def load_trio_win_odds(race_keys, winners_by_race):
    print("[load] trio odds ...", flush=True)
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            boards = defaultdict(dict)
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
                    boards[rk][parts] = fv
            for rk in chunk:
                w = winners_by_race.get(rk)
                if w is None:
                    continue
                odds = boards.get(rk, {}).get(w)
                if odds is not None:
                    out[rk] = odds
            if (i // 900) % 20 == 0:
                print(f"[load]   trio progress: {i}/{len(race_keys)}", flush=True)
    print(f"[load]   races with win odds: {len(out)}", flush=True)
    return out


FEATURES = [
    "win_max", "win_2nd", "win_3rd", "win_gap12", "win_gap23", "win_gap34",
    "win_sum_top2", "win_sum_top3", "win_entropy",
    "top3_max", "top3_2nd", "top3_3rd", "top3_gap12", "top3_gap23", "top3_gap34",
    "top3_sum_top2", "top3_sum_top3", "top3_entropy",
    "mark_top3_sum", "mark_win_sum", "honmei_is_w1", "taikou_is_w2", "mark_same_line",
    "rp_max", "rp_gap12", "rp_std",
    "n_lines", "max_line_size", "n_solo",
    "grade_enc", "bank_length", "is_indoor", "is_night", "distance",
    "n_senko",
]

GRADE_MAP = {"L級": 0, "A級": 1, "S級": 2, "SA混合": 3}


def build_rows(races, entries_by_race):
    print("[build] features ...", flush=True)
    out = []
    winners_by_race = {}
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_win_pct"] is None or e["pred_top3_pct"] is None for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        winners_by_race[rk] = winners

        wv = sorted((float(e["pred_win_pct"]) for e in ents), reverse=True)
        tv = sorted((float(e["pred_top3_pct"]) for e in ents), reverse=True)
        by_frame = {int(e["frame_no"]): e for e in ents}
        w1_frame = max(by_frame, key=lambda f: float(by_frame[f]["pred_win_pct"]))
        w2_frame = sorted(by_frame, key=lambda f: -float(by_frame[f]["pred_win_pct"]))[1]

        honmei = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 2), None)
        mark_top3_sum = mark_win_sum = 0.0
        if honmei is not None and taikou is not None:
            mark_top3_sum = (float(by_frame[honmei]["pred_top3_pct"])
                             + float(by_frame[taikou]["pred_top3_pct"]))
            mark_win_sum = (float(by_frame[honmei]["pred_win_pct"])
                            + float(by_frame[taikou]["pred_win_pct"]))
        mark_same_line = 0
        if honmei is not None and taikou is not None:
            lg_h, lg_t = by_frame[honmei]["line_group"], by_frame[taikou]["line_group"]
            mark_same_line = 1 if (lg_h is not None and lg_h == lg_t) else 0

        rps = [float(e["race_point"]) for e in ents if e["race_point"] is not None]
        rps.sort(reverse=True)
        rp_max = rps[0] if rps else 0.0
        rp_gap12 = (rps[0] - rps[1]) if len(rps) >= 2 else 0.0
        rp_std = float(np.std(rps)) if len(rps) >= 2 else 0.0

        line_sizes = defaultdict(int)
        for e in ents:
            if e["line_group"] is not None:
                line_sizes[e["line_group"]] += 1
        max_line_size = max(line_sizes.values()) if line_sizes else 0
        n_solo = sum(1 for v in line_sizes.values() if v == 1)
        n_senko = sum(1 for e in ents if e["style"] == "逃")

        row = {
            "race_key": rk, "race_date": meta["race_date"],
            "win_max": wv[0], "win_2nd": wv[1], "win_3rd": wv[2],
            "win_gap12": wv[0] - wv[1], "win_gap23": wv[1] - wv[2], "win_gap34": wv[2] - wv[3],
            "win_sum_top2": wv[0] + wv[1], "win_sum_top3": wv[0] + wv[1] + wv[2],
            "win_entropy": _entropy(wv),
            "top3_max": tv[0], "top3_2nd": tv[1], "top3_3rd": tv[2],
            "top3_gap12": tv[0] - tv[1], "top3_gap23": tv[1] - tv[2], "top3_gap34": tv[2] - tv[3],
            "top3_sum_top2": tv[0] + tv[1], "top3_sum_top3": tv[0] + tv[1] + tv[2],
            "top3_entropy": _entropy(tv),
            "mark_top3_sum": mark_top3_sum, "mark_win_sum": mark_win_sum,
            "honmei_is_w1": 1 if honmei == w1_frame else 0,
            "taikou_is_w2": 1 if taikou == w2_frame else 0,
            "mark_same_line": mark_same_line,
            "rp_max": rp_max, "rp_gap12": rp_gap12, "rp_std": rp_std,
            "n_lines": float(ents[0]["n_lines"] or 0),
            "max_line_size": float(max_line_size), "n_solo": float(n_solo),
            "grade_enc": float(GRADE_MAP.get(meta["grade"], -1)),
            "bank_length": float(meta["bank_length"] or 0),
            "is_indoor": float(meta["is_indoor"] or 0),
            "is_night": float(meta["is_night"]),
            "distance": float(meta["distance"] or 0),
            "n_senko": float(n_senko),
        }
        out.append(row)
    print(f"[build]   rows: {len(out)}", flush=True)
    return out, winners_by_race


def single_feature_aucs(rows, target_key):
    res = []
    y = np.array([r[target_key] for r in rows])
    if len(set(y.tolist())) < 2:
        return res
    for f in FEATURES:
        x = np.array([r[f] for r in rows], dtype=float)
        if np.all(x == x[0]):
            continue
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            continue
        res.append((f, auc))
    res.sort(key=lambda t: -abs(t[1] - 0.5))
    return res


def lift_table(y_true, scores, label, quantiles=(0.05, 0.10, 0.20, 0.30, 0.50)):
    n = len(y_true)
    base = sum(y_true) / n * 100
    order = np.argsort(-np.asarray(scores))
    print(f"    ベースレート({label})={base:.1f}%  n={n}")
    print(f"    {'上位':<8}{'n':>7}{'実際の率':>12}{'リフト':>9}")
    for q in quantiles:
        k = max(1, int(n * q))
        idx = order[:k]
        rate = sum(y_true[i] for i in idx) / k * 100
        lift = rate / base if base else 0
        print(f"    {f'{int(q*100)}%':<8}{k:>7}{rate:>11.1f}%{lift:>8.2f}x")


def main():
    races = load_races()
    race_keys = list(races.keys())
    entries = load_entries(race_keys)
    rows, winners = build_rows(races, entries)
    odds_map = load_trio_win_odds([r["race_key"] for r in rows], winners)

    rows = [r for r in rows if r["race_key"] in odds_map]
    for r in rows:
        o = odds_map[r["race_key"]]
        r["payout"] = o
        r["is_chalk"] = 1 if o < CHALK_MAX else 0
        r["is_upset"] = 1 if o >= UPSET_MIN else 0

    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] 有効レース: {len(rows)}  TRAIN={len(train)} TEST={len(test)}")
    print(f"[main] 硬い(<{CHALK_MAX}倍)率: TRAIN={sum(r['is_chalk'] for r in train)/len(train)*100:.1f}% "
          f"TEST={sum(r['is_chalk'] for r in test)/len(test)*100:.1f}%")
    print(f"[main] 波乱(>={UPSET_MIN}倍)率: TRAIN={sum(r['is_upset'] for r in train)/len(train)*100:.1f}% "
          f"TEST={sum(r['is_upset'] for r in test)/len(test)*100:.1f}%")

    # ===== 単一特徴量AUC（解釈用） =====
    for target, tlabel in (("is_upset", f"波乱(>={UPSET_MIN}倍)"), ("is_chalk", f"硬い(<{CHALK_MAX}倍)")):
        print(f"\n{'='*80}\n単一特徴量AUC（TRAIN・{tlabel}判別）上位15\n{'='*80}")
        aucs = single_feature_aucs(train, target)
        for f, auc in aucs[:15]:
            print(f"  {f:<20} AUC={auc:.4f}")

    # ===== LightGBM分類器（TRAIN学習→TEST評価） =====
    import lightgbm as lgb
    Xtr = np.array([[r[f] for f in FEATURES] for r in train], dtype=float)
    Xte = np.array([[r[f] for f in FEATURES] for r in test], dtype=float)

    for target, tlabel in (("is_upset", f"波乱(>={UPSET_MIN}倍)"), ("is_chalk", f"硬い(<{CHALK_MAX}倍)")):
        print(f"\n{'='*80}\nLightGBM判別器: {tlabel}  (TRAIN学習→TEST評価)\n{'='*80}")
        ytr = np.array([r[target] for r in train])
        yte = np.array([r[target] for r in test])
        m = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=400,
                                learning_rate=0.05, num_leaves=31, min_child_samples=40,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                verbose=-1)
        m.fit(Xtr, ytr)
        ptr = m.predict_proba(Xtr)[:, 1]
        pte = m.predict_proba(Xte)[:, 1]
        print(f"  AUC: TRAIN={roc_auc_score(ytr, ptr):.4f}  **TEST={roc_auc_score(yte, pte):.4f}**")
        print(f"\n  [TEST] リフト表（判別器スコア上位X%を選んだ場合）")
        lift_table(yte.tolist(), pte, tlabel)

        imp = sorted(zip(FEATURES, m.feature_importances_), key=lambda t: -t[1])
        print(f"\n  特徴量重要度 上位10: " + ", ".join(f"{f}({v})" for f, v in imp[:10]))

    # ===== 参考: 実際の配当中央値がスコア帯でどう動くか =====
    print(f"\n{'='*80}\n参考: 波乱判別器スコア十分位別の実配当（TEST）\n{'='*80}")
    ytr = np.array([r["is_upset"] for r in train])
    m = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=400,
                            learning_rate=0.05, num_leaves=31, min_child_samples=40,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(Xtr, ytr)
    pte = m.predict_proba(Xte)[:, 1]
    order = np.argsort(-pte)
    n = len(test)
    print(f"  {'十分位':<10}{'n':>7}{'中央値':>10}{'平均':>10}{'30倍+率':>10}{'5倍未満率':>12}")
    for d in range(10):
        lo, hi = n * d // 10, n * (d + 1) // 10
        idx = order[lo:hi]
        pays = sorted(test[i]["payout"] for i in idx)
        k = len(pays)
        med = pays[k // 2]
        mean = sum(pays) / k
        over30 = sum(1 for p in pays if p >= 30) / k * 100
        under5 = sum(1 for p in pays if p < 5) / k * 100
        print(f"  d{d+1:<9}{k:>7}{med:>9.1f}倍{mean:>9.1f}倍{over30:>9.1f}%{under5:>11.1f}%")


if __name__ == "__main__":
    main()
