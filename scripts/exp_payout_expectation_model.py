"""【波乱度＝配当期待度】の予測モデル構築・評価（2026-07-30）。

[[keirin_roi_validation_crisis_2026_07_29]]系の続き。ユーザー方針の再定義:
「波乱度 = レースとしての配当期待度」（予想難易度とは別概念）。

予測対象は「そのレースで実際に決まった三連複の配当（勝ち組み合わせのodds_value）」
であり、買い目選択に依存しないレース固有の量。目的変数は log(payout) の回帰。

**オッズは特徴量に使わない**（朝イチ入稿時点では最終オッズ未確定のため）。

honest分割: TRAIN 2024-01-01〜2025-12-31 で学習・TEST 2026-01-01〜2026-07-30 で評価。
LightGBM回帰器はTRAINのみで学習しTESTには一切触れない。

評価:
  1. 回帰精度: TEST R²(log payout)・実配当スケールSpearman順位相関
  2. 十分位分析: TESTを予測値で10分位、各分位の実配当中央値/平均/30倍+率/5倍未満率
  3. 単一特徴量比較: 34特徴LightGBM vs top3_sum_top2単独 vs top3_entropy単独
  4. 特徴量重要度 上位10

特徴量構築・honest分割の書き方は scripts/exp_chalk_vs_upset_discrimination.py を流用。
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
JST = timezone(timedelta(hours=9))


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


def decile_table(payouts, scores, title):
    """予測スコアで降順ソートし10分位に分割、実配当の分布を表示。"""
    n = len(payouts)
    order = np.argsort(-np.asarray(scores))
    print(f"\n  {title}")
    print(f"  {'十分位':<10}{'n':>7}{'中央値':>10}{'平均':>10}{'30倍+率':>10}{'5倍未満率':>12}")
    medians = []
    for d in range(10):
        lo, hi = n * d // 10, n * (d + 1) // 10
        idx = order[lo:hi]
        pays = sorted(payouts[i] for i in idx)
        k = len(pays)
        if k == 0:
            continue
        med = pays[k // 2]
        mean = sum(pays) / k
        over30 = sum(1 for p in pays if p >= 30) / k * 100
        under5 = sum(1 for p in pays if p < 5) / k * 100
        medians.append(med)
        print(f"  d{d+1:<9}{k:>7}{med:>9.1f}倍{mean:>9.1f}倍{over30:>9.1f}%{under5:>11.1f}%")
    # 単調性チェック（d1→d10でスコア降順=予測配当高い順としているのでd1が一番高いはず）
    is_monotone_desc = all(medians[i] >= medians[i + 1] - 1e-9 for i in range(len(medians) - 1))
    print(f"  (単調性: d1→d10で中央値が概ね減少 = {is_monotone_desc})")


def eval_model(name, ytr_log, ytr_payout, yte_log, yte_payout, ptr_log, pte_log):
    r2_tr = r2_score(ytr_log, ptr_log)
    r2_te = r2_score(yte_log, pte_log)
    sp_tr, _ = spearmanr(ytr_payout, ptr_log)
    sp_te, _ = spearmanr(yte_payout, pte_log)
    print(f"\n{'='*80}\n{name}\n{'='*80}")
    print(f"  R²(log payout):  TRAIN={r2_tr:.4f}  TEST={r2_te:.4f}")
    print(f"  Spearman順位相関(実配当): TRAIN={sp_tr:.4f}  TEST={sp_te:.4f}")
    gap = r2_tr - r2_te
    print(f"  過学習ギャップ(TRAIN-TEST R²): {gap:.4f}" + ("  ← 要注意" if gap > 0.05 else ""))
    decile_table(yte_payout, pte_log, f"[TEST] {name} 予測十分位別 実配当")
    return {"name": name, "r2_train": r2_tr, "r2_test": r2_te,
            "spearman_train": sp_tr, "spearman_test": sp_te}


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
        r["log_payout"] = math.log(o)

    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] 有効レース: {len(rows)}  TRAIN={len(train)} TEST={len(test)}")
    tr_pay = [r["payout"] for r in train]
    te_pay = [r["payout"] for r in test]
    print(f"[main] TRAIN配当: 中央値={sorted(tr_pay)[len(tr_pay)//2]:.1f}倍 "
          f"平均={sum(tr_pay)/len(tr_pay):.1f}倍 30倍+率={sum(1 for p in tr_pay if p>=30)/len(tr_pay)*100:.1f}%")
    print(f"[main] TEST配当:  中央値={sorted(te_pay)[len(te_pay)//2]:.1f}倍 "
          f"平均={sum(te_pay)/len(te_pay):.1f}倍 30倍+率={sum(1 for p in te_pay if p>=30)/len(te_pay)*100:.1f}%")

    ytr_log = np.array([r["log_payout"] for r in train])
    yte_log = np.array([r["log_payout"] for r in test])
    ytr_pay = np.array([r["payout"] for r in train])
    yte_pay = np.array([r["payout"] for r in test])

    results = []

    # ===== (3) 単一特徴量ベースライン =====
    for feat in ("top3_sum_top2", "top3_entropy"):
        xtr = np.array([r[feat] for r in train], dtype=float)
        xte = np.array([r[feat] for r in test], dtype=float)
        # 単純な単回帰(最小二乗)でlog(payout)を予測 → OOS R²で評価
        A = np.vstack([xtr, np.ones_like(xtr)]).T
        coef, *_ = np.linalg.lstsq(A, ytr_log, rcond=None)
        ptr = A @ coef
        Ate = np.vstack([xte, np.ones_like(xte)]).T
        pte = Ate @ coef
        res = eval_model(f"単一特徴量線形回帰: {feat}", ytr_log, ytr_pay, yte_log, yte_pay, ptr, pte)
        results.append(res)

    # ===== (1)(2)(4) LightGBM 34特徴量回帰 =====
    import lightgbm as lgb
    Xtr = np.array([[r[f] for f in FEATURES] for r in train], dtype=float)
    Xte = np.array([[r[f] for f in FEATURES] for r in test], dtype=float)

    m = lgb.LGBMRegressor(objective="regression", metric="l2", n_estimators=400,
                           learning_rate=0.05, num_leaves=31, min_child_samples=40,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbose=-1)
    m.fit(Xtr, ytr_log)
    ptr = m.predict(Xtr)
    pte = m.predict(Xte)
    res = eval_model("LightGBM 34特徴量回帰", ytr_log, ytr_pay, yte_log, yte_pay, ptr, pte)
    results.append(res)

    imp = sorted(zip(FEATURES, m.feature_importances_), key=lambda t: -t[1])
    print(f"\n  特徴量重要度 上位10:")
    for f, v in imp[:10]:
        print(f"    {f:<20} {v}")

    # ===== まとめ比較表 =====
    print(f"\n{'='*80}\nモデル比較まとめ\n{'='*80}")
    print(f"  {'モデル':<32}{'TRAIN R2':>10}{'TEST R2':>10}{'TRAIN Sp':>10}{'TEST Sp':>10}")
    for r in results:
        print(f"  {r['name']:<32}{r['r2_train']:>10.4f}{r['r2_test']:>10.4f}"
              f"{r['spearman_train']:>10.4f}{r['spearman_test']:>10.4f}")


if __name__ == "__main__":
    main()
