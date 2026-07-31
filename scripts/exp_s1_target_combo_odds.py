"""⚠️ 【2026-07-31・警告】本スクリプトは src/wt_vintage_config.py へ統合される前の
独自 QUARTERS 定義を持ち、参照する四半期 vintage モデル（lgbm_wt_eval_q*）は
2026-07-28 の事故で汚染され 2026-07-31 に削除済みのため、現状では実行できない
（load_model が失敗する）。再利用する場合は wt_vintage_config.monthly_windows()
による月次凍結モデル体系へ移植すること。過去の出力数値は信用しないこと。

S1: 万車券(三連単配当10,000円以上)を積極的に狙うため、購入する三連単2点
（軸→p1→p2 / 軸→p2→p1）の「発走前オッズ」自体をレース選定基準に使えないか
検証する。

ユーザー依頼(2026-07-25、再依頼): 「万車券を狙って取ることを目標とし、万車券が
発生し、低配当が発生しづらいレースの選定をすることで的中率・ROIの向上を検討する」

前回(exp_s1_manshaken_dependency_filter.py)の誤り: 場・グレード等のセグメント別に
過去の万車券発生"実績"を遡及集計して依存度を測ったが、万車券が数百件に1件しか
出ない現状のサンプル数では、各セグメントの実績が1〜2件の偶然に支配され、
train+valで好成績→testで崩壊(多重比較ノイズ)という結果に終わった。

本スクリプトは発想を変え、**発走前に確定しているオッズ情報**（購入する三連単
2点の発走前オッズ）を直接の選定基準にする。これはレースごとの偶然の的中実績
ではなく、レース時点で確定的に観測できる値なので、前回のような遡及ノイズの
問題が構造的に発生しない。

母集団: 現行本番ゲート(top3_gap>=0.15 AND 軸勝率<=0.50)通過の全候補
（的中+非的中）。購入2点の発走前三連単オッズ(min/avg/max)を算出し、
オッズ帯別に的中率・ROI・万車券比率を集計する。

正規プロトコル: train+val(〜2026-03-31)でオッズ帯の傾向を探索し、
test(2026-04-01〜)で一度だけ評価する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s1_target_combo_odds.py
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_PATH = Path("/tmp/exp_s1_target_combo_odds_cache.pkl")

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import S1W_STAKE, s1w_gate, s1w_select

QUARTERS = [
    ("2024-01-01", "2024-03-31", "lgbm_wt_eval_q2401", "lgbm_wt_win_q2401"),
    ("2024-04-01", "2024-06-30", "lgbm_wt_eval_q2404", "lgbm_wt_win_q2404"),
    ("2024-07-01", "2024-09-30", "lgbm_wt_eval_q2407", "lgbm_wt_win_q2407"),
    ("2024-10-01", "2024-12-31", "lgbm_wt_eval_q2410", "lgbm_wt_win_q2410"),
    ("2025-01-01", "2025-03-31", "lgbm_wt_eval_q2501", "lgbm_wt_win_q2501"),
    ("2025-04-01", "2025-06-30", "lgbm_wt_eval_q2504", "lgbm_wt_win_q2504"),
    ("2025-07-01", "2025-09-30", "lgbm_wt_eval_q2507", "lgbm_wt_win_q2507"),
    ("2025-10-01", "2025-12-31", "lgbm_wt_eval_w3", "lgbm_wt_win_w3"),
    ("2026-01-01", "2026-04-12", "lgbm_wt_eval_w2", "lgbm_wt_win_w2"),
    ("2026-04-13", "2026-07-22", "lgbm_wt_eval", "lgbm_wt_win_eval"),
]

TRAIN_VAL_END = "2026-03-31"
MANSHAKEN_MIN = 10000


def _load_trifecta_boards(race_keys):
    tri = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trifecta' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = tuple(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    tri[rk][parts] = fv
    return tri


def collect(eval_model_name, win_model_name, date_from, date_to):
    model = load_model(eval_model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        meta_map = dict(c.execute(
            "SELECT race_key, race_date || '|' || race_no || '|' || venue_id || '|' || "
            "COALESCE(grade,'') FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins: dict[str, list[tuple[int, int]]] = {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    tri_bd = _load_trifecta_boards(df["race_key"].unique().tolist())
    pm = _load_payouts_wt(df["race_key"].unique().tolist())

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        tri = tri_bd.get(rk)
        if not tri:
            continue
        board: set[int] = set()
        for k in tri:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = s1w_select(win_probs, top3_probs)
        if sel is None:
            continue
        axis, p1, p2, top3_gap = sel
        # 現行本番ゲート（top3_gap>=0.15 AND 軸勝率<=0.50）を母集団の基準にする
        if not s1w_gate(top3_gap, win_probs[axis]):
            continue
        if axis not in board or p1 not in board or p2 not in board:
            continue
        combo_a, combo_b = (axis, p1, p2), (axis, p2, p1)
        odds_a = tri.get(combo_a)
        odds_b = tri.get(combo_b)
        buy = [c for c, o in ((combo_a, odds_a), (combo_b, odds_b)) if o is not None]
        if not buy:
            continue
        odds_list = [o for o in (odds_a, odds_b) if o is not None]

        order3 = tuple(fno for _, fno in fin[:3])
        hit = order3 in buy
        trifecta_pay = pm.get(rk, {}).get(("trifecta", order3), 0) if hit else 0
        pay = trifecta_pay * S1W_STAKE // 100 if hit else 0
        bet = len(buy) * S1W_STAKE

        meta = meta_map.get(rk, "|||")
        race_date, race_no, venue_id, grade = (meta.split("|") + [""] * 4)[:4]

        races.append({
            "race_key": rk, "race_date": race_date, "race_no": race_no,
            "venue_id": venue_id, "grade": grade,
            "top3_gap": top3_gap, "axis_win_prob": win_probs[axis],
            "odds_a": odds_a, "odds_b": odds_b,
            "odds_min": min(odds_list), "odds_max": max(odds_list),
            "odds_avg": sum(odds_list) / len(odds_list),
            "bet": bet, "pay": pay, "hit": hit, "trifecta_pay": trifecta_pay,
        })
    return races


def _stats(rows):
    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    bet = sum(r["bet"] for r in rows)
    pay = sum(r["pay"] for r in rows)
    man = sum(1 for r in rows if r["hit"] and r["trifecta_pay"] >= MANSHAKEN_MIN)
    roi = pay / bet * 100 if bet else 0.0
    hit_rate = hits / n * 100 if n else 0.0
    man_rate_of_hits = man / hits * 100 if hits else 0.0
    return {
        "n": n, "hits": hits, "hit_rate": hit_rate, "bet": bet, "pay": pay,
        "roi": roi, "man": man, "man_rate_of_hits": man_rate_of_hits,
    }


def _bucket_recall(rows, all_rows, thresh, label):
    base = [r for r in all_rows if r["trifecta_pay"] >= thresh]
    kept = [r for r in rows if r["trifecta_pay"] >= thresh]
    n_base = len(base)
    n_kept = len(kept)
    recall = n_kept / n_base * 100 if n_base else 0.0
    return f"{label}: {n_kept}/{n_base}残存({recall:.1f}%)"


def _odds_bucket(rows, edges, key="odds_avg"):
    buckets = defaultdict(list)
    for r in rows:
        v = r[key]
        lo = None
        for e in edges:
            if v >= e:
                lo = e
            else:
                break
        buckets[lo].append(r)
    return buckets


EDGES = [0, 10, 20, 30, 50, 75, 100, 150, 200]


def _print_bucket_table(rows, label):
    buckets = _odds_bucket(rows, EDGES)
    print(f"\n--- 購入2点オッズ帯別（{label}）---")
    print(f"{'odds帯':<12}{'n':>7}{'hit%':>8}{'ROI':>9}{'万車券数':>9}{'万車券/的中%':>13}")
    keys = sorted(buckets.keys(), key=lambda k: (k is None, k))
    for i, k in enumerate(keys):
        sub = buckets[k]
        s = _stats(sub)
        hi = EDGES[EDGES.index(k) + 1] if k in EDGES and EDGES.index(k) + 1 < len(EDGES) else "+"
        seg_label = f"{k}〜{hi}倍" if hi != "+" else f"{k}倍+"
        print(f"{seg_label:<12}{s['n']:>7}{s['hit_rate']:>7.1f}%{s['roi']:>8.1f}%"
              f"{s['man']:>9}{s['man_rate_of_hits']:>12.1f}%")
    return buckets


def main():
    if CACHE_PATH.exists():
        print(f"[cache] {CACHE_PATH} からロード", flush=True)
        with open(CACHE_PATH, "rb") as f:
            all_races = pickle.load(f)
    else:
        all_races = []
        for date_from, date_to, eval_model, win_model in QUARTERS:
            print(f"[collect] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
            rs = collect(eval_model, win_model, date_from, date_to)
            print(f"[collect]   {len(rs)}R収集(現行ゲート通過・的中+非的中)", flush=True)
            all_races.extend(rs)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(all_races, f)
        print(f"[cache] {CACHE_PATH} に保存", flush=True)

    train_val = [r for r in all_races if r["race_date"] <= TRAIN_VAL_END]
    test = [r for r in all_races if r["race_date"] > TRAIN_VAL_END]
    print(f"\n収集完了: 現行ゲート通過 全候補 {len(all_races)}R "
          f"(train+val={len(train_val)}R / test={len(test)}R)\n")

    base = _stats(train_val)
    print("=" * 110)
    print(f"[train+val ベースライン] n={base['n']} 的中={base['hits']}({base['hit_rate']:.1f}%) "
          f"投資={base['bet']:,} 回収={base['pay']:,} ROI={base['roi']:.1f}%  "
          f"万車券={base['man']}件(的中の{base['man_rate_of_hits']:.1f}%)")
    print("=" * 110)

    _print_bucket_table(train_val, "train+val")

    print(f"\n{'='*110}\n[フィルター候補: 購入2点の発走前オッズ下限で足切り(train+valで探索)]\n{'='*110}")
    for thresh in (10, 15, 20, 25, 30, 40, 50, 75, 100):
        sub = [r for r in train_val if r["odds_min"] >= thresh]
        s = _stats(sub)
        print(f"  odds_min>={thresh:>3}倍: n={s['n']:>5}({s['n']/base['n']*100:5.1f}%)  "
              f"hit%={s['hit_rate']:5.1f}%  ROI={s['roi']:7.1f}%  "
              f"万車券={s['man']:>3}件({s['man_rate_of_hits']:5.1f}%的中中)  "
              f"{_bucket_recall(sub, train_val, MANSHAKEN_MIN, '万車券再現率')}")

    print(f"\n{'='*110}\n[★ test（一度だけ評価）]\n{'='*110}")
    test_base = _stats(test)
    print(f"[testベースライン(フィルターなし)] n={test_base['n']} 的中={test_base['hits']}"
          f"({test_base['hit_rate']:.1f}%) 投資={test_base['bet']:,} 回収={test_base['pay']:,} "
          f"ROI={test_base['roi']:.1f}%  万車券={test_base['man']}件"
          f"(的中の{test_base['man_rate_of_hits']:.1f}%)")
    _print_bucket_table(test, "test")
    for thresh in (10, 15, 20, 25, 30, 40, 50, 75, 100):
        sub = [r for r in test if r["odds_min"] >= thresh]
        s = _stats(sub)
        print(f"  odds_min>={thresh:>3}倍: n={s['n']:>5}({s['n']/test_base['n']*100:5.1f}%)  "
              f"hit%={s['hit_rate']:5.1f}%  ROI={s['roi']:7.1f}%  "
              f"万車券={s['man']:>3}件({s['man_rate_of_hits']:5.1f}%的中中)  "
              f"{_bucket_recall(sub, test, MANSHAKEN_MIN, '万車券再現率')}")


if __name__ == "__main__":
    main()
