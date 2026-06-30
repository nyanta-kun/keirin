"""7+車 硬いレース除外 & 流し馬スコアフィルタ 検証

提案1: 合成オッズ / 最安目オッズ の閾値でフィルタ
  - 現行: 最安目 gami >= 5.0
  - 検証: gami >= 7.0, 10.0, 15.0
  - 合成オッズ(synth) = 1 / Σ(1/odds_i) でのフィルタ比較

提案2: 流し馬の競走スコア(race_point)フィルタ
  - thirds の中で pivot2 の race_point を大幅に下回る選手を除外
  - 閾値: -5pt, -10pt, -15pt, -20pt 差

設計: train=2023-07〜2026-02 / test=2026-03〜最新
     7+車・gap12>=0.10・最終オッズ上限値（7+車Sランク相当）
     バイアス対策: finish_order>=1 で完走判定・n_entriesはwt_racesで確認
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
from collections import defaultdict

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.database import get_connection

TRAIN_FROM = "2023-07-01"
TRAIN_TO   = "2026-02-28"
TEST_FROM  = "2026-03-01"
TEST_TO    = "2026-06-30"
GAP12_MIN  = 0.10  # 7+車Sランク
GAMI_BASE  = 5.0   # 現行最安目閾値


def roi_summary(pays, bets):
    n = len(pays)
    hits = sum(1 for p in pays if p > 0)
    tb = sum(bets)
    tp = sum(pays)
    roi = tp / tb if tb > 0 else 0
    import numpy as np
    # 95% CI (bootstrap相当の近似: Wilson on hit/miss)
    hit_rate = hits / n if n > 0 else 0
    z = 1.96
    ci_lo = max(0, hit_rate - z * (hit_rate * (1 - hit_rate) / n) ** 0.5) if n > 0 else 0
    ci_hi = hit_rate + z * (hit_rate * (1 - hit_rate) / n) ** 0.5 if n > 0 else 0
    return {
        "n": n, "hits": hits, "roi": roi, "hit_rate": hit_rate,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def load_trio_board(race_keys):
    """race_key -> {frozenset: odds_value}"""
    board = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is not None:
                    try:
                        parts = re.split(r"[-=]", str(comb))
                        key = frozenset(int(x) for x in parts)
                        board[rk][key] = float(od)
                    except ValueError:
                        pass
    return board


def load_n_entries_map(race_keys):
    """race_key -> n_entries（出走表ベース・欠車含む）"""
    m = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, n_entries FROM wt_races "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, ne in c.execute(q, chunk):
                m[rk] = ne
    return m


def collect(model, f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    # 7+車のみ（出走表n_entries基準）
    rks = df["race_key"].unique().tolist()
    ne_map = load_n_entries_map(rks)
    valid_rks = {rk for rk, ne in ne_map.items() if ne is not None and int(ne) >= 7}
    df = df[df["race_key"].isin(valid_rks)].copy()
    # 完走者のみ（欠車除外）
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    trio_board = load_trio_board(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        ne = ne_map.get(rk, 0)
        if ne < 7:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 < GAP12_MIN:
            continue

        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3 = frozenset(fin["frame_no"].astype(int).tolist())

        frames = g["frame_no"].astype(int).tolist()
        pivot1, pivot2 = frames[0], frames[1]
        thirds = frames[2:]

        rp_list = g["race_point"].tolist()  # pred_prob降順
        rp_p1 = rp_list[0] if len(rp_list) > 0 else 0.0
        rp_p2 = rp_list[1] if len(rp_list) > 1 else 0.0

        # 流し馬の race_point リスト
        rp_thirds = [(frames[i], rp_list[i]) for i in range(2, n)]

        bd = trio_board.get(rk, {})
        # 流し馬ごとの三連複オッズ
        combo_odds = {}
        for fn in thirds:
            key = frozenset({pivot1, pivot2, fn})
            if key in bd:
                combo_odds[fn] = bd[key]

        if not combo_odds:
            continue

        rows.append({
            "race_key": rk,
            "gap12": gap12,
            "pivot1": pivot1,
            "pivot2": pivot2,
            "thirds": thirds,
            "top3": top3,
            "combo_odds": combo_odds,  # {frame_no: odds}
            "rp_p1": float(rp_p1) if rp_p1 == rp_p1 else 0.0,
            "rp_p2": float(rp_p2) if rp_p2 == rp_p2 else 0.0,
            "rp_thirds": [(fn, float(rp) if rp == rp else 0.0) for fn, rp in rp_thirds],
        })
    return rows


def eval_strategy(rows, gami_thresh=5.0, synth_thresh=None, rp_gap_thresh=None):
    """
    rows: collect() の結果
    gami_thresh: 最安目オッズの最低閾値（これ未満のレースはスキップ）
    synth_thresh: 合成オッズの最低閾値（None=無効）
    rp_gap_thresh: rp_p2 - rp_third の差がこれ以上の流し馬を除外（None=無効・正の値で除外）
    """
    pays = []
    bets = []
    for r in rows:
        filtered_thirds = list(r["thirds"])

        # race_point フィルタ（rp_p2より rp_gap_thresh 以上低い流し馬を除外）
        if rp_gap_thresh is not None:
            rp_p2 = r["rp_p2"]
            rp_map = dict(r["rp_thirds"])  # {fn: rp_value}
            keep = []
            for fn in filtered_thirds:
                if fn not in rp_map:
                    keep.append(fn)  # rp不明は残す
                    continue
                diff = rp_p2 - rp_map[fn]  # 正 = 流し馬の方が低スコア
                if diff < rp_gap_thresh:  # 差が閾値未満 → 残す
                    keep.append(fn)
            filtered_thirds = keep

        # combo_odds が取得できた流し馬のみ
        valid_thirds = [t for t in filtered_thirds if t in r["combo_odds"]]
        if not valid_thirds:
            continue

        combo_odds = {t: r["combo_odds"][t] for t in valid_thirds}

        # 最安目フィルタ
        min_odds = min(combo_odds.values())
        if min_odds < gami_thresh:
            continue

        # 合成オッズフィルタ
        if synth_thresh is not None:
            synth = 1.0 / sum(1.0 / o for o in combo_odds.values())
            if synth < synth_thresh:
                continue

        # 的中判定
        top3 = r["top3"]
        hit_pay = 0
        for t in valid_thirds:
            if frozenset({r["pivot1"], r["pivot2"], t}) == top3:
                hit_pay = int(combo_odds[t] * 100)
                break

        n_pts = len(valid_thirds)
        pays.append(hit_pay)
        bets.append(n_pts * 100)

    return roi_summary(pays, bets)


def main():
    print("モデルロード中...", flush=True)
    model = load_model("lgbm_wt")

    print(f"データ収集中 train={TRAIN_FROM}〜{TRAIN_TO}...", flush=True)
    tr = collect(model, TRAIN_FROM, TRAIN_TO)
    print(f"  → train: {len(tr)}R", flush=True)

    print(f"データ収集中 test={TEST_FROM}〜{TEST_TO}...", flush=True)
    te = collect(model, TEST_FROM, TEST_TO)
    print(f"  → test: {len(te)}R", flush=True)

    print()
    print("=" * 80)
    print("【実験1: 最安目・合成オッズ 閾値フィルタ】（gap12≥0.10 7+車Sランク）")
    print(f"{'戦略':<28} {'TR_ROI':>7} {'TR_n':>6} {'TR_hit':>7} | "
          f"{'TE_ROI':>7} {'TE_n':>6} {'TE_hit':>7} {'TE_CI':>18}")
    print("-" * 80)

    scenarios_1 = [
        ("現行: gami≥5.0 (ALL)",      5.0,  None, None),
        ("gami≥7.0",                   7.0,  None, None),
        ("gami≥10.0",                 10.0,  None, None),
        ("gami≥15.0",                 15.0,  None, None),
        ("synth≥5.0 (合成)",           5.0,   5.0, None),
        ("synth≥8.0 (合成)",           5.0,   8.0, None),
        ("synth≥10.0 (合成)",          5.0,  10.0, None),
        ("synth≥15.0 (合成)",          5.0,  15.0, None),
    ]

    for label, gami, synth, _ in scenarios_1:
        s_tr = eval_strategy(tr, gami_thresh=gami, synth_thresh=synth)
        s_te = eval_strategy(te, gami_thresh=gami, synth_thresh=synth)
        print(
            f"{label:<28} {s_tr['roi']:>6.1%} {s_tr['n']:>6} {s_tr['hit_rate']:>6.1%} | "
            f"{s_te['roi']:>6.1%} {s_te['n']:>6} {s_te['hit_rate']:>6.1%} "
            f"[{s_te['ci_lo']:>4.1%},{s_te['ci_hi']:>5.1%}]"
        )

    print()
    print("=" * 80)
    print("【実験2: 流し馬 race_point フィルタ】（gami≥5.0 + gap12≥0.10 前提）")
    print(f"{'戦略':<32} {'TR_ROI':>7} {'TR_n':>6} {'TR_hit':>7} | "
          f"{'TE_ROI':>7} {'TE_n':>6} {'TE_hit':>7} {'TE_CI':>18}")
    print("-" * 80)

    scenarios_2 = [
        ("現行: フィルタなし",    None),
        ("rp差>=5pt除外",          5.0),
        ("rp差>=10pt除外",        10.0),
        ("rp差>=15pt除外",        15.0),
        ("rp差>=20pt除外",        20.0),
        ("rp差>=25pt除外",        25.0),
    ]

    for label, rp_gap in scenarios_2:
        s_tr = eval_strategy(tr, gami_thresh=GAMI_BASE, rp_gap_thresh=rp_gap)
        s_te = eval_strategy(te, gami_thresh=GAMI_BASE, rp_gap_thresh=rp_gap)

        # 点数削減効果
        def avg_pts(rows, rp_gap_thresh):
            pts = []
            for r in rows:
                ft = list(r["thirds"])
                if rp_gap_thresh is not None:
                    rp_p2 = r["rp_p2"]
                    ft = [fn for fn in ft if fn in r["combo_odds"]]
                    ft = [fn for fn in ft
                          if rp_p2 - dict(r["rp_thirds"]).get(fn, 0.0) < rp_gap_thresh]
                else:
                    ft = [fn for fn in ft if fn in r["combo_odds"]]
                v = [r["combo_odds"][t] for t in ft if t in r["combo_odds"]]
                if not v or min(v) < GAMI_BASE:
                    continue
                pts.append(len(ft))
            return sum(pts) / len(pts) if pts else 0

        avg_tr = avg_pts(tr, rp_gap)
        avg_te = avg_pts(te, rp_gap)
        print(
            f"{label:<32} {s_tr['roi']:>6.1%} {s_tr['n']:>6} {s_tr['hit_rate']:>6.1%} | "
            f"{s_te['roi']:>6.1%} {s_te['n']:>6} {s_te['hit_rate']:>6.1%} "
            f"[{s_te['ci_lo']:>4.1%},{s_te['ci_hi']:>5.1%}]"
            f"  avg_pts(te)={avg_te:.1f}"
        )

    # 追加: 市場が堅いと見て荒れたケース分析（除外損失分析）
    print()
    print("=" * 80)
    print("【参考: gami閾値で除外されたレースの実際の払い戻し分布】")
    print("（市場が堅いと見て実際に荒れた＝高配当を見逃したケース）")
    print("-" * 80)
    for thresh in [5.0, 7.0, 10.0, 15.0]:
        # thresh未満で除外されたレースの「もし買っていた場合の払い戻し」
        excluded_pays = []
        excluded_bets = []
        for r in te:
            valid_thirds = [t for t in r["thirds"] if t in r["combo_odds"]]
            if not valid_thirds:
                continue
            combo_odds = {t: r["combo_odds"][t] for t in valid_thirds}
            min_odds = min(combo_odds.values())
            if min_odds >= thresh:
                continue  # 除外対象でない
            # 除外されたレース: もし全点買っていたら
            top3 = r["top3"]
            hit_pay = 0
            for t in valid_thirds:
                if frozenset({r["pivot1"], r["pivot2"], t}) == top3:
                    hit_pay = int(combo_odds[t] * 100)
                    break
            excluded_pays.append(hit_pay)
            excluded_bets.append(len(valid_thirds) * 100)

        s = roi_summary(excluded_pays, excluded_bets)
        if s["n"] > 0:
            high_pay = sum(1 for p in excluded_pays if p >= 2000)
            very_high = sum(1 for p in excluded_pays if p >= 5000)
            print(
                f"  gami<{thresh:.0f}で除外(test): n={s['n']:>5} ROI={s['roi']:>5.1%} "
                f"hit={s['hit_rate']:>5.1%} "
                f"(配当≥2000円:{high_pay}件 ≥5000円:{very_high}件)"
            )


if __name__ == "__main__":
    main()
