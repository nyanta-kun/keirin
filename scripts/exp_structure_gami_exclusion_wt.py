"""ガミ条件のセマンティクス比較 + 得点/指数の偏りによる朝除外検討（2026-07-09）

Sim A: gami をレース単位除外（min(全目) < thr → レースごと見送り・買い目カットなし）
       として扱った場合の thr∈{5,6,7} × SO∈{なし, ≥8} の対象数とROI。
       現行の「買い目カット」セマンティクスとの比較用。

Sim B: 朝時点でオッズなしに計算できる構造統計
       （score_sd / score_gap2r / pred_sd / pred_top2sum, notify_prerace_wt._score_stats と同定義）
       の帯ごとに、現行本番判定（gami≥7カット+SO≥8+gap23+SS/S）での
       「オッズ起因の不成立率（全目ガミ or SO<8）」「購入成立率」「購入ROI」を集計。
       → 不成立率が高く購入貢献ゼロの帯 = 朝除外の候補条件。

モデル: lgbm_wt_june_eval / REF 2026-03-01〜05-31・OOS 2026-06-01〜06-19
オッズ: wt_odds（最終オッズ近似）
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

CAND_GAP12 = 0.07
S_GAP12 = 0.10
SYNTH_ODDS_MIN = 8.0
GAP23_MIN = 1.0
GAMI_THR = 7.0

REF_FROM, REF_TO = "2026-03-01", "2026-05-31"
OOS_FROM, OOS_TO = "2026-06-01", "2026-06-19"


def load_trio_board(race_keys):
    board = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is not None and 0 < float(od) < 9000:
                    try:
                        key = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                        board[rk][key] = float(od)
                    except ValueError:
                        pass
    return board


def collect(model, date_from, date_to):
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) >= 7})].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    board = load_trio_board(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 5:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 < CAND_GAP12:
            continue
        gap23_pt = (p[1] - p[2]) * 100.0
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        frames = g["frame_no"].astype(int).tolist()
        p1, p2, thirds = frames[0], frames[1], frames[2:]

        bd = board.get(rk, {})
        legs = {t: bd[frozenset({p1, p2, t})] for t in thirds
                if frozenset({p1, p2, t}) in bd}
        if not legs:
            continue

        # 構造統計（notify_prerace_wt._score_stats と同定義・オッズ非依存）
        scores = sorted(g["race_point"].dropna().tolist(), reverse=True)
        score_sd = score_gap2r = None
        if len(scores) >= 5:
            m = sum(scores) / len(scores)
            score_sd = (sum((x - m) ** 2 for x in scores) / len(scores)) ** 0.5
            score_gap2r = (scores[0] + scores[1]) / 2 - sum(scores[2:]) / (len(scores) - 2)
        pv = sorted(p, reverse=True)
        pm = sum(pv) / len(pv)
        pred_sd = (sum((x - pm) ** 2 for x in pv) / len(pv)) ** 0.5
        pred_top2sum = pv[0] + pv[1]

        rows.append({
            "rk": rk, "gap12": gap12, "gap23_pt": gap23_pt,
            "p1": p1, "p2": p2, "top3": top3, "legs": legs,
            "score_sd": score_sd, "score_gap2r": score_gap2r,
            "pred_sd": pred_sd, "pred_top2sum": pred_top2sum,
        })
    return rows


def verdict_current(r, thr=GAMI_THR, so_min=SYNTH_ODDS_MIN):
    """現行本番（買い目カット）判定。returns (verdict, fail_reason, bet, pay)"""
    valid = {t: o for t, o in r["legs"].items() if o >= thr}
    if not valid:
        return "skip", "全目ガミ", 0, 0
    synth = 1.0 / sum(1.0 / o for o in valid.values())
    if synth < so_min:
        return "skip", "SO", len(valid) * 100, 0
    if r["gap23_pt"] < GAP23_MIN:
        return "skip", "gap23", 0, 0
    if len(valid) > 3 and r["gap12"] < S_GAP12:
        return "skip", "gap12", 0, 0
    pay = 0
    for t, o in valid.items():
        if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
            pay = int(o * 100)
            break
    return ("SS" if len(valid) <= 3 else "S"), None, len(valid) * 100, pay


def verdict_racelevel(r, thr, use_so, use_gap23=True):
    """レース単位除外（カットなし・全目購入）判定。"""
    legs = r["legs"]
    if min(legs.values()) < thr:
        return "skip", 0, 0
    if use_so:
        synth = 1.0 / sum(1.0 / o for o in legs.values())
        if synth < SYNTH_ODDS_MIN:
            return "skip", 0, 0
    if use_gap23 and r["gap23_pt"] < GAP23_MIN:
        return "skip", 0, 0
    if len(legs) > 3 and r["gap12"] < S_GAP12:
        return "skip", 0, 0
    pay = 0
    for t, o in legs.items():
        if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
            pay = int(o * 100)
            break
    return "buy", len(legs) * 100, pay


def sim_a(rows, label, days):
    print(f"\n===== Sim A: レース単位除外セマンティクス {label}（候補{len(rows)}R / {days}日） =====")
    print(f"{'thr':>4} {'SO≥8':>5} {'R数':>5} {'R/日':>5} {'的中':>4} {'投資':>9} {'払戻':>9} {'ROI':>7}")
    for thr in (5.0, 6.0, 7.0):
        for use_so in (False, True):
            res = [verdict_racelevel(r, thr, use_so) for r in rows]
            buys = [x for x in res if x[0] == "buy"]
            bets = sum(b for _, b, _ in buys)
            pays = sum(p for _, _, p in buys)
            hits = sum(1 for _, _, p in buys if p > 0)
            roi = pays / bets if bets else 0.0
            print(f"{thr:>4.1f} {'あり' if use_so else 'なし':>5} {len(buys):>5} "
                  f"{len(buys)/days:>5.1f} {hits:>4} {bets:>9,} {pays:>9,} {roi:>6.1%}")


def sim_b(rows_ref, rows_oos):
    print("\n===== Sim B: 構造統計の帯別 → 現行判定の成立/不成立（REF+OOS結合） =====")
    rows = rows_ref + rows_oos
    for r in rows:
        r["verdict"], r["fail"], r["bet"], r["pay"] = verdict_current(r)

    n_buy = sum(1 for r in rows if r["verdict"] != "skip")
    print(f"候補 {len(rows)}R / 購入成立 {n_buy}R "
          f"(オッズ起因不成立: 全目ガミ {sum(1 for r in rows if r['fail']=='全目ガミ')}"
          f" / SO {sum(1 for r in rows if r['fail']=='SO')})")

    for feat, better in (("score_gap2r", "大=軸堅い"), ("score_sd", "大=格差大"),
                         ("pred_sd", "大=指数偏り"), ("pred_top2sum", "大=2強")):
        vals = [r[feat] for r in rows if r[feat] is not None]
        qs = np.percentile(vals, [20, 40, 60, 80])
        bands = [(-1e9, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], qs[3]), (qs[3], 1e9)]
        print(f"\n── {feat}（{better}）5分位 ──")
        print(f"{'帯':<22} {'R数':>5} {'オッズ落ち':>8} {'購入':>5} {'成立率':>7} "
              f"{'的中':>4} {'投資':>8} {'払戻':>8} {'ROI':>7}")
        for lo, hi in bands:
            sel = [r for r in rows if r[feat] is not None and lo <= r[feat] < hi]
            oddsfail = sum(1 for r in sel if r["fail"] in ("全目ガミ", "SO"))
            buys = [r for r in sel if r["verdict"] != "skip"]
            bets = sum(r["bet"] for r in buys)
            pays = sum(r["pay"] for r in buys)
            hits = sum(1 for r in buys if r["pay"] > 0)
            roi = pays / bets if bets else 0.0
            print(f"[{lo:>8.3f},{hi:>8.3f}) {len(sel):>5} {oddsfail/len(sel) if sel else 0:>7.1%} "
                  f"{len(buys):>5} {len(buys)/len(sel) if sel else 0:>6.1%} "
                  f"{hits:>4} {bets:>8,} {pays:>8,} {roi:>6.1%}")


def main():
    print("モデルロード中...", flush=True)
    model = load_model("lgbm_wt_june_eval")
    rows_ref = collect(model, REF_FROM, REF_TO)
    rows_oos = collect(model, OOS_FROM, OOS_TO)

    sim_a(rows_ref, f"REF {REF_FROM}〜{REF_TO}", 92)
    sim_a(rows_oos, f"OOS {OOS_FROM}〜{OOS_TO}", 19)
    sim_b(rows_ref, rows_oos)


if __name__ == "__main__":
    main()
