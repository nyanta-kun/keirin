"""gami閾値 7.0→6.0 緩和の影響検証（2026-07-09）

本番の prerace 判定ロジック（notify_prerace_wt._determine_live_rank 相当）を再現し、
gami閾値 T ∈ {5, 6, 7} でのレース数・ROI を比較する。
特に「6で追加され7で落ちる差分帯 [6,7)」の単独ROIを提示する。

判定再現:
  候補ゲート: 7+車 ∧ gap12≥0.07
  valid = 3連複オッズ ≥ T の目
  valid=0 → 見送り / SO(valid)<8 → 見送り / gap23<1pt → 見送り
  1〜3目 → SS購入(validのみ) / 4目以上 ∧ gap12≥0.10 → S購入(validのみ) / それ以外見送り

モデル: lgbm_wt_june_eval（2022-12〜2026-05学習）→ 2026-06 が真OOS
期間:  REF 2026-03-01〜05-31（半インサンプル・参考）/ OOS 2026-06-01〜06-19
注意:  wt_odds は最終オッズ。本番は発走15分前オッズのため帯の所属が多少ずれる。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

CAND_GAP12 = 0.07
S_GAP12 = 0.10
SYNTH_ODDS_MIN = 8.0
GAP23_MIN = 1.0  # %pt

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


def load_n_entries_map(race_keys):
    m = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, n_entries FROM wt_races "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, ne in c.execute(q, chunk):
                m[rk] = ne
    return m


def collect(model, date_from, date_to):
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    ne_map = load_n_entries_map(df["race_key"].unique().tolist())
    valid_rks = {rk for rk, ne in ne_map.items() if ne is not None and int(ne) >= 7}
    df = df[df["race_key"].isin(valid_rks)].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    trio_board = load_trio_board(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 3:
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
        pivot1, pivot2 = frames[0], frames[1]
        thirds = frames[2:]

        bd = trio_board.get(rk, {})
        combo_odds = {}
        for fn in thirds:
            key = frozenset({pivot1, pivot2, fn})
            if key in bd:
                combo_odds[fn] = bd[key]
        if not combo_odds:
            continue

        rows.append({
            "race_key": rk, "gap12": gap12, "gap23_pt": gap23_pt,
            "pivot1": pivot1, "pivot2": pivot2,
            "top3": top3, "combo_odds": combo_odds,
        })
    return rows


def decide(row, thr):
    """本番 _determine_live_rank 再現。returns (rank, bet, pay, gami)"""
    combo_odds = row["combo_odds"]
    gami = min(combo_odds.values())
    valid = {t: o for t, o in combo_odds.items() if o >= thr}
    if not valid:
        return None, 0, 0, gami
    synth = 1.0 / sum(1.0 / o for o in valid.values())
    if synth < SYNTH_ODDS_MIN:
        return None, 0, 0, gami
    if row["gap23_pt"] < GAP23_MIN:
        return None, 0, 0, gami

    if len(valid) > 3 and row["gap12"] < S_GAP12:
        return None, 0, 0, gami
    rank = "SS" if len(valid) <= 3 else "S"

    pay = 0
    for t, o in valid.items():
        if frozenset({row["pivot1"], row["pivot2"], t}) == row["top3"]:
            pay = int(o * 100)
            break
    return rank, len(valid) * 100, pay, gami


def summarize(results):
    n = len(results)
    bets = sum(b for _, b, _ in results)
    pays = sum(p for _, _, p in results)
    hits = sum(1 for _, _, p in results if p > 0)
    roi = pays / bets if bets else 0.0
    return n, hits, bets, pays, roi


def run_window(rows, label):
    print(f"\n===== {label}（候補 {len(rows)}R = 7+車 gap12≥{CAND_GAP12}） =====")
    print(f"{'閾値':<8} {'rank':<5} {'R数':>5} {'的中':>5} {'投資':>9} {'払戻':>9} {'ROI':>8}")
    print("-" * 60)
    per_thr_keys = {}
    for thr in (5.0, 6.0, 7.0):
        bought = {}
        for row in rows:
            rank, bet, pay, _ = decide(row, thr)
            if rank:
                bought[row["race_key"]] = (rank, bet, pay)
        per_thr_keys[thr] = bought
        for rk_label in ("SS", "S", "計"):
            sel = [(r, b, p) for (r, b, p) in bought.values()
                   if rk_label == "計" or r == rk_label]
            n, hits, bets, pays, roi = summarize(sel)
            thr_s = f"gami≥{thr:.0f}" if rk_label == "SS" else ""
            print(f"{thr_s:<8} {rk_label:<5} {n:>5} {hits:>5} {bets:>9,} {pays:>9,} {roi:>7.1%}")
        print("-" * 60)

    # 差分帯: thr=6 で買い、thr=7 では買わない（or 買い目が違う）レース
    only6 = {rk: v for rk, v in per_thr_keys[6.0].items() if rk not in per_thr_keys[7.0]}
    n, hits, bets, pays, roi = summarize(list(only6.values()))
    print(f"差分帯 [6→7で消えるレース]: {n}R 的中{hits} 投資{bets:,} 払戻{pays:,} ROI {roi:.1%}")
    # 同一レースでも買い目点数が変わる分の差分
    both = set(per_thr_keys[6.0]) & set(per_thr_keys[7.0])
    d_bet = sum(per_thr_keys[6.0][rk][1] - per_thr_keys[7.0][rk][1] for rk in both)
    d_pay = sum(per_thr_keys[6.0][rk][2] - per_thr_keys[7.0][rk][2] for rk in both)
    print(f"差分帯 [共通レースの追加買い目分]: 追加投資{d_bet:,} 追加払戻{d_pay:,}"
          f" ROI {d_pay / d_bet:.1%}" if d_bet else "差分帯 [共通レースの追加買い目分]: なし")


def main():
    print("モデルロード中 (lgbm_wt_june_eval)...", flush=True)
    model = load_model("lgbm_wt_june_eval")

    rows_ref = collect(model, REF_FROM, REF_TO)
    run_window(rows_ref, f"REF {REF_FROM}〜{REF_TO}（半インサンプル・参考）")

    rows_oos = collect(model, OOS_FROM, OOS_TO)
    run_window(rows_oos, f"OOS {OOS_FROM}〜{OOS_TO}（真OOS）")


if __name__ == "__main__":
    main()
