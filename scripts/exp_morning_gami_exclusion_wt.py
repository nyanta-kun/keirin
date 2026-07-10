"""朝時点オッズによる候補除外条件の検討（2026-07-09）

「朝時点で人気が極端に偏っており、直前まで待ってもガミ（購入不成立）に
なりそうなレース」を朝の時点で候補から除外できるかを検証する。

データ: wt_odds_snapshot (snapshot_type='morning', 2026-06-08〜06-18, 727R)
        vs wt_odds（最終オッズ・prerace近似）

手順:
  1. モデル(lgbm_wt_june_eval)で候補ゲート(7+車, gap12≥0.07)を再現
  2. 朝オッズで gami_morning(最安目) / SO_morning(全目合成) を計算
  3. 最終オッズで本番prerace判定（gami≥7カット+SO≥8+gap23≥1pt+SS/S規則）
  4. 朝指標の帯ごとに「最終的に購入成立した率」を集計
     → 購入成立ゼロの朝帯 = 安全に朝除外できる条件
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
GAP23_MIN = 1.0
GAMI_THR = 7.0  # 現行本番閾値で最終判定

FROM, TO = "2026-06-08", "2026-06-18"


def load_board(table_sql, params):
    board = defaultdict(dict)
    with get_connection() as c:
        for rk, comb, od in c.execute(table_sql, params):
            if od is not None and 0 < float(od) < 9000:
                try:
                    key = frozenset(int(x) for x in re.split(r"[-=]", str(comb)))
                    board[rk][key] = float(od)
                except ValueError:
                    pass
    return board


def main():
    print("モデルロード中...", flush=True)
    model = load_model("lgbm_wt_june_eval")
    df = build_features_wt(load_raw_data_wt(min_date=FROM, max_date=TO))

    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (FROM, TO)))
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) >= 7})].copy()
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    morning = load_board(
        "SELECT race_key, combination, odds_value FROM wt_odds_snapshot "
        "WHERE bet_type='trio' AND snapshot_type='morning' "
        "AND substr(race_key,1,8) BETWEEN ? AND ?",
        (FROM.replace("-", ""), TO.replace("-", "")))
    final = load_board(
        "SELECT race_key, combination, odds_value FROM wt_odds "
        "WHERE bet_type='trio' AND substr(race_key,1,8) BETWEEN ? AND ?",
        (FROM.replace("-", ""), TO.replace("-", "")))

    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 3 or rk not in morning:
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

        def legs(bd):
            return {t: bd[frozenset({p1, p2, t})] for t in thirds
                    if frozenset({p1, p2, t}) in bd}

        lm, lf = legs(morning[rk]), legs(final.get(rk, {}))
        if not lm or not lf:
            continue

        gami_m = min(lm.values())
        so_m = 1.0 / sum(1.0 / o for o in lm.values())
        # 朝時点で「現行prerace判定を朝オッズに適用したらどうなるか」
        vm = {t: o for t, o in lm.items() if o >= GAMI_THR}
        so_m_valid = 1.0 / sum(1.0 / o for o in vm.values()) if vm else 0.0

        # 最終オッズで本番判定
        vf = {t: o for t, o in lf.items() if o >= GAMI_THR}
        verdict = "skip"
        pay = 0
        if vf:
            so_f = 1.0 / sum(1.0 / o for o in vf.values())
            if so_f >= SYNTH_ODDS_MIN and gap23_pt >= GAP23_MIN:
                if len(vf) <= 3:
                    verdict = "SS"
                elif gap12 >= S_GAP12:
                    verdict = "S"
        if verdict != "skip":
            for t, o in vf.items():
                if frozenset({p1, p2, t}) == top3:
                    pay = int(o * 100)
                    break

        rows.append({
            "rk": rk, "gami_m": gami_m, "so_m": so_m, "so_m_valid": so_m_valid,
            "n_valid_m": len(vm), "verdict": verdict,
            "bet": len(vf) * 100 if verdict != "skip" else 0, "pay": pay,
        })

    print(f"候補（7+車 gap12≥{CAND_GAP12} ∧ 朝スナップショットあり）: {len(rows)}R")
    n_buy = sum(1 for r in rows if r["verdict"] != "skip")
    print(f"うち最終購入成立: {n_buy}R\n")

    def band_report(key, bands, label):
        print(f"── 朝{label} 帯別 → 最終購入成立率 ──")
        print(f"{'帯':<14} {'候補R':>6} {'購入成立':>8} {'成立率':>7} {'的中':>5} {'投資':>8} {'払戻':>8}")
        for lo, hi in bands:
            sel = [r for r in rows if lo <= r[key] < hi]
            nb = [r for r in sel if r["verdict"] != "skip"]
            bets = sum(r["bet"] for r in nb)
            pays = sum(r["pay"] for r in nb)
            hits = sum(1 for r in nb if r["pay"] > 0)
            rate = len(nb) / len(sel) if sel else 0.0
            print(f"[{lo:>4},{hi:>4}) {len(sel):>6} {len(nb):>8} {rate:>6.1%} {hits:>5} {bets:>8,} {pays:>8,}")
        print()

    band_report("gami_m", [(0, 2), (2, 4), (4, 6), (6, 7), (7, 10), (10, 20), (20, 9999)],
                "最安目オッズ(gami_morning)")
    band_report("so_m_valid", [(0, 1), (1, 4), (4, 6), (6, 8), (8, 12), (12, 9999)],
                "ガミカット後合成オッズ(SO_morning)")

    # 朝時点で「全目<7（=朝は全目ガミ）」だったレースの復活率
    dead_m = [r for r in rows if r["n_valid_m"] == 0]
    revive = [r for r in dead_m if r["verdict"] != "skip"]
    print(f"朝時点 全目<{GAMI_THR:.0f}倍（朝は購入不可相当）: {len(dead_m)}R → 最終購入成立 {len(revive)}R")
    for r in revive:
        print(f"  復活: {r['rk']} gami_m={r['gami_m']:.1f} verdict={r['verdict']} pay={r['pay']}")

    # 朝時点でSO(valid)<8だったレースの復活率
    so_dead = [r for r in rows if 0 < r["so_m_valid"] < SYNTH_ODDS_MIN]
    so_rev = [r for r in so_dead if r["verdict"] != "skip"]
    print(f"朝時点 SO(ガミカット後)<{SYNTH_ODDS_MIN:.0f}: {len(so_dead)}R → 最終購入成立 {len(so_rev)}R")
    for r in so_rev:
        print(f"  復活: {r['rk']} so_m_valid={r['so_m_valid']:.1f} verdict={r['verdict']} pay={r['pay']}")


if __name__ == "__main__":
    main()
