"""クリーン分割モデルの評価（2026-07-10）

指定モデルで test 期間（学習に未使用）と 7月フォワード期間の
SS（三連複レース単位）/ S・S+（三連単1着固定F）規則の成績を算出する。

使い方:
  .venv/bin/python scripts/eval_clean_split_wt.py --model lgbm_wt_h1eval \
      --windows 2026-04-01:2026-06-30 2026-07-01:2026-07-09
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import line_score_features, ss_policy, st_normal_allowed

# 本番条件（notify_prerace_wt.py と揃える）
CAND_GAP12 = 0.07
SS_GAP12 = 0.10
SS_GAMI = 7.0
GAP23_MIN = 1.0
ST_GAP12 = 0.15
ST_GAMI = 10.0
STP_GAP12 = 0.25
STP_GAP34 = 0.04
ST_STAKE = 100
STP_STAKE = 200
# doc53 統合ポリシー（選抜/4分戦/ライン格差増額・S通常gami15）は
# src.strategy_wt の ss_policy / st_normal_allowed を参照（単一実装）


def load_boards(race_keys):
    trio = defaultdict(dict)
    tri = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','trifecta') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                if od is None or not (0 < float(od) < 90000):
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if bt == "trio":
                    trio[rk][frozenset(parts)] = float(od)
                elif len(parts) == 3:
                    tri[rk][tuple(parts)] = float(od)
    return trio, tri


def collect(model, date_from, date_to):
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rt_map = dict(c.execute(
            "SELECT race_key, race_type FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        line_map: dict[str, list] = {}
        for rk_, lg_, rp_ in c.execute(
            "SELECT e.race_key, e.line_group, e.race_point FROM wt_entries e "
            "JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.race_date BETWEEN ? AND ?", (date_from, date_to)):
            line_map.setdefault(rk_, []).append((lg_, rp_))
    # 7車ちょうど限定（本番 wave_picks_wt / notify_prerace_wt と同一母集団。
    # >=7 だと実運用が買わない8/9車が混入し検証と実績が乖離する。2026-07-12）
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) == 7})].copy()
    df = df[df["finish_order"] >= 1].copy()
    if df.empty:
        return []
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    trio_bd, tri_bd = load_boards(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        if len(g) < 5:
            continue
        p = g["pred_prob"].tolist()
        gap12 = p[0] - p[1]
        if gap12 < CAND_GAP12:
            continue
        fin = {}
        for _, row in g.iterrows():
            fo = int(row["finish_order"])
            if fo in (1, 2, 3):
                fin[fo] = int(row["frame_no"])
        if len(fin) < 3:
            continue
        frames = g["frame_no"].astype(int).tolist()
        avg_gap, n_lines, all_solo = line_score_features(line_map.get(rk, []))
        rows.append({
            "rk": rk, "gap12": gap12,
            "gap23_pt": (p[1] - p[2]) * 100.0,
            "gap34": (p[2] - p[3]) if len(p) >= 4 else 0.0,
            "p1": frames[0], "p2": frames[1], "r3": frames[2],
            "frames": frames,
            "top3": frozenset(fin.values()),
            "order": (fin[1], fin[2], fin[3]),
            "trio": trio_bd.get(rk, {}),
            "tri": tri_bd.get(rk, {}),
            # doc53 統合ポリシー用コンテキスト
            "race_type": rt_map.get(rk),
            "avg_gap": avg_gap, "n_lines": n_lines, "all_solo": all_solo,
        })
    return rows


def eval_ss(rows):
    n = h = b = pp = 0
    for r in rows:
        legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                for t in r["frames"][2:]}
        legs = {t: o for t, o in legs.items() if o}
        if not legs or min(legs.values()) < SS_GAMI:
            continue
        if r["gap12"] < SS_GAP12 or r["gap23_pt"] < GAP23_MIN:
            continue
        # doc53: 選抜/4分戦は見送り・ライン格差>=1.5は増額
        skip_reason, stake = ss_policy(
            r["race_type"], r["avg_gap"], r["n_lines"], r["all_solo"])
        if skip_reason:
            continue
        pay = 0
        for t, o in legs.items():
            if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
                pay = int(o * stake)
                break
        n += 1
        h += 1 if pay > 0 else 0
        b += len(legs) * stake
        pp += pay
    return n, h, b, pp


def eval_st(rows, plus_only=False, base_only=False):
    n = h = b = pp = 0
    for r in rows:
        if r["gap12"] < ST_GAP12:
            continue
        combos = {}
        for s in (r["p2"], r["r3"]):
            for t in r["frames"]:
                if t in (r["p1"], s):
                    continue
                ov = r["tri"].get((r["p1"], s, t))
                if ov:
                    combos[(r["p1"], s, t)] = ov
        if not combos or min(combos.values()) < ST_GAMI:
            continue
        is_plus = r["gap12"] >= STP_GAP12 and r["gap34"] >= STP_GAP34
        # doc53: S通常帯は min>=ST_BASE_GAMI(15) ∧ 非選抜（S+帯は現行のまま）
        if not is_plus and not st_normal_allowed(r["race_type"], min(combos.values())):
            continue
        if plus_only and not is_plus:
            continue
        if base_only and is_plus:
            continue
        stake = STP_STAKE if is_plus else ST_STAKE
        pay = int(combos[r["order"]] * stake) if r["order"] in combos else 0
        n += 1
        h += 1 if pay > 0 else 0
        b += len(combos) * stake
        pp += pay
    return n, h, b, pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", nargs="+", required=True,
                    help="FROM:TO 形式（例 2026-04-01:2026-06-30）")
    args = ap.parse_args()

    print(f"モデル: {args.model}", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        rows = collect(model, f, t)
        days = len({r["rk"][:8] for r in rows}) or 1
        print(f"\n===== {f} 〜 {t}（候補{len(rows)}R / 開催{days}日） =====")
        print(f"{'区分':<14} {'R数':>5} {'R/日':>5} {'的中率':>6} {'投資':>9} {'払戻':>9} {'ROI':>7}")
        for label, fn in (("SS(三連複)", eval_ss),
                          ("S計(三連単)", eval_st),
                          ("  うちS+帯", lambda r: eval_st(r, plus_only=True)),
                          ("  うちS通常", lambda r: eval_st(r, base_only=True))):
            n, h, b, pp = fn(rows)
            if n == 0:
                print(f"{label:<14} {'0':>5}")
                continue
            print(f"{label:<14} {n:>5} {n/days:>5.1f} {h/n:>6.1%} {b:>9,} {pp:>9,} {pp/b:>6.1%}")


if __name__ == "__main__":
    main()
