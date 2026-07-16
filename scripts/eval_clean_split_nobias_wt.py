"""現行 SS の欠車バイアス除去版バックテスト（2026-07-15）。

※ S/S+（三連単1着固定F）は優位性なしのため 2026-07-15 に全廃・評価対象から除外。

eval_clean_split_wt.py（公式）との差分は1点のみ:
  公式: `df = df[df["finish_order"] >= 1]` → **完走者のみでランキング**
        （モデル上位が落車/欠車したレースで3位以下が繰り上がる=未来情報。
         doc65 バイアス①・keirin-survivor-bias-inflation で約4倍過大と判明）
  本版: **全出走車（発走前情報）でランキング**し、ゲート(gap12/gap23/gami)も
        発走前ランキングで判定。非完走(finish_order 0/NULL)絡みの買い目は
        - refund モード: 返還（賭け金除外）… 欠車には正しく、落車/失格には楽観
        - lost モード:   外れ（賭け金没収）… 落車/失格には正しく、欠車には悲観
        真値は両者の間（データ上区別不能のため両バウンドを提示）。

使い方:
  .venv/bin/python scripts/eval_clean_split_nobias_wt.py --model lgbm_wt_2026h1_eval \
      --windows 2026-04-01:2026-06-30 2026-07-01:2026-07-10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import line_score_features, ss_policy
from eval_clean_split_wt import (
    load_boards, CAND_GAP12, SS_GAP12, SS_GAMI, GAP23_MIN,
)


def collect_nobias(model, date_from, date_to):
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
    df = df[df["race_key"].isin({rk for rk, ne in ne_map.items() if ne and int(ne) == 7})].copy()
    # ★公式との唯一の差分: finish_order>=1 フィルタを行わない（全出走車ランキング）
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
        nonfin = set()
        for _, row in g.iterrows():
            fo = row["finish_order"]
            fo = None if (fo is None or fo != fo) else int(fo)
            if fo in (1, 2, 3):
                fin[fo] = int(row["frame_no"])
            if fo is None or fo == 0:
                nonfin.add(int(row["frame_no"]))
        if len(fin) < 3:
            # 3完走未満（中止等）は集計対象外（公式も候補に乗らない）
            continue
        frames = g["frame_no"].astype(int).tolist()
        avg_gap, n_lines, all_solo = line_score_features(line_map.get(rk, []))
        rows.append({
            "rk": rk, "gap12": gap12,
            "gap23_pt": (p[1] - p[2]) * 100.0,
            "gap34": (p[2] - p[3]) if len(p) >= 4 else 0.0,
            "p1": frames[0], "p2": frames[1], "r3": frames[2],
            "frames": frames, "nonfin": nonfin,
            "top3": frozenset(fin.values()),
            "order": (fin[1], fin[2], fin[3]),
            "trio": trio_bd.get(rk, {}),
            "tri": tri_bd.get(rk, {}),
            "race_type": rt_map.get(rk),
            "avg_gap": avg_gap, "n_lines": n_lines, "all_solo": all_solo,
        })
    return rows


def eval_ss(rows, mode="refund"):
    n = h = b = pp = n_dns = 0
    for r in rows:
        legs = {t: r["trio"].get(frozenset({r["p1"], r["p2"], t}))
                for t in r["frames"][2:]}
        legs = {t: o for t, o in legs.items() if o}
        if not legs or min(legs.values()) < SS_GAMI:
            continue
        if r["gap12"] < SS_GAP12 or r["gap23_pt"] < GAP23_MIN:
            continue
        skip_reason, stake = ss_policy(
            r["race_type"], r["avg_gap"], r["n_lines"], r["all_solo"])
        if skip_reason:
            continue
        pay = bet = 0
        hit = False
        touched_dns = False
        for t, o in legs.items():
            combo = frozenset({r["p1"], r["p2"], t})
            if combo & r["nonfin"]:
                touched_dns = True
                if mode == "refund":
                    continue          # 返還=賭け金除外
            bet += stake
            if combo == r["top3"]:
                pay += int(o * stake)
                hit = True
        if bet == 0:
            continue
        n += 1
        n_dns += 1 if touched_dns else 0
        h += 1 if hit else 0
        b += bet
        pp += pay
    return n, h, b, pp, n_dns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm_wt_2026h1_eval")
    ap.add_argument("--windows", nargs="+", required=True)
    args = ap.parse_args()

    print(f"モデル: {args.model}（全出走車ランキング=欠車バイアス除去版）", flush=True)
    model = load_model(args.model)

    for w in args.windows:
        f, t = w.split(":")
        rows = collect_nobias(model, f, t)
        days = len({r["rk"][:8] for r in rows}) or 1
        print(f"\n===== {f} 〜 {t}（候補{len(rows)}R / 開催{days}日） =====")
        for mode in ("refund", "lost"):
            tag = "返還(楽観)" if mode == "refund" else "外れ(悲観)"
            print(f"--- 非完走絡み目={tag} ---")
            print(f"{'区分':<14} {'R数':>5} {'R/日':>5} {'的中率':>6} {'投資':>9} "
                  f"{'払戻':>9} {'ROI':>7} {'欠車絡みR':>8}")
            for label, fn in (
                    ("SS(三連複)", lambda r: eval_ss(r, mode)),):
                n, h, b, pp, nd = fn(rows)
                if n == 0:
                    print(f"{label:<14} {'0':>5}")
                    continue
                print(f"{label:<14} {n:>5} {n/days:>5.1f} {h/n:>6.1%} {b:>9,} "
                      f"{pp:>9,} {pp/b:>6.1%} {nd:>8}")


if __name__ == "__main__":
    main()
