"""2026-07 の全日を新方式（SS=7PLUS_R / S=7PLUS_ST/STP）で再構築する。

ローカルの完全データ + クリーンモデル(lgbm_wt: 学習≤2026-06-30)で判定を再導出し、
picks_history 行（買い #7R/#7ST + 見送り #CAND）を JSON に出力する。
オッズゲートは wt_odds（最終オッズ）で近似（バックテストと同一基準）。

使い方:
  .venv/bin/python scripts/backfill_july_newrules_wt.py --from 2026-07-01 --to 2026-07-09 \
      --out /tmp/july_newrules_rows.json
  # → scp して VPS 側で apply_picks_rows_wt.py で適用
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_clean_split_wt as E

ST_STAKE = 100
STP_STAKE = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--model", default="lgbm_wt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = E.load_model(args.model)
    rows = E.collect(model, args.date_from, args.date_to)
    print(f"候補（7+車 gap12>=0.07）: {len(rows)}R")

    out = []
    n_ss = n_st = n_stp = n_mw = 0
    for r in rows:
        rk = r["rk"]
        race_date = f"{rk[:4]}-{rk[4:6]}-{rk[6:8]}"
        p1, p2, r3 = r["p1"], r["p2"], r["r3"]
        thirds = [t for t in r["frames"][2:]]
        trio_top3_pay = int(r["trio"].get(r["top3"], 0) * 100) // 10 * 10 if r["top3"] in r["trio"] else 0

        # ── SS 判定（レース単位 min全目>=7 ∧ gap12>=0.10 ∧ gap23>=1pt）──
        legs = {t: r["trio"].get(frozenset({p1, p2, t})) for t in thirds}
        legs = {t: o for t, o in legs.items() if o}
        gami = min(legs.values()) if legs else None
        ss_buy = (legs and gami >= E.SS_GAMI
                  and r["gap12"] >= E.SS_GAP12 and r["gap23_pt"] >= E.GAP23_MIN)
        ss_hit = False
        ss_pay = 0
        if ss_buy:
            for t, o in legs.items():
                if frozenset({p1, p2, t}) == r["top3"]:
                    ss_hit = True
                    ss_pay = int(o * 100) // 10 * 10
                    break

        # ── S/S+ 判定（三連単F min全目>=10 ∧ gap12>=0.15）──
        st_buy = False
        st_rank = None
        st_combos = {}
        if r["gap12"] >= E.ST_GAP12:
            for s in (p2, r3):
                for t in r["frames"]:
                    if t in (p1, s):
                        continue
                    ov = r["tri"].get((p1, s, t))
                    if ov:
                        st_combos[(p1, s, t)] = ov
            if st_combos and min(st_combos.values()) >= E.ST_GAMI:
                st_buy = True
                is_plus = r["gap12"] >= E.STP_GAP12 and r["gap34"] >= E.STP_GAP34
                st_rank = "7PLUS_STP" if is_plus else "7PLUS_ST"
        st_stake = STP_STAKE if st_rank == "7PLUS_STP" else ST_STAKE
        st_hit = st_buy and r["order"] in st_combos
        st_pay = int(st_combos[r["order"]] * st_stake) // 10 * 10 if st_hit else 0

        gap23_pt = round(r["gap23_pt"], 2)
        pg = round(gami, 2) if gami is not None else None

        if ss_buy:
            out.append({
                "race_date": race_date, "race_key": f"{rk}#7R", "rank": "7PLUS_R",
                "pred_combo": f"{p1}-{p2}-{','.join(str(t) for t in legs)}",
                "n_combos": len(legs), "hit": int(ss_hit), "payout": ss_pay,
                "trio_payout": trio_top3_pay, "bet_amount": len(legs) * 100,
                "miwokuri": False, "prerace_gami": pg, "gap23": gap23_pt,
            })
            n_ss += 1
        if st_buy:
            out.append({
                "race_date": race_date, "race_key": f"{rk}#7ST", "rank": st_rank,
                "pred_combo": f"3連単F: {p1}→{p2},{r3}→全",
                "n_combos": len(st_combos), "hit": int(st_hit), "payout": st_pay,
                "trio_payout": trio_top3_pay, "bet_amount": len(st_combos) * st_stake,
                "miwokuri": False, "prerace_gami": pg, "gap23": gap23_pt,
            })
            if st_rank == "7PLUS_STP":
                n_stp += 1
            else:
                n_st += 1
        if not ss_buy and not st_buy:
            # 見送り（候補記録）: hit は三連複全目換算の参考値・bet 0
            mw_hit = r["top3"] in {frozenset({p1, p2, t}) for t in thirds}
            out.append({
                "race_date": race_date, "race_key": f"{rk}#CAND", "rank": "7PLUS_CAND",
                "pred_combo": f"{p1}-{p2}-{','.join(str(t) for t in thirds)}",
                "n_combos": len(thirds), "hit": int(mw_hit), "payout": 0,
                "trio_payout": trio_top3_pay, "bet_amount": 0,
                "miwokuri": True, "prerace_gami": pg, "gap23": gap23_pt,
            })
            n_mw += 1

    Path(args.out).write_text(json.dumps({
        "date_from": args.date_from, "date_to": args.date_to, "rows": out,
    }, ensure_ascii=False), encoding="utf-8")
    inv = sum(x["bet_amount"] for x in out)
    ret = sum(x["payout"] for x in out)
    print(f"SS買い {n_ss} / S買い {n_st} / S+買い {n_stp} / 見送り {n_mw}")
    print(f"投資 {inv:,} → 払戻 {ret:,}  (ROI {ret/inv:.1%})" if inv else "投資なし")
    print(f"→ {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
