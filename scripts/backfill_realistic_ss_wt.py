"""picks_history を実精算方式（2026-07-15）で再構築する（SS=7PLUS_R のみ）。

指数ランキング・買い目は発走前のオッズ盤面掲載車（欠車除く・落車失格含む）で作成し、
落車・失格絡みの買い目は購入のまま外れ計上する（eval_clean_split_wt.collect の実精算方式）。
購入ポリシー（2026-07-16〜: 選抜レースのみ見送り・常に100円/点）も適用する。

クリーンモデル lgbm_wt（学習<=2026-06-30）で判定を再導出し、
picks_history 行（買い #7R + 見送り #CAND）を JSON に出力する。
学習期間内の日付は in-sample のため参考値（従来の 7/10 再構築と同じ扱い）。

使い方:
  KEIRIN_DB_URL=... .venv/bin/python scripts/backfill_realistic_ss_wt.py \
      --from 2025-07-01 --to 2025-12-31 --out /tmp/realistic_rows_2025h2.json
  # → apply_picks_rows_wt.py で適用（範囲内の route='wt' 行を全削除して置換）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_clean_split_wt as E
from src.strategy_wt import ss_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--model", default="lgbm_wt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = E.load_model(args.model)
    rows = E.collect(model, args.date_from, args.date_to)
    print(f"候補（7車 gap12>={E.CAND_GAP12}）: {len(rows)}R", flush=True)

    out = []
    n_ss = n_mw = 0
    for r in rows:
        rk = r["rk"]
        race_date = f"{rk[:4]}-{rk[4:6]}-{rk[6:8]}"
        p1, p2 = r["p1"], r["p2"]
        thirds = list(r["frames"][2:])
        trio_top3_pay = int(r["trio"].get(r["top3"], 0) * 100) // 10 * 10 if r["top3"] in r["trio"] else 0
        trifecta_top3_pay = (int(r["tri"].get(r["order"], 0) * 100) // 10 * 10
                             if r["order"] in r["tri"] else 0)

        # ── SS 判定（min全目>=7 ∧ gap12>=0.10 ∧ gap23>=1pt ∧ 非選抜）──
        legs = {t: r["trio"].get(frozenset({p1, p2, t})) for t in thirds}
        legs = {t: o for t, o in legs.items() if o}
        gami = min(legs.values()) if legs else None
        skip_reason, stake = ss_policy(
            r["race_type"], r["avg_gap"], r["n_lines"], r["all_solo"])
        ss_buy = (legs and gami >= E.SS_GAMI
                  and r["gap12"] >= E.SS_GAP12 and r["gap23_pt"] >= E.GAP23_MIN
                  and not skip_reason)
        ss_hit = False
        ss_pay = 0
        if ss_buy:
            for t, o in legs.items():
                if frozenset({p1, p2, t}) == r["top3"]:
                    ss_hit = True
                    ss_pay = (int(o * 100) // 10 * 10) * (stake // 100)
                    break

        gap23_pt = round(r["gap23_pt"], 2)
        gap12_v = round(r["gap12"], 4)
        gap34_v = round(r["gap34"], 4)
        pg = round(gami, 2) if gami is not None else None

        if ss_buy:
            out.append({
                "race_date": race_date, "race_key": f"{rk}#7R", "rank": "7PLUS_R",
                "pred_combo": f"{p1}-{p2}-{','.join(str(t) for t in legs)}",
                "n_combos": len(legs), "hit": int(ss_hit), "payout": ss_pay,
                "trio_payout": trio_top3_pay, "trifecta_payout": trifecta_top3_pay,
                "bet_amount": len(legs) * stake,
                "miwokuri": False, "prerace_gami": pg,
                "gap23": gap23_pt, "gap12": gap12_v, "gap34": gap34_v,
            })
            n_ss += 1
        else:
            # 見送り（候補記録）: hit は三連複全目換算の参考値・bet 0
            mw_hit = r["top3"] in {frozenset({p1, p2, t}) for t in thirds}
            out.append({
                "race_date": race_date, "race_key": f"{rk}#CAND", "rank": "7PLUS_CAND",
                "pred_combo": f"{p1}-{p2}-{','.join(str(t) for t in thirds)}",
                "n_combos": len(thirds), "hit": int(mw_hit), "payout": 0,
                "trio_payout": trio_top3_pay, "trifecta_payout": trifecta_top3_pay,
                "bet_amount": 0,
                "miwokuri": True, "prerace_gami": pg,
                "gap23": gap23_pt, "gap12": gap12_v, "gap34": gap34_v,
            })
            n_mw += 1

    Path(args.out).write_text(json.dumps({
        "date_from": args.date_from, "date_to": args.date_to, "rows": out,
    }, ensure_ascii=False), encoding="utf-8")
    inv = sum(x["bet_amount"] for x in out)
    ret = sum(x["payout"] for x in out)
    hits = sum(x["hit"] for x in out if not x["miwokuri"])
    print(f"SS買い {n_ss}（的中 {hits}） / 見送り {n_mw}")
    print(f"投資 {inv:,} → 払戻 {ret:,}  (ROI {ret/inv:.1%})" if inv else "投資なし")
    print(f"→ {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
