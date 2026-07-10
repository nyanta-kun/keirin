"""2026-07-10 の旧コード判定分を新ルール（SS=7PLUS_R / S=7PLUS_ST/STP）で遡及再判定する。

一回性の移行スクリプト（doc52・新体系デプロイ 2026-07-10 21:47 JST）。

- SS: decisions に記録済みの 15分前実測 trio leg_odds で _determine_live_rank を再適用
- S/S+: 15分前の三連単オッズは未記録のため、当日収集済み wt_odds の三連単オッズ
  （日中取得で更新済み・ほぼ最終値）で _determine_st_rank を適用（odds_source=retro_db）
- decisions を新判定で上書き（旧判定は legacy として保持）・picks_history を即時更新
- 対象は decided_at < 21:50（旧コード判定分）のみ。新コード判定分（21:55〜）は触らない

使い方（VPS上で）:
  export KEIRIN_DB_URL=...
  .venv/bin/python3 scripts/rescore_today_newrules.py 2026-07-10
  .venv/bin/python3 scripts/notify_results_wt.py 2026-07-10 --silent   # 再採点
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import notify_prerace_wt as NP
from src.database import get_connection

CUTOFF_DECIDED_AT = "21:50:00"  # これ以前の判定＝旧コード分のみ再判定


def load_candidates(today: str) -> dict:
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    cmap = {}
    for suffix in ("_candidates.json", "_night_candidates.json"):
        p = picks_dir / f"wave_picks_wt_{today}{suffix}"
        if p.exists():
            for c in json.loads(p.read_text(encoding="utf-8")):
                cmap[c["race_key"]] = c
    return cmap


def load_trifecta_odds(rk: str) -> list[dict]:
    with get_connection() as c:
        rows = c.execute(
            "SELECT combination, odds_value FROM wt_odds "
            "WHERE race_key = ? AND bet_type = 'trifecta'", (rk,)).fetchall()
    return [{"combination": r["combination"] if isinstance(r, dict) else r[0],
             "odds_value": r["odds_value"] if isinstance(r, dict) else r[1]} for r in rows]


def main() -> None:
    today = sys.argv[1] if len(sys.argv) > 1 else "2026-07-10"
    cmap = load_candidates(today)
    decisions = NP._load_decisions(today)

    targets = [
        (rk, dec) for rk, dec in decisions.items()
        if "#" not in rk
        and dec.get("decided_at", "99:99:99") < CUTOFF_DECIDED_AT
        and rk in cmap
    ]
    print(f"対象: {len(targets)}R（旧コード判定分・cutoff {CUTOFF_DECIDED_AT}）")

    n_ss_buy = n_ss_skip = n_st_buy = 0
    for rk, dec in sorted(targets):
        pick = cmap[rk]
        p1, p2 = pick.get("pivot1"), pick.get("pivot2")
        legacy = {"decision": dec.get("decision"), "rank": dec.get("rank")}

        # ── SS 再判定（15分前実測 trio leg_odds）──
        leg_odds = dec.get("leg_odds") or {}
        odds_data = {"trio": [{"combination": f"{p1}-{p2}-{t}", "odds_value": float(o)}
                              for t, o in leg_odds.items() if o]}
        ss_rank, ss_thirds, ss_odds = NP._determine_live_rank(pick, odds_data)

        base_rec = {
            "pivot1": p1, "pivot2": p2,
            "all_min_odds": round(min(ss_odds.values()), 2) if ss_odds else dec.get("all_min_odds"),
            "leg_odds": {str(t): o for t, o in ss_odds.items()},
            "retro_newrules": True,
            "legacy": legacy,
            **{k: dec[k] for k in ("score_mean", "score_sd", "score_gap2r",
                                   "pred_sd", "pred_top2sum") if k in dec},
        }
        if ss_rank == "7PLUS_R":
            n_pts = len(ss_thirds)
            combo_str = f"{p1}-{p2}-{','.join(str(t) for t in ss_thirds)}"
            NP._save_decision(today, rk, {
                **base_rec, "decision": "buy", "rank": "7PLUS_R",
                "thirds": [int(t) for t in ss_thirds],
            })
            NP._save_picks_history_state(rk, False, "7PLUS_R", new_pred=(combo_str, n_pts))
            if ss_odds:
                NP._save_prerace_gami(rk, min(ss_odds.values()))
            n_ss_buy += 1
            print(f"  {rk}: SS買い ({n_pts}点)  [旧: {legacy['decision']}/{legacy.get('rank')}]")
        else:
            NP._save_decision(today, rk, {**base_rec, "decision": "skip"})
            NP._save_picks_history_state(rk, True)
            n_ss_skip += 1

        # ── S/S+ 再判定（当日DBの三連単オッズ・retro近似）──
        tri = load_trifecta_odds(rk)
        st_rank, st_combos, st_leg_odds, st_stake = NP._determine_st_rank(
            pick, {"trifecta": tri} if tri else None)
        st_bought = st_rank in ("7PLUS_ST", "7PLUS_STP")
        r3 = int(pick["thirds"][0]) if pick.get("thirds") else None
        st_rec = {
            "decision": "buy" if st_bought else "skip",
            "pivot1": p1,
            "seconds": ([int(p2), r3] if r3 is not None else []),
            "stake": st_stake if st_bought else 0,
            "st_min_odds": round(min(st_leg_odds.values()), 2) if st_leg_odds else None,
            "odds_source": "retro_db",  # 15分前オッズ未記録のため当日DB値で近似
            "retro_newrules": True,
        }
        if st_bought:
            st_rec["rank"] = st_rank
            st_rec["combos"] = [f"{a}-{b}-{c}" for a, b, c in st_combos]
            st_rec["leg_odds"] = st_leg_odds
            NP._save_decision(today, f"{rk}#ST", st_rec)
            NP._insert_st_pick(rk, today, st_rank, pick, st_combos, st_stake)
            n_st_buy += 1
            print(f"  {rk}: 三連単{'S+' if st_rank == '7PLUS_STP' else 'S'} "
                  f"({len(st_combos)}点×{st_stake}円)")
        else:
            NP._save_decision(today, f"{rk}#ST", st_rec)

    print(f"\n完了: SS買い {n_ss_buy} / SS見送り {n_ss_skip} / 三連単S買い {n_st_buy}")
    print("次: notify_results_wt.py で再採点してください")


if __name__ == "__main__":
    main()
