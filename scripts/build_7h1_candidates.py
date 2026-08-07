#!/usr/bin/env python3
"""RANK_7H1（穴推奨・本命バスト型）の候補を生成する。

## 位置づけ

7H1 は既存6ランクと **候補生成の入口が違う**。既存は `wave-picks-wt` が作る
選手単位の予測から軸2車を選ぶが、7H1 は**レース単位のバスト予測モデル**
（`lgbm_wt_favbust`）を使うため、専用の生成経路を持つ。
出力する候補JSONの形式と保存先は既存ランクに合わせてあるので、
`notify_prerace_wt.py` からは同じように読める。

## 処理

1. 対象日の7車レースと出走表を DB から取得
2. 本番モデル（`lgbm_wt_eval` / `_win` / `_bad`）で選手単位の予測を作る
3. `favbust_features.build_favbust_row()` で67特徴を組む
   （**軸1 != WT◎ のレースはここで落ちる**＝母集団外）
4. `lgbm_wt_favbust` でバスト確率を予測
5. `strategy_wt.rank_7h1_build_legs()` で買い目（三連複BOX・三連単フォーメーション）
6. `strategy_wt.rank_7h1_daily_select()` で抜け度と当日相対順位により選別

## 使い方

    PYTHONPATH=. .venv/bin/python scripts/build_7h1_candidates.py \\
        --date 2026-08-06 [--out data/wave_picks_wt_2026-08-06_s7h1_candidates.json]
    # 過去日を honest に作る場合は月次vintageを明示する
    #   --eval-model lgbm_wt_eval_m2608 --win-model lgbm_wt_win_m2608 \\
    #   --bad-model lgbm_wt_bad_m2608 --favbust-model lgbm_wt_favbust_m2608

⚠️ 過去分の再構築で本番モデル（全期間学習）を使うと in-sample になる。
   必ず vintage を指定すること（`backfill_7h1_rank_wt.py` はそうする）。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.wt_vintage_config import assert_vintage_for_past
from src.database import get_connection  # noqa: E402
from src.models.trainer import load_model  # noqa: E402
from src.preprocessing.favbust_features import (  # noqa: E402
    build_favbust_row, feature_vector, roles_of,
)
from src.preprocessing.feature_wt import (  # noqa: E402
    build_features_wt, load_raw_data_wt, prepare_X,
)
from src.strategy_wt import (  # noqa: E402
    RANK_7H1_NE, RANK_7H1_TF_UNIT, rank_7h1_build_legs, rank_7h1_daily_select,
    rank_7h1_trio_stakes,
)


def load_context(date_from: str, date_to: str) -> tuple[dict, dict]:
    with get_connection() as c:
        meta = {}
        for r in c.execute(
                "SELECT r.race_key, r.race_date, r.race_no, r.venue_id, r.grade, "
                "       r.race_type, r.day_index, r.start_at, r.distance, "
                "       v.bank_length, v.is_indoor, v.name AS venue_name "
                "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
                "WHERE r.n_entries = ? AND r.cancel = 0 "
                "  AND r.race_date BETWEEN ? AND ?",
                (RANK_7H1_NE, date_from, date_to)):
            meta[r["race_key"]] = dict(r)
        keys = sorted(meta)
        ents: dict[str, list[dict]] = defaultdict(list)
        for i in range(0, len(keys), 700):
            ch = keys[i:i + 700]
            q = ("SELECT race_key, frame_no, name, pred_win_pct, pred_top3_pct, "
                 "       prediction_mark, race_point, line_group, line_size, line_pos, "
                 "       is_line_leader, n_lines, style, prefecture, player_class "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(ch)))
            for r in c.execute(q, ch):
                ents[r["race_key"]].append(dict(r))
    return meta, dict(ents)


def build(date_from: str, date_to: str, eval_model: str, win_model: str,
          bad_model: str, favbust_model: str) -> list[dict]:
    meta_all, ents_all = load_context(date_from, date_to)
    if not meta_all:
        return []
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    df = df[df["race_key"].isin(set(meta_all))].copy()
    if df.empty:
        return []
    ev, wi, ba = load_model(eval_model), load_model(win_model), load_model(bad_model)
    fb = load_model(favbust_model)
    X = prepare_X(df)
    pp3 = ev.predict_proba(X)[:, 1]
    ppw = wi.predict_proba(X)[:, 1]
    pbd = ba.predict_proba(X)[:, 1]
    preds: dict[str, dict[int, tuple]] = defaultdict(dict)
    for rk, fno, a, b, c in zip(df["race_key"], df["frame_no"], pp3, ppw, pbd):
        preds[rk][int(fno)] = (float(a), float(b), float(c))

    rows, feats = [], []
    for rk, pr in preds.items():
        ents = ents_all.get(rk)
        if not ents or len(ents) != RANK_7H1_NE:
            continue
        row = build_favbust_row(meta_all[rk], ents, pr)
        if row is None:
            continue                       # 軸1 != WT◎（母集団外）
        fav = row.pop("_fav")
        roles = roles_of(ents, fav)
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
        trio, tf = rank_7h1_build_legs(others, roles)
        if not trio or not tf:
            continue                       # 別ライン先頭が居ない等で買い目が組めない
        # 🔴 ここで出す金額は**表示用の目安**。実際の配分は入稿時点のオッズで
        #    決め直す（`netkeirin_submit_wt._normalize_multi_candidate`）。
        #    候補生成は朝で板が薄く、入稿は開催単位の3波でずれるため。
        u_tf = RANK_7H1_TF_UNIT
        _st = rank_7h1_trio_stakes([frozenset(t) for t in trio], None, len(tf))
        u_trio = min(_st.values()) if _st else 0
        total = sum(_st.values()) + u_tf * len(tf)
        if not u_trio or not u_tf:
            continue
        m = meta_all[rk]
        name_of = {int(e["frame_no"]): e.get("name") for e in ents}
        rows.append({
            "race_key": rk, "race_date": m["race_date"],
            "venue_name": m.get("venue_name"), "race_no": m.get("race_no"),
            "start_time": m.get("start_at"), "race_type": m.get("race_type"),
            "n_entries": RANK_7H1_NE,
            "fav": fav, "fav_name": name_of.get(fav),
            "gap12": round(row["fav_ppw_gap12"], 6),
            "others": others,
            "roles": {str(k): v for k, v in roles.items()},
            "legs_trio": ["=".join(str(x) for x in sorted(t)) for t in trio],
            "legs_tf": tf,
            "stake_trio": u_trio, "stake_tf": u_tf, "bet_amount": total,
        })
        feats.append(feature_vector(row))

    if not rows:
        return []
    probs = fb.predict(np.array(feats, dtype=float))
    for r, p in zip(rows, probs):
        r["bust_prob"] = float(p)
    # 選別は日単位の相対順位で行う（開催の質でバスト確率の水準が動くため）
    out: list[dict] = []
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["race_date"]].append(r)
    for day in sorted(by_day):
        out.extend(rank_7h1_daily_select(by_day[day]))
    out.sort(key=lambda c: (c["race_date"], -c["bust_prob"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="対象日（YYYY-MM-DD）")
    ap.add_argument("--to", dest="date_to", default=None, help="範囲指定の終端")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-model", default="lgbm_wt_eval")
    ap.add_argument("--win-model", default="lgbm_wt_win")
    ap.add_argument("--bad-model", default="lgbm_wt_bad")
    ap.add_argument("--favbust-model", default="lgbm_wt_favbust")
    args = ap.parse_args()

    # 過去日に本番モデルを当てると in-sample になるので落とす（2026-08-08）。
    # 既定値が本番モデル名なので、指定を忘れると**無言で**そうなっていた。
    _end = args.date_to or args.date
    if _end:
        assert_vintage_for_past(_end, {"bad": args.bad_model, "eval": args.eval_model, "favbust": args.favbust_model, "win": args.win_model})

    d_to = args.date_to or args.date
    cands = build(args.date, d_to, args.eval_model, args.win_model,
                  args.bad_model, args.favbust_model)
    # 既存ランクと同じ data/picks/ 配下へ出す（notify_prerace_wt.py が読む場所）
    out = Path(args.out) if args.out else (
        REPO / "data" / "picks" / f"wave_picks_wt_{args.date}_s7h1_candidates.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"[保存先] {out}  (7H1候補 {len(cands)}件)")
    for c in cands:
        print(f"  {c['venue_name']}{c['race_no']}R  本命{c['fav']}"
              f"({c['fav_name']}) 抜け度{c['gap12'] * 100:.1f}pt "
              f"バスト確率{c['bust_prob'] * 100:.1f}%  "
              f"三連複{len(c['legs_trio'])}点×{c['stake_trio']}円 + "
              f"三連単{len(c['legs_tf'])}点×{c['stake_tf']}円 = {c['bet_amount']}円")


if __name__ == "__main__":
    main()
