#!/usr/bin/env python3
"""指定日の **7B候補JSONだけ** を生成する（他ランクの候補ファイルに触れない）。

## なぜ必要か

7B（RANK_7B）は 2026-08-03 に新設したが、その日の 8:00 朝バッチ
（`wave-picks-wt`）は 7B のデプロイ前に走り終えていたため
`wave_picks_wt_{date}_s7b_candidates.json` が存在しない。

当日分を後から用意する手段として `wave-picks-wt` を再実行すると、
**7S/7A/9S/9A の候補ファイルもすべて作り直される**。朝と異なる結果が出れば
`write_candidates_wt.py` 経由で picks_history の候補行が変わり、既に入稿・
発走前判定が済んだレースとの整合が崩れる。それを避けるため、7B の候補
ファイルだけを書き出す限定スクリプトとして分離した。

## 何をするか / しないか

する:
  - 指定日の7車立てレースについて、本番と同じ手順で7B候補を算出
    （rank_7s_select_axis → order_disagree → rank_7b_daily_select）
  - `data/picks/wave_picks_wt_{date}_s7b_candidates.json` を書き出す

しない:
  - 他ランクの候補ファイルの読み書き
  - picks_history / wt_entries など DB への書き込み（**読み取り専用**）

## 使い方

    PYTHONPATH=. .venv/bin/python scripts/gen_7b_candidates_only.py 2026-08-03
    PYTHONPATH=. .venv/bin/python scripts/gen_7b_candidates_only.py 2026-08-03 --dry-run
    # 発走済みレースを除外（入稿用途ではこちらを推奨）
    PYTHONPATH=. .venv/bin/python scripts/gen_7b_candidates_only.py 2026-08-03 --future-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wt_vintage_config import assert_vintage_for_past
from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.cli.main import _venue_name
from src.strategy_wt import (
    rank_7b_daily_select, rank_7b_order_disagree, rank_7b_select_legs,
    rank_7s_field_entropy, rank_7s_select_axis, rank_7s_wt_mark3_overlap_n,
    rank_7s_wt_overlap_n,
)

JST = timezone(timedelta(hours=9))
N_CAR = 7


def _fmt_start(start_at: str | int | None) -> str:
    if not start_at:
        return "--:--"
    try:
        return datetime.fromtimestamp(int(start_at), tz=JST).strftime("%H:%M")
    except (ValueError, TypeError, OSError):
        return "--:--"


def build(target_date: str, model_name: str, win_model_name: str,
          future_only: bool, bad_model_name: str = "lgbm_wt_bad") -> list[dict]:
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=target_date, max_date=target_date))
    if df.empty:
        print("[gen-7b] 対象レースなし（特徴量が空）")
        return []

    with get_connection() as c:
        # race_type は新7B（準決勝限定・2026-08-05〜）のゲートに必須。
        # 欠けると rank_7b_daily_select が**黙って0件を返す**ので必ず取る。
        meta = {
            rk: {"n_entries": ne, "venue_id": vid, "race_no": rno, "start_at": st,
                 "race_type": rt}
            for rk, ne, vid, rno, st, rt in c.execute(
                "SELECT race_key, n_entries, venue_id, race_no, start_at, race_type "
                "FROM wt_races WHERE race_date = ?", (target_date,))
        }
        rks7 = [rk for rk, m in meta.items() if m["n_entries"] and int(m["n_entries"]) == N_CAR]
        marks: dict[str, dict[int, int]] = {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, pmv in c.execute(q, chunk):
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
        # 会場名は venue_info（src/cli/main.py の venue_map と同じ引き方）。
        # wt_races には venue_name 列は無く venue_id しか持たない。
        try:
            venues = {str(k): v for k, v in
                      c.execute("SELECT venue_code, name FROM venue_info")}
        except Exception:  # noqa: BLE001 - venue_info 未整備でも候補生成は続ける
            venues = {}

    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        print("[gen-7b] 7車立てレースなし")
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    # 3ヘッド軸選定（2026-08-04〜・7車立てのみ）。本番の wave-picks-wt と同じ軸を
    # 引くため、7B候補もここで pred_bad を持たせる。モデルが無ければ None のまま
    # 従来の重なり方式へフォールバックする。
    try:
        df["pred_bad"] = load_model(bad_model_name).predict_proba(X)[:, 1]
    except FileNotFoundError:
        df["pred_bad"] = None
        print(f"[gen-7b] {bad_model_name} が見つかりません。従来の重なり方式で軸選定します。")

    now = int(time.time())
    raw: list[dict] = []
    n_started = 0
    for rk, g in df.groupby("race_key"):
        m = meta.get(rk)
        if not m or len(g) != N_CAR:
            continue
        if future_only and m["start_at"] and int(m["start_at"]) <= now:
            n_started += 1
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        bad_probs = None
        if "pred_bad" in g.columns and not g["pred_bad"].isna().any():
            bad_probs = {int(r.frame_no): float(r.pred_bad) for r in g.itertuples(index=False)}
        sel = rank_7s_select_axis(win_probs, top3_probs, bad_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel

        mk = marks.get(rk, {})
        wt_honmei = next((f for f, v in mk.items() if v == 1), None)
        wt_taikou = next((f for f, v in mk.items() if v == 2), None)
        wt_ana = next((f for f, v in mk.items() if v == 3), None)
        others = sorted(set(top3_probs) - {axis1, axis2})

        raw.append({
            "race_key": rk,
            "venue_name": _venue_name(venues, m["venue_id"]),
            "race_no": int(m["race_no"]),
            "start_time": _fmt_start(m["start_at"]),
            "axis1": axis1, "axis2": axis2,
            "axis_sum": round(axis_sum, 4),
            "entropy": round(rank_7s_field_entropy(top3_probs), 4),
            "wt_overlap_n": rank_7s_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou),
            "wt_mark3_overlap_n": rank_7s_wt_mark3_overlap_n(
                axis1, axis2, wt_honmei, wt_taikou, wt_ana),
            "order_disagree": rank_7b_order_disagree(win_probs, wt_honmei),
            "race_type": m.get("race_type"),
            "wt_ana": wt_ana,
            "others": others,
            "top3_probs": {str(k): round(v, 6) for k, v in top3_probs.items()},
            "legs_7b": rank_7b_select_legs(others, top3_probs, wt_ana),
        })

    if future_only and n_started:
        print(f"[gen-7b] 発走済みのため除外: {n_started}レース")
    selected = rank_7b_daily_select(raw)
    # 0件のとき原因を切り分けられるよう母集団の内訳を出す（race_type 欠損で
    # 黙って0件になる事故を検知するため。src/cli/main.py 側と同じ方針）。
    _n_ov2 = sum(1 for c in raw if c.get("wt_overlap_n") == 2)
    _n_agree = sum(1 for c in raw
                   if c.get("wt_overlap_n") == 2 and c.get("order_disagree") is False)
    _n_rt_missing = sum(1 for c in raw if c.get("race_type") is None)
    print(f"[gen-7b] 7B候補 {len(selected)}件 / 生候補 {len(raw)}件中")
    print(f"[gen-7b] 母集団: overlap2={_n_ov2} → 順序一致={_n_agree} → 準決勝="
          f"{len(selected)}  (race_type欠損 {_n_rt_missing}件)")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--model", default="lgbm_wt_eval")
    ap.add_argument("--win-model", default="lgbm_wt_win")
    ap.add_argument("--bad-model", default="lgbm_wt_bad")
    ap.add_argument("--future-only", action="store_true",
                    help="発走時刻を過ぎたレースを除外する（入稿用途で推奨）")
    ap.add_argument("--dry-run", action="store_true", help="ファイルを書かず内容だけ表示")
    args = ap.parse_args()

    # 過去日に本番モデルを当てると in-sample になるので落とす（2026-08-08）。
    # 既定値が本番モデル名なので、指定を忘れると**無言で**そうなっていた。
    assert_vintage_for_past(args.date, {"eval": args.model,
                                        "win": args.win_model,
                                        "bad": args.bad_model})

    cands = build(args.date, args.model, args.win_model, args.future_only,
                  args.bad_model)
    for c in cands:
        print(f"  {c['venue_name']}{c['race_no']}R {c['start_time']} "
              f"軸{c['axis1']}={c['axis2']} 相手{c['legs_7b']} "
              f"(△{c['wt_ana']}除外 / entropy={c['entropy']})")

    if args.dry_run:
        print("[gen-7b] DRY RUN（ファイル未書き込み）")
        return

    out = (Path(__file__).parent.parent / "data" / "picks"
           / f"wave_picks_wt_{args.date}_s7b_candidates.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=2)
    print(f"[gen-7b] 書き出し: {out}")


if __name__ == "__main__":
    main()
