#!/usr/bin/env python3
"""RANK_9H1（9車・高配当狙い）の候補を生成する。

## 位置づけ

9H1 は既存ランクと**候補生成の入口が違う**。既存は `wave-picks-wt` が作る選手単位の
予測から軸2車を選ぶが、9H1 は**レース単位の波乱スコア**（`lgbm_upset_screen`）で
レースを選ぶため、7H1 と同じく専用の生成経路を持つ。
出力する候補JSONの形式と保存先は既存ランクに合わせてあるので、
`notify_prerace_wt.py` からは同じように読める。

## 処理

1. 対象日の**9車ちょうど**のレースと出走表を DB から取得
2. `upset_features.build_upset_row()` で31特徴を組む
   （**事前欠車で行数が9に足りないレースはここで落ちる**＝母集団外）
3. `lgbm_upset_screen` で波乱スコアを予測
4. 本番モデルで選手単位の3着内率を出し、`strategy_wt.rank_9h1_build_legs()` で
   買い目（三連単フォーメーション6点）を作る
5. `strategy_wt.rank_9h1_daily_select()` で波乱スコアの絶対閾値により選別

## 使い方

    PYTHONPATH=. .venv/bin/python scripts/build_9h1_candidates.py \\
        --date 2026-08-08 [--out data/wave_picks_wt_2026-08-08_s9h1_candidates.json]
    # 過去日を honest に作る場合は vintage を明示する
    #   --eval-model lgbm_wt_eval_m2608 --screen-model lgbm_upset_screen_m2608 \\
    #   --score-min 0.3021

⚠️ 過去分の再構築で本番モデル（全期間学習）を使うと in-sample になる。
   必ず vintage を指定すること。
⚠️ **閾値 `--score-min` もモデルごとに違う**。vintage を指定したら、その vintage の
   9車スコア p80（`train_upset_screen.py` が最後に出力する）を必ず併せて渡すこと。
   本番の絶対値を当てると件数が数割ずれる。

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

from src.database import get_connection  # noqa: E402
from src.models.trainer import load_model  # noqa: E402
from src.preprocessing.feature_wt import (  # noqa: E402
    build_features_wt, load_raw_data_wt, prepare_X,
)
from src.preprocessing.upset_features import (  # noqa: E402
    build_upset_row, feature_vector,
)
from src.wt_vintage_config import assert_vintage_for_past  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_9H1_NE, RANK_9H1_SCORE_MIN, rank_9h1_build_legs, rank_9h1_daily_select,
    rank_9h1_stakes,
)

RACE_COLS = ("n_entries", "grade", "race_type", "day_index", "distance",
             "start_at", "bank_length", "is_indoor")


def _load_day(date: str) -> dict[str, list[dict]]:
    """対象日の9車レースの出走表を race_key ごとに返す。"""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT e.race_key, r.race_date, r.venue_id, r.race_no,
                   r.n_entries, r.grade, r.race_type, r.day_index, r.distance,
                   r.start_at,
                   e.frame_no, e.name, e.race_point, e.line_group, e.line_size,
                   e.style, e.player_class, e.s_count, e.b_count,
                   e.first_rate, e.third_rate, e.prediction_mark,
                   v.bank_length, v.is_indoor, v.name AS venue_name
            FROM wt_entries e
            JOIN wt_races r USING(race_key)
            LEFT JOIN venue_info v ON v.venue_code = r.venue_id
            WHERE r.cancel=0 AND r.n_entries={RANK_9H1_NE} AND r.race_date='{date}'
        """)
        by_race: dict[str, list[dict]] = defaultdict(list)
        for e in cur:
            by_race[e["race_key"]].append(dict(e))
    return by_race


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="対象日 YYYY-MM-DD")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-model", default="lgbm_wt_eval",
                    help="3着内率モデル（買い目の並び順に使う）")
    ap.add_argument("--screen-model", default="lgbm_upset_screen",
                    help="波乱スコアモデル（レース選別に使う）")
    ap.add_argument("--score-min", type=float, default=RANK_9H1_SCORE_MIN,
                    help="波乱スコアの採用閾値。**vintage を使うときは必ず併せて渡す**")
    args = ap.parse_args()

    # 🔴 過去日を本番モデル（全期間学習）でスコアすると model-vintage look-ahead に
    #    なる。既定値が本番モデル名なので、指定を忘れると**無言で**そうなっていた
    #    （2026-08-08 に 7H1 側で機械的に止める仕組みが入ったので追随する）。
    assert_vintage_for_past(
        args.date, {"eval": args.eval_model, "screen": args.screen_model})

    by_race = _load_day(args.date)
    if not by_race:
        print(f"{args.date}: 9車立てのレースがありません")
        _write([], args.date, args.out)
        return
    print(f"{args.date}: 9車立て {len(by_race)}R")

    screen = load_model(args.screen_model)

    # 選手単位の3着内率（**買い目の並び順にだけ**使う。レース選別には使わない）。
    # 期間指定は既存の 7H1 候補生成（`build_7h1_candidates.py`）と同じ形に揃える。
    feats = build_features_wt(load_raw_data_wt(min_date=args.date, max_date=args.date))
    p3: dict[str, dict[int, float]] = {}
    if feats is not None and len(feats):
        # ⚠️ 3着内率モデルは sklearn API（`predict_proba`）、波乱スコアは
        #    LightGBM Booster（`predict`）で**呼び方が違う**。取り違えると
        #    確率でない値が入る。
        probs = load_model(args.eval_model).predict_proba(prepare_X(feats))[:, 1]
        for rk, fn, pr in zip(feats["race_key"], feats["frame_no"], probs):
            p3.setdefault(rk, {})[int(fn)] = float(pr)

    cands = []
    for rk, ents in by_race.items():
        race = {k: ents[0].get(k) for k in RACE_COLS}
        row = build_upset_row(ents, race)
        if row is None:
            continue                       # 事前欠車で9行そろわない＝母集団外
        probs = p3.get(rk)
        if not probs or len(probs) != RANK_9H1_NE:
            print(f"  {rk}: 3着内率が {len(probs or {})}/{RANK_9H1_NE} 件しか無く skip")
            continue
        legs = rank_9h1_build_legs(probs)
        if not legs:
            continue
        score = float(screen.predict(np.array([feature_vector(row)]))[0])
        unit, total = rank_9h1_stakes(len(legs))
        name_of = {int(e["frame_no"]): e.get("name") for e in ents}
        order = [f for f, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
        cands.append({
            "race_key": rk, "race_date": ents[0]["race_date"],
            "venue_name": ents[0].get("venue_name"), "race_no": ents[0].get("race_no"),
            "start_time": ents[0].get("start_at"),
            "race_type": ents[0].get("race_type"),
            "n_entries": RANK_9H1_NE,
            "upset_score": round(score, 6),
            "order": order,                    # モデル3着内率の降順（印の割当に使う）
            "lead": order[4],                  # 1着固定車（＝3着内率5位）
            "lead_name": name_of.get(order[4]),
            "legs": legs, "stake": unit, "bet_amount": total,
        })

    picked = rank_9h1_daily_select(cands, score_min=args.score_min)
    print(f"波乱スコア >= {args.score_min:.4f} で {len(picked)}/{len(cands)}R を採用")
    for c in picked:
        print(f"  {c['race_key']} {c.get('venue_name')}{c.get('race_no')}R "
              f"score={c['upset_score']:.4f} 1着固定={c['lead']}番 "
              f"{len(c['legs'])}点×{c['stake']:,}円")
    _write(picked, args.date, args.out)


def _write(cands: list[dict], date: str, out: str | None) -> None:
    # 既存ランクと同じ data/picks/ 配下へ出す（notify_prerace_wt.py が読む場所）
    path = Path(out) if out else (
        REPO / "data" / "picks" / f"wave_picks_wt_{date}_s9h1_candidates.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")     # 書き込み途中を読まれないよう原子的に置く
    tmp.write_text(json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(f"[保存先] {path}  (9H1候補 {len(cands)}件)")


if __name__ == "__main__":
    main()
