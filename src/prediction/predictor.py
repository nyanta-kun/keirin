"""
予想生成モジュール

学習済みモデルを使って、指定レースの3連複・3連単予想を出力する。
"""
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..preprocessing.feature_engineer import FEATURE_COLS, build_features, LINE_POSITION_MAP
from ..database import get_connection


@dataclass
class RacePrediction:
    race_key: str
    rider_probs: list[dict]          # [{frame_no, player_name, prob}, ...]
    trifecta_box: list[dict]         # 3連複上位N点 [{combo, score}, ...]
    trifecta: list[dict]             # 3連単上位N点
    line_prediction: list[list[int]] # ライン予想 [[先頭枠, 番手枠, ...], ...]


def predict_race(model, race_key: str, top_n: int = 10) -> RacePrediction | None:
    """1レースの予想を生成"""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT e.frame_no, e.player_id, p.name as player_name,
                   e.gear_ratio, e.racing_score,
                   e.recent_win_rate_3m, e.recent_top3_rate_3m,
                   e.recent_win_rate_6m, e.recent_top3_rate_6m,
                   e.days_since_last_race, e.venue_win_rate,
                   e.quinella_rate, e.period, e.player_class,
                   e.line_position,
                   e.prefecture AS player_prefecture,
                   r.grade, r.distance, r.venue_code,
                   vi.bank_length, vi.is_indoor,
                   vi.prefecture AS venue_prefecture
            FROM race_entries e
            JOIN races r ON e.race_key = r.race_key
            LEFT JOIN players p ON e.player_id = p.player_id
            LEFT JOIN venue_info vi ON vi.venue_code = r.venue_code
            WHERE e.race_key = ?
            ORDER BY e.frame_no
        """, (race_key,)).fetchall()

    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    df["race_key"] = race_key
    df["race_date"] = "2099-01-01"  # 予測時はダミー
    df["finish_position"] = np.nan
    df["top3_flag"] = 0

    df = build_features(df)

    # 欠損補完
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    X = df[FEATURE_COLS].values
    probs = model.predict_proba(X)[:, 1]
    df["prob"] = probs

    # 選手別確率
    rider_probs = []
    for _, row in df.sort_values("prob", ascending=False).iterrows():
        rider_probs.append({
            "frame_no": int(row["frame_no"]),
            "player_name": row.get("player_name") or f"枠{int(row['frame_no'])}",
            "prob": float(row["prob"]),
            "racing_score": row.get("racing_score"),
            "gear_ratio": row.get("gear_ratio"),
            "line_position": row.get("line_position"),
        })

    frames = df["frame_no"].tolist()

    # 3連複: C(n,3) の組み合わせを確率の積でスコアリング
    prob_map = dict(zip(df["frame_no"].tolist(), probs.tolist()))
    box_combos = []
    for combo in itertools.combinations(sorted(frames), 3):
        score = prob_map[combo[0]] * prob_map[combo[1]] * prob_map[combo[2]]
        box_combos.append({
            "combo": "=".join(map(str, combo)),
            "frames": list(combo),
            "score": float(score),
        })
    box_combos.sort(key=lambda x: x["score"], reverse=True)

    # 3連単: 上位3連複を展開して順列スコアリング
    straight_combos = []
    for bc in box_combos[:20]:
        for perm in itertools.permutations(bc["frames"]):
            # 着順確率の近似: 1着の重み付け（1着は最高確率者が来やすい）
            weights = [prob_map[f] ** (3 - i) for i, f in enumerate(perm)]
            score = np.prod(weights)
            straight_combos.append({
                "combo": "-".join(map(str, perm)),
                "frames": list(perm),
                "score": float(score),
            })
    straight_combos.sort(key=lambda x: x["score"], reverse=True)

    # ライン予想（脚質から簡易推定）
    line_pred = _estimate_lines(df)

    return RacePrediction(
        race_key=race_key,
        rider_probs=rider_probs,
        trifecta_box=box_combos[:top_n],
        trifecta=straight_combos[:top_n],
        line_prediction=line_pred,
    )


def _estimate_lines(df: pd.DataFrame) -> list[list[int]]:
    """脚質から簡易ライン推定（先行→番手の順）"""
    lines: dict[str, list[int]] = {"先行": [], "捲り": [], "差し": [], "追い込み": [], "不明": []}
    for _, row in df.iterrows():
        pos = row.get("line_position") or "不明"
        lines.setdefault(pos, []).append(int(row["frame_no"]))

    result = []
    for pos in ["先行", "捲り", "差し", "追い込み", "不明"]:
        if lines.get(pos):
            result.append(lines[pos])
    return result


def format_prediction(pred: RacePrediction) -> str:
    """予想結果を文字列フォーマット"""
    lines = [f"\n{'='*50}", f"レース: {pred.race_key}", f"{'='*50}"]

    lines.append("\n【選手別 3着内確率】")
    for r in pred.rider_probs:
        bar = "█" * int(r["prob"] * 20)
        lines.append(
            f"  枠{r['frame_no']} {r['player_name']:<8} "
            f"{r['prob']:.1%} {bar}  "
            f"(得点:{r['racing_score'] or '-':.1f}  ギア:{r['gear_ratio'] or '-'})"
        )

    lines.append("\n【ライン予想】")
    for i, line in enumerate(pred.line_prediction, 1):
        if line:
            lines.append(f"  ライン{i}: {' → '.join(map(str, line))}")

    lines.append(f"\n【3連複 上位予想】")
    for i, bc in enumerate(pred.trifecta_box[:5], 1):
        lines.append(f"  {i}. {bc['combo']}  (スコア:{bc['score']:.4f})")

    lines.append(f"\n【3連単 上位予想】")
    for i, sc in enumerate(pred.trifecta[:5], 1):
        lines.append(f"  {i}. {sc['combo']}  (スコア:{sc['score']:.4f})")

    return "\n".join(lines)
