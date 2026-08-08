"""RANK_9H1（9車・高配当狙い）の「波乱スコア」特徴量の単一正本。

## このモジュールが存在する理由

9H1 の選別は「**そのレースが高配当で決着するか**」を予測する**レース単位**の
二値分類で、既存ランクの選手単位モデル（`lgbm_wt_eval` / `_win` / `_bad`）とも、
7H1 の本命単位モデル（`favbust_features`）とも粒度が違う。

**学習（`scripts/train_upset_screen.py`）・過去分の再構築（`scripts/backfill_9h1_rank_wt.py`）・
本番の候補生成（`scripts/build_9h1_candidates.py`）がすべてこのモジュールを通る。**
検証で使ったものと本番で動くものが別実装になる事故を構造的に防ぐのが目的なので、
**ここ以外で波乱スコアの特徴量を組み立ててはいけない**。

## 🔴 6/7/9車を1つの母集団として学習する（車数を特徴に入れる）

9車は 4,738R しか無く、9車だけで学習すると効果が検出できない
（Δratio +0.090 で 95%CI の下限が +0.001）。6/7/9車を統合すると
**同じ9車の評価が Δratio +0.131 [+0.028, +0.230]・月次一貫性 75%→90%** へ改善する。
壁は標本数そのものだった。

そのため本モジュールの特徴は **車数が違っても意味が変わらない形**に揃えてある:

- ライン数・単騎数・脚質の人数は **すべて出走車数で割った割合**にする
- 競走得点の分散・上位との差は元々スケールフリーなのでそのまま使う
- `n_entries` 自体も特徴に入れる（車数固有の水準はモデルに吸収させる）

⚠️ 学習の目的変数も**車数ごとの分位**で定義すること（絶対オッズ閾値のままだと
基準率が 7車 9.8% / 9車 24.0% と違うため、モデルは「これは9車か」を当てるだけに
なる）。詳細は `scripts/train_upset_screen.py`。

## 🔴 オッズを使わない

本ランクは**朝の入稿で完結する**ことが要件なので、特徴は出走表と番組表だけから
作る。オッズを入れれば精度は上がるが、それは「市場の写し」になって回収率にならない
（市場と同じ向きの分類器は精度がどれだけ高くても ROI にならない）。
`wt_entries.pred_win_pct` / `pred_top3_pct` は **WINTICKET が表示する予想率**で
オッズではないが、2024-01 以降しか埋まっていないため**特徴には含めない**
（買い目の並び順にだけ使う。`RANK_9H1` の1着固定順位）。
"""
from __future__ import annotations

import math
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))

#: 特徴量の順序の正本。学習・推論の両方がこの順で組む。
#: ⚠️ 増減したら必ず再学習すること（`load_model` は列数不一致で落ちる）。
UPSET_FEATURE_COLS: tuple[str, ...] = (
    # 出走車数（車数固有の水準をモデルに吸収させるために明示的に入れる）
    "n_entries",
    # 競走得点の分布 — 「番組がフラットか」。車数に依らない量
    "rp_std", "rp_mean", "rp_range", "rp_gap12", "rp_gap23", "rp_top2_edge",
    # ライン構成 — 競輪はライン戦なので隊列の作りが決着形を決める。**割合**で持つ
    "line_ratio", "max_line_ratio", "solo_ratio", "line_rp_gap12", "line_rp_std",
    # 脚質構成 — 逃げが多いと潰し合いになるという定番の仮説。**割合**で持つ
    "nige_ratio", "makuri_ratio", "oikomi_ratio",
    # 実績のばらつき
    "first_max", "first_std", "third_std", "s_mean", "b_mean",
    # 級班の混在度（同じ級班ばかりだと力が拮抗する）
    "class_ratio", "top_class_share",
    # 番組・会場
    "day_index", "distance", "bank_length", "is_indoor", "hour",
    "grade_enc", "rtype_enc",
    # WT公式印（記者の本命がどこにいるか。市場の代理変数だがオッズではない）
    "mark1_rp_rank_ratio", "mark1_line_size_ratio",
)


def _enc(v: Any) -> int:
    """カテゴリの**決定的**ハッシュ。

    ⚠️ 組み込みの `hash()` を使ってはいけない。文字列ハッシュはプロセスごとに
    ランダム化される（PYTHONHASHSEED）ため、**学習時と推論時で別の値になり、
    同じ入力でも結果が変わる**。検証段階で実際に踏み、同一条件の再実行で
    評価指標が大きく動いた。
    """
    return zlib.crc32(str(v).encode()) % 1000 if v is not None else -1


def _hour(start_at: Any) -> int:
    """発走時刻（JST の時）。モーニング / 通常 / ナイター / ミッドナイトの別を粗く表す。"""
    try:
        return datetime.fromtimestamp(int(start_at), timezone.utc).astimezone(JST).hour
    except (TypeError, ValueError, OSError, OverflowError):
        return -1


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_upset_row(entries: list[dict], race: dict) -> dict[str, float] | None:
    """1レース分の波乱スコア特徴を組む。組めない場合は None。

    Args:
        entries: そのレースの `wt_entries` 行（**出走車数ぶん全部**）。
                 事前欠車は行自体が消えるため、`race["n_entries"]` と行数が
                 食い違うレースは「その車数で発走していない」＝対象外にする。
        race:    `wt_races` 行 + 会場情報（`bank_length` / `is_indoor`）。

    Returns:
        `UPSET_FEATURE_COLS` と同じキーを持つ dict。
    """
    ne = int(race.get("n_entries") or 0)
    if ne < 5 or len(entries) != ne:
        return None

    rp = sorted((_f(e.get("race_point")) for e in entries), reverse=True)
    lines: dict[Any, list[dict]] = defaultdict(list)
    for e in entries:
        key = e.get("line_group")
        lines[key if key is not None else f"solo{e.get('frame_no')}"].append(e)
    line_rp = sorted((sum(_f(x.get("race_point")) for x in v) for v in lines.values()),
                     reverse=True)
    styles = Counter((e.get("style") or "")[:1] for e in entries)
    classes = Counter(e.get("player_class") or "" for e in entries)
    first = [_f(e.get("first_rate")) for e in entries]
    third = [_f(e.get("third_rate")) for e in entries]

    mark1 = next((e for e in entries if e.get("prediction_mark") == 1), None)
    rp_rank = {e.get("frame_no"): i for i, e in
               enumerate(sorted(entries, key=lambda x: -_f(x.get("race_point"))))}

    return {
        "n_entries": float(ne),
        "rp_std": _std(rp), "rp_mean": sum(rp) / ne,
        "rp_range": rp[0] - rp[-1],
        "rp_gap12": rp[0] - rp[1], "rp_gap23": rp[1] - rp[2],
        "rp_top2_edge": sum(rp[:2]) / 2 - sum(rp[2:]) / (ne - 2),
        "line_ratio": len(lines) / ne,
        "max_line_ratio": max(len(v) for v in lines.values()) / ne,
        "solo_ratio": sum(1 for v in lines.values() if len(v) == 1) / ne,
        "line_rp_gap12": (line_rp[0] - line_rp[1]) if len(line_rp) > 1 else 0.0,
        "line_rp_std": _std(line_rp),
        "nige_ratio": styles.get("逃", 0) / ne,
        "makuri_ratio": styles.get("捲", 0) / ne,
        "oikomi_ratio": styles.get("追", 0) / ne,
        "first_max": max(first), "first_std": _std(first), "third_std": _std(third),
        "s_mean": sum(int(e.get("s_count") or 0) for e in entries) / ne,
        "b_mean": sum(int(e.get("b_count") or 0) for e in entries) / ne,
        "class_ratio": len(classes) / ne,
        "top_class_share": max(classes.values()) / ne,
        "day_index": float(race.get("day_index") or 0),
        "distance": float(race.get("distance") or 0),
        "bank_length": _f(race.get("bank_length")),
        "is_indoor": float(race.get("is_indoor") or 0),
        "hour": float(_hour(race.get("start_at"))),
        "grade_enc": float(_enc(race.get("grade"))),
        "rtype_enc": float(_enc(race.get("race_type"))),
        "mark1_rp_rank_ratio": (rp_rank.get(mark1.get("frame_no"), -1) / ne
                                if mark1 is not None else -1.0),
        "mark1_line_size_ratio": (int(mark1.get("line_size") or 1) / ne
                                  if mark1 is not None else -1.0),
    }


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def feature_vector(row: dict[str, float]) -> list[float]:
    """`build_upset_row()` の戻り値を `UPSET_FEATURE_COLS` の順に並べる。"""
    return [float(row[c]) for c in UPSET_FEATURE_COLS]
