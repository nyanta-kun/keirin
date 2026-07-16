# G41: EXデータ未使用3列の Phase1 AUC 実験

## 目的

WINTICKET EXデータのうち FEATURE_COLS_WT に未収録の3列を追加し、
Phase1 AUC ゲートを突破するか検証する。
DBに100%収録済みのためスクレイプ不要・即実施可能。

## 背景

`wt_entries` には以下3列が存在するが FEATURE_COLS_WT に含まれていない：

| 列名 | 意味 | nonzero率 | corr(top3) |
|---|---|---|---|
| `ex_left_behind_pct` | ちぎられ率（番手が先頭についていけなかった率） | 33.3% | +0.081 |
| `ex_split_line_pct` | ちぎり率（先頭が番手を引き離した率） | 43.9% | -0.070 |
| `ex_snatch_pct` | 飛びつき成功率（別ラインへの割り込み率） | 14.7% | +0.035 |

`ex_spurt_pct`（かまし）・`ex_thrust_pct`（つっぱり）は既に FEATURE_COLS_WT 収録済み。

### 期間定義（全タスク共通）

| 期間 | 範囲 |
|---|---|
| TRAIN | 2023-07-01 〜 2025-06-30 |
| VAL   | 2025-07-01 〜 2026-02-28 |
| HOLD  | 2026-03-01 〜 2026-06-15 |

### ゲート基準
- Phase1: VAL+HOLD の AUC 改善 ≥ +0.001（★マーク）
- Phase2: ROI >100% 全3期間（Phase1通過時のみ評価）

### 参考: 類似実験スクリプト
`scripts/exp_line_cohesion_wt.py` が同じ構造（Phase1→Phase2）を実装済み。
インポートは以下を使う:
```python
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X, FEATURE_COLS_WT
from exp_segment_first_wt import TRAIN, VAL, HOLD, LGB_PARAMS
from src.database import get_connection
```

## 成果物

1. **`scripts/exp_ex_unused_wt.py`**: 実験スクリプト
2. **`docs/analysis/41-ex-unused-features.md`**: 結果レポート

## 実験手順

### Step 1: データロード

```python
df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))
```

`ex_left_behind_pct`, `ex_split_line_pct`, `ex_snatch_pct` は
`build_features_wt()` が返す DataFrame に wt_entries の列として既に含まれている。

### Step 2: 分布確認

各列の nonzero 率・平均値（top3=1 vs top3=0）を表示。

### Step 3: Phase1 AUC 比較

以下6モデルを比較（全て TRAIN 期間のみ `finish_order >= 1` で学習）:
- Base: `FEATURE_COLS_WT`
- +left: `FEATURE_COLS_WT + ["ex_left_behind_pct"]`
- +split: `FEATURE_COLS_WT + ["ex_split_line_pct"]`
- +snatch: `FEATURE_COLS_WT + ["ex_snatch_pct"]`
- +all3: `FEATURE_COLS_WT + ["ex_left_behind_pct", "ex_split_line_pct", "ex_snatch_pct"]`

各モデルを VAL・HOLD・VAL+HOLD で `roc_auc_score` 評価。
VAL+HOLD で改善 ≥ +0.001 なら「★」を表示。

### Step 4: 特徴量重要度（+all3 モデル）

`feature_importances_` で上位12特徴量を表示。新3列の順位を記録。

### Step 5: Phase2 ROI（Phase1通過時のみ）

`exp_line_cohesion_wt.py` の `compute_roi_records()` 関数と同じパターンで
C0戦略（trio・ガミ≥5倍・≤6車）の ROI を TRAIN/VAL/HOLD 各期間で計算。

## レポートテンプレ（`docs/analysis/41-ex-unused-features.md`）

```markdown
# doc41: EXデータ未使用3列特徴量実験（2026-06-15）

> **結論**: [Phase1 通過/不通過・採用/不採用]

## 仮説
（略）

## 新特徴量
（列ごとに nonzero 率・top3 平均差を記載）

## Phase1: AUC
| 期間 | Base | +left | +split | +snatch | +all3 |
|---|---|---|---|---|---|

## 特徴量重要度（+all3・上位12）
（表）

## Phase2: ROI（Phase1通過時のみ）
（表）

## 結論
（採用/不採用・理由）

## ハーネス
python3 scripts/exp_ex_unused_wt.py
```

## 受け入れ基準

- `python3 scripts/exp_ex_unused_wt.py` がエラーなく完了すること
- `docs/analysis/41-ex-unused-features.md` に Phase1 AUC 表・特徴量重要度・結論が記載されること
- Phase1 通過/不通過の判定が明記されること

## 触ってよいファイル

- `scripts/exp_ex_unused_wt.py`（新規作成）
- `docs/analysis/41-ex-unused-features.md`（新規作成）

## 禁止事項

- `src/preprocessing/feature_wt.py` の変更（FEATURE_COLS_WT 更新は本タスクのスコープ外）
- `data/` 配下の DB ファイルの変更（読み取りのみ）
- git commit
- crontab 変更
