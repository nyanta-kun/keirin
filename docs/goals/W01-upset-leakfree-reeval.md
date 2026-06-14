# W01: 波乱モデル × リーク無し再評価（doc18対応）

## 目的

`upset_model.py` のインフラを使い、**リーク無しセマンティクス（doc18）** で波乱スコアとROIの関係を再評価する。
既存の `lgbm_upset.pkl` は 2026-03-01 まで学習済み（リーク）のため、TRAIN期間のみで再学習し
VAL・HOLDOUT で「高波乱確率レースを選別するとROIが改善するか」を検証する。

## 背景（必読）

- `docs/analysis/02-upset-prediction.md`: 波乱スコア Q4 → ROI 598%/627%（biasあり・本タスクの再評価対象）
- `docs/analysis/18-backtest-bias-rescore.md`: 3バイアス（①欠車生存・②≤6車完走者基準・③モデルリーク）
- `src/evaluation/upset_model.py`: 波乱モデルの学習・評価インフラ（trio払戻 ≥ 2000 が波乱定義）
- `src/evaluation/backtest_wt.py`: G01修正済み（doc18対応・void_rules.py共通化済み）
- `data/models/lgbm_wt_eval.pkl`: リーク無し評価用モデル（2023-07〜2025-06学習・TEST=2026-03〜）

### 期間定義（全タスク共通）

| 期間 | 範囲 | 用途 |
|------|------|------|
| TRAIN | 2023-07-01 〜 2025-06-30 | 波乱モデル学習 |
| VAL   | 2025-07-01 〜 2026-02-28 | 中間評価 |
| HOLDOUT | 2026-03-01 〜 2026-06-14 | 最終一発評価 |

## 成果物

1. **`scripts/exp_upset_leakfree_wt.py`**: 実験スクリプト（下記手順を実装）
2. **`docs/analysis/27-upset-leakfree.md`**: 結果レポート（下記テンプレに従う）

## 実験手順

### Step 1: データ準備（doc18セマンティクス）

```python
# 1. build_features_wt で全エントリー取得（min_date=2023-07-01）
# 2. 出走表基準での≤6車フィルタ（race_key ごとの行数で判定・完走者数ではない）
# 3. 結果確定レースのみ（finish_order between 1,3 で3頭そろうもの）
# 4. ランキングは全エントリー（欠車=finish_order=0 を含めてprob降順でrank付け）
```

### Step 2: 波乱モデルの再学習（リーク無し）

```python
# upset_model.build_race_features(df_train) でレース特徴量を構築
# upset_model.add_upset_target(df_race, upset_threshold=2000) で is_upset フラグ
# TRAIN期間のみで lgbm_upset_eval として学習・保存
# AUCをTRAIN/VAL/HOLD各期間で計算して報告
```

### Step 3: ROI評価（doc18セマンティクス）

波乱スコアの四分位（Q1〜Q4）でレースを層別し、各層で `backtest_wt.py` の標準戦略を実行:

- 戦略: wave-picks-wt デフォルト（SS=3連単・S/A=3連複・pivot1-pivot2-{3rd}・ガミ≥5のみ）
- 欠車処理: `void_rules.py` の `should_void_pick` 準拠
- 払戻: `wt_odds`（最終オッズ上限値・実運用は下振れ）
- 報告: Q1/Q2/Q3/Q4 × TRAIN/VAL/HOLD のROI表（的中率・R数・ROI・最大払戻除去ROI）

### Step 4: bootstrap CI

各セル（Q×期間）に対してbootstrap 1000回でROIの95%CIを計算。
**通過基準: VAL と HOLD の両方で ROI CI 下限 > 100%** を満たすセルが存在するか確認。

## レポートテンプレ（`docs/analysis/27-upset-leakfree.md`）

```markdown
# 27: 波乱モデル × リーク無し再評価

## 0. 結論（先出し）
- 波乱モデルAUC（TRAIN/VAL/HOLD）
- ROI単調性（Q1<Q2<Q3<Q4）が再現するか
- VAL・HOLD両方で CI下限>100%のセルが存在するか → YES/NO
- 推奨アクション

## 1. 波乱モデル性能（AUC）
## 2. ROI × 波乱スコア四分位（全期間）
## 3. bootstrap CI
## 4. 過学習チェック（doc02との比較）
## 5. 結論
```

## 受け入れ基準

- スクリプトが `python scripts/exp_upset_leakfree_wt.py` で実行完了すること（エラーなし）
- 結果レポートに Q1-Q4 × 3期間の ROI 表が記載されること
- 既存テスト pass（`python -m pytest tests/ -x -q` が通ること）

## 触ってよいファイル

- `scripts/exp_upset_leakfree_wt.py`（新規作成）
- `docs/analysis/27-upset-leakfree.md`（新規作成）
- `data/models/lgbm_upset_eval.pkl`（必要なら保存）

## 禁止事項

- `src/` 配下の変更（読み取りのみ）
- `lgbm_wt.pkl` / 本番モデルの変更
- git commit
- crontab 変更
