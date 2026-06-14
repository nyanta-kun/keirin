# W03: 波乱スコア × fav_mismatch 交差分析

## 目的

**upset スコア Q4（高波乱確率）と fav_mismatch（モデル1位≠市場本命）の交差条件**を
doc18 セマンティクスで評価する。両シグナルは独立した機序を持つため、組み合わせが
単独よりも強い edge を持つ可能性がある。

| シグナル | 機序 | 単独リーク無し結果 |
|----------|------|-------------------|
| upset Q4 | ライン構造的に崩れやすい | C1=69-76%（不通過） |
| fav_mismatch | 市場価格の誤り | C2=不通過（死亡） |
| **交差** | 両方が同時に成立 | **未検証**（本タスク） |

独立なら掛け合わせで edge が生まれる可能性がある。逆に相関が高ければ情報は重複。

## 背景（必読）

- `docs/analysis/13-shape-style-market-gap.md`: fav_mismatch の定義と元バックテスト結果
- `docs/analysis/18-backtest-bias-rescore.md`: C2(fav_mismatch)の leak-free 結果（死亡）
- `src/cli/main.py:1434`: `fav_mismatch` タグ記録（wave-picks-wt で既に実装済み）
- `scripts/exp_leakfree_rescore_wt.py:75`: `market_fav()` の実装（market_fav は trio 盤面から逆算）

### fav_mismatch の定義（再確認）

```python
# trio盤面のオッズから市場本命を特定（逆算）
mkt_fav = market_fav(trio_board)  # 最も低い3連複オッズに含まれる頻度が最大の選手
fav_mismatch = (mkt_fav is not None) and (mkt_fav != pivot1)
# pivot1 = モデル pred_prob 最大の選手
```

### upset スコアの定義（W01 の再学習モデルを使用）

```python
# upset_model.py の build_race_features + predict_proba
# Q4 = スコア上位25%（TRAIN期間でパーセンタイルを固定）
```

## 実験手順

### Step 1: 交差レース特定

各レースで:
1. upset_prob を計算（TRAIN期間学習モデル）
2. fav_mismatch フラグを計算（market_fav vs pivot1 比較）
3. 交差セル定義:
   - Cell A: upset Q4 + fav_mismatch=True
   - Cell B: upset Q4 + fav_mismatch=False
   - Cell C: upset Q1-Q3 + fav_mismatch=True
   - Cell D: upset Q1-Q3 + fav_mismatch=False（ベースライン）

### Step 2: ROI評価

全4セルで現行戦略（SS=3連単・S/A=3連複・pivot1-pivot2-{3rd}・ガミ≥5）を適用:
- doc18 セマンティクス（全エントリーランキング・出走表基準≤6車・欠車void）
- TRAIN / VAL / HOLD 各期間で集計
- bootstrap CI（1000回）

### Step 3: 独立性チェック

upset Q4 と fav_mismatch の共起率（φ係数・χ²検定）を計算し、
両シグナルが独立かどうかを確認。相関が高いなら情報重複として記録。

### Step 4: 結論判定

**通過基準**: Cell A（交差）が VAL・HOLD 両方で bootstrap CI 下限 > 100%

## 成果物

1. **`scripts/exp_upset_fav_mismatch_wt.py`**: 実験スクリプト
2. **`docs/analysis/29-upset-fav-mismatch.md`**: 結果レポート

## レポートテンプレ（`docs/analysis/29-upset-fav-mismatch.md`）

```markdown
# 29: 波乱スコア × fav_mismatch 交差分析

## 0. 結論
- Cell A（交差）の VAL・HOLD ROI と CI
- 独立性（φ係数・共起率）
- 通過/不通過の判定と理由

## 1. セル定義と各セルのサンプル数（R数）
## 2. ROI × セル × 期間（3期間）
## 3. bootstrap CI
## 4. 独立性・相関分析
## 5. 機序の考察（なぜ交差が効く/効かないか）
## 6. 結論
```

## 受け入れ基準

- `python scripts/exp_upset_fav_mismatch_wt.py` が完走（エラーなし）
- 4セル × 3期間の ROI 表と CI が出力されること
- 独立性指標（φ係数）が報告されること
- 既存テスト pass

## 触ってよいファイル

- `scripts/exp_upset_fav_mismatch_wt.py`（新規作成）
- `docs/analysis/29-upset-fav-mismatch.md`（新規作成）

## 禁止事項

- `src/` 配下の変更
- 本番モデル・cron の変更
- git commit
