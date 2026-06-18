# W02: 波乱Q4 × 代替買い目設計（ライン崩れ前提）

## 目的

高波乱確率レース（upset スコア Q4・上位25%）において、**現行の pivot1-pivot2-{3rd} 戦略に代わる
買い目構造を探索する**。Q4 では86%確率でライン決着が崩れるため、pivot1・pivot2 が同一ライン
のケースでは両者とも圏外になりやすい。これを前提とした代替フォーメーションをリーク無しで評価する。

## 背景（必読）

- `docs/analysis/02-upset-prediction.md`: 波乱Q4はライン崩れ86%・現行current(3点)がQ4で最良
  → **ただしdoc18バイアスあり（ROI数値は無効）。買い目優劣の順序（current>案A>案B>box5）は頑健な可能性**
- `docs/analysis/18-backtest-bias-rescore.md`: doc18 3バイアス定義（必読）
- `src/evaluation/backtest_wt.py`: G01修正済み（doc18セマンティクス）
- `src/evaluation/upset_model.py`: 波乱スコア計算インフラ
- `data/models/lgbm_upset.pkl`: 既存波乱モデル（TRAIN期間限定版が利用可能な場合はそちらを優先）

### 期間定義

| TRAIN | 2023-07-01〜2025-06-30 | VAL | 2025-07-01〜2026-02-28 | HOLD | 2026-03-01〜2026-06-14 |

### 現行戦略の問題点（Q4において）

- pivot1: モデルrank1（多くの場合、強いラインの先頭）
- pivot2: モデルrank2（しばしば pivot1 と同一ライン）
- 3rd: rank3-5 から選出

ライン崩れ時: pivot1 のライン全体が沈む → pivot1・pivot2 両方が外れる → 現行3点がすべてハズレ

## 探索する代替フォーメーション

全フォーメーションで:
- 対象: ≤6車 × doc18セマンティクス × upset Q4（上位25%スコア）
- ガミ最安オッズ ≥ 5.0 倍フィルタ（本番同様）
- 払戻: `wt_odds`（最終オッズ上限値）

### F1: pivot1 × クロスライン軸（2軸流し）

pivot2 を「pivot1 と異なる line_group のモデル最高確率選手」に置き換える。
- pivot1: モデルrank1（既存と同じ）
- cross_pivot: pivot1 以外の line_group から pred_prob 最大の選手
- 3rd: モデル上位5名の残り（3点流し）
- 目的: ライン崩れ時に cross_pivot が台頭するシナリオを拾う

```python
# line_group が利用可能な場合（wt_entries.line_group）:
#   cross_pivot = df[df.line_group != pivot1.line_group].nlargest(1, 'pred_prob').iloc[0]
# line_group が NaN の場合:
#   cross_pivot = df[df.player_id != pivot1.player_id].nlargest(1, 'pred_prob').iloc[0]
```

### F2: pivot1 単軸 × ワイド選択（1軸多点流し）

pivot1 を軸に固定し、残り全員から上位N人を3rd候補として 1-{2nd_candidates}-(pivot1 以外全員) を購入:
- pivot1 固定 × rank2,3,4,5 から任意2頭 BOX（組み合わせ数は C(4,2)=6点）
- SS: trifecta pivot1 → (rank2-5全員から1頭) → (rank2-5から別の1頭)=C(4,2)=6点

### F3: 逆張り（下位線本命・ライン崩れ前提の直接購入）

ライン崩れを「非最強ライン先頭が1着」として解釈:
- second_line: pivot1 と異なる line_group のうち max pred_prob の選手
- third_line: 上位2ラインに属さない選手で最大 pred_prob
- second_line → pivot1 → 任意 の3連複（2点）

## 実験手順

1. `upset_model.py` で upset スコアを計算（TRAIN期間学習モデル使用）
2. Q4 レース（スコア上位25%）を抽出
3. 各フォーメーション（F1/F2/F3 + 現行 current）で ROI をTRAIN/VAL/HOLD で計算
4. bootstrap CI（1000回）で各セルの不確実性を定量化
5. doc02 の数字（current Q4=598%）と比較してどれだけ変化したかを記録

## 成果物

1. **`scripts/exp_upset_bet_design_wt.py`**: 実験スクリプト
2. **`docs/analysis/28-upset-bet-design.md`**: 結果レポート

## レポートテンプレ（`docs/analysis/28-upset-bet-design.md`）

```markdown
# 28: 波乱Q4 × 代替買い目設計

## 0. 結論
- 各フォーメーションの VAL・HOLD ROI（doc18）
- 現行 current に対する改善/悪化
- 推奨フォーメーション or 「全不通過・現行維持」

## 1. フォーメーション定義と点数
## 2. ROI比較（TRAIN/VAL/HOLD × フォーメーション）
## 3. bootstrap CI
## 4. 考察（なぜ差が生じるか/生じないか）
## 5. 結論
```

## 受け入れ基準

- `python scripts/exp_upset_bet_design_wt.py` が完走すること（エラーなし）
- current + F1/F2/F3 の TRAIN/VAL/HOLD × ROI 表が出力されること
- bootstrap CI が各セルで計算されること
- 既存テスト pass

## 触ってよいファイル

- `scripts/exp_upset_bet_design_wt.py`（新規作成）
- `docs/analysis/28-upset-bet-design.md`（新規作成）

## 禁止事項

- `src/` 配下の変更
- 本番モデル・cron の変更
- git commit
