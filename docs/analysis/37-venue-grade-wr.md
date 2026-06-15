# doc37: venue × grade 限定 rolling WR 特徴量実験（2026-06-15）

> **Phase1 不通過（AUC +0.0001）・採用見送り**

## 仮説

現行 `venue_wr` は選手の会場別生涯勝率（S/A/L 混合）。
`grade_group`（S/A）で分割した `venue_grade_wr` を追加することで
AUC・ROI が改善するか検証した。

## 実験結果

### Phase1: AUC

| 期間 | Base AUC | 拡張 AUC | 差分 |
|---|---|---|---|
| VAL | 0.7721 | 0.7722 | +0.0001 |
| HOLD | 0.7764 | 0.7763 | -0.0001 |
| **VAL+HOLD** | **0.7734** | **0.7734** | **+0.0001** |

Phase1: **不通過**（閾値 +0.001 に対し +0.0001）

### Phase2: ROI

| 期間 | Base ROI | 拡張 ROI | 差分 | n(base/ext) |
|---|---|---|---|---|
| TRAIN | 83.6% | 86.7% | +3.2pp | 307/319 |
| VAL | 67.7% | 72.5% | +4.8pp | 72/75 |
| HOLD | 87.5% | 82.9% | -4.6pp | 27/30 |

Phase2: **未評価**（Phase1 不通過のため）

### 特徴量重要度

`venue_grade_wr` は上位 10 特徴量に入らず（≈0.5%以下）。

### データ特性

- `venue_grade_wr` 非ゼロ率: 39.8%（60.2% が初回出走で履歴なし = 0 埋め）
- TRAIN での非ゼロ率: 35.7%

## 解釈

**なぜ効果がないか**:
- `venue_wr`（会場×混合勝率）と `player_class_enc`（選手クラス）の組み合わせが
  `venue_grade_wr`（会場×grade勝率）の情報をほぼ再現できてしまう
- grade_enc バグ修正（doc35）でも AUC への影響 +0.0001 だったことと整合
- 競輪では選手クラス（S1/S2/A1/A2）が会場×グレードの成績分布を既に内包

## 結論

| 項目 | 判定 |
|---|---|
| venue_grade_wr | Phase1 不通過・採用見送り |

## ハーネス

```bash
python3 scripts/exp_venue_grade_wr_wt.py
```
