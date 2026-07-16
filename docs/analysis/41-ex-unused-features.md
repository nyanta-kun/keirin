# doc41: EXデータ未使用3列の Phase1 AUC 実験（2026-06-15）

> **Phase1 不通過（全モデル AUC +0.0001〜+0.0002）・採用見送り**

## 仮説

`wt_entries` に格納済みだが `FEATURE_COLS_WT` に未収録の3列を追加することで
AUC が改善し、予測精度向上につながるか検証する。

## 対象列

| 列名 | 意味 | nonzero率 | corr(top3) |
|---|---|---|---|
| `ex_left_behind_pct` | ちぎられ率 | 33.0% | +0.081 |
| `ex_split_line_pct` | ちぎり率（前者をちぎった率） | 43.5% | -0.072 |
| `ex_snatch_pct` | 飛びつき成功率 | 15.7% | +0.035 |

※ `ex_spurt_pct`・`ex_thrust_pct` は既に `FEATURE_COLS_WT` 収録済み

## 実験条件

| 項目 | 設定 |
|---|---|
| 期間 | TRAIN: 2023-07-01〜2025-06-30 / VAL: 2025-07-01〜2026-02-28 / HOLD: 2026-03-01〜2026-06-15 |
| 学習サンプル数 | 382,955 (finish_order >= 1) |
| モデル | LGBMClassifier (n_estimators=500, lr=0.05, leaves=31) |
| 比較対象 | Base / +left / +split / +snatch / +all3 |
| Phase1 gate | VAL+HOLD AUC ≥ +0.001 で "★" |

## 実験結果

### Phase1: AUC

| 期間 | Base | +left_behind | +split_line | +snatch | +all3 |
|---|---|---|---|---|---|
| VAL | 0.7721 | +0.0002 | +0.0000 | +0.0001 | +0.0003 |
| HOLD | 0.7763 | +0.0002 | +0.0002 | +0.0001 | +0.0002 |
| **VAL+HOLD** | **0.7734** | **+0.0002** | **+0.0001** | **+0.0001** | **+0.0002** |

Phase1: **不通過**（最大 +0.0002・閾値 +0.001 未満）

### 特徴量重要度（+all3 モデル・上位12）

3つの新規列はいずれも上位12に入らなかった。
（`score_z` 7.4%, `race_point` 7.2%, `line_frac` 5.4%, ... `win_6m` 3.4% の順）

### Phase2: ROI

Phase1 不通過のためスキップ。

## 解釈

**なぜ効果がないか**:

1. **`ex_spurt_pct`・`ex_thrust_pct` が代替済み**: 同じ上がり戦術系の2列が既に収録されており、
   追加3列はほぼ同一の戦術傾向を重複して表現するため情報増分が小さい。

2. **corr の低さ**: `ex_snatch_pct` は corr(top3)=+0.035 と極めて小さく、
   そもそも3着以内との相関が弱い。`ex_split_line_pct` は -0.072 と
   負の相関を持つが、この方向性は `ex_spurt_pct`（強い末脚=勝ちに行く）で
   既に捉えられている可能性が高い。

3. **疎性**: `ex_snatch_pct` は nonzero 率 15.7%。LightGBM は疎な特徴量から
   有効なスプリットを学習しにくく、ほぼ定数扱いになる。

4. **市場への織り込み**: `prediction_mark`（winticket AI 印）が市場参加者の
   戦術評価を集約しており、個別 EX 列の追加情報を吸収している。

## 結論

| 特徴量 | 判定 |
|---|---|
| `ex_left_behind_pct` | Phase1 不通過（+0.0002）・採用見送り |
| `ex_split_line_pct` | Phase1 不通過（+0.0001）・採用見送り |
| `ex_snatch_pct` | Phase1 不通過（+0.0001）・採用見送り |
| `+all3` 組み合わせ | Phase1 不通過（+0.0002）・採用見送り |

既存 DB に収録済みの未使用 EX 列はこれで全て検証済み。
新しい EX 列を追加するには新規スクレイプが必要。

## ハーネス

```bash
python3 scripts/exp_ex_unused_wt.py
```
