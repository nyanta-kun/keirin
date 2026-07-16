# doc40: 師匠情報特徴量実験（2026-06-15）

> **結論**: [スクレイプ後に実行して記入]

## 仮説

keirin.jp に掲載されている JKA 登録師匠（メンター）の情報を特徴量化する。
師弟関係がライン形成・走法選択に影響を与えれば、予測精度を向上させる
可能性がある。

## 事前評価（実験前）

| 観点 | 内容 |
|---|---|
| データ入手性 | keirin.jp 静的 HTML・スクレイプ可能・robots.txt ALLOW |
| Coverage 見込み | ~40-60%（師匠情報未掲載選手が多い） |
| 期待度 | 弱（下記の事前根拠を参照） |
| 類似実験 | doc38 ライン連携コヒージョン（Phase1 不通過 +0.0001） |

### 事前根拠の弱め要因

- **同期≠同ライン**: 同一養成所同期の同ライン率 13.3% vs 異期 22.9%（師匠は先輩=異期のため）
  → 師弟関係も同様に「ライン組む動機にならない」可能性が高い
- **年齢差・地域分散**: 師匠は引退後もページに残るため、同一レース出走率は低い
- **疎性**: mentor_in_race が実際に 1 になるケースは稀（ペア共走率が低い）

## 新特徴量

| 特徴量 | 内容 |
|---|---|
| `mentor_in_race` | この選手の師匠が同一レースに出走しているか (0/1) |
| `is_mentor_of_someone` | この選手が同一レースの誰かの師匠であるか (0/1) |

## 実験結果

> **実行後に記入**（スクレイプ完了後に `python3 scripts/exp_mentor_feature_wt.py`）

### Phase1: AUC

| 期間 | Base | +mentor_in_race | +is_mentor | +両方 |
|---|---|---|---|---|
| VAL | - | - | - | - |
| HOLD | - | - | - | - |
| **VAL+HOLD** | - | - | - | - |

Phase1: [未実行]

### データ特性

- `mentor_in_race`: nonzero [未実行]
- `is_mentor_of_someone`: nonzero [未実行]

## 結論

| 項目 | 判定 |
|---|---|
| mentor_in_race | [未実行] |
| is_mentor_of_someone | [未実行] |

## ハーネス

```bash
# 1. データ収集（約22分・再開可能）
python3 scripts/scrape_mentors_wt.py
python3 scripts/scrape_mentors_wt.py --stats

# 2. 実験
python3 scripts/exp_mentor_feature_wt.py
```
