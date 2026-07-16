# doc42: WINTICKET EX フィールド拡張調査

**日付**: 2026-06-15  
**ステータス**: 調査完了・評価継続中  
**スクリプト**: `scripts/inspect_winticket_ex_fields.py`, `scripts/exp_ex_extend_wt.py`

---

## 目的

WINTICKET の PRELOADED_STATE JSON に存在するが未抽出の EX フィールドを特定する。  
既取得5項目 (exSpurt, exThrust, exLeftBehind, exSplitLine, exSnatch) 以外の残りフィールドを調査した。

---

## 調査手法

`inspect_winticket_ex_fields.py` を実行し、最新レースの `records_raw` を直接解析した。

```
venue_id=47 (松阪), race_date=2026-06-15, race_no=1
records_raw: 7 選手分 / 63 キー
```

---

## 発見結果

### A. ex* グループ（競技戦術系）

| JSON キー | 構造 | 意味 | 充足率 | 既取得 |
|-----------|------|------|--------|--------|
| exSpurt | `{total, succeeded, percentage}` | まくりの成功率 | ~100% | 取得済 |
| exThrust | 同上 | 差しの成功率 | ~100% | 取得済 |
| exLeftBehind | 同上 | 残りの成功率 | ~100% | 取得済 |
| exSplitLine | 同上 | 切りの成功率 | ~100% | 取得済 |
| exSnatch | 同上 | 出し抜けの成功率 | ~100% | 取得済 |
| **exCompete** | 同上 | **競りの勝率** | **7.8%** | **未取得** |

`exCompete` が唯一の未取得 ex* フィールドであった。  
充足率が 7.8% と低いのは、競りが稀なケースであるためと考えられる。

### B. 成績系フィールド（位置・天候・バンク・時間帯別）

各フィールドは `{first, second, third, others, total, firstPercentage, ...}` の構造を持つ。

#### 時間帯別成績

| JSON キー | 意味 | 充足率(サンプル20R) |
|-----------|------|------|
| hourTypeNormal | 通常時間帯 | 96.3% |
| hourTypeMorning | モーニング | 98.5% |
| hourTypeNight | ナイター | 98.5% |
| hourTypeMidnight | ミッドナイト | 100.0% |
| hourTypeSummertime | サマータイム | 0%（未使用） |

#### 天候別成績

| JSON キー | 意味 | 充足率 |
|-----------|------|--------|
| weatherSunny | 晴れ | 100.0% |
| weatherCloudy | 曇り | 100.0% |
| weatherRainy | 雨 | 99.3% |
| weatherSnowy | 雪 | ほぼ0（稀） |

#### バンク周長別成績

| JSON キー | 意味 | 充足率 |
|-----------|------|--------|
| trackDistance333 | 333m バンク | 100.0% |
| trackDistance400 | 400m バンク | 100.0% |
| trackDistance500 | 500m バンク | 94.0% |

#### 位置別成績（ライン）

| JSON キー | 意味 | 充足率 |
|-----------|------|--------|
| linePositionFirst | ライン先頭 | 68.7% |
| linePositionSecond | ライン2番手 | 76.1% |
| linePositionThird | ライン3番手 | 61.9% |
| lineSingleHorseman | 1人旅（ライン外） | 81.3% |
| lineCompete | 競り込み成績 | 32.1% |

#### レース種別成績

| JSON キー | 意味 | 備考 |
|-----------|------|------|
| raceTypeQualifyingRound | 予選 | 充足率未計測 |
| raceTypeSemifinal | 準決勝 | 充足率未計測 |
| raceTypeFinal | 決勝 | 充足率未計測 |
| raceTypeLoserRound | 敗者戦 | 充足率未計測 |
| raceTypeSpecial | 特別レース | 充足率未計測 |

### C. その他フィールド（評価対象外）

| JSON キー | 構造 | 備考 |
|-----------|------|------|
| gradeRaceSummaries | list | 常に空リスト（充足率 0%） |
| latestCupResults | list | 直近開催成績（生データ・詳細） |
| latestVenueResults | list | 直近同バンク成績（生データ） |
| previousCupResults | list | 前回開催成績（生データ） |
| currentCupResults | list | 今回開催中の成績（生データ） |

---

## フィールドの特性評価

### 天候・バンク・時間帯別成績の位置づけ

これらのフィールドは **当日のレース条件に対応した選手の過去成績** を提供する。  
現在の `FEATURE_COLS_WT` には `weather_code`（当日天候）や `venue_id`（バンクを間接的に示す）は含まれるが、  
**選手個人の天候別/バンク別勝率** は含まれていない。

- `weather*` 充足率 ~100% → 即座に特徴量化可能
- `trackDistance*` 充足率 ~98% → 即座に特徴量化可能  
  ただし `venue_id` + `venue_info.bank_length` で代替可能な情報が含まれる
- `hourType*` 充足率 ~96-100% → 即座に特徴量化可能

### 位置別成績の制約

`linePosition*` は充足率 62-76% であり、ラインデータが存在しない選手（1人旅等）は欠損する。  
`lineSingleHorseman`（1人旅成績）は 81.3% で比較的高い。  
現在の `line_pos`（ライン内の位置）と組み合わせると相互作用特徴量として有望。

---

## AUC 評価（サンプル評価・参考値）

HOLD 最新 20 レース（134 エントリー）のサンプルフェッチによる予備評価:

```
期間          Base AUC    + EX   diff    n
-------------------------------------------------
VAL           0.7721     0.7721  +0.0000  127889
HOLD(全)       0.7764     0.7764  +0.0000   54833
HOLD(smp)     0.8514     0.8514  +0.0000    134
```

**注意**: サンプルが全体の 134/54833 エントリー（0.2%）に留まるため、  
TRAIN 期間に新特徴量データが存在せず、モデルが特徴量を利用できていない。  
この結果は参考値に過ぎない。

### 正確な評価のための条件

1. TRAIN/VAL 期間（2023-07-01〜2026-02-28）の全レースを再フェッチ
2. DB 書き込み可能にした上でカラム追加してフェッチ結果を保存
3. `build_features_wt()` に天候別/バンク別/時間帯別成績特徴量を追加
4. モデル再学習 → AUC 比較

---

## 推奨事項

### Phase1 候補（充足率・新規性の観点から）

| 優先度 | フィールド | 充足率 | 理由 |
|--------|-----------|--------|------|
| 高 | `weatherSunny/Cloudy/Rainy` + current weather | ~100% | 選手個人の条件別勝率を追加 |
| 高 | `trackDistance333/400/500` + 当日バンク | ~98% | 選手のバンク適性を個人レベルで表現 |
| 中 | `hourType*` + 当日時間帯 | ~96-100% | ミッドナイト等の適性 |
| 低 | `linePosition*` | 62-76% | `line_pos` との相関が高い可能性 |
| 低 | `exCompete` | 7.8% | 稀少なケース、欠損処理が困難 |
| 対象外 | `gradeRaceSummaries` | 0% | 常に空 |

### 実装上の注意

- これらのフィールドは `wt_entries` に存在しないため、スクレイパーでの取得と DB カラム追加が必要
- `src/scraper/winticket.py` の `entries.append({...})` にフィールド追加
- `database.py` の `wt_entries` テーブル定義にカラム追加
- 全期間再収集（`collect-wt-range`）が必要

### Phase1 のハードル

**公開情報の壁**: これらのフィールドは WINTICKET 公式サイトに表示される情報であり、  
他の参加者も同様に参照できる。doc36-39 の検証結果（既存 DB 特徴量は全て Phase1 不通過）と  
同じ構造的限界がある可能性が高い。

ただし「選手個人の条件特異的な過去成績」は現在のモデルに含まれていない新情報であり、  
Phase1 通過の可能性はゼロではない。

---

## ファイル

- `scripts/inspect_winticket_ex_fields.py` — JSON キー調査スクリプト（実行済み）
- `scripts/exp_ex_extend_wt.py` — 実験ハーネス（サンプリングモード）
