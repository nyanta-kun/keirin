# G44: WINTICKET 条件別成績スクレイパー + AUC実験

## 目的

WINTICKET の PRELOADED_STATE JSON に含まれる「条件別成績」（天候別・バンク周長別・
時間帯別・位置別）を選手ごとに取得し、Phase1 AUC ゲートを突破するか検証する。

## 背景（G42 の続き）

G42 調査で発見された25フィールドのうち以下4グループは Coverage が高く評価価値がある：

| グループ | JSONキー群 | Coverage |
|---|---|---|
| 天候別成績 | `weatherSunny/Cloudy/Rainy` | ~100% |
| バンク周長別 | `trackDistance333/400/500` | ~98% |
| 時間帯別 | `hourTypeNormal/Morning/Night/Midnight` | 96-100% |
| 位置別 | `linePositionFirst/Second/Third/lineSingleHorseman/lineCompete` | 32-81% |

### 既存アプローチとの違い

G43（身体測定）と同じ構造：
- 選手ごとの直近レース1件をフェッチして条件別成績を取得
- `data/player_ex_stats.csv` に保存後、`wt_entries` に `player_id` でジョイン
- フルリフェッチ不要（選手ごとの現在の累積成績を使用）

### キーとなる実験設計

**マッチ特徴量**（最も期待値高）：
- `track_match_top3_pct`: 当該レースのバンク周長（venue_info.bank_length）での成績
- `hour_match_top3_pct`: 当該レースの時間帯（start_at から推定）での成績
- これらは「今日の条件に対する選手の適性」を表す → rolling stats と独立の可能性

### 期間定義

| 期間 | 範囲 |
|---|---|
| TRAIN | 2023-07-01 〜 2025-06-30 |
| VAL   | 2025-07-01 〜 2026-02-28 |
| HOLD  | 2026-03-01 〜 2026-06-15 |

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD）

## 成果物

1. **`scripts/scrape_winticket_ex_stats.py`**: 条件別成績スクレイパー（選手直近レース）
2. **`scripts/exp_ex_condition_wt.py`**: Phase1 AUC 実験ハーネス
3. **`docs/analysis/44-ex-condition-stats.md`**: 調査・実験レポート

## 実装手順

### Step 1: scrape_winticket_ex_stats.py

**スクレイプ戦略**:
1. DB から選手ごとの直近レース（cup_id/day_index/race_no/venue_id）を取得
2. レースごとにグループ化（1ページで複数選手のデータを同時取得）
3. 各レースページを `WinticketScraper._get()` でフェッチ
4. `_extract_state()` → `_get_query("FETCH_KEIRIN_RACE")` → `records_raw` 取得
5. `records_raw[player_id]` から条件別成績を抽出
6. `data/player_ex_stats.csv` に保存（再開可能）

**取得フィールド（`firstPercentage` + top3率を計算）**:
```python
CONDITION_FIELDS = {
    "weather_sunny":     "weatherSunny",
    "weather_cloudy":    "weatherCloudy",
    "weather_rainy":     "weatherRainy",
    "track_333":         "trackDistance333",
    "track_400":         "trackDistance400",
    "track_500":         "trackDistance500",
    "hour_normal":       "hourTypeNormal",
    "hour_morning":      "hourTypeMorning",
    "hour_night":        "hourTypeNight",
    "hour_midnight":     "hourTypeMidnight",
    "pos_first":         "linePositionFirst",
    "pos_second":        "linePositionSecond",
    "pos_third":         "linePositionThird",
    "pos_single":        "lineSingleHorseman",
    "pos_compete":       "lineCompete",
}
# 各フィールドから top3_pct = (first+second+third) / max(1,total) * 100
```

**レート制限**: 1.5 req/sec（WinticketScraper デフォルト）  
**推定所要時間**: 最大786レース × 1.5s ≈ 20分（実際はキャッシュで短縮）

### Step 2: exp_ex_condition_wt.py

1. `data/player_ex_stats.csv` ロード（Coverage確認）
2. `build_features_wt()` でベース特徴量取得
3. 以下の特徴量を追加:
   - **マッチ特徴量**: `track_match_top3_pct`（当日バンク）、`hour_match_top3_pct`（当日時間帯）
   - **天候差分**: `rain_vs_sunny_diff = weather_rainy_top3_pct - weather_sunny_top3_pct`
   - **位置特徴**: `pos_first_top3_pct`（先頭位置での成績）
   - **全条件追加**: `+all_condition`
4. Phase1 AUC 比較: Base / +track_match / +hour_match / +weather_diff / +pos_first / +all

### Step 3: docs/analysis/44-ex-condition-stats.md

調査設計・スクレイプ結果・Coverage・AUC 結果・結論を記載。

## 受け入れ基準

- `python3 scripts/scrape_winticket_ex_stats.py --limit 10` が完了し10選手分のデータが出力されること
- `python3 scripts/exp_ex_condition_wt.py` がエラーなく完了すること
- `docs/analysis/44-ex-condition-stats.md` に AUC 結果・結論が記載されること

## 触ってよいファイル

- `scripts/scrape_winticket_ex_stats.py`（新規作成）
- `scripts/exp_ex_condition_wt.py`（新規作成）
- `docs/analysis/44-ex-condition-stats.md`（新規作成）
- `data/player_ex_stats.csv`（新規作成）

## 禁止事項

- `src/` 配下の変更
- `data/keirin_wt.db` への書き込み
- git commit
- crontab 変更
