# データ収集ガイド

> 最終更新: 2026-06-06  
> ※旧 `docs/data-sources.md` の内容を本ファイルに統合済み

---

## データソース一覧

| ソース | URL | 方式 | 用途 |
|--------|-----|------|------|
| **競輪ステーション** | keirin-station.com | requests + BS4 | メイン収集（本番稼働中）|
| **winticket** | winticket.jp | requests / SSR JSON | 並び情報・事前オッズ（実装済み・収集前）|

> 公式APIは存在しない。どちらも認証不要。

---

## 1. keirin-station ルート

### 概要

| 項目 | 内容 |
|------|------|
| URL | `https://keirin-station.com/keirindb/` |
| 認証 | 不要 |
| 方式 | requests + BeautifulSoup4 |
| レート制限 | リクエスト間 1.5秒固定（`KeirinStationScraper`） |

### URL 構造

```
出走表: GET /keirindb/race/member/{venue_code}/{yyyymmdd}/{race_no}/
結果:   GET /keirindb/race/result/{venue_code}/{yyyymmdd}/{race_no}/
オッズ: GET /keirindb/race/odds/{venue_code}/{yyyymmdd}/{race_no}/{bet_type_no}/
```

### 収集データ

| テーブル | 主要カラム |
|--------|-----------|
| `races` | race_key, venue_code, race_date, race_no, grade, distance, start_time |
| `race_entries` | frame_no, player_id, racing_score, gear_ratio, recent_win_rate_3m, recent_top3_rate_3m, line_position, quinella_rate, period, player_class, prefecture |
| `race_results` | finish_position |
| `odds` | bet_type（trifecta/trio/quinella/exacta/win/place/wide）, combination, payout |

> keirin-station の `racing_score` = JKA 競走得点。`recent_win_rate_3m` は直近3ヶ月勝率（0.0〜1.0スケール）。

### 並列処理の仕組み

```
collect-date
  └── _scan_all_venues （全会場コードを並列スキャン）
        └── _collect_venues_parallel （最大4会場同時）
              └── _collect_one_venue
                    ├── _get_collected_race_keys（DBスキップ判定）
                    └── _fetch_race_parallel（出走表+結果を同時取得）
```

| 設定 | 値 |
|------|---|
| `MAX_VENUE_WORKERS` | 4 |
| リクエスト間隔 | 1.5秒 |
| 再試行 | 最大3回（1秒・3秒・6秒待機） |

### CLI コマンド

```bash
source .venv/bin/activate

# DB初期化（初回のみ）
python -m src.cli.main init

# 1日分
python -m src.cli.main collect --date 2026-06-05

# 月次
python -m src.cli.main collect-month --year 2026 --month 6

# 範囲（最新から逆順 / 推奨）
python -m src.cli.main collect-reverse --from 2025-01

# rolling 統計再計算（collect後に実行）
python -m src.cli.main compute-stats --force

# 収集状況確認
python -m src.cli.main status
```

### 収集状況（2026-06-08 時点）

| 項目 | 値 |
|------|---|
| 収録開始 | 2022-12-30 |
| 収録終了 | 2026-06-08（最新） |
| 総レース数 | **94,830 レース** |
| race_entries | 670,168件 |
| race_results | 659,024件（カバー率 98.3%） |

---

## 2. winticket ルート

### 概要

| 項目 | 内容 |
|------|------|
| URL | `https://www.winticket.jp/keirin/` |
| 認証 | 不要（SSRページにデータ埋め込み） |
| 方式 | requests / `window.__PRELOADED_STATE__` JSON 抽出 |
| レート制限 | リクエスト間 2.0秒（`WinticketScraper`） |
| 対応会場数 | 43会場（winticket 掲載分のみ） |

### URL 構造

```
出走表: GET /keirin/{slug}/racecard/{cupId}/{day_index}/{race_no}
オッズ: GET /keirin/{slug}/odds/{cupId}/{day_index}/{race_no}

cupId = YYYYMMDD（イベント開始日）+ venue_id（2桁 JKA コード）
例: 2026060421 = 2026-06-04開始・弥彦（21）
```

### PRELOADED_STATE の取得方法

winticket は React SSR で全データを `window.__PRELOADED_STATE__` に埋め込む。
TanStack Query (React Query) のキャッシュ形式で格納。

```python
marker = "window.__PRELOADED_STATE__ = "
# → JSON をブレース深度で抽出 → tanStackQuery.queries[] から queryKey で検索
# FETCH_KEIRIN_RACE      → 出走表・ライン・結果
# FETCH_KEIRIN_RACE_ODDS → 全オッズデータ
# FETCH_KEIRIN_CUP_RACES → 開催日程（cupId/day_index 特定用）
```

### 収集データ（keirin-station にはないもの）

| テーブル | 主要カラム（winticket 固有） |
|--------|----------------------------|
| `wt_entries` | race_point, style, prediction_mark（AI印）|
| `wt_entries` | s_count, h_count, b_count（セクター回数）|
| `wt_entries` | ex_spurt_pct, ex_thrust_pct 等（上がり戦術率）|
| `wt_entries` | line_group, line_size, line_pos, is_line_leader, n_lines（並び情報）|
| `wt_odds` | trifecta / trio / exacta / quinella / quinellaPlace の事前オッズ |

> `race_point` = keirin-station の `racing_score` 相当。`first_rate` = 勝率（%表記）。

### cupId 自動探索ロジック

イベントは複数日にわたるため、target_date に対して当日〜3日前の開始日を順に試す。
`FETCH_KEIRIN_CUP_RACES` 内の schedules[].date と照合して一致すれば確定。

### CLI コマンド

```bash
source .venv/bin/activate

# 動作確認
python -m src.cli.main collect-wt --date 2026-06-05 --dry-run

# 1日分（レース + オッズ同時取得）
python -m src.cli.main collect-wt --date 2026-06-05

# 範囲（最新から逆順）
python -m src.cli.main collect-wt-range --from 2025-06

# 収集状況確認
python -m src.cli.main status-wt
```

---

## 注意事項

### 再実行の安全性

- `INSERT OR REPLACE` / `INSERT OR IGNORE` を使用しており **重複実行しても安全**
- 収集済み race_key は両ルートともスキップ（中断後の再開が可能）

### アンチスクレイピング

- keirin-station: `MAX_VENUE_WORKERS=4` を増やすと IP バンのリスクあり
- winticket: 単一ドメインのため `MAX_VENUE_WORKERS=2` に抑制（2.0秒間隔）

### winticket 未対応会場

会津(14)・八戸(15)・一宮(41)・大津(52)・観音寺(72)・門司(82)等は winticket 非掲載のため対象外。
keirin-station には全会場のデータが存在する。
