# データ収集ガイド

## 概要

競輪ステーション（https://keirin-station.com）をデータソースとして、レース情報・出走表・結果・払戻金をスクレイピングし SQLite DBに格納する。

---

## データソース

### 競輪ステーション（メイン）

| 項目 | 内容 |
|------|------|
| URL | https://keirin-station.com/keirindb/ |
| 認証 | 不要（ログインなし） |
| スクレイピング方式 | requests + BeautifulSoup4 |
| レート制限 | リクエスト間 2〜5秒のランダムウェイト |

#### URL構造

```
スケジュール検索: POST /keirindb/search/race/
開催情報:         GET  /keirindb/stadium/information/{venue_code}/{yyyymmdd}/
出走表:           GET  /keirindb/race/member/{venue_code}/{yyyymmdd}/{race_no}/
オッズ:           GET  /keirindb/race/odds/{venue_code}/{yyyymmdd}/{race_no}/{bet_type_no}/
結果:             GET  /keirindb/race/result/{venue_code}/{yyyymmdd}/{race_no}/
```

#### 開催場コード（主要）

| コード | 開催場 | コード | 開催場 |
|--------|--------|--------|--------|
| 11 | 函館 | 31 | 岐阜 |
| 14 | 弥彦 | 34 | 富山 |
| 21 | 立川 | 37 | 福井 |
| 22 | 松戸 | 44 | 防府 |
| 23 | 千葉 | 46 | 小倉 |
| 24 | 川崎 | 47 | 久留米 |
| 28 | 静岡 | 50 | 別府 |

---

## 取得データ項目

### レース情報（`races` テーブル）
- `race_key` : 一意識別子（例: `20250401_21_01`）
- `venue_code` / `race_date` / `race_no`
- `grade` : グレード（G1/G2/G3/A級等）
- `distance` : 距離（m）

### 出走情報（`race_entries` テーブル）
- `frame_no` : 枠番（1〜9）
- `player_id` : 選手ID（競輪ステーション管理番号）
- `gear_ratio` : ギア比（例: 3.92）
- `racing_score` : 競走得点
- `recent_win_rate_3m` : 直近3ヶ月勝率
- `recent_top3_rate_3m` : 直近3ヶ月3着内率
- `line_position` : 脚質（先行/差し/追い込み等）

### 結果（`race_results` テーブル）
- `finish_position` : 着順（1〜9）
- `frame_no` / `player_id`

### 払戻金（`odds` テーブル）

| `bet_type` | 賭式 | 組み合わせ例 |
|------------|------|-------------|
| `win` | 単勝 | `1` |
| `place` | 複勝 | `1` |
| `quinella` | 2車複 | `1=3` |
| `exacta` | 2車単 | `1-3` |
| `wide` | ワイド | `1=3` |
| `trifecta_box` | 3連複 | `1=2=3` |
| `trifecta` | 3連単 | `1-3-2` |

> `=` 区切りは複式（順不同）、`-` 区切りは単式（着順あり）

---

## CLI コマンド

### 環境のアクティベート

```bash
source .venv/bin/activate
```

### DB初期化（初回のみ）

```bash
python src/cli/main.py init
```

### 収集コマンド一覧

| コマンド | 用途 | 例 |
|---------|------|----|
| `collect` | 1日分を収集 | `python src/cli/main.py collect --date 2025-11-01` |
| `collect-month` | 1ヶ月分を収集 | `python src/cli/main.py collect-month --year 2025 --month 11` |
| `collect-range` | 年月範囲を一括収集 | `python src/cli/main.py collect-range --from 2025-07` |
| `status` | 収集状況を確認 | `python src/cli/main.py status` |

#### `collect-range` オプション

```bash
# 2025年7月〜今月まで（--to省略で今月まで自動）
python src/cli/main.py collect-range --from 2025-07

# 期間指定
python src/cli/main.py collect-range --from 2025-01 --to 2025-06

# 動作確認のみ（DBに保存しない）
python src/cli/main.py collect-range --from 2025-07 --dry-run
```

### バックグラウンド実行

```bash
# バックグラウンドで実行してログをファイルに保存
nohup python src/cli/main.py collect-range --from 2025-07 > logs/collect.log 2>&1 &
echo "PID: $!"
```

---

## 並列処理の仕組み

```
collect-range
  └── collect_month (月ごとに実行)
       └── _collect_venues_parallel  ← 最大3会場を同時並列
            └── _collect_one_venue（スレッドごとに独立）
                 ├── _get_collected_race_keys  ← DB照合でスキップ判定
                 └── _fetch_race_parallel      ← 出走表+結果を同時取得
                      ├── scrape_race_detail（スレッドA）
                      └── scrape_race_result（スレッドB）
```

| 改善点 | 効果 |
|--------|------|
| 収集済みレースをスキップ | 再実行・追加収集で無駄なリクエストを排除 |
| 出走表+結果の並列取得 | 1レースあたりの待ち時間を約半分に短縮 |
| 最大3会場同時並列 | スループット約3倍 |
| 開催場単位バッチDB書き込み | DB書き込みのオーバーヘッドを削減 |
| SQLite WALモード | マルチスレッド書き込み時のロック競合を最小化 |

---

## 注意事項・制限

### アンチスクレイピング対策
- リクエスト間に **2〜5秒のランダムウェイト** を設ける（変更: `src/scraper/base.py` の `delay_min/max`）
- 接続切断（`RemoteDisconnected`）が発生した場合、**最大3回まで自動リトライ**（1回目3秒、2回目6秒待機）
- `MAX_VENUE_WORKERS = 3`（`src/scraper/pipeline.py`）を増やしすぎるとIPバンのリスクあり

### データ品質
- **出走表のパーサー精度**: 府県・選手名の解析は正規表現ベースのため一部誤パースあり（選手IDは正確に取得済み）
- **枠番と選手IDのマッピング**: 出走表（枠番基準）と結果（選手ID基準）を着順テーブルで紐付け。欠損がある場合は `frame_{N}` で仮IDを付与
- **会場名未定義**: `会場61` 等の表示は venue_code の名称マッピング漏れ。データ収集には影響なし

### 再実行の安全性
- `INSERT OR IGNORE` / `INSERT OR REPLACE` を使用しており、**重複実行しても安全**
- 収集済みの race_key はスキップするため、中断後の再開も可能

---

## 収集状況（2026-05-23 時点）

| 月 | レース数 | 会場数 |
|----|---------|-------|
| 2025-01 | 783 | 22 |
| 2025-02 | 759 | 21 |
| 2025-03 | 780 | 19 |
| 2025-04 | 768 | 23 |
| 2025-05 | 747 | 19 |
| 2025-06 | 723 | 22 |
| 2025-07〜 | 収集中 | - |
| **合計** | **4,800+** | - |

---

## ファイル構成

```
src/scraper/
├── base.py            # 基底クラス（リクエスト・リトライ・ウェイト）
├── keirin_station.py  # 競輪ステーション スクレイパー
└── pipeline.py        # 並列収集パイプライン

src/
└── database.py        # DBスキーマ定義・接続管理

src/cli/
└── main.py            # CLIエントリーポイント

data/
└── keirin.db          # SQLite データベース
```
