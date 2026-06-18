# G05: 気象（風）データ収集・バックフィル

## 目的
Web予想ロジック監査（doc20）・選手コメント検証の結論で、市場が織り込みにくい
**唯一の外部残候補=風**。過去気象は遡及取得できるため、money-flowと違い
**今すぐ全期間のリーク無し検証が可能**。その素材を収集する。

## データソース
- **Open-Meteo Historical Weather API**（無料・APIキー不要）:
  `https://archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..&start_date=..&end_date=..&hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,precipitation`
- 対象期間: 2022-12-01 〜 実行日（wt データ有効期間に一致）
- レート配慮: 会場ごとに長期間を一括取得（リクエスト数を最小化）・失敗時リトライ

## 成果物
1. `src/scraper/weather.py`（新規）:
   - **会場座標テーブル**: 全43会場（`src/scraper/winticket.py` の VENUE_SLUGS と同じ
     venue_id をキーに、競輪場の緯度経度）。競輪場所在地は既知の公開情報から記載し、
     精度は市レベル（±数km）で可。`venue_info.is_indoor`（ドーム）も参照できるよう join 前提。
   - Open-Meteo クライアント（hourly 取得・JST変換に注意。API は timezone パラメータ対応）
2. DB テーブル（`CREATE TABLE IF NOT EXISTS`・新規）:
   ```sql
   wt_weather(venue_id TEXT, dt_hour TEXT,  -- 'YYYY-MM-DD HH:00' JST
              wind_speed REAL, wind_dir REAL, wind_gust REAL,
              temp REAL, precip REAL,
              PRIMARY KEY(venue_id, dt_hour))
   ```
3. `scripts/collect_weather.py`（新規）: `--from/--to`・`--venue` 指定でバックフィル。
   `INSERT OR REPLACE` で再実行安全。日次差分実行も同スクリプトで可能に。
4. 結合ヘルパー: `weather_for_race(race_key)` — `wt_races.start_at` に最も近い時刻の
   会場気象を返す（G06 から import される）。
5. **バックフィル実行**: 全会場×2022-12〜実行日を実際に取得し、
   `wt_races` の (venue_id, 日付) に対するカバレッジ ≥95% を確認・報告。

## 受け入れ基準
- カバレッジレポート（会場別・期間別の欠損）を出力
- 座標テーブルのユニットテスト（43会場全件・緯度経度が日本国内範囲）
- 既存テスト50pass維持

## 触ってよいファイル
- `src/scraper/weather.py`（新規）
- `scripts/collect_weather.py`（新規）
- `tests/`（新規テストファイル）

## 禁止事項
- 既存テーブルのスキーマ変更・cron変更・git commit
- winticket.py 等既存スクレイパーの変更（VENUE_SLUGS は読むだけ）
