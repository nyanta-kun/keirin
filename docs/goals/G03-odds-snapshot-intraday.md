# G03: オッズスナップショット多時点化（money-flow 素材収集）

## 目的
7+クローズ・≤6車リーク無し再採点の両方が指す唯一の市場内残候補=
**「朝→直前のオッズ変動（money-flow）」**の素材を増やす。
現状は morning（8時台・`snapshot_morning_odds_wt.py`）と evening（16時台・
`evening_picks_wt.sh` 経由）の2時点のみ。これを日中複数時点に拡張する。

## 現状スキーマ（変更禁止・準拠すること）
```sql
wt_odds_snapshot(race_key, bet_type, combination, odds_value,
                 snapshot_type, snapshot_at,
                 UNIQUE(race_key, bet_type, combination, snapshot_type))
```
- 既存 snapshot_type: 'morning' / 'evening'。蓄積は 2026-06-09〜（morning 5日345R）。

## 成果物
1. `scripts/snapshot_intraday_odds_wt.py`（新規）:
   - 実行時刻から `snapshot_type='h{HH}'`（例 h10, h12, h14, h18, h20）を自動決定
   - 当日 `wt_races` の**未発走レース**（start_at > now）の trio / trifecta / wide オッズを
     winticket から取得し snapshot 保存（`INSERT OR REPLACE`・既存 morning スクリプトの実装を流用）
   - 発走済み・中止レースはスキップ。リクエスト間 sleep でサーバ負荷配慮
   - `--report`: レース毎に「morning → h{XX} → 確定」のドリフト系列を表示
2. 発走時刻相対の参照ヘルパー: 同スクリプト内に「race_key と任意時点Tに対し、
   T-60分/T-30分に最も近い snapshot を返す」関数（G04 の money-flow 分析から import 可能に）
3. cron 提案: `data/cron_proposal_moneyflow_20260613.txt` に追加行
   （例: `0 10,12,14,18,20 * * *` で本スクリプト実行・ログ出力先は既存 cron と同様式）。
   **crontab への書込みは絶対にしない**（リモートからはTCCでハング・ユーザーがTerminal.appから適用）。

## 受け入れ基準
- 実際に1回実行し、当日の未発走レースのスナップショットが保存されることを確認
  （土曜=開催日なので当日データがあるはず。なければ翌営業日想定のドライラン結果を報告）。
- UNIQUE制約により同一時点の再実行が安全（REPLACE）であることをテストで確認。

## 触ってよいファイル
- `scripts/snapshot_intraday_odds_wt.py`（新規）
- `data/cron_proposal_moneyflow_20260613.txt`（新規）
- `tests/`（新規テストファイル）

## 禁止事項
- `daily_picks_wt.sh` / `evening_picks_wt.sh` / crontab の変更・git commit
- wt_odds_snapshot スキーマ変更
