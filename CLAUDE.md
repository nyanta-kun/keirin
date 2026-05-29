# CLAUDE.md — 競輪AI予想システム開発ガイド

## ドキュメント更新ルール

以下の変更を行った際は、必ず `docs/prediction-factors.md` を合わせて更新すること。

| 変更内容 | 更新箇所 |
|----------|---------|
| `FEATURE_COLS` に特徴量を追加・削除 | 特徴量一覧テーブル（v1/v2）+ 更新履歴 |
| `race_entries` / `races` / `venue_info` のカラム追加・変更 | DBカラム列・取得済み未使用項目 |
| スクレイパーで新しいフィールドを取得開始 | 対応する特徴量行の「DBカラム/計算元」列 |
| `compute-stats` の計算ロジック変更 | 対応する特徴量の説明 |
| モデル再学習（AUC更新） | 概要のバージョン・AUC値 + 更新履歴 |
| 新コマンド追加 | モデル更新手順セクション |

更新時は「最終更新」日付と「更新履歴」テーブルも必ず記入する。

## キーファイル

```
src/preprocessing/feature_engineer.py  # FEATURE_COLS・build_features()
src/database.py                         # スキーマ・venue_infoマスタ
src/scraper/keirin_station.py           # スクレイピング
src/scraper/pipeline.py                 # DB保存・スキップロジック
src/preprocessing/rolling_stats.py     # compute-stats
src/cli/main.py                         # CLIコマンド
docs/prediction-factors.md             # 予想ファクター仕様書（要メンテ）
CONTINUATION.md                         # セッション引継ぎメモ
```

## 設計方針

- `FEATURE_COLS` はモデル互換性のため変更時は必ず再学習する
- `_get_collected_race_keys` は `race_entries` にデータがあるものだけをスキップ（races テーブルのみでは不十分）
- データ有効期間: 直近2年（2024-06-01〜現在）
- 収集方向: 最新から過去へ（`collect-reverse`）
- `INSERT OR REPLACE` を使うため再収集は安全
