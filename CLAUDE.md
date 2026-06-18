# CLAUDE.md — 競輪AI予想システム開発ガイド

## ドキュメント更新ルール

以下の変更を行った際は、必ず `docs/prediction-factors.md` を合わせて更新すること。

| 変更内容 | 更新箇所 |
|----------|---------|
| `FEATURE_COLS` に特徴量を追加・削除 | 特徴量一覧テーブル + 更新履歴 |
| `FEATURE_COLS_WT` に特徴量を追加・削除 | winticket 特徴量一覧テーブル |
| `race_entries` / `wt_entries` のカラム追加・変更 | 対応する特徴量行 |
| スクレイパーで新しいフィールドを取得開始 | 対応する特徴量行の「DBカラム/計算元」列 |
| `compute-stats` の計算ロジック変更 | 対応する特徴量の説明 |
| モデル再学習（AUC更新） | 概要のバージョン・AUC値 + 更新履歴 |
| 新コマンド追加 | `docs/system-architecture.md` のコマンド一覧 |
| 戦略変更（閾値・ランク条件） | `docs/bet-structure-guide.md` + `docs/prediction-factors.md` |

更新時は「最終更新」日付と「更新履歴」テーブルも必ず記入する。

## キーファイル

### winticket ルート（★本番稼働中・2026-06-08〜）

```
src/scraper/winticket.py                # PRELOADED_STATE JSON スクレイパー
src/scraper/pipeline_wt.py              # wt収集（レース+オッズ同時・結果ありのみスキップ）
src/preprocessing/feature_wt.py        # FEATURE_COLS_WT（39特徴・rolling統合）・build_features_wt() / add_rolling_features_wt()
src/evaluation/backtest_wt.py           # wt用バックテスト（通常/--tiered/--value）
src/models/trainer.py                   # train_lgbm（feature_cols/weight_col引数で両ルート共用）
src/cli/main.py                         # CLIコマンド（collect-wt/train-wt/backtest-wt/wave-picks-wt等）
scripts/daily_picks_wt.sh               # 日次運用（cron 8:00）
scripts/notify_results_wt.py            # wt成績採点・通知・picks_history(route='wt')
```
重要: `finish_order=0`は欠車/失格=着外。top3判定は `between(1,3)`（0を3着内に誤算入するバグを2026-06-08修正、性能激変）。

### keirin-station ルート（収集停止・ロールバック用に保持）

```
src/preprocessing/feature_engineer.py  # FEATURE_COLS（24特徴量）・build_features()
src/scraper/keirin_station.py           # スクレイピング（2026-06-08 収集停止）
src/scraper/pipeline.py / rolling_stats.py
data/models/lgbm.pkl (=lgbm_v6)         # 保持。日次/週次cronはwt版に切替済
```

### ドキュメント

```
CONTINUATION.md                         # セッション引継ぎメモ（最重要）
docs/prediction-factors.md             # 予想ファクター仕様書（要メンテ）
docs/system-architecture.md            # システム構成・CLIコマンド一覧
docs/data-collection.md                # データ収集手順（ks + winticket）
docs/bet-structure-guide.md            # 買い目戦略（SS/S/A）
```

## 設計方針

- `FEATURE_COLS` / `FEATURE_COLS_WT` はモデル互換性のため変更時は必ず再学習する
- `_get_collected_race_keys` は `race_entries` にデータがあるものだけをスキップ（races テーブルのみでは不十分）
- winticket の `_get_collected_keys` は `wt_entries` を参照（同様）
- データ有効期間: winticket 2022-12〜現在（本番）/ keirin-station は2026-06-08で凍結
- 収集方向: 最新から過去へ（`collect-reverse` / `collect-wt-range`）
- `INSERT OR REPLACE` を使うため再収集は安全
- **2026-06-08 winticketルートへ完全移行**（wtがks同等以上を確認）。ks収集停止・cronはwt版。ks資産はロールバック用に保持
- finish_order=0(欠車)は着外。top3は `between(1,3)` で判定（DNS誤算入バグ修正済）
- **バックテストの3バイアスに注意（2026-06-12発見・docs/analysis/18）**: ①ランキングは必ず全エントリーで行う（完走者のみ=欠車生存バイアス×stale oddsで黒字が捏造される・旧 `_apply_pred_prob_wt`系は該当）②≤6車判定は出走表基準（`_filter_by_n_riders`を欠車除去後に適用すると7車立てが混入）③モデルは評価期間外で学習（週次再学習済みlgbm_wtはリーク）。標準実装= `exp_leakfree_rescore_wt.py`。本番忠実ではC0現行戦略含む全レバー~70-90%＝**採否判断はlive実測(picks_history)のみ**
