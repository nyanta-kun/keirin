# システムアーキテクチャ

> 最終更新: 2026-06-06

---

## 概要

競輪AI予想システム「穴車AI」。6車立て以下レースを3段階ランク（SS/S/A）で予想し、
Discord 通知と X(Twitter) 配信を自動化する CLI ベースのシステム。

**2ルート構成:**
- **keirin-station ルート** — 本番稼働中。競輪ステーション（keirin-station.com）経由でデータ収集・学習
- **winticket ルート** — 実装済み・収集前。winticket.jp 経由。ライン情報・事前オッズが追加取得可能

---

## ディレクトリ構成（実際）

```
keirin/
├── src/
│   ├── database.py                    # DBスキーマ・venue_infoマスタ・init_db/migrate_db
│   ├── scraper/
│   │   ├── keirin_station.py          # 競輪ステーション スクレイパー（requests+BS4）
│   │   ├── pipeline.py                # ks収集パイプライン（並列4会場・出走表+結果並列取得）
│   │   ├── winticket.py               # winticket スクレイパー（PRELOADED_STATE JSON解析）
│   │   └── pipeline_wt.py             # wt収集パイプライン（並列2会場・オッズ同時取得）
│   ├── preprocessing/
│   │   ├── feature_engineer.py        # FEATURE_COLS（24特徴量）・build_features()
│   │   ├── feature_wt.py              # FEATURE_COLS_WT（30特徴量）・build_features_wt()
│   │   └── rolling_stats.py           # compute-stats（6ヶ月勝率・場別勝率・前走日数）
│   ├── models/
│   │   └── trainer.py                 # train_lgbm/train_baseline/save_model/load_model
│   ├── prediction/
│   │   └── predictor.py               # predict_race・format_prediction
│   ├── evaluation/
│   │   ├── backtest.py                # run_backtest/run_threshold_analysis/run_day_simulation
│   │   └── upset_model.py             # 波乱レース予測モデル
│   ├── notify/
│   │   └── discord.py                 # Discord Webhook 通知（send/send_file）
│   └── cli/
│       └── main.py                    # CLIエントリーポイント（全コマンド定義）
├── scripts/
│   ├── notify_picks.py                # wave-picks 通知 + PDF生成 → Discord
│   └── notify_results.py              # 前日結果 + 的中履歴 → Discord
├── data/
│   ├── keirin.db                      # SQLite DB（94,830レース / WALモード）
│   ├── models/                        # lgbm.pkl（v6）/ lgbm_wt.pkl（未学習）等
│   └── picks/                         # wave_picks_YYYY-MM-DD.txt / _detail.json / _detail.pdf
├── config/                            # 設定ファイル（.env: DISCORD_WEBHOOK_URL）
├── docs/                              # ドキュメント
├── notebooks/                         # Jupyter（探索・分析用）
├── tests/
├── requirements.txt
└── CLAUDE.md                          # 開発ガイド（このリポジトリ固有ルール）
```

---

## CLI コマンド一覧

### keirin-station ルート

| コマンド | 説明 |
|---------|------|
| `init` | DB初期化（初回のみ） |
| `status` | 収集状況確認 |
| `collect --date` | 1日分収集 |
| `collect-month --year --month` | 1ヶ月分収集 |
| `collect-range --from [--to]` | 年月範囲を順方向収集 |
| `collect-reverse --from [--to]` | 年月範囲を逆順収集（最新優先） |
| `compute-stats [--force]` | rolling統計計算（6ヶ月勝率・場別勝率等） |
| `train [--model] [--from] [--test-from] [--save-as]` | LightGBM学習 |
| `backtest` | 戦略別バックテスト |
| `analyze` | 閾値フィルター分析 |
| `weekly [--days]` | 直近N日の日別・場別集計 |
| `day-sim --date` | 1日分の購入シミュレーション |
| `venue` | 会場別的中率・回収率 |
| `predict --race-key` | 1レース予想 |
| `wave-picks [--date]` | SS/S/A 3段階ランク予想生成 |
| `upset-train` | 波乱レース予測モデル学習 |
| `upset-backtest` | 波乱モデル × 戦略バックテスト |

### winticket ルート

| コマンド | 説明 |
|---------|------|
| `status-wt` | 収集状況確認 |
| `collect-wt [--date]` | 1日分収集（レース+オッズ） |
| `collect-wt-range --from [--to]` | 年月範囲を逆順収集 |
| `train-wt [--from] [--test-from] [--save-as]` | winticket 用LightGBM学習 |
| `backtest-wt [--from] [--to] [--model] [--max-riders] [--min-gap12]` | winticket 用 買い目バックテスト（wt_odds の実オッズ使用） |
| `wave-picks-wt [--date] [--min-trio-odds]` | オッズフィルター付き予想生成 |

---

## データフロー

### keirin-station ルート

```
keirin-station.com
  └── scraper/keirin_station.py  (requests + BS4)
        └── scraper/pipeline.py  (4会場並列 / 出走表+結果並列)
              └── database.py    (races / race_entries / race_results / odds)
                    └── preprocessing/feature_engineer.py  (FEATURE_COLS 24特徴量)
                          └── models/trainer.py  (LightGBM 時系列CV)
                                └── data/models/lgbm.pkl
                                      └── cli wave-picks
                                            └── scripts/notify_picks.py → Discord
```

### winticket ルート

```
winticket.jp (PRELOADED_STATE JSON / SSR)
  └── scraper/winticket.py  (requests / tanStackQuery解析)
        └── scraper/pipeline_wt.py  (2会場並列 / レース+オッズ同時取得)
              └── database.py    (wt_races / wt_entries / wt_odds)
                    └── preprocessing/feature_wt.py  (FEATURE_COLS_WT 39特徴量・rolling込/DNS処理済)
                          └── models/trainer.py  (同一trainer / feature_cols引数)
                                └── data/models/lgbm_wt.pkl
                                      └── cli wave-picks-wt (オッズフィルター付き)
```

---

## DBスキーマ（概要）

### keirin-station テーブル

| テーブル | 内容 |
|--------|------|
| `races` | レース情報（race_key, venue_code, race_date, grade, distance, start_time） |
| `race_entries` | 出走情報（24カラム: racing_score, gear_ratio, win_rate, 脚質 等） |
| `race_results` | 着順結果（frame_no, finish_position） |
| `odds` | 払戻金（bet_type: trifecta/trio/quinella 等, payout） |
| `venues` | 会場マスタ |
| `players` | 選手マスタ |
| `venue_info` | 場マスタ（bank_length, is_indoor, prefecture） |
| `picks_history` | 予想履歴（hit/payout 集計用） |

### winticket テーブル

| テーブル | 内容 |
|--------|------|
| `wt_races` | レース情報（cup_id, day_index, grade, start_at 等） |
| `wt_entries` | 出走情報（34カラム: race_point, 脚質, lineup情報, 戦術率 等） |
| `wt_odds` | 事前オッズ（bet_type: trifecta/trio/exacta/quinella 等, odds_value） |

---

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| スクレイピング | Python 3.10, requests, BeautifulSoup4 |
| データ管理 | SQLite 3（WALモード）, pandas |
| ML | LightGBM, scikit-learn（baseline用） |
| CLI | Click |
| 通知 | Discord Webhook（urllib.request で実装 / requests不使用） |
| PDF生成 | matplotlib（PNG変換）+ Pillow（PDF結合） |
| 環境管理 | venv（`.venv/`）|

---

## 毎朝の自動実行フロー（想定）

**2026-06-08 以降: winticketルートへ完全移行（ks収集停止）。** 本番日次は `scripts/daily_picks_wt.sh`（cron 7:00）:
```
AM 7:00 （daily_picks_wt.sh）
  ① collect-wt --date $(yesterday)            # 前日結果 再収集（finish_order>=1のみスキップ）
  ② notify_results_wt.py $(yesterday)         # 前日成績採点 → Discord / picks_history(route='wt')
  ③ collect-wt --date $(today)                # 当日出走表+オッズ収集
  ④ wave-picks-wt --date $(today)             # 予想生成（lgbm_wt 39特徴・SS/S/A）
  ⑤ notify_picks.py $(today) wave_picks_wt    # 予想 + PDF → Discord
週次（日 23:30, weekly_retrain_wt.sh）: train-wt 再学習。
```
（旧ksフロー daily_picks.sh / notify_results.py / wave-picks は廃止。lgbm_v6等は保持＝ロールバック用）

---

## 開発経緯（簡略）

| 時期 | 内容 |
|------|------|
| 2026-02 | v1.0 本番稼働（LightGBM 13特徴量 / AUC 0.7444） |
| 2026-05 | v2〜v4: 特徴量24個・時系列CVへ修正・データ拡張 |
| 2026-06-02 | wave-picks SS/S/A 3段階ランク戦略策定 |
| 2026-06-04 | v6: 2023年〜データ追加収集（94,722R）/ AUC 0.7575 / ホールドアウト9ヶ月検証 |
| 2026-06-05 | S ランクに ratio<1.6 上限追加（低配当レース除外） |
| 2026-06-06 | winticket ルート実装完了（未収集）|
