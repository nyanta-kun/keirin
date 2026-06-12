# システムアーキテクチャ

> 最終更新: 2026-06-13

---

## 概要

競輪AI予想システム「穴車AI」。6車立て以下レースを3段階ランク（SS/S/A）で予想し、
Discord 通知と X(Twitter) 配信を自動化する CLI ベースのシステム。

**2ルート構成（2026-06-08 winticketへ完全移行）:**
- **winticket ルート（★本番稼働中）** — winticket.jp 経由。ライン情報・全組合せ事前オッズを取得。lgbm_wt（39特徴）。
- **keirin-station ルート（収集停止・ロールバック保持）** — keirin-station.com 経由。lgbm_v6（24特徴）。2026-06-08で収集凍結。

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
│   │   ├── feature_engineer.py        # FEATURE_COLS（24特徴量・ks/ロールバック）・build_features()
│   │   ├── feature_wt.py              # FEATURE_COLS_WT（39特徴量・rolling込/DNS処理済）・build_features_wt()
│   │   └── rolling_stats.py           # compute-stats（6ヶ月勝率・場別勝率・前走日数）
│   ├── strategy_wt.py                  # 波乱/非本命ゲート（top3_sum・upset_tier・passes_upset_gate）
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
│   ├── daily_picks_wt.sh              # ★本番日次（cron 8:00）
│   ├── evening_picks_wt.sh            # ★本番夕方（cron 16:00）夜レース用2段階生成
│   ├── intraday_results_wt.sh         # ★本番日中（cron 0,10-23時）当日結果逐次収集・通知なし
│   ├── weekly_retrain_wt.sh           # ★本番週次（cron 日23:30）
│   ├── notify_picks.py                # wave-picks 通知 + PDF生成 → Discord
│   ├── notify_results_wt.py           # wt前日結果採点 + picks_history(route='wt') → Discord
│   ├── snapshot_morning_odds_wt.py    # 朝オッズ退避(wt_odds_snapshot) / --report ドリフト計測
│   ├── snapshot_intraday_odds_wt.py   # 日中オッズスナップショット（money-flow素材・G03）
│   ├── live_report_wt.py              # live実測レポート（ランク別/タグ別成績・ドリフト分布・標本数・G02）
│   ├── collect_weather.py             # 気象データバックフィル（Open-Meteo API・G05）
│   ├── exp_moneyflow_wt.py            # money-flow 検証ハーネス（G04）
│   ├── exp_wind_wt.py                 # 風×バンク特徴リーク無し検証（G06）
│   ├── exp_highpay_fusion_wt.py       # 高配当検知×新シグナル合成（ゲート判定・G07）
│   └── analyze_*/backtest_*_wt.py     # 各種検証スクリプト
├── data/
│   ├── keirin.db                      # SQLite DB（wt 96,455R + ks凍結 / WALモード）
│   ├── models/                        # lgbm_wt.pkl（=v1・本番）/ lgbm.pkl（=v6・ロールバック）等
│   └── picks/                         # wave_picks_wt_YYYY-MM-DD.txt / _detail.json / _detail.pdf
├── config/                            # 設定ファイル（.env: DISCORD_WEBHOOK_URL）
├── docs/                              # ドキュメント
├── notebooks/                         # Jupyter（探索・分析用）
├── tests/
├── requirements.txt
└── CLAUDE.md                          # 開発ガイド（このリポジトリ固有ルール）
```

---

## CLI コマンド一覧

### winticket ルート（★本番）

| コマンド | 説明 |
|---------|------|
| `status-wt` | 収集状況確認 |
| `collect-wt [--date]` | 1日分収集（レース+オッズ同時） |
| `collect-wt-range --from [--to]` | 年月範囲を逆順収集 |
| `train-wt [--from] [--test-from] [--save-as]` | winticket 用LightGBM学習（39特徴） |
| `backtest-wt [--from] [--to] [--model] [--max-riders] [--min-gap12] [--tiered] [--value]` | 買い目バックテスト（wt_odds 実オッズ使用） |
| `wave-picks-wt [--date] [--min-trio-odds] [--gami-skip-odds] [--b-rank-odds] [--upset-gate]` | SS/S/A 予想生成＋ガミ3段階／波乱ゲート |

**wave-picks-wt の主要フラグ（2026-06-08 追加）:**
- `--gami-skip-odds 3.0`：3点中1点でも朝オッズ<3倍ならレース見送り
- `--b-rank-odds 5.0`：最安目が3〜5倍未満ならBランク（購入は各自判断・別枠）
- `--upset-gate Q1_loose|Q2|Q3`：top3_sum波乱ゲート（opt-in。省略時は全pickに upset_tier タグ付けのみ）
- `--stake-tilt`：波乱スコア(top3_sum)で賭け金傾斜（opt-in・既定off）
- `--ss-trifecta-box`：SS層の3連単を pred1,pred2 1-2着BOX(6点)に拡張（opt-in・既定off=3点で本番不変。検証=`docs/analysis/10-le6-fav-position.md`）

補助スクリプト:
- `scripts/snapshot_morning_odds_wt.py [date]`（朝オッズ退避）/ `--report`（朝→最終ドリフト計測）
- `scripts/snapshot_intraday_odds_wt.py [--date]`（日中オッズスナップショット・money-flow素材）
- `scripts/live_report_wt.py [--from] [--to] [--format md]`（live実測レポート・ランク別/タグ別成績・ドリフト分布・必要標本数推定）
- `scripts/collect_weather.py [--from] [--to]`（気象データバックフィル・全43会場・Open-Meteo Historical API）
- `scripts/exp_moneyflow_wt.py [--from] [--to] [--report]`（money-flow検証ハーネス・ドリフト記述統計・スマートマネー仮説）
- `scripts/exp_wind_wt.py [--from] [--to]`（風×バンク特徴のリーク無し LGBM 検証）
- `scripts/exp_highpay_fusion_wt.py [--report]`（高配当×新シグナル合成・G06/G04ゲート判定）

### keirin-station ルート（収集停止・ロールバック保持）

`init` / `status` / `collect[-month/-range/-reverse]` / `compute-stats` / `train` / `backtest` / `analyze` / `weekly` / `day-sim` / `venue` / `predict` / `wave-picks` / `upset-train` / `upset-backtest`（2026-06-08 以降 日常運用では未使用）。

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
| `wt_entries` | 出走情報（34カラム: race_point, 脚質, lineup情報, 戦術率, finish_order 等） |
| `wt_odds` | 事前オッズ（bet_type: trifecta/trio/exacta/quinella 等, odds_value・最終値で上書き） |
| `wt_odds_snapshot` | オッズスナップショット（snapshot_type='morning'/'h06'/'h10'等・初回値保持。朝→最終ドリフト計測・money-flow用） |
| `wt_weather` | 気象データ（venue_id×dt_hour PK・wind_speed/wind_gust/temperature 等。Open-Meteo API 経由バックフィル済） |

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

## 毎朝の自動実行フロー（本番稼働中）

**2026-06-08 以降: winticketルートへ完全移行（ks収集停止）。** 本番日次は `scripts/daily_picks_wt.sh`（cron 8:00）:
```
AM 8:00 （daily_picks_wt.sh）
  ① collect-wt --date $(yesterday)               # 前日結果 再収集（finish_order>=1のみスキップ）
  ② notify_results_wt.py $(yesterday)            # 前日成績採点 → Discord / picks_history(route='wt')
  ③ collect-wt --date $(today)                   # 当日出走表+オッズ収集
  ④ snapshot_morning_odds_wt.py $(today)         # 朝オッズを wt_odds_snapshot に退避（ドリフト計測用）
  ⑤ wave-picks-wt --date $(today) \
       --gami-skip-odds 3.0 --b-rank-odds 5.0    # 予想生成（lgbm_wt 40特徴・SS/S/A＋ガミ3段階）
  ⑥ notify_picks.py $(today) wave_picks_wt       # 予想 + PDF → Discord（Bは各自判断・成績/ツイート対象外）
日中（0,10-23時, intraday_results_wt.sh）: collect-wt --date $(today) で当日結果を逐次収集（未終了のみ・通知なし・最終R23:30発走を0:00でカバー）。
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
| 2026-06-04 | ks v6: 2023年〜追加収集 / AUC 0.7575 / ホールドアウト9ヶ月検証 |
| 2026-06-05 | S ランクに ratio<1.6 上限追加（低配当レース除外） |
| 2026-06-07 | winticket 全期間収集（96k）。ローリング特徴移植・ks比較検証 |
| **2026-06-08** | **DNS(欠車)バグ修正 → winticket本番移行**（lgbm_wt_v1・39特徴・CV AUC 0.7720）。3タスク分析（`docs/analysis/`）→ 波乱ゲート(`strategy_wt.py`)・ガミ回避3段階・朝オッズ前向き計測を実装 |
| **2026-06-09** | n_senko 特徴追加（FEATURE_COLS_WT 39→40）。SS三連単BOX(6点)・ワイド1点推奨(opt-in)実装。7+クローズ（公開オッズ内を3経路で確定）。fav_mismatch タグ記録開始。夕方2段階生成（`evening_picks_wt.sh`）実装。 |
| **2026-06-10** | 欠車無効化（`notify_results_wt._void_by_dns`）・結果バックフィル実装。会場取得漏れバグ修正（ks references → wt_races）。`linePrediction=null` クラッシュ修正。 |
| **2026-06-12** | バックテスト3バイアス発見（`docs/analysis/18`）。夕方cron 16:00登録完了。各種検証スクリプト追加（gap13打ち切り・B閾値緩和全滅・条件先行新方式なし・高配当10点リーク無し不通過・コメント特徴無情報・Web予想監査不通過）。 |
| **2026-06-13** | G01〜G07完了（`backtest_wt.py`リーク無し化・live実測レポート・日中オッズスナップショット・money-flow検証ハーネス・気象データ収集・風特徴検証・高配当融合ゲート）。 |
