# 予想ファクター仕様書

> **最終更新**: 2026-05-27  
> **モデルバージョン**: v1（13特徴量、lgbm_v1.pkl）→ v1.5（20特徴量）→ v2（24特徴量）→ **v3実用版（24特徴量、lgbm.pkl = lgbm_v3.pkl）**

---

## 概要

本システムは LightGBM を使用した3着内確率予測モデル。  
特徴量は `src/preprocessing/feature_engineer.py` の `FEATURE_COLS` で管理し、  
DBカラムの定義は `src/database.py` で管理する。

---

## 特徴量一覧

### v1（バックアップ / 13特徴量）

モデルファイル: `data/models/lgbm_v1.pkl`  
CV AUC: **0.7444**

| # | 変数名 | カテゴリ | 説明 | DBカラム |
|---|--------|----------|------|----------|
| 1 | `racing_score` | 選手成績 | JKA競走得点（総合評価指標） | `race_entries.racing_score` |
| 2 | `gear_ratio` | 選手成績 | ギヤ倍数（例: 3.92） | `race_entries.gear_ratio` |
| 3 | `recent_win_rate_3m` | 選手成績 | 直近3ヶ月勝率（サイト掲載値） | `race_entries.recent_win_rate_3m` |
| 4 | `recent_top3_rate_3m` | 選手成績 | 直近3ヶ月3着内率 | `race_entries.recent_top3_rate_3m` |
| 5 | `line_pos_enc` | 戦術 | 脚質エンコード（先行=0/捲り=1/差し=2/追い込み=3） | `race_entries.line_position` |
| 6 | `frame_no` | 枠・位置 | 車番（1〜9） | `race_entries.frame_no` |
| 7 | `score_rank` | レース内相対 | 競走得点のレース内順位（1=最高） | 派生 |
| 8 | `score_z` | レース内相対 | 競走得点のレース内偏差値（clip±5） | 派生 |
| 9 | `wr_rank` | レース内相対 | 3ヶ月勝率のレース内順位 | 派生 |
| 10 | `top3r_rank` | レース内相対 | 3ヶ月3着内率のレース内順位 | 派生 |
| 11 | `is_inner` | 枠・位置 | 内枠フラグ（車番1〜3 = 1） | 派生 |
| 12 | `is_outer` | 枠・位置 | 外枠フラグ（車番7以上 = 1） | 派生 |
| 13 | `grade_enc` | レース条件 | グレードエンコード（GP=7/G1=6/G2=5/G3=4/F1=3/F2=2/A=1） | `races.grade` |

---

### v1.5（旧バージョン / 20特徴量）

モデルファイル: `data/models/lgbm_v15_final.pkl`（2026-05-27に lgbm_v2 へ更新）  
CV AUC: **0.7495**  
追加条件: `compute-stats` 実行済み + `venue_info` 登録済み

v1に以下を追加:

| 変数名 | 説明 | 備考 |
|--------|------|------|
| `recent_win_rate_6m` | 直近6ヶ月勝率 | compute-stats |
| `recent_top3_rate_6m` | 直近6ヶ月3着内率 | compute-stats |
| `wr_trend` | 勝率トレンド（3m − 6m） | 派生 |
| `venue_win_rate` | 同場通算勝率 | compute-stats |
| `days_since_last_race` | 前走経過日数 | compute-stats |
| `bank_length_enc` | バンク長 / 100 | venue_info静的データ |
| `is_indoor` | 屋内バンクフラグ | venue_info静的データ |

---

### v2（旧 / 24特徴量）

モデルファイル: `data/models/lgbm_v2.pkl`  
CV AUC: **0.7526** ± 0.0021（**過楽観**: GroupKFold に未来漏洩あり）  
データ期間: 2024-06-01〜2026-04（54,422レース）

---

### v3実用版（現在使用中 / 24特徴量）

モデルファイル: `data/models/lgbm.pkl`（= `lgbm_v3.pkl`）  
CV AUC: **0.7490** ± 0.0034（日付ベース時系列CV・漏洩なし）  
データ期間: 2024-06-01〜2026-05（54,491レース）

v2 と特徴量は同一。**CV手法を GroupKFold → 日付ベース時系列 fold に修正。**

> **修正背景**: GroupKFold は同一レースを同一 fold に入れるが、時系列順を保証しない。結果として全 fold で訓練・検証の日付範囲が重複し CV AUC が +0.0036 過楽観だった。正しい時系列 CV では 0.7490、ホールドアウト（2026-05）実測値 0.7481 と整合。

v1.5 に以下4特徴量を追加（v2から継続）:

| 変数名 | 説明 | DBカラム | カバレッジ |
|--------|------|----------|----------|
| `quinella_rate` | 2連対率（2着内率） | `race_entries.quinella_rate` | 100% |
| `period_norm` | 期別 / 100（小さい値=ベテラン） | `race_entries.period` | 100% |
| `player_class_enc` | 登録クラス（SS=6/S1=5/S2=4/A1=3/A2=2/A3=1/B=0） | `race_entries.player_class` | 92%（欠損=-1） |
| `is_home` | 地元フラグ（登録府県 == 開催場府県 = 1） | 派生（`player_prefecture` vs `venue_prefecture`） | 6.9%が1（スパース） |

> **除外特徴量**: `line_leader_score` は `line_group` データ 0% のため除外。`FEATURE_COLS_V2` 定義（25特徴量）には残すが、`FEATURE_COLS`（現行）には含まない。

#### バックテスト（テスト期間: 2026-05 / 1947レース）

| 戦略 | v1.5 | v2 | v3 |
|------|------|----|----|
| 3連複上位3頭BOX(1点) | 109.6% | 116.4% | **118.5%** |
| 3連単上位3頭マルチ(6点) | 106.2% | 115.5% | **116.3%** |
| 3連単1着固定×4頭(12点) | - | 100.5% | **102.1%** |

---

### 取得済みだが未使用の項目

DBには存在するが現在のモデルに組み込んでいない項目。データが揃い次第評価する。

| DBカラム | 説明 | 課題 |
|----------|------|------|
| `races.distance` | レース距離（m） | 欠損が多い（スクレイピング精度） |
| `races.weather` | 天候 | 欠損が多い |
| `races.track_condition` | 路面状態 | 欠損が多い |
| `race_entries.line_group` | 何番目のライン班か | パイプライン未保存（要実装） |
| `race_entries.prefecture` | 登録府県（is_home算出に使用済み） | 単体特徴量としては未追加 |

---

### 今後の課題（未実装）

| ファクター | 実装難易度 | 説明 |
|-----------|-----------|------|
| S/H/B分類 | 中 | 競走タイプ（スプリント/ハーフ/バランス）の分類。出走表要調査 |
| コメント解析 | 高 | 「好調」「練習中」等のテキストを調子スコアに変換 |
| オッズ活用 | 中 | 市場オッズを特徴量化（逆張り戦略との組み合わせ） |
| 対戦成績 | 高 | 同一レース出場選手間の過去着順関係（相性指標） |
| 決勝進出率 | 高 | グレードレースでの決勝進出実績 |
| ライン相性 | 高 | 同ライン選手の組み合わせ実績 |
| ライン人数 | 中 | 同ライン内の選手数（多い=先行有利の傾向） |

---

## 場マスタデータ（venue_info）

`src/database.py` の `VENUE_STATIC` で管理。55会場分を登録済み。

| 項目 | 説明 |
|------|------|
| `bank_length` | バンク周長（m）: 250 / 333 / 400 / 500 |
| `is_indoor` | 屋内バンク: 1（千葉のみ） |
| `prefecture` | 開催府県（地元フラグ算出に使用） |

---

## 実装場所

```
src/
├── database.py                      # DBスキーマ・venue_infoマスタ・migrate_db()
├── scraper/
│   ├── keirin_station.py            # スクレイピング（_parse_entry_table）
│   └── pipeline.py                  # DB保存ロジック（_write_race）
└── preprocessing/
    ├── feature_engineer.py          # FEATURE_COLS・build_features()
    └── rolling_stats.py             # compute-stats（6ヶ月勝率・場別勝率等を算出）
```

---

## フェーズ別データ収集・モデル更新手順

段階的にデータを拡充しながらモデルを順次改善する。各フェーズ完了後に以下を実行。

```bash
# Phase 1: 直近4ヶ月（2026-02〜2026-05）→ v1.5モデル
python -m src.cli.main collect-reverse --from 2026-02 --to 2026-05
python -m src.cli.main compute-stats --force
python -m src.cli.main train --model lgbm --from 2025-11-01 --test-from 2026-05-01 --save-as lgbm_v15

# Phase 2: 半年追加（2025-08〜2026-01）→ v1.5再学習
python -m src.cli.main collect-reverse --from 2025-08 --to 2026-01
python -m src.cli.main compute-stats --force
python -m src.cli.main train --model lgbm --from 2025-08-01 --test-from 2026-05-01 --save-as lgbm_v15_p2

# Phase 3: 半年追加（2025-02〜2025-07）→ 再学習
python -m src.cli.main collect-reverse --from 2025-02 --to 2025-07
python -m src.cli.main compute-stats --force
python -m src.cli.main train --model lgbm --from 2025-02-01 --test-from 2026-05-01 --save-as lgbm_v15_p3

# Phase 4: 残り（2024-06〜2025-01）+ v2特徴量へ切り替え → 再学習
python -m src.cli.main collect-reverse --from 2024-06 --to 2025-01
python -m src.cli.main compute-stats --force
# feature_engineer.py の FEATURE_COLS = FEATURE_COLS_V2 に変更
python -m src.cli.main train --model lgbm --from 2024-06-01 --test-from 2026-05-01 --save-as lgbm_v2
```

---

## 更新履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2026-05-27 | **v3実用版** | CV手法を GroupKFold → 日付ベース時系列 fold に修正（未来漏洩解消）。特徴量は v2 と同一（24特徴量）。CV AUC 0.7490（正直値）。バックテスト 2026-05: 3連複3頭BOX 118.5% |
| 2026-05-27 | **v2実用版** | quinella_rate / period_norm / player_class_enc / is_home の4特徴量を追加（24特徴量）。全期間データ（2024-06〜）で再学習。CV AUC 0.7526（GroupKFold漏洩により過楽観）。line_leader_score は line_group データ未収集のため除外 |
| 2026-05-27 | コードレビュー | backtest.py: venues→venue_info統一・run_threshold_analysis高速化・print_venue_analysis閾値動的化。pipeline.py: スキップ時stats誤カウント修正。database.py: player_idインデックス追加。feature_engineer.py: top3_flag NaNガード追加 |
| 2026-05-26 | v1.5設計・移行 | FEATURE_COLS を v1.5（20特徴量）に更新。スキップロジックを quinella_rate IS NOT NULL に変更。フェーズ別再収集・再学習計画を策定。train コマンドに --save-as / --test-from 追加 |
| 2026-05-26 | v2設計 | quinella_rate / period / player_class / is_home / bank_length / venue_win_rate / days_since_last_race / line_leader_score を追加設計。venue_infoテーブル作成。collect-reverseコマンド追加 |
| 2026-05-25 | v1.1 | 場×戦略フィルター廃止（過学習のため）。collect_dateを全会場スキャン方式に変更 |
| 2026-02-24 | v1.0 | LightGBM v1モデル本番稼働（CV AUC 0.7444） |
