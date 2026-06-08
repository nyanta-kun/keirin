# 予想ファクター仕様書

> **最終更新**: 2026-06-08  
> **現行モデル（keirin-station）**: lgbm_v6 / 24特徴量 / CV AUC **0.7575**  
> **winticket モデル**: lgbm_wt（lgbm_wt_v1）/ **39特徴量** / CV AUC **0.7720** / Test AUC 0.7742（2022-12〜2026-06 全期間収集済 96,355R）  
> **現行戦略**: 6車立て以下 SS/S/A 3段階ランク（2026-06-05改定）

---

## 概要

LightGBM を使用した「3着以内（top3）確率」の二値分類モデル。  
選手×レースを1行として特徴量化し、wave-picks コマンドで予想を生成する。

特徴量管理:
- keirin-station ルート: `src/preprocessing/feature_engineer.py` の `FEATURE_COLS`
- winticket ルート: `src/preprocessing/feature_wt.py` の `FEATURE_COLS_WT`

---

## 1. keirin-station ルート（本番稼働中）

### 1-1. 現行特徴量（v6実用版 / 24特徴量）

モデルファイル: `data/models/lgbm.pkl`（= `lgbm_v6.pkl`）

#### 選手成績（7項目）

| 変数名 | 説明 | DBカラム |
|--------|------|----------|
| `racing_score` | JKA競走得点 | `race_entries.racing_score` |
| `gear_ratio` | ギヤ倍数（例: 3.92） | `race_entries.gear_ratio` |
| `recent_win_rate_3m` | 直近3ヶ月勝率（0-1） | `race_entries.recent_win_rate_3m` |
| `recent_top3_rate_3m` | 直近3ヶ月3着内率 | `race_entries.recent_top3_rate_3m` |
| `recent_win_rate_6m` | 直近6ヶ月勝率（compute-stats） | `race_entries.recent_win_rate_6m` |
| `recent_top3_rate_6m` | 直近6ヶ月3着内率 | `race_entries.recent_top3_rate_6m` |
| `wr_trend` | 勝率トレンド（3m − 6m） | 派生 |

#### 会場・場別（3項目）

| 変数名 | 説明 | DBカラム |
|--------|------|----------|
| `venue_win_rate` | 同会場での通算勝率（compute-stats） | `race_entries.venue_win_rate` |
| `bank_length_enc` | バンク周長 / 100 | `venue_info.bank_length` |
| `is_indoor` | 屋内バンクフラグ（千葉のみ 1） | `venue_info.is_indoor` |

#### レース内相対（6項目）

| 変数名 | 説明 |
|--------|------|
| `score_rank` | 競走得点のレース内順位（1=最高） |
| `score_z` | 競走得点のレース内偏差値（clip±5） |
| `wr_rank` | 3ヶ月勝率のレース内順位 |
| `top3r_rank` | 3ヶ月3着内率のレース内順位 |
| `is_inner` | 内枠フラグ（車番1〜3） |
| `is_outer` | 外枠フラグ（車番7以上） |

#### 戦術・選手属性（5項目）

| 変数名 | 説明 | DBカラム |
|--------|------|----------|
| `line_pos_enc` | 脚質（先行=0/捲り=1/差し=2/追い込み=3） | `race_entries.line_position` |
| `frame_no` | 車番（1〜9） | `race_entries.frame_no` |
| `quinella_rate` | 2着内率（連対率） | `race_entries.quinella_rate` |
| `player_class_enc` | クラス（SS=6/S1=5/S2=4/A1=3/A2=2/A3=1/B=0） | `race_entries.player_class` |
| `is_home` | 地元フラグ（登録府県 == 開催場府県） | 派生 |

#### レース条件（3項目）

| 変数名 | 説明 | DBカラム |
|--------|------|----------|
| `grade_enc` | グレード（GP=7/G1=6/G2=5/G3=4/F1=3/F2=2/A=1） | `races.grade` |
| `days_since_last_race` | 前走からの経過日数（compute-stats） | `race_entries.days_since_last_race` |
| `period_norm` | 期別 / 100（小さい=ベテラン） | `race_entries.period` |

---

### 1-2. モデル履歴

| バージョン | 特徴量数 | CV AUC | データ期間 | 備考 |
|-----------|---------|--------|-----------|------|
| v1 | 13 | 0.7444 | 〜2026-02 | ベースライン |
| v1.5 | 20 | 0.7495 | 〜2026-03 | rolling stats 追加 |
| v2 | 24 | 0.7526 | 〜2026-04 | GroupKFold（未来漏洩あり）|
| v3 | 24 | 0.7490 | 〜2026-05 | 日付ベース時系列CV（漏洩修正）|
| v4 | 24 | 0.7467 | 〜2026-02 | テスト期間3ヶ月に拡大 |
| v5 | 24 | 0.7466 | 〜2026-06-04 | 全DB再学習 |
| **v6（現行）** | 24 | **0.7575** | 〜2025-05（学習）| 2023年〜追加収集・ホールドアウト9ヶ月検証 |

---

### 1-3. バックテスト結果（v6 / ホールドアウト 2025-06〜2026-02）

> **真の独立テスト（戦略チューニング未使用）**

| ランク | 条件 | 買い目 | 件数 | 的中率 | ROI | avg的中払戻 |
|--------|------|--------|------|--------|-----|------------|
| **SS** | gap12≥0.15 & ratio<1.3 | 3連単 3点300円 | 157R | 19.7% | **3,944%** | 52,287円 |
| **S** | gap12≥0.15 & ratio [1.3, 1.6) | 3連複 3点300円 | 691R | 50.9% | **158%** | 928円 |
| **A** | gap12 [0.06, 0.15) | 3連複 3点300円 | 767R | 44.6% | **228%** | 1,515円 |
| **合計** | — | — | **1,615R** | 44.9% | **519%** | — |

月別安定性（2025-06〜2026-02）: **9ヶ月連続プラス**。SS のみ 2025-08 が月 ROI 56%（唯一の赤字月）。

---

### 1-4. 取得済みだが未使用の項目

| DBカラム | 説明 | 課題 |
|----------|------|------|
| `races.distance` | レース距離（m） | 欠損多い |
| `races.weather` | 天候 | 欠損多い |
| `race_entries.line_group` | ライン班番号 | パイプライン未保存 |

---

## 2. winticket ルート（収集済み・本番モデル稼働）

### 2-1. 特徴量一覧（FEATURE_COLS_WT / 39特徴量）

モデルファイル: `data/models/lgbm_wt.pkl`（= lgbm_wt_v1, 2023-07〜2026-02学習 / CV AUC 0.7720）

> **重要（DNS処理）**: `finish_order=0` は欠車/失格＝着外。`top3_flag` および全評価で `between(1,3)` 判定（0を3着内に誤算入していたバグを2026-06-08修正）。これがwt性能を大きく改善した（A層ROI 70%→187%）。

#### keirin-station ルートと共通の概念

| winticket 変数名 | 対応する ks 変数 | 説明 |
|-----------------|----------------|------|
| `race_point` | `racing_score` | JKA競走得点相当（winticket 表示値） |
| `first_rate_norm` | `recent_win_rate_3m` | 勝率（winticket は%表記 → /100 変換）|
| `third_rate_norm` | `recent_top3_rate_3m` | 3着内率 |
| `style_enc` | `line_pos_enc` | 脚質エンコード |
| `period_norm` | `period_norm` | 期 / 100 |
| `player_class_enc` | `player_class_enc` | クラスエンコード（同一マッピング）|
| `gear_ratio` | `gear_ratio` | ギヤ比 |
| `grade_enc` / `bank_length_enc` / `is_indoor` | 同 | 共通 |
| `is_inner` / `is_outer` / `frame_no` | 同 | 枠番 |
| `score_rank` / `score_z` / `wr_rank` / `top3r_rank` | 同 | レース内相対 |
| `is_home` | `is_home` | 地元フラグ |

#### winticket 固有の新特徴量（12項目）

| 変数名 | 説明 | DBカラム |
|--------|------|----------|
| `line_size` | 同ライン内の選手数 | `wt_entries.line_size` |
| `line_pos` | ライン内ポジション（1=先頭） | `wt_entries.line_pos` |
| `is_line_leader` | ライン先頭フラグ | `wt_entries.is_line_leader` |
| `n_lines` | レース内のライン数 | `wt_entries.n_lines` |
| `is_isolated` | 単騎（line_size==1）フラグ | 派生 |
| `line_frac` | レース内でのライン規模比率 | 派生 |
| `s_count` | 先行セクター回数 | `wt_entries.s_count` |
| `h_count` | ホームセクター回数 | `wt_entries.h_count` |
| `b_count` | バックセクター回数 | `wt_entries.b_count` |
| `ex_spurt_pct` | 追い込み率（0-1に正規化） | `wt_entries.ex_spurt_pct` |
| `ex_thrust_pct` | 捲り率（0-1に正規化） | `wt_entries.ex_thrust_pct` |
| `prediction_mark` | winticket AI印（0=なし/1=本命/2=対抗/3=単穴/4=連下） | `wt_entries.prediction_mark` |

#### ks流ローリング特徴（9項目・2026-06-08追加 / `add_rolling_features_wt`）

選手の過去成績から point-in-time（現レース日より前のみ・欠車除外）で計算。学習時は履歴 merge、予測時は当日 as-of 計算。

| 変数名 | 説明 |
|--------|------|
| `win_3m` / `win_6m` | 直近3ヶ月 / 6ヶ月の1着率 |
| `top3_3m` / `top3_6m` | 直近3ヶ月 / 6ヶ月の3着内率 |
| `quin_3m` / `quin_6m` | 直近3ヶ月 / 6ヶ月の2着内率 |
| `venue_wr` | 当該会場での過去勝率 |
| `days_since` | 前走からの日数 |
| `wr_trend` | 勝率トレンド（win_3m − win_6m） |

---

### 2-2. オッズ活用方針

> オッズはモデルの特徴量に**含めない**。AI予想後の購入判断に使用する。

- AI が予想を生成
- `wt_odds` テーブルから対象組み合わせのオッズを取得・表示
- `wave-picks-wt --min-trio-odds N` で N 倍未満の組み合わせを自動フィルタ
- 低オッズ = 市場が既に織り込み済み → 配当価値が低い

---

### 2-3. winticket ルートの学習・実行手順

```bash
# 1. データ収集（最低2,000レース推奨）
python -m src.cli.main collect-wt-range --from 2025-06

# 2. モデル学習
python -m src.cli.main train-wt --from 2025-06-01 --test-from 2026-03-01

# 3. 予想生成（オッズフィルター付き）
python -m src.cli.main wave-picks-wt --date 2026-06-06 --min-trio-odds 3.0
```

---

### 2-4. 波乱/非本命ゲート（`src/strategy_wt.py`・2026-06-08 試験実装）

3タスク分析（`docs/analysis/01〜03`）が収束した「本命が堅いレースは低ROI、本命が割れた波乱余地レースが高ROI」を、確定前指標 **`top3_sum`（上位3頭の pred_prob 合計）** のloose四分位で実装。

| 帯（TRAIN四分位カット） | top3_sum | TRAIN ROI | TEST(OOS) ROI |
|---|---|---|---|
| Q1_loose（波乱余地大） | < 1.70 | 1224% | **1136%**（125R・最大払戻除外934%）|
| Q2 | 1.70–1.90 | 193% | 224% |
| Q3 | 1.90–2.08 | 112% | 103% |
| Q4_chalk（本命堅） | ≥ 2.08 | 88% | 107% |

- カット定数 `UPSET_TOP3SUM_CUTS=(1.70, 1.90, 2.08)` は TRAIN 2023-07〜2026-02 の四分位。**モデル再学習で確率分布が変わったら `scripts/exp_upset_gate_wt.py` で再計測すること**。
- `wave-picks-wt` は各pickに `top3_sum`/`upset_tier` を**タグ付け（既定・detail.json記録）**。`--upset-gate Q1_loose|Q2|Q3` で本命堅レースを見送るopt-inフィルター。**既定は全件出力＝本番挙動不変**（前向き検証用）。
- ⚠️ ROIは**最終データbacktest=実運用上限値**。live検証は picks_history(route='wt') × detail.jsonの `upset_tier` で別途。

## 3. 場マスタデータ（venue_info）

`src/database.py` の `VENUE_STATIC` で管理。55会場分登録済み。  
winticket 対応会場（43場）は `src/scraper/winticket.py` の `VENUE_SLUGS` を参照。

| 項目 | 説明 |
|------|------|
| `bank_length` | バンク周長（m）: 250 / 333 / 400 / 500 |
| `is_indoor` | 屋内バンク: 1（千葉のみ）|
| `prefecture` | 開催府県（地元フラグ算出に使用）|

---

## 4. 今後の課題

| 課題 | 状況 | 方針 |
|------|------|------|
| winticket データ収集・モデル検証 | **未着手** | keirin-station との比較検証。優位性なければ廃棄 |
| SS 的中率 95%CI 改善（現 ±6.2%） | 実運用中 | 約2〜3ヶ月の実績蓄積で ±5% 未満へ |
| 春季ホールドアウト（2026-06〜）| 未蓄積 | 2026年夏以降に再評価 |
| lgbm_v7 再学習 | 未定 | 実運用SS 30件以上または実績が大幅乖離した場合 |
| line_leader_score の有効化 | 未実装 | keirin-station の `line_group` 収集が先決（winticket では取得可能）|

---

## 5. 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-06-08(夜) | 3タスク分析（`docs/analysis/01〜03`）→ 全タスクが「波乱/非本命レースが高ROI」に収束。**波乱ゲート `src/strategy_wt.py` 試験実装**（`top3_sum` loose四分位・Q1_loose TEST ROI 1136%）。`wave-picks-wt` に `upset_tier` タグ付け＋`--upset-gate` opt-inフィルター。③レポート`top2_sum<0.80`はスケール誤りで撤回。朝オッズ前向き計測（`wt_odds_snapshot`＋`snapshot_morning_odds_wt.py`）を仕込み。AUC↑≠ROI↑・AI印はROI低下を再確認、閾値は現状維持。|
| 2026-06-08 | winticket 全期間収集完了（96,355R）。**DNS(finish_order=0)バグ修正**（着外を3着内に誤算入していた）でwt性能大幅改善（A層ROI 70%→187%・S 364%・SS 1205%・合計336%、ks同等以上）。**ks流ローリング特徴9項目追加→FEATURE_COLS_WT 30→39特徴**。lgbm_wt_v1 学習（CV AUC 0.7720/Test 0.7742）。wave-picks-wt 実運用化（発走時刻バグ修正）。notify_results 成績バグ修正（公開予想採点・月次ROI 102%→49%再採点）。|
| 2026-06-06 | winticket ルート（30特徴量 FEATURE_COLS_WT）設計・実装完了を追記。model-overview.md を本ファイルに統合。|
| 2026-06-05 | S ランクに ratio<1.6 上限追加（低配当レース除外）。ホールドアウト再検証: S 727R→392R / ROI 149.8%→177.1% / avg配当 928円→1,170円 |
| 2026-06-04 | lgbm_v6 学習完了。学習 52,472R / ホールドアウト 1,615R（9ヶ月）。ROI: SS 3,944% / S 158% / A 228% |
| 2026-06-03 | SS/S/A 3段階ランク戦略導入。7車立て・upset_prob戦略検討（不採用）。バックテスト全面再検証 |
| 2026-06-02 | wave-picks 6車立て以下 jiku2_3 戦略確定。lgbm_v5 再学習（AUC 0.7466） |
| 2026-05-27 | v3（時系列CV修正）→ v2（24特徴量）→ v1.5（20特徴量）段階的改善 |
| 2026-05-26 | v2 設計（quinella_rate / period / player_class / is_home / bank_length 追加）|
| 2026-02-24 | v1.0 本番稼働（13特徴量 / AUC 0.7444）|
