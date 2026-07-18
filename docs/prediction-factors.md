# 予想ファクター仕様書

> **最終更新**: 2026-07-19  
> **本番モデル（winticket）**: `lgbm_wt` / **44特徴量** / TRAIN+VAL 2022-12-01〜2026-02-28 / 88,769R学習 / AUC 0.7717  
> **評価専用モデル**: `lgbm_wt_train_only` / TRAINのみ 2022-12-01〜2025-06-30 / 70,540R / AUC 0.7774（VAL期間評価用・HOLD汚染なし）  
> **モデル設計方針**: TRAIN(〜2025-06-30) / VAL(2025-07〜2026-02-28) / HOLD(2026-03〜現在) の3分割。VAL評価=train_only、HOLD評価+live予想=lgbm_wt(TRAIN+VAL)。いずれも評価期間を学習に含まない。  
> **ロールバック保持（keirin-station）**: lgbm_v6 / 24特徴量 / CV AUC 0.7575（2026-06-08 収集停止）  
> **現行戦略（2026-07-17 再設計・2026-07-19 Phase B/複合シグナル拡張・S2/S3 の2ペーパーランクのみ）**: S2=波乱ライン連れ込み三連複（7車・ent≥1.84∧mto≥4.3×穴×同L逃相方×目≥15）、S3=**◎不一致**×軸信頼ゲート（gap12≥0.10 OR 1着モデル内順位≥3 OR p_win/p_top3比≤0.30・3way OR）×システム◎×同L逃相方の三連複・目≥15（7車）。正規プロトコル（1年検証→テスト1回）で合格したのはこの2つのみ。**S1（6車三連単）・A（一致波乱二連単）は検証ROI100%超なしのため 2026-07-17 全廃**（行は picks_history_r_archive / picks_history_a_archive へ退避）。旧S1（7車三連複・7PLUS_R・実賭け）は 2026-07-16 全廃済み。詳細は CLAUDE.md「現行ランク体系」。

---

## 概要

LightGBM を使用した「3着以内（top3）確率」の二値分類モデル。  
選手×レースを1行として特徴量化し、wave-picks コマンドで予想を生成する。

特徴量管理:
- keirin-station ルート: `src/preprocessing/feature_engineer.py` の `FEATURE_COLS`
- winticket ルート: `src/preprocessing/feature_wt.py` の `FEATURE_COLS_WT`

---

## 1. keirin-station ルート（収集停止・ロールバック保持 / 2026-06-08〜）

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

## 2. winticket ルート（★本番稼働中 / 2026-06-08〜）

### 2-1. 特徴量一覧（FEATURE_COLS_WT / 44特徴量）

モデルファイル:
- `data/models/lgbm_wt.pkl`（**本番・live予想用** / TRAIN+VAL 2022-12-01〜2026-02-28 / AUC 0.7717 / 88,769R / 2026-06-17学習）
- `data/models/lgbm_wt_train_only.pkl`（**VAL評価専用** / TRAINのみ 2022-12-01〜2025-06-30 / AUC 0.7774 / 70,540R）
- `data/models/lgbm_wt_v2.pkl`（退避版・2023-07〜2026-06-14・HOLD汚染あり・参照のみ）

> **重要（DNS処理）**: `finish_order=0` は欠車/失格＝着外。`top3_flag` および全評価で `between(1,3)` 判定（0を3着内に誤算入していたバグを2026-06-08修正）。これがwt性能を大きく改善した（A層ROI 70%→187%）。

> **重要（6車立て以下は未使用）**: 現行戦略は **7車以上（7+車）専用**。≤6車ではオッズ構造上の優位性が確認できないため対象外（`docs/analysis/47`・`docs/analysis/08-le6-roadmap.md` 参照）。

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

#### 競走得点トレンド特徴（4項目・2026-07-16追加 / `add_rp_trend_features_wt`）

選手の競走得点の時系列変化（成長/好不調）を捉える。履歴は `wt_entries.race_point` × `wt_races.race_date`。**`finish_order` 未確定（NULL）の過去行は値を集計から除外**（wave-picks の AIスコア上書きが恒久残存する行の汚染対策・行自体は当日レースの merge キーとして保持）。`> 20` はゼロ・欠損系の除外。同一選手・同一日の複数走は median で1点に集約（得点は節内で不変）。rolling は `closed="left"` で当日を除外＝point-in-time保証。rp_prev は直前の非NaN実値。履歴不足（新人等）は 0.0 補完。

| 変数名 | 説明 | DBカラム/計算元 |
|--------|------|----------------|
| `rp_prev_delta` | 今回得点 − 前回出走時（前回の異なる race_date）の得点 | `wt_entries.race_point` の選手別 shift(1) |
| `rp_delta_90` | 今回得点 − 過去90日の平均得点（当日を含まない） | 同 rolling("90D", closed="left") 平均との差 |
| `rp_delta_180` | 今回得点 − 過去180日の平均得点 | 同 rolling("180D") |
| `rp_trend` | 過去90日平均 − 過去180日平均（中期トレンド） | 上記2平均の差 |

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

- カット定数の既定値 `UPSET_TOP3SUM_CUTS_DEFAULT=(1.70, 1.90, 2.08)`（TRAIN 2023-07〜2026-02 の四分位）。**週次再学習後に `scripts/recompute_upset_cuts_wt.py` が train分布で自動再計測し `data/models/upset_cuts_wt.json` に保存**、`strategy_wt._load_cuts()` がこれを優先採用（無ければ既定値）。手動再計測も同スクリプトで可能。
- `wave-picks-wt` は各pickに `top3_sum`/`upset_tier` を**タグ付け（既定・detail.json記録）**。`--upset-gate Q1_loose|Q2|Q3` で本命堅レースを見送るopt-inフィルター。**既定は全件出力＝本番挙動不変**（前向き検証用）。
- ⚠️ ROIは**最終データbacktest=実運用上限値**。live検証は picks_history(route='wt') × detail.jsonの `upset_tier` で別途。

### 2-5. ガミ回避オッズ3段階（`--gami-skip-odds`/`--b-rank-odds`・2026-06-08 採用）

3点の**最安目の朝オッズ**で振り分け（日次cron既定 `--gami-skip-odds 3.0 --b-rank-odds 5.0`）:
- **<3倍**: 見送り（明確なガミ）/ **3〜5倍未満**: Bランク（別枠・購入は各自判断）/ **≥5倍**: 通常推奨(SS/S/A)。

検証は `scripts/analyze_gami_threshold_wt.py`、詳細は `docs/bet-structure-guide.md`。「安い目カット」より「レース単位振り分け」が ROI・総損益とも上（TEST 全件286%→<5倍除外636%、総利益ほぼ維持）。Bランクは推奨合計に含めない（detail.json `rank="B"`/`base_rank`保持）。`top3_sum` 波乱ゲートと同義シグナルだが、こちらは朝オッズ基準（ドリフト計測中）。

## 3. 場マスタデータ（venue_info）

`src/database.py` の `VENUE_STATIC` で管理。55会場分登録済み。  
winticket 対応会場（43場）は `src/scraper/winticket.py` の `VENUE_SLUGS` を参照。

| 項目 | 説明 |
|------|------|
| `bank_length` | バンク周長（m）: 250 / 333 / 400 / 500 |
| `is_indoor` | 屋内バンク: 1（千葉のみ）|
| `prefecture` | 開催府県（地元フラグ算出に使用）|
| `straight_len` | 直線長（m）: 2026-06-12 追加（`docs/analysis/20-web-logic-audit.md`・宇都宮500等48行の誤記訂正済） |
| `cant_deg` | カント角（度）: 2026-06-12 追加（同上） |

> **注**: `straight_len`/`cant_deg` は FEATURE_COLS_WT には含まれない（風特徴 G06 Phase1 不通過と同様、レース内相対に無効）。venue_info 副産物として記録。

---

## 4. 今後の課題

| 課題 | 状況 | 方針 |
|------|------|------|
| wt実運用ROIの実測 | 蓄積中 | `picks_history(route='wt')` で朝-確定ズレ込みの真のROIを測定。backtestは最終データ上限値（実測ks 1週間49%）|
| 朝→最終オッズ ドリフト計測 | 2026-06-08〜蓄積開始 | `snapshot_morning_odds_wt.py --report`。ガミ3段階(3倍/5倍)が朝オッズで妥当か検証 |
| 波乱ゲート(top3_sum)の本番反映可否 | 検証中 | detail.json `upset_tier` × picks_history で帯別live ROIを確認後に判断 |
| Bランク(3〜5倍)の実成績検証 | 蓄積中 | 「Bを買うべきだったか」を detail.json `rank=B` × 結果で事後検証 |
| ~~週次再学習でのカット定数再計測~~ | **解決済(2026-06-08・自動化)** | `weekly_retrain_wt.sh` が `recompute_upset_cuts_wt.py` を実行→`upset_cuts_wt.json` 更新→`strategy_wt` が自動採用 |
| ~~L級（ガールズ）クラス未マッピング~~ | **解決済(2026-06-08)** | `feature_wt._CLASS_MAP` に `cls4`(L級ガールズ→7)・`cls1`(S級下位→4)を追加し再学習。AUC中立(0.7719/0.7741)・`player_class_enc=-1` 解消 |

> 詳細な検証レポートは `docs/analysis/01〜03`（特徴ablation・波乱予測・オッズ活用）を参照。

---

## 5. 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-07-19 | **S3(M)ゲート3way OR拡張**: システム◎の p_win/p_top3 比 ≤0.30（`M_RATIO_MAX`）を第3項としてOR追加（既存 gap12≥0.10 OR win_rank≥3 に統合）。win_rank（順位・離散量）の連続量版。加法差(diff=p_top3-p_win)は無判別力で不採用、乗法比(ratio)のみ有効と判明（`exp_composite_prob_diff_wt.py`）。正規プロトコル: 検証158.2%(531R)→158.6%(671R)・テスト149.5%(152R)→154.3%(186R)、母数さらに+22〜26%。`m_axis_gate`（`src/strategy_wt.py`）拡張・`src/cli/main.py`/`scripts/backfill_um_rank_wt.py`/`scripts/notify_prerace_wt.py` 対応。ペーパートレード継続（live実測フォロー中）。 |
| 2026-07-16 | **特徴追加: 競走得点トレンド4特徴（rp_prev_delta / rp_delta_90 / rp_delta_180 / rp_trend）**。選手単位の得点時系列変化＝成長/好不調シグナル（`add_rp_trend_features_wt`・point-in-time・closed="left" で当日除外）。A/B検証: ΔAUC +0.0009〜0.001 / 1位勝率 +0.15pt・2独立窓で方向一致。FEATURE_COLS_WT 40→44。 |
| 2026-06-17 | **モデル設計刷新（3分割・汚染なし）**: lgbm_wt を TRAIN+VAL（2022-12-01〜2026-02-28）で再学習（AUC 0.7717・88,769R）。lgbm_wt_train_only（TRAINのみ）を VAL評価専用に分離。旧lgbm_wt（HOLD汚染あり）を lgbm_wt_v2 として退避。現行戦略を **7+車専用**（6車立て以下は使用しない）に明記。HOLD バックテスト結果: SS 137.8%★ / S 138.8%★ / A 99.4% / 合計 134.3%★（2026-03〜06-16・3,076R）。 |
| 2026-06-13 | **ドキュメント同期（G08）**: G01〜G07完了に伴い各ドキュメントを更新。venue_info に `straight_len`/`cant_deg` 追加記録（`docs/analysis/20-web-logic-audit.md` 副産物・宇都宮500等48行誤記訂正済）。FEATURE_COLS_WT への変更なし（G06風特徴 Phase1 不通過・無情報）。新規スクリプト（G02〜G07）を `docs/system-architecture.md` に追記。|
| 2026-06-12 | **バックテスト3バイアス修正（G01 移植）**: `backtest_wt.py` 本体に①欠車生存バイアス（全エントリーでランキング）②≤6車フィルタ位置（pred_prob付与前=出走表基準）③欠車void（DNS含む組の不計上）を移植。`src/evaluation/void_rules.py` 新設。`--eval-model` オプション追加。スポットチェック ROI 80.4%（doc18 の~84% と同オーダー）。|
| 2026-06-09 | **特徴追加: n_senko（レース内の逃げ人数＝展開シグナル）**。4サイト監査(oddspark等)→n_linesと独立の波乱シグナルと検証→特徴量化。FEATURE_COLS_WT 39→40。再学習で holdout AUC 0.7778→**0.7784**・層別合計393→**404%**(A層220→251%)＝小幅改善・非劣化で採用。外部の穴俗説(333初日/ミッドナイト/A級波乱)はデータ非再現で棄却。|
| 2026-06-08(夜9) | **波乱の解剖＋脚質バグ修正**: `docs/analysis/04-upset-anatomy.md`（波乱は n_lines が最大の事前条件・波乱時の伏兵は「非本命ライン先頭・指数3-4位」）。探索中に **`style_enc` 全件-1（脚質特徴が死亡）** を発見＝winticket値は `逃/両/追` だが `_STYLE_MAP` が旧表記前提でキー不一致。`逃=0/両=1/追=2` に修正し再学習（AUC中立 0.7777→0.7778＝s_count等で実質取込済だが特徴正常化・脚質次元分析が可能に）。|
| 2026-06-08(夜8) | **波乱ステーク傾斜(方針A)実装**: `--stake-tilt`（top3_sum帯で賭け金傾斜 Q1_loose×2/Q2×1/Q3,Q4見送り。`strategy_wt.stake_units`）。検証 `scripts/exp_stake_tilt_wt.py`（eval OOS・上限値）: **TEST ROI フラット351%→傾斜745%**（最大除640%・train/test順序一致）。既定off（分散増・上限値のためlive実測後に有効化判断）。テスト36件pass。|
| 2026-06-08(夜7) | **M-1/M-2 修正**: M-1 推論特徴を `prepare_X`(fillna0) に統一（wave-picks/eval/backtest）＋ build_features_wt 末尾で FEATURE_COLS_WT 保証fill（dropna vs fillna の skew排除）。M-2 学習母集団を `finish_order≥1` に統一（DNS負例除去）。再学習で **holdout AUC 0.7741→0.7777 改善**（fit 562,265行）。テスト30件pass。|
| 2026-06-08(夜6) | **コードレビュー指摘の修正**（`docs/analysis/code-review-2026-06-08.md`）: H-1 配信/評価モデル分離（`train-wt --full-refit`で全データ配信・`--no-promote`・メタsidecar、weeklyを評価→配信→カット→世代退避に再編）/ M-5 世代退避(`data/models/archive/`) / L-5 pipefail / L-13 recompute非ゼロ終了 / H-2 pytest基盤(tests/・26件)。配信モデルを全データ再学習に切替（holdout監視AUC 0.7741 維持）。|
| 2026-06-08(夜5) | **波乱ゲート カット定数の自動再計測**: `weekly_retrain_wt.sh` に `recompute_upset_cuts_wt.py` を追加。再学習後の train分布で top3_sum 四分位を再計測→`data/models/upset_cuts_wt.json`(gitignore)→`strategy_wt._load_cuts()` が優先採用（無ければ既定値）。現行再計測値 (1.693/1.901/2.075)＝既定とほぼ不変。|
| 2026-06-08(夜4) | **L級(ガールズ)クラスのマッピング追加**: `feature_wt._CLASS_MAP` に `cls4`(L級→7)・`cls1`(S級下位→4)。約7.7%が `player_class_enc=-1` だった問題を解消し再学習（CV AUC 0.7719/Test 0.7741＝中立）。モデルは新マッピングで再学習済（lgbm_wt 上書き）。|
| 2026-06-08(夜3) | **ガミ回避を3段階化**: `--gami-skip-odds 3.0 --b-rank-odds 5.0`。最安目<3倍=見送り / 3〜5倍未満=**Bランク（購入者判断にゆだねる別枠）** / ≥5倍=通常推奨。Bランクは推奨合計に含めず（detail.json `rank="B"`/`base_rank`）。日次cron反映。|
| 2026-06-08(夜2) | ガミ回避レーススキップ採用（当初 `--gami-skip-odds 5.0` 単一閾値・後に夜3で3段階化）。検証 `scripts/analyze_gami_threshold_wt.py`（<3倍点含むレースは集団で収支ゼロ＝スキップが「安い目カット」より ROI・総損益とも上、TEST 286%→636%@5倍）。|
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
