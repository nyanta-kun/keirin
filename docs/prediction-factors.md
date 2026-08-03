# 予想ファクター仕様書

> **最終更新**: 2026-08-02  
> **本番モデル（winticket）**: `lgbm_wt` / **46特徴量**（2026-07-31にex_spurt_pct/ex_thrust_pctをtrain/serve skewのため除外し48→46・下記参照） / 全期間(2022-12-01〜) full-refit / 706,230エントリー・99,776R学習 / holdout AUC 0.7793（test-from=直近90日で評価後、全データ再学習）  
> **評価専用モデル**: `lgbm_wt_eval` / TRAINのみ〜test-from(直近90日)前 / 658,339エントリー / holdout AUC 0.7765（honest backtest再構築用・HOLD汚染なし）  
> **1着専用モデル**: `lgbm_wt_win` / holdout AUC 0.8258（`lgbm_wt_win_eval` holdout AUC 0.8214）。S1の軸選定に使用（2026-07-19導入）  
> **モデル設計方針**: `weekly_retrain_wt.sh`（毎週日曜23:30）が①`--test-from`直近90日でholdout評価→AUCゲート（top3系≥0.75・win系≥0.78、前回比-0.02超悪化で昇格中止）→②全データfull-refitで配信モデル更新→③波乱ゲートcut再計測→④世代退避（`data/models/archive/`）の順で実行。  
> **ロールバック保持（keirin-station）**: lgbm_v6 / 24特徴量 / CV AUC 0.7575（2026-06-08 収集停止）  
> **現行戦略（2026-07-23時点・S1/SS+/SS/S の4ペーパーランク）**: S1=win軸1着固定×3着内モデル相手2車（7車・軸=1着専用モデル1位×top3_gap≥0.15×軸単勝勝率≤50%×軸級班≠S1/A1×三連単2点流し・目下限なし）。S2(旧U)・S3(旧M)は対象レース数・的中率・期待値の観点で継続困難と判断し2026-07-21に全廃（過去行は picks_history_u_archive / picks_history_m_archive へ退避）。S4(単勝×複勝指数トップ3重なり軸×波乱度選出の三連複2軸総流し・7車)は軸2車がWINTICKET公式◎◯と重なる数で**SS**（重なり0・全く重ならない）と**S**（重なり1・片方だけ重なる、日次axis_sum上位10件）に再編、SSのうち軸級班に各グレード最上位(S1/A1)を含まないレースを観察用サブランク**SS+**として表示分岐（2026-07-23導入・買い目はSSと同一）。**旧新S1（6車三連単）・A（一致波乱二連単）は検証ROI100%超なしのため 2026-07-17 全廃**（行は picks_history_r_archive / picks_history_a_archive へ退避）。旧S1（7車三連複・7PLUS_R・実賭け）は 2026-07-16 全廃済み。詳細は CLAUDE.md「現行ランク体系」。  
> **【2026-07-23・重要】race_point特徴量の自己参照汚染を修正**: 2026-06-18〜07-23、`wt_entries.race_point`（`score_z`等の学習特徴量の入力）がAI予測確率で上書きされ続け週次再学習が自己汚染していたバグを発見・修正（詳細は本ファイル更新履歴・kiseki側メモリ`keirin_race_point_feature_leak_2026_07_23`）。汚染期間の生データ再取得・モデル全面再学習・S1/S4のtailウィンドウ(2026-04-13〜)再構築まで完了済み。
> **【2026-07-31・train/serve skew検出→FEATURE_COLS_WTから除外済み】** `ex_spurt_pct`/`ex_thrust_pct`（2-1節「winticket 固有の新特徴量」参照。旧記述は「2-2節」と誤記していたため本更新で訂正）が開催中に値が更新される（同一開催Day1と最終日でそれぞれ5.36%/2.39%のペアで値が変化）ことを確認。`_get_collected_keys`が結果確定済みレースのみスキップする仕様と組み合わさると学習データにそのレース自身の結果を反映した値が混入しうる。12ヶ月・約194,000サンプルのA/B測定（`scripts/exp_ab_leaky_ex_features.py`）でAUC寄与が事実上ゼロ（eval 0.7732→0.7731 / win 0.8233→0.8233・honest ROIも有意差なし）と確認された上で、「性能が上がるから」ではなく「予測に何も貢献していないのにリスクだけを抱えているため」`FEATURE_COLS_WT`から除外（48→46特徴・ユーザー承認済み）。除外後の全モデル再学習はPMが別途実施する。詳細は該当箇所の注記・更新履歴参照。
> **【2026-07-31・ランク名体系化】** 内部rank名・suffixを表示ラベル基準へ全面改名（commit `f31f84b`。表示ラベルは変更なし）。`SEVEN_S7`→`RANK_7S`・`SEVEN_7A`→`RANK_7A`・`NINE_S9`→`RANK_9S`・`NINE_9A`→`RANK_9A`・`SEVEN_SS`→`RANK_7SS`（**`RANK_7SS`は2026-08-02に全廃**・下記）。以降の本文・更新履歴に残る旧名表記はいずれも改名前の記述として読むこと。対応表・命名規則は CLAUDE.md「現行ランク体系」節の「ランク名体系化」サブ節を参照。
> **【2026-08-03・7B新設】** `RANK_7B`（◎◯一致だが順序・相手で不一致・三連複3点）を新設し、現行は **7S / 7A / 7B / 9S / 9A の5ペーパーランク**になった。詳細は更新履歴の2026-08-03エントリ・CLAUDE.md「現行ランク体系」節を参照。
> **【2026-08-02・7SS全廃】** `RANK_7SS`（波乱軸選出・穴レース検知・モデル非依存）を全廃し、現行は **7S / 7A / 9S / 9A の4ペーパーランク**になった（commit `7048db5`）。live実績が picks_history 全期間 **n=16,298・ROI 73.5%**、2026年の月次も1月以外すべて70%以下と控除率75%を下回り続けたため（ユーザー判断）。候補生成・ライブ判定・採点・Discord通知・Web表示・DB実績（16,298行）をすべて停止／削除。判定ロジック（`rank_7ss_*` / `backfill_7ss_rank_wt.py` / `tests/test_7ss_void_by_dns_unification.py`）は再設定に備え残置（探索用の `exp_7ss_*.py` 6本はgit未追跡だったため2026-08-02に削除・復元不可）。詳細・再開手順は kiseki側メモリ `keirin_7ss_abolition_2026_08_02`。

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

### 2-1. 特徴量一覧（FEATURE_COLS_WT / 46特徴量）

> 補足: 本節の表題の特徴量数は2026-07-19のS/B展開特徴追加（44→48）に追従できて
> いなかった既存の記載漏れがあり、本更新（2026-07-31・48→46）と合わせて実値に
> 訂正した。

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

#### winticket 固有の新特徴量（10項目）

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
| `prediction_mark` | winticket AI印（0=なし/1=本命/2=対抗/3=単穴/4=連下） | `wt_entries.prediction_mark` |

> **【2026-07-31・除外済み】`ex_spurt_pct`（追い込み率相当）/ `ex_thrust_pct`
> （捲り率相当）は FEATURE_COLS_WT から除外**（48→46特徴）。同一開催の
> Day1時点の値と最終日時点の値をn=221,551ペアで比較した結果、
> `ex_spurt_pct` は5.36%のペアで値が変化（変化時の平均+8.2pt）、
> `ex_thrust_pct` は2.39%のペアで変化（変化時の平均+22.9pt）していた。
> つまりこの2特徴量は**開催中に値が更新される**（train/serve skew）。一方
> `race_point` / `first_rate_norm` / `third_rate_norm`（相当する
> `first_rate`/`second_rate`/`third_rate`）/ `s_count` / `b_count` は
> 0.00〜0.14%とほぼ完全に開催単位で固定＝安全と確認済み。
> `_get_collected_keys`（`src/scraper/pipeline_wt.py:168-183`）は
> 「結果確定済み（`finish_order>=1`）」のレースのみをスキップ対象とし、
> 未確定レースは結果が付くまで再収集を続ける仕様のため、上記2特徴量は
> 学習データに「そのレース自身の発走後の結果を反映した値」が混入しうる
> （`sb_dyn` 系特徴で過去に確認された自己参照汚染と同型の構造）。
> 12ヶ月・約194,000サンプルのA/B測定（`scripts/exp_ab_leaky_ex_features.py`）で
> AUC寄与が事実上ゼロ（eval AUC 0.7732→0.7731 / win AUC 0.8233→0.8233・honest
> ROIも有意差なし）と確認された上で、**「性能が上がるから」ではなく「予測に
> 何も貢献していないのにリスクだけを抱えているため」除外**した（ユーザー承認済み）。
> 除外後、`ex_spurt_pct`/`ex_thrust_pct`の SELECT・0-1正規化自体（`load_raw_data_wt`/
> `build_features_wt`）は分析用途・将来のpoint-in-time化のためコードには残置。
> `ex_left_behind_pct`（変化ペア21.4%）/ `ex_split_line_pct`（24.9%）/
> `ex_snatch_pct`（1.5%）も同様に開催中更新される実測があるが、これらは元々
> FEATURE_COLS_WTに含まれておらずSELECTのみのため対応不要（将来の誤採用防止のため記録）。
> 除外後の全モデル再学習はPMが別途実施する。詳細はkiseki側メモリ参照
> （本ファイル更新履歴にも記録）。

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

#### S/B展開特徴（4項目・2026-07-19採用 / `add_sb_dyn_features_wt`）

レース単位のS（スタンディング先頭）/B（バック先頭）取得・上がりタイム由来のローリング特徴。
データ源は`wt_entries.res_standing`/`res_back`/`final_half`（2024-01〜バックフィル済み）。
全て過去レースのみ・`closed="left"`90日窓でpoint-in-time保証。

| 変数名 | 説明 |
|--------|------|
| `b_rate_90` | 直近90日のB（バック先頭）取得率 |
| `s_rate_90` | 直近90日のS（スタンディング先頭）取得率 |
| `fh_rel_90` | 直近90日の上がり相対値平均（自上がり − レース中央値・負=速い） |
| `fh_best_rate_90` | 直近90日の「レース内上がり最速」率 |

A/B検証（`exp_sb_dyn_ab.py`・2独立窓×5seed）: ΔAUC +0.013/+0.011・指数1位3着内率 +0.93pt/+1.10pt・
重要度2〜9位/48。FEATURE_COLS_WT 44→48。DNF（`finish_order<1`）除外バグを発見・修正の上で採用
（詳細はkiseki側メモリ`keirin_sb_dynamics_pipeline`）。

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

# 3. 隊列位置バイアス補正（2段目モデル）を学習 ※2-6節。weekly_retrain_wt.sh が毎週実行する
python -m src.cli.main train-poscal-wt

# 4. 予想生成（オッズフィルター付き）
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

### 2-6. 隊列位置バイアス補正（`src/preprocessing/position_calib.py`・2026-08-03 採用）

3着内モデル（`lgbm_wt`）は**隊列後方の選手を系統的に過大評価する**。後方の選手は
最終コーナーを外へ膨らんで回るため実走距離が伸びる、という物理的な理由による。

**距離損の実測**（1着の上がりタイム＝最終半周の `捲−差`。速度差は逆算に織り込み済み）:

| バンク | 車数 | 捲−差 | 推定余剰距離 | 内側からの幅 Δr |
|---|---|---|---|---|
| 400m | 7車 | −0.190秒 | 3.13m | 約1.0m（≒1車幅）|
| 333m | 7車 | −0.203秒 | 3.31m | 約1.1m |
| 500m | 7車 | −0.262秒 | 5.34m | 約1.7m |
| 500m | **9車** | **−0.371秒** | **8.01m** | **約2.6m** |

**車数が増えるほど不利が増す**（同一500mバンクで9車は7車の1.50倍）。

**補正の設計**: 偏りは「モデルが高く買った後方選手」に集中する選択効果のため、
セル一律の加算補正では順位が動かず効かない。`logit(pred_prob)` に加えてレース内相対の
**正規化位置**（`pc_pos_frac`）・競走得点差（`pc_rp_adv`）・**正規化予測順位**
（`pc_rank_frac`）・B取得確率（`pc_p_b`）を入力にした2段目モデルで組み替える。
位置と順位を車数で正規化してあるため、**9車で「後方」が4車以上になる等の境界差を
閾値のハードコードなしに吸収する**。

| モデル | 役割 | 学習 |
|---|---|---|
| `lgbm_wt_b` | B（最終バック先頭）取得確率。隊列推定に使う | 全期間 |
| `lgbm_wt_poscal` | 2段目。補正後の3着内確率を返す | 内側窓12ヶ月（ベースモデルが見ていない期間）|

### ⚠️ 適用は9車立てのみ（`POSCAL_MIN_ENTRIES = 8`）

全車数に適用した初版では軸2車的中が 7車 +0.56pt / 9車 +1.41pt（いずれも有意）だったが、
`wt_overlap_n`（軸2車とWT公式印◎◯の重なり）との整合を確認したところ、
**7車の改善は「軸が◎◯の側へ寄った＝overlapが動いた」ことで全て説明され、
軸の質そのものは改善していない**ことが判明した。

| 母集団を固定した比較 | 7車 | 9車 |
|---|---|---|
| overlapが変わらないレース | 54.2→54.2% (−0.01pt・**ns**) | 41.8→42.3% (+0.55pt・**有意**) |
| ランク対象(overlap≤1のまま) | 39.0→39.2% (+0.25pt・**ns**) | 29.2→31.7% (+2.50pt・**有意**) |
| ランク対象の母集団 | 4,346→4,285（純 −61件） | 477→459（純 −18件）|

7S/7A は `overlap∈{0,1}`（＝市場と不一致）だけを対象とするランクなので、
overlap が 1→2 へ動いたレースは**改善ではなく7Bへの再分類**にすぎない。
さらに 7S/7A は既に12件/日まで枯渇しており、見返りゼロで母集団を1.4%削ることになる。
よって**7車には適用しない**。

9車は overlap を固定しても改善が残る＝市場追従では説明できない実質的な改善がある。
車数が増えるほど外を回る距離が伸びる（500mバンクで9車は7車の1.50倍）という
物理的根拠とも整合する。

**効果**（本番コードでの再現・test 2025-10-01〜2026-08-03・9車 n=1,848）:

| 指標 | 補正前 → 補正後 |
|---|---|
| 軸2車ともに3着内（全体） | 39.7 → 41.1%（+1.41pt・有意）|
| 同（9S/9A対象に固定・n=360）| 29.2 → 31.7%（+2.50pt・有意）|
| Brier | 0.18308 → 0.18031 |
| 改善した月 | 9/11 |

**ROIは改善しない**（三連複5点 73.9→74.3%）。的中率が下がるほど配当が上がって
相殺されるため、狙いは軸2車の信頼度＝表示品質の向上にある。
⚠️ **軸1（単独1位）の3着内率・1着的中率はいずれも有意差なし**。
1着を決め打つ買い目（三連単1着固定）へ転用してはいけない。

```bash
python -m src.cli.main train-poscal-wt      # weekly_retrain_wt.sh が②''で毎週実行
```

⚠️ `weekly_retrain_wt.sh` では **③カット再計測より必ず前**に置くこと。
`recompute_upset_cuts_wt.py` は `_apply_pred_prob_wt` 経由で補正後の pred_prob 分布を
見るため、順序を入れ替えると旧確率のカット定数が新モデルに適用され波乱帯がずれる。

---

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
| 2026-08-03 | **隊列位置バイアス補正（2段目モデル `lgbm_wt_b` / `lgbm_wt_poscal`）を全ランクの軸選定に導入**: 3着内モデルが**隊列後方の選手を系統的に過大評価**していることを発見し、2段目モデルで補正した。原因は物理的なもので、後方の選手は最終コーナーを外へ膨らんで回るため実走距離が伸びる。上がりタイム（最終半周）の `捲−差` から逆算した余剰距離は 400mバンク7車で **3.1m（≒1車幅・0.19秒）**、500mバンク9車では **8.0m（≒2.6車幅・0.37秒）** で、**車数が増えるほど不利が増す**（同一500mバンクで9車は7車の1.50倍）。honest split（学習≤2025-09-30 / test 2025-10-01〜2026-08-03）で測った「実3着内率−モデル予測」は前方が −0.9〜+2.2% なのに対し後方は全予測順位帯で **−3.5〜−5.4%（9車は−6.4〜−10.9%）**。偏りが「モデルが高く買った後方選手」に集中する**選択効果**のため、セル一律の加算補正では順位がほとんど動かず無効（軸2車的中 +0.43pt・Spearman −0.0014）。`logit(pred_prob)` + レース内相対の**正規化位置**・競走得点差・予測順位を入力にした2段目モデルで組み替える方式を採用（位置と予測順位は車数で正規化してあるため、9車で「後方」が4車以上になる等の境界差を閾値のハードコードなしに吸収する）。**適用は9車立てのみ（`POSCAL_MIN_ENTRIES=8`）**。全車数に適用した初版では 7車 +0.56pt / 9車 +1.41pt（いずれも有意）だったが、`wt_overlap_n`（軸2車とWT公式印◎◯の重なり）との整合を確認したところ **7車の改善は「軸が◎◯側へ寄った＝overlapが動いた」ことで全て説明され、軸の質は改善していない**と判明した（overlapが変わらないレースに限ると 7車 54.2→54.2%・−0.01pt・ns / 9車 41.8→42.3%・+0.55pt・有意。ランク対象(overlap≤1のまま)でも 7車 39.0→39.2%・ns / 9車 29.2→31.7%・+2.50pt・有意）。7S/7A は overlap∈{0,1}＝市場と不一致のレースだけを対象とするため、overlap が 1→2 へ動くのは改善ではなく**7Bへの再分類**にすぎず、しかも既に12件/日まで枯渇している 7S/7A の母集団を 4,346→4,285（純-61件・-1.4%）と削ってしまう。よって7車には適用しない。**9車の実測**: 軸2車ともに3着内 39.7→41.1%（+1.41pt・95%CI [+0.22,+2.54]・有意）、9S/9A対象に固定しても 29.2→31.7%（+2.50pt・95%CI [+0.28,+4.44]・有意）、Brier 0.18308→0.18031、月次9/11で改善。**ROIは改善しない**（三連複5点 73.9→74.3%）。的中率が下がるほど配当が上がって相殺されるためで、狙いは軸2車の信頼度＝表示品質の向上にある。**軸1（単独1位）の3着内率と1着的中率はいずれも有意差なし**のため、1着を決め打つ買い目（三連単1着固定）へ転用してはいけない。前段の検証として「後方かつ競走得点差が小さい選手を軸候補から除外して次点を繰り上げる」ハードルールも測ったが、発動率30.5%まで上げても確認窓ROIは76.4→76.1%で不変・的中率はむしろ悪化（52.0→51.1%）で**不採用**（差し替え候補である予測3位の実3着内率51.5%が、除外対象52.7%とほぼ同じ＝同じ穴に落ちるため）。実装: `src/preprocessing/position_calib.py`（`add_position_features` / `apply_position_calibration`）・`src/models/poscal_trainer.py`（`train_position_calibrator`）・`src/cli/main.py`（`train-poscal-wt` コマンド・`wave-picks-wt` へ結線）・`src/evaluation/backtest_wt.py`（`_apply_pred_prob_wt` にも適用し本番と一致させる）・`scripts/weekly_retrain_wt.sh`（②''。**③カット再計測より必ず前**に置く。`recompute_upset_cuts_wt.py` は `_apply_pred_prob_wt` 経由で補正後の分布を見るため、順序を入れ替えると旧確率のカット定数が新モデルに適用され波乱帯がずれる）。テスト9件追加（`tests/test_position_calib.py`。7車で適用されないことの回帰テストを含む）・全475件通過。**未実施**: 過去 picks_history の再構築（新旧混在状態）。詳細は kiseki メモリ `keirin_position_disadvantage_verification_2026_08_03`。 |
| 2026-08-03 | **7B(`RANK_7B`)を新設（◎◯一致だが順序・相手で市場と不一致・三連複3点）**: モデルが WINTICKET 公式印へ収束した結果 `wt_overlap_n==2`（軸2車が◎◯と完全一致）が母集団の約8割を占めるようになり、これを全除外している 7S/7A の対象が **2024-01の39件/日 → 2025年以降12件/日** まで枯渇した（2026-08-02・08-03 は◎◯不一致レースを全件採用しても6件・5件。entropy/axis_sum をどれだけ緩めても1件も増えない状態）。ユーザー方針「単純な市場との不一致を避ける方針自体は正しい。ただし◎◯が一致してもある程度の配当を見込める・的中率が他より高い・相手を絞れる時は価値がある」を受け、honest 月次凍結vintage（2025-01〜2026-08・579日・36,791候補。`scripts/exp_7s7a_overlap2_conditional_value.py` / `_sweep.py` / `_disagreement.py`）で検証した。**否定的結果**: overlap2 は的中率56.3%（7S/7Aは36.9%）と高い一方 ROI 72.4%・的中中央値4.0倍・ガミ率60.2%で配当が消えており、entropy×相手集中度×買い目点数の120セルを掃引しても ROI は全セル 72〜76% で完全に平坦＝**「的中率が高い」と「配当が残る」は確率の集中度という同一軸の両端で同時には立たない**（高ROIの隠れ帯は存在しない）。**採用した帯**: 相手からWT△(ana)を除外すると的中中央値3.4→6.1倍・ガミ率42.4→10.8%・20倍超の的中175→435本へ回復し、さらに順序不一致（単勝モデルの最上位≠◎・overlap2の11.2%）を重ねると **ROI 78.7%（7S+7Aの78.3%と同水準）・5.58件/日・的中率25.2%・ガミ率8.1%・的中中央値6.8倍・20倍超64本・最高116.6倍**、年次2025:78.6%/2026:78.8%・四半期7期すべて68〜89%・月次ROI60%割れ0回/20ヶ月と安定。**ROIは改善しない（控除率75%の壁）**ため、件数・見せ場・ガミ率の改善が目的の増枠という位置づけ。実装: `src/strategy_wt.py`（`RANK_7B_*`・`rank_7b_order_disagree/select_legs/daily_select`）・`src/cli/main.py`（候補JSON `_s7b_candidates.json`）・`scripts/notify_prerace_wt.py`（`judge_rank_7b` は朝の候補ではなく**発走前の盤面から相手を再計算**）・`scripts/notify_results_wt.py`・`scripts/write_candidates_wt.py`・`scripts/backfill_7b_rank_wt.py` / `scripts/rebuild_7b_walkforward_pg.py`。netkeirin自動入稿は `RANK_CONFIGS` に配線したうえで `netkeirin_settings` に **enabled=false** を明示投入してある（`_is_enabled()` が fail-open のため、行が無いと自動でONになってしまう）。 |
| 2026-08-02 | **7SS(`RANK_7SS`)を全廃**: 2026-07-31に「高配当の的中頻度（見せ場）」を目的として導入した波乱軸選出ランク（モデル非依存・race_point×WT公式印×ライン構成で判定）を、live実績の蓄積により全廃した。picks_history 全期間 **n=16,298・的中31.6%・ROI 73.5%**、2026年の月次ROIは 94.4/61.0/56.3/61.1/69.3/70.2/60.3% と1月以外すべて70%以下で、控除率75%を一貫して下回っており有効な推奨として成立していなかった（導入時のTEST ROI 71.0%も既に控除率割れだった）。加えて軸の較正でも、7SSが意図的に軸へ据える「市場人気4位以下」帯だけが市場実測を下回る（全期間 -1.9pt / TEST窓 -5.9pt）ことが判明。S1全廃時の教訓に従い**候補生成（`src/cli/main.py`）・ライブ判定（`notify_prerace_wt.py`）・採点/DB追記（`notify_results_wt.py`）の3経路すべて**を停止し、単一正本 `CURRENT_PAPER_RANKS` から `ABOLISHED_PAPER_RANKS` へ移動（これにより `_query_stats`・`live_report_wt.RANKS`・`save_model_eval.PAPER_RANKS` が自動追随）。DBは picks_history 16,298行・model_evaluation 1行を削除、netkeirin_settings の7SS行は enabled=false（退避はMac側 `data/backup/`）。kiseki側も `_PAPER_RANK_LABELS` から除去しWeb表示・集計・入稿設定から削除（commit `f10d6d6`）。判定ロジック本体は将来の再設定に備え残置。副次的に `keirin_webhook.py::_MANUAL_ALLOWED_RANKS` がsubmit側と乖離し廃止済みランクを許可していた既存バグも是正。テスト447件全通過。 |
| 2026-07-31 | **Phase3-a: `ex_spurt_pct`/`ex_thrust_pct`をtrain/serve skewのためFEATURE_COLS_WTから除外（48→46特徴・下記エントリの続報）**: 直下のエントリで検出したtrain/serve skew（同一開催中に値が更新される・5.36%/2.39%のペアで変化）を受け、12ヶ月・約194,000サンプルでA/B測定（`scripts/exp_ab_leaky_ex_features.py`）を実施。arm A（48特徴・現行）vs arm B（46特徴・2特徴除外）でeval(3着内) AUC 0.7732→0.7731、win(1着) AUC 0.8233→0.8233と**AUCへの寄与は事実上ゼロ**（honest ROIも有意差なし: 7S 77.9%[68.8,88.1]→86.3%[76.3,96.8] / 7A 76.4%[69.2,84.0]→75.6%[68.9,82.9]・いずれも重複区間）。**除外の理由は「性能が上がるから」ではなく「予測に何も貢献していないのにリスクだけを抱えているから」**（ユーザー承認済み）。`src/preprocessing/feature_wt.py`の`FEATURE_COLS_WT`から2特徴を削除（SELECT・0-1正規化処理自体は分析用途・将来のpoint-in-time化のため残置）。`tests/test_sb_dyn_wt.py`/`tests/test_rp_trend_wt.py`の特徴量数アサーションを48→46に更新し、`tests/test_feature_prepare.py`に2特徴が`FEATURE_COLS_WT`へ再混入しないことを保証する回帰テストを新規追加（H2H特徴が一度採用→撤回された前例への対策）。既存の全モデル（`lgbm_wt`/月次凍結vintage等）は48特徴で学習済みのため本変更単体では特徴量数不一致になるが、モデル再学習はPMが別途実施する方針でありコード変更時点では意図的に未実施。 |
| 2026-07-31 | **`ex_spurt_pct`/`ex_thrust_pct`のtrain/serve skewを検出（対応は保留）→ 同日中に上記エントリで除外を決定**: 同一開催のDay1時点と最終日時点の値をn=221,551ペアで比較した結果、`ex_spurt_pct`は5.36%のペア（変化時平均+8.2pt）、`ex_thrust_pct`は2.39%のペア（変化時平均+22.9pt）で値が開催中に更新されていることを確認。一方`race_point`/`first_rate`/`second_rate`/`third_rate`/`s_count`/`b_count`は0.00〜0.14%とほぼ完全に開催単位で固定＝安全。`_get_collected_keys`（`src/scraper/pipeline_wt.py:168-183`）が結果確定済み（`finish_order>=1`）レースのみをスキップし未確定レースの再収集を続ける仕様と組み合わさると、学習データにそのレース自身の発走後の結果を反映した値が混入しうる（`sb_dyn`系特徴で過去に確認された自己参照汚染と同型の構造）。対応方針はA/B測定で影響を測ってから判断することがユーザー承認済みで、本記録時点ではFEATURE_COLS_WTからの除外・point-in-time化等は未実施（同日中に上記エントリで除外が確定）。 |
| 2026-07-31 | **`finish_order=0`の意味を精査（void判定3実装の食い違いを検出）**: 実データ検証で、事前確定の欠車（取消・除外）は`wt_entries`に行自体が作られず物理削除される（`src/scraper/pipeline_wt.py:236-249`・`src/scraper/winticket.py:274`）ため、DB上に残る`finish_order=0`行（9,528件・5,474レース）は事前欠車ではなく発走後の落車・失格・棄権（DNF）であることが判明（99.6%に`res_standing`/`res_back`の実測値あり・該当レースの98.5%は行数が`wt_races.n_entries`と一致）。本番`notify_results_wt.py::_void_by_dns`は「オッズ盤面掲載車」基準で正しく実装されていたが、`src/evaluation/void_rules.py`（→`backtest_wt.py`が参照）の旧docstringは「完走者（`finish_order>=1`）基準」と誤記しており、DNFを欠車＝返還扱いしていたためバックテストROIが本番の実損益より構造的に高く出ていた（kiseki側修正済み・詳細はCLAUDE.md「変更時チェックリスト」参照）。 |
| 2026-07-28 | **S9/7A/9A候補時点表示を追加**: 従来S1/S7のみ朝の候補生成直後に`picks_history`へプレースホルダ行を書き込み推奨ページへ候補表示していたが、S9・7A・9Aは未実装で発走15分前判定（オッズ取得後）まで行自体が存在せずページに現れなかった（ユーザー指摘。軸2車選定・波乱度・ゲート判定はいずれもオッズ非依存でモデル計算のみから確定するため技術的制約はない）。`scripts/write_candidates_wt.py::_write_paper_candidates()`にS9(`#9S9`)・7A(`#7A`)・9A(`#9A`)の書き込みブロックを追加。あわせてS7/S9の候補時点pred_comboを汎用プレースホルダー`"axis1=axis2-候補"`から実際の残り車番号リスト`"axis1=axis2-3,4,5,..."`（`_third_list()`）に変更（対象車数7/9は候補生成時点で確定しているため軸2車を除く残りの顔ぶれはオッズなしで一意に決まり、発走15分前判定後の最終表記と同じ形式で表示できる）。kiseki側`frontend/src/app/keirin/page.tsx`の`isPaperRank`にも`NINE_S9`/`SEVEN_7A`/`NINE_9A`を追加（候補行はprerace_gami未設定のためガミ判定チップを非表示にする対象に含める）。詳細はkiseki側メモリ`keirin_s9_7a_9a_candidate_display_2026_07_28`。 |
| 2026-07-28 | **H2H(頭対頭対戦成績)特徴を実装したが本番採用は撤回**: netkeirin「対戦表」に着想を得た特徴（`h2h_win_rate`/`h2h_n_total`/`h2h_net_norm`、`wt_entries.finish_order`の自前履歴からpoint-in-timeで再現・netkeirinスクレイピング不要）を`add_h2h_features_wt()`として実装し`FEATURE_COLS_WT`へ組み込み、全期間honest再学習・vintage 18モデル再学習込みでS1/S7/S9をhonest全期間walk-forward再検証したところ、S1(ROI 443.0%→363.5%・-79.5pt)・S9(412.8%→286.8%・-126.0pt)が悪化し、事前の単独検証で改善していたS7(401.1%→424.8%・+23.7pt)のみ改善という結果になった。的中率・AUCは3戦略とも一貫して改善していたが、エキゾチック券種特有の払戻の歪みでROIの符号が割れた。S1/S7/S9は共有モデル（`lgbm_wt`/`lgbm_wt_win`）を使うため部分採用はできず、ユーザー判断で全面撤回。本番モデル・vintage 18モデル・tail evalモデルは全て実装前の状態（48特徴）に再学習し直して復元済み。`add_h2h_features_wt()`自体はコードとして残すが`FEATURE_COLS_WT`には含めず未使用（picks_historyへの書き込みは一切発生していない・検証は全てdry-run）。詳細はkiseki側メモリ`keirin_netkeirin_h2h_feature_2026_07_28`。 |
| 2026-07-23 | **【重要】race_point特徴量の自己参照汚染を根本修正**: 2026-06-18のcommitで`wt_entries.race_point`（`score_rank`/`score_mean`/`score_std`/`score_z`という実モデル特徴量の入力）をAI予測確率(`pred_prob_pct`)で上書きする処理が`wave_picks_wt`内に混入し、`weekly_retrain_wt.sh`が汚染されたrace_pointを特徴量として取り込み続ける自己参照汚染が約5週間（2026-06-18〜07-23）放置されていた（2026-07-19導入の`pred_top3_pct`が既に同じ表示目的を汚染なく満たしており、この上書き自体が既に不要だった）。上書きコード削除・`race_point=0.0`（デビュー戦等未点数選手・全期間1,505R該当）をNaN扱いに修正・健全性チェック+自動リトライ（`scripts/check_race_point_sanity.py`・`daily_picks_wt.sh`組み込み）実装・汚染期間(2026-06-18〜07-23・36日)の生データ再取得・汚染モデル破棄（`lgbm_wt`/`lgbm_wt_win`/`lgbm_wt_eval`/`lgbm_wt_win_eval`と6/21以降の世代退避8世代）・全期間再学習（holdout AUC: top3系0.777〜0.779・win系0.821〜0.826でいずれも旧値と同等〜微改善）・S1/S4のtailウィンドウ(2026-04-13〜07-22)再構築（S1: 779R的中7.8%ROI208.4%／S4: 1001R的中37.7%ROI153.1%）まで完了。**教訓**: モデル特徴量として使う列に表示専用の値を上書きしてはならない。詳細はkiseki側メモリ`keirin_race_point_feature_leak_2026_07_23`。 |
| 2026-07-23 | **SS+観察サブランク新設**: S1で発見した軸級班denyフィルターがS4のSS/Sにも効くか検証（`exp_s4_axis_class_deny_analysis.py`・正規プロトコル）。Sは日次axis_sum上位10件の「枠付き相対選出」のため格上軸候補を除外すると別候補が繰り上がり実際には悪化する一方、SSは無制限採用（枠なし）のため単純な足切りで改善が残る（train+val ROI222.3%→351.6%・全期間237.1%→362.2%）。結論: Sには適用せず、SS内の軸格上非該当サブセットのみ新表示ランク**SS+**として観察（買い目・購入対象は変更しない表示分岐）。`strategy_wt.s4_gate_label()`に集約、`backfill_s4_rank_wt.py`等で過去分も再構築。kiseki側の表示対応（`RANK_STYLE`更新漏れの「非」バッジバグ修正・推奨ガイドのSS+カード追加とSS+/SS/S/S1表示順統一）も同日完了。 |
| 2026-07-22 | **S1軸級班denyフィルター追加**: 軸選手が各グレード内の最上位クラス（S1級・A1級）の場合を除外する条件を`s1w_gate`に追加。「格上」認定選手が軸だと市場も同じ判断をしており配当が低くなりやすいと判明。的中率は変えず母数を約半分に絞りROIを底上げ（143.3%→182.5%・5万円以上高配当は85.7%残存）。同日、`top3_gap`閾値を0.22→0.15へ復帰し軸単勝勝率≤50%ゲートも新設（本命決着＝低配当レースの除外）。過去分バックフィル完了（13,489R・ROI143.3%→182.5%）。副産物としてVPS PGの`wt_odds`欠落(2024〜2026-05分)を発見・2,332万件移植で解消。 |
| 2026-07-19 | **特徴追加: S/B展開特徴（4項目・b_rate_90/s_rate_90/fh_rel_90/fh_best_rate_90）**。レース単位のS/B取得・上がりタイム由来のローリング特徴（詳細は本ファイル2-1節参照）。A/B検証でΔAUC+0.011〜0.013・指数1位3着内率+0.9〜1.1pt。FEATURE_COLS_WT 44→48。DNF(`finish_order<1`)除外バグを発見・修正の上で採用。 |
| 2026-07-21 | **S2/S3を全廃・S4をSS/Sの2ランクへ再編（同日中の最終方針）**: S2/S3は直前まで厳選作業を続けていたが（下記エントリ参照）、対象レース数・的中率・期待値の観点でユーザーが継続困難と判断し全廃を決定。既存picks_history行（S2:1155件・S3:801件）は`scripts/archive_u_m_abolition_wt.py`（新設）で`picks_history_u_archive`/`picks_history_m_archive`へ退避。候補生成（`src/cli/main.py`のU/M候補ブロック）・Discord通知呼び出し（`scripts/notify_prerace_wt.py`の`_process_u_candidates`/`_process_m_candidates`呼び出し）を停止（関数定義自体は過去日再採点スクリプト互換のため残置）。`scripts/save_model_eval.py`の`PAPER_RANKS`からも除外。S1は現状維持（払戻の大きさが予想購入者へのアピール材料になるためブラッシュアップ方向を継続検討）。S4は「今後の予想データのベース」と位置づけ、直前に導入したWT◎◯重なり考慮ロジック（下記エントリ）の区分をそのまま表示ランクに昇格: 重なり0→**SS**、重なり1→**S**。`picks_history.gate_label`列（元はS3のOR gate内訳記録用）を流用しSS/Sを格納（新規カラム不要）。`scripts/backfill_s4_rank_wt.py`・`scripts/notify_prerace_wt.py`（`_insert_s4_pick`/`_build_s4_message`）を対応済み。Web/Discord/サマリー/グラフの表示側対応は別途 kiseki リポジトリで実施。 |
| 2026-07-21 | **S4の選出方式をWT◎◯重なり考慮版へ変更**: 軸2車がWINTICKET公式予想の◎◯（`prediction_mark`∈{1,2}）と重なるとROIが下がるのではというユーザー仮説を`exp_s4_wt_axis_overlap.py`で検証（honest全期間・四半期walk-forwardモデル）。日次Top10選出内で重なり数別に分解した結果、的中率はほぼ横ばい（33〜37%）なのにROIが重なり数に応じ単調悪化（重なり0=408.1%／重なり1=148.7%／重なり2=完全一致=**75.7%赤字**）と判明。ユーザー指示により選出方式を変更: 重なり0は該当があれば無条件で全件採用（上限なし）、重なり1はaxis_sum昇順で固定`S4_DAILY_TOP_N`=10件、重なり2・WTマーク欠損は除外。`s4_wt_overlap_n()`/`s4_daily_select()`（`src/strategy_wt.py`）新設、`src/cli/main.py`（候補生成でprediction_mark取得・重なり判定を追加）/`scripts/backfill_s4_rank_wt.py`（日次選出ロジック差し替え）対応。honest全期間再構築（`rebuild_s4_walkforward.py`）: 9,220R(10R/日)・ROI128.1% → **9,927R(10.77R/日)・ROI131.3%**に改善。新旧の検証手法（独立スクリプトとbackfillスクリプト）で完全一致する数値を確認済み。 |
| 2026-07-21 | **S2/S3の購入条件を厳選**: 「購入機会が減っても的中率を上げてROIを改善したい」というユーザー要望を受け実施。S3は`m_axis_gate`をgap12/ratioとの3way ORからwin_rank単独ゲートへ縮小（honest全期間再構築でgap12単独87.9%・ratio単独88.2%がいずれも赤字と判明したため）。あわせて買い目オッズ下限をS2と分離し`M_LEG_MIN_ODDS`=20倍（旧15倍）に新設。正規プロトコル: 検証158.2%→188.4%(531→272R)・テスト149.5%→175.7%(152→72R)。honest全期間（`rebuild_s3_walkforward.py`）: 95.9%(3114R)→**120.4%(801R)**に黒字転換。S2は`U_MTO_MIN`を4.3→4.5へ引き上げ。正規プロトコル(本番モデル`lgbm_wt_val25`): 検証97.9%→103.7%・テスト107.4%→114.9%。honest全期間（`rebuild_s2_walkforward.py`新設）: 4.3=81.6%(1251R)→4.5=84.8%(1155R)。**S2は厳選後も全期間honestでは損失圏内**（2024〜2025年前半が40-70%台で低迷し2025年Q3以降90-150%台に改善する時系列トレンドあり・8月末の採否判定では直近実績を重視）。両ランクとも2024-01-01〜の過去分をwalk-forwardモデルで再構築済み。 |
| 2026-07-19 | **S1新設計導入（本番稼働）**: win軸1着固定（1着専用モデル`lgbm_wt_win`のレース内1位）×3着内モデルで軸を除いた上位2頭(p1,p2)を相手に、top3_gap(p1-p2の3着内確率差)≥0.15（`S1W_TOP3_GAP_MIN`）で三連単2点流し（軸→p1→p2, 軸→p2→p1・目下限なし）。旧S1（7車三連複7PLUS_R）・新S1（6車三連単SIX_S1）はいずれも全廃されたが、この構造は未検証だった。正規プロトコル: 検証145.8%(n=9949)→テスト135.3%(n=2851・約28R/日)、閾値0.08〜0.20で単調改善。S2/S3との重複4.3%とほぼ独立。月次11/16・年次2025/2026年とも100%超（S2/S3の9/16より高い一貫性）。払戻分布は少数の高額配当に偏る（上位3件除外でROI99.2%）ためレース単位ROIのmean±2SDでは不合格だが、同基準はS2/S3も不合格＝三連系券種の構造的性質と確認済み。`s1w_select`/`s1w_gate`（`src/strategy_wt.py`）新設・`src/cli/main.py`/`scripts/notify_prerace_wt.py`/`scripts/notify_results_wt.py`/`scripts/write_candidates_wt.py`/`scripts/backfill_s1w_rank_wt.py`（新規）/`scripts/save_model_eval.py` 対応。内部rank `SEVEN_S1`・suffix `#7S1`。全期間(2024-01-01〜)バックフィル済み。ペーパートレードで運用開始。 |
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
