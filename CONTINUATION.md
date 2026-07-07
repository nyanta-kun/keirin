# セッション引継ぎメモ（最終更新 2026-07-07）

コンテキストリセット後にここから再開すること。

---

## ★現在の状態サマリ（2026-07-07 時点）

### ★★ 発走前判定の永続化 = 採点・表示の正本化（2026-07-07）

**背景**: SS/S昇格の通知済みレース 240R 中 139R (58%) が翌朝の採点で見送り誤記されていた
（例: 7/7 取手9R SS3点通知→DB見送り表示）。逆に prerace「条件不成立」レースを旧採点が
pg>=7 で購入扱いにする「幻の購入」も混在（例: 7/7 45_02 通知なしなのに bet 500 計上）。

**方針（ユーザー確定）**: 推奨/ガミ見送りは**発走15分前の判定から事後変更しない**。
Web表示・サマリー・Discord通知は全てこの判定に揃える。

**実装（コミット 7eea4b1 ほか）**:
- `notify_prerace_wt.py`: 判定を `data/prerace_decisions_{date}.json` に確定記録
  （decision buy/skip・rank・カット後買い目 thirds・レグ別発走前オッズ・競走得点統計）。
  S/SS とも prerace_gami を購入レグ最安値で上書き・pred_combo/n_combos も即時反映。
- `notify_results_wt.py`: decisions を最優先で採点（txt ランク・最終オッズで上書きしない）。
  txt に載らない SS 昇格（gap12 0.07-0.10 候補）も採点対象に注入。
- kiseki 側 (コミット 9aa8c3a): summary/stats の gami 再フィルタ撤廃 → `bet_amount>0` 基準。
- バックフィル: `backfill_prerace_decisions.py` で 6/27〜7/7 の判定を prerace.log +
  candidates + 最終オッズ近似で復元し全日再採点済み。誤記 139→0（残 4 件は欠車返還で正しい）。

**注意**: 過去日の decisions json は `backfilled: true`・レグは最終オッズ近似（44%が approx）。
7/8 以降は prerace 実測値で記録される。

### gami 閾値検証（2026-07-07・詳細は kiseki メモリ keirin_gami_relax_verification）
- 見送り帯[3,7)は歴史12ヶ月 ROI 0.80 / live 0.76 で回収不能 → **7.0 維持**
- 7月窓 (7/1-7/7) S束 prerace実測: thr7 0.627 / thr6 0.538 / thr5 0.820（全て赤字・小n）
- 復活買い（直前オッズで拾い直し）も live 0.60-0.75 / 歴史 0.79 で全滅

### 競走得点構造（2026-07-07 検証・decisions json に score_sd/score_gap2r 記録開始）
- 7車 22,236R: 得点SD・上位2と残りの格差が大きいほど上位堅い（top1 3着内 67→79%）
  ・下位好走率低下（70→58%）・trio配当低下（中央値 1210→680円）。得点mean は逆相関（下級戦ほど堅い）
- 現行買い(min>=7) で sd>=Q1 を残すと ROI 2.87→3.0-3.6（両半期頑健）。除外帯も上限値 1.5-1.7 で
  黒字傾向のため**即除外はせず live 蓄積後に判断**

---

## ★現在の状態サマリ（2026-06-18 時点）

### ★ バグ修正完了（2026-06-18）

| バグ | 原因 | 修正 |
|---|---|---|
| **miwokuri=FALSE バグ** | `get_connection()` が `KEIRIN_DB_URL` 設定時に PG へ直接書き込み → 後続の migrate がSQLite（miwokuri=TRUE のまま）で上書き | `notify_results_wt.py` `main()` 冒頭で `KEIRIN_DB_URL` を pop し `get_connection()` を常に SQLite に向ける。`_sync_vps(db_url)` を明示引数化 |
| **PGデータ不正（#CAND miwokuri=FALSE）** | 上記バグで 2026-06-17(50件)・2026-06-18(35件) の #CAND が miwokuri=FALSE | PG: `UPDATE keirin.picks_history SET miwokuri=TRUE WHERE race_key LIKE '%#CAND'` で修正済み |
| **Discord レース数水増し** | `_query_stats` / `_query_stats_rank` が miwokuri フィルタなし → 見送り行がCOUNT(\*)に混入 | `AND NOT COALESCE(miwokuri, 0)` を両関数に追加 |

**ROI計算バグ調査結果**: kiseki WEB `_aggregate()`・`backtest_wt.py`・`save_model_eval.py` 全箇所で ROI = `SUM(payout WHERE hit=1) / SUM(bet_amount ALL)` と正しく実装されていることを確認。「的中レースの投資のみで割る」バグは現時点の全コードには存在しない。

---

## ★現在の状態サマリ（2026-06-17 時点）

### ★★ モデル設計刷新（2026-06-17）

**3分割・汚染なし設計**に移行済み:

| モデル | 学習期間 | 役割 | AUC | R数 |
|--------|----------|------|-----|-----|
| `lgbm_wt` | TRAIN+VAL 2022-12-01〜2026-02-28 | **live本番予想 + HOLD評価** | 0.7717 | 88,769R |
| `lgbm_wt_train_only` | TRAINのみ 2022-12-01〜2025-06-30 | **VAL評価専用** | 0.7774 | 70,540R |
| `lgbm_wt_v2`（退避） | 2023-07〜2026-06-14 | HOLD汚染あり・参照のみ | 0.7747 | — |

**正しいバックテスト結果（汚染なし）**:
- **VAL（lgbm_wt_train_only）**: 139.6%★ 6,866R ｜ SS 158.8% / S 132.9% / A 100.7%
- **HOLD（lgbm_wt）**: 134.3%★ 3,076R ｜ SS 137.8% / S 138.8% / A 99.4%

HOLD月別（lgbm_wt）: 2026-03 115.5%★ / 04 138.0%★ / 05 146.1%★ / 06(1-16) 139.0%★  
→ **全12ヶ月黒字（VAL8ヶ月+HOLD4ヶ月）**

更新ファイル: `scripts/save_model_eval.py`（VAL_MODEL_NAME/HOLD_MODEL_NAME 分離）/ `docs/prediction-factors.md` / `docs/bet-structure-guide.md` / kiseki PR#39

**6車立て以下は完全に未使用**（7+車専用戦略のみ本番稼働）

### 実運用
- **live picks (7+車)**: 2026-06-16〜 開始（SS/S/A 3ランク体制・初回判断目安 2026-06-23頃）
  - **7+車 SSランク**: gap12≥0.07 + ガミ目カット後残り1〜3目 → HOLD 137.8%★（lgbm_wt）
  - **S/Aランク**: gami≥5倍全目通過 + gap12≥0.10(S) / [0.07,0.10)(A)
  - 6月1-16日live実績: SS 240R/84.4% / S 162R/87.9% / A 47R/67.9%（449R・16日間・分散大）
  - **2026-06-17**: 0件推奨（日中32件・夜18件がgami条件不成立）
- **fav_mismatch タグ**: 1R のみ記録（2026-06-11〜）。バックテスト根拠否定済み
- **money-flow snapshot cron**: ユーザーの Terminal 適用待ち（`data/cron_proposal_moneyflow_20260613.txt`）

### 2026-06-17 実装完了（SSランク体制）

| 変更 | 内容 |
|---|---|
| `src/cli/main.py` | 7+車を SS/S/A 3ランクに分割。combo_odds_map 導入（per-combo オッズ評価）。candidates.json（gap12のみ）出力追加 |
| `scripts/notify_picks.py` | SS/S/A 別セクション・ヘッダー。`🚲` アイコン |
| `scripts/notify_results_wt.py` | `7PLUS_SS`/`7PLUS_S`/`7PLUS_A` 別集計対応 |
| `scripts/notify_prerace_wt.py` | candidates.json 優先読み込み・`_determine_live_rank()` でリアルタイム rank 判定・合成オッズ表示 |
| `docs/analysis/49-7plus-rank-rules.md` | SS/S/A ランクルール・検証結果まとめ |
| kiseki frontend | `RANK_STYLE` に 7PLUS_SS/7S/7A 追加・HelpSection（ルール説明）追加 |

### G41-G44 + 手法実験フェーズ（2026-06-16 完了）

| Goal | 内容 | 結果 |
|---|---|---|
| **G41** EXデータ未使用3列 | `ex_left_behind_pct` / `ex_split_line_pct` / `ex_snatch_pct` | **Phase1 不通過** AUC +0.0001〜+0.0002（閾値+0.001未達）|
| **G42** WINTICKET EXデータ拡張調査 | JSON全キー調査・25フィールド新発見 | 評価保留→G44で実装 |
| **G43** keirin.jp 身体測定 | 体重・背筋力・肺活量・太もも・胸囲 | **Phase1 不通過**（AUC ±0.0000・全2,719選手スクレイプ済み）|
| **G44** WINTICKET 条件別成績 | 天候/バンク/時間帯/位置別成績 | **Phase1 通過★・Phase2 不通過**（AUC +0.003台・ROI<100%・市場効率の壁）|
| **LambdaRank** | LGBMRanker（binary/ordinal）vs 二値分類 | **Phase1 不通過** AUC -0.014〜-0.023（大幅悪化）→ クローズ |
| **条件別ROIスキャン** | n_entries×grade×gap12×bank_length 体系スキャン | **★15セル通過**（doc45）→ gap12>0.10・S級が最有力 |

#### 条件別ROIスキャン 主要結果（2026-06-16・doc45）
- **全体ベースライン**: TRAIN 211.1% / VAL 132.9% / HOLD 174.4%（363/83/32R）
  - doc18（70-90%）との差: 週次再学習リーク③を除去したTRAIN-onlyモデルの汎化性能
- **最有力条件 gap12>0.10**: VAL 139.9%(52R) / HOLD 220.7%(20R) ← サンプル最大・最安定
- **S級**: VAL 201.7%(35R) / HOLD 151.0%(14R)
- **400m×S級**: VAL 117.4%(24R) / HOLD 193.6%(11R)
- **n6×gap12>0.07**: VAL 122.7%(56R) / HOLD 186.0%(21R) ← 最大サンプル
- 注意: DNS生存バイアス残存（欠車除外→結果やや楽観的・ただし競輪の欠車は払い戻しのため影響限定的）
- ハーネス: `scripts/exp_conditional_split_wt.py`

#### G42 発見フィールド（要評価）
- `exCompete`（競りの勝率）: Coverage 7.8%（疎い）
- `weatherSunny/Cloudy/Rainy`（天候別成績）: ~100% Coverage
- `trackDistance333/400/500`（バンク周長別成績）: ~98% Coverage
- `hourTypeNormal/Morning/Night`（時間帯別成績）: 96-100% Coverage
- `linePositionFirst/Second/Third`（位置別成績）: 32-81% Coverage
→ 全評価には wt_entries の TRAIN 期間 (~2023-07〜) フルリフェッチが必要（3h+）

### doc51: 三連複 ライン軸2 + 指数下位除外（2026-06-16） → **全不採用**

| 戦略 | VAL | HOLD | 判定 |
|---|---|---|---|
| S0 現行3点 | 73.3% | 82.9% | 基準 |
| L1: 同一ライン指数2位を軸2 + 下位1除外 | 70.7% | 91.2% | **不採用** |
| L2: 同一ライン指数2位を軸2 + 下位2除外 | 86.3% | 80.7% | **不採用** |
| R1: AI軸2固定 + 下位1除外 | 62.3% | 77.1% | **不採用** |
| R2: AI軸2固定 + 下位2除外 | 51.8% | 92.8% | **不採用** |

**核心**: 軸2変更は64%のレースで発動するが、その際 S0 ROI 92.4% → L1 83.5% と悪化。
AI確率2位はモデルが全特徴量を統合した最良の2番手予測であり、ライン内指数2位より優秀。
指数下位除外も doc50 の P2 系と同型で VAL 51-62%（現行 73% を大幅に下回る）。
→ **現行 S0（AI確率順3点固定）を維持。変更に意義なし。**
- ハーネス: `scripts/exp_line_axis2_wt.py` / 詳細: `docs/analysis/51-line-axis2-trim.md`
- データ資産: `wt_entries.line_group`（NULL率0%・全期間カバー）が存在確認済み

### doc49: 三連単フォーメーション条件別ROI（2026-06-16）

| 戦略 | TRAIN | VAL | HOLD | 判定 |
|---|---|---|---|---|
| S0 三連複 現行 | 83.6% | 67.7% | 87.5% | 不通過 |
| **T7: 三連単P1→P2→{P3,P4} gap12≥0.10&gap23≥0.05 2点** | 51.8% | **112.6%★** | **209.2%★** | **Phase2 ボーダー** |
| T4: 三連単P1→P2→{P3,P4} gap12≥0.10 2点 | 59.6% | 70.9% | 147.6%★ | 不通過(VAL) |
| その他 T1-T6・T8 | — | <100% | — | 不通過 |

**T7 の注意**: VAL n=19（閾値 30R 未達）/ HOLD n=12 / 約0.3R/日（稀レース）
- **着順精度**: gap12≥0.15帯は pred2 の2着率が 16.1%（固定不可）/ gap12 0.10-0.12帯が p2_2nd=33.3%でバランス最良
- **gap23 鍵**: gap23 高位(≥0.082)で P1=1着&P2=2着 = 20.0%（最高）
- **特別推奨案**: 条件=≤6車・ガミ≥5倍・gap12≥0.10・gap23≥0.05 → pred1→pred2→{pred3,pred4} 2点
- 詳細: `docs/analysis/49-trifecta-formation.md` / ハーネス: `scripts/exp_trifecta_formation_wt.py`

### doc50: 三連複 買い目削減 O10戦略（2026-06-16）

**逆説的発見**: pred5（最低確率の3着目）が ROI 111.4%★ で最高・pred3 は 64.7%（市場の過小評価）

| 戦略 | VAL | HOLD | 判定 |
|---|---|---|---|
| S0 現行3点 | 73.3%(76R) | 82.9%(30R) | 不通過 |
| P2 pred3+pred4削減 | **36.7%** | 113.0%★ | **不通過(VAL悪化)** |
| **O10 各目≥10倍のみ** | **112.2%★(58R)** | **100.0%★(19R)** | **Phase2 通過** |
| O15 各目≥15倍 | 200.2%★(34R) | **0.0%(11R)** | 不通過(HOLD崩壊) |

**O10実装**: `wave-picks-wt --min-combo-odds 10.0`（個別コンボ単位オッズ足切り・live検証用）
- 76R→58R（-24%）・3.0点→1.4点（-53%）・コスト約50%削減
- **HOLD n=19 は小サンプル → live 50R 蓄積後に初回判断**
- ハーネス: `scripts/exp_trio_trim_wt.py` / 詳細: `docs/analysis/50-trio-trim.md`

### 次のアクション（優先順）
1. **live実測の継続観察**: `scripts/live_report_wt.py` で随時確認。
   - **7+車 SS/S/A**: 初回判断目安 2026-06-23頃（12.9R/日 × 7日 ≈ 90R）。`picks_history` で `rank IN ('7PLUS_SS','7PLUS_S','7PLUS_A')` を確認（6車以下は未使用）
2. **★★ 7+車 SSランク live検証（2026-06-17 実装済み）**: candidates.json で prerace cron が gap12候補全件を監視・live オッズで SS/S/A を再判定。詳細: `docs/analysis/49-7plus-rank-rules.md`
3. **live成績をグレード別にトラッキング**: `picks_history` で S級/A級 を分けて観察
4. **★T7 三連単 特別推奨 live 追跡**: gap12≥0.10 & gap23≥0.05 の 2点フォーメーション（約0.3R/日）。50-60R蓄積後（約5〜6ヶ月）に初回判断。実装済みスクリプト: `exp_trifecta_formation_wt.py` 参照
5. **★O10 個別コンボ足切り live 検証**: `--min-combo-odds 10.0` を追加して手動テスト可能。50R蓄積後に初回判断（約3〜4週間）。`picks_history` で `combo_odds_trim` タグで別集計予定
6. **money-flow cron 適用**: Terminal から `crontab -e` で登録。≥1,624R 蓄積後（約9ヶ月）に `exp_moneyflow_wt.py --report`
7. **中間オッズ帯フィルタのlive検証**: 朝オッズデータが数十R蓄積後に `snapshot_morning_odds_wt.py --report` で確認
8. **JKA師匠情報実験（低優先・任意）**: `scrape_mentors_wt.py`（約22分）→ `exp_mentor_feature_wt.py`。Phase1 不通過予測。doc40 に結果記入

---

## ★★ 既存DB特徴量候補の検証結果（2026-06-15 全完了）

以下の候補を全て実験済み。**公開情報内では残る即着手アイデアはほぼ枯渇**。

| 候補 | 結果 | 詳細 |
|---|---|---|
| **朝→夕方 intraday drift** | 方法論確立・C0対象4R/5日（≈12ヶ月待ち） | `doc36` / `exp_evening_morning_drift_wt.py` |
| **venue×grade rolling WR** | Phase1 不通過（AUC +0.0001） | `doc37` / `exp_venue_grade_wr_wt.py` |
| **ライン連携コヒージョン** | Phase1 不通過（AUC +0.0001・疎すぎ） | `doc38` / `exp_line_cohesion_wt.py` |
| **S-model ハイパーパラメータ** | 全4セット HOLD 88〜94%（15R・ノイズ範囲内）・採用見送り | `doc39` / `exp_s_hyperparam_wt.py` |

**結論**: 既存 DB の公開情報内では Phase1 を突破できる新特徴量がない。
市場効率の壁を超えるには **新情報源（非公開・未価格化データ）** が必要。

---

## ★ 残る候補（WEBスクレイピング）

| 候補 | 内容 | 期待度 | 状況 |
|---|---|---|---|
| **試走タイム** | 選手の当日練習ラップタイム | 最高 | 調査済: 公開限定・難易度中〜高・推奨度低 |
| **JKA 師匠情報** | keirin.jp 選手詳細の師匠リンク | 弱 | **実装済み**（スクレイパー+ハーネス待機中） |
| **コンピューター指数** | netkeirin 等の独自AI指数 | 低 | 調査済: market redundancy リスク高→スキップ推奨 |

### JKA 師匠情報（doc40・2026-06-15実装）
- **URL**: `https://keirin.jp/pc/racerprofile?snum={player_id:06d}`（静的HTML・robots.txt ALLOW）
- **特徴量**: `mentor_in_race`（師匠が同一レースに出走）/ `is_mentor_of_someone`
- **事前評価**: 弱（同期≠同ライン実績・Coverage ~40-60%・doc38 ライン連携と類似の疎シグナル）
- **手順**:
  1. `python3 scripts/scrape_mentors_wt.py` → `data/player_mentors.csv`（約22分）
  2. `python3 scripts/exp_mentor_feature_wt.py` → Phase1 AUC 確認
- 詳細: `docs/analysis/40-mentor-feature.md`

### コンピューター指数（スキップ推奨）
- netkeirin 等の独自AI指数は静的HTML取得可能だが **market redundancy** が高い
- prediction_mark（既存特徴量）と同質の情報→モデルに追加しても AUC 改善なし（doc17 Web予想ロジック監査と同型）
- スクレイプコスト対比で期待値が低いため保留

### 試走タイム（低優先度）
- **WINTICKET**: 出走表に試走タイムなし（確認済）
- **netkeirin**: データ保有の可能性あるが動的ロード
- **各競輪場公式サイト**: 掲載なし
- **難易度**: 中〜高 / **推奨度**: 低（公開が限定的）

---

## ★ 他式別オッズ特徴量 & グレード別モデル実験（2026-06-15）→ **保留**（doc35）

### 他式別オッズ（二連単・ワイド・二連複）を特徴量化
- AUC: +0.022（全特徴量中1〜3位）
- ROI: VAL 悪化・選択レース数が半減 → Phase2 不通過
- 結論: 5式別市場は統合されており裁定不可。市場特徴量を加えるとモデルが市場追随になりガミ≥5倍レースが消える。

### グレード別モデル（S级 × S-model）
- S级専用モデル: TRAIN 150%★ / VAL 107%★ / HOLD 88%（15R）
- 全体（S→S-model, A→A-model）: TRAIN 113%★ / VAL 97.3% / HOLD 107%★
- Phase2 不通過（VAL 97.3%・閾値 100% 未達）。これまでの実験で最高に近い結果。
- 今後: live成績をグレード別にトラッキングし、100R蓄積後に S级専用モデルの採否を判断

### 副産物: grade_enc バグ修正
- `feature_wt.py` の grade_map が ks 形式（GP/G1...）のまま wt では全件 fillna=1 だった
- 修正済み（S級→3 / A級→2 / L級→1）。AUC への影響は +0.0001 で実質軽微。

詳細: `docs/analysis/35-crossmarket-grade-model.md`

---

## ★★fav_mismatch リーク無し単独検証（2026-06-15）→ **バイアス崩壊確認**（doc34）

doc13 の「1168%/576% = 最強新レバー」はリーク無しで：
- fav_mismatch=True: TRAIN 79.7%, VAL 95.4%, HOLD **23.1%** → 全期間100%未達
- fav_mismatch=False: TRAIN 88.9%, VAL 55.4%, HOLD **100.2%** → 逆転（20R・小標本）

3バイアスによる 6〜25倍の過大評価。「最強新レバー」の前提は取り消し。
live タグ蓄積は継続（採否は picks_history ≥100R 後）。
詳細: `docs/analysis/34-fav-mismatch-leakfree.md`。ハーネス: `scripts/exp_fav_mismatch_leakfree_wt.py`。

---

## ★★軸精度・三連複 vs 三連単 条件付き戦略検証（2026-06-15）→ **全戦略不通過**（doc33）

**最重要発見: gap12<0.06（拮抗帯）の pred1 3着以内率は 37.9%（VAL+HOLD）**
三連単（S1/S2）は全帯で三連複より劣る（S2 VAL 32.5%/HOLD 39.8%）。
pred1単軸流し（S4）は 0.06-0.10 帯で 103.2%★（S0 の 113.9% より低い）。
gap12 0.06-0.10 帯のみが唯一 >100% だが 16R の小標本。
**doc10（1-2着BOX★頑健）はバイアス込み数字でリーク無しでは不成立確認 → ロードマップ#7クローズ済み**。
詳細: `docs/analysis/33-axis-accuracy-trifecta.md`。ハーネス: `scripts/exp_axis_trifecta_wt.py`。

---

## ★ライン構造ベース買い目設計検証（2026-06-15）→ **Phase2不通過**（doc32）

拮抗レース（gap12<0.06）でライン構造から軸選択する2×2フレームワーク。
S0(現行)/S1(強番手軸)/S2(拮抗×最強ライン)/S3(拮抗×強番手) 全て全期間ROI<100%。
軸変更も市場効率の壁を超えられないことを確認。
詳細: `docs/analysis/32-line-bet-design.md`。ハーネス: `scripts/exp_line_bet_design_wt.py`。

---

## ★ライン先頭強度・ライン内得点差 特徴量検証（2026-06-14）→ **Phase1通過・Phase2不通過・不採用**（doc31）

leader_rp_gap_vs_best / within_line_rp_gap の3特徴量。Phase1: AUC差+0.0024〜+0.0032（通過）。
Phase2: base 65〜84% vs line 71〜75%（全期間<100%・TRAIN/VAL ではbase下回る）。市場効率の壁を再確認。
詳細: `docs/analysis/31-line-features-leader-gap.md`。ハーネス: `scripts/exp_line_features_wt.py`。

---

## ★波乱予想フェーズ（2026-06-14）→ クローズ（W01〜W04全不通過）

W01（波乱モデルAUC 0.57・ランダム）・W02（4フォーメーション全滅）・W03（upset×fav_mismatch交差=構造的非重複）・W04（総合まとめ）全て不通過。
現行tier選別（gap12大）は隠れた波乱回避フィルターとして機能しており、波乱モデル追加の意義なし。
詳細: `docs/analysis/30-upset-synthesis.md`。

---

## ★★★2026-06-13 ROI100%再挑戦フェーズ（G01〜G08）完了

| Goal | 結論 |
|---|---|
| G01 backtest リーク無し化 | `backtest_wt.py` 本体に3バイアス修正・`void_rules.py` 新設 |
| G02 live実測レポート | `scripts/live_report_wt.py` 新規作成（SS+S+A 10R・ROI 65%・判断100R以上必要） |
| G03 日中オッズスナップショット | `scripts/snapshot_intraday_odds_wt.py`。cron提案=`data/cron_proposal_moneyflow_20260613.txt` |
| G04 money-flow検証ハーネス | `scripts/exp_moneyflow_wt.py`。30R≈2%・評価不能・≥1,624R蓄積後に再実行 |
| G05 気象データ収集 | `src/scraper/weather.py`。全43場・wt_weather 133万行・カバレッジ99.9% |
| G06 風特徴検証 | Phase1 不通過・無情報。`docs/analysis/24-wind-feature.md` |
| G07 高配当融合 | ゲート条件未充足でSKIP。`docs/analysis/25-highpay-fusion.md` |
| G08 ドキュメント同期 | 完了 |

---

## リーク無し再検証で結論が変わったもの（要注意）

| 項目 | 旧結論（バイアス込み） | 新結論（リーク無し） |
|---|---|---|
| **fav_mismatch** | 1168%/576%（最強新レバー） | HOLD 23.1%（バイアス産物） |
| **doc10 1-2着BOX三連単** | ★頑健 >100% | VAL 32.5%/HOLD 39.8%（不成立） |
| **全レバー全般** | 70-90%超多数 | 全て ~70-90%（doc18基準） |

**採否判断は live 実測のみ。backtest は最終オッズ上限値。**

---

## ★★★★2026-06-12(続々) リーク無し再採点（doc18・最重要）

バックテスト3バイアス発見：①欠車生存バイアス×stale odds ②≤6車完走者基準（7車混入33%）③週次再学習リーク。
本番忠実+リーク無しでは現行戦略含む全レバー3期間 ~70-90%。
採否判断は live 実測のみ。標準ハーネス=`exp_leakfree_rescore_wt.py`。
