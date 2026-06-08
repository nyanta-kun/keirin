# コードレビュー統合レポート（2026-06-08）

対象: winticket(wt) ルート本番稼働コードベース。4ロール（シニアエンジニア／データサイエンティスト／AI・MLエンジニア／テスト・QA）の構造化所見を統合し、PM（テックリード兼任）が実コードで裏取り・誤りを是正したうえでの判定レポート。

レビュー範囲: 直近セッションの主要変更（DNSバグ修正・ガミ回避3段階・波乱ゲート・朝オッズ退避・L級クラスマッピング・カット週次再計測・成績採点）。read-only レビュー（コード未変更）。

---

## 1. 総合判定

**sound-with-issues（実運用に足る妥当性はあるが、重要な是正項目あり）**

根拠:
- 本番予測経路の中核（DNS=着外の `between(1,3)` 一貫適用、SS/S/A 判定とオッズ市場の一貫性、ガミ3段階・Bランク除外、波乱ゲートの単調性ガード）は、4ロールすべてが「概ね正しい」と判断し、PM の実コード裏取りでも**本番挙動を壊す致命的バグは検出されなかった**。
- ローリング特徴のリーク有無は3ロールが実検証し**明確な未来漏洩なし**で一致（唯一 `venue_wr` の同日複数走に理論上の日内リーク余地があるが影響レース極小）。
- 一方で、(A) 本番モデル `lgbm_wt` が「直近90日を除外したホールドアウト評価用モデル」で上書きされ続けている運用上の設計欠陥、(B) 自動テストが皆無で回帰検知できない、(C)「バックテストROI=上限値」フレーミングと `picks_history`(=最終オッズ採点) の不整合、という**信頼性・再現性に関わる重要課題**が複数ロール横断で指摘された。これらは収支を直接壊すものではないが、運用判断・OOS主張・訴求の根拠を毀損しうる。

---

## 2. 重大度別 指摘一覧（統合・PM裏取り済）

### CRITICAL
なし（本番予測の正誤を直接壊す確定バグは未検出）。

### HIGH

#### H-1. 本番モデル `lgbm_wt` が「直近90日を除外したホールドアウトモデル」で毎週上書きされている
- ロール: AI・MLエンジニア（high）／ データサイエンティスト（high・別解釈）
- 該当: `scripts/weekly_retrain_wt.sh:18-19`, `src/cli/main.py:993-994,1012-1015`
- 事実（PM裏取り済）:
  - weekly cron は `train-wt --from 2023-07-01 --test-from <90日前> --save-as lgbm_wt_v1` を実行。
  - `main.py:994` で `df_tr = df_train[race_date < test_from]`（直近90日を学習から除外）。
  - `main.py:1014-1015` は `save_as != "lgbm_wt"` のとき**無条件で `lgbm_wt` にもコピー**するため、ホールドアウト評価モデルがそのまま本番名で配信される。
  - `data/models/upset_cuts_wt.json` は `train_to=2026-03-01` を記録 → 本番モデルが test 期間を学習に含んでいないことと整合。
- **PMによるDS高所見の是正**: DSは「TRAIN/TEST AUC が近接 → test混入＝in-sample評価の疑い」とした(確信度中)。PM が実モデルで再計算した結果 **TRAIN(<2026-03)=0.7832 / TEST(>=2026-03)=0.7741（差 約0.009）**。test がリークしていれば TEST AUC は TRAIN に接近・超過し*上振れ*するはずで、観測された「TRAIN がわずかに上」のパターンは**正しいホールドアウトと整合**する。よって「analysis/exp の OOS 主張が in-sample で崩れる」という DS の懸念は**概ね棄却**でき、analysis 評価は真に OOS とみなせる。ただし裏返しとして「本番に配信されているのは全データ学習モデルではなく90日古い打切りモデル」という AI 所見が正しく、データ浪費・最新調子の未反映という実害が残る。
- 推奨対応:
  1. 評価（holdout）と本番 artifact を分離する。weekly では holdout 評価後に `--test-from` 無しで全データ再学習して `lgbm_wt` を生成・保存する。
  2. `pkl` に `train_from/train_to/学習コミット/AUC` のメタデータを埋め込み再現性を担保。
  3. 併せて世代退避（H-2 と統合対応）。

#### H-2. 自動テストが皆無（`tests/` 空・pytest 未使用）。新規ロジックが全て手動検証依存
- ロール: テスト・QAエンジニア（high）
- 該当: `tests/`（PM確認: ファイル0件）, `requirements.txt`（pytest==8.1.0 が宣言のみ・未使用）。PM が `grep -rl "import pytest|def test_"` で**該当ファイル0**を確認。
- 影響: 週次 recompute がカット定数を、週次 retrain がモデル確率分布を毎回書き換えるため、ガミ閾値・Bランク振り分け・波乱ゲートの挙動は**静かにドリフトしうる**。境界値（オッズ=3.0/5.0 ちょうど、top3_sum=1.70/1.90/2.08 ちょうど）、欠車、出走3車未満、結果未確定、オッズ全欠損などのエッジは現状コードレビューのみで担保。
- 推奨対応（DB/モデル不要な純粋関数から優先）:
  - `strategy_wt.upset_tier / passes_upset_gate / _load_cuts`（境界ちょうど・JSON不在/壊れ/非単調 `[2.0,1.9,2.1]`/等値 `c0==c1` で既定値復帰）
  - `notify_results_wt._parse_combo / _parse_picks_full`（Bランク行除外・`→`区切りSS・thirds>3 切詰め）
  - ガミ3段階振り分け（min_leg=2.99/3.0/4.99/5.0/5.01、known空→None素通り）
  - as-of ローリングのリークテスト（合成DBで未来レースが特徴に混入しないこと）

#### H-3. `picks_history` も最終オッズで採点され「実測ROI」ではない（上限値フレーミングの不整合）
- ロール: データサイエンティスト（high）
- 該当: `scripts/notify_results_wt.py`（`_load_payouts_wt`＝wt_odds最終オッズで payout 算出・保存）, `src/strategy_wt.py:16`（「実測は picks_history で検証」とのコメント）
- 事実: `wt_odds` は `INSERT OR REPLACE` で確定（締切直前）オッズに上書きされる。`picks_history` の payout もそれを参照するため、バックテストと**同源の上限値**。朝-確定ズレ・実販売下振れを反映した真の実現ROIではない。実運用の現実アンカーは旧ルートの実測49%/1週間のみ。
- 推奨対応: `picks_history` を「朝オッズ(`wt_odds_snapshot`)ベース」または「実確定払戻の手入力」へ変更し、上限値(最終)と実現値を**別カラムで両建て保存**。strategy_wt のコメントと analysis docs の「実測検証」表現を「現状は最終オッズ採点＝上限値」に修正する。

#### H-4. 小標本層（SS 等）の点推定ROIが単発万車券に極端依存
- ロール: データサイエンティスト（high）
- 該当: `src/evaluation/backtest_wt.py:321-333`（SS=trifecta）, `docs/analysis/02-upset-prediction.md:109-135`
- 事実: test 期間 SS は約60R・的中6本(per-leg) で ROI 672%、**最大払戻1本除去で242%・2本除去で162%**に崩落。SS headline は1〜2の高配当に支配される点推定で信頼区間が極端に広い。N<100 の層を絶対値で戦略採否・訴求に使うのは選択バイアス。
- 推奨対応: N<100 層は点推定でなく**ブートストラップ信頼区間＋最大払戻除去後ROI**を併記。採否は複数期間の順序安定性・的中率・払戻中央値で判断。SS は標本が貯まるまで暫定扱いと明記。

### MEDIUM

#### M-1. train/serve/eval で欠損特徴の扱いが不一致（dropna vs fillna(0)）
- ロール: シニアエンジニア（medium）／ データサイエンティスト（low, 生存者バイアス）
- 該当: `src/evaluation/backtest_wt.py:48-49`(`dropna`) / `src/cli/main.py:1021,1154`(`fillna(0)`)
- 事実: バックテストは `dropna(subset=FEATURE_COLS_WT)` で欠損行除去、本番予測・学習は `fillna(0)`。NaN を含む選手行が落ちると当該レースの n_riders・gap12・ratio・top3_sum・tier 判定が本番と変わりうる。「バックテスト=実運用上限値」の同一母集団前提を崩す潜在リスク。現状は `build_features_wt` が全特徴を fill 済みのため実測で該当0件だが、将来 fill 漏れ特徴を追加すると静かに不整合。
- 推奨対応: 3経路共通の `prepare_X(df)->X` ヘルパを用意し fillna(0) に統一。dropna で実際に落ちる行数/レース数をログ化。

#### M-2. 学習データに DNS(finish_order=0) 行が負例混入し、バックテスト母集団とズレる
- ロール: AI・MLエンジニア（medium）
- 該当: `src/cli/main.py:985`(`df[finish_order.notna()]`) / `src/preprocessing/feature_wt.py:101-103`
- 事実（PM裏取り済）: 学習行選択は `finish_order.notna()` で **DNS(=0) を含む**（2025+ データで finish_order==0 が 3,594 行≒1.4%、全期間では約9,190行相当）。これらは top3_flag=0 の負例として学習に入る一方、`_apply_pred_prob_wt` は `finish_order>=1` で除外（backtest_wt.py:48）。出走していない/失格選手を通常の競争上の敗北として学習させるのは母集団不整合・ラベルノイズで、DNS修正方針（採点対象外）とも一部矛盾。
- 推奨対応: `df_train = df[df['finish_order'] >= 1]` に揃え、train/eval/predict を同一母集団に統一。影響1.4%だが再現性・一貫性のため修正推奨。

#### M-3. 朝オッズ前向き計測は本日初投入で drift データが実質ゼロ → ガミゲートのOOS妥当性が未検証
- ロール: データサイエンティスト（medium）
- 該当: `scripts/snapshot_morning_odds_wt.py`, `wt_odds_snapshot`, `docs/analysis/03-odds-utilization.md`
- 事実: `snapshot_type='morning'` は 2026-06-08 の単一バッチ69Rのみ。snapshot と最終 wt_odds を突合すると trio 2,355ペアで final/morning 比中央値=1.0（同一スクレイプ由来）＝**朝→締切ドリフトを全く捕捉していない**。ガミ3段階（朝オッズ<3倍見送り等）と analyze_gami_*/gate_vs_gami は最終オッズで判定しており、運用の朝オッズ判定との乖離を裏付けるデータが未存在。「高配当は締切直前に下がる」も自前データでは未実証。
- 推奨対応: 発走数時間前に確実に1回 snapshot を取得する cron を確立し数週間分蓄積。蓄積後にガミ閾値(3/5倍)を朝オッズ基準で再評価。それまでガミ仕分け閾値は**暫定扱いと明記**。

#### M-4. `run_tiered_backtest_wt` の hits がレッグ単位加算で的中率/的中数が過大表示
- ロール: データサイエンティスト（medium）
- 該当: `src/evaluation/backtest_wt.py:321-333,342`
- 事実: thirds ループ内で各レッグ的中ごとに `hits += 1` する一方、`hit_rate = hits/races` でレース数で割るため単位不整合。SS で 16hits 表示 vs per-leg 厳密集計 6hits と意味が曖昧。
- 推奨対応: 的中は「そのレースで1点以上的中したか」のレース単位フラグで集計、`的中率=的中レース/対象レース`に統一。点数(レッグ)ベースを出すなら別指標として明示分離。

#### M-5. 週次再学習が `lgbm_wt.pkl` を in-place 上書き。世代バックアップ無くロールバック不能・再現性低下
- ロール: テスト・QAエンジニア（medium）
- 該当: `src/cli/main.py:1012-1015` / `scripts/weekly_retrain_wt.sh:18-20`
- 事実: `lgbm_wt_v1` と `lgbm_wt` は同一内容で毎週上書き、前週世代が残らない。学習期間が毎週ずれるためモデルは毎回変化するが、突然の劣化時に直前版へ戻せず・特定日の予想（モデル×カット）を後日再現できない（カットJSONも上書き）。
- 推奨対応: `lgbm_wt_YYYYMMDD` で世代を残し `lgbm_wt` はコピー/リンク。最低限 `data/models/archive/` へ退避1行。`upset_cuts_wt.json` も同様に世代退避。`train_lgbm` の乱数 seed 固定有無も確認。

### LOW（要点のみ）

| ID | 指摘 | ロール | 該当 |
|----|------|--------|------|
| L-1 | ratio≥1.6でスキップされるレースがガミ・スキップ件数に二重計上（収支影響なし・計測の不正確さ）。PM確認: skipped_gami(1228) は ratio≥1.6 の continue(1298) より前に走る | シニア | main.py:1217-1231,1297 |
| L-2 | 最安オッズ全欠損(None)時はガミ/Bランク判定を素通りし通常推奨に → フィルタ意図と逆のフェイルオープン | シニア | main.py:1224-1231 |
| L-3 | 4〜5車レースで thirds<3点になりコスト集計(×300固定)が実購入点数と乖離。n<3ガード無し | シニア | main.py:1205,1372-1379 / notify_results_wt.py:97 |
| L-4 | `_load_odds` がレースごと個別DB接続・クエリ（N+1）。日次バッチで致命的でないが一括ロード推奨 | シニア | main.py:1086-1102,1209 |
| L-5 | daily/weekly が `set -e` だが `pipefail` 未設定で `\| tee` が python 終了コードをマスク。真の異常も握り潰す（PM確認: daily に pipefail 無し） | シニア・QA | daily_picks_wt.sh:4 / weekly_retrain_wt.sh:3 |
| L-6 | venue_map のキー型不一致の可能性（wt は `str(venue_id)` 参照・venue_code が int 格納なら未ヒット） | シニア | main.py:1198-1199,1080-1081 |
| L-7 | as-of ローリングの逐次ループが当日多数レースで低速。ベクトル化推奨 | シニア・AI | feature_wt.py:249-269 |
| L-8 | value backtest の三連複 combo 確率が独立積近似＋無較正pred_prob。EV絶対値は信頼薄（相対指標と注記/較正推奨） | DS・AI | backtest_wt.py:382-393,396-470 |
| L-9 | L級(cls4)=player_class_enc=7 が SS=6 より上位を含意する順序エンコード。is_girls 二値分離推奨（FEATURE_COLS変更＝再学習要） | AI | feature_wt.py:21-30,120 |
| L-10 | data依存の median 補完(race_point/term)が学習(全期間)と予測(当日)で異なる補完値 → train/serve skew（現状NaN稀で実害小） | AI | feature_wt.py:111-112,123-124 |
| L-11 | `venue_wr` が `expanding().shift(1)`（行位置ベース）で同一選手同日複数走時に日内リーク（実検証で確認・影響レース極小） | AI | feature_wt.py:237-240 |
| L-12 | `picks_history.race_key` が UNIQUE。将来1レース多ランク出力化で後勝ち上書きの回帰リスク（現状は1レース1エントリで非顕在） | QA | database.py:189 / notify_results_wt.py:127-130 |
| L-13 | recompute 単調性NG時に保存スキップだが exit 0。weekly が成功扱いで沈黙（モデル異常シグナルを取りこぼす） | QA | recompute_upset_cuts_wt.py:62-64 / weekly_retrain_wt.sh:24-27 |

---

## 3. ロール横断テーマ（PM総括）

1. **リークの有無 — 結論: 明確な未来漏洩なし（実検証済）**。シニア・DS・AI・QA の4ロールが独立に検証し、`closed='left'` の時間窓・日付ソート・`venue_wr` の `shift(1)` により point-in-time が保たれることで一致。競輪は原則1選手1日1走のため同日リークも実質なし。唯一の残課題は `venue_wr` の同一日複数走（予選・準決の同会場連走）における日内リーク余地(L-11)だが、影響レース数が極小で severity=low。**この点はプロジェクトの最大の強み**。

2. **「上限値」フレーミングの一貫性 — 部分的に破綻**。analysis docs は ROI を「最終オッズ=上限値」と一貫表記しており設計思想は妥当。しかし (a) `picks_history` も最終オッズ採点で「実測」と称している(H-3)、(b) 朝オッズ drift 計測が実質ゼロでガミ閾値の朝オッズ妥当性が未検証(M-3)、という形で**「上限値と実測の分離」がコード/データ上は未実現**。フレーミングを支えるデータ基盤の整備が急務。

3. **モデル運用の再現性・配信品質 — 設計欠陥あり**。本番 `lgbm_wt` が holdout 評価モデルそのもの(H-1) で、かつ in-place 上書きで世代が残らない(M-5)。OOS 評価の健全性(AUC差0.009で確認)とは裏腹に、**本番に配信されるモデルが90日古い打切り学習**という実害があり、評価用と配信用 artifact の分離が必要。

4. **テスト不在 — 横断的な脆弱性(H-2)**。週次でモデル確率分布とカット定数が動くため、境界・冪等性・リークの担保が手動レビューのみ。確率分布ドリフトで静かに壊れる経路（ガミ閾値・Bランク振り分け・波乱ゲート）が多く、回帰検知の仕組みが致命的に不足。

5. **train/serve/eval skew の系統的リスク**。欠損補完(M-1)・median補完(L-10)・DNS母集団(M-2)の3つはいずれも「学習・バックテスト・本番予測で同じ計算を通す」原則の局所的破れ。現状は実測で発現0〜微小だが、特徴追加時に静かに不整合化する構造的弱点。共通 `prepare_X` ヘルパで一元化すべき。

---

## 4. 良い点（strengths・統合）

- **DNS=着外の一貫実装**: top3 判定が学習(`feature_wt` top3_flag)・バックテスト(`between(1,3)`)・採点(`finish_order BETWEEN 1 AND 3`)の全経路で統一され、欠車(0)を着外として正しく排除（4ロール一致・PM確認済）。DNSバグ修正の意図がコード全体で貫徹。
- **SS/S/A とオッズ市場の一貫性**: `is_ss` を先に確定し、ガミ判定(`_find_trifecta_odds`/`_find_trio_odds`)・表示オッズ・実買い目(SS=3連単/S・A=3連複)で同一市場を引く。SSは順序付き tuple、S/Aは順不同 frozenset で正しく区別。
- **Bランク除外の3経路一貫**: 推奨合計コスト・Discord通知・成績採点(picks_history) すべてから正しく除外（rank=None で素通り）。
- **バックテスト層別=本番 serving 条件の完全一致**で検証と本番が同条件評価。
- **ローリング特徴 point-in-time の堅牢性**（最大の強み・上記テーマ1）。
- **防御的設計**: `_load_cuts` の単調性チェック＋JSON不在/壊れ/非単調でのフォールバック、snapshot の `INSERT OR IGNORE` 冪等性（UNIQUE制約裏打ち）、notify の DELETE→INSERT 冪等性。
- **時系列CV**（日付ベース・バーンイン60%・early_stopping を val_set で）で構造的に未来漏洩を防止。
- **オッズ combination 区切り**（順序あり `-`／順序なし `=`）がスクレイパ保存と照合で整合。

---

## 5. 次アクション Top5（優先度順・具体）

1. **【H-1・最優先】本番モデル配信の是正**: weekly_retrain で holdout 評価後に `--test-from` 無しの全データ再学習を別途実行して `lgbm_wt` を生成・配信する。`main.py:1014-1015` の無条件コピーを見直し、評価用(`lgbm_wt_v1`)と配信用(`lgbm_wt`)を分離。pkl に train_from/train_to/コミット/AUC のメタを埋め込む。同時に M-5（世代退避 `lgbm_wt_YYYYMMDD` + archive、カットJSONも）を実施。
2. **【H-2】純粋関数のユニットテスト整備**: `strategy_wt`(upset_tier/passes_upset_gate/_load_cuts 境界・フォールバック)、`notify_results_wt`(_parse_combo/_parse_picks_full・Bランク除外)、ガミ3段階振り分け、as-of リークテストを pytest で追加。CI(または daily/weekly 前)で実行し回帰を固定。
3. **【H-3・M-3】「実測」基盤の整備**: 発走数時間前の morning snapshot cron を確実化し、`picks_history` の payout を朝オッズ/確定払戻ベースに変更（上限値と実現値を別カラム両建て）。数週間の drift 蓄積後にガミ閾値(3/5倍)を朝オッズ基準で再評価。docs の「実測検証」表現を最終オッズ採点である旨に修正。
4. **【M-1・M-2・L-10】train/serve/eval の母集団・補完の統一**: 共通 `prepare_X(df)->X`（fillna(0) 統一）を導入、学習行を `finish_order>=1` に揃え、median 補完値をモデル artifact に固定保存。dropna で落ちる行数をログ化。
5. **【H-4・M-4】小標本ROIの提示方法是正**: N<100 層(SS等)はブートストラップ信頼区間＋最大払戻除去後ROIを併記、的中率をレース単位に統一(M-4)。SS は暫定扱いと analysis docs に明記。あわせて L-5(pipefail)・L-13(recompute 非ゼロ終了/警告) の運用沈黙故障対策を低コストで実施。

---

## 6. 各ロール総評

- **シニアソフトウェアエンジニア（sound-with-issues）**: 中核ロジックは概ね正しく整合。致命バグなし。train/serve skew(dropna vs fillna)、ガミ件数二重計上、4〜5車のコスト集計・n<3ガード欠如、N+1、pipefail 未設定など保守性・エッジケースの low〜medium が残る。
- **データサイエンティスト（sound-with-issues）**: バックテスト中核ロジック・リーク無しは確認。一方で (1)本番モデルの学習範囲、(2)小標本ROIの万車券依存、(3)picks_history も上限値、(4)朝オッズ drift 未計測、を high〜medium で指摘。**PM注記**: (1)の「in-sample疑い」は実AUC再計算（TRAIN0.7832/TEST0.7741・差0.009）でホールドアウト整合と判明し概ね棄却。ただし「配信モデルが holdout モデル」という別問題(H-1)が確定した。
- **AI・MLエンジニア（sound-with-issues）**: リーク無し・train/serve一致を実検証。本番モデルが holdout 配信(H-1)を high で正しく指摘。DNS負例混入(M-2)、L級順序エンコード(L-9)、valueモードの独立性・無較正(L-8)、median補完skew(L-10)、venue_wr 日内リーク(L-11) を指摘。本番のSS/S/Aは順位依存で較正ロバストとの評価は妥当。
- **テスト・QAエンジニア（sound-with-issues）**: 新規ロジックのコアは手追いで正しく、冪等性・フォールバックも堅実と評価。最大の問題はテスト皆無(H-2)、pipefail 未設定(L-5)、モデル in-place 上書き(M-5)、recompute 沈黙(L-13)。回帰検知の仕組み不在を強調。

---

最終更新: 2026-06-08 / 作成: PM（テックリード兼任）レビュー統合。PM裏取り項目: 本番モデルAUC再計算・DNS行数・tests空・pipefail・weekly/main.py 配信ロジック・cuts JSON メタ。
