# セッション引継ぎメモ（最終更新 2026-06-08）

コンテキストリセット後にここから再開すること。

---

## ★★2026-06-08(夜) 3タスク分析 → 波乱ゲート実装 → 朝オッズ計測（最新）

ユーザー3課題①特徴/精度②波乱予測③オッズ活用をサブエージェントで分析（レポート: `docs/analysis/01〜03`）。**3タスクが同一結論に収束**: 「本命が堅いレースは低ROI・本命が割れた波乱余地レースが高ROI」。

- **①** ライン群②だけがROIに寄与(+81pp)。**セクター・AI印(prediction_mark)はROIを下げる**（AI印除去で合計349→386%）。**AUC↑≠ROI↑**。閾値は現状維持推奨（gap_lo 0.06→0.08のみ将来追試候補）。
- **②** 確定前情報だけで「想定ライン崩れ」をOOS AUC 0.735で予測可。**買い目構造変更（点数増/BOX）は投資増でROI希薄化＝劣後**。波乱スコア上位帯に絞るのが最善。
- **③** ⚠️ レポート§3の `top2_sum<0.80`(607%)は**スケール誤りのアーティファクト＝撤回**（実際 top2_sum中央値1.40でほぼ該当なし）。`docs/analysis/03` 冒頭に訂正注記済。最終オッズフィルタは上限値、**朝7:00確定で問題なし**。
- **★統合検証(`scripts/exp_upset_gate_wt.py`)**: 正しい頑健シグナルは **`top3_sum`(上位3頭pred_prob合計)のloose四分位**。TRAIN四分位カット[1.70,1.90,2.08]をTEST(OOS)適用→ **Q1_loose(top3_sum<1.70): TRAIN ROI 1224% / TEST 1136%(125R・最大払戻除外でも934%)** vs Q4_chalk 107%。単調・train/test一致・volume十分・万車券単発非依存。

### ★ガミ回避オッズ3段階 採用（2026-06-08・本番cron反映済）
ユーザー判断。3点の**最安目の朝オッズ**で振り分け（`daily_picks_wt.sh`: `--gami-skip-odds 3.0 --b-rank-odds 5.0`）:
- **<3倍 → 見送り**（明確なガミ）/ **3〜5倍未満 → Bランク**（別枠・購入は各自判断・推奨合計に含めず・detail.json `rank="B"`/`base_rank`保持）/ **≥5倍 → 通常推奨(SS/S/A)**。
- 根拠(閾値スイープ `scripts/analyze_gami_threshold_wt.py`): 総損益は<3〜6倍でほぼ横ばい・ROIは単調上昇・購入R半減 → <3倍点含むレースは集団で収支ゼロの鉄板。「安い目カット」より「レース単位振り分け」が ROI・総損益とも上（TEST 全件286%→<5倍除外636%、総利益ほぼ維持）。
- ⚠️ 朝オッズ基準＝朝→直前ドリフトの影響を受ける（`snapshot_morning_odds_wt.py --report` で計測中）。オッズ不要の同義代替 = `--upset-gate`(top3_sum)。
- スクリプト: `scripts/analyze_gami_policy_wt.py`（カット vs スキップ）, `scripts/analyze_gami_threshold_wt.py`（閾値3〜8倍スイープ）。
- 例: 6/8 いわき平6R（最安3.0倍）→ Bランク(元A)。<3倍の2レースは見送り。

### ★#7 検証: 波乱ゲート(top3_sum・オッズ不要) vs ガミ3段階(最終オッズ) の重複（`scripts/analyze_gate_vs_gami_wt.py`）
- 最安オッズ と top3_sum の順位相関 **−0.76**（looseほど高オッズ）。推奨レースの **85〜87%が一致**（ガミ推奨≥5倍 ⊂ ほぼ ゲートQ1+Q2）。両者ROIとも黒字（test ガミ636%/148R vs ゲートQ1+Q2 492%/201R, Jaccard0.59）＝**相補的で置き換え不要**。
- **最重要**: ドリフトリスクは**Bランク帯(Q2/Q3・3〜5倍付近)に集中**。推奨側Q1は最安中央値24倍・見送り側Q4は2倍で**境界から遠く、朝→直前ドリフトで判定が反転しにくい**＝**推奨(≥5倍)と見送り(<3倍)はドリフトに堅牢**、揺れるのは元々各自判断のBのみ。
- **結論**: 現行ガミ3段階を据え置きでよい。`top3_sum`(detail.jsonにタグ済)はオッズ不要のクロスチェックとして併用可。朝オッズドリフトの実測(A-2)は引き続き有効だが、構造上は両極が安全と確認。

### ★#10 解決: L級(ガールズ)クラスのマッピング追加（2026-06-08）
- `wt_entries.player_class` の未マッピング値 `cls4`(50,452件=grade'L級'ガールズ・混在なし)・`cls1`(2,290件=S級にS1/S2と混在・group欠損)が `player_class_enc=-1`（約7.7%）だった。
- `feature_wt._CLASS_MAP` に `"cls4":7`(L級・別軸識別子)・`"cls1":4`(S2相当)を追加。**winticket.py は変更不要**（保存済み文字列を直接マッピング＝既存96kに即効・旧/新ラベル不整合なし）。
- 再学習: CV AUC **0.7719** / Test **0.7741**（旧0.7720/0.7742と中立）。`-1`残存ゼロ。`train-wt --save-as` は `lgbm_wt.pkl` も同時保存するため**本番モデルは新マッピング版に更新済**（週次cronでも再現）。

### ★#8 解決: 波乱カット定数の自動再計測（2026-06-08）
- 課題: 週次再学習で pred_prob 分布が動くと `top3_sum` 四分位カットがズレる。
- 対応: `scripts/recompute_upset_cuts_wt.py`（train分布で四分位再計測→`data/models/upset_cuts_wt.json`保存・gitignore）を新設し `weekly_retrain_wt.sh` に組込。`src/strategy_wt.py` を `_load_cuts()`（JSON優先・無ければ `UPSET_TOP3SUM_CUTS_DEFAULT`・単調性チェック付）に変更。
- 現行再計測 (1.693/1.901/2.075) は既定(1.70/1.90/2.08)とほぼ不変＝新マッピングでも分布安定。JSON削除時は既定値にフォールバックを確認済。

### 実装済み（試験実装・本番挙動は不変）
- `src/strategy_wt.py`（新規）: `race_signals()` / `upset_tier()` / `passes_upset_gate()`。カット定数 `UPSET_TOP3SUM_CUTS=(1.70,1.90,2.08)`（**再学習で確率分布が変わったら再計測**）。
- `wave-picks-wt`: 各pickに `top3_sum`/`upset_tier` を**タグ付け（既定・detail.jsonに記録）**＋ **opt-in `--upset-gate Q1_loose|Q2|Q3`** で本命堅レースを見送り。既定は全件出力で本番不変＝前向き検証用。
- **朝オッズ前向き計測**（③が指摘・別途承認で実装）: `wt_odds_snapshot` テーブル＋`scripts/snapshot_morning_odds_wt.py`（取得/`--report`ドリフト集計）。`daily_picks_wt.sh` の当日collect-wt直後に退避ステップ追加。**朝→最終ドリフトは過去データでは測定不能→今日(6/9)から蓄積開始**。

### 残TODO（次セッション）
- 波乱ゲートの**実運用前向き検証**: detail.jsonの `upset_tier` × picks_history(route='wt')でlive ROIを帯別集計（数週間蓄積後）。ROI 1136%は**最終データ上限値**＝実運用は割り引いて解釈。
- 朝→最終ドリフト計測: 数日蓄積後 `snapshot_morning_odds_wt.py --report` で市場別に確認。±20%以内割合が高ければ朝オッズフィルタが信頼できる。
- 本番反映判断（ユーザー方針=「見てから個別判断」）: ゲートをcron既定化するか、ステーク傾斜にするかはlive実測後。

---

## ★★現在の本番構成（2026-06-08・最重要サマリ）

**本番ルートは winticket(wt) に完全移行。keirin-station(ks)スクレイピングは停止。**

| 項目 | 状態 |
|------|------|
| 本番モデル | `data/models/lgbm_wt.pkl`（=lgbm_wt_v1, 39特徴, CV AUC 0.7720/Test 0.7742） |
| wtデータ | 96,355R（2022-12〜2026-06）, オッズ3,379万, 結果率99.88% |
| 日次cron | 7:00 `scripts/daily_picks_wt.sh`（collect-wt→notify_results_wt→collect-wt→wave-picks-wt→notify_picks …wave_picks_wt） |
| 週次cron | 日23:30 `scripts/weekly_retrain_wt.sh`（train-wt） |
| バックテスト性能 | A187%/S364%/SS1205%/合計336%（テスト期間・**最終データ上限値**） |
| 戦略 | 6車以下 SS/S/A（gap12/ratio）3連単(SS)/3連複2軸流し(S・A) 3点 |
| ks資産 | lgbm_v6等 保持（ロールバック用）。ks日次/週次スクリプトは未使用 |

**最大の発見**: wtが当初ksに劣って見えたのは `finish_order=0`(欠車)を3着内に誤算入する単一バグが原因。修正後wtはks同等以上。詳細は下記「★★真因判明」節。

**最重要caveat**: 336%は**最終データbacktest（欠車事後判明・朝-確定ズレ未反映）の上限値**。真の実運用ROIは未確定。今後 `picks_history(route='wt')` 蓄積で測定する。ksの実運用実績(修正後)は1週間で49%だった点に留意。

**次の改善候補**: ①wt実運用ROIの実測 ②波乱(ライン崩れ=高ROIゾーン)を狙う特徴/層別の強化 ③朝7:00収集時の出走表/オッズ確定度の確認。

---

## ★成績報告バグ修正＋バックテスト信頼性の重要caveat（2026-06-08）

### notify_results.py の重大バグ（修正済）
- **症状**: 報告月次ROI 102%(53R) だが、実際の公開予想の成績は **49%(35R・損益-5,350円)**。約2倍の過大評価。
- **原因**: `notify_results.py` が予想ファイルを採点せず、翌朝の**再収集データでモデルを再実行**していた。→ ①公開ファイルに無いレースを gap12/ratio で自動算入(松阪11R¥1970等の万車券混入) ②欠車/最終確定で買い目が再導出され公開時と乖離(松阪8R 1-2-5→1-5-2で外れが的中に化ける)。
- **修正**: main()を「公開ファイル(`_parse_picks_full`)の買い目をそのまま採点／未公開レース算入せず」に変更。`scripts/rescore_picks_history.py --apply` で過去分(2026-06)を再採点済。
- 修正後 picks_history(2026-06): 35R 的中15(42.9%) ROI **49.0%**。

### バックテスト信頼性への影響（重要・要認識）
- **リーク無し＝内部的に妥当**: backtest/reaudit_ks/finalize は特徴量が全て事前情報(race_point/rolling=point-in-time/line)、finish_positionは採点のみ。モデルのエッジ測定としては正しい。
- **但し最終データでの楽観バイアスあり**: backtestは**最終確定の出走表**で予想を導出するが、実運用は**朝の暫定データ**で予想する。欠車/直前変更で着順予測が変わると、backtestの的中は実運用で再現不可。6/7松阪8Rが実例(最終データ1-5-2なら的中、朝の1-2-5は外れ)。
- **帰結**: **ks 238%等のbacktest値は「最終データ性能」＝実運用ROIの上限**。実運用(公開予想を正しく採点)は現状49%(35R・1週間・高分散)で、真の実力はこの間。確定には①朝データ・スナップショットでのbacktest ②修正後picks_historyの実運用サンプル蓄積、が必要。
- TODO: 朝収集データを保存し「朝データbacktest」で実運用ROIを正しく見積もる。ks 238%を鵜呑みにしない。

### winticketの値安定性（2026-06-08 検証）
- **winticketの first_rate / race_point は開催(3日間)単位で固定**、開催をまたぐ時だけ更新（野口裕史の時系列で確証: 05-15〜17=fr10.0固定、05-23〜25=12.1固定…）。
- 過去レースのracecard値はレース後も凍結（6/4レースを6/6と6/8で取得→完全一致）。
- **帰結**: ①同レース結果のリーク無し（勝率は開催開始時点＝当該レース・開催中前走を含まない）②朝-レース後で値が変わらない（ks rolling特徴は毎レース更新だが、wtは開催固定）。→ **winticket backtestはこの点でksより live再現性が高い**。
- 残る朝-確定ズレ要因: 欠車(許容済)、ライン/並び変動(未確認)。

### ライン予想精度と波乱の関係（2026-06-08 検証）
- ライン特徴は `linePrediction`（レース前予想ライン）＝クリーン・backtest=live。
- A層を「予想ライン通り(実1-2着が同line_group) vs 崩れ」で分割（テスト期間）:
  - ライン通り: 106R 的中**62%** ROI73%
  - ライン崩れ: 92R 的中**39%** ROI77%
- **ライン予想の正否は的中率を強く左右（62 vs 39%）が、ROIはほぼ同じ(~75%)**。理由: ライン通り=本命決着=低配当、ライン崩れ=高配当だが難。
- **含意**: ライン予想精度↑→的中率↑だが**ROIは天井のまま（本命狙い）**。収益の妙味は「ライン崩れ＝波乱」側。課題は「ラインを当てる」でなく「**波乱を予測する**」こと。ks238%もローリング特徴が波乱兆候を捉えている可能性（ksは詳細ライン無しでも勝てる傍証）。
- **方向性**: wt本番では「波乱予測」を狙う特徴/層別を検討（高配当ゾーンの局所エッジ探索と一致）。

### ★最終本番評価（全期間収集完了後 2026-06-08）
- 収集完了: **96,355レース**(2022-12〜2026-06)・681,960エントリ(結果99.88%)・オッズ3,379万・収集errors0。欠損30会場日(0.32%・winticketカバレッジ外)。
- finalize_wt_eval（全データ・ks同一の2023-07学習・rolling+line+頭数重み, lgbm_wt_v1保存）:
  | 期間 | A層ROI | S層 | 合計 | AUC |
  |---|---|---|---|---|
  | 検証6mo | 86%(327R) | 68% | 83% | 0.768 |
  | テスト3mo | **70%(193R)** | 64% | 69% | 0.773 |
- **結論: winticketは ks(A238%) に届かず**。深い履歴・rolling・ライン・ks同一学習期間を揃えても A層テスト70%。当初仮説(履歴の浅さ)は否定。
- 切り分け: 学習期間深さ❌/rolling❌/ライン❌/本命バイアス除外❌ いずれも個別要因では説明不可。ksの優位は**ks特徴量固有の妙味(波乱)検出力**(同AUC0.77でもksは高配当combo、wtは本命combo選択)。
- **但しks238%も最終データ上限値**。修正後実運用(picks_history)は1週間49%。真の実力は両者の間、要継続検証。
- **確定方針**: **ks=本番勝ち筋(継続運用)**。winticket=単独賭けモデルではks劣後だが、全オッズ盤面(ks払戻検証・EV分析を可能化)＋ライン＋96k研究基盤として価値。今後は「波乱予測」研究 or ks強化に注力。
- 未解明(次セッション候補): ks vs wt の同一レースでの pivot選択差を直接比較し、ksの妙味検出の正体を特定。

### ★★真因判明＆大逆転: DNS(欠車)バグ (2026-06-08)
ks vs wt のデータ差を精査した結果、得点・勝率は同一(相関0.94〜1.0)、ライン情報はwtが豊富。**唯一の決定的差は欠車の表現**:
- ks: `finish_position=NULL`で欠車→全処理で正しく除外
- wt: `finish_order=0`で欠車→コードが `finish_order<=3` で判定し**0を「3着内」に誤算入**(9,190件/1.35%)。
  - 影響: ①学習target `top3_flag` 汚染 ②backtest top3_set破壊(欠車混入で4要素→的中不能) ③≤6車判定狂い(ks890 vs wt477レース)
- **修正**: `feature_wt.py` top3_flag を `between(1,3)`、`backtest_wt.py` の `_apply_pred_prob_wt`(>=1除外)＋全`finish_order<=3`→`between(1,3)`、finalize_wt_eval も同様。
- **修正後 finalize_wt_eval (全データ・2023-07学習・rolling+line)**:
  | 層 | 修正前test | 修正後test | ks参考 |
  |---|---|---|---|
  | SS | 0%(3R) | **1205%(56R)** | 1321% |
  | S | 64% | **364%(111R)** | 185% |
  | A | 70% | **187%(286R)** | 238% |
  | 合計 | 69% | **336%** | - |
- **結論大逆転: wtは ks同等以上**(S層はks超)。「wtがksに勝てない」は単一DNSバグが原因だった。ksが勝っていたのは特徴量でなく欠車を正しく除外していたから。
- **重要**: 過去のwt否定的分析(EV妙味37%・本命バイアス・利益条件なし・A層70%)は**全てこのバグで汚染**＝撤回。lgbm_wt_v1再学習済。
- 残caveat: これも最終データbacktest(欠車は事後判明)。実運用は朝-確定ズレ＋picks_history実績蓄積で要検証。

### ①wave-picks-wt 実運用化 完了（2026-06-08）
- ローリング特徴を `feature_wt.py` に統合: `add_rolling_features_wt()`（point-in-time、学習=履歴merge/予測=当日as-of両対応）を `build_features_wt` 末尾で自動実行。`FEATURE_COLS_WT` を39特徴に拡張（rolling 9列追加）。
- 本番モデル再学習: `train-wt --from 2023-07-01 --test-from 2026-03-01 --save-as lgbm_wt_v1`（lgbm_wtも同時保存）。CV AUC 0.7720 / Test 0.7742。
- `wave-picks-wt`（既定model=lgbm_wt）が39特徴で正常動作。発走時刻バグ修正（start_at=unix秒→JST HH:MM、main.py `_fmt_start`）。SS/S/A＋オッズ＋時刻すべて正常。

### ②EV/波乱分析 DNS修正後 再検証（2026-06-08）
- EVバケット別実ROI: 依然較正されず(0-0.8帯64%〜2.0+帯60%、高EV≠高ROI)。**機械的EV買いは不可**。
- ライン通り/崩れ(A層・テスト): ライン通り 153R 65% **ROI158%** / ライン崩れ 136R 49% **ROI278%**。**両方黒字、波乱側が高ROI**＝「価値は波乱側」をDNS修正後データで実証。
- 結論: 収益源は**層別pivot戦略(336%)**。特に波乱(ライン崩れ)レースが高ROI。combo-EVの機械買いではなく「競合レースで上位を当て波乱配当を取る」構造。
- 過去の否定的EV/本命バイアス結論は撤回（DNSバグ汚染が原因だった）。

### ★ks→wt 完全移行 完了（2026-06-08）
ユーザー判断: wtがks以上の精度を確認→運用をwtへ完全移行、ksスクレイピング停止。
- **日次運用**: `scripts/daily_picks_wt.sh`（新規）= collect-wt前日→notify_results_wt→collect-wt当日→wave-picks-wt→notify_picks(wave_picks_wt)。
- **週次再学習**: `scripts/weekly_retrain_wt.sh`（新規）= train-wt（直近90日をtest分割）。
- **cron 切替済**: `daily_picks_wt.sh`(7:00) / `weekly_retrain_wt.sh`(日23:30)。**ks版(daily_picks.sh/weekly_retrain.sh)は呼ばれなくなりks収集停止**。
- 新規/変更スクリプト:
  - `notify_results_wt.py`（新規）: wt結果(finish_order 1-3)+wt_odds で公開買い目を採点、picks_history(route='wt')保存、Discord通知。
  - `notify_picks.py`: 第2引数でプレフィックス指定可（`wave_picks_wt`）に汎用化。
  - `pipeline_wt._get_collected_keys`: 「結果あり(finish_order>=1)のみスキップ」に修正→前日再収集で結果取得可（ks同方針）。
  - `picks_history` に `route` 列追加（ks/wt区別）。INSERT OR REPLACE で移行時の重複race_key解消。
- 検証: 全スクリプト構文OK。notify_results_wt/notify_picks をmock送信で2026-06-06採点・通知正常確認。
- **注意**: ksの venue絞り込み(`_venues_racing_on`)はks `races`参照だがks停止後はフォールバックで全43会場走査（動作OK・やや低速）。winticket自体の find_cup_info で開催判定されるため問題なし。

### 残TODO
- wt実運用 picks_history 蓄積で、朝-確定ズレ込みの**真の実運用ROI**測定（backtest 336%は最終データ上限値）。
- 必要なら collect-wt の当日収集タイミング最適化（朝7:00時点の出走表確定度）。
- ks資産(lgbm_v6等)は当面保持（ロールバック用）。

## 🗄 アーカイブ: 2026-06-07 収集フェーズ＆検証ログ（記録として保持・以下すべて履歴）

> **以下はすべて履歴**。winticket 全期間収集は**完了済**（96,455R）。当時の否定的結論（EV妙味なし・本命バイアス・利益条件なし・wtがksに劣る等）は**DNS(欠車)バグ汚染が原因で撤回済**（最新の正しい結論は本ファイル冒頭の「★★現在の本番構成」「★★真因判明」を参照）。「現在の状態」テーブルのみ現行情報。収集手順・暫定モデル・各種実験は再現性のため残置。

### 🔄 (旧) ON RESUME（2026-06-07時点・winticket全期間収集中）

**1. 収集状況を確認:**
```bash
ps -p $(cat data/logs/collect_wt.pid) && echo ALIVE || echo DEAD   # プロセス生存確認
.venv/bin/python3 -m src.cli.main status-wt                          # 件数・期間
tail -3 "$(cat data/logs/collect_wt.logpath)"                        # 進捗（[N/43]）
```
- DEADなら再開: `nohup .venv/bin/python3 -m src.cli.main collect-wt-range --from 2022-12 --to 2026-06 > data/logs/collect_wt_resume.log 2>&1 &` （スキップ判定で安全に続きから）
- 全期間完了の目安: 2022-12 まで到達（`[43/43]`）

**2. 収集完了後の本番評価（準備済みスクリプトを順に実行）:**
```bash
# ① 欠損チェック（瞬断由来の取りこぼしを再収集）
PYTHONPATH=. .venv/bin/python3 scripts/gap_check_wt.py            # 検出
PYTHONPATH=. .venv/bin/python3 scripts/gap_check_wt.py --recollect # 欠損あれば再収集
# ② 本番評価: ks流ローリング特徴+ライン情報で学習→9/6/3 OOS層別→ks比較→モデル保存
PYTHONPATH=. .venv/bin/python3 scripts/finalize_wt_eval.py \
    --train-from 2023-07-01 --val-from 2025-09-01 --test-from 2026-03-01 --save-as lgbm_wt_v1
# baseline比較したい場合: 上記に --no-rolling を付けてもう一度
# ③ ks再監査の再現（参考値の確認）
PYTHONPATH=. .venv/bin/python3 scripts/reaudit_ks.py --model lgbm
```
**判定基準**: finalize_wt_eval のテスト期間で A層ROI>100%（理想はks 238%超）かつ検証でも黒字なら、winticketルート成功。

**準備済みスクリプト（全て `scripts/`、本セッションで作成・検証済）:**
- `gap_check_wt.py` — ks比較で欠損会場日を検出/再収集（バグ修正済・動作確認済）
- `finalize_wt_eval.py` — 本番評価一括（ローリング特徴計算込み）
- `reaudit_ks.py` — ksルートROI再監査（OOS+jackknife）
- `mine_profitable_conditions_wt.py` — 利益条件マイニング（train/test分離）
- `backtest_963_wt.py` — 9/6/3分割バックテスト
- `exp_wt_rolling.py` — ローリング特徴の効果検証（プロトタイプ）

**重要な結論（詳細は下記「winticket ルート」節）**: 市場は半効率的＝優れた特徴量で勝てる。ksは真のOOSで A238%/S185%（リークなし・jackknife頑健・払戻データ正確）。wtは弱い特徴量で敗北→ks流ローリング特徴移植で改善するが**まだksに未到達**（下記2026-06-07夜の追加検証）。

#### wt深いデータ(2023-12〜)での学習・検証・バックテスト（2026-06-07夜）
現データ(2023-12〜2026-06)で finalize_wt_eval（学習2024-06〜/検証〜2026-02/テスト2026-03〜）:
- ローリング特徴版: 検証A67%/合計68%、テストA78%/合計76%、AUC0.774。
- **ksとの決定的差＝当たり配当**: ks A層は当たり平均**1361円**(51%的中・ROI232%)、wt A層は**約430円**(52%的中・ROI75%)。**wtは本命、ksは妙味を選ぶ**(3倍の配当差)。
- **本命バイアス除外実験**(`scripts/exp_wt_no_mark.py`): prediction_mark(AI印)も表示勝率も外しても当たり平均420-430円のまま改善せず。→ favorite-biasは特定特徴でなく**wtモデル全体が市場consensusに収束**。
- **残る仮説**: ①ks学習期間が2年(2023-07〜)と深い vs wt1.2年 ②ks固有特徴(racing_score計算法/line_pos_enc)。
- **次の決定的検証(全期間完了後)**: wtを**ksと同じ2023-07〜で学習**しフェア比較。並ばなければ ks本番が勝ち筋確定、winticketはライン情報の補助に。
- 追加スクリプト: `scripts/exp_wt_no_mark.py`(本命バイアス除外実験)。

---

## 現在の状態

### winticket ルート（★本番稼働中 — 2026-06-08〜）

| 項目 | 状態 |
|------|------|
| スクレイパー | `src/scraper/winticket.py` ✅（日付照合/オッズ/ワイドの3バグ修正済） |
| パイプライン | `src/scraper/pipeline_wt.py` ✅（会場絞り込み＋4並列、結果ありのみスキップ） |
| 特徴量 | `src/preprocessing/feature_wt.py` ✅（**39特徴**・rolling統合・DNS処理修正済） |
| バックテスト | `src/evaluation/backtest_wt.py` ✅（通常/`--tiered`/`--value`、DNS修正済） |
| DBテーブル | `wt_races` / `wt_entries` / `wt_odds`、`picks_history.route` |
| CLIコマンド | `collect-wt` / `collect-wt-range` / `train-wt` / `backtest-wt` / `wave-picks-wt` / `status-wt` |
| データ | **96,355R**（2022-12〜2026-06）/ オッズ3,379万 / 結果率99.88% |
| モデル | `lgbm_wt.pkl`（=lgbm_wt_v1, CV AUC 0.7720 / Test 0.7742） |
| 日次/週次 | `daily_picks_wt.sh`(cron 7:00) / `weekly_retrain_wt.sh`(cron 日23:30) |

### keirin-station ルート（★収集停止・ロールバック用に保持）

| 項目 | 値 |
|------|---|
| 状態 | **2026-06-08 ks→wt移行により収集停止**。日次/週次cronはwt版に切替済 |
| DB | `data/keirin.db`（races/race_entries/odds は移行時点で凍結） |
| モデル | `lgbm.pkl`(=lgbm_v6, CV AUC 0.7575) 保持 |
| 旧戦略実績 | ホールドアウトROI SS3944%/S158%/A228%（**最終データbacktest=上限値**。実運用は修正後49%/1週間） |
| 旧スクリプト | `daily_picks.sh`/`weekly_retrain.sh`/`notify_results.py`/`wave-picks` 未使用（保持） |

#### 修正した重大バグ（収集が0件だった原因）
1. **日付照合**（`winticket.py find_cup_info`）: スケジュール日付が `YYYYMMDD` 形式なのに `YYYY-MM-DD` で比較 → cup情報が永遠に見つからなかった。両者を `YYYYMMDD` に正規化して比較するよう修正。
2. **オッズDB書き込み**（`pipeline_wt.py _write_race`）: `combination` がリスト `[1,2,3]` のまま SQLite にバインド → 全レース書き込み失敗。順序市場は `-`、順不同市場は `=` で結合した文字列に変換。
3. **ワイドのオッズ0**（`winticket.py fetch_odds`）: quinellaPlace は `odds=0`／`minOdds`にレンジ下限 → `minOdds` フォールバックを追加。`wt_odds` は `INSERT OR REPLACE` に変更。

#### 中間バックテスト結果（2026-06-07 / 直近13ヶ月 lgbm_wt_interim）
- 学習 20,502R（2025-06〜2026-02）/ テスト 7,333R（2026-03〜）
- **CV AUC 0.7668 / ホールドアウト Test AUC 0.7720**（ks版 CV 0.7575 を上回る = モデルの判別力は良好）
- バックテスト（6車以下・gap12≥0.06・314R）: **全戦略 ROI < 100%**（最高 exacta_21 86.6% / 3連複 jiku2_3 72.3%）
- **SS/S/A層別バックテスト**（`backtest-wt --tiered` 新規実装）: SS 693.6%(4R・サンプル過小) / S 75.9%(39R) / A 74.8%(198R) / 合計 84.4%(241R)
- **決定的な原因**: S+A の的中三連複の払戻が **中央値310円・平均440円**（121的中中59件が<300円）。payout欠損は0件＝計算は正常。つまり **モデルが本命を当てすぎて配当が安い**（AUC0.772＝予測は優秀だが、本命はオッズに織り込み済みで儲からない）。
- **ks route A-rank 215% との乖離の解釈**: ①ksの閾値(gap12 0.15/ratio 1.3,1.6/0.06)はks確率分布向けにチューニング済み→wt確率分布には別チューニングが必要 ②儲かるのは SS層(ratio<1.3=競合レースでAI的中時に高配当)だが13ヶ月で4Rと希少。全期間ならSSサンプル増。③ks報告値が楽観的だった可能性も排除できない。
- **本番(全期間)での課題**: wt確率分布に合わせた gap12/ratio 閾値の再最適化（訓練期間のみで）＋SS層の十分なサンプルでの検証。

#### バリュー(EV)バックテストの結果と重要結論（2026-06-07）
`backtest-wt --value`（EV=モデル組合せ確率×オッズ で買い目選択）を新規実装し検証：
- 組合せ確率を **正規化積** と **Plackett-Luce** の2方式で推定 → どちらも **ROI 約35〜47%**（tier戦略84%より悪化）
- EV≥1（モデルが市場より高評価＝割安）の三連複を買っても **的中率0.5%・ROI 37%** で大敗
- **結論（重要）**: モデルが市場と乖離する買い目＝**モデルが間違っている**（市場の方が正確）。つまり **モデルは市場(パリミュチュエル)に対する優位性を持たない**。
- **理由**: wtモデルの特徴量（得点・勝率・ライン・AI印）は**市場も見ている公開情報**。同じ情報からは市場効率を超えられない（効率的市場仮説）。AUC0.772は「予測は優秀」だが「予測≠利益」。
- **ks route の 200%+ ROI も同手法で再監査すべき**（過学習/期間依存の可能性。同じEV検証にかけて本物か確認）。
- **残された優位性の探索余地**: ①確定前の早期オッズ（市場収束前）の妙味 ②小規模プール/特定条件で市場が systematic に誤る場面の実証的発見 ③予測精度自体を「情報商材」として提供（利益保証ではなく）。

#### 次アクション（ユーザー決定 2026-06-07）: ①全期間収集完了を待つ → ②ksルートROIを再監査
**ks odds テーブルの制約**: `payout`（当選組の実払戻）は有るが `odds_value`（全組合せ事前オッズ盤面）は**全NULL**。→ ks単体ではEV計算不可。
**再監査の設計**: winticket は全組合せのオッズ盤面を持ち同一実レースをカバーするため、**wtオッズ盤面 × ksモデルの買い目** でEVを算出する。手順:
1. 全期間収集完了後、本番wtモデル `lgbm_wt_v1` 学習（2023-01〜 / test 2025-06〜）
2. **ksモデル(lgbm_v6)のA/S/SS買い目**を、重複期間(2025-06〜)で wt_odds の実オッズに紐付け、(a)payout整合性チェック（ks payout ≒ wt odds×100 か）(b)EV分布を算出
3. ksの報告ROI（A215%/SS3944%）が、(i)本物の市場優位 か (ii)過学習/期間依存アーティファクト かを判定
4. wt本番モデルでも tiered + value(EV) バックテストを全期間で再実行（SS層サンプル十分）
**仮説**: 両ルートとも公開情報のみ使用 → 効率的市場を超えられないはずで、ks報告値も過学習の可能性。wtオッズ盤面で厳密検証する。

#### 決定的診断: EVバケット別の実ROI（2026-06-07・279,138組合せ）
全trio組合せをモデルEVでバケット分けし実ROIを集計した結果、**EVが高いほど実ROIが低い（逆相関）**:
| EVバケット | 件数 | 的中率 | 実ROI |
|---|---|---|---|
| (0,0.5]本命 | 45757 | 7.5% | **63.9%** |
| (0.8,1.0] | 27302 | 2.1% | 66.3% |
| (1.0,1.2] | 23771 | 1.5% | 51.8% |
| (1.5,2.0] | 36789 | 0.6% | 38.9% |
| (3.0,100] 大穴 | 28737 | 0.2% | **33.5%** |
- **モデルが「割安(高EV)」と判断した買い目ほど実際は最悪のROI**＝EVが反予測的(anti-calibrated)。
- 最良の本命バケットですら63.9%（テラ銭25%控除後の効率水準~75%を下回る）→ **どのEV帯にも黒字ゾーンが存在しない**。
- **シグモイドでEV>1から賭け比重を上げる案は、最悪ゾーン(33%)に賭けを集中させ逆効果**。
- 確率較正(isotonic)をしても「正EVの買い目が存在しない」を正しく示すだけで、利益は生まない。ranking自体が逆なので較正では救えない。
- **結論の強化**: 現特徴量(公開情報)では、テラ銭を超える賭け戦略は構築不可能。残る可能性は「特定レース条件の局所的エッジ(SS的拮抗レース)を全期間データの十分サンプルで探索」のみ。

#### 競馬ML記事の概念検証（2026-06-07 / note dijzpeb n1afb70e3c981）
記事の2概念を競輪に適用検証：
1. **頭数バイアス対策（weight=1/n_riders）**: `train_lgbm(weight_col=...)` を実装し検証 → **効果ほぼ無し**。6車以下AUC 0.7568→0.7561、EVバケット別ROIも依然 単調減少（anti-calibrated）のまま。理由: 競馬は6〜18頭で偏り大だが、競輪は5〜9車と幅が狭くバイアスが小さい。※理論的に正しいので本番モデルでは採用継続（無害＋6車注力時に妥当）。
2. **確率整合性**: 既に Plackett-Luce（スケール不変の coherent joint）でcombo確率を算出済み＝実質取り込み済み。
- **総括**: 両概念は健全なML衛生だが、**市場優位性は生まない**。EVバケットの anti-calibration は全介入で不変＝問題は賭け方やバイアスではなく「公開情報のみで市場効率を超えられない」点。

#### keibaAI-v2 リポジトリの概念検証（2026-06-07 / github keibaAI-community/keibaAI-v2）
取り込んだ新概念: **レース内標準化スコア(z-score)** × **閾値スイープ** × **可変ベット(0点〜)**（StdScorePolicy + BetPolicy*Box + Simulator）。競輪は単勝/複勝が無いため、閾値超え選手集合→**三連複/三連単BOX**で組成。
検証結果（interim 2026-03〜・/tmp/exp_threshold_sweep.py）:
| 券種 | z閾値 | 購入R | 的中率 | ROI | std |
|---|---|---|---|---|---|
| 三連複 | 0.5 | 2217 | 23.6% | 68.6% | 0.03 |
| 三連複 | 0.8 | 239 | 20.1% | 62.9% | 0.11 |
| 三連単 | 0.8 | 239 | 20.1% | 68.4% | 0.17 |
| 三連単 | 1.0 | **19** | 15.8% | **151.1%** | 1.36 |
- 大量ゾーン(z 0.3〜0.8)は一貫して **65〜69%**（量的エッジ無し、既存結論を再確認）。
- **z≥1.0の高選択性ゾーンのみ151%**（ただし19R・std1.36＝分散ノイズと区別不能）。前回SS層693%(4R)と同一現象。
- **収穫**: レース内z-scoreは gap12/ratio より principled な選択指標。3手法(pivot/EV/閾値)が独立に「高選択性レースにのみ局所エッジの可能性」を示唆。
- **全期間データの決定的価値**: 43ヶ月なら z≥1.0 が約200R集まり、151%が持続(本物)か70%回帰(ノイズ)か判定可能。これが収集完了後の最重要検証。
- TODO: `backtest-wt` に score_z 閾値モード(`--zscore-threshold`)を正式実装。exp は scripts/explore_zscore_threshold_wt.py に移管済。

#### skyley記事＋識別子特徴量の検証（2026-06-07 / skyley.com 競馬予想AI）
- 記事の最重要発見=**「馬主名」が最高重要度特徴**（"目利き馬主"の偏った勝率を1カテゴリで捕捉）。記事の実戦成績は的中34%/**回収率81%**で著者結論「**大幅な利益は困難**」。
- 競輪版検証（選手ID・県をカテゴリ追加 / scripts/explore_identity… 相当・/tmp/exp_identity.py）:
  | 特徴 | AUC全体 | AUC6車 | z≥1.0三連複ROI |
  |---|---|---|---|
  | baseline | 0.7720 | 0.7568 | 64.2%(19R) |
  | +選手ID | 0.7516 | 0.7486 | 72.0%(56R) |
  | +選手ID+県 | 0.7529 | 0.7473 | **92.0%**(49R) |
  - interim(13ヶ月)では選手IDは**全体AUC悪化=過学習**（選手あたり試行不足）。だが高選択性ゾーンROIは上昇(64→92%)。
  - **全期間データ(選手サンプル5-10倍)で再検証する価値あり**。現状 production FEATURE_COLS_WT には追加しない。
- **独立3事例(skyley 81% / note / 自作 65-85%)が同一の~80%天井に収束** = 公開情報＋控除率25%の構造的限界。「市場に勝てない」は固着ではなく頑健な実証的結論。残る希望は高選択性ゾーン局所エッジ(全期間で要検証)。

#### 利益条件マイニング（out-of-sample検証 / 2026-06-07）
方針確定（ユーザー）: 全レースで勝つのではなく**的中率×払戻で100%超になる条件を絞り込む**。過去の「場×戦略フィルター」過学習廃止の教訓から、**訓練期間で条件抽出→テスト期間(OOS)で検証**を徹底。
ツール: `scripts/mine_profitable_conditions_wt.py`（本番2軸流し三連複を各条件で層別、train/test別ROI算出）。
interim結果（train〜2026-02 / test 2026-02〜）:
- 全体 train85.0% / test74.7%。**頑健候補(train>105%&test>100%)=該当なし**。
- `n_riders==9` train121%→**test84%**、`n_riders==5` train107%→**test72%** = 訓練で勝てて見えた条件がOOSで崩壊＝**過学習の典型**（過去の失敗と同じ罠を再現確認）。
- 13ヶ月では規律ある利益条件は存在しない。希少高選択性条件(n≤6&ratio<1.3 はtest33Rのみ)は**サンプル過小で判定不能** → 全期間43ヶ月で数百Rに増やして本判定。
- 全期間で追加すべき条件: ライン構成(n_lines/隊列), prediction_mark とモデルの不一致, 会場×バンク, 期別/選手ID。すべてOOS検証必須。

#### 9/6/3分割バックテスト（2026-06-07 / scripts/backtest_963_wt.py）
学習9mo(2025-01〜09, 18353R) → 検証6mo(2025-10〜2026-03) → テスト3mo(2026-04〜06)。学習に頭数重み付け採用。
- AUC: 検証0.7686 / テスト0.7702 = **安定＝モデルは過学習せず良く汎化**。
- 全体ROI(2軸流し3点): 検証 三連複68.8%/三連単71.2%、テスト 三連複68.3%/三連単67.6%（極めて安定、全て~68%）。
- 条件別 検証→テスト: 全条件60〜79%、**頑健条件(検証>105%&テスト>100%)=該当なし**。
- **3重検証（手法×分割×外部事例）すべてで利益条件なしを確認**。AUC0.77安定が示すのは「予測力は本物だが控除率25%は超えられない」こと。
- 唯一の未確定: テスト3moでは希少条件が小サンプル(n≤6&gap12≥0.15=91R)。全期間43moで数百R化した時の最終判定のみ残る。

#### ★重大訂正: ksルート再監査で「市場は勝てる」と判明（2026-06-07 / scripts/reaudit_ks.py）
これまでの「公開情報では市場に勝てない（効率的市場）」結論は**誤り**だった。
- ks本番SS/S/A戦略を**真のOOS(2026-03〜06、閾値調整未使用)**で再監査:
  | 層 | 対象R | 的中率 | ROI | jackknife(最大払戻除外) |
  |---|---|---|---|---|
  | A | 291 | 51% | **238%** | **216%** |
  | S | 152 | 41% | 185% | 158% |
  | SS | 73 | 29% | 1321% | 1118% |
- **過学習でも分散(万車券依存)でもない** — 最大払戻を除いてもA216%/S158%。報告HO(2025-06〜)も同様(A228%/S189%)。
- **払戻データ検証**: ks(oddspark確定払戻) と wt(winticket最終オッズ×100) を38,390組合せで突合 → **中央値比1.000・99.5%が±10%以内・完全一致**。ROI差はデータ由来ではない。
- **ksが勝ちwtが負ける真因＝特徴量の質**:
  - ks=自作ローリング特徴(recent_win_rate_3m/6m, wr_trend, venue_win_rate, days_since_last_race, quinella_rate 等)を全履歴DBから独自計算 → 市場が見落とすsignalを抽出
  - wt=winticket画面表示の素朴な勝率(first_rate等)=市場全員が見る数字=オッズに織込済 → エッジ無し
- **結論**: 市場は**半効率的**。優れた特徴量エンジニアリングは公開情報からでも市場を上回れる。wtの「~80%天井」は効率的市場の壁ではなく弱い特徴量のアーティファクトだった。
- **次アクション**: ①ks rolling特徴の**リーク監査**(point-in-timeか。as-of計算の確認) ②wtルートにks流rolling特徴を移植し、winticketのライン情報と統合して更に上を狙う ③ks本番は既に有効＝継続運用。

#### リーク監査＋wtローリング特徴の効果検証（2026-06-07）
- **ks rolling特徴リーク監査: 合格**。`rolling_stats.py:86` `past = p_res[p_res["race_date"] < race_date]` ＝現レースより厳密に過去のみ。未来参照なし → ks 238%は本物確定。
- **wtにks流ローリング特徴を実装し検証**（`scripts/exp_wt_rolling.py`、point-in-time `closed='left'`）:
  追加特徴: win/top3/quin の3m/6m率, venue_wr, days_since, wr_trend。9/6/3分割OOS:
  | | testAUC | A-rank ROI | 合計 |
  |---|---|---|---|
  | baseline(30特徴) | 0.7695 | 70% | 71% |
  | +rolling | 0.7707 | **84%** | **83%** |
  - **test A-rank 70→84%(+14pp)、合計71→83%(+12pp)** OOS改善。方向性確定。
  - まだks238%未満の理由＝**wt履歴が2025-01〜と浅くlookback不足**。全期間(2022-12〜)収集完了で2-3年の深い履歴→ks級のローリング特徴品質に到達見込み＋wtはライン情報保有でks超え狙える。
- **収集完了後のTODO**: ①wtローリング特徴を本実装(precompute step + `FEATURE_COLS_WT`拡張 + 再学習) ②深い履歴でのA/S層OOS再評価 ③ライン特徴との統合最適化。`scripts/exp_wt_rolling.py`に計算ロジック保全済。

#### 収集ジョブ管理（再開方法）
- PID: `data/logs/collect_wt.pid` / ログパス: `data/logs/collect_wt.logpath` 記載
- 中断・PC再起動時の再開（収集はスキップ判定＋`INSERT OR REPLACE`で安全）:
  ```bash
  nohup .venv/bin/python3 -m src.cli.main collect-wt-range --from 2022-12 --to 2026-06 > data/logs/collect_wt_resume.log 2>&1 &
  ```
- 進捗確認: `.venv/bin/python3 -m src.cli.main status-wt`

#### 学習・バックテスト方針（ユーザー決定: 暫定→全期間の2段階）
**① 暫定**（直近13ヶ月 2025-06〜2026-06 が揃った時点 = 収集ログ `[14/43]` 到達後）:
```bash
.venv/bin/python3 -m src.cli.main train-wt --from 2025-06-01 --test-from 2026-03-01 --save-as lgbm_wt_interim
.venv/bin/python3 -m src.cli.main backtest-wt --from 2026-03-01 --model lgbm_wt_interim --max-riders 6 --min-gap12 0.06
```
**② 本番**（全期間収集完了 = 収集ログ `All done` 後）:
```bash
.venv/bin/python3 -m src.cli.main train-wt --from 2023-01-01 --test-from 2025-06-01 --save-as lgbm_wt_v1
.venv/bin/python3 -m src.cli.main backtest-wt --from 2025-06-01 --model lgbm_wt_v1 --max-riders 6 --min-gap12 0.06
```

#### 既知の軽微課題（収集後に対応可）
- ガールズ戦のクラスが `cls4`/`cls1` で未マッピング（`_CLASS_MAP` に L級が無く `player_class_enc=-1`）。全体の約8%。`winticket.py _CLASS_MAP` と `feature_wt._CLASS_MAP` に L級マッピング追加を検討。

---

### (旧/参考) keirin-station 戦略（ロールバック用・ホールドアウト=上限値）

ks `wave-picks`（lgbm_v6）のホールドアウトROI（2025-06〜2026-02）: SS 3944% / S 158% / A 228%。**最終データbacktest=上限値**（実運用は修正後1週間49%）。閾値定義は wt と共通（gap12/ratio）。

> （アーカイブここまで。以下は現行情報）

---

## 確認コマンド（最新・winticket 本番）

```bash
source .venv/bin/activate

# 状況・予想・成績
python -m src.cli.main status-wt
python -m src.cli.main wave-picks-wt --date $(date +%F) --gami-skip-odds 3.0 --b-rank-odds 5.0
python scripts/notify_results_wt.py $(date -v-1d +%F)          # 前日成績（mac）

# バックテスト/検証
python -m src.cli.main backtest-wt --from 2026-03-01 --tiered
python scripts/backtest_monthly_rank_wt.py                      # 月×ランク（ガミ3段階込み）
python scripts/snapshot_morning_odds_wt.py --report            # 朝→最終オッズ ドリフト
```

cron: `daily_picks_wt.sh`(7:00) / `weekly_retrain_wt.sh`(日23:30)。

---

## 重要な設計メモ

### winticket ルートの固有特徴量（keirin-station にはない）

- **並び情報**: `line_size`, `line_pos`, `is_line_leader`, `n_lines` — 構造化済みJSON
- **セクター回数**: `s_count`（先行）, `h_count`（ホーム）, `b_count`（バック）
- **上がり戦術率**: `ex_spurt_pct`, `ex_thrust_pct`
- **winticket AI印**: `prediction_mark`（0=なし / 1=本命 / 2=対抗 / 3=単穴 / 4=連下）
- **事前オッズ**: `wt_odds` テーブル（3連複/3連単/2車複 etc.）

### オッズの扱い方針

> オッズは市場原理の結果。AI予想後の購入判断・低オッズ時の見直しに使う。

- `wave-picks-wt` はオッズをモデル特徴量に**含めない**
- 予想生成後に `wt_odds` テーブルから取得・表示
- `--min-trio-odds` フラグで低オッズ組み合わせを自動フィルタ

### バンクロール管理（参考）

- 推奨最低資金: 100,000円（日次予算 残高の5〜8%）
- 30,000円スタートは破産リスクが高い
- 1日投資目安: SS/S/A 合計 約1,800〜2,400円（300円×6〜8件）
