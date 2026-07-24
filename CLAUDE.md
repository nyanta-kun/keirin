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
docs/bet-structure-guide.md            # 買い目戦略（旧体系の歴史的記録。現行は CLAUDE.md ランク体系参照）
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

## 現行ランク体系（2026-07-21〜・実精算方式・**S1/SS/S の3ペーパーランク（S2/S3全廃・S4はSS/Sへ再編）**）

**2026-07-21 同日中の再編**: S2(7PLUS_U)/S3(7PLUS_M)は対象レース数・的中率・期待値の観点で
継続困難と判断し全廃（過去行は`picks_history_u_archive`/`picks_history_m_archive`へ退避）。
S4(SEVEN_S4)は今後の予想データのベースと位置づけ、軸2車がWINTICKET公式◎◯と重なるかで
**SS**（重なり0・全く重ならない）と**S**（重なり1・片方だけ重なる）の2ランクに再編して表示
（内部rankは`SEVEN_S4`のまま・`gate_label`列で"SS"/"S"を区別）。S1は現状維持で継続検討中
（払戻の大きさが予想購入者へのアピール材料になりうるため、ブラッシュアップの方向性を継続検討）。

**2026-07-17 再設計確定**: 正規プロトコル（学習〜2025-03-31／検証=2025-04-01〜2026-03-31 の1年で条件選択
／テスト=2026-04-01〜07-15 で1回評価・モデル `lgbm_wt_val25`）による全ランク再検証の結果、
**合格は S2（現行条件のまま）と S3（新定義）のみ**。S1（6車三連単）・A（一致波乱二連単）は
検証ROI100%超の条件が存在せず全廃（新S1候補スイープ=適応型2車軸トリオ/m1 1着固定三連単も
検証ROI≥95%のセルなしで全滅）。
**2026-07-16 指数改定: 競走得点トレンド4特徴を追加（FEATURE_COLS_WT 40→44・全モデル再学習・バックフィル再構築済み）**

- **S1（新設計・win軸1着固定・2026-07-19導入、閾値は同日07-19に0.15→0.22、07-22に0.15へ再変更）** = `SEVEN_S1`（suffix `#7S1`・**ペーパートレード**）: 軸=1着専用モデル(`lgbm_wt_win`)のレース内1位（固定）× 相手=3着内モデルで軸を除いた上位2頭(p1,p2) × top3_gap(p1-p2の3着内確率差)≥`S1W_TOP3_GAP_MIN` × 三連単 軸→p1→p2, 軸→p2→p1 の2点流し（目オッズ下限なし）。旧S1（7車三連複7PLUS_R）・新S1（6車三連単SIX_S1）はいずれも全廃されたが「win軸固定×3着内モデル相手選定」は未検証だった構造。正規プロトコル: top3_gap閾値0.08〜0.20で検証・テストとも単調に改善（0.15で検証145.8%(n=9949・約27R/日)→テスト135.3%(n=2851)）。07-19同日中にユーザー要望（母数を1日15R以下へ絞り的中率向上）でスイープを延長し0.22へ引き上げ（検証171.6%(15.2R/日・的中18.1%)→テスト146.0%(15.3R/日・的中18.2%)）。旧win_rank/gap12モデルと同型のリーク（下記参照）を修正した四半期walk-forwardモデルで全期間(`scripts/rebuild_s1_walkforward.py`)を再構築した結果、honest全期間実績（0.22時点・2024-01-01〜2026-07-18）= 14,363R・約15.3R/日・的中17.3%・ROI**123.0%**。S2/S3との重複4.3%とほぼ独立。払戻分布は少数の高額配当に偏る。
  **【2026-07-22 再変更・高配当取りこぼし防止】** 万車券(配当≥10,000円)分析（`exp_s1_manshaken_analysis.py`）で「top3_gapを上げても万車券は増えない」と判明（万車券のtop3_gap平均は的中全体よりむしろ低い）。一方「軸の単勝勝率が低いほど高配当」の傾向を確認し、`exp_s1_20x_filter_design.py`で軸勝率フィルターを評価。ユーザー判断で**`S1W_TOP3_GAP_MIN`を0.22→0.15へ復帰**し、**`S1W_AXIS_WIN_PROB_MAX=0.50`新設**（軸の単勝勝率がこれを超えるレースを除外＝本命決着を回避）。honest全期間(軸勝率≤50%フィルター単体): n=13,510(53.5%)・的中率10.7%(元16.2%)・ROI146.3%(元120.3%)・20倍以上再現率65.9%・万車券再現率84.0%。**S1は「軸=win1位固定」のため的中率と高配当は構造的トレードオフ**（的中率を保ったまま高配当だけ追加で拾うことは不可）。コードは`src/strategy_wt.py`/`src/cli/main.py`/`scripts/backfill_s1w_rank_wt.py`に反映済み。
  **【2026-07-22 過去分honest再構築 完了】** VPS PG直接参照で全期間(2024-01-01〜2026-07-22)を四半期ごとに分割実行し再構築（picks_history本番反映済み）: 13,489R・約14.4R/日・的中10.6%・ROI143.3%（旧0.22時点比: 14,363R→13,489R・的中17.3%→10.6%・ROI123.0%→143.3%）。四半期別ROIは81.4%〜263.2%とばらつくが全期間で黒字幅拡大。再構築の過程で**VPS PostgreSQLの`wt_odds`が2026-06-01以降のみのミラーで2024〜2026-05分が丸ごと欠落している**ことが判明（`wt_races`はVPS PG側が全期間完全なのに対し`wt_odds`は逆にローカルMac SQLiteのみ全期間完全という非対称構成だった）。ローカルSQLite（2022-12-01〜2026-07-10・3,469万件）から不足分（2024-01-01〜2026-05-31・2,332万件）をCSV export→scp→`\copy`+`ON CONFLICT DO NOTHING`でVPS PGへ一括移植し解消（VPS disk 60G→65G/99G・所要ディスク+5GB、メモリ影響なし）。**これによりVPS PGのみでwt_oddsに依存する過去分honest再構築が可能になった**（S1は本件で実施確認済み。S2/S3/S4は全廃済みだが今後同型の再構築が必要な場面があればこの移植により`rebuild_*_walkforward.py`のローカルSQLite依存を回避できる）。実行時の注意点: `rebuild_s1_walkforward.py`本体は「ローカルSQLiteが完全な履歴を持つ」旧前提でKEIRIN_DB_URLを読み取り時にpopする設計のままなので、VPS PG一本化で読む場合は環境変数をpopしない別スクリプトが必要（本件では四半期ごとに分割した単発スクリプトを都度実行）。過去分は `scripts/rebuild_s1_walkforward.py`（旧`backfill_s1w_rank_wt.py --wipe`は四半期対応前の単一モデル版・リーク混入のため非推奨）。この直後、ローカルSQLite（`data/keirin.db`）自体を廃止（Mac対話セッションも`~/.zshrc`の`KEIRIN_DB_URL`でVPS PGをデフォルト参照するよう変更・VPS PGが名実ともに唯一のデータソースに）。
  **【2026-07-22 軸級班denyフィルター追加・高配当特化】** ユーザー要望「高的中率を目指すが高配当は捨てない（低配当になりそうなレースを省き高配当の的中率を上げる）」でセグメント別分析（`scripts/exp_s1_segment_deny_analysis.py`・正規プロトコル: train+val〜2026-03-31で選定→test 2026-04-01〜07-22で一度だけ評価）を実施。venue_id/grade/distance/line構成等のうち、**軸選手の級班（player_class）が最も明確なシグナル**: 各グレード内の最上位クラス（S1級/A1級）が軸の場合、的中率は同水準のまま配当が低くなりやすい（train+val: ROI138.5%→173.5%・test: ROI178.9%→246.2%・いずれも的中率は完全に不変）。5万円以上の高配当payoutは全期間7件中6件(85.7%)が残存・カットは1件のみ（払戻額が大きいほど残存率が上がる=高配当ほどこのフィルターで保護される）。`S1W_DENY_AXIS_CLASS={"S1","A1"}`として`s1w_gate()`に統合（`src/strategy_wt.py`/`src/cli/main.py`/`scripts/backfill_s1w_rank_wt.py`）。honest全期間再構築（2024-01-01〜2026-07-22）: **6,426R・約6.9R/日・的中10.6%（変化なし）・ROI182.5%**（旧143.3%から改善）。母数はほぼ半減（13,489→6,426R）するが「SS/Sが別途ある」との判断で許容。過去分は本番picks_history反映済み・kiseki help表示も更新要（次回確認）。
- **S4（単勝×複勝指数トップ3重なり軸×波乱度選出・2026-07-21導入・同日中にWT◎◯重なり考慮版へ改良）** = `SEVEN_S4`（suffix `#7S4`・**ペーパートレード**）: 軸2車 = `pred_win_pct`（単勝指数）上位3 ∩ `pred_top3_pct`（複勝指数）上位3 の重なり車から選定（重なり>=2なら`pred_top3_pct`上位2、重なり==1ならその1車+残りの`pred_top3_pct`最上位。重なり0は対象外・実データで58,616中1件のみ）。波乱度指数 = 軸2車の`pred_top3_pct`合計（`axis_sum`）。**レース全体のエントロピー（拮抗度）で絞るとROIが悪化する（絞り込みなし85.7%→73.5%）ことを確認し不採用**。当初はaxis_sum昇順で日次上位`S4_DAILY_TOP_N`件を採用する方式で、N=15→10へ変更後のhonest全期間実績（2024-01-01〜2026-07-20）= 9,220R・10R/日・的中35.2%・ROI128.1%だった。
  **同日中の追加検証（ユーザー仮説）**: 軸2車がWINTICKET公式予想の◎◯（`prediction_mark`∈{1,2}）と重なる場合に期待値が下がるかを`exp_s4_wt_axis_overlap.py`で検証（honest全期間・四半期walk-forwardモデル）。重なり数別に日次Top10選出内訳を見ると、的中率はほぼ横ばい（33〜37%）なのにROIが重なり数に応じて単調悪化（重なり0=ROI408.1%／重なり1=148.7%／重なり2=完全一致=**75.7%・赤字**）と判明。コンセンサスピック（WT予想と完全一致）は市場に織り込まれ払戻が縮む構造。
  → **選出方式を変更**（`strategy_wt.s4_wt_overlap_n()` / `s4_daily_select()`）: 重なり0（WT◎◯と全く重ならない）は該当があれば無条件で全件採用（本数上限なし）、重なり1（片方だけ重なる）はaxis_sum昇順で固定`S4_DAILY_TOP_N`=10件、重なり2（完全一致）・WTマーク欠損は完全除外。1日あたりの採用本数は重なり0の発生数に応じて可変。honest全期間実績（新方式）= **9,927R・10.77R/日・的中36.3%・ROI131.3%**（旧方式128.1%から改善）。内訳: 重なり0(943R)的中39.4%/ROI232.8%・重なり1(8984R)的中36.0%/ROI120.6%。過去分は `scripts/rebuild_s4_walkforward.py`（`scripts/backfill_s4_rank_wt.py`が新方式に対応済み）
  **【2026-07-21 同日中・表示ランク再編】** S4は今後の予想データのベースと位置づけ、ユーザー指示によりWeb/Discord/サマリー/グラフ全てで内部区分をそのまま表示ランクとする: 重なり0→**SS**（ROI232.8%）、重なり1→**S**（ROI120.6%）。内部rank `SEVEN_S4` はそのままで、`picks_history.gate_label`列に"SS"/"S"を格納して区別する（新規カラム不要・既存の`gate_label`（元はS3のOR gate内訳用）を流用）。`notify_prerace_wt.py`の`_insert_s4_pick`/`_build_s4_message`が対応済み（Discord通知の見出しも"SS"/"S"表示）。
  **【2026-07-22・朝夕統合再選出への再設計】** 上記の「重なり1は固定`S4_DAILY_TOP_N`件」は、朝(`daily_picks_wt.sh`)と夕(`evening_picks_wt.sh`)が別プロセスで独立にこの上限を適用していたため、1日最大20件になるバグと化していた（発覚の経緯は`keirin_s4_gate_label_bug_and_candidate_visibility_2026_07_22`）。さらにhonest全期間検証で「朝の部(19時未満発走)だけでS候補が10件に達する日が57.2%」と判明し、朝が先着で夜の優良候補を取りこぼす構造的懸念が確認された。**`S4_HALF_CAP`=6を新設し、朝夕それぞれの一次選出を6件に縮小**。夕方バッチの最後に`scripts/s4_evening_reselect.py`を実行し、朝夜の生候補（`_s4_raw_candidates.json`/`_night_s4_raw_candidates.json`に新たに永続化）を統合してaxis_sumランキングを組み直す（`strategy_wt.s4_evening_reselect()`）。ただし既に買い判定済み（`bet_amount>0`）のレースは実購入を取り消せないため維持し、未判定分だけ日次合計`S4_DAILY_TOP_N`(10)件へトリムする。honest全期間バックテスト: 現行(朝夕別選出)ROI117.7%(理論上限との選出一致率76.5%) → **新設計ROI120.8%(理論上限120.6%とほぼ同等・一致率89.5%)**。過去分再構築(`scripts/backfill_s4_rank_wt.py`)は1日分データを最初から統合済みのため影響を受けない（`cap=S4_DAILY_TOP_N`を明示指定して従来通りの理論上限相当を再現）。詳細: `keirin_s4_evening_reselect_2026_07_22`
  **【関連: 「非」バッジ再発バグ】** `notify_results_wt.py`の毎時採点処理がpicks_history行をDELETE+`INSERT OR REPLACE`で再作成する際、列リストに`gate_label`が含まれておらず、対象レースが再採点されるたびにgate_labelがNULLに巻き戻り「非」表示になるバグがあった（2026-07-21発見・修正済み）。picks_historyに新規列を追加する際は、この`INSERT OR REPLACE`列リスト（S1/S4/旧U/M共通）に必ず追加すること。あわせてS4は候補時点（買い判定成立前）から`write_candidates_wt.py`がプレースホルダ行を書き込むようになり、Webで候補になった時点からS/SSバッジが表示される（見送りは的中したかを`miwokuri=True・bet=0`のまま参考記録）。詳細: `keirin_s4_gate_label_bug_and_candidate_visibility_2026_07_22`
  **【2026-07-23・SS+観察サブランク新設】** S1で発見した軸級班denyフィルターがS4のSS/Sにも効くか検証（`scripts/exp_s4_axis_class_deny_analysis.py`・正規プロトコル）。**単純な母集団相関ではS/SSとも改善に見えたが、Sは日次axis_sum上位10件の「枠付き相対選出」のため、格上軸候補を除外すると別候補が繰り上がり、実際にシミュレーションすると悪化する**（train+val ROI116.3%→111.5%・test 132.6%→119.2%、両期間で一貫して悪化）。一方SSは無制限採用（枠なし）のため単純に足切りされるだけで、繰り上がり効果を考慮しても改善が残る（train+val ROI222.3%→351.6%・全期間237.1%→362.2%、的中率は不変〜微増）。**結論: Sには適用せず、SS内の軸格上非該当サブセットのみ新表示ランク"SS+"として観察する**（ユーザー判断・実際の買い目・購入対象は変更しない表示分岐のみ）。
  実装: `strategy_wt.s4_gate_label(wt_overlap_n, axis1_class, axis2_class)`（軸級班情報が両方揃いいずれもS1/A1でなければ"SS+"、それ以外は従来通り"SS"、重なり1は"S"）に集約。`src/cli/main.py`（S4候補生成時にaxis1_class/axis2_class追加）・`scripts/notify_prerace_wt.py`（`_process_s4_candidates`/`_build_s4_message`）・`scripts/write_candidates_wt.py`（候補時点表示）・`scripts/backfill_s4_rank_wt.py`（過去分再構築）・`scripts/notify_results_wt.py`（日次結果通知のSS+/SS/S 3分割）・`scripts/save_model_eval.py`（`PAPER_RANKS`にSS+追加）に反映。既存SS行（951件）はSQL UPDATEでgate_label='SS+'/'SS'へ即時分割済み（367件がSS+へ・honest実績: SS+ ROI360.9%(367件)・SS ROI158.9%(584件)・S ROI119.9%(9104件)）。
  **【2026-07-23完了】** kiseki側backend `_display_rank`/frontend `RANK_ORDER`/`RANK_BADGE_STYLE`は同日中に対応済みだったが、`RankBadge`用の`RANK_STYLE`マップ（レース詳細の個別バッジ表示）だけ更新漏れで「非」表示になっていたバグを発見・修正（`keirin_ss_plus_display_fixes_2026_07_23`）。あわせて`/keirin/help`の`RANKS`カード一覧にSS+カードを追加し表示順をSS+/SS/S/S1に変更、Web指数ラベルを「単勝指数/複勝指数/指数」→「単勝率/複勝率/競走得点」へ改称（表示値の実態を反映）。
- **旧新S1（SIX_S1・6車三連単・2026-07-17 全廃）**: 3独立窓では110/103/113%だったが正規プロトコルの1年検証で最良70.3%・100%超なし（「直近だけ良い」レジーム依存を検出）。6車全域・9車・新S1候補も全滅 → 全廃。`#6S1` 行は `picks_history_r_archive` へ退避（`scripts/archive_s1_a_abolition_wt.py`）
- **A（7PLUS_A・2026-07-17 全廃）**: 正規プロトコルで検証最良88.5-94.2%・100%超なし → 全廃。`#7A` 行は `picks_history_a_archive` へ退避（同上スクリプト）。旧・買い目カット方式Aランク（〜2026-06-19）の行も同テーブルに退避済み
- **旧S1（7PLUS_R・7車三連複・2026-07-16 全廃）**: 検証期間ROI 67.3%・代替条件の全探索で黒字なし。過去行（7PLUS_R/7PLUS_CAND/7PLUS_SS/7PLUS_S）は `picks_history_r_archive` へ退避。wave-picks の SS txtセクション・#CAND 書き込み・ガミ判定は停止済み（ss_policy 等は互換のため残置）
- **S2（旧U・7PLUS_U・2026-07-21 全廃）**: 波乱見込み×穴×同ライン「逃」相方の三連複2車軸流し。廃止直前にmto閾値を4.3→4.5へ厳選したが、honest全期間再構築（`scripts/rebuild_s2_walkforward.py`・四半期walk-forwardモデル）で確認したところ4.3=ROI81.6%(1251R)→4.5=ROI84.8%(1155R)と全期間では依然として損失圏内（2024〜2025年前半が40-70%台で低迷）。対象レース数・的中率・期待値の観点で継続困難と判断し全廃。過去行（1155件）は `picks_history_u_archive` へ退避（`scripts/archive_u_m_abolition_wt.py`）。judge_u/`_process_u_candidates`等のロジックは過去日再採点・分析スクリプト互換のため残置（呼び出し元のみ停止）
- **S3（旧M・7PLUS_M・2026-07-21 全廃）**: ◎不一致×軸信頼ゲートの三連複2車軸流し。廃止直前にwin_rank単独ゲート化+目≥20倍で honest全期間ROIを95.9%→120.4%(801R)まで改善させていたが、S2と合わせて対象レース数・的中率・期待値の観点で継続困難と判断し全廃（過去の閾値変遷・リーク発覚の経緯は本ファイルのgit履歴・メモリ`keirin_composite_ratio_gate`参照）。過去行（801件）は `picks_history_m_archive` へ退避（`scripts/archive_u_m_abolition_wt.py`）。judge_m/`_process_m_candidates`/`m_axis_gate`等のロジックは過去日再採点・分析スクリプト互換のため残置（呼び出し元のみ停止）
- S1/SS/S は live 100R以上で採否判定（実賭け昇格 or 廃止。月次判定は分散的に禁物・40R全外れ月も想定内）。詳細はメモリ `keirin_s1_redesign_sweep` / `keirin_s1_win_axis_paper` / `keirin_s1_threshold_axis_win_prob_2026_07_22`（S1閾値0.15復帰+軸勝率ゲート）/ `keirin_picks_history_data_loss_2026_07_20`（S4）/ `keirin_s2_s3_tightening_2026_07_21`（S2/S3全廃の経緯）/ `keirin_s4_wt_overlap_selection_2026_07_21`（S4→SS/S再編）/ `keirin_s4_evening_reselect_2026_07_22`（S4朝夕統合再選出・現行設計）/ `keirin_s4_gate_label_bug_and_candidate_visibility_2026_07_22`（非バッジ再発バグ・候補可視化機能）
- **廃止済みランク**: S/S+（`7PLUS_ST`/`7PLUS_STP`・三連単1着固定F）は 2026-07-15 に全廃・過去分もDB削除（`keirin.picks_history_st_archive` に退避）。SO≥8フィルタ・旧≤6車 SS/S/A/B・ワイドも廃止済み。旧ドキュメント・メモに残る記載は無効
- **実精算方式（2026-07-15〜）**: バックテスト・採点とも、指数ランキング＝発走前のオッズ盤面掲載車（欠車除く・落車失格含む）、落車失格絡みの買い目＝外れ計上（返還しない）、欠車のみ返還。完走者ランキングの旧方式は約2-4倍過大で全面廃止
- 見送り=miwokuri=TRUE。**実賭けランクは現在なし**（全ランクペーパー・名目賭金）。Webサマリーのトップラインは `rank IN ('SEVEN_S1','SEVEN_S4')`（S4は`gate_label`でSS/Sに分割表示）の名目合算
- `prerace_decisions_{date}.json` が採点/Web/サマリー/Discord の正本（15分前判定を事後変更しない）。キーは S1=`{rk}#S1` / S4(SS/S)=`{rk}#S4`（廃止済みのS2=`{rk}#U`/S3=`{rk}#M`キーは過去日分のみ存在）
- **落車失格レースの学習除外は棄却**（除外するとS1テスト122.8→87.9%に劣化した検証あり。落車の事前予測情報は不存在＝事後情報での母集団選別になる）。`WT_EXCLUDE_DNF_RACES=1` のオプトインのみ残置

## Web指数表示（単勝率・複勝率・競走得点・2026-07-19導入、2026-07-23ラベル変更）

- kiseki側 `/keirin` の出走表（EntryTable）に、既存の**競走得点**（`race_point`）に加えて
  **単勝率**（1着専用モデル`lgbm_wt_win`の予測確率）・**複勝率**（3着内モデルの予測確率）を
  単→複→競走得点の順で表示する（2026-07-23: 「単勝指数/複勝指数/指数」から改称。
  表示値の実態＝AI予測確率／公式得点であることを明確化するため）
- `wt_entries.pred_win_pct` / `pred_top3_pct`（%スケール・小数1位）に格納。`wave-picks-wt`実行時に
  `pred_prob`/`pred_win`算出直後（候補選定の前）に全出走馬分をUPDATEする（`src/cli/main.py`）
- PG側は kiseki alembic `n0p1q2r3s4t5`で追加。SQLite側は`src/database.py::migrate_db()`
- 過去分（2024-01-01〜）は `scripts/backfill_index_pct_wt.py` で四半期walk-forwardモデルを使い
  リークなしで一括反映済み（491,582/705,079件・2026-07-19実施）

**【重要・設計原則】`wt_entries.race_point`を表示専用の値で上書きしてはならない**。
2026-06-18のcommitで、この列（`feature_wt.py`の`score_rank`/`score_mean`/`score_std`/
`score_z`という実モデル学習特徴量の入力）を`pred_prob_pct`（AI予測確率）で上書きする
処理が`wave_picks_wt`内に混入し、`weekly_retrain_wt.sh`（毎週日曜23:30）が汚染された
race_pointを特徴量として取り込み続けるという自己参照汚染が約5週間（2026-06-18〜07-23）
放置されていた（2026-07-19導入の`pred_top3_pct`が既に同じ表示目的を汚染なく満たして
おり、この上書き自体が既に不要だった）。2026-07-23、上書きコード削除・
race_point=0.0（デビュー戦等未点数選手・欠損扱いへ修正）・健全性チェック+自動リトライ
（`scripts/check_race_point_sanity.py`）・汚染期間の生データ再取得・汚染モデル破棄・
全期間再学習・S1/S4のtailウィンドウ(2026-04-13〜)再構築まで完了。
**教訓**: モデル特徴量として使う列に対して「表示のための書き込み」を絶対に行わない。
表示専用の値は必ず別カラム（`pred_win_pct`/`pred_top3_pct`パターン）を新設すること。
新しい特徴量列やUPDATE文を追加する際は`grep "UPDATE wt_entries SET"`で他の書き込み
経路と衝突していないか必ず確認する。詳細はメモリ`keirin_race_point_feature_leak_2026_07_23`。

## Mac / VPS データアーキテクチャ（2026-07-22 VPS PG一本化完了・確定）

**VPS PostgreSQL（`hrdb`.`keirin`スキーマ）が唯一の本番データソース**。
VPS（`/home/ysuzuki/keirin`・GitHubの本リポジトリと同一cloneが常駐）が
daily_picks_wt.sh/evening_picks_wt.sh/notify_prerace_wt.py（毎分・8-23時）等の
cronを自前で実行し、日次データ収集・ライブ判定・通知を独立して行っている。
`wt_races`はVPS PGで2022-12-01〜当日まで欠損なし。

**ローカルMacのSQLite（`data/keirin.db`）は2026-07-22に正式に廃止した**。
Mac対話シェルも `~/.zshrc` に `KEIRIN_DB_URL=postgresql://...@sekito-stable.com:5432/hrdb`
をグローバル export するよう変更し、対話セッション・crontab（週次再学習
`weekly_retrain_wt.sh`含む）とも常にVPS PGを参照する。**`get_connection()`
（`src/database.py`）は2026-07-24、`KEIRIN_DB_URL`未設定時にローカルSQLiteへ
無言フォールバックする実装から、`RuntimeError`を送出する実装へ変更済み**
（テストのみ`KEIRIN_ALLOW_SQLITE_FALLBACK=1`で明示的に許可・`tests/conftest.py`
が自動設定）。`notify_results_wt.py`が持っていた旧Mac/VPS二重モード判定
（ローカルSQLiteの鮮度をヒューリスティックで検知し書き込み先を自動切替する
仕組み）も同日に削除済み。

**背景（2026-07-21の近未遂インシデント）**: 上記の一本化前は、crontab経由の
実行は`KEIRIN_DB_URL`を引き継ぐが対話的なターミナル/SSHセッションは引き継がない
ため、`KEIRIN_DB_URL`を明示的にexportしない限り`get_connection()`がデフォルトで
ローカルSQLite（更新停止済みの不完全なコピー）を見てしまう罠があった。この
思い込みでVPS本番データを誤ってwipeしかけたインシデント寸前が発生し、これが
一本化・SQLite廃止・無言フォールバック廃止に至った直接のきっかけ。

**運用ルール**:
- VPSは**メモリ1.9GB（空き実測101MB・buff/cache込みでも1.1GB程度）と限られており、
  ライブ本番処理と同居している**。重いバックテスト・モデル再学習等の計算処理は
  引き続きMacで行い、VPSには「完了した結果の書き込み」または「軽量な直接クエリ」
  のみを行うこと。VPS上でのフル学習・大規模walk-forward計算は避ける（PGは
  VPS上で稼働しているため、Mac側からの重いクエリもVPS DBサーバー自体には
  相応の負荷がかかる点は変わらず留意）。
- `rebuild_*_walkforward.py` 系スクリプトのコメントに残る「ローカルSQLite=
  完全な履歴」「PG側は直近数ヶ月のみのミラー」等の記述は2026-06-20以前の
  旧アーキテクチャ前提で、現在は不正確。読む・改修する際は鵜呑みにしないこと。
- keirinスクリプトをMacで修正した場合、VPS本番に反映するには必ず`git push`→
  `ssh sekito "cd /home/ysuzuki/keirin && git pull"`まで実施すること。VPSの
  cronはVPS上の別checkoutを実行するため、Macでの編集だけでは本番挙動は変わらない。
- **【2026-07-22追記】`wt_odds`はVPS PG側こそ不完全だった**（`wt_races`とは逆パターン）。
  VPS PGの`wt_odds`は2026-06-01以降のミラーのみで2024〜2026-05分が丸ごと欠落しており、
  ローカルMac SQLiteだけが2022-12-01〜の全履歴を持っていた。S1のhonest再構築
  （2026-07-22）時にこの非対称に気づき、不足分2,332万件をCSV export→scp→VPS上で
  `\copy`+`ON CONFLICT DO NOTHING`により一括移植して解消済み（VPS disk使用量+5GB・
  メモリ影響なし）。**現在はVPS PGの`wt_odds`も2022-12-01〜今日まで完全**なので、
  今後`wt_odds`に依存する過去分honest再構築はVPS PGのみで完結できる（ローカル
  SQLiteの`wt_races`停止（7/10）による制約は依然残るが、`wt_odds`自体はもう
  ボトルネックではない）。他のテーブル（`wt_entries`等）にも同様の非対称ミラー
  範囲が隠れている可能性があるため、大規模rebuild前は対象テーブルのVPS PG側
  カバレッジを個別に確認すること（`wt_races`が完全だからと言って他表も完全とは
  限らない、が今回の教訓）。

## スキーマ管理ルール（picks_history 等 keirin スキーマ）

- **DDL は「kiseki 側 alembic」と「本リポジトリ src/database.py::migrate_db()（SQLite用）」の両方に必ず追加する**
  （gap23 列が両方から漏れて本番 PG に手動ALTERだけで存在する「幽霊カラム」になった事故あり → 2026-07-12 に両側へ正式化済み: kiseki alembic `j6k7l8m9n0p1` / migrate_db）
- **gap カラムのスケール**: gap12 / gap34 = 0-1 スケール、**gap23 のみ pt（%ポイント・×100済み）**。歴史的経緯によるもので変更不可。読み書き時に注意
- 閾値定数（GAMI_THRESHOLD=7.0 等）は `src/cli/main.py` / `scripts/notify_prerace_wt.py` / `scripts/write_candidates_wt.py` に多重定義。**変更時は3ファイル + kiseki フロント（page.tsx）を必ず grep して揃える**
- **新しいテーブルを追加したら `src/database.py::_pg_translate()` の keirin スキーマ自動付与regex（2箇所: INSERT系/通常SQL系）にもテーブル名を必ず追加する**。INSERT OR REPLACE/IGNORE文はテーブル名を直接展開するため regex 漏れでも動くが、素のSELECT/UPDATE/DELETEはこのregexだけが唯一のスキーマ付与経路なので、漏れると `relation "xxx" does not exist` で本番クラッシュする（2026-07-24発覚: `netkeirin_submissions`追加時にregexへの追加を忘れ、`_already_submitted()`のSELECTが機能追加以来一度もschema解決できず、netkeirin入稿が導入(2026-07-23)以来一度も成功していなかった。INSERT経路は正しく動いていたため気づかれなかった）
