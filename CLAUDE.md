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
src/preprocessing/feature_wt.py        # FEATURE_COLS_WT（46特徴・rolling統合。2026-07-31にex_spurt_pct/ex_thrust_pctをtrain/serve skewのため除外し48→46。H2H対戦表特徴は実装のみでFEATURE_COLS_WT非採用・下記参照）・build_features_wt() / add_rolling_features_wt()
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

## 現行ランク体系（2026-08-02〜・実精算方式・**7S/7A/9S/9A の4ペーパーランク**）

> **【2026-08-02・7SS全廃】** `RANK_7SS`（波乱軸選出・穴レース検知）を全廃した
> （ユーザー判断・commit `7048db5`）。live実績が picks_history 全期間
> **n=16,298・ROI 73.5%**、2026年の月次も 94.4/61.0/56.3/61.1/69.3/70.2/60.3% と
> 1月以外すべて70%以下で控除率75%を下回り続け、有効な推奨として成立していなかった
> ため。候補生成・ライブ判定・採点・Discord通知・Web表示・DB実績（16,298行）を
> すべて停止／削除した（退避: Mac側 `data/backup/picks_history_rank_7ss_before_abolition_20260802.csv`）。
> 判定ロジック（`rank_7ss_*` / `backfill_7ss_rank_wt.py` / `exp_7ss_*.py`）は
> 将来の再設定に備えて残置してある。再開手順はメモリ
> `keirin_7ss_abolition_2026_08_02` 参照。
> **以降の本節の記述で 7SS/9SS に触れている箇所は、いずれも廃止前の履歴として読むこと。**

### ランク名体系化（2026-07-31・commit `f31f84b`）— 内部rank名・suffixの正本

**表示ラベル（Web/Discord/netkeirinの見え方）は一切変更していない。** 内部rank名と
suffixだけを表示ラベル基準に体系化した。以降のコード・ドキュメントで参照すべき
正本は以下の対応表（**本節より下の履歴記述は意図的に改名前の表記のまま残して
いる**ため、混同しないよう本節を先に参照すること）:

| 内部rank（新） | suffix（新） | 表示ラベル | 内部rank（旧） | suffix（旧） |
|---|---|---|---|---|
| `RANK_7S`  | `#7S`  | 7S  | `SEVEN_S7` | `#7S7` |
| `RANK_7A`  | `#7A`  | 7A  | `SEVEN_7A` | `#7A`（変更なし） |
| ~~`RANK_7SS`~~ | ~~`#7SS`~~ | ~~7SS~~ | `SEVEN_SS` | `#7SS`（**2026-08-02 全廃**） |
| `RANK_9S`  | `#9S`  | 9S  | `NINE_S9`  | `#9S9` |
| `RANK_9A`  | `#9A`  | 9A  | `NINE_9A`  | `#9A`（変更なし） |

**命名規則**: 内部rank = `RANK_` + 表示ラベル、suffix = `#` + 表示ラベル
（3表現が完全に1対1で覚えるべき対応がゼロになる）。suffixの実質変更は
`#7S7`→`#7S`・`#9S9`→`#9S`の2件のみ（元々`SEVEN_S7`/`NINE_S9`だけ表示ラベル
と非対称だった不揃いの是正そのもの）。

定数・関数・ファイル名も同基準へ統一済み:
- 定数: `S7_*`→`RANK_7S_*` / `S9_*`→`RANK_9S_*` / `S7A_STAKE`→`RANK_7A_STAKE` /
  `S9A_STAKE`→`RANK_9A_STAKE` / `SEVENSS_*`→`RANK_7SS_*`
- 関数: `s7_daily_select`→`rank_7s_daily_select`・`s7_gate_label`→
  `rank_7s_gate_label`・`s7_evening_reselect`→`rank_7s_evening_reselect`・
  `sevenss_score`/`field_features`/`select_axis`→`rank_7ss_*` 等。
  `notify_prerace_wt.py`は`judge_s7`→`judge_rank_7s`・`_process_s7_candidates`→
  `_process_rank_7s_candidates`等`_load_`/`_insert_`/`_build_`系17関数を含む
- ファイル名: `backfill_s7/s7a/s9/s9a_rank_wt.py`→`backfill_7s/7a/9s/9a_rank_wt.py`、
  `rebuild_s7/s7a/s9/s9a_walkforward_pg.py`→`rebuild_7s/7a/9s/9a_walkforward_pg.py`、
  `s7_evening_reselect.py`→`reselect_7s_evening.py`（`backfill_7ss_rank_wt.py`は
  改名前から既に規則に合致していたため変更なし）

**なぜ改名したか**: S4→S7改名（2026-07-27）・S1〜S3全廃・「7SS」の意味が2度
変わる（軸格上非該当サブランク→2026-07-27にSS統合で廃止→2026-07-31に波乱軸
選出の別戦略として再利用）という経緯が積み重なり、`SEVEN_S7`→suffix`#7S7`な
のに`SEVEN_7A`→suffix`#7A`、`NINE_S9`→suffix`#9S9`なのに`NINE_9A`→suffix`#9A`
という不揃いが生じていた。将来`9SS`等を追加する拡張性も同時に確保している。

廃止済みランク（`SEVEN_S1`/`7PLUS_*`/`SIX_S1`/`M_*`/`U_*`）は改名対象外で
現状の名前のまま残置（過去日再採点・分析スクリプト互換のため）。旧名→新名の
機械的な変換が必要な場合（過去CSV/Discordログ/netkeirin_submissions等、DB
移行対象外の履歴データの読み解き）は`src/strategy_wt.py`の
`LEGACY_RANK_NAME_MAP`/`LEGACY_SUFFIX_MAP`を参照すること。

> ## 🔴🔴 【2026-07-30・最重要】本節以下のROI数値はすべて無効です
>
> **本節および以下の各ランク説明に記載されているROI数値（S1 123.0%/143.3%/182.5%/443.9%、
> S7 128.1%/131.3%/158.3%/405.0%、7A 107.4%、9A 108.4%、entropyゲート454.7%/266.1% 等）は
> すべて汚染されたモデルによる見かけ上の値であり、実際には再現しません。**
>
> **原因**: 四半期vintageモデル18本(`lgbm_wt_eval_q24xx`等)が2026-07-19のバックフィル後、
> 2026-07-28にH2H特徴量のアドホック実験で「復元」のつもりで2回上書き・再学習されていた。
> 学習データがその間に変化していた(race_point NaN補完修正・sb_dynバグ修正)ため同じ設定での
> 再学習は重みの復元にならず、実データ検証で pred_top3_pct 最大72.7pt差・軸1位判定の
> 入れ替わり8.5% を確認。元の重みは`.gitignore`対象で復元不可能。
>
> **クリーンな月次凍結vintageモデル（`docs/vintage_model_policy.md`参照）で再検証した
> 正しい全期間honest実績（2024-01-01〜2026-07-30）**:
>
> | ランク | 件数 | 的中率 | **正しいROI** | 旧記載値 |
> |---|---|---|---|---|
> | S1（三連単2点） | 1,497R | 15.2% | **80.0%** | 143.3%〜443.9% |
> | S7（三連複5点） | 575R | 34.1% | **78.6%** | 131.3%〜405.0% |
> | 参考: ランダム期待値（控除率25%） | — | — | 75.0% | — |
>
> **両ランクとも控除率の壁（75%）にほぼ張り付いており、黒字ではありません。**
> 7A/S9/9Aは誤解防止のため picks_history から破棄済み（再構築は保留）。
>
> さらに市場エッジ診断で **モデルの予測精度はオッズに負ける**（Brier 0.024892 vs
> 市場 0.024496・logloss 0.102513 vs 0.099616・全469,280組）ことが実証され、
> 券種変更・パターン分類・軸選定10方式・硬い除外・3列目絞り込み・ライン相関補正・
> 組単位結合モデル・高配当帯の市場ギャップ・1車選択の切替という全アプローチで
> ROI100%超は達成できないと確定しました。
>
> 詳細は kiseki メモリ `keirin_clean_baseline_market_efficiency_2026_07_30.md` を参照。
> **以下の履歴記述は「当時どう判断したか」の記録として残しますが、数値は信用しないこと。**

**【2026-07-27・内部名S4→S7へ統一・9車版S9と7A/9A境界ランクを新設】** 表示が一度も「S4」に
ならず(SS+/SS/S)、内部コード名(S4/SEVEN_S4)と表示ランクの対応が分かりにくかったため、
9車版が既に「S9-」接頭辞を使っていたのに合わせ7車版も**S7**（`SEVEN_S7`・suffix `#7S7`）へ
統一（`S4_*`定数→`S7_*`・`s4_*`関数→`s7_*`・ファイル名`backfill_s4_rank_wt.py`→
`backfill_s7_rank_wt.py`等）。表示は**7SS**（重なり0）/**7S**（重なり1）、9車版は**9SS**/**9S**。
同時に、S7/S9それぞれの日次ボリュームが小さい（7車合計約1.2R/日・9車約0.25R/日）ため、
ROIはやや落ちても的中率のあるゾーンで推奨数を増やしたいというユーザー要望に対応し、
S7の3ゲート・S9の2ゲートのうち**ちょうど1つだけ不合格**の候補を境界ランク**7A**/**9A**として
新設（`s7a_daily_select`/`s9a_daily_select`・S7/S9とは論理的に排他）。
honest全期間再構築（2024-01-01〜2026-07-26・四半期walk-forward）: 7A 6,029R・的中42.0%・
ROI107.4%（直近7四半期連続100%超）／9A 1,032R・的中35.1%・ROI108.4%。7A/9Aは`gate_label`
なし（S7/S9のSS/S区分はゲート合格が前提のため境界ケースには付与しない）。

**【2026-07-27・観察用サブランク「SS+」を廃止】** 7SS+/9SS+（軸2車の級班に各グレード
最上位を含まないSS内訳・2026-07-23導入）はサンプル数が少なすぎるとのユーザー判断により
廃止しSSへ統合（買い目・投資額はSS+/SSで元々同一のため実害なし）。以下の記述で「SS+」
に言及する箇所は歴史的経緯として読むこと（`s7_gate_label()`は現在wt_overlap_n==0を
常に"SS"へ返す）。

**2026-07-21 同日中の再編**: S2(7PLUS_U)/S3(7PLUS_M)は対象レース数・的中率・期待値の観点で
継続困難と判断し全廃（過去行は`picks_history_u_archive`/`picks_history_m_archive`へ退避）。
S4(SEVEN_S4)は今後の予想データのベースと位置づけ、軸2車がWINTICKET公式◎◯と重なるかで
**SS**（重なり0・全く重ならない）と**S**（重なり1・片方だけ重なる）の2ランクに再編して表示
（内部rankは`SEVEN_S4`のまま・`gate_label`列で"SS"/"S"を区別）。S1は現状維持で継続検討中
（払戻の大きさが予想購入者へのアピール材料になりうるため、ブラッシュアップの方向性を継続検討）。
※上記の`SEVEN_S4`/`#7S4`は2026-07-27に`SEVEN_S7`/`#7S7`へ改名済み（下記参照）。

**2026-07-17 再設計確定**: 正規プロトコル（学習〜2025-03-31／検証=2025-04-01〜2026-03-31 の1年で条件選択
／テスト=2026-04-01〜07-15 で1回評価・モデル `lgbm_wt_val25`）による全ランク再検証の結果、
**合格は S2（現行条件のまま）と S3（新定義）のみ**。S1（6車三連単）・A（一致波乱二連単）は
検証ROI100%超の条件が存在せず全廃（新S1候補スイープ=適応型2車軸トリオ/m1 1着固定三連単も
検証ROI≥95%のセルなしで全滅）。
**2026-07-16 指数改定: 競走得点トレンド4特徴を追加（FEATURE_COLS_WT 40→44・全モデル再学習・バックフィル再構築済み）**

- **【2026-07-31・S1 全廃】** ユーザー判断により「現在有効なデータとは言えない」として、以下の全ROI数値（旧設計・汚染モデル時代のもの含む）は無効。過去分picks_history（`SEVEN_S1`・1,504件・2024-01-02〜2026-07-30）はVPS PGから削除済み（バックアップ: `data/backup/picks_history_s1_discarded_20260731.csv`）。候補生成（`src/cli/main.py`のS1候補ブロック）・ライブ判定呼び出し（`scripts/notify_prerace_wt.py`の`_process_s1_candidates`呼び出し）・欠損自動補完対象（`scripts/backfill_missing_prerace_wt.py`）を全て停止（S2/S3全廃と同じ設計: `s1w_select`/`s1w_gate`/`judge_s1`等のロジック自体は過去日再採点・分析スクリプト互換のため残置し、呼び出し元のみ停止）。以下は導入当時の履歴として参考保持のみ。
- **S1（旧・新設計・win軸1着固定・2026-07-19導入、閾値は同日07-19に0.15→0.22、07-22に0.15へ再変更）** = `SEVEN_S1`（suffix `#7S1`・**ペーパートレード・2026-07-31全廃**）: 軸=1着専用モデル(`lgbm_wt_win`)のレース内1位（固定）× 相手=3着内モデルで軸を除いた上位2頭(p1,p2) × top3_gap(p1-p2の3着内確率差)≥`S1W_TOP3_GAP_MIN` × 三連単 軸→p1→p2, 軸→p2→p1 の2点流し（目オッズ下限なし）。旧S1（7車三連複7PLUS_R）・新S1（6車三連単SIX_S1）はいずれも全廃されたが「win軸固定×3着内モデル相手選定」は未検証だった構造。正規プロトコル: top3_gap閾値0.08〜0.20で検証・テストとも単調に改善（0.15で検証145.8%(n=9949・約27R/日)→テスト135.3%(n=2851)）。07-19同日中にユーザー要望（母数を1日15R以下へ絞り的中率向上）でスイープを延長し0.22へ引き上げ（検証171.6%(15.2R/日・的中18.1%)→テスト146.0%(15.3R/日・的中18.2%)）。旧win_rank/gap12モデルと同型のリーク（下記参照）を修正した四半期walk-forwardモデルで全期間(`scripts/rebuild_s1_walkforward.py`)を再構築した結果、honest全期間実績（0.22時点・2024-01-01〜2026-07-18）= 14,363R・約15.3R/日・的中17.3%・ROI**123.0%**。S2/S3との重複4.3%とほぼ独立。払戻分布は少数の高額配当に偏る。
  **【2026-07-22 再変更・高配当取りこぼし防止】** 万車券(配当≥10,000円)分析（`exp_s1_manshaken_analysis.py`）で「top3_gapを上げても万車券は増えない」と判明（万車券のtop3_gap平均は的中全体よりむしろ低い）。一方「軸の単勝勝率が低いほど高配当」の傾向を確認し、`exp_s1_20x_filter_design.py`で軸勝率フィルターを評価。ユーザー判断で**`S1W_TOP3_GAP_MIN`を0.22→0.15へ復帰**し、**`S1W_AXIS_WIN_PROB_MAX=0.50`新設**（軸の単勝勝率がこれを超えるレースを除外＝本命決着を回避）。honest全期間(軸勝率≤50%フィルター単体): n=13,510(53.5%)・的中率10.7%(元16.2%)・ROI146.3%(元120.3%)・20倍以上再現率65.9%・万車券再現率84.0%。**S1は「軸=win1位固定」のため的中率と高配当は構造的トレードオフ**（的中率を保ったまま高配当だけ追加で拾うことは不可）。コードは`src/strategy_wt.py`/`src/cli/main.py`/`scripts/backfill_s1w_rank_wt.py`に反映済み。
  **【2026-07-22 過去分honest再構築 完了】** VPS PG直接参照で全期間(2024-01-01〜2026-07-22)を四半期ごとに分割実行し再構築（picks_history本番反映済み）: 13,489R・約14.4R/日・的中10.6%・ROI143.3%（旧0.22時点比: 14,363R→13,489R・的中17.3%→10.6%・ROI123.0%→143.3%）。四半期別ROIは81.4%〜263.2%とばらつくが全期間で黒字幅拡大。再構築の過程で**VPS PostgreSQLの`wt_odds`が2026-06-01以降のみのミラーで2024〜2026-05分が丸ごと欠落している**ことが判明（`wt_races`はVPS PG側が全期間完全なのに対し`wt_odds`は逆にローカルMac SQLiteのみ全期間完全という非対称構成だった）。ローカルSQLite（2022-12-01〜2026-07-10・3,469万件）から不足分（2024-01-01〜2026-05-31・2,332万件）をCSV export→scp→`\copy`+`ON CONFLICT DO NOTHING`でVPS PGへ一括移植し解消（VPS disk 60G→65G/99G・所要ディスク+5GB、メモリ影響なし）。**これによりVPS PGのみでwt_oddsに依存する過去分honest再構築が可能になった**（S1は本件で実施確認済み。S2/S3/S4は全廃済みだが今後同型の再構築が必要な場面があればこの移植により`rebuild_*_walkforward.py`のローカルSQLite依存を回避できる）。実行時の注意点: `rebuild_s1_walkforward.py`本体は「ローカルSQLiteが完全な履歴を持つ」旧前提でKEIRIN_DB_URLを読み取り時にpopする設計のままなので、VPS PG一本化で読む場合は環境変数をpopしない別スクリプトが必要（本件では四半期ごとに分割した単発スクリプトを都度実行）。過去分は `scripts/rebuild_s1_walkforward.py`（旧`backfill_s1w_rank_wt.py --wipe`は四半期対応前の単一モデル版・リーク混入のため非推奨）。この直後、ローカルSQLite（`data/keirin.db`）自体を廃止（Mac対話セッションも`~/.zshrc`の`KEIRIN_DB_URL`でVPS PGをデフォルト参照するよう変更・VPS PGが名実ともに唯一のデータソースに）。
  **【2026-07-22 軸級班denyフィルター追加・高配当特化】** ユーザー要望「高的中率を目指すが高配当は捨てない（低配当になりそうなレースを省き高配当の的中率を上げる）」でセグメント別分析（`scripts/exp_s1_segment_deny_analysis.py`・正規プロトコル: train+val〜2026-03-31で選定→test 2026-04-01〜07-22で一度だけ評価）を実施。venue_id/grade/distance/line構成等のうち、**軸選手の級班（player_class）が最も明確なシグナル**: 各グレード内の最上位クラス（S1級/A1級）が軸の場合、的中率は同水準のまま配当が低くなりやすい（train+val: ROI138.5%→173.5%・test: ROI178.9%→246.2%・いずれも的中率は完全に不変）。5万円以上の高配当payoutは全期間7件中6件(85.7%)が残存・カットは1件のみ（払戻額が大きいほど残存率が上がる=高配当ほどこのフィルターで保護される）。`S1W_DENY_AXIS_CLASS={"S1","A1"}`として`s1w_gate()`に統合（`src/strategy_wt.py`/`src/cli/main.py`/`scripts/backfill_s1w_rank_wt.py`）。honest全期間再構築（2024-01-01〜2026-07-22）: **6,426R・約6.9R/日・的中10.6%（変化なし）・ROI182.5%**（旧143.3%から改善）。母数はほぼ半減（13,489→6,426R）するが「SS/Sが別途ある」との判断で許容。過去分は本番picks_history反映済み・kiseki help表示も更新要（次回確認）。
  **【2026-07-25・万車券依存セグメント除外フィルターは不採用】** 「大穴的中(万車券・配当≥10,000円)を軸により不要だったレースを分析し除外できないか」の依頼で検証（`exp_s1_manshaken_dependency_filter.py`）。現行S1はROI(万車券抜)=78.2%と万車券なしでは単体赤字という構造は確認したが、venue/grade/distance/line構成/軸級班/軸脚質/逃げ人数の9次元で「万車券依存幅」を機械抽出したdenyフィルターはtrain+val(ROI138.5%→175.2%)では好成績もtest(2026-04-01〜07-22)でROI178.9%→85.7%へ崩壊（万車券再現率7件中1件のみ）。各セグメントの万車券サンプルが数百件中0〜2件と極小で多重比較ノイズをパターンと誤認していたと判明し**不採用**（既存の軸級班S1/A1除外フィルターの方が同一test期間でROI178.9%→246.2%と頑健に改善しており優位）。詳細メモリ`keirin_s1_manshaken_dependency_filter_2026_07_25`。
  **【2026-07-26・フィールドentropyゲート追加】** S4/S9で有効だったフィールド全体の指数エントロピー（拮抗度・オッズ非依存）シグナルがS1でも独立に機能するか検証（`exp_s1w_entropy_wt.py`・2024Q1のみで閾値決定→残り9四半期へ完全ブラインド適用の真のwalk-forward・9四半期全てで方向一致）。`S1W_ENTROPY_MAX=1.7571`を`s1w_gate()`に追加（entropy≤閾値: n=1,686・的中14.9%・ROI454.7%／entropy>閾値: 的中8.5%・ROI71.5%赤字）。既存の軸単勝勝率・軸級班denyフィルターとは独立な追加シグナル。honest全期間再構築（quarterly walk-forward・VPS PG本番反映済み）: 6,515R→**1,857R・的中15.5%・ROI443.9%**（10四半期中9四半期が125%以上・赤字四半期ゼロ）。
- **S7（現行内部rank `SEVEN_S7`・suffix `#7S7`。単勝×複勝指数トップ3重なり軸×波乱度選出・2026-07-21導入・同日中にWT◎◯重なり考慮版へ改良・2026-07-27に内部名S4→S7へ改名）** = **ペーパートレード**（以下の履歴は導入当時の`SEVEN_S4`/`#7S4`表記のまま残す）: 軸2車 = `pred_win_pct`（単勝指数）上位3 ∩ `pred_top3_pct`（複勝指数）上位3 の重なり車から選定（重なり>=2なら`pred_top3_pct`上位2、重なり==1ならその1車+残りの`pred_top3_pct`最上位。重なり0は対象外・実データで58,616中1件のみ）。波乱度指数 = 軸2車の`pred_top3_pct`合計（`axis_sum`）。**レース全体のエントロピー（拮抗度）で絞るとROIが悪化する（絞り込みなし85.7%→73.5%）ことを確認し不採用**。当初はaxis_sum昇順で日次上位`S4_DAILY_TOP_N`件を採用する方式で、N=15→10へ変更後のhonest全期間実績（2024-01-01〜2026-07-20）= 9,220R・10R/日・的中35.2%・ROI128.1%だった。
  **同日中の追加検証（ユーザー仮説）**: 軸2車がWINTICKET公式予想の◎◯（`prediction_mark`∈{1,2}）と重なる場合に期待値が下がるかを`exp_s4_wt_axis_overlap.py`で検証（honest全期間・四半期walk-forwardモデル）。重なり数別に日次Top10選出内訳を見ると、的中率はほぼ横ばい（33〜37%）なのにROIが重なり数に応じて単調悪化（重なり0=ROI408.1%／重なり1=148.7%／重なり2=完全一致=**75.7%・赤字**）と判明。コンセンサスピック（WT予想と完全一致）は市場に織り込まれ払戻が縮む構造。
  → **選出方式を変更**（`strategy_wt.s4_wt_overlap_n()` / `s4_daily_select()`）: 重なり0（WT◎◯と全く重ならない）は該当があれば無条件で全件採用（本数上限なし）、重なり1（片方だけ重なる）はaxis_sum昇順で固定`S4_DAILY_TOP_N`=10件、重なり2（完全一致）・WTマーク欠損は完全除外。1日あたりの採用本数は重なり0の発生数に応じて可変。honest全期間実績（新方式）= **9,927R・10.77R/日・的中36.3%・ROI131.3%**（旧方式128.1%から改善）。内訳: 重なり0(943R)的中39.4%/ROI232.8%・重なり1(8984R)的中36.0%/ROI120.6%。過去分は `scripts/rebuild_s4_walkforward.py`（`scripts/backfill_s4_rank_wt.py`が新方式に対応済み）
  **【2026-07-21 同日中・表示ランク再編】** S4は今後の予想データのベースと位置づけ、ユーザー指示によりWeb/Discord/サマリー/グラフ全てで内部区分をそのまま表示ランクとする: 重なり0→**SS**（ROI232.8%）、重なり1→**S**（ROI120.6%）。内部rank `SEVEN_S4` はそのままで、`picks_history.gate_label`列に"SS"/"S"を格納して区別する（新規カラム不要・既存の`gate_label`（元はS3のOR gate内訳用）を流用）。`notify_prerace_wt.py`の`_insert_s4_pick`/`_build_s4_message`が対応済み（Discord通知の見出しも"SS"/"S"表示）。
  **【2026-07-22・朝夕統合再選出への再設計】** 上記の「重なり1は固定`S4_DAILY_TOP_N`件」は、朝(`daily_picks_wt.sh`)と夕(`evening_picks_wt.sh`)が別プロセスで独立にこの上限を適用していたため、1日最大20件になるバグと化していた（発覚の経緯は`keirin_s4_gate_label_bug_and_candidate_visibility_2026_07_22`）。さらにhonest全期間検証で「朝の部(19時未満発走)だけでS候補が10件に達する日が57.2%」と判明し、朝が先着で夜の優良候補を取りこぼす構造的懸念が確認された。**`S4_HALF_CAP`=6を新設し、朝夕それぞれの一次選出を6件に縮小**。夕方バッチの最後に`scripts/s4_evening_reselect.py`を実行し、朝夜の生候補（`_s4_raw_candidates.json`/`_night_s4_raw_candidates.json`に新たに永続化）を統合してaxis_sumランキングを組み直す（`strategy_wt.s4_evening_reselect()`）。ただし既に買い判定済み（`bet_amount>0`）のレースは実購入を取り消せないため維持し、未判定分だけ日次合計`S4_DAILY_TOP_N`(10)件へトリムする。honest全期間バックテスト: 現行(朝夕別選出)ROI117.7%(理論上限との選出一致率76.5%) → **新設計ROI120.8%(理論上限120.6%とほぼ同等・一致率89.5%)**。過去分再構築(`scripts/backfill_s4_rank_wt.py`)は1日分データを最初から統合済みのため影響を受けない（`cap=S4_DAILY_TOP_N`を明示指定して従来通りの理論上限相当を再現）。詳細: `keirin_s4_evening_reselect_2026_07_22`
  **【関連: 「非」バッジ再発バグ】** `notify_results_wt.py`の毎時採点処理がpicks_history行をDELETE+`INSERT OR REPLACE`で再作成する際、列リストに`gate_label`が含まれておらず、対象レースが再採点されるたびにgate_labelがNULLに巻き戻り「非」表示になるバグがあった（2026-07-21発見・修正済み）。picks_historyに新規列を追加する際は、この`INSERT OR REPLACE`列リスト（S1/S4/旧U/M共通）に必ず追加すること。あわせてS4は候補時点（買い判定成立前）から`write_candidates_wt.py`がプレースホルダ行を書き込むようになり、Webで候補になった時点からS/SSバッジが表示される（見送りは的中したかを`miwokuri=True・bet=0`のまま参考記録）。詳細: `keirin_s4_gate_label_bug_and_candidate_visibility_2026_07_22`
  **【2026-07-23・SS+観察サブランク新設】** S1で発見した軸級班denyフィルターがS4のSS/Sにも効くか検証（`scripts/exp_s4_axis_class_deny_analysis.py`・正規プロトコル）。**単純な母集団相関ではS/SSとも改善に見えたが、Sは日次axis_sum上位10件の「枠付き相対選出」のため、格上軸候補を除外すると別候補が繰り上がり、実際にシミュレーションすると悪化する**（train+val ROI116.3%→111.5%・test 132.6%→119.2%、両期間で一貫して悪化）。一方SSは無制限採用（枠なし）のため単純に足切りされるだけで、繰り上がり効果を考慮しても改善が残る（train+val ROI222.3%→351.6%・全期間237.1%→362.2%、的中率は不変〜微増）。**結論: Sには適用せず、SS内の軸格上非該当サブセットのみ新表示ランク"SS+"として観察する**（ユーザー判断・実際の買い目・購入対象は変更しない表示分岐のみ）。
  実装: `strategy_wt.s4_gate_label(wt_overlap_n, axis1_class, axis2_class)`（軸級班情報が両方揃いいずれもS1/A1でなければ"SS+"、それ以外は従来通り"SS"、重なり1は"S"）に集約。`src/cli/main.py`（S4候補生成時にaxis1_class/axis2_class追加）・`scripts/notify_prerace_wt.py`（`_process_s4_candidates`/`_build_s4_message`）・`scripts/write_candidates_wt.py`（候補時点表示）・`scripts/backfill_s4_rank_wt.py`（過去分再構築）・`scripts/notify_results_wt.py`（日次結果通知のSS+/SS/S 3分割）・`scripts/save_model_eval.py`（`PAPER_RANKS`にSS+追加）に反映。既存SS行（951件）はSQL UPDATEでgate_label='SS+'/'SS'へ即時分割済み（367件がSS+へ・honest実績: SS+ ROI360.9%(367件)・SS ROI158.9%(584件)・S ROI119.9%(9104件)）。
  **【2026-07-23完了】** kiseki側backend `_display_rank`/frontend `RANK_ORDER`/`RANK_BADGE_STYLE`は同日中に対応済みだったが、`RankBadge`用の`RANK_STYLE`マップ（レース詳細の個別バッジ表示）だけ更新漏れで「非」表示になっていたバグを発見・修正（`keirin_ss_plus_display_fixes_2026_07_23`）。あわせて`/keirin/help`の`RANKS`カード一覧にSS+カードを追加し表示順をSS+/SS/S/S1に変更、Web指数ラベルを「単勝指数/複勝指数/指数」→「単勝率/複勝率/競走得点」へ改称（表示値の実態を反映）。
  **【2026-07-24・axis_sum上限フィルタ追加（三連複5倍未満レースの除外）】** ユーザー要望「三連複が5倍を下回るレースは購入対象から除外したい（朝一入稿のため直前オッズは見られない）」への対応。買い目が軸2車+5点流し（1点100円・計500円）のため三連複配当<500円(5倍)は的中しても賭け金割れになる構造で、honest全期間（2024-01-01〜2026-07-23・935日）で実際にSEVEN_S4選出レースの18.8%がこの状態（このバケット単体でROI 38.8%の恒常的な純損失）。直前盤面オッズは`wt_odds_snapshot`(morning)が2026-06-08〜とまだ7週間分しかなくバックテストに使えないため、発走前確定のwalk-forward特徴量のみで代替指標を探索した結果、既存の`axis_sum`（軸2車のpred_top3_pct合計・波乱度指数）が唯一有効な予測材料と判明（他特徴量を組み合わせてもAUC改善なし＝axis_sumと相関0.83で情報量重複・公開情報の壁）。ただしAUC0.64(train)/0.67(test)止まりで市場オッズほど鋭くは分離できない。ユーザー判断で`S4_AXIS_SUM_MAX=1.3`を採用（`src/strategy_wt.py`の`s4_daily_select()`に統合・次点繰り上げなしで足切り＝S1のS1/A1級班denyフィルタと同じ設計）。過去分は`scripts/rebuild_s4_walkforward_pg.py`で本番反映済み（ローカルSQLite廃止済みのためKEIRIN_DB_URLをpopしないPG直読み版・`rebuild_s4_walkforward.py`本体は旧ローカルSQLite前提のまま未修整で現在は使用不可）。
  **honest全期間実績（2024-01-01〜2026-07-23・935日・本番picks_history反映済みの実測値）**: 全体 10.75件/日→**6.27件/日**(-42%)・的中36.3%→33.7%・**ROI 131.3%→158.3%**。内訳: SS+ 0.39→0.29件/日・的中38.1%・**ROI491.3%**／SS 0.62→0.37件/日・的中31.1%・**ROI201.6%**／S 9.74→5.61件/日・的中33.6%・**ROI138.5%**。
  探索時点の簡易試算（`pred_combo`テキストに小数1桁で丸め保存されたaxis_sum値から後付けフィルタしたもの）ではROI147.1%・7.83件/日と見積もっていたが、実際の`s4_daily_select()`は丸め前のフル精度axis_sumで判定するため境界付近のレースがより多く除外され、上記の本番実測値の方が正しい（母集団が小さくなった分ROIは押し上げ方向）。
  **【2026-07-25・オッズ下限カットは3仮説とも不採用（開示文で対応）】** 「SS+/SS/Sの三連複5点流しでROIを改善できないか」の依頼で3仮説を検証（`keirin_s4_odds_floor_cut_verification_2026_07_25`）。①占有率(axis_sum)下限カット・②軸単勝率下限カットはいずれも**逆効果**（低axis_sum・低軸単勝率ほどROIが高い単調傾向＝axis_sumは元々「低いほど波乱で高配当」という設計そのものが収益源のため）。③目単位の最終確定オッズ(`wt_odds`)による下限カットはROI改善に有効（例: Sランクcut無し139%→10倍超のみ167%）だが、朝夕バッチで買い目を確定する現行入稿フローの時点ではまだ確定していないオッズのため自動化不可。ユーザー判断で①②は採用せず、③はnetkeirin入稿コメントへ開示文として先行実装し利用者判断に委ねる方針（全体のコメント/タイトル設計自体は`netkeirin_title_comment_design`で別途保留中）。
  **【2026-07-26・entropyゲート導入と件数cap撤廃→デプロイ移行期の一時的な回帰→フェイルセーフ強化】** SS+/SS/Sの的中率20-30%・3桁配当連発を受け、フィールド全体の指数エントロピー（拮抗度・オッズ非依存）が高配当レースの判別に使えるか検証（2024Q1のみでしきい値決定→残り7四半期に完全ブラインド適用・一貫して成立）: entropy≤1.8329で30倍+的中の74%を独占（ROI266.1% vs 閾値超78.1%赤字）。axis_sum選定を7四半期中6四半期で上回り無相関(spearman≈-0.08)の独立シグナルと判明したため、`S4_ENTROPY_MAX=1.8329`を採用し**件数cap(`S4_HALF_CAP`/`S4_DAILY_TOP_N`)を撤廃**（axis_sum選定との相関がなく枠取り合い対策自体が不要になったため）。ところがデプロイ当日、entropyフィールドを持たない旧形式の朝バッチ候補JSONが夕方の統合再選出を経由した際、`c.get("entropy", 0.0)`のデフォルト値が「常にゲート通過」として扱われ1日26件という異常が発生。原因を修正（entropy欠損時のデフォルトを0.0→`float("inf")`＝フェイルオープンからフェイルクローズへ）し、`S4_DAILY_CAP=12`を安全網として再導入（entropyゲート通過候補が日次合計でこれを超える場合のみentropy昇順でトリム。通常運用ではほぼ発火しない設計・honest全期間ではcap値8/10/12/15/無制限のいずれでもROI/件数が完全同一）。あわせてロック済み（既に買い判定済み）候補をゲート適用**前**に保護するよう防御的に修正（従来はゲート適用後に分離しておりentropy欠損等でロック済み候補が保護されない抜け穴があった）。
  **【2026-07-27・軸2車が◎◯△のうち2つと一致する場合を除外】** 「2軸がWINTICKETの◎◯△のうち2車と一致すると市場人気と重なり不人気が来てもオッズが高くならない」というユーザー仮説を検証: 軸2車のうち2車が◎◯△いずれかと一致する母集団はROI182.9%、それ以外はROI434.4%で、払戻トップ5は全て「2車一致」に該当しない側に集中。`s4_wt_mark3_overlap_n()`（既存の◎◯(mark1/2)のみ見る`wt_overlap_n`に対し△(mark3)も加味した拡張版）を新設し、`S7_MARK3_OVERLAP_MAX=1`（軸2車の両方が◎◯△いずれかと一致=2の場合を除外・欠損時はフェイルセーフとして除外扱い）をS4/S9両方に追加。
  **【2026-07-27・内部名S4→S7へ統一】** 上記の通り`SEVEN_S4`/`#7S4`/`S4_*`/`s4_*`は全て`SEVEN_S7`/`#7S7`/`S7_*`/`s7_*`へ改名済み（詳細は本節冒頭）。honest全期間再構築（mark3ゲート込み・quarterly walk-forward・VPS PG本番反映済み）: 2,313R→**1,076R・的中40.6%・ROI405.0%**（全10四半期で改善）。表示は重なり0→**7SS**、重なり1→**7S**（旧SS+は2026-07-27にSSへ統合済み・上記参照）。
- **S9（S7の9車立て版・独立ランク・2026-07-26〜27導入）** = `NINE_S9`（suffix `#9S9`・**ペーパートレード**）: 2026-08開催予定「ドリームレース」（S級・過去3回とも9車立て）をターゲットに含めるため、7車専用だったS7ロジック（軸選定`s4_select_axis`・entropy計算・WT◎◯重なり判定・表示ランク付けは車数非依存で共通利用）を9車立てへ拡張。9車立ては全レースの8.0%（約6.5件/日）で、ユーザー判断によりS7とは独立した別ランクとして実装（表示・集計を分離。買い目コストが7点流し=700円とS7の5点=500円で異なるため）。ゲートは`S9_ENTROPY_MAX=1.9938`（2024Q1のentropy下位25%点を9四半期へブラインド適用・全て方向一致）と`S7_MARK3_OVERLAP_MAX`共有（`axis_sum`は9車では未較正のため導入せず・複数の未較正閾値の積み増しによる過学習回避）。表示は重なり0→**9SS**、重なり1→**9S**。honest全期間実績（mark3ゲート込み・2024-01-01〜2026-07-25）: **233R・的中48.9%・ROI412.8%**（全10四半期改善）。翌朝の結果確定・採点・Discordサマリー通知は`[9車 S9]`として既存`[7+車]`ヘッダーとは別メッセージで送信（低頻度のS9行がmsg[:1900]切り詰めで消えるのを防ぐため）。
- **7A/9A（S7/S9の境界ランク・ボリューム拡大・2026-07-27導入）** = `SEVEN_7A`（suffix `#7A`）/ `NINE_9A`（suffix `#9A`）・**ペーパートレード**: S7/S9は日次ボリュームが小さい（7車合計約1.2R/日・9車約0.25R/日）ため、ROIはやや落ちても的中率のあるゾーンで推奨数を増やしたいというユーザー要望に対応。S7の3ゲート（axis_sum≤1.3・entropy≤1.8329・mark3≤1）のうち**ちょうど1つだけ**不合格の候補を7A、S9の2ゲート（entropy≤1.9938・mark3≤1）のうちちょうど1つだけ不合格の候補を9Aとして新設（`s7a_daily_select`/`s9a_daily_select`・S7/S9とは論理的に排他）。wt_overlap==2（WT◎◯完全一致）側を含めた単純なaxis_sum区切りも検討したが全10四半期でROI70〜96%と一度も100%を安定して超えず不採用。honest全期間再構築（2024-01-01〜2026-07-26・四半期walk-forward）: **7A 6,029R・的中42.0%・ROI107.4%**（直近7四半期連続100%超）／**9A 1,032R・的中35.1%・ROI108.4%**。`gate_label`列は使わない（境界ケースにはSS/S区分を付与しない）。
  **【2026-07-28・候補時点でのページ表示に対応】** S1/S7は朝の候補生成直後に`picks_history`へプレースホルダ行を書き込みページに当日中から候補表示していたが、S9/7A/9Aは同じ仕組みが未実装で発走15分前判定（オッズ取得後）が走るまでページに現れなかった（ユーザー指摘。軸2車選定・波乱度・ゲート判定はいずれもオッズ非依存でモデル計算のみから確定するため技術的制約はない）。`scripts/write_candidates_wt.py::_write_paper_candidates()`にS9/7A/9Aの書き込みブロックを追加し3ランクとも候補時点表示に対応。あわせてS7/S9の候補時点pred_comboを汎用プレースホルダー`"axis1=axis2-候補"`から実際の残り車番号リスト`"axis1=axis2-3,4,5,..."`（`_third_list()`。対象車数7/9は候補生成時点で確定済みのためオッズなしで一意に決まる）に変更。詳細メモリ`keirin_s9_7a_9a_candidate_display_2026_07_28`。
- **【2026-07-28・H2H(頭対戦成績)特徴を実装したが本番採用は撤回】** netkeirin「対戦表」に着想を得た特徴（`h2h_win_rate`/`h2h_n_total`/`h2h_net_norm`・`wt_entries.finish_order`の自前履歴からpoint-in-timeで再現・netkeirinスクレイピング不要）を`add_h2h_features_wt()`として`src/preprocessing/feature_wt.py`に実装し全期間honest再学習・vintage 18モデル再学習込みでS1/S7/S9をhonest全期間walk-forward再検証したところ、S1(ROI443.0%→363.5%)・S9(412.8%→286.8%)が悪化し、単独事前検証で改善していたS7(401.1%→424.8%)のみ改善という結果になった。S1/S7/S9は共有モデル(`lgbm_wt`/`lgbm_wt_win`)を使うため部分採用はできず**全面撤回**。本番モデル・vintageモデルは全て実装前の状態（48特徴）に再学習し直して復元済み。`add_h2h_features_wt()`自体はコードとして残るが`FEATURE_COLS_WT`には含まれず未使用。詳細メモリ`keirin_netkeirin_h2h_feature_2026_07_28`。
- **旧新S1（SIX_S1・6車三連単・2026-07-17 全廃）**: 3独立窓では110/103/113%だったが正規プロトコルの1年検証で最良70.3%・100%超なし（「直近だけ良い」レジーム依存を検出）。6車全域・9車・新S1候補も全滅 → 全廃。`#6S1` 行は `picks_history_r_archive` へ退避（`scripts/archive_s1_a_abolition_wt.py`）
- **A（7PLUS_A・2026-07-17 全廃）**: 正規プロトコルで検証最良88.5-94.2%・100%超なし → 全廃。`#7A` 行は `picks_history_a_archive` へ退避（同上スクリプト）。旧・買い目カット方式Aランク（〜2026-06-19）の行も同テーブルに退避済み
- **旧S1（7PLUS_R・7車三連複・2026-07-16 全廃）**: 検証期間ROI 67.3%・代替条件の全探索で黒字なし。過去行（7PLUS_R/7PLUS_CAND/7PLUS_SS/7PLUS_S）は `picks_history_r_archive` へ退避。wave-picks の SS txtセクション・#CAND 書き込み・ガミ判定は停止済み（ss_policy 等は互換のため残置）
- **S2（旧U・7PLUS_U・2026-07-21 全廃）**: 波乱見込み×穴×同ライン「逃」相方の三連複2車軸流し。廃止直前にmto閾値を4.3→4.5へ厳選したが、honest全期間再構築（`scripts/rebuild_s2_walkforward.py`・四半期walk-forwardモデル）で確認したところ4.3=ROI81.6%(1251R)→4.5=ROI84.8%(1155R)と全期間では依然として損失圏内（2024〜2025年前半が40-70%台で低迷）。対象レース数・的中率・期待値の観点で継続困難と判断し全廃。過去行（1155件）は `picks_history_u_archive` へ退避（`scripts/archive_u_m_abolition_wt.py`）。judge_u/`_process_u_candidates`等のロジックは過去日再採点・分析スクリプト互換のため残置（呼び出し元のみ停止）
- **S3（旧M・7PLUS_M・2026-07-21 全廃）**: ◎不一致×軸信頼ゲートの三連複2車軸流し。廃止直前にwin_rank単独ゲート化+目≥20倍で honest全期間ROIを95.9%→120.4%(801R)まで改善させていたが、S2と合わせて対象レース数・的中率・期待値の観点で継続困難と判断し全廃（過去の閾値変遷・リーク発覚の経緯は本ファイルのgit履歴・メモリ`keirin_composite_ratio_gate`参照）。過去行（801件）は `picks_history_m_archive` へ退避（`scripts/archive_u_m_abolition_wt.py`）。judge_m/`_process_m_candidates`/`m_axis_gate`等のロジックは過去日再採点・分析スクリプト互換のため残置（呼び出し元のみ停止）
- **S1（win軸1着固定・三連単2点流し・2026-07-31 全廃）**: ユーザー判断により「現在有効なデータとは言えない」として全廃。過去行（1,504件・SEVEN_S1）はpicks_historyから完全削除（archive退避ではなくDELETE。バックアップCSVのみ`data/backup/`に保持）。S2/S3と同じ設計で`s1w_select`/`s1w_gate`/`judge_s1`等のロジックは過去日再採点・分析スクリプト互換のため残置（呼び出し元のみ停止）。**欠損自動補完スクリプト（`scripts/backfill_missing_prerace_wt.py`）からもS1を除外済み**（除外しないと翌日以降「S1が毎日欠損している」と誤検知し自動再生成・再挿入され続け全廃の指示が上書きされる事故になるため。S3全廃時に判明した同型の教訓を適用）
- S1/7SS/7S/9SS/9S/7A/9A は live 100R以上で採否判定（実賭け昇格 or 廃止。月次判定は分散的に禁物・40R全外れ月も想定内）。競輪は2026-07-23より[[feedback_keirin_monitoring_phase_2026_07_23]]の**結果監視フェーズ**（明示指示なくロジック改修・新ランク提案をしない）。詳細はメモリ `keirin_s1_redesign_sweep` / `keirin_s1_win_axis_paper` / `keirin_s1_threshold_axis_win_prob_2026_07_22`（S1閾値0.15復帰+軸勝率ゲート）/ `keirin_picks_history_data_loss_2026_07_20`（S4）/ `keirin_s2_s3_tightening_2026_07_21`（S2/S3全廃の経緯）/ `keirin_s4_wt_overlap_selection_2026_07_21`（S4→SS/S再編）/ `keirin_s4_evening_reselect_2026_07_22`（S4朝夕統合再選出・現行設計）/ `keirin_s4_gate_label_bug_and_candidate_visibility_2026_07_22`（非バッジ再発バグ・候補可視化機能）/ `keirin_s1_manshaken_dependency_filter_2026_07_25`（万車券依存フィルター不採用）/ `keirin_s4_odds_floor_cut_verification_2026_07_25`（オッズ下限カット不採用・開示文対応）/ `keirin_s4_entropy_gate_2026_07_26`（entropyゲート・件数cap撤廃・デプロイ移行期バグ）/ `keirin_s9_and_mark3_gate_2026_07_27`（S9新設+mark3ゲート+S1entropyゲート）/ `keirin_netkeirin_h2h_feature_2026_07_28`（H2H特徴・S1/S9悪化のため撤回）/ `keirin_s9_7a_9a_candidate_display_2026_07_28`（S9/7A/9A候補時点表示）
- **廃止済みランク**: S/S+（`7PLUS_ST`/`7PLUS_STP`・三連単1着固定F）は 2026-07-15 に全廃・過去分もDB削除（`keirin.picks_history_st_archive` に退避）。SO≥8フィルタ・旧≤6車 SS/S/A/B・ワイドも廃止済み。7SS+/9SS+（軸格上非該当サブランク・2026-07-23導入）は2026-07-27にSS/9SSへ統合済み。旧ドキュメント・メモに残る記載は無効
- **実精算方式（2026-07-15〜）**: バックテスト・採点とも、指数ランキング＝発走前のオッズ盤面掲載車（欠車除く・落車失格含む）、落車失格絡みの買い目＝外れ計上（返還しない）、欠車のみ返還。完走者ランキングの旧方式は約2-4倍過大で全面廃止
- 見送り=miwokuri=TRUE。**実賭けランクは現在なし**（全ランクペーパー・名目賭金）。Webサマリーのトップラインは `rank IN ('SEVEN_S1','SEVEN_S7','NINE_S9','SEVEN_7A','NINE_9A')`（S7/S9は`gate_label`でSS/Sに分割表示・7A/9Aはgate_labelなし）の名目合算
- `prerace_decisions_{date}.json` が採点/Web/サマリー/Discord の正本（15分前判定を事後変更しない）。キーは S1=`{rk}#S1` / S7(SS/S)=`{rk}#S7` / S9(SS/S)=`{rk}#S9` / 7A=`{rk}#7A` / 9A=`{rk}#9A`（廃止済みのS2=`{rk}#U`/S3=`{rk}#M`キー・旧S7=`{rk}#S4`キーは過去日分のみ存在）
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
- **【2026-07-22追記・2026-07-29訂正】`wt_odds`はVPS PG側こそ不完全だった**
  （`wt_races`とは逆パターン）。VPS PGの`wt_odds`は2026-06-01以降のミラーのみで
  2024〜2026-05分が丸ごと欠落しており、ローカルMac SQLiteだけが2022-12-01〜の
  全履歴を持っていた。S1のhonest再構築（2026-07-22）時にこの非対称に気づき、
  不足分2,332万件をCSV export→scp→VPS上で`\copy`+`ON CONFLICT DO NOTHING`により
  一括移植（VPS disk使用量+5GB）。**しかしこの移植は「2024-01-01〜2026-05-31」に
  限定されており、2022-12-01〜2023-12-31分（全レースの29.4%・29,444レース）は
  対象外のまま今日（2026-07-29）まで欠落し続けていた**（下記「現在はVPS PGの
  wt_oddsも2022-12-01〜今日まで完全」という記述は誤りだった。外部データ取得の
  根本監査[[keirin_s7_foundational_rethink_2026_07_29]]で発覚）。
  `scripts/backfill_wt_odds_2022_2023.py`でwinticket.jpから再取得し解消済み
  （2年以上前のレースでも最終オッズページが引き続き提供されていることを確認済み）。
  他のテーブル（`wt_entries`等）にも同様の非対称ミラー範囲が隠れている可能性が
  あるため、大規模rebuild前は対象テーブルのVPS PG側カバレッジを個別に確認する
  こと。「完全」という記述は必ず実データでCOUNT(*)突合してから書くこと（今回の
  最大の教訓：ドキュメントの「解消済み」記載を鵜呑みにせず実データで裏取りした
  結果、8日間気づかれていなかった欠落が発覚した）。

## スキーマ管理ルール（picks_history 等 keirin スキーマ）

- **DDL は「kiseki 側 alembic」と「本リポジトリ src/database.py::migrate_db()（SQLite用）」の両方に必ず追加する**
  （gap23 列が両方から漏れて本番 PG に手動ALTERだけで存在する「幽霊カラム」になった事故あり → 2026-07-12 に両側へ正式化済み: kiseki alembic `j6k7l8m9n0p1` / migrate_db）
- **gap カラムのスケール**: gap12 / gap34 = 0-1 スケール、**gap23 のみ pt（%ポイント・×100済み）**。歴史的経緯によるもので変更不可。読み書き時に注意
- 閾値定数（GAMI_THRESHOLD=7.0 等）は `src/cli/main.py` / `scripts/notify_prerace_wt.py` / `scripts/write_candidates_wt.py` に多重定義。**変更時は3ファイル + kiseki フロント（page.tsx）を必ず grep して揃える**
- **新しいテーブルを追加したら `src/database.py::_pg_translate()` の keirin スキーマ自動付与regex（2箇所: INSERT系/通常SQL系）にもテーブル名を必ず追加する**。INSERT OR REPLACE/IGNORE文はテーブル名を直接展開するため regex 漏れでも動くが、素のSELECT/UPDATE/DELETEはこのregexだけが唯一のスキーマ付与経路なので、漏れると `relation "xxx" does not exist` で本番クラッシュする（2026-07-24発覚: `netkeirin_submissions`追加時にregexへの追加を忘れ、`_already_submitted()`のSELECTが機能追加以来一度もschema解決できず、netkeirin入稿が導入(2026-07-23)以来一度も成功していなかった。INSERT経路は正しく動いていたため気づかれなかった）

## 変更時チェックリスト（2026-07-31・データ整合性レビューで新設）

2026-07-31 に7領域のデータ整合性レビューを実施した結果、検出された重大事項は
ほぼ全て単一の根本原因に還元されることが判明した:

> **同じ知識（ランク集合・void 判定・モデル期間定義・列リスト）が複数ファイルに
> 独立してコピーされ、更新が同期していない。**

サマリーのランク漏れは3回（過去2回に加え今回`SEVEN_SS`欠落を検出）、ランク全廃時の
経路漏れは2回（S3全廃時に事故が起きかけ、今回S1全廃で4経路目まで漏れていた）
反復しており、同型の事故を繰り返さないため以下をチェックリスト化する。

### ランク新設・全廃時に確認する7経路

本ファイル内の「S1 全廃」の記述（上記「現行ランク体系」節）は当初「候補生成・
ライブ判定・欠損自動補完の3箇所」を止めれば十分という前提で書かれていたが、
2026-07-31 のレビューで**実際には最低7経路を止める必要がある**と判明した
（**3箇所という記述はその時点の理解であり、以下の通り7箇所へ拡張する**）。
ランクを新設・全廃する際は必ず以下7つ全てを確認すること:

1. **候補生成** — `src/cli/main.py` のランク別候補生成ブロック。加えて
   `scripts/reconcile_walkforward_tail.sh`（→`rebuild_s1_walkforward_pg.py --tail-only`
   等）のような**定期rebuild/reconcileスクリプトも「候補生成」の一種**として
   対象に含めること（picks_historyを対象ランクでDELETE→INSERTし直す設計のため、
   これを止め忘れると廃止済みランクの直近月分だけ自動的に復活する。S1全廃の
   「4経路目」としてここで発覚。cronはPAUSED中で即時事故はなかった。commit `33ba316`）
2. **ライブ判定（発走15分前判定）** — `scripts/notify_prerace_wt.py` の
   ランク別ハンドラ呼び出し（`_process_s1_candidates`等）
3. **欠損自動補完** — `scripts/backfill_missing_prerace_wt.py`。対象ランク
   リストに廃止済みランクを残したままにすると、翌日以降「毎日欠損している」
   と誤検知して最終オッズで自動的にpicks_history行を再生成し続け、全廃の
   指示そのものが上書きされる事故になる（S3全廃時に判明した教訓をS1全廃でも
   適用済み）
4. **候補書き込み（当日中の候補表示）** — `scripts/write_candidates_wt.py::_write_paper_candidates()`
   （S1のブロックがここに残存しpicks_historyへINSERTし続けていたことが
   2026-07-31に発覚。commit `3775101`）
5. **Discord通知** — `scripts/notify_prerace_wt.py` のメッセージ生成・
   pick挿入関数（`_insert_rank_7s_pick` / `_build_rank_7s_message` 等ランク別関数。
   2026-07-31のランク全面改名で`_insert_s7_pick`/`_build_s7_message`から
   さらに改称済み。旧名 `_insert_s4_pick`/`_build_s4_message` は2026-07-27の
   S4→S7改名時点で既に置き換わっていた）
6. **サマリー集計** — `scripts/notify_results_wt.py::_query_stats` のIN句
   および同ファイルの `_PAPER_SUFFIXES` 定数。**いずれも commit `5fb70c1` で
   単一正本 `CURRENT_PAPER_RANKS`（`src/strategy_wt.py`）からの導出に変わった**
   ので、通常はこの2箇所を直接編集する必要はない（正本を直せば波及する）:
   `_PAPER_SUFFIXES`（51行）/ `_QUERY_STATS_RANKS_SQL`（316行）
7. **netkeirin入稿** — `scripts/netkeirin_submit_wt.py` の `RANK_CONFIGS` /
   `RANK_ORDER`（80/88行）。**`_is_enabled()`（230行）はfail-open**（設定行が
   存在しない・削除されている場合、無効ではなく常時ONとして扱われる）ため、
   `RANK_CONFIGS`/`RANK_ORDER`から確実に除去しないと自動入稿が止まらない
   （S1がここに残存し常時ON扱いだったことが2026-07-31に発覚。commit `3775101`）

### ランク集合の定義箇所一覧（2026-07-31 commit `5fb70c1` で単一正本化 済）

**現在は `src/strategy_wt.py` の `CURRENT_PAPER_RANKS`（`PaperRankSpec` のタプル）が
唯一の正本**で、下記4箇所は全てそこから導出される。ランクを追加・削除する際は
**正本1箇所だけを直せばよい**。`ABOLISHED_PAPER_RANKS` が廃止済みランクの
ブラックリストを持ち、`tests/test_paper_rank_single_source.py` の
`test_all_four_locations_agree_on_current_rank_universe` が4箇所の矛盾を
機械的に検出する（再発防止の本体）。

以下は単一正本化に至った経緯の記録。ランクの集合は4箇所に独立して
ハードコードされており、2026-07-31時点で内容が全て食い違っていた:

- `scripts/notify_results_wt.py:313` `_query_stats` のIN句（`SEVEN_SS`が欠落
  していた）
- `scripts/notify_results_wt.py:1172,1198` `_PAPER_SUFFIXES`（全廃済みの
  `#6S1` が残置されていた）
- `scripts/save_model_eval.py:52-56` `PAPER_RANKS`（S9・7A・9A・7SSが欠落
  していた）
- `scripts/live_report_wt.py:35` `RANKS`（2026-07-16全廃済みの`7PLUS_R`
  のみで3週間以上n=0の空振りになっていた。2026-07-31 commit `3775101`で
  現行ランクへ修正）

**上記4箇所は commit `5fb70c1` で正本からの導出に置き換わったため、個別更新は
不要になった。** ただし正本を経由しない新しいランク参照を書けば同じ問題が
再発するので、ランク集合を扱うコードは必ず `CURRENT_PAPER_RANKS` を
import すること。加えて `netkeirin_submit_wt.py` の
`_is_enabled()` のような **fail-open な有効/無効判定は「消し忘れると危険」
ではなく「消さない限り有効のまま」という逆方向の罠**になる点に特に注意する
こと（新設定を追加し忘れても気づきにくいが、廃止時に除去し忘れると即座に
常時ON扱いになる非対称なリスク）。

### void（欠車返還）判定を変更する際に同時に見るべき3実装

「欠車で返還・失格で没収」の境界判定は3箇所に独立実装されており、
2026-07-31時点で判定基準が食い違っていた:

- 本番 `scripts/notify_results_wt.py::_void_by_dns`（95行）— 正。**オッズ
  盤面掲載車**（board）基準。DNF（発走後の落車・失格・棄権。`finish_order=0`
  として記録される）は盤面に残ったまま＝外れ計上（没収）
- `src/evaluation/void_rules.py`（→`backtest_wt.py`が参照）— 旧docstringは
  「完走者（`finish_order>=1`）基準」と記載しており本番と不一致だった
  （2026-07-31是正済み。DNFを誤って欠車＝返還扱いしていたため、バックテスト
  ROIが本番の実損益より構造的に高く出ていた）
- `scripts/backfill_*_rank_wt.py` 系 — 「盤面サイズ完全一致以外は候補
  プールから全体除外」という3つ目の基準

void判定ロジックを変更する際は、この3実装を必ず同時に確認し、
「完走者基準」と「盤面掲載車基準」を混同しないこと。

### `finish_order = 0` の意味（重要・void判定・特徴量設計の前提）

（本ファイル上記「winticket ルート」節の「`finish_order=0`は欠車/失格=着外」
という記述は着順判定としては誤りではないが、「欠車」という語のニュアンスが
不正確なため以下で補足する。）

2026-07-31の実データ検証で、**事前確定の欠車（取消・除外）は`wt_entries`に
行自体が作られず物理削除される**（`src/scraper/pipeline_wt.py:236-249`・
`src/scraper/winticket.py:274`）ため、DB上に残る`finish_order=0`行は事前
欠車ではなく、**発走後の落車・失格・棄権（DNF）**であることが判明した。
実測: `finish_order=0`の行は9,528件・5,474レースあり、うち99.6%に
`res_standing`/`res_back`の実測値（＝発走が確定していた証拠）が入っており、
該当レースの98.5%は行数が`wt_races.n_entries`と一致する（＝出走予定馬は
全員行として存在し、その一部がDNFになっただけで、事前欠車によって行数が
減っているわけではない）。top3判定を`between(1,3)`とする既存実装はこのまま
引き続き正しい（DNFは着外扱いで問題ない）が、「finish_order=0=（事前）欠車」
という理解のままvoid判定・特徴量設計を行うと、上記のvoid_rules.py旧版と
同型の取り違えが起きるため、**「finish_order=0はDNF（発走後の非完走）」
「事前欠車は行自体が存在しない」**という区別を前提とすること。

### モデル期間定義（QUARTERS等）は `src/wt_vintage_config.py` を単一正本とする

四半期・月次のモデル学習期間定義（QUARTERS等）は`src/wt_vintage_config.py`
に単一正本化済み（旧6ファイル重複を統合）。**新しいスクリプト（`exp_*.py`
含む）でモデル期間を扱う場合は、独自にQUARTERS等を定義せず必ず
`wt_vintage_config`をimportすること**。2026-07-31時点で独自QUARTERS定義を
持つexpスクリプトが21本残存し、`docs/vintage_model_policy.md`の削除宣言に
反して汚染済み四半期vintageモデル28ファイルが未削除のまま残っていた
（commit `1a9bdac`で削除・独自QUARTERS系expスクリプトに警告コメントを付与）。
うち2本（`exp_s7_gate_staged_audit.py`/`exp_s7_reduced_box_wt_marks.py`）は
独自変数`S7_AXIS_SUM_MAX=1.3`をハードコードしており、現行の本番定数
`RANK_7S_AXIS_SUM_MAX`（`src/strategy_wt.py:329`、値は`1.5`）と食い違って
いた（2026-07-31のランク全面改名で本番定数名は`S7_AXIS_SUM_MAX`→
`RANK_7S_AXIS_SUM_MAX`へ変更済み。expスクリプト側のハードコード変数名は
改名対象外のため`S7_AXIS_SUM_MAX`のまま残る）。

### pure関数の仕様変更時は同一コミットでテストを更新する

`rank_7s_gate_label()`（当時の名前は`s7_gate_label()`。2026-07-31のランク
全面改名commit `f31f84b`で改称済み）のようなpure関数の仕様を変更する際、
同一コミットで対応テストを更新しないと、CIが赤いままmasterへ積み上がる。
CIのdeployジョブは`needs: test`でtestジョブに従属するため、**テストが
赤い間はpushしても本番へ自動デプロイされない**（気づかれにくい形で
リリースが止まる）。2026-07-31、commit `e994758`で`s7_gate_label()`
（当時の名前）を「重なり0も"S"を返す」仕様へ変更した際にテストを更新し
忘れ、masterのCIが赤いまま放置されていたのをcommit `f8811b8`で修復した。

### Discord通知 `send(content, channel)` の `channel` は必須引数

`src/notify/discord.py::send(content, channel)`（39行）は`channel`が必須
引数（省略不可）。2026-07-24のDiscord 5チャンネル分割（commit `8a3abe4c`）
で`channel`が追加されて以降、shellスクリプト内で`python -c "..."`の
ワンライナーとして`send('...')`を1引数だけで呼んでいる箇所は静的解析にも
テストにも引っかからずTypeErrorでサイレント失敗する。
`scripts/daily_picks_wt.sh`（当時92行目）の呼び出しがこのパターンで約1週間
失敗し続け、「race_point異常で本日の推奨をスキップした」という最重要
アラートが届いていなかった（2026-07-31、commit `66386b6`で`channel='system'`
を追加し修復）。**shell内から`python -c`でPython関数を呼ぶ箇所は特に注意し、
`channel`引数の指定漏れがないか個別に確認すること**。

### 本日実施した是正（2026-07-31）

上記データ整合性レビューを受けて以下を同日中に修正済み:

- commit `f8811b8` — `s7_gate_label()`（2026-07-31改名後は`rank_7s_gate_label()`）
  のテスト未更新によるCI赤字を修復
- commit `3775101` — S1全廃の残存2経路（候補書き込み・netkeirin入稿）を停止、
  `live_report_wt.py`のランク集合を現行へ更新
- commit `33ba316` — S1自動再生成の第4経路（`reconcile_walkforward_tail.sh`）
  を停止
- commit `1a9bdac` — 汚染済み四半期vintageモデル28ファイルを削除、独自
  QUARTERS系expスクリプトに警告を付与
- commit `66386b6` — 運用shellにDB URLガード・多重起動防止(flock)を追加、
  Discordアラートの引数漏れ（`channel`必須化未対応）を修復
- commit `bd127b1` — モデル保存のアトミック化（`src/models/model_io.py`新設）、
  vintage凍結保護のrm耐性強化
