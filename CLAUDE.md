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

- **S1（新設計・win軸1着固定・2026-07-19導入、同日中に閾値再調整）** = `SEVEN_S1`（suffix `#7S1`・**ペーパートレード**）: 軸=1着専用モデル(`lgbm_wt_win`)のレース内1位（固定）× 相手=3着内モデルで軸を除いた上位2頭(p1,p2) × top3_gap(p1-p2の3着内確率差)≥0.22（`S1W_TOP3_GAP_MIN`・2026-07-19に0.15→0.22へ引き上げ）× 三連単 軸→p1→p2, 軸→p2→p1 の2点流し（目オッズ下限なし）。旧S1（7車三連複7PLUS_R）・新S1（6車三連単SIX_S1）はいずれも全廃されたが「win軸固定×3着内モデル相手選定」は未検証だった構造。正規プロトコル: top3_gap閾値0.08〜0.20で検証・テストとも単調に改善（採用時点0.15で検証145.8%(n=9949・約27R/日)→テスト135.3%(n=2851)）。**同日中にユーザー要望（母数を1日15R以下へ絞り的中率向上）でスイープを延長**（`exp_s1w_gap_tighten.py`）、0.22で検証171.6%(15.2R/日・的中18.1%)→テスト146.0%(15.3R/日・的中18.2%)と確認し採用値を更新。旧win_rank/gap12モデルと同型のリーク（下記参照）を修正した四半期walk-forwardモデルで全期間(`scripts/rebuild_s1_walkforward.py`)を再構築した結果、honest全期間実績（2024-01-01〜2026-07-18）= 14,363R・約15.3R/日・的中17.3%・ROI**123.0%**。S2/S3との重複4.3%とほぼ独立。払戻分布は少数の高額配当に偏る。過去分は `scripts/rebuild_s1_walkforward.py`（旧`backfill_s1w_rank_wt.py --wipe`は四半期対応前の単一モデル版・リーク混入のため非推奨）
- **S4（単勝×複勝指数トップ3重なり軸×波乱度選出・2026-07-21導入・同日中にWT◎◯重なり考慮版へ改良）** = `SEVEN_S4`（suffix `#7S4`・**ペーパートレード**）: 軸2車 = `pred_win_pct`（単勝指数）上位3 ∩ `pred_top3_pct`（複勝指数）上位3 の重なり車から選定（重なり>=2なら`pred_top3_pct`上位2、重なり==1ならその1車+残りの`pred_top3_pct`最上位。重なり0は対象外・実データで58,616中1件のみ）。波乱度指数 = 軸2車の`pred_top3_pct`合計（`axis_sum`）。**レース全体のエントロピー（拮抗度）で絞るとROIが悪化する（絞り込みなし85.7%→73.5%）ことを確認し不採用**。当初はaxis_sum昇順で日次上位`S4_DAILY_TOP_N`件を採用する方式で、N=15→10へ変更後のhonest全期間実績（2024-01-01〜2026-07-20）= 9,220R・10R/日・的中35.2%・ROI128.1%だった。
  **同日中の追加検証（ユーザー仮説）**: 軸2車がWINTICKET公式予想の◎◯（`prediction_mark`∈{1,2}）と重なる場合に期待値が下がるかを`exp_s4_wt_axis_overlap.py`で検証（honest全期間・四半期walk-forwardモデル）。重なり数別に日次Top10選出内訳を見ると、的中率はほぼ横ばい（33〜37%）なのにROIが重なり数に応じて単調悪化（重なり0=ROI408.1%／重なり1=148.7%／重なり2=完全一致=**75.7%・赤字**）と判明。コンセンサスピック（WT予想と完全一致）は市場に織り込まれ払戻が縮む構造。
  → **選出方式を変更**（`strategy_wt.s4_wt_overlap_n()` / `s4_daily_select()`）: 重なり0（WT◎◯と全く重ならない）は該当があれば無条件で全件採用（本数上限なし）、重なり1（片方だけ重なる）はaxis_sum昇順で固定`S4_DAILY_TOP_N`=10件、重なり2（完全一致）・WTマーク欠損は完全除外。1日あたりの採用本数は重なり0の発生数に応じて可変。honest全期間実績（新方式）= **9,927R・10.77R/日・的中36.3%・ROI131.3%**（旧方式128.1%から改善）。内訳: 重なり0(943R)的中39.4%/ROI232.8%・重なり1(8984R)的中36.0%/ROI120.6%。過去分は `scripts/rebuild_s4_walkforward.py`（`scripts/backfill_s4_rank_wt.py`が新方式に対応済み）
  **【2026-07-21 同日中・表示ランク再編】** S4は今後の予想データのベースと位置づけ、ユーザー指示によりWeb/Discord/サマリー/グラフ全てで内部区分をそのまま表示ランクとする: 重なり0→**SS**（ROI232.8%）、重なり1→**S**（ROI120.6%）。内部rank `SEVEN_S4` はそのままで、`picks_history.gate_label`列に"SS"/"S"を格納して区別する（新規カラム不要・既存の`gate_label`（元はS3のOR gate内訳用）を流用）。`notify_prerace_wt.py`の`_insert_s4_pick`/`_build_s4_message`が対応済み（Discord通知の見出しも"SS"/"S"表示）。
- **旧新S1（SIX_S1・6車三連単・2026-07-17 全廃）**: 3独立窓では110/103/113%だったが正規プロトコルの1年検証で最良70.3%・100%超なし（「直近だけ良い」レジーム依存を検出）。6車全域・9車・新S1候補も全滅 → 全廃。`#6S1` 行は `picks_history_r_archive` へ退避（`scripts/archive_s1_a_abolition_wt.py`）
- **A（7PLUS_A・2026-07-17 全廃）**: 正規プロトコルで検証最良88.5-94.2%・100%超なし → 全廃。`#7A` 行は `picks_history_a_archive` へ退避（同上スクリプト）。旧・買い目カット方式Aランク（〜2026-06-19）の行も同テーブルに退避済み
- **旧S1（7PLUS_R・7車三連複・2026-07-16 全廃）**: 検証期間ROI 67.3%・代替条件の全探索で黒字なし。過去行（7PLUS_R/7PLUS_CAND/7PLUS_SS/7PLUS_S）は `picks_history_r_archive` へ退避。wave-picks の SS txtセクション・#CAND 書き込み・ガミ判定は停止済み（ss_policy 等は互換のため残置）
- **S2（旧U・7PLUS_U・2026-07-21 全廃）**: 波乱見込み×穴×同ライン「逃」相方の三連複2車軸流し。廃止直前にmto閾値を4.3→4.5へ厳選したが、honest全期間再構築（`scripts/rebuild_s2_walkforward.py`・四半期walk-forwardモデル）で確認したところ4.3=ROI81.6%(1251R)→4.5=ROI84.8%(1155R)と全期間では依然として損失圏内（2024〜2025年前半が40-70%台で低迷）。対象レース数・的中率・期待値の観点で継続困難と判断し全廃。過去行（1155件）は `picks_history_u_archive` へ退避（`scripts/archive_u_m_abolition_wt.py`）。judge_u/`_process_u_candidates`等のロジックは過去日再採点・分析スクリプト互換のため残置（呼び出し元のみ停止）
- **S3（旧M・7PLUS_M・2026-07-21 全廃）**: ◎不一致×軸信頼ゲートの三連複2車軸流し。廃止直前にwin_rank単独ゲート化+目≥20倍で honest全期間ROIを95.9%→120.4%(801R)まで改善させていたが、S2と合わせて対象レース数・的中率・期待値の観点で継続困難と判断し全廃（過去の閾値変遷・リーク発覚の経緯は本ファイルのgit履歴・メモリ`keirin_composite_ratio_gate`参照）。過去行（801件）は `picks_history_m_archive` へ退避（`scripts/archive_u_m_abolition_wt.py`）。judge_m/`_process_m_candidates`/`m_axis_gate`等のロジックは過去日再採点・分析スクリプト互換のため残置（呼び出し元のみ停止）
- S1/SS/S は live 100R以上で採否判定（実賭け昇格 or 廃止。月次判定は分散的に禁物・40R全外れ月も想定内）。詳細はメモリ `keirin_s1_redesign_sweep` / `keirin_s1_win_axis_paper` / `keirin_picks_history_data_loss_2026_07_20`（S4）/ `keirin_s2_s3_tightening_2026_07_21`（S2/S3全廃の経緯）/ `keirin_s4_wt_overlap_selection_2026_07_21`（S4→SS/S再編）
- **廃止済みランク**: S/S+（`7PLUS_ST`/`7PLUS_STP`・三連単1着固定F）は 2026-07-15 に全廃・過去分もDB削除（`keirin.picks_history_st_archive` に退避）。SO≥8フィルタ・旧≤6車 SS/S/A/B・ワイドも廃止済み。旧ドキュメント・メモに残る記載は無効
- **実精算方式（2026-07-15〜）**: バックテスト・採点とも、指数ランキング＝発走前のオッズ盤面掲載車（欠車除く・落車失格含む）、落車失格絡みの買い目＝外れ計上（返還しない）、欠車のみ返還。完走者ランキングの旧方式は約2-4倍過大で全面廃止
- 見送り=miwokuri=TRUE。**実賭けランクは現在なし**（全ランクペーパー・名目賭金）。Webサマリーのトップラインは `rank IN ('SEVEN_S1','SEVEN_S4')`（S4は`gate_label`でSS/Sに分割表示）の名目合算
- `prerace_decisions_{date}.json` が採点/Web/サマリー/Discord の正本（15分前判定を事後変更しない）。キーは S1=`{rk}#S1` / S4(SS/S)=`{rk}#S4`（廃止済みのS2=`{rk}#U`/S3=`{rk}#M`キーは過去日分のみ存在）
- **落車失格レースの学習除外は棄却**（除外するとS1テスト122.8→87.9%に劣化した検証あり。落車の事前予測情報は不存在＝事後情報での母集団選別になる）。`WT_EXCLUDE_DNF_RACES=1` のオプトインのみ残置

## Web指数表示（単勝指数・複勝指数・2026-07-19導入）

- kiseki側 `/keirin` の出走表（EntryTable）に、既存の指数（`race_point`＝競走得点）に加えて
  **単勝指数**（1着専用モデル`lgbm_wt_win`の予測確率）・**複勝指数**（3着内モデルの予測確率）を
  単→複→指数の順で表示する
- `wt_entries.pred_win_pct` / `pred_top3_pct`（%スケール・小数1位）に格納。`wave-picks-wt`実行時に
  `pred_prob`/`pred_win`算出直後（候補選定の前）に全出走馬分をUPDATEする（`src/cli/main.py`）
- PG側は kiseki alembic `n0p1q2r3s4t5`で追加。SQLite側は`src/database.py::migrate_db()`
- 過去分（2024-01-01〜）は `scripts/backfill_index_pct_wt.py` で四半期walk-forwardモデルを使い
  リークなしで一括反映済み（491,582/705,079件・2026-07-19実施）。ローカルSQLiteが直近数日分
  停止しているため直近1週間程度は欠落あり得る（次回`wave-picks-wt`実行で自然に埋まる）

## スキーマ管理ルール（picks_history 等 keirin スキーマ）

- **DDL は「kiseki 側 alembic」と「本リポジトリ src/database.py::migrate_db()（SQLite用）」の両方に必ず追加する**
  （gap23 列が両方から漏れて本番 PG に手動ALTERだけで存在する「幽霊カラム」になった事故あり → 2026-07-12 に両側へ正式化済み: kiseki alembic `j6k7l8m9n0p1` / migrate_db）
- **gap カラムのスケール**: gap12 / gap34 = 0-1 スケール、**gap23 のみ pt（%ポイント・×100済み）**。歴史的経緯によるもので変更不可。読み書き時に注意
- 閾値定数（GAMI_THRESHOLD=7.0 等）は `src/cli/main.py` / `scripts/notify_prerace_wt.py` / `scripts/write_candidates_wt.py` に多重定義。**変更時は3ファイル + kiseki フロント（page.tsx）を必ず grep して揃える**
