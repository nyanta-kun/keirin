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

## 現行ランク体系（2026-07-17〜・実精算方式・**S2/S3 の2ペーパーランクのみ**）

**2026-07-17 再設計確定**: 正規プロトコル（学習〜2025-03-31／検証=2025-04-01〜2026-03-31 の1年で条件選択
／テスト=2026-04-01〜07-15 で1回評価・モデル `lgbm_wt_val25`）による全ランク再検証の結果、
**合格は S2（現行条件のまま）と S3（新定義）のみ**。S1（6車三連単）・A（一致波乱二連単）は
検証ROI100%超の条件が存在せず全廃（新S1候補スイープ=適応型2車軸トリオ/m1 1着固定三連単も
検証ROI≥95%のセルなしで全滅）。
**2026-07-16 指数改定: 競走得点トレンド4特徴を追加（FEATURE_COLS_WT 40→44・全モデル再学習・バックフィル再構築済み）**

- **S2（旧U）** = `7PLUS_U`（suffix `#7U`・**ペーパートレード＝賭けなし記録のみ**）: 波乱見込み（指数エントロピー≥1.84 ∧ 盤面min三連複≥4.3・凍結値）× 穴（市場4-7位∧モデル3位内∧ライン先頭/番手）× 同ライン「逃」相方の三連複2車軸流し・目≥15倍のみ。正規プロトコル: 検証127.8%(n=320)→テスト117.1%(n=87)
- **S3（2026-07-17 新定義・2026-07-19 3way OR拡張）** = `7PLUS_M`（suffix `#7M`・ペーパー）: **WT◎≠システム◎（不一致）** ∧ 以下いずれか（軸信頼ゲート）: (a) gap12≥0.10（`M_GAP12_MIN`）(b) システム◎の1着モデル(`lgbm_wt_win`)内順位≥3（`M_WIN_RANK_MIN`・Phase B 2026-07-19導入）(c) システム◎の p_win/p_top3 比≤0.30（`M_RATIO_MAX`・(b)の連続量版・2026-07-19導入）。× システム◎ × 同ライン「逃」相方の三連複2車軸流し・目≥15倍のみ。**旧定義の波乱ゲート（entropy≥1.84∧mto≥4.3）は廃止**。正規プロトコル: gap12単独 検証111.8%(221R)→テスト104.4%(62R) → gap12 OR win_rank 検証158.2%(531R)→テスト149.5%(152R・母数1.7倍) → gap12 OR win_rank OR ratio 検証158.6%(671R)→テスト154.3%(186R・母数さらに+22〜26%)（`exp_axis_redesign.py` → `exp_win_axis_sweep_wt.py` → `exp_composite_prob_diff_wt.py`）。S2(buy)と同一ペアなら S2 優先で S3 は記録しない。過去分は `scripts/backfill_um_rank_wt.py --wipe-m`
- **S1（SIX_S1・6車三連単・2026-07-17 全廃）**: 3独立窓では110/103/113%だったが正規プロトコルの1年検証で最良70.3%・100%超なし（「直近だけ良い」レジーム依存を検出）。6車全域・9車・新S1候補も全滅 → 全廃。`#6S1` 行は `picks_history_r_archive` へ退避（`scripts/archive_s1_a_abolition_wt.py`）
- **A（7PLUS_A・2026-07-17 全廃）**: 正規プロトコルで検証最良88.5-94.2%・100%超なし → 全廃。`#7A` 行は `picks_history_a_archive` へ退避（同上スクリプト）。旧・買い目カット方式Aランク（〜2026-06-19）の行も同テーブルに退避済み
- **旧S1（7PLUS_R・7車三連複・2026-07-16 全廃）**: 検証期間ROI 67.3%・代替条件の全探索で黒字なし。過去行（7PLUS_R/7PLUS_CAND/7PLUS_SS/7PLUS_S）は `picks_history_r_archive` へ退避。wave-picks の SS txtセクション・#CAND 書き込み・ガミ判定は停止済み（ss_policy 等は互換のため残置）
- S2/S3 は live 100R以上で採否判定（実賭け昇格 or 廃止。月次判定は分散的に禁物・40R全外れ月も想定内）。詳細はメモリ `keirin_u_rank_paper` / `keirin_s1_redesign_sweep`
- **廃止済みランク**: S/S+（`7PLUS_ST`/`7PLUS_STP`・三連単1着固定F）は 2026-07-15 に全廃・過去分もDB削除（`keirin.picks_history_st_archive` に退避）。SO≥8フィルタ・旧≤6車 SS/S/A/B・ワイドも廃止済み。旧ドキュメント・メモに残る記載は無効
- **実精算方式（2026-07-15〜）**: バックテスト・採点とも、指数ランキング＝発走前のオッズ盤面掲載車（欠車除く・落車失格含む）、落車失格絡みの買い目＝外れ計上（返還しない）、欠車のみ返還。完走者ランキングの旧方式は約2-4倍過大で全面廃止
- 見送り=miwokuri=TRUE。**実賭けランクは現在なし**（全ランクペーパー・名目賭金）。Webサマリーのトップラインは `rank IN ('7PLUS_U','7PLUS_M')` の名目合算
- `prerace_decisions_{date}.json` が採点/Web/サマリー/Discord の正本（15分前判定を事後変更しない）。キーは S2=`{rk}#U` / S3=`{rk}#M`
- **落車失格レースの学習除外は棄却**（除外するとS1テスト122.8→87.9%に劣化した検証あり。落車の事前予測情報は不存在＝事後情報での母集団選別になる）。`WT_EXCLUDE_DNF_RACES=1` のオプトインのみ残置

## スキーマ管理ルール（picks_history 等 keirin スキーマ）

- **DDL は「kiseki 側 alembic」と「本リポジトリ src/database.py::migrate_db()（SQLite用）」の両方に必ず追加する**
  （gap23 列が両方から漏れて本番 PG に手動ALTERだけで存在する「幽霊カラム」になった事故あり → 2026-07-12 に両側へ正式化済み: kiseki alembic `j6k7l8m9n0p1` / migrate_db）
- **gap カラムのスケール**: gap12 / gap34 = 0-1 スケール、**gap23 のみ pt（%ポイント・×100済み）**。歴史的経緯によるもので変更不可。読み書き時に注意
- 閾値定数（GAMI_THRESHOLD=7.0 等）は `src/cli/main.py` / `scripts/notify_prerace_wt.py` / `scripts/write_candidates_wt.py` に多重定義。**変更時は3ファイル + kiseki フロント（page.tsx）を必ず grep して揃える**
