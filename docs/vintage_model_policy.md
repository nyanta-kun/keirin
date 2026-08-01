# 月次vintageモデル方針（2026-07-29策定）

## 背景・経緯

2026-07-28、H2H特徴量のアドホック実験により、四半期凍結vintageモデル18本
（`lgbm_wt_eval_q2401.pkl`等）が無断で2回上書き・再学習される事故が発生した。
「同じ設定で再学習すれば元に戻る」という誤解のもとで行われたが、学習データ
自体がバックフィル実行時点(2026-07-19)からわずかに変化していたため
（race_point NaN補完修正・sb_dynバグ修正等）、実際には全く別の重みを持つ
モデルに置き換わっていた（実データ検証: pred_top3_pct平均絶対差2.42pt・
最大72.7pt・軸1位判定の入れ替わり8.5%）。オリジナルの重みは`.gitignore`
対象のため復元不可能だった。

この事故に加え、従来の四半期QUARTERS+静的TAIL_FROMという2層設計自体にも、
TAIL_FROMと実際の`lgbm_wt_eval`のtest_from（週次で「実行日-90日」にスライド）
が乖離し続けるバグがあり、直近2週間分が常にリーク区間化する構造的欠陥を
抱えていた。

これらを踏まえ、期間設定・モデル管理を根本から作り直した
（[[keirin_s7_foundational_rethink_2026_07_29]]参照）。

## データ期間ポリシー

- **2022-12-01 〜 2023-12-31（約13ヶ月）を全モデル共通の「ベース学習データ」
  として固定する。** この期間はwinticketのAI予測列(`pred_win_pct`/
  `pred_top3_pct`)が導入される前であり、常に学習データとしてのみ使う
  （評価窓には使わない）。
- **2024-01以降は月単位でモデルを学習し、スライドさせる。** 月Mのモデルは
  「ベース学習データ + 2024-01〜M月の前月末」を学習データとし、M月のレース
  （進行中の当月を含む）をスコアリングする専用モデルとして固定的に使う。
- 「未確定tail」という別概念は撤廃した。当月分がまだ終わっていなくても、
  「その月のレースは全て前月末までのデータで学習したモデルでスコアする」
  という契約は当月中ずっと不変であり、旧TAIL_FROM方式のようなドリフトが
  構造的に発生しない。

## 命名規則

- `lgbm_wt_eval_mYYMM`（複勝=top3モデル）/ `lgbm_wt_win_mYYMM`（1着=winモデル）
- 例: 2024年1月分 → `lgbm_wt_eval_m2401` / `lgbm_wt_win_m2401`
- 旧命名（`_q2401`等の四半期・`_w2`/`_w3`等の非標準命名）は新規作成しない。
  既存の旧ファイルは新体系構築後に削除する。

## 単一の正本（`src/wt_vintage_config.py`）

期間定義は`src/wt_vintage_config.py::monthly_windows()`のみを正本とする。
以前は`rebuild_s1/s7/s9/s7a/s9a_walkforward_pg.py`・`backfill_index_pct_wt.py`
の6ファイルに同一内容のQUARTERS定数がコピーされており、将来どれか1つだけ
更新されて食い違うリスクを常に抱えていた。現在はこの6ファイル
（2026-07-31のランク全面改名でファイル名は`rebuild_s1_walkforward_pg.py`・
`rebuild_7s/9s/7a/9a_walkforward_pg.py`・`backfill_index_pct_wt.py`へ変更済み）
全てが`monthly_windows()`をimportして使う設計に統一済み。

新しい月を追加する・境界を変更する場合は、このファイル1箇所を直せば
全ての本番スクリプトに反映される。

## モデル構築

`scripts/train_monthly_vintage_models.py`が2024-01から現在の月まで、
`monthly_windows()`が生成する全窓に対応するeval/winモデルペアを一括学習する。

```bash
# 全期間の初回一括構築（数時間規模のジョブ）
.venv/bin/python scripts/train_monthly_vintage_models.py

# 中断後の再開（既存pklがある月をスキップ）
.venv/bin/python scripts/train_monthly_vintage_models.py --only-missing
```

**月初の自動生成（2026-08-01・F-4対応で整備済み）**: 新しい月が始まるたびに、
その月用のvintageモデルを作成する必要がある。`scripts/ensure_monthly_vintage.sh`
が「`train_monthly_vintage_models.py --only-missing`で不足月を学習 →
`sync_models_to_vps.sh`でVPSへ配布」を1コマンドで行う（マニフェスト更新は
`save_model()`内で学習と同時に自動的に行われるため別手順は不要）。**Mac crontab
への登録（`5 0 1 * *` = 毎月1日 00:05）も 2026-08-01 にユーザー承認のうえ完了した**
（あわせて週次行末尾の rsync 直書きを `sync_models_to_vps.sh` へ差し替え済み。
反映前の crontab は Mac 上の `~/crontab_mac_backup_20260801.txt` に保全）。

## 書き込み保護

`src/models/trainer.py::save_model()`に、凍結命名規則
（`_q\d{4}$` / `_w\d+$` / `_m\d{4}$`）に一致する名前への上書きガードを実装済み
（2026-07-29）。

- 初回保存時、ファイルを自動的に読み取り専用化（chmod 444）する
  （`save_model()`を経由しない直接書き込みからも保護する第二の防御線）。
- 既存ファイルに同名で再度保存しようとすると`FileExistsError`で拒否される。
- 意図的な再構築が必要な場合のみ、`save_model(..., force=True)`
  （CLIからは`train-wt --force-overwrite-vintage`）を明示的に指定する。

これにより、2026-07-28のような「アドホック実験による無断上書き」は
構造的に発生しなくなった。

## rebuild系スクリプトの使い方

```bash
# 全期間の再構築
PYTHONPATH=. .venv/bin/python scripts/rebuild_7s_walkforward_pg.py

# 直近月（今月）のみの日次軽量再構築
PYTHONPATH=. .venv/bin/python scripts/rebuild_7s_walkforward_pg.py --tail-only

# モデル不足の月を除外して続行する（2026-08-01追加・F-4対応）
PYTHONPATH=. .venv/bin/python scripts/rebuild_7s_walkforward_pg.py --skip-missing-models
```

（2026-07-31のランク全面改名（commit `f31f84b`）で`rebuild_s7_walkforward_pg.py`
は`rebuild_7s_walkforward_pg.py`へ改名。同様に`rebuild_s7a/s9/s9a_walkforward_pg.py`
も`rebuild_7a/9s/9a_walkforward_pg.py`へ改名済み。`rebuild_s1_walkforward_pg.py`
はS1が改名対象外のため変更なし。）

S7/S9/7A/9Aは同一のインターフェース（`--dry-run`/`--tail-only`/
`--skip-missing-models`）に統一されている（`--skip-missing-models`は
2026-08-01追加・下記「月初の実害・再発防止」参照）。`rebuild_s1_walkforward_pg.py`
はS1全廃（2026-07-31）に伴い過去分析用の手動実行専用として残置されており、
本改修の対象外（S1はpicks_historyへの自動再生成を行わない運用のため、
月初のモデル不足でも実害が生じない）。

## 運用状況（2026-07-30時点）

### 構築・デプロイ完了
- 月次vintageモデル **62本**（2024-01〜2026-07の31ヶ月 × eval/win）構築完了
- 全モデルが `chmod 444` で読み取り専用化済み（書き込み保護が機能していることを実データで確認済み）
- commit `875568b` を push・VPS `git pull` 完了
- **月次モデル62本（104MB）をVPSへ rsync 配布済み**
  （VPSには従来 vintage モデルが1本も存在しなかったため、`--tail-only` が月次モデルを
  要求する新設計では配布が必須になった）
- `wt_entries.pred_win_pct`/`pred_top3_pct` を全期間 502,522件クリーン再計算済み
  （`backfill_index_pct_wt.py`）→ DB格納値も月次モデル由来になり信頼できる状態

### picks_history の再構築状況
| rank | 件数 | 状態 |
|---|---|---|
| SEVEN_S1 | 1,497 | ✅クリーン月次モデルで再構築済み（ROI 80.0%） |
| SEVEN_S7 | 575 | ✅クリーン月次モデルで再構築済み（ROI 78.6%） |
| SEVEN_7A / NINE_S9 / NINE_9A | 0 | ⚠️破棄済み（誤解防止・再構築は保留） |

破棄分（8,272件）は `data/backup/picks_history_discarded_20260730_134153.csv` に
バックアップ済み（.gitignore対象外のため未コミット）。

（注: 上記表のrank名は2026-07-31時点＝ランク全面改名（commit `f31f84b`）前の
表記のまま残している。改名後の対応は `SEVEN_S7`→`RANK_7S`・`SEVEN_7A`→
`RANK_7A`・`NINE_S9`→`RANK_9S`・`NINE_9A`→`RANK_9A`。`SEVEN_S1`は改名対象外の
ため変更なし。詳細はCLAUDE.md「現行ランク体系」節の「ランク名体系化」サブ節参照。）

### ⚠️ 日次cronの状態（重要）
`scripts/reconcile_walkforward_tail.sh` の日次cron（VPS 00:50）は
**2026-07-27にユーザー判断で停止中**（crontabに `# [PAUSED 2026-07-27 by user request]`）。
再開する場合、新設計では月次モデルが必要になるが**既に配布済みなので動作するはず**。
ただし再開前に `--tail-only` の動作確認を推奨。

### 月初の実害・再発防止（2026-08-01・F-4対応）

上記「未整備」の懸念は2026-08-01に日付が月替わりした直後に実際に発生した。
`scripts/rebuild_7s_walkforward_pg.py` が
`FileNotFoundError: data/models/lgbm_wt_eval_m2608.pkl` で失敗し、
全期間・全窓を計算し終えた最後の窓（当月分）でようやく不足が判明する設計だった
ため、**それまでの計算（実測約40分規模）が丸ごと失われた**（picks_historyへの
書き込みは全窓計算完了後にまとめて行う設計だったため、途中で例外が出ると何も
書き込まれずプロセスが落ちる）。

以下2点を整備して再発を防止した:

1. **月初の自動モデル生成**: `scripts/ensure_monthly_vintage.sh`を新設。
   `train_monthly_vintage_models.py --only-missing`で不足月のみ学習し、
   成功後に`sync_models_to_vps.sh`でVPSへ配布する。多重起動防止（mkdirロック）・
   `KEIRIN_DB_URL`未設定チェック・失敗時のDiscord通知（`system`チャンネル）を
   備える。**Mac crontab への組み込みは 2026-08-01 にユーザー承認のうえ完了済み**
   （下記エントリをそのまま登録した）:
   ```
   5 0 1 * * /Users/ysuzuki/GitHub/keirin/scripts/ensure_monthly_vintage.sh \
     >> /Users/ysuzuki/GitHub/keirin/data/logs/cron.log 2>&1
   ```
   （月初00:05実行。週次retrain(日曜23:30)と重ならず、`reconcile_walkforward_tail.sh`
   の00:50 VPS cron（現在PAUSED）や08:00の`daily_picks_wt.sh`より確実に前に完了させる
   狙い。`train_monthly_vintage_models.py`は学習失敗時に非ゼロ終了するよう
   2026-08-01に修正済み——従来は`failed`件数を集計しても常にexit 0していたため、
   自動化ラッパーが学習失敗を検知できずVPS配布まで進んでしまう不備があった。）

2. **rebuild系4本（7s/7a/9s/9a）の防御を強化**（`src/wt_rebuild_common.py`新設）:
   - **事前チェック**: build_rows(重い計算)を始める前に全窓のモデル存在を検証する。
     不足があれば`--skip-missing-models`未指定時は計算を一切開始せず即座に
     SystemExit(1)で終了する（今回のような「40分計算後に失敗して結果を失う」を
     構造的に防止）。
   - **`--skip-missing-models`オプション**: 指定時は不足窓を除外し、
     モデルが揃っている窓のみで処理を継続する。除外した窓・続行の両方を
     Discordの`system`チャンネルへ通知する。
   - **wipe/insertの単一トランザクション化**: 従来は`wipe_rows_pg()`/
     `insert_rows_pg()`が別々の接続・別トランザクションで実行されており、
     wipe成功後にinsertが失敗するとpicks_historyが空のまま残るリスクがあった
     （2026-08-01実際に発生・バックアップから復旧）。`rebuild_pg_atomic()`は
     窓ごとのwipe+insertを単一の`with get_connection()`ブロック内で実行し、
     `src/database.py::get_connection()`のコンテキストマネージャが持つ
     「正常終了時に一括commit・例外時に自動rollback」という性質を利用して
     アトミック性を確保する。
   - **0件安全策**: 挿入対象行が0件の窓はwipe自体をスキップする
     （置き換えデータが無いのに削除だけ行う事故を防止）。全窓が0件の場合は
     `get_connection()`自体を呼ばずDBに一切触れない。いずれもDiscordへ警告する。

### 汚染済み四半期モデルの削除（2026-07-31実施）

本ポリシーには「既存の旧ファイルは新体系構築後に削除する」と明記していたが、
2026-07-30の点検（B-7レビュー）で**実際には未削除のまま`data/models/`に残存**
していることが判明した。削除前に以下を確認した上で、ユーザー承認を得て削除を実施した。

- 削除前安全確認: `scripts/daily_picks_wt.sh` / `evening_picks_wt.sh` /
  `intraday_results_wt.sh` / `weekly_retrain_wt.sh` / `notify_prerace_wt.py` /
  `backfill_missing_prerace_wt.py` / `reconcile_walkforward_tail.sh` /
  `src/prediction/predictor.py` / `src/cli/main.py`、および VPS crontab
  （読み取りのみ）を grep し、`_q24`/`_q25` 系モデル名への参照が **0件**
  であることを確認。`rebuild_s1/s7/s7a/s9/s9a_walkforward_pg.py` と
  `backfill_index_pct_wt.py` は全て `wt_vintage_config.monthly_windows()`
  経由で月次モデルのみを参照していることも確認済み。
- 全ファイルが`.gitignore`対象（`data/models/*.pkl` / `*.meta.json`）で
  Git追跡外であることを`git check-ignore -v`で確認し、`rm`で削除
  （`git rm`は不使用）。
- **削除ファイル: 28件**
  （`lgbm_wt_eval_q{2401,2404,2407,2410,2501,2504,2507}.{pkl,meta.json}` 7組14件 +
  `lgbm_wt_win_q{2401,2404,2407,2410,2501,2504,2507}.{pkl,meta.json}` 7組14件）。
- 月次モデル124ファイル（62本×pkl/meta.json）・本番モデル
  （`lgbm_wt.pkl`/`lgbm_wt_win.pkl`/`lgbm_wt_eval.pkl`/`lgbm_wt_win_eval.pkl`/
  `lgbm_wt_train_only.pkl`/`lgbm_wt_val25.pkl`）・`upset_cuts_wt.json`は
  削除対象外として保持を確認済み。

## 既知の制約

- `wt_odds`は2022-12-01〜2023-12-31分が長期間欠落していたが、
  `scripts/backfill_wt_odds_2022_2023.py`（2026-07-29実行）で解消済み。
- `wt_entries.pred_win_pct`/`pred_top3_pct`（Web表示用）は
  `scripts/backfill_index_pct_wt.py`で同じ月次モデル体系を使い一括反映する
  （2026-07-29に月次体系対応・ローカルSQLite依存を除去）。
