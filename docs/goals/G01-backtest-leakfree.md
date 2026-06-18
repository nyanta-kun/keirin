# G01: backtest_wt.py 本体のリーク無し化（doc18 残タスク）

## 目的
`docs/analysis/18-backtest-bias-rescore.md` で発見された3バイアスのうち①②を
`src/evaluation/backtest_wt.py` 本体に移植し、今後のすべてのバックテストが
本番忠実セマンティクスで動くようにする。③（モデルリーク）は CLI オプションで回避可能にする。

## 背景（必読: docs/analysis/18、CLAUDE.md 末尾の3バイアス注意）
- ①欠車生存バイアス: 旧 `_apply_pred_prob_wt` 系は完走者のみでランキング=欠車を事前に知っている。
  stale odds と組み合わさり黒字が捏造される（差分44RだけでROI842%）。
- ②≤6車判定が完走者基準: `_filter_by_n_riders(6)` を欠車除去後に適用すると7車立てが33%混入。
- 標準実装は `scripts/exp_leakfree_rescore_wt.py`（出走表基準≤6車・全エントリーでランキング・
  欠車は `notify_results_wt._void_by_dns` 同等の返還/除外ルール）。これを読んでから着手すること。

## 成果物
1. `src/evaluation/backtest_wt.py` 修正:
   - ランキング（pred_prob順位付け）は**全エントリー**で実施
   - `≤6車` フィルタは**出走表基準**（`wt_entries` の行数 / `wt_races.n_entries`）
   - 採点時の欠車処理: 軸(p1/p2)欠車=レース無効（返還・不計上）/ 相手欠車=その目のみ除外 /
     全相手欠車=無効（`notify_results_wt._void_by_dns` と同一ルール。ロジックの共通化が可能なら共通化）
   - 通常/`--tiered`/`--value` の全モードに適用
2. `src/cli/main.py` の `backtest-wt`: 評価用モデル名を指定するオプション（既存に `--model` 系が
   あるか確認し、無ければ追加）。ヘルプに「週次再学習済み lgbm_wt は評価期間にリークする」旨を明記。
3. テスト: 欠車を含む合成データで ①全エントリーランキング ②出走表基準フィルタ ③void採点 を検証。

## 受け入れ基準
- 修正後の `backtest-wt` を doc18 と同条件（C0現行3層×ガミ≥5・リーク無しモデル・同期間）で
  実行し、`exp_leakfree_rescore_wt.py` の結果（~85%帯）とROIがオーダー一致することをスポットチェック。
  一致しない場合は差分原因を特定して報告（無理に合わせない）。
- 既存テスト全pass＋新規テスト。

## 触ってよいファイル（これ以外は変更禁止）
- `src/evaluation/backtest_wt.py`
- `src/cli/main.py`（backtest-wt コマンド部のみ）
- `src/notification/` 配下は**読み取りのみ**（void ルール参照。共通化する場合は新規 util モジュール
  `src/evaluation/void_rules.py` を新設し notify 側は変更しない）
- `tests/`（新規テストファイル）

## 禁止事項
- `wave-picks-wt` / cron / `notify_results_wt.py` の挙動変更
- git commit
