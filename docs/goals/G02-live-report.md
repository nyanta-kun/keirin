# G02: live実測レポートCLI（採否判断の唯一の裁定者を見える化）

## 目的
doc18 の結論「採否判断は live実測（picks_history）のみ」を実行可能にする。
現状 picks_history(route='wt') は蓄積中だが、ランク別・タグ別・ドリフト込みの
統一レポートが無い。日次/随時実行できる集計CLIを作る。

## 成果物
`scripts/live_report_wt.py`（標準入出力のみ・Discord通知なし・DB書込みなし）:

1. **ランク別成績**: SS / S / A / B / WIDE 別に n・的中率・投資額・払戻・ROI・
   bootstrap CI（`scripts/roi_robustness_wt.py` の流儀を再利用）・最大払戻除去ROI。
   B と WIDE は本流(SS/S/A)と必ず別集計（notify_results_wt と同じ区分）。
2. **タグ別成績**: `data/picks/wave_picks_wt_*_detail.json`（昼・夜both）から
   `fav_mismatch` / `top3_sum`帯 / `upset_tier` 等のタグを race_key で picks_history に
   突合し、タグ有無別の live ROI を出す（fav_mismatch は 2026-06-11朝から記録開始）。
3. **朝→確定ドリフト割引率**: `data/logs/wide_monitor.jsonl` と
   `wt_odds_snapshot`(morning/evening) vs `wt_odds`(確定) から、オッズ帯別の
   ドリフト率分布（中央値・分位）を算出。「backtest上限値×割引率=期待live ROI」の換算表を出す。
4. **必要標本数の推定**: 現在の的中率・払戻分布を所与として、「ROIのCI下限が100%を
   超える判定に必要な残レース数」を bootstrap で概算（SS/S/A合算と各タグ帯について）。
5. `--from/--to` 期間指定、`--format md` で markdown 出力。

## 背景データ（実装前に必ずスキーマ確認）
- `picks_history`: route='wt'、WIDE は race_key `#W` 接尾、n_combos 列あり
- 欠車無効化済みレースは採点から除外されている（notify_results_wt._void_by_dns）
- 既知の実績: ks実測49%（backtest上限値との乖離の先例）

## 受け入れ基準
- 実データで実行し、レポートを `docs/analysis/22-live-report-initial.md` として保存
  （現時点の live 成績スナップショット＋「ROI100%判定までに必要な標本数」を明記）。
- ユニットテスト（合成picks_historyでの集計・void除外・WIDE分離）。既存テスト50pass維持。

## 触ってよいファイル
- `scripts/live_report_wt.py`（新規）
- `docs/analysis/22-live-report-initial.md`（新規）
- `tests/`（新規テストファイル）

## 禁止事項
- picks_history への書込み・cron/通知系の変更・git commit
