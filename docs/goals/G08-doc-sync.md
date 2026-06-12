# G08: ドキュメント同期（全タスク完了後）

## 目的
G01〜G07 の成果を CLAUDE.md のドキュメント更新ルールに従って既存ドキュメント体系に反映する。

## 依存
G01〜G07 の完了報告（オーケストレータから渡される）

## 成果物
1. `docs/analysis/08-le6-roadmap.md`: 新フェーズ（money-flow / 風 / live計測基盤）を
   「次に検証する問い」に統合。クローズ済み項目の整理。各 Goal の結論を1行で反映。
2. `CONTINUATION.md`: 冒頭に 2026-06-13 セッション節を追加
   （フェーズ計画・G01〜G07の結果要約・次のアクション。既存の書式・★記法に合わせる）。
3. `docs/system-architecture.md`: 新スクリプト/コマンド（live_report_wt.py・
   snapshot_intraday_odds_wt.py・collect_weather.py・exp_*）をコマンド一覧に追加。
4. `docs/prediction-factors.md`: 特徴量変更があった場合のみ（G06 は FEATURE_COLS_WT を
   変更しない設計のため、原則「候補として言及」レベル）。「最終更新」日付と更新履歴を記入。
5. `docs/goals/README.md`: ステータス表を各 Goal の結果（done/partial/blocked と1行要約）で更新。

## 受け入れ基準
- 各ドキュメントの既存トーン・書式（日本語・★記法・テーブル形式）に一致
- 事実は G01〜G07 の完了報告と各 docs/analysis/2x にあるものだけを書く（推測で盛らない）

## 触ってよいファイル
- `CONTINUATION.md` / `docs/analysis/08-le6-roadmap.md` / `docs/system-architecture.md` /
  `docs/prediction-factors.md` / `docs/goals/README.md`

## 禁止事項
- コード変更・git commit
