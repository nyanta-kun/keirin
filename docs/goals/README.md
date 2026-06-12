# ゴール管理 — ROI100%再挑戦フェーズ計画（2026-06-13）

三連複・三連単を対象に、レース選定・予想方法・購入効率化で高配当的中×ROI100%を目指す。
データベースは winticket（必要に応じ外部追加取得）。

## 前提（全タスク共通・必読）

過去検証の結論（`docs/analysis/18` ほか）により、**公開オッズ内の情報だけでは全レバーが
リーク無し再採点で~70-90%**（7+はクローズ、≤6車も同様の壁の可能性大）。本計画は:

1. **検証基盤を本番忠実・リーク無しに統一**する（過去の黒字捏造を二度と起こさない）
2. **公開オッズを超える残候補=「money-flow（朝→直前オッズ変動）」「風（外部気象）」**を収集・検証する
3. **採否判断は live実測（picks_history）のみ**——その計測基盤を整える

### 全実装タスク共通の規律（doc18セマンティクス）
- ランキングは**全エントリー（出走表）**で行う。欠車を事前に知らない。
- ≤6車判定は**出走表基準**（完走者基準だと7車立てが33%混入する）
- モデルは**評価期間外で学習**（週次再学習済み `lgbm_wt` をバックテストに使うとリーク）
- 参考標準実装: `scripts/exp_leakfree_rescore_wt.py`
- `finish_order=0` は欠車=着外。top3判定は `between(1,3)`
- 最終オッズ(`wt_odds`)は**上限値**（実運用は下振れ）
- 検証合格基準: 3期間（TRAIN/VAL/HOLDOUT）すべて>100% + bootstrap CI + 最大払戻除去
- **本番挙動（cron・`daily_picks_wt.sh` 既定）の変更禁止**。新機能は opt-in
- **crontab への書込み禁止**（リモートからはTCCでハング）→ 提案ファイル生成のみ
- **git commit 禁止**（オーケストレータが実施）
- 既存テスト 50 pass を維持＋新規分を追加
- CLAUDE.md のドキュメント更新ルールに従う

## フェーズとタスク

| Phase | Goal | 内容 | 依存 |
|---|---|---|---|
| 1 検証・計測基盤 | [G01](G01-backtest-leakfree.md) | `backtest_wt.py` 本体のリーク無し化（doc18残タスク） | なし |
| 1 検証・計測基盤 | [G02](G02-live-report.md) | live実測レポートCLI（picks_history集計・ドリフト割引率・必要標本数） | なし |
| 2 新情報源 | [G03](G03-odds-snapshot-intraday.md) | オッズスナップショット多時点化（money-flow素材） | なし |
| 2 新情報源 | [G04](G04-moneyflow-harness.md) | money-flow検証ハーネス＋初期観察 | なし(G03と独立) |
| 2 新情報源 | [G05](G05-weather-collection.md) | 気象（風）データ収集・バックフィル | なし |
| 2 新情報源 | [G06](G06-wind-verification.md) | 風×バンク特徴のリーク無し検証 | G05 |
| 3 統合 | [G07](G07-highpay-fusion.md) | 高配当検知×新シグナル合成（事前登録セルのみ） | G04, G06 |
| 4 整備 | [G08](G08-doc-sync.md) | ドキュメント同期（roadmap/CONTINUATION/architecture） | 全部 |

## ステータス

| Goal | 状態 | 1行要約 |
|---|---|---|
| G01 | done | `backtest_wt.py` 本体に3バイアス（欠車生存バイアス・≤6車フィルタ位置・欠車void）を修正移植。`void_rules.py` 新設。スポットチェック ROI 80.4%（doc18 の~84% と同オーダー）。|
| G02 | done | `scripts/live_report_wt.py` を新規作成。ランク別・タグ別成績集計・ドリフト分布・必要標本数推定を出力。初期観察 `docs/analysis/22-live-report-initial.md` に保存（SS+S+A 9R・ROI 56%・判断には最低100R必要）。|
| G03 | done | `scripts/snapshot_intraday_odds_wt.py` 新規作成。当日未発走レースのオッズを時点スナップショットとして保存。初回実行で72レース/25,263行保存確認。cron提案ファイル生成（書込はユーザー操作）。|
| G04 | done | `scripts/exp_moneyflow_wt.py` 新規作成。ドリフト記述統計・ガミ帯反転率・スマートマネー仮説検証の3事前登録セルを実装。初期観察 `docs/analysis/23-moneyflow-initial.md` に保存（最小標本数≈1,624R・約9ヶ月待ち）。|
| G05 | done | `src/scraper/weather.py` 新規作成。Open-Meteo API 経由で全43場・2022-12〜2026-06 をバックフィル。wt_weather 1,331,280行・カバレッジ99.9%達成。|
| G06 | done | `scripts/exp_wind_wt.py` 新規作成。風×バンク特徴をリーク無し LGBM に追加して AUC 差を測定 → VAL +0.0002・HOLD +0.0003 で Phase1 不通過（閾値 ±0.001 未満）。本番変更ゼロ。|
| G07 | done | ゲート条件（G06 Phase1 不通過 かつ G04 標本 30R ≈ 最小 1,624R の 2%）が成立したため事前登録4セル全てを SKIP。`scripts/exp_highpay_fusion_wt.py` と `docs/analysis/25-highpay-fusion.md` を生成。|
| G08 | done | CONTINUATION.md・`docs/analysis/08-le6-roadmap.md`・`docs/system-architecture.md`・`docs/prediction-factors.md`・`docs/goals/README.md` を G01〜G07 の完了報告を正として同期。|
