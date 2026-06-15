# セッション引継ぎメモ（最終更新 2026-06-15）

コンテキストリセット後にここから再開すること。

---

## ★現在の状態サマリ（2026-06-15 時点）

### 実運用
- **live picks**: SS+S+A 10R / ROI 64.8% / CI[14%,126%]（判断最低100R必要・約2週間で達成見込み）
- **fav_mismatch タグ**: 1R のみ記録（2026-06-11〜）。バックテスト根拠否定済み（下記参照）
- **money-flow snapshot cron**: ユーザーの Terminal 適用待ち（`data/cron_proposal_moneyflow_20260613.txt`）

### 次のアクション（優先順）
1. **live実測の継続観察**: `scripts/live_report_wt.py` で随時確認。100R到達後（約2週間）に初回判断
2. **live成績をグレード別にトラッキング**: `picks_history` で S級/A級 を分けて観察（S级専用モデルの採否判断に使用）
3. **money-flow cron 適用**: Terminal から `crontab -e` で登録。≥1,624R 蓄積後（約9ヶ月）に `exp_moneyflow_wt.py --report`
4. **中間オッズ帯フィルタのlive検証**: 朝オッズデータが数十R蓄積後に `snapshot_morning_odds_wt.py --report` で確認

---

## ★★ 即着手できる候補（待ち不要・2026-06-15 時点）

### A. 既存DBから計算できる特徴量（新データ不要）

| 候補 | 内容 | 期待度 | 実装難度 |
|---|---|---|---|
| **宵→朝 オッズドリフト** | `wt_odds_snapshot` に `evening`(158k行)・`morning`(216k行) 既存。前日夕→朝の drift を money-flow の長時間軸として即評価可 | 高 | 低（データあり） |
| **H2H（対戦成績）** | 同一出走者間の過去勝率。`wt_entries` から算出可 | 中 | 中 |
| **venue×grade 限定 rolling WR** | S级 × 小倉 の勝率など。現行 `venue_wr` の分割版 | 中 | 低 |
| **S-model ハイパーパラメータ最適化** | S级専用モデル（TRAIN 15k行）向け設定。HOLD 88%→100%超えを狙う | 中 | 低 |

### B. WEBスクレイピング（新情報源）

| 候補 | 内容 | 期待度 | 理由 |
|---|---|---|---|
| **試走タイム** | 選手の当日練習ラップタイム。一部会場が公式サイト掲載。直近コンディションの最直接指標 | 最高 | 市場が手動でしか参照できない → 非効率の余地最大 |
| **JKA 師匠/訓練所同期情報** | jka.or.jp の選手詳細。同一師匠 → ライン連携強度の代理変数 | 中 | 競輪場の連携ネットワークを静的に補強 |
| **コンピューター指数** | netkeirin・競輪ラボ等の独自AI指数。現行モデルと独立したシグナルの可能性 | 中 | 市場に部分的にしか折り込まれていない可能性 |
| **選手コンディション記事** | 専門紙（競輪ダービー等）の記者コメント。欠場予告・機材変更情報など | 低-中 | 取得・構造化が困難 |

### 最優先推奨: 宵→朝ドリフト分析（即日着手可）

**なぜ最優先か:**
- データ収集済み・追加スクレイパー不要
- 従来の money-flow（朝→直前）に加え「前日夕→朝」という長時間軸を追加できる
- 時系列ドリフト = 「前夜に入った smart money」の方向を反映する可能性
- 実装: `exp_moneyflow_wt.py` を `evening` snapshot にも対応させる

---

## ★ 他式別オッズ特徴量 & グレード別モデル実験（2026-06-15）→ **保留**（doc35）

### 他式別オッズ（二連単・ワイド・二連複）を特徴量化
- AUC: +0.022（全特徴量中1〜3位）
- ROI: VAL 悪化・選択レース数が半減 → Phase2 不通過
- 結論: 5式別市場は統合されており裁定不可。市場特徴量を加えるとモデルが市場追随になりガミ≥5倍レースが消える。

### グレード別モデル（S级 × S-model）
- S级専用モデル: TRAIN 150%★ / VAL 107%★ / HOLD 88%（15R）
- 全体（S→S-model, A→A-model）: TRAIN 113%★ / VAL 97.3% / HOLD 107%★
- Phase2 不通過（VAL 97.3%・閾値 100% 未達）。これまでの実験で最高に近い結果。
- 今後: live成績をグレード別にトラッキングし、100R蓄積後に S级専用モデルの採否を判断

### 副産物: grade_enc バグ修正
- `feature_wt.py` の grade_map が ks 形式（GP/G1...）のまま wt では全件 fillna=1 だった
- 修正済み（S級→3 / A級→2 / L級→1）。AUC への影響は +0.0001 で実質軽微。

詳細: `docs/analysis/35-crossmarket-grade-model.md`

---

## ★★fav_mismatch リーク無し単独検証（2026-06-15）→ **バイアス崩壊確認**（doc34）

doc13 の「1168%/576% = 最強新レバー」はリーク無しで：
- fav_mismatch=True: TRAIN 79.7%, VAL 95.4%, HOLD **23.1%** → 全期間100%未達
- fav_mismatch=False: TRAIN 88.9%, VAL 55.4%, HOLD **100.2%** → 逆転（20R・小標本）

3バイアスによる 6〜25倍の過大評価。「最強新レバー」の前提は取り消し。
live タグ蓄積は継続（採否は picks_history ≥100R 後）。
詳細: `docs/analysis/34-fav-mismatch-leakfree.md`。ハーネス: `scripts/exp_fav_mismatch_leakfree_wt.py`。

---

## ★★軸精度・三連複 vs 三連単 条件付き戦略検証（2026-06-15）→ **全戦略不通過**（doc33）

**最重要発見: gap12<0.06（拮抗帯）の pred1 3着以内率は 37.9%（VAL+HOLD）**
三連単（S1/S2）は全帯で三連複より劣る（S2 VAL 32.5%/HOLD 39.8%）。
pred1単軸流し（S4）は 0.06-0.10 帯で 103.2%★（S0 の 113.9% より低い）。
gap12 0.06-0.10 帯のみが唯一 >100% だが 16R の小標本。
**doc10（1-2着BOX★頑健）はバイアス込み数字でリーク無しでは不成立確認 → ロードマップ#7クローズ済み**。
詳細: `docs/analysis/33-axis-accuracy-trifecta.md`。ハーネス: `scripts/exp_axis_trifecta_wt.py`。

---

## ★ライン構造ベース買い目設計検証（2026-06-15）→ **Phase2不通過**（doc32）

拮抗レース（gap12<0.06）でライン構造から軸選択する2×2フレームワーク。
S0(現行)/S1(強番手軸)/S2(拮抗×最強ライン)/S3(拮抗×強番手) 全て全期間ROI<100%。
軸変更も市場効率の壁を超えられないことを確認。
詳細: `docs/analysis/32-line-bet-design.md`。ハーネス: `scripts/exp_line_bet_design_wt.py`。

---

## ★ライン先頭強度・ライン内得点差 特徴量検証（2026-06-14）→ **Phase1通過・Phase2不通過・不採用**（doc31）

leader_rp_gap_vs_best / within_line_rp_gap の3特徴量。Phase1: AUC差+0.0024〜+0.0032（通過）。
Phase2: base 65〜84% vs line 71〜75%（全期間<100%・TRAIN/VAL ではbase下回る）。市場効率の壁を再確認。
詳細: `docs/analysis/31-line-features-leader-gap.md`。ハーネス: `scripts/exp_line_features_wt.py`。

---

## ★波乱予想フェーズ（2026-06-14）→ クローズ（W01〜W04全不通過）

W01（波乱モデルAUC 0.57・ランダム）・W02（4フォーメーション全滅）・W03（upset×fav_mismatch交差=構造的非重複）・W04（総合まとめ）全て不通過。
現行tier選別（gap12大）は隠れた波乱回避フィルターとして機能しており、波乱モデル追加の意義なし。
詳細: `docs/analysis/30-upset-synthesis.md`。

---

## ★★★2026-06-13 ROI100%再挑戦フェーズ（G01〜G08）完了

| Goal | 結論 |
|---|---|
| G01 backtest リーク無し化 | `backtest_wt.py` 本体に3バイアス修正・`void_rules.py` 新設 |
| G02 live実測レポート | `scripts/live_report_wt.py` 新規作成（SS+S+A 10R・ROI 65%・判断100R以上必要） |
| G03 日中オッズスナップショット | `scripts/snapshot_intraday_odds_wt.py`。cron提案=`data/cron_proposal_moneyflow_20260613.txt` |
| G04 money-flow検証ハーネス | `scripts/exp_moneyflow_wt.py`。30R≈2%・評価不能・≥1,624R蓄積後に再実行 |
| G05 気象データ収集 | `src/scraper/weather.py`。全43場・wt_weather 133万行・カバレッジ99.9% |
| G06 風特徴検証 | Phase1 不通過・無情報。`docs/analysis/24-wind-feature.md` |
| G07 高配当融合 | ゲート条件未充足でSKIP。`docs/analysis/25-highpay-fusion.md` |
| G08 ドキュメント同期 | 完了 |

---

## リーク無し再検証で結論が変わったもの（要注意）

| 項目 | 旧結論（バイアス込み） | 新結論（リーク無し） |
|---|---|---|
| **fav_mismatch** | 1168%/576%（最強新レバー） | HOLD 23.1%（バイアス産物） |
| **doc10 1-2着BOX三連単** | ★頑健 >100% | VAL 32.5%/HOLD 39.8%（不成立） |
| **全レバー全般** | 70-90%超多数 | 全て ~70-90%（doc18基準） |

**採否判断は live 実測のみ。backtest は最終オッズ上限値。**

---

## ★★★★2026-06-12(続々) リーク無し再採点（doc18・最重要）

バックテスト3バイアス発見：①欠車生存バイアス×stale odds ②≤6車完走者基準（7車混入33%）③週次再学習リーク。
本番忠実+リーク無しでは現行戦略含む全レバー3期間 ~70-90%。
採否判断は live 実測のみ。標準ハーネス=`exp_leakfree_rescore_wt.py`。
