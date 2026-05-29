# 継続作業メモ（2026-05-25）

コンテキストリセット後にここから再開すること。

---

## 現在の状態

- DB: `data/keirin.db`、2025-01-20〜2026-06-02、499日、36,168レース
- モデル: `data/models/lgbm.pkl`、LightGBM、CV AUC 0.7444
- 訓練/テスト分割: 2026-02-24 を境界

---

## 完了した作業

1. **場×戦略フィルター廃止**
   - `src/evaluation/backtest.py` の `VENUE_STRATEGY_FILTER = {}` に設定済み
   - 理由: テストデータで設計したため過学習。訓練期間では除外対象場の多くがROI 130〜170%と高かった
   - `src/cli/main.py` の `weekly` コマンドも `--venue-filter` デフォルトを False に変更済み

2. **パイプライン修正（以前のセッションで完了）**
   - `collect_date`: 常に全会場スキャン（`_scan_all_venues`）を使用
   - `collect_month`: 毎日iterate方式に変更

---

## 次にやること（優先順）

### Step 1: データ収集期間の拡張

keirin-station.comは**2024年以降のデータ**のみ取得可能（2023年以前は404）。
現在のDBに欠けている期間: **2024-06-01 〜 2025-01-19**（約7.5ヶ月）

```bash
source .venv/bin/activate

# 2024年6〜12月を収集（各月順番に）
python -m src.cli.main collect-month --year 2024 --month 6
python -m src.cli.main collect-month --year 2024 --month 7
python -m src.cli.main collect-month --year 2024 --month 8
python -m src.cli.main collect-month --year 2024 --month 9
python -m src.cli.main collect-month --year 2024 --month 10
python -m src.cli.main collect-month --year 2024 --month 11
python -m src.cli.main collect-month --year 2024 --month 12

# 2025年1月（既存データの前）
python -m src.cli.main collect-month --year 2025 --month 1
```

注意: 2025-01-20以前はDBに既存データが入っている可能性があるため、
`collect_month`のスキップ機能（`_get_collected_race_keys`）が働く。

### Step 2: 選手の加齢・トレンド特徴量の追加

**問題**: 現在の特徴量は `recent_win_rate_3m`（直近3ヶ月のみ）。
選手の加齢・好不調トレンドが反映されていない。

**必要な対応**:

#### 2a. スクレイパーに選手年齢・追加スタッツを追加
`src/scraper/keirin_station.py` の `scrape_race_detail()` で出走表から追加取得:
- 選手の年齢または生年月日（出走表に記載あり）
- 直近6ヶ月・12ヶ月勝率（サイトに掲載されている場合）

#### 2b. DBスキーマに列追加
`src/database.py` の `race_entries` テーブルに追加:
```sql
ALTER TABLE race_entries ADD COLUMN age INTEGER;
ALTER TABLE race_entries ADD COLUMN recent_win_rate_6m REAL;
ALTER TABLE race_entries ADD COLUMN recent_top3_rate_6m REAL;
```

#### 2c. 特徴量エンジニアリングに追加
`src/preprocessing/feature_engineer.py` に:
- `age`: 選手年齢（加齢効果の反映）
- 直近3ヶ月と6ヶ月の勝率差（トレンド: 上昇中/下降中）

#### 2d. モデル再訓練
新特徴量追加後、`python -m src.cli.main train --model lgbm`

### Step 3: データ充足後に場フィルター再評価

データが各場500R以上（全期間）集まったら:
1. 訓練期間のみのROI・信頼区間を計算
2. 95%CIが完全に100%未満の場のみをフィルター対象とする
3. テスト期間での効果を検証

---

## 重要な設計メモ

### バンクロール管理
- jiku1戦略: ケリー基準f*=5.0%、1,200円賭けに最低24,000円必要（フルKelly）
- 推奨: 100,000円以上でスタート、日次予算は残高の5〜8%
- 30,000円スタートは実質破産リスクが高い

### 訓練/テスト分割の扱い
- 現在のモデルは2026-02-24以前で訓練
- テスト期間: 2026-02-24〜現在
- 新データ追加後もこの境界を保持するか、または最新6ヶ月をテストにするか検討

### 特徴量の現状評価
- `venue_code` は特徴量に**含まれていない**（→ 場の系統的な違いがモデルに反映されていない可能性あり）
- `venue_code` を特徴量に追加することで場ごとの傾向を学習させることを検討

---

## 確認コマンド

```bash
# DB状態確認
python -m src.cli.main status

# バックテスト
python -m src.cli.main backtest

# 直近7日の場別集計
python -m src.cli.main weekly --days 7
```
