# G43: keirin.jp 身体測定データ スクレイパー + 実験ハーネス

## 目的

keirin.jp 選手プロフィールページから身体測定データ（体重・背筋力・肺活量・太もも周径・胸囲）を取得し、
競走成績予測に有効かを Phase1 AUC で検証する実験ハーネスを整備する。

## 背景

keirin.jp の選手詳細ページ（`https://keirin.jp/pc/racerprofile?snum={player_id:06d}`）には
以下の身体測定データが掲載されている（ページ fetch で確認済み）：

| データ | 意味 | 期待度 |
|---|---|---|
| 体重 (weight_kg) | 直近の体重 | 中〜高（パワーウェイトレシオに影響） |
| 背筋力 (back_strength_kg) | 背筋力（kg）| 高（脚力の代理変数） |
| 肺活量 (lung_capacity_cc) | 肺活量（cc）| 中（有酸素能力） |
| 太もも周径 (thigh_cm) | 太もも周径（cm）| 中（筋肉量の代理） |
| 胸囲 (chest_cm) | 胸囲（cm）| 低（体格の代理） |

**既存の師匠スクレイパーと同じURL**: `scrape_mentors_wt.py` と同じ HTML 構造から取得可能。

### 既存参考実装

`scripts/scrape_mentors_wt.py`:
- 同 URL からのスクレイプ（keirin.jp/pc/racerprofile?snum=XXXXXX）
- `get_player_ids()` で `wt_entries` から全 player_id 取得
- `data/player_mentors.csv` に CSV 保存（再開可能）
- 2 req/sec のレート制限（`time.sleep(0.5)`）

### HTML 解析ヒント

師匠スクレイパーのパターン:
```python
idx = html.find("師匠")
snippet = html[idx : idx + 600]
m = re.search(r"snum=(\d{6})", snippet)
```

身体測定は同じページに以下のパターンで掲載:
- `<td>身長</td>` → 次の `<td>` に値（例: `163.0cm`）
- `<td>体重</td>` → 次の `<td>` に値（例: `59.0kg`）
- `<td>背筋力</td>` → 次の `<td>` に値（例: `190.0kg`）
- `<td>胸囲</td>` → 次の `<td>` に値（例: `100.0cm`）
- `<td>太もも</td>` または `太もも周径` → 次の `<td>` に値
- `<td>肺活量</td>` → 次の `<td>` に値（例: `4800cc`）

※ 実際の HTML 構造はページ fetch で確認すること（表組みのため th/td パターンが異なる可能性あり）

### 期間定義

| 期間 | 範囲 |
|---|---|
| TRAIN | 2023-07-01 〜 2025-06-30 |
| VAL   | 2025-07-01 〜 2026-02-28 |
| HOLD  | 2026-03-01 〜 2026-06-15 |

Phase1 gate: AUC 改善 ≥ +0.001（VAL+HOLD）

## 成果物

1. **`scripts/scrape_physicals_wt.py`**: 身体測定スクレイパー（再開可能・`--limit N` オプション付き）
2. **`scripts/exp_physical_wt.py`**: Phase1 AUC 実験ハーネス
3. **`docs/analysis/43-physical-features.md`**: 調査・実験計画レポート（実行前の構造分析まで）

## 実装手順

### Step 1: scrape_physicals_wt.py の作成

`scrape_mentors_wt.py` をベースに以下を変更：
- 出力先: `data/player_physicals.csv`
- 取得フィールド: `height_cm`, `weight_kg`, `back_strength_kg`, `lung_capacity_cc`, `thigh_cm`, `chest_cm`
- `--stats` オプション: Coverage 表示（各列の非欠損率）

実装する関数:
```python
def scrape_physicals(player_id: int, session) -> dict:
    """体重・背筋力・肺活量・太もも・胸囲を dict で返す。取得できない項目は None。"""
    ...
```

**動作確認**: `--limit 5` で5人分をテスト実行し、取得値が正常かを確認して結果を表示すること。

### Step 2: HTML パターンの確認（Step 1 の前作業）

まず 1 件フェッチして HTML を確認:
```python
import requests
r = requests.get("https://keirin.jp/pc/racerprofile?snum=015830")
# print(r.text) して身体測定の HTML 構造を確認
```

keirin.jp は静的 HTML で robots.txt は `/pc/` を ALLOW。

### Step 3: exp_physical_wt.py の作成

`scrape_mentors_wt.py` + `exp_mentor_feature_wt.py` のパターンを踏襲：

1. `data/player_physicals.csv` から測定値をロード
2. Coverage 確認（各列の非欠損率を表示）
3. `build_features_wt()` で base features 取得
4. 以下の新特徴量を追加:
   - `weight_kg`: 体重（そのまま）
   - `back_strength_kg`: 背筋力
   - `lung_capacity_cc`: 肺活量
   - `thigh_cm`: 太もも周径
   - `bsr_per_weight`: back_strength_kg / weight_kg（パワーウェイトレシオ代理）
5. Phase1 AUC 比較:
   - Base vs +weight vs +back_strength vs +lung_capacity vs +all_physical

**注意**: `scrape_physicals_wt.py` の全件実行（≈22分）は本タスクでは行わない。
`exp_physical_wt.py` はスクレイプ済みデータが存在する場合のみ AUC 評価を実行し、
データなしの場合は「スクレイプ後に再実行」旨を表示して終了する。

### Step 4: docs/analysis/43-physical-features.md の作成

以下を記載:
- 仮説・根拠
- HTML 構造解析結果（実際に確認したパターン）
- 5人サンプルのスクレイプ結果（値の範囲・Coverage）
- AUC 結果（データがある場合） or 期待評価（データなしの場合）
- 実行手順

## 受け入れ基準

- `python3 scripts/scrape_physicals_wt.py --limit 5` が実行完了し、5人分の身体測定値が出力されること
- `scripts/exp_physical_wt.py` がエラーなく完了すること（データなし → 「スクレイプ後に再実行」で正常終了も可）
- `docs/analysis/43-physical-features.md` に HTML 構造分析・サンプル値・実行手順が記載されること

## 触ってよいファイル

- `scripts/scrape_physicals_wt.py`（新規作成）
- `scripts/exp_physical_wt.py`（新規作成）
- `docs/analysis/43-physical-features.md`（新規作成）
- `data/player_physicals.csv`（新規作成・スクレイプ結果）

## 禁止事項

- `src/` 配下の変更
- `data/keirin_wt.db` への書き込み
- 全件スクレイプの実行（`--limit 5` のテストのみ）
- git commit
- crontab 変更
