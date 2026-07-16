# G42: WINTICKET EXデータ拡張調査・スクレイパー追加

## 目的

WINTICKET の PRELOADED_STATE JSON に存在するが現在未抽出の EX フィールドを特定し、
スクレイパーとDBスキーマを拡張して実験ハーネスを整備する。

## 背景

WINTICKET EXデータは11項目あり、現在取得済みは5項目のみ：

| 取得済み | 未取得（追加候補） |
|---|---|
| exSpurt（かまし成功率） | **競りの勝率**（exConflict 等） |
| exThrust（つっぱり成功率） | 位置別成績 |
| exLeftBehind（ちぎられ率） | レース種別成績 |
| exSplitLine（ちぎり率） | バンク周長別成績 |
| exSnatch（飛びつき成功率） | 天候別成績 |
|  | 時間帯別成績 |

既存スクレイパー: `src/scraper/winticket.py`
- `records_raw.get(player_id)` が EX データ源
- 現在 `rec.get("exSpurt", {}).get("percentage")` 等で percentage のみ取得

### WINTICKET ページ構造

URL パターン:
```
https://www.winticket.jp/keirin/{venue_slug}/racecard/{cup_id}/{day_index}/{race_no}
```

PRELOADED_STATE JSON から `FETCH_KEIRIN_RACECARD` → `records` → player_id → EXデータ

## 成果物

1. **`scripts/inspect_winticket_ex_fields.py`**: JSON キー調査スクリプト（既存レース1件をフェッチして records_raw の全キーを表示）
2. **`scripts/exp_ex_extend_wt.py`**: 発見されたフィールドを使った Phase1 AUC 実験ハーネス
3. **`docs/analysis/42-ex-winticket-extend.md`**: 調査結果レポート

スクレイパー本体（`src/scraper/winticket.py`）の変更は **本タスクのスコープ外**。
発見内容をレポートに記載し、スクレイパー拡張の実装提案を `docs/analysis/42-ex-winticket-extend.md` に記述する。

## 調査・実験手順

### Step 1: JSON キー調査

`scripts/inspect_winticket_ex_fields.py` を作成し、以下を実行：
1. `WinticketScraper` を使って直近の任意のレース1件をフェッチ
2. `records_raw` 内の全キーを表示（player 1件分の `rec` の全キーを `pprint` で出力）
3. `ex*` で始まるキーを抽出・列挙

```python
from src.scraper.winticket import WinticketScraper
# 既存の cup_id/day_index/race_no を適当に指定してフェッチ
# records_raw.get(player_id) の全キーを表示
```

利用可能な cup_id は `wt_races` テーブルの最新レコードから取得可能。

### Step 2: 追加フィールドの特定

調査結果から以下を確認：
- `exConflict` (競りの勝率) が存在するか
- percentage 以外に count/successCount 等の sub-key があるか
- 位置別・レース種別・バンク周長別・天候別・時間帯別の JSON キー名

### Step 3: 実験ハーネス作成

Step 2 で発見されたフィールドについて、`exp_ex_extend_wt.py` を作成：
- wt_entries に新フィールドが存在しない場合、直近データのみ（HOLD期間）でJSONから取得するロジックを実装
- Phase1 AUC チェックを実施（データが十分にある場合のみ）
- データ不足の場合は「現状では評価不可・スクレイパー拡張後に再実行」と明記

### Step 4: レポート作成

`docs/analysis/42-ex-winticket-extend.md` に以下を記載：
- 発見された全 JSON キー一覧
- 追加価値のあるフィールドの推奨リスト
- スクレイパー拡張のための実装提案（コードスニペット付き）
- AUC 結果（データがある場合）

## レポートテンプレ

```markdown
# doc42: WINTICKET EXデータ拡張調査（2026-06-15）

> **結論**: [新規発見フィールド N 件、AUC改善 X/不明]

## 発見された EX JSON キー一覧
（全キーを列挙）

## 追加推奨フィールド
（競り勝率等・JSON key・期待値）

## スクレイパー拡張実装案
（winticket.py への追加コードスニペット）

## AUC 結果（データがある場合）
（表）

## 結論と次のアクション
```

## 受け入れ基準

- `python3 scripts/inspect_winticket_ex_fields.py` が実行完了し EX フィールドの一覧が出力されること
- `docs/analysis/42-ex-winticket-extend.md` に発見フィールド一覧とスクレイパー拡張案が記載されること

## 触ってよいファイル

- `scripts/inspect_winticket_ex_fields.py`（新規作成）
- `scripts/exp_ex_extend_wt.py`（新規作成）
- `docs/analysis/42-ex-winticket-extend.md`（新規作成）

## 禁止事項

- `src/scraper/winticket.py` の変更（調査のみ・変更不可）
- `data/` 配下の DB ファイルへの書き込み
- git commit
- crontab 変更
