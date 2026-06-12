# G06: 風×バンク特徴のリーク無し検証

## 目的
G05 で収集した風データが、≤6車のレース選別・予想精度に**市場の織り込みを超える**
情報を持つかを、doc17/doc18 の標準プロトコル（3期間・リーク無し学習・本番忠実採点）で検証する。

## 依存
- G05 完了（`wt_weather` テーブル・`weather_for_race` ヘルパー）

## 仮説（doc20 の残課題）
バンク詳細（直線長・カント）単体は「決まり手シフト6-8ppは実在するがレース内相対に無効」で
不通過だった。風は**日時変動する外生変数**であり、同じバンクでも日によって逃げ/差しの
有利が変わる＝レース内相対に乗る可能性が唯一残っている。

## 検証設計（事前登録・これ以外のセルを追わない）
`scripts/exp_wind_wt.py`（新規）:

1. **特徴**（候補・wt_weather × venue_info）:
   - `wind_speed` / `wind_gust`（is_indoor=1 の会場は 0 扱いまたは除外）
   - `wind_speed × style_enc`（逃げ/追い込み別の影響）
   - `wind_speed × straight_len` / `× cant_deg`（venue_info に追加済みの列）
   - 降水・気温は補助（風が主仮説）
2. **Phase1: 情報量検定**（選手コメント検証と同形式）:
   - TRAIN期間限定で学習したリーク無しLGBM（ベース=FEATURE_COLS_WT 相当）に
     風特徴を追加し、VAL/HOLDOUT の AUC 差・logloss 差を測る。
   - 決まり手（`wt_entries.factor`）の事後分析: 風速帯×決まり手分布のシフトが
     実在するか（doc20 のバンク6-8ppシフトの風版）。
   - **不通過基準: AUC差が±0.001未満なら無情報と判定し、ROI検証に進まず終了**
     （コメント検証 doc 同様）。
3. **Phase2（Phase1通過時のみ）: ROI検定**:
   - doc18セマンティクス（全エントリーランキング・出走表基準≤6車・欠車void・上限値注記）
   - 現行C0戦略 × 風ゲート（事前登録: 強風≥7m/s の屋外レースのみ等、2セルまで）
   - 3期間 TRAIN/VAL/HOLDOUT（期間は `exp_segment_first_wt.py` / `exp_leakfree_rescore_wt.py` に合わせる）
4. レポート `docs/analysis/24-wind-feature.md`:
   結論（通過/不通過を明確に）・全セルの数字・陰性なら陰性と書く
   （過去の Phase1 不通過ドキュメントと同形式）。**本番 FEATURE_COLS_WT は変更しない**
   （通過した場合も「候補」として報告に留め、採用判断はユーザー）。

## 受け入れ基準
- end-to-end 実行完了・レポート生成・判定の明記
- 本番モデル/特徴/設定への変更ゼロ

## 触ってよいファイル
- `scripts/exp_wind_wt.py`（新規）
- `docs/analysis/24-wind-feature.md`（新規）

## 禁止事項
- `src/preprocessing/feature_wt.py`（FEATURE_COLS_WT）の変更・本番モデル再学習・git commit
