# doc44: WINTICKET 条件別成績特徴量実験

最終更新: 2026-06-16

## 仮説・背景

WINTICKET の PRELOADED_STATE JSON には選手の「条件別成績」が含まれており、
G42（doc42）の調査で4グループ・15フィールドを新規発見した。

これらは「選手が特定の条件（天候・バンク周長・時間帯・位置）でどれだけ3着内に入るか」を
表すキャリア累積統計であり、rolling_top3_3m 等が持たない以下の独立情報を含む可能性がある：

- **バンク周長マッチ**: 今日の競技場（333/400/500m）が選手の得意バンクか否か
- **時間帯適性**: 選手がナイター/デイ/モーニング戦で異なる成績を持つか
- **天候適性**: 雨天/晴天で特定選手が有利/不利か
- **位置別成績**: ライン先頭（独走型）か番手追走型かが rolling stats と独立か

| フィールドグループ | JSONキー群 | 期待度 |
|---|---|---|
| 天候別成績 | `weatherSunny/Cloudy/Rainy` | 中（雨天専門選手が一部存在？）|
| バンク周長別 | `trackDistance333/400/500` | 中（venue_wr で代替済みの可能性）|
| 時間帯別 | `hourTypeNormal/Morning/Night/Midnight` | 低〜中 |
| 位置別 | `linePositionFirst/Second/Third/lineSingleHorseman` | 中（スタイル依存）|

### アプローチ（G43 身体測定と同型）

選手ごとの直近レース1件をフェッチして条件別成績（キャリア累積統計）を取得。
`data/player_ex_stats.csv` に保存後、`wt_entries` に `player_id` でジョインして評価。

## スクレイプ結果（2026-06-16）

786レース・2434選手を約20分でスクレイプ（1.5 req/sec）。

## Coverage 確認（--stats）

```
取得済み選手数 : 2434
  weather_sunny_top3_pct             :  2429 / 2434 (99.8%)
  weather_cloudy_top3_pct            :  2420 / 2434 (99.4%)
  weather_rainy_top3_pct             :  2364 / 2434 (97.1%)
  track_333_top3_pct                 :  2371 / 2434 (97.4%)
  track_400_top3_pct                 :  2423 / 2434 (99.5%)
  track_500_top3_pct                 :  2122 / 2434 (87.2%)
  hour_normal_top3_pct               :  2412 / 2434 (99.1%)
  hour_morning_top3_pct              :  1725 / 2434 (70.9%)
  hour_night_top3_pct                :  2385 / 2434 (98.0%)
  hour_midnight_top3_pct             :  1946 / 2434 (80.0%)
  pos_first_top3_pct                 :  1681 / 2434 (69.1%)
  pos_second_top3_pct                :  1959 / 2434 (80.5%)
  pos_third_top3_pct                 :  1535 / 2434 (63.1%)
  pos_single_top3_pct                :  2062 / 2434 (84.7%)
  pos_compete_top3_pct               :   612 / 2434 (25.1%)
```

エントリー単位（wt_entries JOIN後）の notna 率:
- track_match_top3_pct: 92.6% / hour_match_top3_pct: 95.0%
- rain_vs_sunny_diff: 95.7% / pos_first_top3_pct: 69.4%

## Phase1 AUC 結果

> **結論: Phase1 通過 ★（+track / +hour 個別通過・+all 最大改善 +0.0037）**

### 評価モデル

| モデル | 特徴量 | 説明 |
|---|---|---|
| Base | FEATURE_COLS_WT | ベースライン |
| +track | +track_match_top3_pct | 当日バンク周長での top3 率 |
| +hour | +hour_match_top3_pct | 当日時間帯での top3 率 |
| +weather | +rain_vs_sunny_diff | 雨 - 晴天 top3 率差 |
| +pos | +pos_first_top3_pct | 先頭位置での top3 率 |
| +all | 全5特徴量 | 全条件特徴量 |

### AUC 表

| 期間 | Base | Δ+track | Δ+hour | Δ+weather | Δ+pos | Δ+all |
|------|------|---------|--------|-----------|-------|-------|
| VAL | 0.7721 | +0.0034 | +0.0030 | -0.0001 | +0.0008 | +0.0038 |
| HOLD | 0.7764 | +0.0034 | +0.0028 | -0.0000 | +0.0002 | +0.0036 |
| **VAL+HOLD** | **0.7734** | **+0.0034★** | **+0.0029★** | **-0.0000** | **+0.0006** | **+0.0037★** |

Phase1 gate（VAL+HOLD 改善 ≥ +0.001）: **+track・+hour・+all が通過**

## 特徴量重要度（+all モデル・上位15）

| 順位 | 特徴量 | 重要度 |
|------|--------|--------|
| 1 | score_z | 7.4% |
| 2 | race_point | 6.6% |
| 3 | line_frac | 4.9% |
| 4 | period_norm | 4.1% |
| 5 | top3_6m | 3.9% |
| 6 | rain_vs_sunny_diff ← | 3.8% |
| 7 | third_rate_norm | 3.7% |
| 8 | track_match_top3_pct ← | 3.6% |
| 9 | hour_match_top3_pct ← | 3.4% |
| 10 | quin_6m | 3.4% |

条件別成績3特徴が重要度上位10位内に入った。

## Phase2: ROI 結果

> **結論: Phase2 不通過（全モデル・全期間 < 100%）**

| モデル | TRAIN | VAL | HOLD | n (TR/VA/HO) |
|---|---|---|---|---|
| Base | 83.6% | 67.7% | 87.5% | 307/72/27 |
| +track | 85.5% | 75.8% | 81.5% | 318/75/29 |
| +hour | 86.2% | 83.5% | 85.8% | 325/78/29 |
| +all | 81.2% | 74.7% | 88.8% | 321/81/28 |

C0戦略（trio・ガミ≥5倍・≤6車・リーク無し）にて評価。

## 解釈

- **track_match_top3_pct** と **hour_match_top3_pct** は AUC を有意に改善する（+0.003台）
- しかし ROI は改善しない → 市場オッズが既にバンク周長・時間帯の選手適性を織込済み
- `venue_wr`（選手の会場別勝率・既存特徴量）で捉えられていない「バンク周長カテゴリ」の追加情報が AUC を上げるが、市場も同じ情報を持っている
- `rain_vs_sunny_diff` は AUC への貢献ゼロ（天候は当日情報で既存特徴量に織込済み）

## 結論

**Phase1 通過・Phase2 不通過 → G44 クローズ**

条件別成績（天候・バンク・時間帯・位置別）は AUC +0.003 の有意な改善をもたらすが、
市場効率の壁を超えた ROI 改善には至らない。
公開情報から構築できる特徴量の追加による Phase2 突破は極めて困難と判断。

## 実行手順

```bash
# Step1: スクレイプ（全2434選手・約20分）
python3 scripts/scrape_winticket_ex_stats.py

# Step1 テスト（10選手だけ）
python3 scripts/scrape_winticket_ex_stats.py --limit 10

# Step2: Coverage 確認
python3 scripts/scrape_winticket_ex_stats.py --stats

# Step3: AUC 実験
python3 scripts/exp_ex_condition_wt.py
```

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/scrape_winticket_ex_stats.py` | WINTICKET 条件別成績スクレイパー |
| `scripts/exp_ex_condition_wt.py` | Phase1 AUC + Phase2 ROI 実験ハーネス |
| `data/player_ex_stats.csv` | 出力 CSV（player_id + 15特徴量） |
| `docs/analysis/42-ex-winticket-extend.md` | G42 JSON調査結果（フィールド一覧）|
