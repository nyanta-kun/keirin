# doc43: keirin.jp 身体測定特徴量実験

最終更新: 2026-06-15

## 仮説・背景

JKA（競輪の公益財団法人）が各選手の登録時に計測・公開している身体測定値が、
top3 予測の追加情報を持つ可能性を検証する。

| 特徴量 | 仮説 |
|--------|------|
| weight_kg | 体重は惰性力（運動量）に直結。重い選手は番手・ゴール前の直線で有利? |
| back_strength_kg | 背筋力はスプリント力の代理変数。強いほど先行・捲り有利? |
| lung_capacity_cc | 持久力の代理。長距離ライン展開で後半に息切れしないか? |
| thigh_cm | 太もも周径は筋肉量の代理。大きいほど瞬発力高? |
| chest_cm | 肺・心臓容積の代理。持久力と相関? |
| bsr_per_weight | 背筋力 / 体重 = 「パワーウェイト比」。スプリント効率の直接指標 |

**事前評価の注意点**:
- 身体測定は入団時（23歳前後）の固定値。加齢・負傷・コンディション変化を反映しない
- モデル既存特徴量（rolling_top3_3m / odds 等）が既に選手の「現在の強さ」を織込済み
- → 身体測定の独立情報量は限定的と予想されるが、長期的な体格プロフィール効果を検証

## HTML 構造確認（2026-06-15 確認済み）

URL: `https://keirin.jp/pc/racerprofile?snum={player_id:06d}`

```html
<p class="midasi2_fsz">■身長・体重・体力等</p>
<table>
  <tbody>
    <!-- 行1: ヘッダ（星座〜体重） -->
    <tr>
      <td class="tbl_header al-c">星座</td>
      <td class="tbl_header al-c">九星</td>
      <td class="tbl_header al-c">血液型</td>
      <td class="tbl_header al-c">身長</td>
      <td class="tbl_header al-c">体重</td>
      <td class="nb-a" colspan="2"></td>
    </tr>
    <!-- 行2: 値 -->
    <tr>
      <td class="al-c">双子座</td>
      <td class="al-c">一白</td>
      <td class="al-c">O </td>
      <td class="al-c">168.5cm</td>
      <td class="al-c">73.0kg</td>
      <td class="nb-a" colspan="2"></td>
    </tr>
    <!-- 行3: ヘッダ（胸囲〜肺活量） -->
    <tr>
      <td class="tbl_header al-c">胸囲</td>
      <td class="tbl_header al-c">太股</td>
      <td class="tbl_header al-c">背筋力</td>
      <td class="tbl_header al-c">肺活量</td>
    </tr>
    <!-- 行4: 値（"-" = 非公開） -->
    <tr>
      <td class="al-c">103.0cm</td>
      <td class="al-c">64.0cm</td>
      <td class="al-c">164.0kg</td>
      <td class="al-c">-</td>
    </tr>
  </tbody>
</table>
```

**パース方針**:
- `class="tbl_header"` を含む `<tr>` をヘッダ行として認識
- 次の非ヘッダ行を値行として、ヘッダと zip してマッピング
- 値: `"168.5cm"` → 正規表現 `[\d.]+` で数値部分を抽出 → float
- 欠損: `"-"` / 空文字 / `"―"` → `None`（CSV では空欄）
- keirin.jp の表記: 「太もも周径」は **「太股」** と略記（注意）

**古い選手（登録番号 10014〜10661 等）**:
- ページは存在するが `■身長・体重・体力等` セクションが表示されない
- おそらく旧形式ページ or 引退選手でデータ非公開
- スクレイパーは全フィールド `None` を返す（正常動作）

## --limit 5 実行結果（2026-06-15）

```
$ python3 scripts/scrape_physicals_wt.py --limit 5
Total: 5  Done: 0  Remaining: 5
  5/5
Done. Run with --stats for coverage summary.
```

```
$ cat data/player_physicals.csv
player_id,height_cm,weight_kg,back_strength_kg,lung_capacity_cc,thigh_cm,chest_cm
10014,,,,,,
10465,,,,,,
10577,,,,,,
10615,,,,,,
10661,,,,,,
```

**注**: wt_entries 最古登録番号（10014〜10661）はいずれも keirin.jp 新形式ページ非対応
のため全フィールドが空欄。これは正常動作（エラーではない）。

**実際の取得例（登録番号 15000〜15004 の手動テスト）**:

| player_id | height_cm | weight_kg | back_strength_kg | lung_capacity_cc | thigh_cm | chest_cm |
|-----------|-----------|-----------|-----------------|-----------------|----------|----------|
| 15000 | 176.7 | 76.0 | 189.0 | 4380.0 | 60.0 | 97.6 |
| 15001 | 174.5 | 80.0 | 203.0 | 6480.0 | 62.0 | 96.3 |
| 15002 | 171.7 | 72.8 | 166.0 | 5600.0 | 59.2 | 96.0 |
| 15003 | 173.0 | 76.5 | 171.0 | 4820.0 | 64.5 | 97.3 |
| 15004 | 177.0 | 79.0 | 192.0 | 5200.0 | 61.0 | 99.0 |

**値の範囲（5選手サンプル）**:
- 身長: 171.7〜177.0 cm（平均 174.6 cm）
- 体重: 72.8〜80.0 kg
- 背筋力: 166〜203 kg（bsr_per_weight: 2.1〜2.5）
- 肺活量: 4380〜6480 cc（player 15830 は "-" で非公開）
- 太もも周径: 59.2〜64.5 cm
- 胸囲: 96.0〜99.0 cm

**Coverage 予測**:
- 身長・体重・背筋力・太もも・胸囲: 現役選手（登録番号 14000 以上）で 80〜90% 程度と推定
- 肺活量: 50〜70% 程度（非公開選手多い）
- 古い選手（10000〜13999 台）は取得不可能（約 30%）

## 実行手順

```bash
# Step1: スクレイプ（全2719選手・約22分）
python3 scripts/scrape_physicals_wt.py

# Step1 テスト（5人だけ）
python3 scripts/scrape_physicals_wt.py --limit 5

# Step2: Coverage 確認
python3 scripts/scrape_physicals_wt.py --stats

# Step3: AUC 実験（要: 全件スクレイプ済み）
python3 scripts/exp_physical_wt.py
```

## AUC 実験結果（2026-06-15 全件スクレイプ後）

### スクレイプ結果 Coverage

```
取得済み選手数 : 2719
  height_cm           :  2430 / 2719  (89.4%)
  weight_kg           :  2426 / 2719  (89.2%)
  back_strength_kg    :  2087 / 2719  (76.8%)
  lung_capacity_cc    :  1466 / 2719  (53.9%)
  thigh_cm            :  2231 / 2719  (82.1%)
  chest_cm            :  1966 / 2719  (72.3%)
```

エントリー単位（wt_entries とのJOIN後）の nonzero 率:
- weight_kg: 95.3% / back_strength_kg: 81.0% / lung_capacity_cc: 62.3%
- thigh_cm: 87.0% / chest_cm: 77.9% / bsr_per_weight: 80.9%

### Phase1 AUC 結果

> **結論: Phase1 不通過（全特徴量で AUC 改善ゼロ〜微減）**

| 期間 | Base | +weight | +bsr | +lung | +all |
|------|------|---------|------|-------|------|
| VAL | 0.7721 | -0.0000 | -0.0002 | -0.0001 | +0.0000 |
| HOLD | 0.7763 | -0.0002 | -0.0001 | -0.0003 | -0.0002 |
| VAL+HOLD | 0.7734 | -0.0001 | -0.0002 | -0.0001 | -0.0000 |

Phase1 gate（VAL+HOLD 改善 ≥ +0.001）: **全組み合わせ不通過**

### 特徴量重要度（全身体測定追加・上位15）

weight_kg が 2.9% で 13位にランクインするが、AUC への貢献はゼロ。
既存 rolling 特徴量（score_z, race_point, top3_6m 等）に吸収されている。

| 順位 | 特徴量 | 重要度 |
|------|--------|--------|
| 1 | score_z | 7.2% |
| 2 | race_point | 7.0% |
| 3 | line_frac | 5.1% |
| ... | ... | ... |
| 13 | weight_kg | 2.9% ← |

### 解釈

- 身体測定値は **入団時の固定値**。現在の調子・成績を反映しない
- モデルの rolling 特徴量（score_z, top3_6m 等）が既に選手の「現在の実力」を完全に織込済み
- 市場オッズも同様に選手の現状評価を反映 → 固定身体値の独立情報量はゼロ
- bsr_per_weight（パワーウェイト比）も既存 score_z に吸収されていると解釈

### 結論

**Phase1 不通過 → G43 クローズ**

身体測定（体重・背筋力・肺活量・太もも・胸囲）は競走成績予測への寄与なし。
固定身体値は現役選手の成績 rolling 統計で既に内包されており独立情報量を持たない。

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/scrape_physicals_wt.py` | keirin.jp 身体測定スクレイパー |
| `scripts/exp_physical_wt.py` | Phase1 AUC + Phase2 ROI 実験ハーネス |
| `data/player_physicals.csv` | 出力 CSV（player_id + 6特徴） |
