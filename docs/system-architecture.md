# システムアーキテクチャ

## 概要

競輪AI予想システム。3連複・3連単・ライン予想を機械学習で実現。
フェーズ1: CLIバックエンド（精度検証）→ フェーズ2: Webフロントエンド

---

## システム構成

```
keirin/
├── data/
│   ├── raw/              # 生データ（スクレイピング結果）
│   │   ├── races/        # レース情報
│   │   ├── players/      # 選手情報
│   │   └── odds/         # オッズデータ
│   ├── processed/        # 前処理済みデータ
│   └── features/         # 特徴量データ
├── src/
│   ├── scraper/          # データ収集モジュール
│   │   ├── netkeirin.py      # netkeirin (Selenium)
│   │   ├── keirin_station.py # 競輪ステーション (requests)
│   │   ├── oddspark.py       # OddsPark (requests)
│   │   └── pipeline.py       # 統合パイプライン
│   ├── preprocessing/    # データ前処理
│   │   ├── cleaner.py        # データクリーニング
│   │   ├── feature_engineer.py # 特徴量エンジニアリング
│   │   └── merger.py         # データ統合
│   ├── models/           # 予測モデル
│   │   ├── baseline.py       # ロジスティック回帰ベースライン
│   │   ├── xgboost_model.py  # XGBoostモデル
│   │   ├── lgbm_model.py     # LightGBMモデル
│   │   └── ensemble.py       # アンサンブル
│   ├── prediction/       # 予想生成
│   │   ├── line_predictor.py     # ライン予想
│   │   ├── trifecta_predictor.py # 3連複・3連単予想
│   │   └── formatter.py          # 結果フォーマット
│   ├── evaluation/       # モデル評価
│   │   ├── backtester.py     # バックテスト
│   │   └── metrics.py        # 評価指標
│   └── cli/              # CLIインターフェース
│       └── main.py
├── notebooks/            # Jupyter（探索・分析用）
├── tests/                # テスト
├── docs/                 # ドキュメント
├── config/
│   └── settings.yaml     # 設定ファイル
└── requirements.txt
```

---

## データフロー

```
[スクレイピング]
netkeirin (Selenium)
競輪ステーション (requests) → [データ統合] → [前処理] → [特徴量] → [モデル] → [予想出力]
OddsPark (requests)
```

---

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| スクレイピング | Python 3.12+, Selenium 4, requests, BeautifulSoup4 |
| データ管理 | SQLite（開発）→ PostgreSQL（本番）, pandas |
| ML | scikit-learn, XGBoost, LightGBM |
| CLI | argparse または click |
| Web（フェーズ2） | FastAPI（バックエンド）+ Next.js（フロントエンド） |
| 環境管理 | uv（パッケージ管理）|

---

## 予想ロジック概要

### ライン予想
1. 選手間の連携履歴・地域別グループ分析
2. 先行/番手/追い込みポジション推定
3. ライン強度スコアリング

### 3連複・3連単予想
1. 各選手の「3着以内確率」を計算（多クラス分類）
2. 上位候補の組み合わせを生成（確率の積でスコアリング）
3. ライン情報で補正（同ライン選手の連携ボーナス）
4. 期待値フィルター（オッズ × 確率 > 1.0 を推奨）

---

## 特徴量設計（予定）

### 選手特徴量
- 競走得点（スコア）
- 直近3/6/12ヶ月の勝率・連対率・3着内率
- ギア比
- 同開催場での過去成績
- 前走からの間隔・疲労指標
- ライン位置（先行/番手/追い込み）

### レース特徴量
- グレード（G1/G2/G3/F1/F2）
- 開催場コード
- 距離
- 天候・気温

### オッズ特徴量（参考）
- 単勝オッズ（市場評価の指標として）
- オッズ変動率（締め切り前後の変化）
