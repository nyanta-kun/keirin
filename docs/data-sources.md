# 競輪データソース調査

## 調査日: 2026-05-22

## 結論サマリー

**公式APIは存在しない。**主要データ取得手段はWebスクレイピング。

---

## 利用可能なデータソース

### スクレイピング対象サイト

| サイト | URL | データ内容 | スクレイピング難易度 |
|--------|-----|-----------|------------------|
| **netkeirin (推奨)** | https://keirin.netkeiba.com/db/ | 選手・レース・成績の包括的DB | ⚠️ 高（Seleniumが必要） |
| 競輪ステーション | https://keirin-station.com/keirindb/search/ | レース検索・選手検索・成績 | 低〜中 |
| OddsPark競輪 | https://www.oddspark.com/keirin | レースデータ・オッズ・選手 | 低〜中 |
| WINTICKET | https://www.winticket.jp/keirin/odds/ | リアルタイムオッズ | 中（レート制限あり） |
| keirinodds.com | https://keirinodds.com/ | オッズ時系列データ（無料） | 低 |
| KEIRIN.JP公式 | https://keirin.jp/pc/search | レース・選手情報 | ブラウザ操作が必要 |

### 注意事項

- **netkeirin**: 2024年11月以降、アンチスクレイピング強化。requests+BeautifulSoupは不可。Seleniumでのブラウザシミュレーション必須。過剰アクセスで24時間IPブロック。
- **User-Agent必須**: 未設定はHTTP 400エラー
- **レート制限**: リクエスト間隔 1〜5秒以上推奨

### 有料データサービス

| サービス | 提供者 | 概要 |
|---------|--------|------|
| team-nave.com API | Team-Nave Inc. | 競輪システム開発向けDB API（法人向け） |
| AIcast | AIcast | 1万件以上の過去データを使ったAI予想API |
| Gamboo データ分析 | Gamboo Inc. | 競輪・オートレースデータ分析（一部有料） |

---

## 取得可能なデータフィールド

### レース情報
- レースID・開催スケジュール
- 開催場・トラック情報
- レースグレード（G1/G2/G3/F1/F2）
- トラックコンディション・天候
- 距離・周回数

### 選手情報（予想に重要）
- 競走得点（Racing Score）
- パワーランク
- 直近成績（勝率・連対率・3着内率）
- **ギア比**（重要特徴量）
  - 男子: 上限4.00
  - 女子: 上限3.80
  - 多用: 3.92, 3.85, 3.86, 3.93
- フレーム・自転車スペック
- 地元・近況コメント

### 並び（ライン）情報
- 選手グループ・連携情報
- 先行・番手・追い込み戦術

### オッズ・配当情報
- 単勝・複勝・2連複・2連単・3連複・3連単オッズ
- オッズ時系列データ
- 払戻金情報

---

## 開発参考リソース

- [競輪AI予想開発 (Note.com)](https://note.com/pabu_keirin/n/n284c4988c466)
- [ニューラルネットワーク競輪予想 (Qiita)](https://qiita.com/GOTOinfinity/items/877fc90168d84d8d1297)
- [競輪データスクレイピング (Note.com)](https://note.com/shota_umemura/n/n59819cf2a663)
- [競輪ギア比解説](https://keirin-brother.com/beginner/gear/)
