# W04: 波乱予想フェーズ統合（W01〜W03 の知見合成）

## 目的

W01・W02・W03 の結果を統合し、**波乱予想AIフェーズの最終判定と次のアクションを確定する**。
3つの検証が揃ってから実行する（W01/W02/W03 完了が前提条件）。

## 入力（必読ファイル）

- `docs/analysis/27-upset-leakfree.md`（W01の結果）
- `docs/analysis/28-upset-bet-design.md`（W02の結果）
- `docs/analysis/29-upset-fav-mismatch.md`（W03の結果）
- `docs/analysis/08-le6-roadmap.md`（既存ロードマップとの整合）
- `docs/goals/README.md`（ゴール管理の更新対象）

## 判定ロジック

W01/W02/W03 の各結果を読み、以下の判定を行う:

### 通過判定（優先順）

| 優先度 | 条件 | アクション |
|--------|------|-----------|
| 1 | W03 交差（upset Q4 + fav_mismatch）が VAL・HOLD 両方 CI下限>100% | opt-in CLI フラグ設計（下記） |
| 2 | W02 いずれかのフォーメーションが VAL・HOLD 両方 CI下限>100% | フォーメーション変更を opt-in で実装 |
| 3 | W01 Q4 が VAL・HOLD 両方 CI下限>100% | upset フィルタを opt-in で実装 |
| 4 | 全不通過 | 波乱予想フェーズをクローズ・知見を記録 |

複数通過の場合は独立性を確認し、相関が高ければ最強条件のみ採用。

### opt-in CLI フラグ（通過条件があった場合）

通過条件の実装は `wave-picks-wt` コマンドの新オプションとして opt-in で追加:

```bash
# 例: upset フィルタ（W01通過の場合）
python -m src.cli.main wave-picks-wt --upset-filter

# 例: クロスライン代替買い目（W02-F1通過の場合）
python -m src.cli.main wave-picks-wt --upset-cross-line

# 例: 波乱×fav_mismatch 交差（W03通過の場合）
python -m src.cli.main wave-picks-wt --upset-fav-mismatch
```

実装は **スクリプト側の概念実証のみ**（`src/` への実装は別タスク）。
「こういうフラグを追加すべき」という仕様書として記録する。

## 成果物

1. **`docs/analysis/30-upset-synthesis.md`**: 統合レポート（下記テンプレに従う）
2. **`docs/goals/README.md` の更新**: W01〜W04 のステータス欄を記入
3. **`CONTINUATION.md` の先頭追記**: 波乱フェーズの結論（3行以内）

### ドキュメント更新は最小限

`CONTINUATION.md` の先頭に追記するのは **波乱フェーズ結論の1パラグラフのみ**。
`08-le6-roadmap.md` の更新は波乱フェーズ結論（通過/クローズ）のみを1行追記。

## レポートテンプレ（`docs/analysis/30-upset-synthesis.md`）

```markdown
# 30: 波乱予想フェーズ統合レポート

## 0. 最終判定（先出し）

| Goal | 結論 | VAL ROI | HOLD ROI |
|------|------|---------|---------|
| W01 upset Q4 単独 | 通過/不通過 | X% | X% |
| W02 代替フォーメーション | 通過/不通過（最良案） | X% | X% |
| W03 upset×fav_mismatch | 通過/不通過 | X% | X% |

**採用アクション**: [具体的な次ステップ or クローズ]

## 1. W01 要約（upset Q4 ROI の再評価）
## 2. W02 要約（代替フォーメーションの比較）
## 3. W03 要約（交差条件の検証）
## 4. 統合判定と独立性確認
## 5. 実装仕様（通過条件がある場合）or クローズ理由（全不通過の場合）
## 6. 構造的知見（市場効率・波乱予測の限界）
## 7. 次のアクション
```

## 受け入れ基準

- `docs/analysis/30-upset-synthesis.md` が作成されること
- W01/W02/W03 の全結果が要約されること
- `docs/goals/README.md` の W01〜W04 ステータスが記入されること
- `CONTINUATION.md` の先頭に波乱フェーズ結論が追記されること
- 実装仕様 or クローズ宣言が明記されること

## 触ってよいファイル

- `docs/analysis/30-upset-synthesis.md`（新規作成）
- `docs/goals/README.md`（W01-W04 ステータス欄のみ更新）
- `CONTINUATION.md`（先頭への追記のみ）
- `docs/analysis/08-le6-roadmap.md`（1行追記のみ）

## 禁止事項

- `src/` の変更
- `wave-picks-wt` / cron の実際の変更（仕様書のみ）
- git commit
