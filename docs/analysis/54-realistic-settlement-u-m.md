# doc54: 実精算方式への全面改定と S/S+ 全廃・U/M ペーパートレード新設（2026-07-15〜16）

## 経緯

doc65（survivor bias 調査・2026-07-15）で「過去のプラス表示は欠車生存バイアスの人工物」と確定した
のを受け、ユーザー指示により以下を実施した。

## 1. S/S+（三連単1着固定F・7PLUS_ST/STP）全廃（2026-07-15・d753cce/5952a5c）

- 優位性なし（実精算で ROI 70-90% = 控除率の壁）のため、生成・発走前購入・通知・採点・集計から全除去
- **過去分も破棄**: keirin.picks_history の 3,684行 + model_evaluation 4行を削除
  （`keirin.picks_history_st_archive` / `keirin.model_evaluation_st_archive` に退避済み）
- 一時スクリプト（rescore_today_newrules / backfill_july_newrules_wt / exp_st_* ）も削除

## 2. 実精算方式（2026-07-15・4e7b1d1）

バックテスト（save_model_eval / eval_clean_split_wt）・ライブ採点（notify_results_wt）を統一:

- 指数ランキング・買い目 = **発走前のオッズ盤面掲載車**（欠車除く・落車失格含む）
- 落車・失格・棄権絡みの買い目 = **購入のまま外れ計上**（返還しない＝実際の車券精算と同一）
- 欠車（盤面から消えている＝発走前に判明・実際も返還）のみ返還
- 欠車と落車失格の判別 = 最終オッズ盤面への掲載有無（`_board_frames`）
- 旧・完走者ランキング方式は未来情報リークで約2〜4倍過大 → 全面廃止
- picks_history は 2025-07-01〜2026-07-09 を実精算で再構築（apply_picks_rows_wt +
  backfill_realistic_ss_wt）。7/10以降は実ライブ判定（prerace_decisions 正本）を保持

実精算での SS 公式値: モデル評価 VAL 501R ROI 63.5% / HOLD 166R 76.2%、
クリーン分割テスト91日 157R 的中26.1% ROI 77.4% = **SS は損益分岐圏**。

## 3. SS ポリシー簡素化（2026-07-16・b132f9a）

doc53 の 4分戦カット・ライン格差≥1.5増額は実精算再検証（exp_ss_policy_realistic_wt.py）で
窓間方向不一致（4分戦: テスト有効/VAL逆効果、格差帯: テスト110%/VAL56%）と判明し削除。
**選抜カットのみ維持**（選抜セグメントは全3窓一貫で ROI 26/39/0%）。常に100円/点。

## 4. U/M ペーパートレード新設（2026-07-16・77e4ec9 / 05b86a6）

一連の探索（波乱予測→穴の型→軸2選定→相手選定）で、大標本2窓とも ROI>100% の構成を2本発見:

| ランク | 条件 | テスト | VAL |
|---|---|---|---|
| U (7PLUS_U) | 波乱見込み × 穴（市場4-7位∧モデル3位内∧先頭/番手）× 同L逃相方 × 三連複目≥15 | 110.9% | 118.9% |
| M (7PLUS_M) | 波乱見込み × WT◎≠システム◎ × システム◎ × 同L逃相方 × 目≥15 | 121.8% | 120.9% |

- 波乱見込み = 指数エントロピー≥1.84 ∧ 盤面min三連複≥4.3（2026-01〜06 lgbm_wt 分布 Q3 で凍結）
- 閾値感度は両窓で単調勾配（孤立ピークでない）・◎/WT印と独立・券種は三連複が最良
  （三連単穴→相方は分散大、逃相方→穴は織り込み済み）
- 月別は高分散（U: 13ヶ月合計109.5%・勝ち月6/13・40R全外れ月あり）
- **多重比較上振れの懸念があるため賭けなしのペーパー運用**。live 100R以上（2026-08末目安）で採否判定
- パイプライン: wave-picks-wt が `_u_candidates.json` / `_m_candidates.json` →
  notify_prerace_wt の judge_u / judge_m（15分前オッズ）→ decisions `{rk}#U`/`{rk}#M` →
  picks_history `{rk}#7U`/`{rk}#7M` → notify_results_wt 採点（ヘッダー合計不算入・独立行）

## 5. 不成立と確定した領域（再検証不要）

- ◎一致（本命側・全体84%）: 軸2×目オッズ帯×レース選別×ワイド1点の全構成で ROI 65-84%。
  的中率は最大64%（◎一致∧ent≤1.81×市場2位）まで上がるがオッズが正確に相殺（的中率×オッズ≒0.75）
- 見送り緩和×高オッズ目のみ購入（62-75%）・欠車判明レースの残存目（窓間不一致）・
  波乱見込みの単純券種化（ワイド77-82%/二車複82-95%）・3車目の選手属性カット（全属性で窓間反転）

検証ハーネス: exp_upset_trio30 / exp_upset_dark_riders / exp_dark_rider_roi / exp_dark_axis2 /
exp_dark_pair_features / exp_mismatch_m1 / exp_agree_m1 / exp_share_threshold /
exp_scratch_races_realistic / exp_skip_relax_highodds / exp_ss_policy_realistic（全て scripts/）
