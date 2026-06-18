"""高配当検知×新シグナル合成 — G07（ゲート判定・スキップ記録）。

目的 (G07: docs/goals/G07-highpay-fusion.md):
  既存検証で「万車券**検知**は有効（top3_sum AUC0.73・doc14/doc21）だが
  **価格化（買って勝つ）は公開オッズ内では全滅**（doc21: 36セル全滅）」と判定済み。
  残る可能性は「検知×公開オッズ外の新シグナル（G04 money-flow / G06 風）の合成」のみ。

事前登録セル（最大4・G04/G06の結果でゲート判定）:
  a) 検知Q1 × money-flow「推奨目短縮」ゲート
  b) 検知Q1 × 「市場本命が朝→直前で交代」ゲート
  c) 検知Q1 × 強風ゲート（G06 Phase1通過時のみ）
  d) fav_mismatch × 上のいずれか最良

ゲート条件 (docs/goals/G07 記載):
  G06 が Phase1 不通過 かつ G04 の smart money 検定が無方向なら、
  本タスクは「合成の前提シグナルなし」として検証せず、その旨を docs に記録して終了。

本スクリプトの動作:
  1) ゲート条件の判定（G04/G06の結果ファイルを参照）
  2) ゲート成立なら「スキップ確定・根拠」を標準出力に出力し終了
  3) --report フラグで docs/analysis/25-highpay-fusion.md を生成

使い方:
  python3 scripts/exp_highpay_fusion_wt.py           # ゲート判定のみ
  python3 scripts/exp_highpay_fusion_wt.py --report  # docs/analysis/25 生成
"""

import sys
import argparse
from pathlib import Path
from datetime import date

# ── 参照ドキュメント（ゲート判定根拠） ─────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "analysis"
G04_DOC = DOCS_DIR / "23-moneyflow-initial.md"
G06_DOC = DOCS_DIR / "24-wind-feature.md"
G07_DOC = DOCS_DIR / "25-highpay-fusion.md"


# ── ゲート判定ロジック ──────────────────────────────────────────────────────

def check_g06_gate() -> tuple[bool, str]:
    """G06（風）が Phase1 不通過かを確認する。

    Returns:
        (gate_passed, reason)
        gate_passed=True  → 「G06不通過」条件が成立（スキップ方向）
        gate_passed=False → G06が通過しているためc)セルは検証可能
    """
    if not G06_DOC.exists():
        return False, "G06ドキュメント未生成（G06未実行）"
    text = G06_DOC.read_text(encoding="utf-8")
    # "Phase1 不通過" の文言を確認
    if "Phase1 不通過" in text or "Phase1 不通過（無情報）" in text:
        return True, "G06: Phase1 不通過（AUC差 ±0.0003 < 閾値 ±0.001）→ セルc)対象外"
    return False, "G06: Phase1 通過 → セルc)は評価対象"


def check_g04_gate() -> tuple[bool, str]:
    """G04（money-flow）の smart money 検定が「無方向」かを確認する。

    判定基準: 標本数が最小基準の5%未満なら「無方向・採否判定不能」とみなす。
    現状: 30R = 最小1624Rの1.8%

    Returns:
        (gate_passed, reason)
        gate_passed=True  → 「無方向」条件が成立（スキップ方向）
        gate_passed=False → 検定で有意方向が確認されたためa)/b)セルは評価可能
    """
    if not G04_DOC.exists():
        return False, "G04ドキュメント未生成（G04未実行）"
    text = G04_DOC.read_text(encoding="utf-8")
    # 結論文の確認: 「採否を判定できない」が含まれていれば無方向
    if "採否を判定できない" in text or "統計的結論を出せる標本数に達していない" in text:
        return True, (
            "G04: smart money 検定が無方向（標本30R = 最小1624Rの約2%・採否判定不能）"
            " → セルa)/b)対象外"
        )
    return False, "G04: 検定で有意方向あり → セルa)/b)は評価対象"


def evaluate_gate() -> dict:
    """ゲート条件を評価して結果辞書を返す。"""
    g06_skip, g06_reason = check_g06_gate()
    g04_skip, g04_reason = check_g04_gate()

    # ゲート条件: G06不通過 かつ G04無方向 → 全4セルがスキップ対象
    full_skip = g06_skip and g04_skip

    cell_status = {
        "a": "SKIP" if g04_skip else "TODO",
        "b": "SKIP" if g04_skip else "TODO",
        "c": "SKIP" if g06_skip else "TODO",
        "d": "SKIP" if (g04_skip and g06_skip) else "TODO",
    }

    return {
        "full_skip": full_skip,
        "g06_skip": g06_skip,
        "g06_reason": g06_reason,
        "g04_skip": g04_skip,
        "g04_reason": g04_reason,
        "cell_status": cell_status,
        "generated_date": str(date.today()),
    }


# ── レポート生成 ────────────────────────────────────────────────────────────

def build_report(gate: dict) -> str:
    """docs/analysis/25-highpay-fusion.md の内容を生成する。"""
    cell_rows = "\n".join(
        f"| {k} | {v} |"
        for k, v in gate["cell_status"].items()
    )

    # セル別説明
    cell_desc = {
        "a": "検知Q1 × money-flow「推奨目短縮」ゲート",
        "b": "検知Q1 × 「市場本命が朝→直前で交代」ゲート",
        "c": "検知Q1 × 強風ゲート（G06 Phase1通過時のみ）",
        "d": "fav_mismatch × 上のいずれか最良",
    }
    cell_table_rows = "\n".join(
        f"| {k} | {cell_desc[k]} | {v} |"
        for k, v in gate["cell_status"].items()
    )

    skip_reason_section = ""
    if gate["full_skip"]:
        skip_reason_section = """
## スキップ根拠

ゴールファイル（`docs/goals/G07-highpay-fusion.md`）の定義に従い、
以下の両条件が成立したため検証をスキップした:

| 条件 | 状態 | 詳細 |
|------|------|------|
| G06 Phase1 不通過 | **成立** | {g06_reason} |
| G04 smart money 無方向 | **成立** | {g04_reason} |

合成検証が意味を持つのは「上乗せするシグナルが単体で有効」な場合のみ。
現状は両シグナルとも有効性未確認であり、多重比較を増やすだけになる。
""".format(**gate)
    else:
        skip_reason_section = """
## 注記

ゲート条件が成立していないため、該当セルは今後実装が必要。
"""

    return f"""# G07: 高配当検知×新シグナル合成 — ゲート判定レポート（{gate["generated_date"]}）

> G07 の検証スクリプト: `scripts/exp_highpay_fusion_wt.py`
> ゴール定義: `docs/goals/G07-highpay-fusion.md`

---

## 背景

既存検証の結論（`docs/analysis/14`・`docs/analysis/21`）:

- 万車券**検知**は有効: `top3_sum` Q1帯で万車券率52%・AUC0.73
- **価格化（買って勝つ）は公開オッズ内では全滅**: 36セル全滅（doc21・doc18セマンティクス）
  → 「的中10-20%と万車券中心は市場価格上両立しない」

残る可能性 = 「検知（有効） × 公開オッズ外の新シグナル」の合成のみ。
本タスクは事前登録4セルでこれを検証する計画だった。

---

## ゲート条件の判定

| 依存 | 判定結果 | 詳細 |
|------|----------|------|
| G06（風検証） | **不通過** | {gate["g06_reason"]} |
| G04（money-flow） | **無方向** | {gate["g04_reason"]} |

---

## 事前登録セル別ステータス

| セル | 内容 | 判定 |
|------|------|------|
{cell_table_rows}

- **SKIP**: ゲート条件成立によりスキップ（多重比較防衛）
- **TODO**: 条件成立時に実装が必要（現状なし）
{skip_reason_section}
---

## 合成が機能しない構造的理由

```
万車券的中の算数:
  三連単10点×100円 = 1,000円/R
  的中率10% でROI100%には平均獲得配当 ≥ 10,000円 (100倍) が必要
  → 「万車券(≥100倍)を常用的に取る」こと

市場効率の壁（doc21確認済み）:
  [的中10-20%帯] は中央配当 2,000〜4,000円 → 10,000円到達が難しい
  [万車券のみ購入帯] は的中率 1〜2% → 10%的中率に届かない
  ↓
  モデルシグナルだけでは両立不可能（市場が正しい価格付けをしている）

新シグナル（money-flow・風）が有効なら突破できる可能性があったが:
  - 風: AUC差±0.0003（無情報・Phase1不通過）
  - money-flow: 標本30Rで採否判定不能（最小1,624Rの約2%）
```

---

## 今後の対応方針

| シナリオ | 条件 | 対応 |
|----------|------|------|
| money-flow 蓄積後 | ≥1,624R（約9ヶ月・G03稼働継続） | `exp_moneyflow_wt.py` 再実行 → セルa)/b)を評価 |
| money-flow 有意方向確認 | セルa)/b)いずれかが3期間>100% | 本スクリプトのTODOセクションを実装 |
| 新外部情報源獲得 | 市場が織り込みにくい情報（落車明け等） | 新しいゴールとして設計し直す |

**現時点の結論: 合成の前提シグナルが未確立のため検証不実施。**
採否判断は `picks_history` の live 実測値に依存する。

---

## 本番への影響

**ゼロ。** 本スクリプトは純粋な検証ハーネスであり、
`FEATURE_COLS_WT`・本番モデル・cron・`daily_picks_wt.sh` に一切変更を加えない。

---

*ハーネス*: `scripts/exp_highpay_fusion_wt.py`
*ゴール*: `docs/goals/G07-highpay-fusion.md`
*生成日*: {gate["generated_date"]}
"""


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G07: 高配当検知×新シグナル合成 — ゲート判定"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="docs/analysis/25-highpay-fusion.md を生成する",
    )
    args = parser.parse_args()

    gate = evaluate_gate()

    # 標準出力にゲート判定を表示
    print("=" * 60)
    print("G07: 高配当検知×新シグナル合成 — ゲート判定")
    print("=" * 60)
    print(f"  G06（風）:         {gate['g06_reason']}")
    print(f"  G04（money-flow）: {gate['g04_reason']}")
    print()

    if gate["full_skip"]:
        print("【判定】全4セルをスキップ — 合成の前提シグナルが未確立")
        print()
        print("根拠（docs/goals/G07-highpay-fusion.md より）:")
        print("  G06 が Phase1 不通過 かつ G04 の smart money 検定が無方向なら、")
        print("  本タスクは「合成の前提シグナルなし」として検証せず、")
        print("  その旨を docs に記録して終了してよい（無意味な多重比較を増やさない）。")
    else:
        print("【判定】ゲート条件が部分的または完全に不成立 → 対象セルを実装すること")

    print()
    print("セル別ステータス:")
    cell_desc = {
        "a": "検知Q1 × money-flow「推奨目短縮」ゲート",
        "b": "検知Q1 × 「市場本命が朝→直前で交代」ゲート",
        "c": "検知Q1 × 強風ゲート（G06 Phase1通過時のみ）",
        "d": "fav_mismatch × 上のいずれか最良",
    }
    for k, v in gate["cell_status"].items():
        print(f"  {k}) [{v:4s}] {cell_desc[k]}")
    print()

    if args.report:
        report_text = build_report(gate)
        G07_DOC.parent.mkdir(parents=True, exist_ok=True)
        G07_DOC.write_text(report_text, encoding="utf-8")
        print(f"レポート生成: {G07_DOC}")
    else:
        print("レポートを生成するには --report フラグを追加してください。")


if __name__ == "__main__":
    main()
