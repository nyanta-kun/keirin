"""S1条件選択のやり直し（正規プロトコル）。

学習: 〜2025-12-31（lgbm_wt_val26）
検証: 2026-01-01〜2026-03-31 — 条件選択はここだけで行う
テスト: 2026-04-01〜2026-07-15 — 検証で選んだ条件のみ最終確認（選択に使わない）

S1候補ファミリー: 6車・モデル順位ベースの小点数（三連単/三連複）× gap12閾値
"""
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, "/Users/ysuzuki/GitHub/keirin")

from exp_s1_v2 import collect  # 6車/7車レース収集（tri_pay/trio_pay/order3/frames/gap12）

STAKE = 100
MODEL = "lgbm_wt_val25"
VAL = ("2025-04-01", "2026-03-31")
TEST = ("2026-04-01", "2026-07-15")

# 買い目ファミリー（モデル順位 m = frames）
def V_tri_12_34(m):  # 三連単 1→2→{3,4} 2点（現行S1形）
    return {(m[0], m[1], m[2]), (m[0], m[1], m[3])}, "tri"
def V_tri_12_3(m):   # 三連単 1→2→3 1点
    return {(m[0], m[1], m[2])}, "tri"
def V_tri_box12_34(m):  # 三連単 1,2裏表→{3,4} 4点
    return {(a, b, c) for a, b in permutations(m[:2], 2) for c in m[2:4]}, "tri"
def V_tri_1_23_234(m):  # 三連単 1→{2,3}→{2,3,4} 4点
    return {(m[0], b, c) for b in m[1:3] for c in m[1:4] if c != b}, "tri"
def V_trio_123(m):   # 三連複 1-2-3 1点
    return {frozenset(m[:3])}, "trio"
def V_trio_12_34(m): # 三連複 1-2軸+{3,4} 2点
    return {frozenset({m[0], m[1], m[2]}), frozenset({m[0], m[1], m[3]})}, "trio"

VARIANTS = [
    ("三連単1→2→{3,4} 2点", V_tri_12_34),
    ("三連単1→2→3 1点", V_tri_12_3),
    ("三連単1,2裏表→{3,4} 4点", V_tri_box12_34),
    ("三連単1→{2,3}→{2,3,4} 4点", V_tri_1_23_234),
    ("三連複1-2-3 1点", V_trio_123),
    ("三連複1-2軸+{3,4} 2点", V_trio_12_34),
]
THRESHOLDS = [0.00, 0.06, 0.08, 0.10, 0.11, 0.12, 0.14]


def settle(races, fn, th):
    n = hits = bet = pay = 0
    for r in races:
        if r["gap12"] < th:
            continue
        buy, kind = fn(r["frames"])
        n += 1
        bet += len(buy) * STAKE
        if kind == "tri":
            if r["order3"] in buy:
                hits += 1
                pay += r["tri_pay"] * STAKE // 100
        else:
            if r["top3"] in buy:
                hits += 1
                pay += r["trio_pay"] * STAKE // 100
    roi = pay / bet * 100 if bet else 0
    return n, hits, roi


def main():
    val_races = [r for r in collect(MODEL, *VAL) if r["ne"] == 6]
    test_races = [r for r in collect(MODEL, *TEST) if r["ne"] == 6]
    print(f"検証 {VAL[0]}〜{VAL[1]}: 6車 {len(val_races)}R / "
          f"テスト {TEST[0]}〜{TEST[1]}: {len(test_races)}R")

    print("\n== 検証期間スイープ（条件選択はここだけ） ==")
    results = []
    for name, fn in VARIANTS:
        for th in THRESHOLDS:
            n, h, roi = settle(val_races, fn, th)
            if n >= 60:  # 最低限の母数
                results.append((roi, name, th, n, h))
                print(f"  {name:<24} gap12>={th:.2f}: n={n:4d} "
                      f"的中={h/n*100:5.1f}% ROI={roi:6.1f}%")

    # 選択規則: 検証ROI最大（的中率5〜25%の範囲内）
    eligible = [x for x in results if 5 <= x[4] / x[3] * 100 <= 25]
    eligible.sort(reverse=True)
    print("\n== 検証ROI上位（的中率5-25%制約） ==")
    for roi, name, th, n, h in eligible[:5]:
        print(f"  {name} gap12>={th:.2f}: 検証ROI {roi:.1f}% (n={n})")

    if not eligible:
        print("採用候補なし")
        return
    roi, name, th, n, h = eligible[0]
    fn = dict(VARIANTS)[name]
    tn, thit, troi = settle(test_races, fn, th)
    print(f"\n== 【選択条件のテスト評価（1回のみ）】 ==")
    print(f"  選択: {name} × gap12>={th:.2f}（検証ROI {roi:.1f}%・n={n}）")
    print(f"  テスト: n={tn} 的中={thit/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")
    # 参考: 上位2-3位のテスト値（選択ではない・頑健性の参考）
    print("  [参考] 検証2-3位のテスト値:")
    for roi2, name2, th2, n2, h2 in eligible[1:3]:
        fn2 = dict(VARIANTS)[name2]
        tn2, th2h, troi2 = settle(test_races, fn2, th2)
        print(f"    {name2} gap12>={th2:.2f}: テスト n={tn2} ROI={troi2:.1f}%")
    # 参考: 現行S1（三連単1→2→{3,4}×0.11）のテスト値
    tnc, thc, troic = settle(test_races, V_tri_12_34, 0.11)
    print(f"  [参考] 現行S1(三連単2点×0.11): テスト n={tnc} 的中={thc/tnc*100 if tnc else 0:.1f}% ROI={troic:.1f}%")


if __name__ == "__main__":
    main()
