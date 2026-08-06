"""買い目の表示整形（フォーメーション表記への畳み込み）。

7H1 は三連複+三連単の併買のため、買い目は全目の列挙で持っている:

    三連複 10点: 1=3=7 1=3=5 1=3=4 3=5=7 3=4=7 3=4=5 1=5=7 1=4=7 1=4=5 4=5=7
    三連単  8点: 7-3-1 7-3-5 7-3-4 7-3-6 7-1-3 7-1-5 7-1-4 7-1-6

これは Discord でも Web でもほぼ読めない。構造が
「N車BOX」「1着1車 × 2着n車 × 3着m車」であることを検算したうえで畳む:

    三連複 1,3,4,5,7 BOX
    三連単 7-1,3-1,3,4,5,6

⚠️ **畳めない構造なら None を返し、呼び出し側は元の列挙を出すこと。**
   省略して誤った買い目を見せるより冗長を選ぶ。

⚠️ **picks_history.pred_combo に書く文字列は畳んではいけない。** 採点・再構築が
   全目の列挙を前提にしている（`notify_results_wt.py`）。畳むのは表示だけ。

Web 側（kiseki `frontend/src/lib/keirinCombo.ts`）に同じロジックがある。
表記を変えるときは両方直すこと。
"""
from __future__ import annotations

from itertools import combinations


def fold_trio_box(legs: list[str]) -> str | None:
    """三連複の全目が「N車BOX」ならその表記を返す。違えば None。

    legs: ["1=3=7", "1=3=5", ...]（順不同）
    """
    try:
        sets = [tuple(sorted(int(x) for x in leg.split("="))) for leg in legs]
    except ValueError:
        return None
    if not sets or any(len(s) != 3 for s in sets):
        return None
    got = set(sets)
    if len(got) != len(sets):          # 重複目があるなら BOX とは呼べない
        return None
    cars = sorted({x for s in sets for x in s})
    if got != set(combinations(cars, 3)):
        return None
    return ",".join(str(c) for c in cars) + " BOX"


def fold_trifecta_formation(legs: list[str]) -> str | None:
    """三連単の全目が「1着-2着候補-3着候補」の直積ならその表記を返す。違えば None。

    legs: ["7-3-1", "7-3-5", ...]（順不同）
    """
    try:
        rows = [tuple(int(x) for x in leg.split("-")) for leg in legs]
    except ValueError:
        return None
    if not rows or any(len(r) != 3 for r in rows):
        return None
    got = set(rows)
    if len(got) != len(rows):
        return None
    if len({r[0] for r in rows}) != 1:  # 1着が複数あるならフォーメーションでない
        return None
    first = rows[0][0]
    seconds = sorted({r[1] for r in rows})
    thirds = sorted({r[2] for r in rows})
    # 直積から「1着・2着と重複する3着」を除いたものと完全一致するときだけ畳める
    expected = {(first, s, t) for s in seconds for t in thirds
                if t != first and t != s}
    if expected != got:
        return None
    return (f"{first}-{','.join(map(str, seconds))}"
            f"-{','.join(map(str, thirds))}")
