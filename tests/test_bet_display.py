"""買い目のフォーメーション表記への畳み込み（src/bet_display.py）。

⚠️ 畳めない構造で **None を返す**ことが本質。省略して誤った買い目を見せるより
   冗長（元の列挙）を選ぶ設計なので、その分岐を必ず固定しておく。
"""
from __future__ import annotations

from itertools import combinations

from src.bet_display import fold_trifecta_formation, fold_trio_box


class TestFoldTrioBox:
    def test_5車box_を畳む(self):
        legs = ["=".join(map(str, c)) for c in combinations([1, 3, 4, 5, 7], 3)]
        assert fold_trio_box(legs) == "1,3,4,5,7 BOX"

    def test_4車box_を畳む(self):
        legs = ["3=5=7", "2=3=5", "2=3=7", "2=5=7"]
        assert fold_trio_box(legs) == "2,3,5,7 BOX"

    def test_並び順に依存しない(self):
        legs = ["7=3=1", "1=3=5", "4=3=1", "3=5=7", "3=4=7",
                "3=4=5", "1=5=7", "1=4=7", "1=4=5", "4=5=7"]
        assert fold_trio_box(legs) == "1,3,4,5,7 BOX"

    def test_box_でなければ畳まない(self):
        # 1車欠けた 9点。BOX(10点)と誤認して余分な目を表示してはいけない
        legs = ["=".join(map(str, c)) for c in combinations([1, 3, 4, 5, 7], 3)][:9]
        assert fold_trio_box(legs) is None

    def test_重複目があれば畳まない(self):
        assert fold_trio_box(["1=2=3", "1=2=3"]) is None

    def test_空や不正入力は畳まない(self):
        assert fold_trio_box([]) is None
        assert fold_trio_box(["1=2"]) is None
        assert fold_trio_box(["1=2=x"]) is None


class TestFoldTrifectaFormation:
    def test_1着固定フォーメーションを畳む(self):
        legs = ["7-3-1", "7-3-5", "7-3-4", "7-3-6",
                "7-1-3", "7-1-5", "7-1-4", "7-1-6"]
        assert fold_trifecta_formation(legs) == "7-1,3-1,3,4,5,6"

    def test_3着に1着2着が現れない場合も畳む(self):
        legs = ["3-5-6", "3-5-4", "3-5-7", "3-5-2",
                "3-7-6", "3-7-4", "3-7-5", "3-7-2"]
        assert fold_trifecta_formation(legs) == "3-5,7-2,4,5,6,7"

    def test_1着が複数なら畳まない(self):
        assert fold_trifecta_formation(["7-3-1", "6-3-1"]) is None

    def test_直積に欠けがあれば畳まない(self):
        legs = ["7-3-1", "7-3-5", "7-3-4", "7-3-6",
                "7-1-3", "7-1-5", "7-1-4"]          # 7-1-6 が無い
        assert fold_trifecta_formation(legs) is None

    def test_重複目があれば畳まない(self):
        assert fold_trifecta_formation(["7-3-1", "7-3-1"]) is None

    def test_空や不正入力は畳まない(self):
        assert fold_trifecta_formation([]) is None
        assert fold_trifecta_formation(["7-3"]) is None
        assert fold_trifecta_formation(["7-3-x"]) is None


def test_本番の7H1買い目が畳めること():
    """2026-08-06 別府7R（実データ）。Web 側と同じ結果になること。"""
    trio = ["1=3=7", "1=3=5", "1=3=4", "3=5=7", "3=4=7",
            "3=4=5", "1=5=7", "1=4=7", "1=4=5", "4=5=7"]
    tf = ["7-3-1", "7-3-5", "7-3-4", "7-3-6",
          "7-1-3", "7-1-5", "7-1-4", "7-1-6"]
    assert fold_trio_box(trio) == "1,3,4,5,7 BOX"
    assert fold_trifecta_formation(tf) == "7-1,3-1,3,4,5,6"
