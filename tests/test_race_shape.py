"""レース構造ラベル（netkeirin タイトル後半）の検査。

守りたいのは3点:
  1. ラベルの判定順序（solo > duo > mixed > line/split/clash）が崩れないこと
  2. 「定義上あり得ない組み合わせ」が**無言で**フォールバックしないこと
     （7SS × split/clash・7A × solo）
  3. ランク一覧が二重管理にならないこと（TITLE_TEMPLATES ↔ SHAPE_TITLES）
"""
from __future__ import annotations

import pytest

from src.race_shape import (
    RANK_ALIASES,
    SHAPE_CLASH,
    SHAPE_DUO,
    SHAPE_LINE,
    SHAPE_MIXED,
    SHAPE_NOTES,
    SHAPE_SOLO,
    SHAPE_SPLIT,
    SHAPE_TITLES,
    STAKE_NOTE_EQUAL,
    classify_shape,
    shape_note_text,
    shape_title_text,
    stake_note_text,
)


def _entries(win, top3, *, lines=None, styles=None):
    """車番1..n の出走表を組む。win/top3 はパーセント（正規化前の生確率）。"""
    n = len(win)
    lines = lines if lines is not None else list(range(1, n + 1))   # 全員単騎
    styles = styles if styles is not None else ["追"] * n
    return [
        {"frame_no": i + 1, "pred_win_pct": win[i], "pred_top3_pct": top3[i],
         "line_group": lines[i], "style": styles[i]}
        for i in range(n)
    ]


# 1着率が1車だけ突出。3着内率も同じ形にしておく。
SOLO_WIN = [55, 12, 8, 8, 7, 6, 4]
FLAT_TOP3 = [45, 43, 42, 41, 40, 39, 38]
# 3着内率の2位と3位が離れる（1着率は横並び）。
FLAT_WIN = [20, 18, 16, 15, 14, 9, 8]
DUO_TOP3 = [80, 75, 30, 28, 26, 24, 22]
# どちらも横並び＝混戦。
MIXED_WIN = [17, 16, 15, 14, 14, 13, 11]
MIXED_TOP3 = [46, 45, 44, 43, 42, 41, 39]
# 中位（solo でも duo でも mixed でもない）＝ライン系ラベルへ落ちる帯。
PLAIN_WIN = [30, 15, 14, 13, 12, 9, 7]
PLAIN_TOP3 = [60, 50, 42, 40, 38, 36, 34]


def test_solo_wins_over_other_labels():
    e = _entries(SOLO_WIN, FLAT_TOP3)
    assert classify_shape("7S", e, 1, 2) == SHAPE_SOLO


def test_duo_when_top3_gap_is_wide():
    e = _entries(FLAT_WIN, DUO_TOP3)
    assert classify_shape("7S", e, 1, 2) == SHAPE_DUO


def test_mixed_when_both_gaps_are_narrow():
    e = _entries(MIXED_WIN, MIXED_TOP3)
    assert classify_shape("7S", e, 1, 2) == SHAPE_MIXED


def test_plain_falls_back_to_line_when_axes_share_a_line():
    e = _entries(PLAIN_WIN, PLAIN_TOP3, lines=[1, 1, 2, 2, 2, 3, 3])
    assert classify_shape("7S", e, 1, 2) == SHAPE_LINE


def test_plain_falls_back_to_split_when_axes_are_on_different_lines():
    e = _entries(PLAIN_WIN, PLAIN_TOP3, lines=[1, 2, 2, 2, 3, 3, 1])
    assert classify_shape("7S", e, 1, 2) == SHAPE_SPLIT


def test_plain_becomes_clash_when_three_or_more_front_runners():
    e = _entries(PLAIN_WIN, PLAIN_TOP3, lines=[1, 2, 2, 2, 3, 3, 1],
                 styles=["逃", "逃", "逃", "追", "追", "追", "追"])
    assert classify_shape("7S", e, 1, 2) == SHAPE_CLASH


def test_solo_lines_are_never_treated_as_the_same_line():
    """line_group が NULL 同士（単騎×単騎）を同ラインにしてはいけない。"""
    e = _entries(PLAIN_WIN, PLAIN_TOP3, lines=[None] * 7)
    assert classify_shape("7S", e, 1, 2) == SHAPE_SPLIT


@pytest.mark.parametrize("rank", ["7A", "9A"])
def test_rank_7a_never_reports_solo(rank):
    """7A は q20 ゲートで「本命が割れた」レースだけを通す＝1車抜けと両立しない。"""
    e = _entries(SOLO_WIN, FLAT_TOP3, lines=[1, 1, 2, 2, 2, 3, 3])
    assert classify_shape(rank, e, 1, 2) == SHAPE_LINE


def test_missing_indices_still_classify_by_line():
    """指数未算出でもライン系ラベルは決まる（確率を使わないため）。"""
    e = [{"frame_no": i + 1, "pred_win_pct": None, "pred_top3_pct": None,
          "line_group": 1 if i < 2 else 2, "style": "追"} for i in range(7)]
    assert classify_shape("7C", e, 1, 2) == SHAPE_LINE


def test_no_entries_returns_none():
    assert classify_shape("7C", [], 1, 2) is None


@pytest.mark.parametrize("nine,seven", sorted(RANK_ALIASES.items()))
def test_nine_car_ranks_share_the_seven_car_wording(nine, seven):
    """購入者に車立ての区別は出さない＝9系と7系は同一文言。"""
    for shape in SHAPE_TITLES[seven]:
        assert shape_title_text(nine, shape) == shape_title_text(seven, shape)


def test_shape_title_text_returns_no_warning_for_defined_pairs():
    for rank, table in SHAPE_TITLES.items():
        for shape in table:
            text, warning = shape_title_text(rank, shape)
            assert text, f"{rank}/{shape} のテキストが空"
            assert warning is None, f"{rank}/{shape} で警告が出た: {warning}"


@pytest.mark.parametrize("shape", [SHAPE_SPLIT, SHAPE_CLASH])
def test_rank_7ss_warns_loudly_on_impossible_shapes(shape):
    """7SS は軸2車が同一ラインであることが選出条件。別ライン系は壊れた合図。"""
    text, warning = shape_title_text("7SS", shape)
    assert text, "フォールバックのテキストは必要（タイトル空は禁止）"
    assert warning is not None, "無言でフォールバックしてはいけない"


def test_fallback_when_shape_is_unknown():
    for rank in SHAPE_TITLES:
        text, warning = shape_title_text(rank, None)
        assert text
        assert warning is None


EXPECTED_COVERAGE = {
    "7S": {SHAPE_SOLO, SHAPE_DUO, SHAPE_LINE, SHAPE_SPLIT, SHAPE_CLASH, SHAPE_MIXED},
    "7A": {SHAPE_DUO, SHAPE_LINE, SHAPE_SPLIT, SHAPE_CLASH, SHAPE_MIXED},
    "7B": {SHAPE_SOLO, SHAPE_DUO, SHAPE_LINE, SHAPE_SPLIT, SHAPE_CLASH, SHAPE_MIXED},
    "7C": {SHAPE_SOLO, SHAPE_DUO, SHAPE_LINE, SHAPE_SPLIT, SHAPE_CLASH, SHAPE_MIXED},
    "7SS": {SHAPE_SOLO, SHAPE_DUO, SHAPE_LINE, SHAPE_MIXED},
    "7H1": {SHAPE_SOLO, SHAPE_DUO, SHAPE_LINE, SHAPE_SPLIT, SHAPE_CLASH, SHAPE_MIXED},
}


@pytest.mark.parametrize("table", [SHAPE_TITLES, SHAPE_NOTES])
def test_every_rank_defines_the_common_shapes(table):
    """7SS(split/clash) と 7A(solo) 以外は6ラベルすべてを持つこと。

    タイトルと本文で**同じ網羅**であること。片方だけラベルを足すと、タイトルは
    構造を語っているのに本文は既定文、という食い違いが静かに生まれる。
    """
    assert {k: set(v) for k, v in table.items()} == EXPECTED_COVERAGE


def test_shape_notes_never_leak_the_bet():
    """本文冒頭はプレビュー表示されうる。車番・点数を書かない（仕様書 §4-3）。"""
    for rank, table in SHAPE_NOTES.items():
        for shape, text in table.items():
            assert "{axis" not in text, f"{rank}/{shape} に軸の差し込みがある"
            for banned in ("番・", "点買", "点流し"):
                assert banned not in text, f"{rank}/{shape} に買い目の露出: {banned}"


def test_shape_note_text_mirrors_title_behaviour():
    for rank, table in SHAPE_NOTES.items():
        for shape in table:
            text, warning = shape_note_text(rank, shape)
            assert text and warning is None
    # 定義上あり得ない組み合わせはタイトルと同じく警告を返す
    _text, warning = shape_note_text("7SS", SHAPE_SPLIT)
    assert warning is not None


@pytest.mark.parametrize("nine,seven", sorted(RANK_ALIASES.items()))
def test_nine_car_ranks_share_the_seven_car_note(nine, seven):
    for shape in SHAPE_NOTES[seven]:
        assert shape_note_text(nine, shape) == shape_note_text(seven, shape)


def test_stake_note_is_honest_about_equal_allocation():
    """均等になったレースで「オッズに応じて配分」と書かないこと（仕様書 §4-6）。"""
    for rank in ("7S", "7A", "7B", "7C", "7SS", "7H1", "9H1"):
        assert stake_note_text(rank, tilted=False) == STAKE_NOTE_EQUAL
        assert "均等ではなく" in stake_note_text(rank, tilted=True)


def test_gami_claim_is_limited_to_high_pay_ranks():
    """ガミ抑制を売り文句にできるのは 7H1/9H1 のみ（仕様書 §1・§4-6）。"""
    for rank in ("7H1", "9H1"):
        assert "ガミ" in stake_note_text(rank, tilted=True)
    for rank in ("7S", "7A", "7B", "7C", "7SS", "9S", "9A"):
        assert "ガミ" not in stake_note_text(rank, tilted=True)
        assert "ガミ" not in stake_note_text(rank, tilted=False)


def test_title_templates_and_shape_titles_stay_in_sync():
    """ランク一覧の二重管理を検出する（本リポジトリで繰り返し事故になった型）。"""
    from scripts.update_netkeirin_templates import (
        TITLE_TEMPLATES, _check_consistency,
    )
    assert _check_consistency() == []
    for rank, tpl in TITLE_TEMPLATES.items():
        assert "{shape}" in tpl
        # 会場・R番号・日付は netkeirin 側で別欄に出るのでタイトルには入れない
        for banned in ("{venue}", "{race_no}", "{date}", "{axis1}", "{axis2}"):
            assert banned not in tpl, f"{rank}: {banned} はタイトルに入れない"


def test_comment_body_starts_with_the_race_note_not_the_axes():
    """本文の冒頭（＝プレビュー想定）に軸2車を出さない（仕様書 §4-3）。"""
    from scripts.update_netkeirin_templates import COMMENT_TEMPLATES
    for rank, tpl in COMMENT_TEMPLATES.items():
        assert tpl.startswith("{shape_note}"), f"{rank}: 冒頭が見解になっていない"
        head = tpl.split("【二軸】")[0]
        for banned in ("{axis1}", "{axis2}"):
            assert banned not in head, f"{rank}: プレビュー部分に {banned} がある"


def test_comment_body_has_no_self_promotion_block():
    """【予想者より】（実績の宣伝・お気に入り登録の依頼）を含まない。

    2026-08-09 にユーザー指示で全ランクから削除した。**復活させないこと。**
    以前は「集客導線は【参考データ】より前に置く」として本文へ挟んでいたので、
    テンプレートを触るときに戻してしまいやすい。
    """
    from scripts.update_netkeirin_templates import COMMENT_TEMPLATES
    for rank, tpl in COMMENT_TEMPLATES.items():
        for banned in ("【予想者より】", "ウマい！", "お気に入り登録", "的中実績"):
            assert banned not in tpl, f"{rank}: 削除済みの宣伝文（{banned}）が復活している"
        assert "オッズ" in tpl, f"{rank}: 最終オッズ確認の一文が無い"
        assert tpl.rstrip().endswith("参考にご活用ください。"), (
            f"{rank}: 出走表は本文末尾へ自動追記されるので【参考データ】が最後であること"
        )


def test_comment_body_does_not_mention_semifinals_for_7b():
    """7B の「準決勝」は購入者に伝わりにくいので書かない（2026-08-09）。"""
    from scripts.update_netkeirin_templates import COMMENT_TEMPLATES
    assert "準決勝" not in COMMENT_TEMPLATES["7B"]


def test_comment_body_has_no_bet_explanation_block():
    """【この買い目について】（狙いの説明＋{stake_note}）を含まない。

    2026-08-09 にユーザー指示で全ランクから削除した。**復活させないこと。**

    ⚠️ このテストは以前 `test_comment_body_differentiates_ranks` として
       「全ランクが同一文だった状態へ戻らないこと」を守っていた（7C/7SS の
       実態不一致の再発防止）。ブロックごと消した結果**全ランクの本文は
       意図的に同一**になったので、ガードの向きを反転させてある。
       狙いの差はタイトルの `{shape}` と、商品に表示される買い目そのもので伝える。
    """
    from scripts.update_netkeirin_templates import COMMENT_TEMPLATES
    for rank, tpl in COMMENT_TEMPLATES.items():
        for banned in ("【この買い目について】", "{stake_note}"):
            assert banned not in tpl, f"{rank}: 削除済みの {banned} が復活している"


def test_submitted_titles_stay_within_display_width():
    """一覧で切れないよう、組み上がったタイトルは20字以内に収める。"""
    from scripts.update_netkeirin_templates import TITLE_TEMPLATES
    for rank, tpl in TITLE_TEMPLATES.items():
        base = RANK_ALIASES.get(rank, rank)
        for shape, text in SHAPE_TITLES[base].items():
            title = tpl.replace("{shape}", text)
            assert len(title) <= 20, f"{rank}/{shape} が長すぎる ({len(title)}字): {title}"
