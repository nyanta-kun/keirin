"""看板レース専用の文面（`--marquee`）が実態と食い違わないことを検査する。

## 背景

2026-08-09 に「メイン会場の決勝・特選クラス（看板レース）には**必ず**推奨を出す」
方針を採った。既存ランクの文面はこれと両立しない:

    7A/9A: 「本命が割れ、相手次第で配当が伸びるレースだけを絞ってお届けして
             います。**毎日は出ません。**」

- 「毎日は出ません」 … 必ず出す方針と正面から矛盾する
- 「本命が割れ」     … 看板レースには断然人気のいる決勝も含まれる
                       （2026-08-09 和歌山12R は軸1の3着内率 95.6%）

7B で旧文面が現行条件と正反対だった事故（2026-08-06 是正）と同じ型なので、
**文面側で機械的に禁じる**。

## 守る不変条件

1. 看板用テンプレートに「絞って出している」「拮抗している」と読める断定が無い
2. 看板用テンプレートのプレースホルダが `_apply_template` で全て置換される
   （未定義の `{...}` は例外にならず**そのまま商品に出る**ため、テストでしか気づけない）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.netkeirin_submit_wt import (  # noqa: E402
    _MARQUEE_COMMENT_TEMPLATE,
    _MARQUEE_TITLE_TEMPLATE,
    _apply_template,
)

# 必ず出す方針・混在する拮抗度と食い違う言い回し。
# 「絞って出す」「毎日は出ない」「本命が割れている」を意味する語を禁じる。
FORBIDDEN = (
    "毎日は出ません",
    "毎日は出さ",
    "本命が割れ",
    "だけを絞って",
    "拮抗",
    "混戦",
)


@pytest.mark.parametrize("template", [_MARQUEE_TITLE_TEMPLATE, _MARQUEE_COMMENT_TEMPLATE])
def test_marquee_template_has_no_contradicting_claim(template: str) -> None:
    """必ず出す方針・断然人気の決勝と食い違う断定を含まない。"""
    hits = [w for w in FORBIDDEN if w in template]
    assert not hits, (
        f"看板レース用の文面に実態と食い違う表現が含まれています: {hits}\n"
        "看板レースは『必ず出す』うえ拮抗度も混ざるため、選別・拮抗の断定はできません。"
    )


def test_marquee_comment_has_no_self_promotion_block() -> None:
    """看板レース用の文面も【予想者より】を含まない（2026-08-09 ユーザー指示）。

    ランク別テンプレート（`update_netkeirin_templates.COMMENT_TEMPLATES`）とは
    **別に定義されている**ので、片方だけ消すと看板レースにだけ宣伝が残る。
    """
    for banned in ("【予想者より】", "ウマい！", "お気に入り登録", "的中実績",
                   "【この買い目について】", "{stake_note}", "【このレースについて】"):
        assert banned not in _MARQUEE_COMMENT_TEMPLATE, (
            f"看板レース用の文面に削除済みの記載（{banned}）が復活している"
        )


@pytest.mark.parametrize("template", [_MARQUEE_TITLE_TEMPLATE, _MARQUEE_COMMENT_TEMPLATE])
def test_marquee_template_placeholders_are_all_substituted(template: str) -> None:
    """`_apply_template` が看板用テンプレートの `{...}` を全て置換できる。

    `_apply_template` は未定義のプレースホルダを**素通しする**設計（ユーザーが
    設定画面で書いた任意の `{...}` で落とさないため）。したがって綴り間違いは
    例外にならず、`{race_typ}` のような文字列がそのまま商品に載る。
    """
    out = _apply_template(
        template,
        venue_name="和歌山",
        race_no=12,
        rank_key="9A",
        target_date="2026-08-09",
        axis1=3,
        axis2=5,
        shape="ライン決着に妙味",
        shape_note="軸2車が同じライン。",
        stake_note="金額は想定オッズに応じて配分しています。",
        race_type="決勝",
    )
    left = re.findall(r"\{[^}\n]{1,20}\}", out)
    assert not left, f"置換されなかったプレースホルダが残っています: {left}"


def test_marquee_title_is_the_fixed_phrase() -> None:
    """タイトルは固定文字列「本日の二軸」（2026-08-09 ユーザー指示）。

    ⚠️ 当初は「{race_type}の二軸｜{shape}」で種別とレース形を出していたが、
       看板レースは商品名を揃える方針に変更した。**変数を戻さないこと。**
       どのレースでも同じ文字列になるので、`_apply_template` を通しても不変。
    """
    assert _MARQUEE_TITLE_TEMPLATE == "本日の二軸"
    title = _apply_template(
        _MARQUEE_TITLE_TEMPLATE,
        venue_name="佐世保", race_no=12, rank_key="7A", target_date="2026-08-09",
        axis1=5, axis2=1, shape="別線に妙味", shape_note="", stake_note="",
        race_type="ガールズ決勝",
    )
    assert title == "本日の二軸", title
