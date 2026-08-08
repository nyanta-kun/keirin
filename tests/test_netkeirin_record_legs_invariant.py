"""入稿の記録経路が全ランクで成立することを構造的に検査する。

## 守る不変条件

`_process_rank` は入稿成功後に記録用の買い目を組む:

    record_legs = legs if (is_multi or is_formation or tilt_source) else _legs_for_record(...)

`_legs_for_record` は「軸＋相手を均等割り」する前提なので `_stake_per_line(cfg, ...)` を
呼ぶ。ここは `cfg["stake_budget"]` か `cfg["stake_per_line"]` のどちらかが要る。

したがって **どのランクも「左辺で拾われる」か「単価を計算できる」かのどちらか**でなければ
ならない。片方も満たさないランクを足すと、

  submit_pick_multi → **入稿は成功** → ここで KeyError → `_record_submission` に届かない

となり、**netkeirin には出ているのに DB に記録が無い**行が生まれ、さらに例外が
`_process_rank` を抜けて **その波の後続ランクが丸ごと入稿されない**。

実際 9H1 追加（2026-08-08）で `is_formation` が guard から漏れ、2026-08-09 朝の
morning 波は 7H1 の1件を出した直後に落ちて 7SS/7S/7A/7C/7B が1件も入稿されなかった。

⚠️ **個別ランクを名指しで検査しない。** ランクは頻繁に増えるので、
`RANK_CONFIGS` を走査して**全件**に対して不変条件を課す。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.netkeirin_submit_wt import (  # noqa: E402
    RANK_CONFIGS,
    _stake_per_line,
)


def _has_stake_source(cfg: dict) -> bool:
    """`_stake_per_line` が値を返せる設定か。"""
    return bool(cfg.get("stake_budget")) or ("stake_per_line" in cfg)


def _is_prebuilt_legs(cfg: dict) -> bool:
    """候補正規化側で legs を組み終えていて `_legs_for_record` を通らない経路か。"""
    return bool(cfg.get("multi_bet") or cfg.get("formation_bet") or cfg.get("tilt_stakes"))


@pytest.mark.parametrize("rank_key", sorted(RANK_CONFIGS))
def test_全ランクが記録経路のどちらかで成立する(rank_key: str):
    cfg = RANK_CONFIGS[rank_key]
    assert _is_prebuilt_legs(cfg) or _has_stake_source(cfg), (
        f"{rank_key}: multi_bet/formation_bet/tilt_stakes のいずれでもなく、"
        f"stake_budget も stake_per_line も無い。このままでは入稿成功後に "
        f"_legs_for_record → _stake_per_line で落ち、"
        f"netkeirin に出たまま DB へ記録されず後続ランクも止まる。"
    )


@pytest.mark.parametrize("rank_key", sorted(RANK_CONFIGS))
def test_stake_per_lineを通るランクは実際に単価を計算できる(rank_key: str):
    """`_legs_for_record` へ落ちるランクは、単価計算が例外なく通ること。"""
    cfg = RANK_CONFIGS[rank_key]
    if _is_prebuilt_legs(cfg):
        pytest.skip(f"{rank_key} は legs を組み終えた経路なので通らない")
    assert _stake_per_line(cfg, 5) > 0


def test_9h1は組み立て済み経路であること():
    """9H1(三連単フォーメーション)は formation_bet 側で拾われること。

    ここが False に戻ると 2026-08-09 の障害が再発する。
    """
    cfg = RANK_CONFIGS["9H1"]
    assert cfg.get("formation_bet") is True
    assert not _has_stake_source(cfg)      # 単価を持たない＝guard に頼っている
    assert _is_prebuilt_legs(cfg)


def test_guardにis_formationが含まれている():
    """`_process_rank` の guard 式に is_formation が残っていることを確認する。

    ⚠️ ソースを読む検査。実際に `_process_rank` を通す統合テストは
       NetkeirinClient の実通信を伴うため、ここでは guard 式の存在だけを固定する。
    """
    src = (Path(__file__).parent.parent / "scripts" / "netkeirin_submit_wt.py").read_text()
    assert "legs if (is_multi or is_formation or tilt_source) else _legs_for_record" in src, (
        "record_legs の guard から is_formation が消えている"
    )


def test_dry_run側のguardにもis_formationが含まれている():
    """preview の分岐にも is_formation が要る。

    本番経路(`record_legs`)だけ直すと **dry-run だけが落ちる**状態になり、
    「本番で何が出るか確かめる道具」が肝心のときに使えない
    （2026-08-09 に実際にこの状態になった）。
    """
    src = (Path(__file__).parent.parent / "scripts" / "netkeirin_submit_wt.py").read_text()
    assert "if is_multi or is_formation:" in src, (
        "dry-run の detail 分岐から is_formation が消えている"
    )
