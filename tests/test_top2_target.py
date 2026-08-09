"""連帯（2着以内）ターゲット `top2_flag` の定義を固定する（2026-08-09 新設）。

## なぜテストを最初から置くのか

`bad6_flag` は**本番モデル `lgbm_wt_bad` が使っている定義を作るコードが
リポジトリに存在せず**（全git履歴を検索して0件）、2026-08-05 に実測で
同定する羽目になった。モデルだけが正解を知っていて、コードもテストも無い状態は
再現不能で危険。`top2_flag` は同じ轍を踏まないよう、定義とテストを同時に置く。

## 守る不変条件

1. `top2_flag` = 着順が 1 か 2。**DNF/失格（finish_order=0）は含めない**
   （`win_flag` と同じ扱い。`top3_flag` も 1〜3 に限っている）
2. 包含関係 `win ⊆ top2 ⊆ top3` が全行で成り立つ
   — 崩れると `P(2着)=top2−win` / `P(3着)=top3−top2` が負になり、
   着順分解（この列を足した目的そのもの）が破綻する
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_wt import TOP2_TARGET_COL_WT  # noqa: E402


def _flags(finish_order: list) -> pd.DataFrame:
    """`build_features_wt` と**同一の式**でフラグを組む。

    ⚠️ 本体は生データの全列を要求するため単体では呼べない。式を写している以上、
       本体を変えたらここも変える必要がある（下の test_matches_source が検出する）。
    """
    df = pd.DataFrame({"finish_order": finish_order})
    df["top3_flag"] = (df["finish_order"].notna()
                       & (df["finish_order"] >= 1)
                       & (df["finish_order"] <= 3)).astype(int)
    df["top2_flag"] = (df["finish_order"].notna()
                       & (df["finish_order"] >= 1)
                       & (df["finish_order"] <= 2)).astype(int)
    df["win_flag"] = (df["finish_order"].notna()
                      & (df["finish_order"] == 1)).astype(int)
    return df


def test_target_column_name() -> None:
    assert TOP2_TARGET_COL_WT == "top2_flag"


def test_top2_is_first_or_second_only() -> None:
    d = _flags([1, 2, 3, 4, 9, 0, None])
    assert list(d["top2_flag"]) == [1, 1, 0, 0, 0, 0, 0]


def test_dnf_is_not_top2() -> None:
    """DNF/失格（finish_order=0）と欠場（NaN）は2着以内に含めない。"""
    d = _flags([0, None])
    assert d["top2_flag"].sum() == 0


def test_nesting_win_subset_top2_subset_top3() -> None:
    """🔴 win ⊆ top2 ⊆ top3。崩れると着順分解が負になる。"""
    d = _flags([0, 1, 2, 3, 4, 5, 6, 7, 9, None])
    assert ((d["win_flag"] <= d["top2_flag"]).all())
    assert ((d["top2_flag"] <= d["top3_flag"]).all())


def test_positional_decomposition_is_non_negative() -> None:
    """P(2着)=top2−win, P(3着)=top3−top2 が非負になる（この列を足した目的）。"""
    d = _flags([0, 1, 2, 3, 4, None])
    assert ((d["top2_flag"] - d["win_flag"]) >= 0).all()
    assert ((d["top3_flag"] - d["top2_flag"]) >= 0).all()


def test_matches_source() -> None:
    """本体（feature_wt.py）の式と、このテストの写しが一致していること。"""
    src = (Path(__file__).parent.parent / "src" / "preprocessing"
           / "feature_wt.py").read_text()
    assert 'df["top2_flag"] = (df["finish_order"].notna()' in src
    assert '& (df["finish_order"] <= 2)).astype(int)' in src


def test_cli_exposes_top2_target() -> None:
    """学習CLIから `--target top2` が使えること。"""
    src = (Path(__file__).parent.parent / "src" / "cli" / "main.py").read_text()
    assert '"top3", "win", "bad", "top2"' in src
    assert '"top2": TOP2_TARGET_COL_WT' in src
    # 配信用 lgbm_wt を汚染しない安全弁に top2 が入っていること
    assert 'target_kind in ("win", "bad", "top2") and promote' in src
