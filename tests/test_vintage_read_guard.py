"""過去日のスコアリングに本番モデルを使ったら落ちることを固定する（2026-08-08）。

背景（レビュー指摘 M-2）:
  書き込み側 `trainer.save_model` には vintage 名の検査があるのに、
  **読み込み側には対称な防御が無かった**。`backfill_*_rank_wt.py` /
  `build_7h1_candidates.py` の `--eval-model` 等は既定値が本番モデル名
  （`lgbm_wt_eval` 等）なので、過去日を渡しても**エラーにも警告にもならず
  静かに in-sample な数字が出る**。docstring に「vintage を明示すること」と
  書いてあるだけで、機械的には何も止めていなかった。

  本番モデルは全期間を週次（日曜23:30）で再学習しているため、過去へ当てると
  model-vintage look-ahead になる。keirin では「過去のプラス ROI が実は
  人工物だった」という事故を何度も踏んでいる領域なので、機械で止める。
"""
from __future__ import annotations

from datetime import date

import pytest

from src.wt_vintage_config import (
    ALLOW_PRODUCTION_ENV, assert_vintage_for_past, is_vintage_model,
)

TODAY = date(2026, 8, 8)


def test_vintage_name_detection() -> None:
    assert is_vintage_model("lgbm_wt_eval_m2404")
    assert is_vintage_model("lgbm_wt_favbust_m2512")
    assert not is_vintage_model("lgbm_wt_eval")
    assert not is_vintage_model("lgbm_wt_bad")
    # 桁数が違うものを vintage と誤認しない
    assert not is_vintage_model("lgbm_wt_eval_m24")


def test_past_date_with_production_model_raises() -> None:
    with pytest.raises(ValueError, match="本番モデル"):
        assert_vintage_for_past("2026-07-31", {"eval": "lgbm_wt_eval"}, today=TODAY)


def test_past_date_with_vintage_model_passes() -> None:
    assert_vintage_for_past(
        "2026-07-31",
        {"eval": "lgbm_wt_eval_m2607", "win": "lgbm_wt_win_m2607",
         "bad": "lgbm_wt_bad_m2607", "favbust": "lgbm_wt_favbust_m2607"},
        today=TODAY)


def test_one_production_model_among_vintages_is_caught() -> None:
    """1つでも本番モデルが混じっていたら落ちること（混在が一番危ない）。"""
    with pytest.raises(ValueError, match="favbust=lgbm_wt_favbust"):
        assert_vintage_for_past(
            "2026-07-31",
            {"eval": "lgbm_wt_eval_m2607", "favbust": "lgbm_wt_favbust"},
            today=TODAY)


def test_today_with_production_model_passes() -> None:
    """🔴 ライブ予想を壊さないこと。

    `daily_picks_wt.sh` / `evening_picks_wt.sh` は毎日
    `build_7h1_candidates.py --date $TODAY` を本番モデルの既定値で呼ぶ。
    ここで落とすと**朝の予想生成が丸ごと止まる**。当日レースを前日までで
    学習したモデルで評価するのは honest なので、素通しが正しい。
    """
    assert_vintage_for_past("2026-08-08", {"eval": "lgbm_wt_eval"}, today=TODAY)


def test_future_date_with_production_model_passes() -> None:
    assert_vintage_for_past("2026-08-09", {"eval": "lgbm_wt_eval"}, today=TODAY)


def test_none_model_is_ignored() -> None:
    """使わないモデル（None）は検査対象外。"""
    assert_vintage_for_past("2026-07-31", {"bad": None, "eval": "lgbm_wt_eval_m2607"},
                            today=TODAY)


def test_env_escape_hatch_warns_but_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    """調査目的の逃げ道はあるが、黙って通さず必ず警告を出すこと。"""
    monkeypatch.setenv(ALLOW_PRODUCTION_ENV, "1")
    with pytest.warns(RuntimeWarning, match="in-sample"):
        assert_vintage_for_past("2026-07-31", {"eval": "lgbm_wt_eval"}, today=TODAY)


@pytest.mark.parametrize("script", [
    "backfill_7ss_rank_wt", "backfill_7s_rank_wt", "backfill_7a_rank_wt",
    "backfill_7b_rank_wt", "backfill_7c_rank_wt", "backfill_9s_rank_wt",
    "backfill_9a_rank_wt", "backfill_7h1_rank_wt", "backfill_s1w_rank_wt",
    "build_7h1_candidates", "gen_7b_candidates_only",
])
def test_cli_scripts_call_the_guard(script: str) -> None:
    """既定値が本番モデル名の CLI は全てガードを通すこと。

    1本ずつ手で確認する運用だと、次に足すスクリプトでまた漏れる
    （このリポジトリで RANK_ORDER・reconcile 登録・CURRENT_PAPER_RANKS と
    同型の漏れが繰り返し起きている）。
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "scripts"
           / f"{script}.py").read_text(encoding="utf-8")
    assert "assert_vintage_for_past(" in src, (
        f"{script}.py がガードを呼んでいない。--*-model の既定値が本番モデル名なので"
        " 過去日を渡すと無言で in-sample になる")
