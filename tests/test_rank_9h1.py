"""RANK_9H1（9車・高配当狙い）の不変条件テスト。

守りたいのは4つ:
  1. **9車ちょうどのレースだけを対象にする**（7H1 と母集団が排他であること）
  2. **1着はモデル3着内率5位の1車で固定**（人気薄を頭に置くのが本ランクの本体）
  3. **選別が絶対閾値で行われる**（日次の相対順位に戻すと件数が系統的に減る）
  4. **購入額が必ず 10,000円以下・100円単位**
"""
from __future__ import annotations

import pytest

from src import strategy_wt as sw
from src.preprocessing import upset_features as uf


def _p3(order: list[int]) -> dict[int, float]:
    """先頭ほど3着内率が高くなる dict を作る（順位だけが意味を持つ）。"""
    return {f: 90.0 - i * 5.0 for i, f in enumerate(order)}


def test_legs_shape_and_lead_is_fifth():
    order = [1, 2, 3, 4, 5, 6, 7, 8, 9]      # モデル3着内率の降順
    legs = sw.rank_9h1_build_legs(_p3(order))
    # 1着 = 5位(=車番5)固定 / 2着 = 5位を除く上位2車 / 3着 = 5位を除く上位4車
    assert len(legs) == 6
    assert all(x.startswith("5-") for x in legs), "1着が3着内率5位に固定されていない"
    seconds = {x.split("-")[1] for x in legs}
    thirds = {x.split("-")[2] for x in legs}
    assert seconds == {"1", "2"}
    assert thirds == {"1", "2", "3", "4"}
    assert len(set(legs)) == len(legs), "同じ目が重複している"
    for x in legs:
        a, b, c = x.split("-")
        assert len({a, b, c}) == 3, "同じ車が2箇所に入っている"


def test_legs_follow_probability_not_car_number():
    """並びは車番順ではなくモデル3着内率順で決まる。"""
    # 3着内率の降順が [3,1,4,9,2,...] なら 5位は車番2、2着列は車番3と1
    legs = sw.rank_9h1_build_legs(_p3([3, 1, 4, 9, 2, 5, 6, 7, 8]))
    assert all(x.startswith("2-") for x in legs)
    assert {x.split("-")[1] for x in legs} == {"3", "1"}
    assert {x.split("-")[2] for x in legs} == {"3", "1", "4", "9"}


def test_legs_empty_when_field_is_too_small():
    """5位が存在しない小頭数では買い目を組まない（黙って別の車を頭にしない）。"""
    assert sw.rank_9h1_build_legs(_p3([1, 2, 3, 4])) == []


@pytest.mark.parametrize("n_legs", [1, 2, 3, 5, 6, 8, 12])
def test_stakes_never_exceed_budget_and_are_100yen_units(n_legs):
    unit, total = sw.rank_9h1_stakes(n_legs)
    assert unit % sw.STAKE_UNIT == 0
    assert unit >= sw.STAKE_UNIT
    assert total <= sw.RACE_BUDGET, "1レースの予算枠を超えている"
    assert total == unit * n_legs


def test_normal_shape_costs_9600():
    """通常形（6点）は 1,600円/点・計9,600円。"""
    assert sw.rank_9h1_stakes(6) == (1600, 9600)


def _cand(score, **kw):
    d = {"n_entries": 9, "upset_score": score, "legs": ["5-1-2"]}
    d.update(kw)
    return d


def test_daily_select_uses_absolute_threshold():
    """絶対閾値で切る。**日次の相対順位に戻してはいけない**（件数が系統的に減る）。"""
    below = sw.RANK_9H1_SCORE_MIN - 0.01
    above = sw.RANK_9H1_SCORE_MIN + 0.01
    assert sw.rank_9h1_daily_select([_cand(below) for _ in range(20)]) == []
    assert len(sw.rank_9h1_daily_select([_cand(above) for _ in range(20)])) == 20


def test_daily_select_gates():
    above = sw.RANK_9H1_SCORE_MIN + 0.05
    assert sw.rank_9h1_daily_select([_cand(above, n_entries=7)]) == [], \
        "9車限定が効いていない（7H1 と母集団が重なる）"
    assert sw.rank_9h1_daily_select([_cand(above, legs=[])]) == [], \
        "買い目未成立が通っている"
    assert sw.rank_9h1_daily_select([_cand(None)]) == [], "スコア欠損が通っている"


def test_daily_select_sorted_by_score_desc():
    got = sw.rank_9h1_daily_select(
        [_cand(sw.RANK_9H1_SCORE_MIN + d) for d in (0.01, 0.30, 0.10)])
    assert [c["upset_score"] for c in got] == sorted(
        (c["upset_score"] for c in got), reverse=True)


# ── 波乱スコアの特徴量 ────────────────────────────────────────────────


def _entry(frame, rp, line_group, line_size, style="逃", cls="S級", mark=None):
    return {"frame_no": frame, "race_point": rp, "line_group": line_group,
            "line_size": line_size, "style": style, "player_class": cls,
            "s_count": 1, "b_count": 1, "first_rate": 10.0, "third_rate": 30.0,
            "prediction_mark": mark}


@pytest.fixture
def board9():
    return [_entry(1, 110.0, "A", 3, mark=1), _entry(2, 105.0, "A", 3),
            _entry(3, 100.0, "A", 3), _entry(4, 108.0, "B", 2),
            _entry(5, 102.0, "B", 2), _entry(6, 99.0, "C", 2),
            _entry(7, 98.0, "C", 2), _entry(8, 97.0, None, 1),
            _entry(9, 96.0, None, 1)]


@pytest.fixture
def race9():
    return {"n_entries": 9, "grade": "F1", "race_type": "予選", "day_index": 1,
            "distance": 400, "start_at": "1754600000", "bank_length": 400,
            "is_indoor": 0}


def test_feature_row_has_exactly_the_declared_columns(board9, race9):
    row = uf.build_upset_row(board9, race9)
    assert row is not None
    assert set(row) == set(uf.UPSET_FEATURE_COLS), "宣言した列と実際の列が食い違う"
    assert len(uf.feature_vector(row)) == len(uf.UPSET_FEATURE_COLS)


def test_feature_row_is_none_when_scratched(board9, race9):
    """事前欠車で行数が車数に足りないレースは母集団外（None）。"""
    assert uf.build_upset_row(board9[:-1], race9) is None


def test_line_features_are_ratios_not_counts(board9, race9):
    """車数をまたいで学習するため、ライン系は**個数ではなく割合**であること。"""
    row = uf.build_upset_row(board9, race9)
    # ライン A(3車)・B(2車)・C(2車) と単騎2名で 5 グループ（単騎は1車のラインとして数える）
    assert row["line_ratio"] == pytest.approx(5 / 9)
    assert row["max_line_ratio"] == pytest.approx(3 / 9)
    assert row["solo_ratio"] == pytest.approx(2 / 9)
    assert 0.0 <= row["nige_ratio"] <= 1.0


def test_category_encoding_is_deterministic():
    """カテゴリ符号化に組み込みの `hash()` を使っていないこと。

    文字列ハッシュはプロセスごとにランダム化されるため、`hash()` だと
    学習時と推論時で別の値になり、同じ入力でも結果が変わる。
    """
    assert uf._enc("F1") == uf._enc("F1")
    assert uf._enc("F1") == 678        # crc32("F1") % 1000（値が変わったら実装変更）
    assert uf._enc(None) == -1


def test_rank_is_registered_in_the_single_source():
    """`CURRENT_PAPER_RANKS` に載っていること（4つの集計参照先はここから導出する）。"""
    spec = next((s for s in sw.CURRENT_PAPER_RANKS if s.rank == "RANK_9H1"), None)
    assert spec is not None, "CURRENT_PAPER_RANKS に RANK_9H1 が無い"
    assert spec.suffix == "#9H1" and spec.label == "9H1"
    assert spec.in_header_total is False, "穴推奨はヘッダー合計に混ぜない"


# ── netkeirin 入稿への変換 ────────────────────────────────────────────


def test_netkeirin_formation_conversion_roundtrips():
    """入稿へ渡すフォーメーションを展開し直すと、候補の買い目と完全一致すること。

    一致しないまま入稿すると**意図と違う買い目が有料商品として外部へ出る**。
    7H1 で同じ性質を守っているのと同型の回帰テスト。
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from scripts.netkeirin_submit_wt import (
        RANK_CONFIGS, _normalize_formation_candidate,
    )
    from src.netkeirin_client import BET_KIND_TRIFECTA_FORMATION, expand_bet

    order = [3, 8, 7, 9, 5, 1, 2, 4, 6]          # モデル3着内率の降順
    legs_raw = sw.rank_9h1_build_legs(_p3(order))
    cand = {"race_key": "20260808_55_08", "order": order, "legs": legs_raw,
            "n_entries": 9}
    legs, marks, axis1, axis2 = _normalize_formation_candidate(
        cand, RANK_CONFIGS["9H1"])

    assert len(legs) == 1, "9H1 は単一券種（三連単フォーメーション）"
    leg = legs[0]
    assert leg.bet_kind == BET_KIND_TRIFECTA_FORMATION
    assert leg.groups[0] == [order[4]], "1着列が3着内率5位の1車になっていない"
    expanded = expand_bet(BET_KIND_TRIFECTA_FORMATION, leg.groups)
    assert expanded == {tuple(int(x) for x in s.split("-")) for s in legs_raw}
    assert leg.stake_per_line * len(legs_raw) <= sw.RACE_BUDGET

    # 印は order（モデル3着内率の降順）に従う。車番順ではない
    assert axis1 == order[4] and marks[axis1] == "◎"
    assert axis2 == order[0] and marks[order[0]] == "○"
    assert marks[order[1]] == "▲"
    assert marks[order[2]] == "△" and marks[order[3]] == "△"


def test_netkeirin_9h1_config_shape():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.netkeirin_submit_wt import RANK_CONFIGS
    from src.netkeirin_client import ACT_TYPE_LONGSHOT

    cfg = RANK_CONFIGS["9H1"]
    assert cfg["formation_bet"] is True
    assert cfg["n_cars"] == 9
    assert cfg["file_key"] == "s9h1"
    assert "multi_bet" not in cfg, "9H1 は単一券種（7H1 の2券種経路と取り違えない）"
    assert "stake_per_line" not in cfg, "賭け金は点数から決めるので固定額は持たない"
    assert cfg["act_type"] == ACT_TYPE_LONGSHOT      # 勝負アイコンは「穴狙い」
