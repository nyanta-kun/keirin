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


# ── 発走前のライブ判定（盤面・欠車の扱い）───────────────────────────────


def _tf_lookup(cars: list[int]) -> dict:
    """指定車で作れる全順列のオッズ辞書（盤面のダミー）。"""
    from itertools import permutations

    from scripts.notify_prerace_wt import _parse_combo_key
    return {_parse_combo_key("-".join(map(str, p)), True): 100.0
            for p in permutations(cars, 3)}


@pytest.fixture
def cand_9h1():
    order = [3, 8, 7, 9, 5, 1, 2, 4, 6]          # モデル3着内率の降順
    return {"race_key": "20260808_55_08", "venue_name": "和歌山", "race_no": 8,
            "n_entries": 9, "upset_score": 0.3780, "order": order,
            "lead": order[4], "lead_name": "X",
            "legs": sw.rank_9h1_build_legs(_p3(order))}


def test_judge_buys_when_board_is_complete(cand_9h1):
    from scripts.notify_prerace_wt import judge_rank_9h1
    decision, detail = judge_rank_9h1(cand_9h1, _tf_lookup(list(range(1, 10))))
    assert decision == "buy"
    assert detail["dropped"] == 0
    assert detail["stake"] == 1600 and detail["bet_amount"] == 9600
    assert detail["bet_amount"] <= sw.RACE_BUDGET


def test_judge_skips_when_first_place_car_is_scratched(cand_9h1):
    """1着固定車が盤面に無い＝レース無効。残りで組み直したりしない。"""
    from scripts.notify_prerace_wt import judge_rank_9h1
    others = [c for c in range(1, 10) if c != cand_9h1["lead"]]
    decision, detail = judge_rank_9h1(cand_9h1, _tf_lookup(others))
    assert decision == "skip"
    assert "1着固定" in (detail["skip_reason"] or "")


def test_judge_drops_scratched_partners_and_restakes(cand_9h1):
    """相手が欠けた目だけ落とし、残った点数で賭け金を張り直す。"""
    from scripts.notify_prerace_wt import judge_rank_9h1
    board = [c for c in range(1, 10) if c != 6]      # 6番が欠車（買い目には未使用）
    decision, detail = judge_rank_9h1(cand_9h1, _tf_lookup(board))
    assert decision == "buy" and detail["dropped"] == 0

    board2 = [c for c in range(1, 10) if c != 9]     # 9番が欠車（3着列に居る）
    decision2, detail2 = judge_rank_9h1(cand_9h1, _tf_lookup(board2))
    assert decision2 == "buy"
    assert detail2["dropped"] > 0
    assert detail2["bet_amount"] <= sw.RACE_BUDGET
    assert detail2["stake"] * len(detail2["legs"]) == detail2["bet_amount"]


def test_judge_returns_unknown_without_board(cand_9h1):
    """盤面が取れていないときは skip ではなく『不明』（次回再試行）。"""
    from scripts.notify_prerace_wt import judge_rank_9h1
    assert judge_rank_9h1(cand_9h1, {})[0] == "不明"


def test_pred_combo_format_matches_between_write_and_judge(cand_9h1):
    """朝の候補書き込みと発走前判定で pred_combo の形式が一致すること。

    食い違うと**採点と Web 表示が黙って壊れる**（7H1 で同じ性質を守っている）。
    """
    import inspect

    from scripts import notify_prerace_wt as npw
    from scripts import write_candidates_wt as wcw
    judge_src = inspect.getsource(npw._insert_rank_9h1_pick)
    write_src = inspect.getsource(wcw._write_paper_candidates)
    assert '"三単:" + ",".join(detail["legs"])' in judge_src
    assert '"三単:" + ",".join(legs)' in write_src


def test_formation_path_submits_with_longshot_flag():
    """9H1 は `submit_pick_multi` へ **act_type=穴狙い** で送られること。

    ⚠️ `formation_bet` を分岐条件に入れ忘れると `submit_pick`（軸+相手）へ落ち、
       `cfg["bet_kind"]` を持たない 9H1 は KeyError で入稿できない。
       分岐と勝負アイコンの両方をここで固定する。
    """
    import inspect
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts import netkeirin_submit_wt as ns
    from src.netkeirin_client import ACT_TYPE_LONGSHOT

    src = inspect.getsource(ns)
    # 2026-08-09: フラグ列挙から **legs の有無** による分岐へ変更した。
    # formation 経路は `_normalize_formation_candidate` が legs を組むので
    # 引き続き submit_pick_multi 側へ入る。列挙方式は経路が増えるたびに
    # 追記が要り、9H1 追加時に実際に漏れたのでここへは戻さないこと。
    assert "if legs:" in src, \
        "legs を組んだ経路が submit_pick_multi へ回っていない"
    # act_type は cfg 優先で解決される（9H1 は ACT_TYPE_LONGSHOT を持つ）
    assert ns.RANK_CONFIGS["9H1"]["act_type"] == ACT_TYPE_LONGSHOT
    assert ns.RANK_CONFIGS["7H1"]["act_type"] == ACT_TYPE_LONGSHOT
    assert 'act_type=cfg.get(' in src


def test_daily_pipeline_generates_9h1_candidates():
    """朝・夕の両バッチが 9H1 候補を作ること。

    どちらかが漏れると「候補JSONが無い日は静かに0件」になり、
    ログにも異常が出ないまま推奨が止まる（7SS の入稿漏れと同型の fail-closed）。
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for name in ("daily_picks_wt.sh", "evening_picks_wt.sh"):
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "build_9h1_candidates.py" in text, f"{name} が 9H1 候補を作っていない"
    ev = (root / "scripts" / "evening_picks_wt.sh").read_text(encoding="utf-8")
    assert "_night_s9h1_candidates.json" in ev, "夕方分の出力先が _night になっていない"


def test_build_9h1_candidates_refuses_production_models_for_past():
    """過去日を本番モデルでスコアしようとしたら落ちること。

    既定値が本番モデル名なので、`--screen-model` の指定を忘れると**無言で**
    in-sample な数字が出る。7H1 側と同じく機械的に止める。
    """
    import inspect
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts import build_9h1_candidates as b9
    from src.wt_vintage_config import assert_vintage_for_past

    src = inspect.getsource(b9.main)
    assert "assert_vintage_for_past(" in src, "vintage ガードを呼んでいない"
    # 波乱スコアのモデルも検査対象に入っていること（3着内率だけでは片手落ち）
    assert '"screen": args.screen_model' in src

    from datetime import date
    with pytest.raises(ValueError, match="本番モデル"):
        assert_vintage_for_past(
            "2026-07-31",
            {"eval": "lgbm_wt_eval", "screen": "lgbm_upset_screen"},
            today=date(2026, 8, 8))
    # vintage を渡せば通る
    assert_vintage_for_past(
        "2026-07-31",
        {"eval": "lgbm_wt_eval_m2607", "screen": "lgbm_upset_screen_m2607"},
        today=date(2026, 8, 8))
