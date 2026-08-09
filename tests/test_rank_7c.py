"""RANK_7C（ベースモデル・終日の二軸）の不変条件テスト。

守りたいのは5つ:
  1. **軸は pred_top3 上位2車**（3ヘッド軸ではない）。取り違えると別の買い目になる
  2. **相手は3着内率15%以上だけ**。足りないときに最低1車を補ってはいけない
     （「相手が絞れる＝配当が付かない」という指標そのものを消してしまう）
  3. **相手が4点未満のレースは買わない**（低配当回避の本体）
  4. **賭け金は予算枠 ÷ 点数**（固定額にすると投資が実態とずれ ROI が壊れる）
  5. **他ランクとの重複を候補段階で排除しない**（排除は netkeirin 入稿だけ）
"""
from __future__ import annotations

import pytest

from src import strategy_wt as sw


# ── 1. 軸選定 ────────────────────────────────────────────────────────

def test_axis_is_top2_of_pred_top3():
    p3 = {1: 0.30, 2: 0.88, 3: 0.10, 4: 0.55, 5: 0.82, 6: 0.20, 7: 0.15}
    a1, a2, s = sw.rank_7c_select_axis(p3)
    assert (a1, a2) == (2, 5)
    assert s == pytest.approx(1.70)


def test_axis_ties_break_by_frame_no():
    """同値のときは車番の小さい方を上位にする（実行ごとに入れ替わらないこと）。"""
    p3 = {1: 0.50, 2: 0.50, 3: 0.50, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10}
    assert sw.rank_7c_select_axis(p3)[:2] == (1, 2)


def test_axis_returns_none_when_too_few_cars():
    assert sw.rank_7c_select_axis({}) is None
    assert sw.rank_7c_select_axis({1: 0.5}) is None


# ── 2〜3. 相手選択と点数ゲート ────────────────────────────────────────

def test_legs_keep_only_cars_above_threshold():
    p3 = {1: 0.88, 2: 0.82, 3: 0.60, 4: 0.25, 5: 0.16, 6: 0.14, 7: 0.05}
    legs = sw.rank_7c_select_legs([3, 4, 5, 6, 7], p3)
    assert legs == [3, 4, 5]          # 0.14 と 0.05 は落ちる
    assert all(p3[x] >= sw.RANK_7C_LEG_P3_MIN for x in legs)


def test_legs_are_sorted_by_probability_desc():
    p3 = {3: 0.20, 4: 0.60, 5: 0.40, 6: 0.50, 7: 0.30}
    assert sw.rank_7c_select_legs([3, 4, 5, 6, 7], p3) == [4, 6, 5, 7, 3]


def test_legs_do_not_backfill_when_all_below_threshold():
    """🔴 最低1車を補ってはいけない。補うと点数ゲートが機能しなくなる。"""
    p3 = {3: 0.10, 4: 0.08, 5: 0.05, 6: 0.03, 7: 0.01}
    assert sw.rank_7c_select_legs([3, 4, 5, 6, 7], p3) == []


def _cand(sum2, n_legs, rk="20260807_01_01"):
    return {"race_key": rk, "p3_sum_top2": sum2,
            "legs_7c": list(range(3, 3 + n_legs))}


# ── 低配当パターンの除外 ────────────────────────────────────────────

def _p3(vals):
    return {i + 1: v for i, v in enumerate(vals)}


def test_lowpay_pattern_needs_both_gap_and_same_line():
    """🔴 片方だけでは切らない。両方揃ったときだけ見送る。"""
    # 豊橋5R(2026-08-07) 相当: 3-4位差31.8pt ∧ 上位3車が同一ライン
    p3 = {7: 0.825, 4: 0.626, 2: 0.590, 5: 0.272, 1: 0.233, 6: 0.206, 3: 0.168}
    same = {7: 1, 4: 1, 2: 1, 5: 5, 1: 2, 6: 4, 3: 3}
    assert sw.rank_7c_is_lowpay_pattern(p3, same) is True
    # 3車が抜けているが同一ラインではない → 買う
    diff = {7: 1, 4: 2, 2: 3, 5: 5, 1: 2, 6: 4, 3: 3}
    assert sw.rank_7c_is_lowpay_pattern(p3, diff) is False
    # 同一ラインだが3-4位差が小さい → 買う
    flat = {7: 0.825, 4: 0.626, 2: 0.590, 5: 0.560, 1: 0.233, 6: 0.206, 3: 0.168}
    assert sw.rank_7c_is_lowpay_pattern(flat, same) is False


def test_lowpay_pattern_is_false_when_line_unknown():
    """ライン不明を理由に推奨を減らさない（安全側は『買う』）。"""
    p3 = {7: 0.825, 4: 0.626, 2: 0.590, 5: 0.272, 1: 0.233, 6: 0.206, 3: 0.168}
    assert sw.rank_7c_is_lowpay_pattern(p3, None) is False
    assert sw.rank_7c_is_lowpay_pattern(p3, {7: None, 4: 1, 2: 1}) is False


def test_daily_select_excludes_lowpay_pattern():
    c = _cand(1.60, 5, "x")
    assert sw.rank_7c_daily_select([c])          # 通常は通る
    assert sw.rank_7c_daily_select([{**c, "lowpay_pattern": True}]) == []
    # 旧形式（キー欠損）は False 扱いで落とさない
    assert sw.rank_7c_daily_select([{**c, "lowpay_pattern": None}])


def test_daily_select_requires_both_gates():
    cands = [
        _cand(1.60, 5, "a"),   # 通過
        _cand(1.60, 4, "b"),   # 通過（下限ちょうど）
        _cand(1.60, 3, "c"),   # ✂ 点数不足
        _cand(1.43, 5, "d"),   # ✂ 合計不足
        _cand(sw.RANK_7C_P3_SUM_MIN, 4, "e"),  # 通過（閾値ちょうど）
    ]
    got = [c["race_key"] for c in sw.rank_7c_daily_select(cands)]
    assert set(got) == {"a", "b", "e"}


def test_daily_select_sorts_by_confidence_desc():
    cands = [_cand(1.50, 4, "lo"), _cand(1.90, 4, "hi"), _cand(1.70, 4, "mid")]
    assert [c["race_key"] for c in sw.rank_7c_daily_select(cands)] == ["hi", "mid", "lo"]


def test_daily_select_tolerates_missing_fields():
    assert sw.rank_7c_daily_select([{"race_key": "x"}]) == []
    assert sw.rank_7c_daily_select([{"race_key": "x", "p3_sum_top2": 1.9}]) == []


def test_legs_min_is_four():
    """低配当回避の本体。緩めると的中のうち2倍以下の割合が跳ね上がる
    （実測 4点24.9% → 3点45.6% → 2点67.0%）。"""
    assert sw.RANK_7C_LEGS_MIN == 4


def test_selection_thresholds_are_absolute_not_relative():
    """日次の相対順位に変えてはいけない（7H1 で件数が半減した前例）。"""
    assert isinstance(sw.RANK_7C_P3_SUM_MIN, float)
    assert 1.0 < sw.RANK_7C_P3_SUM_MIN < 2.0
    assert 0.0 < sw.RANK_7C_LEG_P3_MIN < 1.0


# ── 4. 賭け金 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expected", [(1, 10000), (2, 5000), (3, 3300),
                                        (4, 2500), (5, 2000)])
def test_unit_stake_splits_budget_and_floors_to_100(n, expected):
    assert sw.rank_7c_unit_stake(n) == expected


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_total_never_exceeds_budget_and_is_100yen_units(n):
    s = sw.rank_7c_unit_stake(n)
    assert s % sw.RANK_7C_UNIT == 0
    assert n * s <= sw.RANK_7C_BUDGET


def test_unit_stake_does_not_crash_on_zero():
    assert sw.rank_7c_unit_stake(0) == sw.RANK_7C_UNIT


# ── 5. 単一正本への登録と重複方針 ─────────────────────────────────────

def test_registered_in_current_paper_ranks():
    spec = next(s for s in sw.CURRENT_PAPER_RANKS if s.rank == "RANK_7C")
    assert (spec.suffix, spec.label) == ("#7C", "7C")
    # 母集団の性格が既存ランクと違うのでヘッダー合計には混ぜない
    assert spec.in_header_total is False


def test_not_in_abolished_blacklist():
    assert "RANK_7C" not in sw.ABOLISHED_PAPER_RANK_NAMES


def test_daily_select_does_not_dedupe_against_other_ranks():
    """🔴 7C は他ランクと同一レースに併存するのが正常。
    picks_history の race_key は `{レースキー}#{suffix}` なので行は共存できる。
    ここで排他にすると母集団が削れ、実測と乖離する。"""
    import inspect
    src = inspect.getsource(sw.rank_7c_daily_select)
    assert "claimed" not in src
    assert "wt_overlap_n" not in src


# ── netkeirin 入稿の優先順位・賭け金解決 ──────────────────────────────

def test_netkeirin_priority_order():
    """優先順位 7H1 > 7SS > 7S > 7A > **7C > 7B**（2026-08-07 ユーザー指定）。

    RANK_ORDER は dict の定義順なので、順序が入れ替わると黙って優先度が変わる。

    ⚠️ 同日中に **7B を 7C の後ろへ移した**。7C との重複では 7C が実質的中率で
       上回る（39.0% vs 31.6%）一方、7B は 7C が拾わないレースを 3.14件/日 持つ。
       「重複は 7C・独自は 7B」を優先順位だけで実現している。
    """
    from scripts import netkeirin_submit_wt as ns
    # 9車ランクは7車ランクと母集団が排他なので、優先順位の検査から外す
    order = [r for r in ns.RANK_ORDER if r not in ("9S", "9A", "9H1")]
    assert order == ["7H1", "7SS", "7S", "7A", "7C", "7B"]


def test_netkeirin_priority_order_9car():
    """9車の優先順位 9H1 > 9S > 9A（2026-08-08・9H1 新設時のユーザー判断）。

    同じ9車レースで重なったら**穴推奨の 9H1 が取る**。9H1 は約1件/日と薄いので
    9S/9A（3.96件/日）が失う分は小さい。入れ替えたければ RANK_CONFIGS の定義順を
    変える（RANK_ORDER はその導出なので、片方だけ直すことはできない）。
    """
    from scripts import netkeirin_submit_wt as ns
    order = [r for r in ns.RANK_ORDER if r in ("9S", "9A", "9H1")]
    assert order == ["9H1", "9S", "9A"]


def test_netkeirin_7c_uses_budget_and_own_axis_keys():
    from scripts import netkeirin_submit_wt as ns
    cfg = ns.RANK_CONFIGS["7C"]
    assert cfg["axis_keys"] == ("axis1_7c", "axis2_7c")
    # 🔴 買う相手は `legs_7c_buy`（三連単=全部 / 三連複=上位2点・2026-08-09）。
    #    選別用の `legs_7c`（4〜5点）を読むと絞り込みが効かない。
    assert cfg["partners_key"] == "legs_7c_buy"
    assert cfg["stake_budget"] == sw.RACE_BUDGET
    assert "stake_per_line" not in cfg      # 固定額と併記すると取り違える
    assert cfg["overlap_expected"] is True  # 衝突は想定内＝失敗集計に混ぜない
    # タイトル・文面は 7A と同じ既定テンプレート（ユーザー指示 2026-08-07）
    assert "default_comment" not in cfg


def test_all_ranks_invest_one_race_budget():
    """🔴 全ランクが1レース RACE_BUDGET 円に揃っていること（2026-08-07 統一）。
    固定単価に戻すと点数が変わったとき投資額がずれ、Web の比較が壊れる。"""
    from scripts import netkeirin_submit_wt as ns
    for rank, n_pts in (("7SS", 5), ("7S", 5), ("7A", 5), ("7B", 3),
                        ("9S", 7), ("9A", 7), ("7C", 4), ("7C", 5)):
        cfg = ns.RANK_CONFIGS[rank]
        total = n_pts * ns._stake_per_line(cfg, n_pts)
        assert sw.RACE_BUDGET - 200 <= total <= sw.RACE_BUDGET, (rank, n_pts, total)


def test_default_comment_does_not_hardcode_point_count():
    """既定文面は 7S/7A/7SS(5点) と 7C(4〜5点) が共有するので点数を書かない。"""
    from scripts import netkeirin_submit_wt as ns
    assert "5点" not in ns._DEFAULT_COMMENT_TEMPLATE


def test_netkeirin_stake_resolution():
    """2026-08-07 に全ランクが予算枠方式へ統一されたので、点数が同じなら
    ランクによらず同じ単価になる。"""
    from scripts import netkeirin_submit_wt as ns
    assert ns._stake_per_line(ns.RANK_CONFIGS["7C"], 4) == 2500
    assert ns._stake_per_line(ns.RANK_CONFIGS["7C"], 5) == 2000
    assert ns._stake_per_line(ns.RANK_CONFIGS["7S"], 5) == 2000
    assert ns._stake_per_line(ns.RANK_CONFIGS["7B"], 3) == 3300
    assert ns._stake_per_line(ns.RANK_CONFIGS["9S"], 7) == 1400


def test_netkeirin_7c_normalizes_with_its_own_axes():
    """`axis1`（3ヘッド軸）が併存していても 7C は自分の軸を使うこと。"""
    from scripts import netkeirin_submit_wt as ns
    cand = {"axis1": 1, "axis2": 2, "axis1_7c": 5, "axis2_7c": 6,
            "legs_7c": [3, 4, 7, 1], "legs_7c_buy": [3, 4]}
    a1, a2, partners, marks = ns._normalize_candidate(cand, ns.RANK_CONFIGS["7C"])
    assert (a1, a2) == (5, 6)
    # 買うのは `legs_7c_buy`。選別用の `legs_7c` を読んではいけない。
    assert partners == [3, 4]
    assert marks == {5: "◎", 6: "○"}


def test_netkeirin_7c_uses_same_template_as_7a():
    """タイトル・文面は 7A と同じ（ユーザー指示 2026-08-07）。
    どちらも rank 固有の上書きを持たず既定テンプレートへ落ちる。"""
    from scripts import netkeirin_submit_wt as ns
    assert "default_comment" not in ns.RANK_CONFIGS["7A"]
    assert "default_comment" not in ns.RANK_CONFIGS["7C"]
    assert "default_title" not in ns.RANK_CONFIGS["7C"]


# ── 発走前のライブ判定（盤面・欠車の扱い）───────────────────────────────

def _trio_lookup(cars):
    from itertools import combinations

    from scripts.notify_prerace_wt import _parse_combo_key
    return {_parse_combo_key("=".join(map(str, sorted(c))), False): 12.3
            for c in combinations(cars, 3)}


@pytest.fixture
def cand_7c():
    """軸=2,5。相手候補 1(0.40) 3(0.30) 4(0.20) 6(0.16) 7(0.05)。"""
    return {"race_key": "20260807_01_01", "venue_name": "T", "race_no": 1,
            "axis1": 2, "axis2": 5, "p3_sum_top2": 1.70,
            "legs_7c": [1, 3, 4, 6],
            "top3_probs": {"1": 0.40, "2": 0.90, "3": 0.30, "4": 0.20,
                           "5": 0.80, "6": 0.16, "7": 0.05}}


def test_judge_buys_and_sets_variable_stake(cand_7c):
    from scripts.notify_prerace_wt import judge_rank_7c
    decision, detail = judge_rank_7c(cand_7c, _trio_lookup([1, 2, 3, 4, 5, 6, 7]))
    assert decision == "buy"
    # 2026-08-09: 相手はギャップ（3着内率の落差 >= 0.15）でだけ削る。
    # このレースは 0.40/0.30/0.20/0.16 と落差が全て 0.15 未満なので**総流し**。
    assert detail["thirds"] == [1, 3, 4, 6]        # 7番(0.05)は足切り
    assert len(detail["combos"]) == 4
    assert detail["stake"] == sw.rank_7c_unit_stake(4) == 2500


def test_judge_keys_leg_odds_by_combo_label():
    """🔴 leg_odds のキーは買い目の文字列。3列目の車番にすると Discord 通知が
    全件『取得不可』になる（2026-08-07 の 7C 初日に実際に発生）。"""
    from scripts.notify_prerace_wt import judge_rank_7c
    cand = {"race_key": "r", "axis1": 2, "axis2": 5, "p3_sum_top2": 1.70,
            "legs_7c": [1, 3, 4, 6],
            "top3_probs": {"1": 0.40, "2": 0.90, "3": 0.30, "4": 0.20,
                           "5": 0.80, "6": 0.16, "7": 0.05}}
    _, detail = judge_rank_7c(cand, _trio_lookup([1, 2, 3, 4, 5, 6, 7]))
    assert set(detail["leg_odds"]) == set(detail["combos"])
    assert all(detail["leg_odds"][c] is not None for c in detail["combos"])


def test_judge_skips_when_legs_fall_below_minimum(cand_7c):
    """🔴 相手が4点未満になったら買わない（低配当回避の本体）。"""
    from scripts.notify_prerace_wt import judge_rank_7c
    cand_7c["top3_probs"]["6"] = 0.10             # 4点 → 3点へ
    decision, detail = judge_rank_7c(cand_7c, _trio_lookup([1, 2, 3, 4, 5, 6, 7]))
    assert decision == "skip"
    assert "3点" in (detail["skip_reason"] or "")


def test_judge_skips_on_scratch(cand_7c):
    from scripts.notify_prerace_wt import judge_rank_7c
    decision, detail = judge_rank_7c(cand_7c, _trio_lookup([1, 2, 3, 4, 5, 6]))
    assert decision == "skip"
    assert "欠車" in (detail["skip_reason"] or "")


def test_judge_skips_when_axis_missing_from_board(cand_7c):
    from scripts.notify_prerace_wt import judge_rank_7c
    cand_7c["axis1"] = 9
    decision, detail = judge_rank_7c(cand_7c, _trio_lookup([1, 2, 3, 4, 5, 6, 7]))
    assert decision == "skip"
    assert "盤面に不在" in (detail["skip_reason"] or "")


def test_judge_returns_unknown_without_board(cand_7c):
    """盤面が取れないときは skip ではなく「不明」で次分に再試行する。"""
    from scripts.notify_prerace_wt import judge_rank_7c
    assert judge_rank_7c(cand_7c, {})[0] == "不明"


def test_judge_recomputes_legs_from_board_not_morning_json(cand_7c):
    """朝の legs_7c を鵜呑みにせず盤面と確率から引き直すこと。"""
    from scripts.notify_prerace_wt import judge_rank_7c
    cand_7c["legs_7c"] = [6, 4]                   # 朝の値が壊れていても
    decision, detail = judge_rank_7c(cand_7c, _trio_lookup([1, 2, 3, 4, 5, 6, 7]))
    # 盤面と確率から引き直すこと。朝の値をそのまま使えば [6, 4] になる。
    assert decision == "buy" and detail["thirds"] == [1, 3, 4, 6]


# ── walk-forward 再構築の登録漏れ防止 ───────────────────────────────

def test_reconcile_registers_7c():
    """rebuild スクリプトがあるのに reconcile へ登録し忘れると、直近日の 7C だけ
    live 記録のまま残り honest な再構築が当たらない（7A/7B で実際に混在が起きた）。"""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert (root / "scripts" / "rebuild_7c_walkforward_pg.py").exists()
    sh = (root / "scripts" / "reconcile_walkforward_tail.sh").read_text()
    assert '"7c:7C"' in sh


def test_backfill_7c_does_not_require_win_or_bad_models():
    """7C は pred_prob だけで軸も相手も決まる。win/bad を必須にすると
    vintage 不足で無意味に窓が落ちる。"""
    import inspect

    from scripts import backfill_7c_rank_wt as bf
    src = inspect.getsource(bf.build_rows)
    assert "load_model(win_model_name)" not in src
    assert "pred_bad" not in src

    from scripts import rebuild_7c_walkforward_pg as rb
    rsrc = inspect.getsource(rb.main)
    assert "require_bad=True" not in rsrc


# ─────────────────────────────────────────────────────────────────────────
# 予算定数の単一正本（2026-08-08 レビュー指摘 L-2）
# ─────────────────────────────────────────────────────────────────────────

def test_rank_budget_constants_derive_from_single_source():
    """ランク別の予算・単位が `RACE_BUDGET`/`STAKE_UNIT` から導かれること。

    以前は `RANK_7C_BUDGET = 10000` / `RANK_7C_UNIT = 100` のように
    **同じ値を別途リテラルで再定義**していた。netkeirin 入稿側
    （`netkeirin_submit_wt.RANK_CONFIGS`）は `RACE_BUDGET` を直接参照するので、
    全ランク一律で予算を改定したときに「実際に入稿される金額」と
    「picks_history に記録される賭け金」が**無言で食い違う**。

    値の一致を確かめるだけでは足りない（改定直後は両方10000で通ってしまう）ので、
    片方を変えたら他方も追随することを確かめる。
    """
    import importlib
    import src.strategy_wt as sw

    assert sw.RANK_7C_BUDGET == sw.RACE_BUDGET
    assert sw.RANK_7C_UNIT == sw.STAKE_UNIT
    assert sw.RANK_7H1_BUDGET_CAP == sw.RACE_BUDGET
    assert sw.RANK_7H1_UNIT == sw.STAKE_UNIT

    # 参照になっているか（リテラル再定義でないか）を実際に確かめる
    orig_budget, orig_unit = sw.RACE_BUDGET, sw.STAKE_UNIT
    try:
        sw.RACE_BUDGET, sw.STAKE_UNIT = 12000, 200
        reloaded = importlib.reload(sw)
        assert reloaded.RANK_7C_BUDGET == reloaded.RACE_BUDGET, (
            "RANK_7C_BUDGET が RACE_BUDGET に追随していない（リテラル再定義に戻された）")
        assert reloaded.RANK_7H1_BUDGET_CAP == reloaded.RACE_BUDGET
    finally:
        importlib.reload(sw)


def test_netkeirin_submit_uses_the_same_budget_source():
    """入稿側の予算も同じ正本を参照していること。"""
    import src.strategy_wt as sw
    from scripts.netkeirin_submit_wt import RANK_CONFIGS

    for rank, cfg in RANK_CONFIGS.items():
        budget = cfg.get("stake_budget")
        if budget is None:
            continue
        assert budget == sw.RACE_BUDGET, (
            f"{rank} の stake_budget({budget}) が RACE_BUDGET({sw.RACE_BUDGET}) と違う。"
            " 入稿額と picks_history の記録額が食い違う")


def test_documented_priority_order_matches_rank_configs():
    """ドキュメントに書かれた入稿優先順位が実装と一致すること（2026-08-08）。

    2026-08-07 に 7B を 7C の後ろへ動かした際、`RANK_CONFIGS` の定義順（正本）は
    正しく変わったのに、**先に書かれていた解説文3箇所が古いまま**残り
    `7H1 > 7SS > 7S > 7A > 7B > 7C` という逆順を主張していた
    （CLAUDE.md / docs/system-architecture.md / strategy_wt.py のコメント）。

    CLAUDE.md は AI エージェントが正本として読むファイルなので、ここに誤った
    順序が書かれていると「ドキュメントに合わせて実装を直す」形で本番の優先順位が
    壊れうる。文字列として機械的に照合する。
    """
    from pathlib import Path
    from scripts.netkeirin_submit_wt import RANK_ORDER

    seven = [r for r in RANK_ORDER if r.startswith("7")]
    expected = " > ".join(seven)

    repo = Path(__file__).resolve().parent.parent
    targets = [
        repo / "CLAUDE.md",
        repo / "docs" / "system-architecture.md",
        repo / "src" / "strategy_wt.py",
        repo / "scripts" / "netkeirin_submit_wt.py",
    ]
    checked = 0
    for p in targets:
        text = p.read_text(encoding="utf-8")
        if "7H1 > 7SS" not in text:
            continue
        checked += 1
        assert expected in text, (
            f"{p.name} の優先順位表記が実装（{expected}）と食い違う")
        # 旧・逆順が残っていないこと
        assert "7A > 7B > 7C" not in text, (
            f"{p.name} に古い優先順位 '7A > 7B > 7C' が残っている（実装は {expected}）")
    assert checked == len(targets), (
        "優先順位を書いているはずのファイルで表記が見つからなかった"
        f"（照合できたのは {checked}/{len(targets)} 件）。"
        " 表記を変えたなら本テストの targets も追随させること")
