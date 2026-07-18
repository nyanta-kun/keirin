"""strategy_wt.m_axis_gate（S3軸信頼ゲート・gap12 OR win_rank OR ratio）の純関数テスト。

2026-07-19 Phase B: win_rank（システム◎の1着モデル内レース順位）ゲートを
gap12（3着内モデルの軸信頼ゲート）にOR統合。両者はほぼ独立したシグナル
（exp_win_axis_sweep_wt.py で重複率5%・相関-0.265を確認）。
2026-07-19 複合シグナル追加: ratio（p_win/p_top3比）を第3項としてOR統合
（exp_composite_prob_diff_wt.py・母数+22〜26%・ROI維持〜微増を確認）。
"""
from src.strategy_wt import M_GAP12_MIN, M_RATIO_MAX, M_WIN_RANK_MIN, m_axis_gate


def test_gap12_only_passes():
    """gap12>=閾値・win_rankなし（1着モデル未ロード）→ gap12ゲートで通過。"""
    passed, gate = m_axis_gate(M_GAP12_MIN, None)
    assert passed is True
    assert gate == "gap12"


def test_gap12_below_and_no_win_rank_fails():
    """gap12<閾値・win_rankなし → 不成立。"""
    passed, gate = m_axis_gate(M_GAP12_MIN - 0.01, None)
    assert passed is False
    assert gate is None


def test_win_rank_only_passes():
    """gap12<閾値・win_rank>=閾値 → win_rankゲートで通過。"""
    passed, gate = m_axis_gate(0.0, M_WIN_RANK_MIN)
    assert passed is True
    assert gate == "win_rank"


def test_win_rank_below_threshold_fails():
    """gap12<閾値・win_rank<閾値 → 不成立。"""
    passed, gate = m_axis_gate(0.0, M_WIN_RANK_MIN - 1)
    assert passed is False
    assert gate is None


def test_both_pass_gap12_takes_priority_label():
    """両方成立時は gap12 ラベルを優先表示する。"""
    passed, gate = m_axis_gate(M_GAP12_MIN, M_WIN_RANK_MIN)
    assert passed is True
    assert gate == "gap12"


def test_win_rank_one_boundary_fails():
    """win_rank=1（一致相当・最上位評価）は M_WIN_RANK_MIN 未満で不成立。"""
    passed, gate = m_axis_gate(0.0, 1)
    assert passed is False
    assert gate is None


def test_ratio_only_passes():
    """gap12<閾値・win_rankなし・ratio<=閾値 → ratioゲートで通過。"""
    passed, gate = m_axis_gate(0.0, None, M_RATIO_MAX)
    assert passed is True
    assert gate == "ratio"


def test_ratio_above_threshold_fails():
    """gap12<閾値・win_rankなし・ratio>閾値 → 不成立。"""
    passed, gate = m_axis_gate(0.0, None, M_RATIO_MAX + 0.01)
    assert passed is False
    assert gate is None


def test_ratio_none_falls_back_to_others():
    """ratio未算出（None）でも gap12/win_rank ゲートは従来通り機能する。"""
    passed, gate = m_axis_gate(M_GAP12_MIN, None, None)
    assert passed is True
    assert gate == "gap12"


def test_all_three_pass_gap12_takes_priority_label():
    """3つとも成立時は gap12 ラベルを優先表示する。"""
    passed, gate = m_axis_gate(M_GAP12_MIN, M_WIN_RANK_MIN, M_RATIO_MAX)
    assert passed is True
    assert gate == "gap12"


def test_win_rank_and_ratio_pass_win_rank_takes_priority_label():
    """gap12不成立・win_rank/ratio両方成立時は win_rank ラベルを優先表示する。"""
    passed, gate = m_axis_gate(0.0, M_WIN_RANK_MIN, M_RATIO_MAX)
    assert passed is True
    assert gate == "win_rank"
