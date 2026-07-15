"""notify_prerace_wt._determine_live_rank の 7PLUS_R 判定テスト（2026-07-10 SS/S置き換え）。

Rランク = レース単位セマンティクス: min(全目) >= GAMI_THRESHOLD(7.0)
∧ gap12 >= 0.10 ∧ gap23 >= 1pt で全目購入。買い目カット・SOフィルタは廃止。
2026-07-16〜: 選抜のみ見送り（4分戦カット・格差増額は実精算再検証で廃止）。旧doc53:
ライン平均得点格差 >= 1.5 は 200円/点に増額。
"""
import notify_prerace_wt as np_wt  # scripts/ は conftest で path 追加済


def _pick(gap12=0.12, thirds=(3, 4, 5, 6, 7), gap23_pct=2.0,
          race_type="Ａ級一般", line_avg_gap=0.5, line_n_lines=3, line_all_solo=False):
    """gap23 は riders の pred_prob_pct (ai_rank 2位-3位差) から計算される。

    race_type / line_* は doc53 ポリシーコンテキスト（candidates.json 由来の形）。
    キーが存在するため _policy_ctx は DB を引かない。
    """
    riders = [
        {"ai_rank": 1, "pred_prob_pct": 60.0},
        {"ai_rank": 2, "pred_prob_pct": 40.0},
        {"ai_rank": 3, "pred_prob_pct": 40.0 - gap23_pct},
    ]
    return {
        "pivot1": 1, "pivot2": 2,
        "thirds": list(thirds),
        "gap12": gap12,
        "riders": riders,
        "race_type": race_type,
        "line_avg_gap": line_avg_gap,
        "line_n_lines": line_n_lines,
        "line_all_solo": line_all_solo,
    }


def _odds_data(leg_odds: dict):
    return {"trio": [
        {"combination": f"1-2-{t}", "odds_value": o} for t, o in leg_odds.items()
    ]}


LEGS_OK = {3: 7.0, 4: 7.2, 5: 10.2, 6: 8.0, 7: 9.0}


def test_r_when_all_legs_above_threshold():
    """全目 min>=7.0 → 7PLUS_R（全目購入・基本100円/点）。SOは適用しない。"""
    rank, thirds, _, stake, reason = np_wt._determine_live_rank(_pick(), _odds_data(LEGS_OK))
    assert rank == "7PLUS_R"
    assert thirds == [3, 4, 5, 6, 7]
    assert stake == np_wt.SS_STAKE
    assert reason is None


def test_no_r_when_min_below_threshold():
    """min < 7.0 → レースごと見送り（買い目カットで残す旧SS挙動はしない）。"""
    legs = {3: 6.9, 4: 8.0, 5: 15.0, 6: 30.0, 7: 40.0}
    rank, thirds, _, _, reason = np_wt._determine_live_rank(_pick(), _odds_data(legs))
    assert rank == "なし"
    assert thirds == []
    assert reason is None  # オッズ条件による見送り


def test_no_ss_cut_revival():
    """旧SS条件（カット後1-3目・高SO）でも min<7 なら見送り = SS廃止の確認。"""
    legs = {3: 6.2, 4: 6.5, 5: 15.0, 6: 30.0, 7: 40.0}  # 旧仕様ならSS(3目)だった形
    rank, _, _, _, _ = np_wt._determine_live_rank(_pick(), _odds_data(legs))
    assert rank == "なし"


def test_no_r_when_gap12_below_010():
    """min>=7 でも gap12 < 0.10 → 不成立。"""
    rank, _, _, _, _ = np_wt._determine_live_rank(_pick(gap12=0.08), _odds_data(LEGS_OK))
    assert rank == "なし"


def test_no_r_when_gap23_below_1pt():
    """min>=7 でも gap23 < 1pt → 不成立。"""
    rank, _, _, _, _ = np_wt._determine_live_rank(_pick(gap23_pct=0.5), _odds_data(LEGS_OK))
    assert rank == "なし"


def test_unknown_when_no_odds():
    """オッズ取得失敗 → 不明（再試行対象）。"""
    rank, _, _, _, _ = np_wt._determine_live_rank(_pick(), None)
    assert rank == "不明"


# ── doc53 統合ポリシー ─────────────────────────────────────────────────────

def test_skip_senbatsu():
    """選抜レースはオッズ条件成立でも見送り（skip_reason="選抜"）。"""
    rank, _, _, _, reason = np_wt._determine_live_rank(
        _pick(race_type="Ａ級選抜"), _odds_data(LEGS_OK))
    assert rank == "なし"
    assert reason == "選抜"


def test_skip_challenge_senbatsu():
    """チャレンジ選抜も選抜として見送り。"""
    rank, _, _, _, reason = np_wt._determine_live_rank(
        _pick(race_type="Ａ級チャレンジ選抜"), _odds_data(LEGS_OK))
    assert rank == "なし"
    assert reason == "選抜"


def test_no_skip_four_lines():
    """4分戦カットは2026-07-16廃止 → ライン数>=4でも購入（100円/点）。"""
    rank, _, _, stake, _ = np_wt._determine_live_rank(
        _pick(line_n_lines=4), _odds_data(LEGS_OK))
    assert rank == "7PLUS_R"
    assert stake == np_wt.SS_STAKE


def test_no_boost_when_line_gap_large():
    """格差増額は2026-07-16廃止 → ライン格差が大きくても常に100円/点。"""
    rank, _, _, stake, _ = np_wt._determine_live_rank(
        _pick(line_avg_gap=2.1), _odds_data(LEGS_OK))
    assert rank == "7PLUS_R"
    assert stake == np_wt.SS_STAKE


def test_no_boost_without_line_info():
    """ライン情報欠損（None）でも通常購入（100円/点）。"""
    rank, _, _, stake, _ = np_wt._determine_live_rank(
        _pick(line_avg_gap=None, line_n_lines=None, line_all_solo=None),
        _odds_data(LEGS_OK))
    assert rank == "7PLUS_R"
    assert stake == np_wt.SS_STAKE


def test_senbatsu_skip_precedes_odds():
    """選抜は他条件が全て良くても見送り。"""
    rank, _, _, _, reason = np_wt._determine_live_rank(
        _pick(race_type="Ｓ級選抜", line_avg_gap=3.0), _odds_data(LEGS_OK))
    assert rank == "なし"
    assert reason == "選抜"
