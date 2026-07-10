"""notify_prerace_wt._determine_live_rank の 7PLUS_R 判定テスト（2026-07-10 SS/S置き換え）。

Rランク = レース単位セマンティクス: min(全目) >= GAMI_THRESHOLD(7.0)
∧ gap12 >= 0.10 ∧ gap23 >= 1pt で全目購入。買い目カット・SOフィルタは廃止。
"""
import notify_prerace_wt as np_wt  # scripts/ は conftest で path 追加済


def _pick(gap12=0.12, thirds=(3, 4, 5, 6, 7), gap23_pct=2.0):
    """gap23 は riders の pred_prob_pct (ai_rank 2位-3位差) から計算される。"""
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
    }


def _odds_data(leg_odds: dict):
    return {"trio": [
        {"combination": f"1-2-{t}", "odds_value": o} for t, o in leg_odds.items()
    ]}


def test_r_when_all_legs_above_threshold():
    """全目 min>=7.0 → 7PLUS_R（全目購入）。SOは適用しない（低SOでも成立）。"""
    legs = {3: 7.0, 4: 7.2, 5: 10.2, 6: 8.0, 7: 9.0}  # 全目合成≈1.6だがSO廃止のため成立
    rank, thirds, _ = np_wt._determine_live_rank(_pick(), _odds_data(legs))
    assert rank == "7PLUS_R"
    assert thirds == [3, 4, 5, 6, 7]


def test_no_r_when_min_below_threshold():
    """min < 7.0 → レースごと見送り（買い目カットで残す旧SS挙動はしない）。"""
    legs = {3: 6.9, 4: 8.0, 5: 15.0, 6: 30.0, 7: 40.0}
    rank, thirds, _ = np_wt._determine_live_rank(_pick(), _odds_data(legs))
    assert rank == "なし"
    assert thirds == []


def test_no_ss_cut_revival():
    """旧SS条件（カット後1-3目・高SO）でも min<7 なら見送り = SS廃止の確認。"""
    legs = {3: 6.2, 4: 6.5, 5: 15.0, 6: 30.0, 7: 40.0}  # 旧仕様ならSS(3目)だった形
    rank, _, _ = np_wt._determine_live_rank(_pick(), _odds_data(legs))
    assert rank == "なし"


def test_no_r_when_gap12_below_010():
    """min>=7 でも gap12 < 0.10 → 不成立。"""
    legs = {3: 7.1, 4: 7.5, 5: 8.9, 6: 8.2, 7: 8.8}
    rank, _, _ = np_wt._determine_live_rank(_pick(gap12=0.08), _odds_data(legs))
    assert rank == "なし"


def test_no_r_when_gap23_below_1pt():
    """min>=7 でも gap23 < 1pt → 不成立。"""
    legs = {3: 7.1, 4: 7.5, 5: 8.9, 6: 8.2, 7: 8.8}
    rank, _, _ = np_wt._determine_live_rank(_pick(gap23_pct=0.5), _odds_data(legs))
    assert rank == "なし"


def test_unknown_when_no_odds():
    """オッズ取得失敗 → 不明（再試行対象）。"""
    rank, _, _ = np_wt._determine_live_rank(_pick(), None)
    assert rank == "不明"
