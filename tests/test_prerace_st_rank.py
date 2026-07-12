"""notify_prerace_wt._determine_st_rank（三連単Sランク）判定テスト（2026-07-10）。

S = 1着=指数1位固定 / 2着=指数2,3位 / 3着=全通り。
gap12>=0.15 ∧ 購入全目の三連単オッズ min>=10 → 7PLUS_ST（100円/点）。
さらに gap12>=0.25 ∧ gap34>=0.04 → 7PLUS_STP（200円/点・増額）。
"""
import notify_prerace_wt as np_wt  # scripts/ は conftest で path 追加済


def _pick(gap12=0.18, thirds=(3, 4, 5, 6, 7), probs=(60.0, 40.0, 30.0, 20.0)):
    """probs = 指数1〜4位の pred_prob_pct（gap34 = (p3-p4)/100）。"""
    riders = [{"ai_rank": i + 1, "pred_prob_pct": p} for i, p in enumerate(probs)]
    return {
        "pivot1": 1, "pivot2": 2,
        "thirds": list(thirds),
        "gap12": gap12,
        "riders": riders,
    }


def _odds_data(base_odds=20.0):
    """全買い目（1着=1, 2着∈{2,3}, 3着=残り）に base_odds を付けた odds_data。"""
    frames = [1, 2, 3, 4, 5, 6, 7]
    items = []
    for s in (2, 3):
        for t in frames:
            if t in (1, s):
                continue
            items.append({"combination": f"1-{s}-{t}", "odds_value": base_odds})
    return {"trifecta": items}


def test_st_basic():
    """gap12>=0.15 ∧ 全目min>=10 → 7PLUS_ST・2×(n-2)=10点・100円/点。"""
    rank, combos, leg_odds, stake = np_wt._determine_st_rank(_pick(), _odds_data(20.0))
    assert rank == "7PLUS_ST"
    assert len(combos) == 10  # 7車: 2着2通り × 3着5通り
    assert stake == np_wt.ST_STAKE
    assert all(c[0] == 1 and c[1] in (2, 3) for c in combos)


def test_stp_upgrade():
    """gap12>=0.25 ∧ gap34>=0.04 → 7PLUS_STP（増額200円/点）。"""
    pick = _pick(gap12=0.30, probs=(60.0, 30.0, 25.0, 15.0))  # gap34=10pt=0.10
    rank, combos, _, stake = np_wt._determine_st_rank(pick, _odds_data(20.0))
    assert rank == "7PLUS_STP"
    assert stake == np_wt.STP_STAKE


def test_no_stp_when_gap34_small():
    """gap12>=0.25 でも gap34<0.04 なら通常 S。"""
    pick = _pick(gap12=0.30, probs=(60.0, 30.0, 25.0, 23.0))  # gap34=2pt=0.02
    rank, _, _, stake = np_wt._determine_st_rank(pick, _odds_data(20.0))
    assert rank == "7PLUS_ST"
    assert stake == np_wt.ST_STAKE


def test_skip_when_min_odds_below_10():
    """購入目に10倍未満の目があればレースごと見送り（レース単位ガミ条件）。"""
    odds = _odds_data(20.0)
    odds["trifecta"][0]["odds_value"] = 8.0  # 1目だけ8倍
    rank, combos, _, _ = np_wt._determine_st_rank(_pick(), odds)
    assert rank == "なし"
    assert combos == []


# ── doc53: S通常帯 min>=15 ∧ 非選抜（S+帯は現行 min>=10 のまま） ──────────

def test_st_normal_skip_when_min_between_10_and_15():
    """S通常帯: min が [10,15) はガミ閾値引き上げ（doc53）により見送り。"""
    rank, combos, _, _ = np_wt._determine_st_rank(_pick(), _odds_data(12.0))
    assert rank == "なし"
    assert combos == []


def test_stp_still_allowed_when_min_between_10_and_15():
    """S+帯は現行条件のまま: min が [10,15) でも成立する。"""
    pick = _pick(gap12=0.30, probs=(60.0, 30.0, 25.0, 15.0))  # gap34=0.10
    rank, _, _, stake = np_wt._determine_st_rank(pick, _odds_data(12.0))
    assert rank == "7PLUS_STP"
    assert stake == np_wt.STP_STAKE


def test_st_normal_skip_senbatsu():
    """S通常帯: 選抜レースは min>=15 でも見送り。"""
    rank, _, _, _ = np_wt._determine_st_rank(
        _pick(), _odds_data(20.0), race_type="Ａ級選抜")
    assert rank == "なし"


def test_stp_allowed_in_senbatsu():
    """S+帯は選抜カットの対象外（現行条件維持）。"""
    pick = _pick(gap12=0.30, probs=(60.0, 30.0, 25.0, 15.0))
    rank, _, _, _ = np_wt._determine_st_rank(pick, _odds_data(20.0), race_type="Ａ級選抜")
    assert rank == "7PLUS_STP"


def test_skip_when_gap12_below_015():
    rank, _, _, _ = np_wt._determine_st_rank(_pick(gap12=0.12), _odds_data(20.0))
    assert rank == "なし"


def test_unknown_when_no_odds():
    rank, _, _, _ = np_wt._determine_st_rank(_pick(), None)
    assert rank == "不明"
    rank2, _, _, _ = np_wt._determine_st_rank(_pick(), {"trifecta": []})
    assert rank2 == "不明"
