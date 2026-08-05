"""波乱ゲート純粋関数のテスト（境界値・ゲート・カット読込フォールバック）。"""
import json
import importlib

import pytest

import src.strategy_wt as sw


# ── upset_tier 境界値（既定カット 1.70 / 1.90 / 2.08）──
@pytest.mark.parametrize("top3_sum, expected", [
    (1.0, "Q1_loose"),
    (1.6999, "Q1_loose"),
    (1.70, "Q2"),       # カット境界は下側の帯に含めない（< 判定）
    (1.80, "Q2"),
    (1.90, "Q3"),
    (2.00, "Q3"),
    (2.08, "Q4_chalk"),
    (2.50, "Q4_chalk"),
])
def test_upset_tier_boundaries(monkeypatch, top3_sum, expected):
    # 既定カットで判定（JSON の影響を排除）
    monkeypatch.setattr(sw, "UPSET_TOP3SUM_CUTS", sw.UPSET_TOP3SUM_CUTS_DEFAULT)
    assert sw.upset_tier(top3_sum) == expected


# ── passes_upset_gate（loose側のみ通す）──
@pytest.mark.parametrize("top3_sum, max_tier, expected", [
    (1.5, "Q1_loose", True),    # Q1_loose は通る
    (1.8, "Q1_loose", False),   # Q2 は Q1ゲートでは通さない
    (1.8, "Q2", True),          # Q2 までなら通る
    (2.0, "Q2", False),         # Q3 は通さない
    (2.0, "Q3", True),
    (2.2, "Q3", False),         # Q4_chalk は常に通さない
])
def test_passes_upset_gate(monkeypatch, top3_sum, max_tier, expected):
    monkeypatch.setattr(sw, "UPSET_TOP3SUM_CUTS", sw.UPSET_TOP3SUM_CUTS_DEFAULT)
    assert sw.passes_upset_gate(top3_sum, max_tier) is expected


# ── _load_cuts のフォールバック ──
def _write(tmp_path, content):
    p = tmp_path / "upset_cuts_wt.json"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_cuts_valid(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [1.6, 1.8, 2.0]}))
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == (1.6, 1.8, 2.0)


def test_load_cuts_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sw, "_CUTS_PATH", tmp_path / "nope.json")
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_corrupt_json(monkeypatch, tmp_path):
    p = _write(tmp_path, "not a json {")
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_non_monotonic(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [2.0, 1.9, 2.1]}))   # 単調でない
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_wrong_length(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [1.7, 1.9]}))
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


def test_load_cuts_equal_values_rejected(monkeypatch, tmp_path):
    p = _write(tmp_path, json.dumps({"cuts": [1.7, 1.7, 2.0]}))   # 等値は単調NG
    monkeypatch.setattr(sw, "_CUTS_PATH", p)
    assert sw._load_cuts() == sw.UPSET_TOP3SUM_CUTS_DEFAULT


# ── stake_units（波乱ステーク傾斜） ──
@pytest.mark.parametrize("top3_sum, expected_mult", [
    (1.5, 2),    # Q1_loose → 2倍
    (1.8, 1),    # Q2 → 1倍
    (2.0, 0),    # Q3 → 見送り
    (2.3, 0),    # Q4_chalk → 見送り
])
def test_stake_units(monkeypatch, top3_sum, expected_mult):
    monkeypatch.setattr(sw, "UPSET_TOP3SUM_CUTS", sw.UPSET_TOP3SUM_CUTS_DEFAULT)
    assert sw.stake_units(top3_sum) == expected_mult


# ── doc53 統合ポリシー（2026-07-12） ─────────────────────────────────────────

from src.strategy_wt import (  # noqa: E402
    SS_STAKE, is_senbatsu, line_score_features,
    ss_policy,
)


class TestLineScoreFeatures:
    def test_basic_two_lines(self):
        # ライン1: 平均90 / ライン2: 平均88 → avg_gap=2.0
        pairs = [(1, 92.0), (1, 88.0), (2, 88.0), (2, 88.0), (3, 85.0), (3, 85.0), (3, 85.0)]
        gap, n_lines, all_solo = line_score_features(pairs)
        assert gap == 2.0
        assert n_lines == 3
        assert all_solo is False

    def test_all_solo(self):
        pairs = [(i, 80.0 + i) for i in range(1, 8)]
        gap, n_lines, all_solo = line_score_features(pairs)
        assert n_lines == 7
        assert all_solo is True
        assert gap == 1.0  # 86-85（単騎も1本のラインとして格差計算）

    def test_missing_line_group(self):
        pairs = [(1, 90.0), (None, 88.0), (2, 85.0)]
        assert line_score_features(pairs) == (None, None, None)

    def test_single_line(self):
        pairs = [(1, 90.0), (1, 88.0)]
        gap, n_lines, all_solo = line_score_features(pairs)
        assert gap is None
        assert n_lines == 1

    def test_empty(self):
        assert line_score_features([]) == (None, None, None)


class TestSsPolicy:
    def test_normal(self):
        assert ss_policy("Ａ級一般", 0.5, 3, False) == (None, SS_STAKE)

    def test_senbatsu_skip(self):
        reason, _ = ss_policy("Ａ級選抜", 0.5, 3, False)
        assert reason == "選抜"

    def test_four_lines_not_skipped(self):
        # 4分戦カットは2026-07-16廃止
        assert ss_policy("Ａ級一般", 3.0, 4, False) == (None, SS_STAKE)

    def test_no_boost(self):
        # 格差増額は2026-07-16廃止（常に100円/点）
        assert ss_policy("Ａ級一般", 2.0, 3, False) == (None, SS_STAKE)

    def test_none_context_fallback(self):
        assert ss_policy(None, None, None, None) == (None, SS_STAKE)



def test_is_senbatsu():
    assert is_senbatsu("Ａ級選抜")
    assert is_senbatsu("Ａ級チャレンジ選抜")
    assert is_senbatsu("Ｌ級ガールズ選抜")
    assert not is_senbatsu("Ａ級特選")  # 特選は選抜ではない
    assert not is_senbatsu("Ａ級一般")
    assert not is_senbatsu(None)


# ── S7 entropyゲート・件数cap撤廃（2026-07-26） ─────────────────────────────

from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_DAILY_CAP, RANK_7S_ENTROPY_MAX, rank_7s_daily_select, rank_7s_evening_reselect,
    rank_7s_field_entropy, rank_7s_wt_mark3_overlap_n,
)


class TestS7FieldEntropy:
    def test_concentrated_distribution_low_entropy(self):
        # 1車に確率が集中 → entropyはほぼ0
        probs = {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
        assert rank_7s_field_entropy(probs) < 0.01

    def test_uniform_distribution_max_entropy(self):
        # 7車均等 → entropy = ln(7)
        import math
        probs = {i: 1.0 for i in range(1, 8)}
        assert rank_7s_field_entropy(probs) == pytest.approx(math.log(7), abs=1e-6)

    def test_zero_total_returns_zero(self):
        assert rank_7s_field_entropy({1: 0.0, 2: 0.0}) == 0.0


class TestS7DailySelect:
    def _cand(self, race_key, axis_sum=1.0, entropy=1.0, wt_overlap_n=0, wt_mark3_overlap_n=0):
        return {"race_key": race_key, "axis_sum": axis_sum, "entropy": entropy,
                "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3_overlap_n}

    def test_overlap0_and_overlap1_pass_gates(self):
        cands = [
            self._cand("r1", wt_overlap_n=0),
            self._cand("r2", wt_overlap_n=1),
        ]
        assert {c["race_key"] for c in rank_7s_daily_select(cands)} == {"r1", "r2"}

    def test_overlap2_and_none_excluded(self):
        cands = [
            self._cand("r1", wt_overlap_n=2),
            self._cand("r2", wt_overlap_n=None),
        ]
        assert rank_7s_daily_select(cands) == []

    def test_axis_sum_gate(self):
        cands = [
            self._cand("ok", axis_sum=RANK_7S_AXIS_SUM_MAX),
            self._cand("ng", axis_sum=RANK_7S_AXIS_SUM_MAX + 0.01),
        ]
        assert {c["race_key"] for c in rank_7s_daily_select(cands)} == {"ok"}

    def test_entropy_gate(self):
        cands = [
            self._cand("ok", entropy=RANK_7S_ENTROPY_MAX),
            self._cand("ng", entropy=RANK_7S_ENTROPY_MAX + 0.01),
        ]
        assert {c["race_key"] for c in rank_7s_daily_select(cands)} == {"ok"}

    def test_missing_entropy_fails_safe(self):
        # entropyキー欠損は「常に通過(0.0扱い)」ではなく「常に除外(inf扱い)」。
        # 2026-07-26に旧形式(entropyフィールド無し)の生候補が誤って全通過した
        # 事故の再発防止テスト。
        cands = [{"race_key": "no_entropy", "axis_sum": 0.5, "wt_overlap_n": 0}]
        assert rank_7s_daily_select(cands) == []

    def test_no_count_cap(self):
        # 2026-07-26以前は重なり1候補が件数capで打ち切られていたが、現行は
        # 閾値ゲートを通過した候補は何件でも全件採用される。
        cands = [self._cand(f"r{i}", wt_overlap_n=1) for i in range(50)]
        assert len(rank_7s_daily_select(cands)) == 50

    def test_sorted_by_axis_sum(self):
        cands = [self._cand("b", axis_sum=0.9), self._cand("a", axis_sum=0.5)]
        assert [c["race_key"] for c in rank_7s_daily_select(cands)] == ["a", "b"]

    def test_mark3_overlap_gate_removed(self):
        # 2026-07-31撤廃: mark3ゲートはS7から撤廃された。
        # wt_mark3_overlap_nの値に関わらず、axis_sum/entropy/wt_overlap_nのみで
        # 判定する（旧仕様ではwt_mark3_overlap_n=2は除外していたが、現行は通過する）。
        cands = [
            self._cand("ok0", wt_mark3_overlap_n=0),
            self._cand("ok1", wt_mark3_overlap_n=1),
            self._cand("was_ng2", wt_mark3_overlap_n=2),
        ]
        assert {c["race_key"] for c in rank_7s_daily_select(cands)} == {"ok0", "ok1", "was_ng2"}

    def test_missing_mark3_overlap_no_longer_matters(self):
        # 2026-07-31撤廃: wt_mark3_overlap_nキーが無くても、他のゲートを
        # 満たせば採用される（mark3自体を見なくなったため）。
        cands = [{"race_key": "no_mark3", "axis_sum": 0.5, "entropy": 1.0, "wt_overlap_n": 0}]
        assert {c["race_key"] for c in rank_7s_daily_select(cands)} == {"no_mark3"}


class TestS7EveningReselect:
    def test_merges_day_and_night_under_cap(self):
        day = [{"race_key": "d1", "axis_sum": 0.5, "entropy": 1.0, "wt_overlap_n": 1, "wt_mark3_overlap_n": 0}]
        night = [{"race_key": "n1", "axis_sum": 0.5, "entropy": 1.0, "wt_overlap_n": 1, "wt_mark3_overlap_n": 0}]
        merged = rank_7s_evening_reselect(day, night)
        assert {c["race_key"] for c in merged} == {"d1", "n1"}

    def test_trims_to_daily_cap_by_entropy_ascending(self):
        # 2026-07-26再導入: 日次合計がRANK_7S_DAILY_CAP(=12)を超える場合のみ
        # entropy昇順（最も自信がある順）で上位のみ残す。
        # entropy値は全てゲート(RANK_7S_ENTROPY_MAX=1.8329)以下に収める。
        day = [{"race_key": f"d{i}", "axis_sum": 0.5, "entropy": 0.01 * i, "wt_overlap_n": 1, "wt_mark3_overlap_n": 0}
               for i in range(8)]
        night = [{"race_key": f"n{i}", "axis_sum": 0.5, "entropy": 0.01 * (8 + i), "wt_overlap_n": 1, "wt_mark3_overlap_n": 0}
                 for i in range(8)]
        merged = rank_7s_evening_reselect(day, night)
        assert len(merged) == RANK_7S_DAILY_CAP
        kept = {c["race_key"] for c in merged}
        # entropy 0.00〜0.11（d0..d7, n0..n3）の12件が残り、0.12以降(n4..n7)は落ちる
        assert kept == {f"d{i}" for i in range(8)} | {f"n{i}" for i in range(4)}

    def test_locked_keys_survive_trim(self):
        # 既に買い判定済み(bet_amount>0記録済み)のレースはトリムで除外しない
        day = [{"race_key": f"d{i}", "axis_sum": 0.5, "entropy": 0.01 * i, "wt_overlap_n": 1, "wt_mark3_overlap_n": 0}
               for i in range(8)]
        night = [{"race_key": f"n{i}", "axis_sum": 0.5, "entropy": 0.01 * (100 + i), "wt_overlap_n": 1, "wt_mark3_overlap_n": 0}
                 for i in range(8)]
        # n7はentropyが最も高く通常なら真っ先に落ちるが、ロック済みなら残る
        merged = rank_7s_evening_reselect(day, night, locked_keys={"n7"})
        assert "n7" in {c["race_key"] for c in merged}
        assert len(merged) == RANK_7S_DAILY_CAP

    def test_locked_keys_survive_even_when_gate_would_reject(self):
        # 2026-07-26修正: ロック済み候補はentropy欠損等でゲート自体に落ちても保護される
        # （2026-07-26のデプロイ移行期に実際に起きた「旧形式raw candidatesがentropy
        # フィールド無し」というケースを想定）。
        day = [{"race_key": "locked_but_gate_fails", "axis_sum": 999.0, "wt_overlap_n": 0}]
        merged = rank_7s_evening_reselect(day, [], locked_keys={"locked_but_gate_fails"})
        assert {c["race_key"] for c in merged} == {"locked_but_gate_fails"}


# ── S9（S7の9車立て版・独立ランク・2026-07-26） ─────────────────────────────

from src.strategy_wt import RANK_9S_ENTROPY_MAX, rank_9s_daily_select  # noqa: E402


class TestS9DailySelect:
    def _cand(self, race_key, axis_sum=1.0, entropy=1.0, wt_overlap_n=0, wt_mark3_overlap_n=0):
        return {"race_key": race_key, "axis_sum": axis_sum, "entropy": entropy,
                "wt_overlap_n": wt_overlap_n, "wt_mark3_overlap_n": wt_mark3_overlap_n}

    def test_overlap0_and_overlap1_pass(self):
        cands = [self._cand("r1", wt_overlap_n=0), self._cand("r2", wt_overlap_n=1)]
        assert {c["race_key"] for c in rank_9s_daily_select(cands)} == {"r1", "r2"}

    def test_overlap2_and_none_excluded(self):
        cands = [self._cand("r1", wt_overlap_n=2), self._cand("r2", wt_overlap_n=None)]
        assert rank_9s_daily_select(cands) == []

    def test_entropy_gate(self):
        cands = [
            self._cand("ok", entropy=RANK_9S_ENTROPY_MAX),
            self._cand("ng", entropy=RANK_9S_ENTROPY_MAX + 0.01),
        ]
        assert {c["race_key"] for c in rank_9s_daily_select(cands)} == {"ok"}

    def test_missing_entropy_fails_safe(self):
        cands = [{"race_key": "no_entropy", "axis_sum": 0.5, "wt_overlap_n": 0}]
        assert rank_9s_daily_select(cands) == []

    def test_no_count_cap(self):
        cands = [self._cand(f"r{i}", wt_overlap_n=1) for i in range(50)]
        assert len(rank_9s_daily_select(cands)) == 50

    def test_mark3_overlap_gate(self):
        cands = [
            self._cand("ok0", wt_mark3_overlap_n=0),
            self._cand("ok1", wt_mark3_overlap_n=1),
            self._cand("ng2", wt_mark3_overlap_n=2),
        ]
        assert {c["race_key"] for c in rank_9s_daily_select(cands)} == {"ok0", "ok1"}

    def test_missing_mark3_overlap_fails_safe(self):
        cands = [{"race_key": "no_mark3", "axis_sum": 0.5, "entropy": 1.0, "wt_overlap_n": 0}]
        assert rank_9s_daily_select(cands) == []


class TestS7WtMark3OverlapN:
    def test_both_axis_match_marks(self):
        # axis1=◎(mark1相当) axis2=△(mark3相当) → 2車ともマッチ=2
        assert rank_7s_wt_mark3_overlap_n(1, 3, wt_honmei=1, wt_taikou=2, wt_ana=3) == 2

    def test_one_axis_matches(self):
        assert rank_7s_wt_mark3_overlap_n(1, 9, wt_honmei=1, wt_taikou=2, wt_ana=3) == 1

    def test_no_axis_matches(self):
        assert rank_7s_wt_mark3_overlap_n(8, 9, wt_honmei=1, wt_taikou=2, wt_ana=3) == 0

    def test_missing_any_mark_returns_none(self):
        assert rank_7s_wt_mark3_overlap_n(1, 2, wt_honmei=1, wt_taikou=2, wt_ana=None) is None
        assert rank_7s_wt_mark3_overlap_n(1, 2, wt_honmei=None, wt_taikou=2, wt_ana=3) is None


# ── S1w_gate（win軸1着固定・2026-07-27にentropy条件追加） ───────────────────

from src.strategy_wt import S1W_ENTROPY_MAX, S1W_TOP3_GAP_MIN, s1w_gate  # noqa: E402


class TestS1wGate:
    def test_top3_gap_gate(self):
        assert s1w_gate(S1W_TOP3_GAP_MIN) is True
        assert s1w_gate(S1W_TOP3_GAP_MIN - 0.01) is False

    def test_axis_win_prob_gate_optional(self):
        # axis_win_prob=None（省略時）はこの条件をスキップ
        assert s1w_gate(S1W_TOP3_GAP_MIN, axis_win_prob=None) is True
        assert s1w_gate(S1W_TOP3_GAP_MIN, axis_win_prob=0.9) is False

    def test_axis_player_class_gate_optional(self):
        assert s1w_gate(S1W_TOP3_GAP_MIN, axis_player_class="S1") is False
        assert s1w_gate(S1W_TOP3_GAP_MIN, axis_player_class="A2") is True

    def test_entropy_gate(self):
        # 2026-07-27新設: entropy<=S1W_ENTROPY_MAXを要求。省略時はスキップ。
        assert s1w_gate(S1W_TOP3_GAP_MIN, entropy=None) is True
        assert s1w_gate(S1W_TOP3_GAP_MIN, entropy=S1W_ENTROPY_MAX) is True
        assert s1w_gate(S1W_TOP3_GAP_MIN, entropy=S1W_ENTROPY_MAX + 0.01) is False

    def test_all_gates_combined(self):
        assert s1w_gate(S1W_TOP3_GAP_MIN, axis_win_prob=0.3,
                         axis_player_class="A2", entropy=1.5) is True
        assert s1w_gate(S1W_TOP3_GAP_MIN, axis_win_prob=0.3,
                         axis_player_class="A2", entropy=2.0) is False


def test_rank_7s_axis_sum_max_value():
    """7S の axis_sum 上限を 1.40 に固定する（2026-08-05・ユーザー承認）。

    掃引窓(2025-07〜2026-07)で候補化し、**掃引に一度も使っていない確認窓
    (2024-07〜2025-06)** で閾値を固定したまま一度きり検証して採用した値。
    確認窓4つすべてで ROI>=75% かつ現行(1.50)を上回った（7S: 82.3→84.4%）。

    ⚠️ 7S+7A の合計では +0.5pt しか変わらない（1.40〜1.50 の帯は消えるのではなく
       大半が 7A へ移るため）。この値を動かすときは 7S だけでなく 7A と合計も
       必ず測ること。詳細は strategy_wt.RANK_7S_AXIS_SUM_MAX のコメント参照。
    """
    from src.strategy_wt import RANK_7S_AXIS_SUM_MAX
    assert RANK_7S_AXIS_SUM_MAX == 1.40
