"""7C の三連単切替が採点まで一貫していることを検査する（段階2・2026-08-09）。

## 守る不変条件

7C は単勝率で三連複と三連単を出し分ける（`RANK_7C_TRIFECTA_PW_MIN`）。
採点が三連複固定のままだと:

  - **着順違いを的中に数える**（`frozenset` 比較のため）
  - `payout` に**買っていない三連複の配当**が入る

`picks_history` は全ROI・的中率分析の台帳なので、ここが狂うと分析全体が
静かに壊れる。**壊れても例外は出ない**＝テストでしか気づけない。

券種の伝達経路は2本ある。両方を固定する:

  1. 発走前 decision の `bet_kind`   → `notify_results_wt.py` が採点に使う
  2. `pred_combo` の `三単:` 接頭辞 → `notify_race_result_wt.py` が表示に使う
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.notify_prerace_wt import judge_rank_7c  # noqa: E402


def _trio_board(cars: list[int]) -> dict:
    """全通りの三連複オッズ盤面（欠車なし）。"""
    from itertools import combinations
    return {frozenset(c): 10.0 for c in combinations(cars, 3)}


def _cand(trifecta: bool) -> dict:
    # 軸1=1 / 軸2=2、相手は 3-4-5-6（3着内率15%以上）、7 は足切り。
    probs = {1: 0.95, 2: 0.80, 3: 0.40, 4: 0.35, 5: 0.30, 6: 0.20, 7: 0.05}
    return {
        "axis1": 1, "axis2": 2,
        # 三連複側のゲート（RANK_7C_TRIO_P3_SUM_MIN）で使う。本番の候補JSONは
        # 必ず持つ（`rank_7c_daily_select` が None を落とすため）。
        "p3_sum_top2": probs[1] + probs[2],
        "top3_probs": {str(k): v for k, v in probs.items()},
        "trifecta_7c": trifecta,
    }


def test_trio_case_keeps_unordered_labels() -> None:
    """切替なしのレースは従来どおり順不同ラベル。"""
    decision, detail = judge_rank_7c(_cand(False), _trio_board(list(range(1, 8))))
    assert decision == "buy", detail
    assert detail["bet_kind"] == "trio"
    assert all("=" not in c for c in detail["combos"])
    # 順不同ラベルは車番昇順（"1-2-3"）
    assert "1-2-3" in detail["combos"]


def test_trifecta_case_emits_ordered_labels_and_bet_kind() -> None:
    """切替レースは「軸1-軸2-相手」の順序付きラベルと bet_kind=trifecta。"""
    decision, detail = judge_rank_7c(_cand(True), _trio_board(list(range(1, 8))))
    assert decision == "buy", detail
    assert detail["bet_kind"] == "trifecta"
    for c in detail["combos"]:
        head = c.split("-")
        assert head[0] == "1" and head[1] == "2", f"1着=軸1/2着=軸2 になっていない: {c}"


def test_trifecta_point_count_equals_full_legs() -> None:
    """🔴 三連単は相手を絞らない（＝足切り後の全点数のまま）。

    点数を増やすとガミ境界（＝点数倍）も上がって切替の根拠が消えるが、
    **減らしても**効果が消える（相手2点の三連単は掃引84.4/確認80.9で
    相手全部の82.9/86.2に劣る）。2026-08-09 に三連複側だけ上位2点へ
    絞ったので、三連単と点数が一致しなくなった点に注意。
    """
    board = _trio_board(list(range(1, 8)))
    _, tf = judge_rank_7c(_cand(True), board)
    _, trio = judge_rank_7c(_cand(False), board)
    # このフィクスチャは相手の落差が全て 0.15 未満なので三連複も総流し＝同点数。
    # （差があるレースでは三連複だけが縮む。`test_7c_buy_plan.py` で検査）
    assert len(tf["combos"]) == 4
    assert len(trio["combos"]) == 4
    assert tf["stake"] == trio["stake"] == 2500


def test_trifecta_gate_uses_trio_board_not_trifecta_board() -> None:
    """🔴 三連単の板が空でも見送りにならない。

    三連単の板は三連複より薄い。そちらでゲートすると 7C の 16.9% が
    「オッズ取得できた目が N点」で黙って消える。**買い方が変わっただけで
    母集団まで変わってはいけない。**
    """
    decision, detail = judge_rank_7c(
        _cand(True), _trio_board(list(range(1, 8))), trifecta_lookup={})
    assert decision == "buy", detail
    assert len(detail["combos"]) >= 4
    # 板が無いので表示用オッズは付かないが、買い目は出る
    assert detail["leg_odds"] == {}


def test_scoring_reads_bet_kind_and_uses_ordered_comparison() -> None:
    """採点側が bet_kind を見て着順比較・三連単配当へ切り替えている。

    ⚠️ ソースを読む検査。`notify_results_wt` の 7C ブロックは DB と
       スクレイパに強く依存しており単体で実行できないため、
       分岐が存在することだけを構造的に固定する。
    """
    src = (Path(__file__).parent.parent / "scripts" / "notify_results_wt.py").read_text()
    assert 'c7_is_tf = dec_7c.get("bet_kind") == "trifecta"' in src, \
        "7C 採点が bet_kind を読んでいない"
    assert "c7_win_key = c7_order3 if c7_is_tf else c7_top3" in src, \
        "三連単でも順不同（frozenset）で比較している"
    assert "odds_payout=(c7_trifecta_pay if c7_is_tf else c7_trio_pay)" in src, \
        "三連単なのに三連複の配当を payout に使っている"


def test_result_notify_is_order_sensitive_for_trifecta() -> None:
    """結果通知が `三単:` を着順込みで判定している。"""
    src = (Path(__file__).parent.parent / "scripts"
           / "notify_race_result_wt.py").read_text()
    assert 'is_tf = combo.startswith("三単:")' in src
    assert "order3[0] == head[0] and order3[1] == head[1]" in src, \
        "着順を見ずに的中判定している"
