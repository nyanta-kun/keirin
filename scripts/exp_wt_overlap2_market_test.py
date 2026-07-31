"""wt_overlap_n==2（軸2車がWT公式◎◯と完全一致）レースを5点trio均等買いした場合の
honest全期間検証（2026-07-31）。S7/7Aの軸選定ロジックをそのまま使い、
overlap==2のみ抽出してROI/的中率を算出する。読み取り専用・DB書き込みなし。
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wt_vintage_config import monthly_windows
from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

STAKE_PER_POINT = 2000.0


def main():
    windows = monthly_windows()
    print(f"[main] 月次窓数: {len(windows)}", flush=True)

    total_n = total_hit = 0
    total_bet = total_ret = 0.0
    monthly_roi = []

    for date_from, date_to, eval_model, win_model in windows:
        candidates, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        if not candidates:
            continue
        m_bet = m_ret = 0.0
        m_n = m_hit = 0
        for c_ in candidates:
            if c_.get("wt_overlap_n") != 2:
                continue
            a, b = c_["axis1"], c_["axis2"]
            others = c_["others"]
            trio = c_["trio"]
            combos = [trio[frozenset({a, b, x})] for x in others if frozenset({a, b, x}) in trio]
            if len(combos) < 2:
                continue
            n_points = len(combos)
            bet = STAKE_PER_POINT * n_points
            actual_top3 = c_["actual_top3"]
            hit_odds = trio.get(actual_top3)
            hit = actual_top3 in [frozenset({a, b, x}) for x in others]
            ret = STAKE_PER_POINT * hit_odds if (hit and hit_odds) else 0.0
            m_n += 1
            m_bet += bet
            m_ret += ret
            if ret > 0:
                m_hit += 1
        total_n += m_n
        total_bet += m_bet
        total_ret += m_ret
        total_hit += m_hit
        roi = m_ret / m_bet * 100 if m_bet else 0.0
        monthly_roi.append(roi if m_bet else None)
        print(f"[{date_from}〜{date_to}] n={m_n} 的中{m_hit} ROI={roi:.1f}%", flush=True)

    print("\n" + "=" * 90)
    print("全期間合計（wt_overlap_n==2レース・5点trio均等買い・2,000円/点）")
    print("=" * 90)
    roi = total_ret / total_bet * 100 if total_bet else 0.0
    hitrate = total_hit / total_n * 100 if total_n else 0.0
    vals = [v for v in monthly_roi if v is not None]
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"{total_n}R 的中{total_hit} ({hitrate:.1f}%) 投資{total_bet:,.0f} → 回収{total_ret:,.0f} "
          f"ROI {roi:.1f}%  月次標準偏差={sd:.1f}")


if __name__ == "__main__":
    main()
