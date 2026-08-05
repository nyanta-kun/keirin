"""軸2の予測確率が低い帯を「除外」するか「別ランク化」するかの検証（2026-08-04）。

ユーザー提案:
  「下限設定、もしくは7SSなど別ランクに設定し、大当たりが引けた場合に
    予想家としての引きにつながるように推奨するか（当てられるか）、を検討します」

直近レビュー（exp_recent_opportunity_review.py・8/1〜8/4）で、外した推奨32件のうち
軸2の p3 が 39.5 / 45.3 / 49.7 / 51.5 と低いものが目立ち、しかもそれらのレースは
264.6倍・159.5倍・149.7倍と大荒れだった。つまり
  「軸2が低確率のレース = 当てにくいが、当たれば大きい」
という帯が存在する可能性がある。これを

  (a) 下限設定して**除外**する（的中率を上げる。件数は減る）
  (b) 別ランクとして**残す**（的中率は低いが大当たりの引きを狙う）

のどちらにすべきかを判断するため、軸2の p2 帯別に
**ROIではなく「高配当的中の頻度」を主指標**として測る。

⚠️ 7SS 全廃の経緯（2026-08-02）を踏まえること: live n=16,298・ROI 73.5%、
   2026年の月次も1月以外すべて70%以下で控除率75%を下回り続けたため廃止された。
   別ランク化しても **ROIは控除率の壁を超えない**前提で、
   「月に何回 20倍/50倍/100倍を当てられるか」で価値を測る。

⚠️ 使用するキャッシュは48特徴時代の vintage モデルで生成したもの
   （60特徴での再学習は実行中）。傾向を見る目的では十分だが、
   採用判定は60特徴で作り直してから行うこと。

DB書き込みなし。

使い方:
    python scripts/exp_axis2_low_prob_rank.py data/exp_7c_cache
"""
from __future__ import annotations

import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX

STAKE = 100


def load(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def score(cands: list[dict], k: int = 5) -> dict:
    """三連複 軸2車 + 相手上位k車（k=5 は総流し）。"""
    n = hit = bet = ret = 0
    pays: list[int] = []
    days = set()
    by_month: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in cands:
        legs = [x for x in sorted(c["others"], key=lambda x: -c["top3_probs"][x])[:k]
                if c["trio_legs"].get(x)]
        if not legs:
            continue
        n += 1
        days.add(c["race_date"])
        ym = str(c["race_date"])[:7]
        stake = len(legs) * STAKE
        bet += stake
        by_month[ym][0] += stake
        top3 = set(c["actual_top3"])
        rest = top3 - {c["axis1"], c["axis2"]}
        if len(top3 & {c["axis1"], c["axis2"]}) == 2 and len(rest) == 1 and rest.pop() in legs:
            hit += 1
            got = c["trio_pay"] * STAKE // 100
            ret += got
            by_month[ym][1] += got
            pays.append(got)
    ndays = len(days) or 1
    mrois = [100.0 * r / b for b, r in by_month.values() if b > 0]
    return {
        "n": n, "per_day": n / ndays, "days": ndays,
        "hit": 100.0 * hit / n if n else 0.0,
        "roi": 100.0 * ret / bet if bet else 0.0,
        "med": statistics.median(pays) / 100 if pays else 0.0,
        "n20": sum(1 for p in pays if p >= 2000),
        "n50": sum(1 for p in pays if p >= 5000),
        "n100": sum(1 for p in pays if p >= 10000),
        "max": max(pays) / 100 if pays else 0.0,
        "m_sd": statistics.pstdev(mrois) if len(mrois) > 1 else 0.0,
    }


HDR = (f"{'区分':22} {'n':>6} {'件/日':>6} {'的中':>6} {'ROI':>7} {'中央値':>7} "
       f"{'20倍超':>7} {'50倍超':>7} {'100倍超':>8} {'最高':>8}")


def row(lbl: str, s: dict) -> str:
    d = s["days"]
    return (f"{lbl:22} {s['n']:6d} {s['per_day']:6.2f} {s['hit']:5.1f}% "
            f"{s['roi']:6.1f}% {s['med']:6.1f}倍 "
            f"{s['n20']:4d}({d/max(s['n20'],1):4.0f}日に1) "
            f"{s['n50']:3d}({d/max(s['n50'],1):4.0f}日に1) "
            f"{s['n100']:3d}({d/max(s['n100'],1):5.0f}日に1) {s['max']:7.1f}倍")


def main() -> None:
    rows = load(Path(sys.argv[1]))
    days = sorted({c["race_date"] for c in rows})
    ov01 = [c for c in rows if c["wt_overlap_n"] in (0, 1)]
    # 現行 7S+7A 相当（2ゲートのうち不合格が1個以下）
    pool = [c for c in ov01
            if ((c["axis_sum"] > RANK_7S_AXIS_SUM_MAX)
                + (c["entropy"] > RANK_7S_ENTROPY_MAX)) <= 1]
    print(f"母集団: 現行7S+7A相当 {len(pool)}件 / {len(days)}日 "
          f"({days[0]}〜{days[-1]})\n")

    p2 = {id(c): c["top3_probs"][c["axis2"]] for c in pool}

    print("【① 軸2の予測確率 p2 帯別】（現行はこの全帯を推奨している）")
    print(HDR)
    edges = [(0.0, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
             (0.65, 0.70), (0.70, 1.01)]
    for lo, hi in edges:
        sub = [c for c in pool if lo <= p2[id(c)] < hi]
        if sub:
            print(row(f"p2 {lo:.2f}〜{hi:.2f}", score(sub)))
    print(row("全体（現行）", score(pool)))
    print()

    print("【② (a) 下限設定: p2 が閾値未満を除外したときの残り】")
    print(HDR)
    for th in (0.50, 0.55, 0.60, 0.65):
        sub = [c for c in pool if p2[id(c)] >= th]
        print(row(f"p2 >= {th:.2f} を推奨", score(sub)))
    print()

    print("【③ (b) 別ランク化: p2 が閾値未満だけを取り出したとき】")
    print("  ※ 評価軸はROIではなく「大当たりの引き」。7SS全廃の経緯から")
    print("     ROIが控除率75%を超えないことは織り込む。")
    print(HDR)
    for th in (0.50, 0.55, 0.60, 0.65):
        sub = [c for c in pool if p2[id(c)] < th]
        if sub:
            print(row(f"p2 < {th:.2f} を別ランク", score(sub)))
    print()

    print("【④ 別ランク候補に相手を絞った場合（点数を減らして配当を残す）】")
    print(HDR)
    low = [c for c in pool if p2[id(c)] < 0.60]
    for k in (5, 4, 3, 2):
        print(row(f"p2<0.60 相手{k}点", score(low, k)))
    print()

    print("【⑤ 参考: 廃止済み 7SS の実績（CLAUDE.md 記載値）】")
    print("  live n=16,298 / ROI 73.5% / 2026年月次 94.4-61.0-56.3-61.1-69.3-70.2-60.3%")
    print("  → 控除率75%を下回り続けたため 2026-08-02 に全廃。")
    print("     別ランク化はこの前例を踏むリスクがあるため、")
    print("     『月次ROIの安定性』ではなく『高配当の引ける頻度』で価値を判断すること。")


if __name__ == "__main__":
    main()
