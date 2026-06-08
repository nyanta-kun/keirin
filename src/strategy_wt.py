"""winticket 波乱/非本命ゲート（確定前情報のみ・朝7:00算出可）

3タスク（特徴ablation・波乱予測・オッズ活用）が収束した結論:
「本命が堅いレースは低ROI、本命が割れた=波乱余地のあるレースが高ROI」。

指標 top3_sum = 上位3頭の pred_prob(=P(top3)) 合計。
  小さい = 上位3頭に確率が集中していない = レースが割れている = 波乱余地大。
  大きい = 鉄板 = 低配当。

検証（lgbm_wt・本番3点戦略・TRAIN 2023-07〜2026-02 → TEST 2026-03〜, OOS）:
  TRAIN四分位カット = [1.70, 1.90, 2.08]
  Q1_loose(top3_sum<1.70): TRAIN ROI 1224% / TEST ROI 1136%（最大払戻除外でも934%）
  Q4_chalk(>2.08):         TRAIN ROI  88% / TEST ROI  107%
  単調・train/test一致・volume十分(test 125R=25%)・万車券単発非依存。

注意: ROIは最終データbacktest=実運用上限値（実測は別途 picks_history で検証）。
ゲートは「本命堅レースを見送り、波乱余地レースに絞る」フィルターとして使う。
"""
from __future__ import annotations

import json
from pathlib import Path

# TRAIN(2023-07-01〜2026-02-28) の top3_sum 四分位カット（既定値＝コミット済フォールバック）。
# 再学習でモデル確率分布が変わると四分位がズレるため、週次再学習後に
# scripts/recompute_upset_cuts_wt.py が data/models/upset_cuts_wt.json を更新し、
# 下記 _load_cuts() がそれを優先採用する（無ければこの既定値）。
UPSET_TOP3SUM_CUTS_DEFAULT = (1.70, 1.90, 2.08)
UPSET_TIERS = ("Q1_loose", "Q2", "Q3", "Q4_chalk")

_CUTS_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "upset_cuts_wt.json"


def _load_cuts() -> tuple[float, float, float]:
    """再計測済みカット(JSON)を読む。無効/不在なら既定値。"""
    try:
        d = json.loads(_CUTS_PATH.read_text(encoding="utf-8"))
        c = d.get("cuts")
        if isinstance(c, (list, tuple)) and len(c) == 3:
            cuts = tuple(float(x) for x in c)
            if cuts[0] < cuts[1] < cuts[2]:   # 単調性チェック
                return cuts  # type: ignore[return-value]
    except Exception:
        pass
    return UPSET_TOP3SUM_CUTS_DEFAULT


# 実効カット（プロセス起動時に確定。日次cronは毎回新プロセスなので最新を反映）
UPSET_TOP3SUM_CUTS = _load_cuts()


def upset_tier(top3_sum: float) -> str:
    """top3_sum を TRAIN 四分位カットで Q1_loose〜Q4_chalk に割り当てる。"""
    c1, c2, c3 = UPSET_TOP3SUM_CUTS
    if top3_sum < c1:
        return "Q1_loose"
    if top3_sum < c2:
        return "Q2"
    if top3_sum < c3:
        return "Q3"
    return "Q4_chalk"


def race_signals(probs_desc: list[float], n_riders: int) -> dict:
    """pred_prob 降順リストから確定前シグナルを計算する。

    probs_desc: そのレースの pred_prob を降順に並べたリスト
    n_riders:   出走車数
    """
    p1 = probs_desc[0] if probs_desc else 0.0
    p2 = probs_desc[1] if len(probs_desc) >= 2 else 0.0
    p3 = probs_desc[2] if len(probs_desc) >= 3 else 0.0
    top3_sum = p1 + p2 + p3
    return {
        "gap12": p1 - p2,
        "ratio": p1 / (3.0 / n_riders) if n_riders else 0.0,
        "top2_sum": p1 + p2,
        "top3_sum": top3_sum,
        "upset_tier": upset_tier(top3_sum),
    }


def passes_upset_gate(top3_sum: float, max_tier: str = "Q1_loose") -> bool:
    """ゲート通過判定。max_tier までの帯（loose側）のみ通す。

    max_tier='Q1_loose' なら最もlooseな四分位のみ、'Q2' なら Q1+Q2 を通す。
    """
    order = {t: i for i, t in enumerate(UPSET_TIERS)}
    return order[upset_tier(top3_sum)] <= order[max_tier]
