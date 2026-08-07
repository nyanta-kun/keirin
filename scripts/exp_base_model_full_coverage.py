"""全レース対象「ベースモデル」の設計掃引（2026-08-07・探索）。

ユーザー依頼:
  「1日の対象レース数が少なく的中体験が乏しい。終日のレースを対象にできそうな
    ベースモデルを1つ構築したい。現行モデルとの重なりは気にしない」

前提データ: `data/exp_cache/axis_detail_7car.pkl`（48,541R・2024-07〜2026-08・
honest walk-forward 予測）。1レースあたり以下を持つ:
  p3: モデル3着内確率 / pw: モデル1着確率 / pb: モデル着外(bad)確率
  board: 三連複35組の**最終**オッズ / top3: 実際の3着内集合 / line / mk(公式印)

評価軸は「的中率（＝的中体験）」を主・ROI を従とする。
控除率75%の壁は既に実証済み（keirin_clean_baseline_market_efficiency_2026_07_30）
なので黒字は目標にしない。**同じ ROI 帯でどれだけ的中率を上げられるか**を測る。

⚠️ board は最終オッズ。市場情報を「入力」に使う案は本来 朝オッズでしか実装
   できないため、ここでの市場系スコアは**上振れした上限値**として読むこと。
   朝→最終のドリフト影響は exp_base_model_morning_drift.py で別途測る。
⚠️ 掃引窓 2025-07-01〜 / 確認窓 2024-07-01〜2025-06-30 で必ず分ける。
⚠️ DB 書き込みなし（読み取りのみ）。
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.strategy_wt import RANK_AXIS2_BAD_WEIGHT, _race_zscore  # noqa: E402

DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
CONFIRM_END = "2025-06-30"
STAKE = 100  # 円/点


# --------------------------------------------------------------------------
# 確率の下ごしらえ
# --------------------------------------------------------------------------
def market_probs(board: dict) -> tuple[dict, dict]:
    """三連複オッズ board から (組の市場確率, 車の市場3着内確率) を返す。

    1/odds を全35組で正規化して控除率を落とす。組の集合は「3着内の集合」その
    ものなので、ある車を含む組の確率和がその車の3着内確率になる。
    """
    raw = {c: 1.0 / o for c, o in board.items() if o and o > 0}
    tot = sum(raw.values())
    if tot <= 0:
        return {}, {}
    qc = {c: v / tot for c, v in raw.items()}
    qk: dict[int, float] = {}
    for c, v in qc.items():
        for k in c:
            qk[k] = qk.get(k, 0.0) + v
    return qc, qk


def model_combo_probs(p3: dict[int, float], cars: list[int]) -> dict:
    """車ごとの3着内確率から組の確率をナイーブ積で作り正規化する。

    相関を無視した粗い近似（memory: 組単位LGBでも市場に届かないと実証済み）。
    ここでは「市場を使わない場合の組レベル順位」の代表として使うだけ。
    """
    out = {}
    for c in itertools.combinations(sorted(cars), 3):
        out[frozenset(c)] = p3[c[0]] * p3[c[1]] * p3[c[2]]
    tot = sum(out.values())
    return {c: v / tot for c, v in out.items()} if tot > 0 else out


# --------------------------------------------------------------------------
# 軸・買い目の構成
# --------------------------------------------------------------------------
def axis_three_head(r) -> tuple[int, int]:
    """現行の3ヘッド軸（本番 rank_7s_select_axis と同一の式）。"""
    a1 = max(r["pw"], key=lambda k: r["pw"][k])
    zp, zb = _race_zscore(r["p3"]), _race_zscore(r["pb"])
    sc = {k: zp[k] - RANK_AXIS2_BAD_WEIGHT * zb[k] for k in r["p3"]}
    a2 = max((k for k in sc if k != a1), key=lambda k: sc[k])
    return a1, a2


def order_by(score: dict[int, float]) -> list[int]:
    return sorted(score, key=lambda k: -score[k])


def legs_axis2(a1: int, a2: int, cars: list[int]) -> list[frozenset]:
    return [frozenset({a1, a2, x}) for x in cars if x not in (a1, a2)]


def legs_box(order: list[int], n: int) -> list[frozenset]:
    return [frozenset(c) for c in itertools.combinations(order[:n], 3)]


def legs_axis1_pairs(a1: int, order: list[int], n: int) -> list[frozenset]:
    """軸1を必ず含め、残り上位 n 車から2車。点数 = C(n,2)。"""
    rest = [x for x in order if x != a1][:n]
    return [frozenset({a1, x, y}) for x, y in itertools.combinations(rest, 2)]


def legs_topk(combo_score: dict, k: int) -> list[frozenset]:
    return sorted(combo_score, key=lambda c: -combo_score[c])[:k]


# --------------------------------------------------------------------------
# 掃引本体
# --------------------------------------------------------------------------
def build(races) -> pd.DataFrame:
    rows = []
    for r in races:
        board = r["board"]
        if len(board) < 35:
            continue  # 盤面欠け（欠車など）は基準線から外す
        cars = sorted(r["p3"])
        if len(cars) != 7:
            continue
        top3 = frozenset(r["top3"])
        if top3 not in board:
            continue

        qc, qk = market_probs(board)
        if not qk:
            continue
        mc = model_combo_probs(r["p3"], cars)

        a1, a2 = axis_three_head(r)
        # 市場だけで選ぶ軸（3着内確率の上位2車）
        m_order = order_by(qk)
        # モデル×市場のブレンド（レース内 z の平均。重み 0.5 固定＝素直な折半）
        zp, zq = _race_zscore(r["p3"]), _race_zscore(qk)
        bl = {k: 0.5 * zp[k] + 0.5 * zq[k] for k in cars}
        b_order = order_by(bl)
        p_order = order_by(r["p3"])

        # ブレンド確率での組スコア（naive積の z ブレンド版）
        bc = {c: qc[c] ** 0.5 * mc[c] ** 0.5 for c in board}

        variants = {
            # --- 現行（比較基準） ---
            "A0_3head_ax2_5": legs_axis2(a1, a2, cars),
            # --- 軸の選び方だけ変える（点数は5点で固定） ---
            "A1_p3top2_ax2_5": legs_axis2(p_order[0], p_order[1], cars),
            "A2_market_ax2_5": legs_axis2(m_order[0], m_order[1], cars),
            "A3_blend_ax2_5": legs_axis2(b_order[0], b_order[1], cars),
            # --- 点数を増やす（軸は現行3ヘッド） ---
            "B1_3head_ax1pairs_6": legs_axis1_pairs(a1, order_by(r["p3"]), 4),
            "B2_p3box4": legs_box(p_order, 4),
            "B3_p3box5": legs_box(p_order, 5),
            "B4_blendbox4": legs_box(b_order, 4),
            "B5_blendbox5": legs_box(b_order, 5),
            # --- 組レベルで直接選ぶ ---
            "C1_model_top5": legs_topk(mc, 5),
            "C2_market_top5": legs_topk(qc, 5),
            "C3_blend_top5": legs_topk(bc, 5),
            "C4_blend_top8": legs_topk(bc, 8),
            "C5_blend_top10": legs_topk(bc, 10),
            "C6_market_top10": legs_topk(qc, 10),
        }

        base = dict(rk=r["rk"], date=r["date"])
        for name, legs in variants.items():
            legs = [c for c in dict.fromkeys(legs) if c in board]
            if not legs:
                continue
            hit = top3 in legs
            odds = board[top3]
            rows.append(dict(
                **base, variant=name, pts=len(legs), hit=int(hit),
                bet=len(legs) * STAKE,
                ret=(round(odds * 100) // 10 * 10) if hit else 0,
                odds=odds if hit else np.nan,
            ))
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (v, w), g in df.groupby(["variant", "win"], sort=False):
        out.append(dict(
            variant=v, win=w, n=len(g), pts=g.pts.mean(),
            hit=100 * g.hit.mean(),
            roi=100 * g.ret.sum() / g.bet.sum(),
            avg_odds=float(g.odds.mean()) if g.hit.sum() else 0.0,
        ))
    return pd.DataFrame(out)


def main() -> None:
    races = pickle.load(open(DETAIL, "rb"))
    print(f"読み込み: {len(races):,}R")
    df = build(races)
    df["win"] = np.where(df.date <= CONFIRM_END, "確認", "掃引")
    a = agg(df)

    piv = a.pivot(index="variant", columns="win",
                  values=["n", "pts", "hit", "roi", "avg_odds"])
    order = ["A0_3head_ax2_5", "A1_p3top2_ax2_5", "A2_market_ax2_5", "A3_blend_ax2_5",
             "B1_3head_ax1pairs_6", "B2_p3box4", "B3_p3box5", "B4_blendbox4",
             "B5_blendbox5", "C1_model_top5", "C2_market_top5", "C3_blend_top5",
             "C4_blend_top8", "C5_blend_top10", "C6_market_top10"]
    print("\n=== 全7車レース対象（無ゲート）・変種別 ===")
    print(f"{'variant':22s} {'点':>4s} | "
          f"{'掃引n':>7s} {'的中%':>6s} {'ROI%':>6s} {'平均配当':>7s} | "
          f"{'確認n':>7s} {'的中%':>6s} {'ROI%':>6s} {'平均配当':>7s}")
    for v in order:
        if v not in piv.index:
            continue
        row = piv.loc[v]
        try:
            print(f"{v:22s} {row[('pts','掃引')]:4.0f} | "
                  f"{row[('n','掃引')]:7,.0f} {row[('hit','掃引')]:6.1f} "
                  f"{row[('roi','掃引')]:6.1f} {row[('avg_odds','掃引')]:7.1f} | "
                  f"{row[('n','確認')]:7,.0f} {row[('hit','確認')]:6.1f} "
                  f"{row[('roi','確認')]:6.1f} {row[('avg_odds','確認')]:7.1f}")
        except KeyError:
            pass

    days_s = df[df.win == "掃引"].date.nunique()
    days_c = df[df.win == "確認"].date.nunique()
    n_s = df[(df.win == "掃引") & (df.variant == "A0_3head_ax2_5")].shape[0]
    n_c = df[(df.win == "確認") & (df.variant == "A0_3head_ax2_5")].shape[0]
    print(f"\n対象レース: 掃引 {n_s:,}R / {days_s}日 = {n_s/days_s:.1f}件/日, "
          f"確認 {n_c:,}R / {days_c}日 = {n_c/days_c:.1f}件/日")

    df.to_pickle(REPO / "data" / "exp_cache" / "base_model_variants.pkl")
    print("→ data/exp_cache/base_model_variants.pkl に保存")


if __name__ == "__main__":
    main()
