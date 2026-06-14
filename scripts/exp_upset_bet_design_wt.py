"""波乱Q4 × 代替買い目設計 — フォーメーション別 ROI 評価（doc28）

目的: 高波乱確率レース（top3_sum Q4=最もloose・upset_tier='Q1_loose'）において
  現行 pivot1-pivot2-3rd 戦略に代わる代替フォーメーションを探索する。

フォーメーション:
  current : 現行 trio3点（pivot1-pivot2-{3rd 3点}）
  F1      : pivot1 × クロスライン軸（pivot2 を異ライン最高確率選手に置換）
  F2      : pivot1 単軸 × rank2-5 から C(4,2)=6点 BOX
  F3      : second_line → pivot1 → 任意 の逆張り（2点）

doc18 セマンティクス厳守:
  ① ランキングは全エントリー（欠車含む）で行う
  ② ≤6車フィルタは出走表基準
  ③ 欠車処理 = 軸欠車→スキップ・相手欠車→その目のみ除外
  ④ モデル = lgbm_wt_eval（TRAIN期間 2023-07〜2025-06 学習、VAL/HOLD は真のOOS）
  ⑤ 払戻 = wt_odds 最終オッズ × 100

upset Q4 = top3_sum が UPSET_TOP3SUM_CUTS[0] 未満（Q1_loose = 最もloose帯 = 上位25%）
線分フィルタ = gap12 ≥ 0.06（本番同様）。最安オッズ ≥ 5.0 倍（ガミ帯回避）。
"""
import sys
import re
import itertools
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _filter_by_n_riders
from src.strategy_wt import UPSET_TOP3SUM_CUTS
from src.database import get_connection
from roi_robustness_wt import roi_summary

# ──────────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────────
TRAIN = ("2023-07-01", "2025-06-30")
VAL   = ("2025-07-01", "2026-02-28")
HOLD  = ("2026-03-01", "2026-06-14")

# upset Q4 = Q1_loose の閾値（top3_sum が これ未満 → 最もloose・波乱余地大）
UPSET_Q4_CUT = UPSET_TOP3SUM_CUTS[0]

# ガミ帯回避: trio最安オッズ ≥ 5.0 倍
MIN_ODDS = 5.0

# gap12 最小値（本番 wave-picks-wt の A 層下限）
MIN_GAP12 = 0.06


# ──────────────────────────────────────────────────────────────────────────────
# オッズ盤面ロード
# ──────────────────────────────────────────────────────────────────────────────
def load_trio_boards(race_keys: list) -> dict:
    """trio の盤面を {race_key: {frozenset: odds_value}} で返す。"""
    trio = defaultdict(dict)
    CHUNK = 900
    with get_connection() as c:
        for i in range(0, len(race_keys), CHUNK):
            chunk = race_keys[i:i + CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = c.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds "
                f"WHERE bet_type='trio' AND race_key IN ({ph})", chunk
            ).fetchall()
            for rk, comb, ov in rows:
                if ov is None or ov <= 0:
                    continue
                try:
                    fr = [int(x) for x in re.split(r"[-=]", str(comb))]
                except ValueError:
                    continue
                if len(fr) == 3:
                    trio[rk][frozenset(fr)] = float(ov)
    return dict(trio)


# ──────────────────────────────────────────────────────────────────────────────
# フォーメーション定義
# ──────────────────────────────────────────────────────────────────────────────
def make_current_combos(frames: list, _line_groups: list) -> list:
    """current: pivot1-pivot2-{3rd 3点}（三連複3点）"""
    p1, p2 = frames[0], frames[1]
    thirds = frames[2:5]
    return [frozenset((p1, p2, t)) for t in thirds]


def make_f1_combos(frames: list, line_groups: list) -> list:
    """F1: pivot1 × クロスライン軸（2軸流し3点）

    pivot2 を「pivot1 と異なる line_group のモデル最高確率選手」に置換。
    line_group が None / 欠損の場合はフォールバック無し（このレースをスキップ）。
    """
    p1 = frames[0]
    lg1 = line_groups[0]
    if lg1 is None:
        return []
    # pivot1 以外の選手から line_group が異なる最高確率の選手（既にprob降順でソート済）
    cross_pivot = None
    for f, lg in zip(frames[1:], line_groups[1:]):
        if lg is not None and lg != lg1:
            cross_pivot = f
            break
    if cross_pivot is None:
        return []
    # 3rd: rank2-5 の中から p1, cross_pivot 以外を最大3人
    thirds = [f for f in frames[1:5] if f not in (p1, cross_pivot)][:3]
    if not thirds:
        return []
    return [frozenset((p1, cross_pivot, t)) for t in thirds]


def make_f2_combos(frames: list, _line_groups: list) -> list:
    """F2: pivot1 単軸 × rank2-5 から C(4,2)=6点 BOX（trio形式）"""
    p1 = frames[0]
    candidates = frames[1:5]
    if len(candidates) < 2:
        return []
    return [frozenset((p1, a, b))
            for a, b in itertools.combinations(candidates, 2)]


def make_f3_combos(frames: list, line_groups: list) -> list:
    """F3: 逆張り second_line 軸（2点）

    second_line: pivot1 と異なる line_group のうち pred_prob 最大の選手
    third_line : 上位2ラインに属さない選手で最大 pred_prob
    購入: frozenset(second_line, pivot1, third_line) × 2 = second_line軸に他2人流し

    実装: (second_line, pivot1) 軸に rank3以降から third_line 候補を3つ流す（2点で上限）
    """
    p1 = frames[0]
    lg1 = line_groups[0]
    # second_line = pivot1 異ラインの最高確率選手
    second_line = None
    second_line_group = None
    for f, lg in zip(frames[1:], line_groups[1:]):
        if lg is not None and lg != lg1:
            second_line = f
            second_line_group = lg
            break
    if second_line is None:
        return []
    # 3rd候補: pivot1 / second_line 以外（最大2人）
    thirds = [f for f, lg in zip(frames, line_groups)
              if f not in (p1, second_line)][:2]
    if not thirds:
        return []
    # second_line → pivot1 → third の三連複
    return [frozenset((second_line, p1, t)) for t in thirds]


FORMATIONS = [
    ("current", make_current_combos),
    ("F1",      make_f1_combos),
    ("F2",      make_f2_combos),
    ("F3",      make_f3_combos),
]


# ──────────────────────────────────────────────────────────────────────────────
# データ収集
# ──────────────────────────────────────────────────────────────────────────────
def collect(date_from: str, date_to: str, model) -> list[dict]:
    """対象期間のレースデータを収集してレース単位の構造体リストを返す。

    doc18 セマンティクス厳守:
      - 出走表基準で≤6車フィルタ（_filter_by_n_riders を使用）
      - pred_prob は全エントリーで計算（_apply_pred_prob_wt相当: prepare_X + model.predict_proba）
      - ランキングは全エントリー（欠車含む）で行う
      - wt_entries.line_group を使用
    """
    print(f"  loading {date_from}〜{date_to}...", flush=True)
    df_raw = load_raw_data_wt(min_date=date_from, max_date=date_to)
    df = build_features_wt(df_raw)

    # ② 出走表基準で≤6車フィルタ
    df = _filter_by_n_riders(df, 6)
    if df.empty:
        return []

    # ① pred_prob を全エントリー（欠車含む）で計算
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    # line_group を wt_entries から取得（feature_wt が load_raw_data_wt に含む）
    # load_raw_data_wt は wt_entries.line_group を含む
    race_keys = df["race_key"].unique().tolist()
    trio_boards = load_trio_boards(race_keys)

    races = []
    for rk, grp in df.groupby("race_key"):
        # 結果確定チェック（3着内に3選手いること）
        fin = grp[grp["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue

        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        bd = trio_boards.get(rk, {})
        if not bd:
            continue

        # ① 全エントリーでランキング（欠車=finish_order=0 を含む）
        grp = grp.sort_values("pred_prob", ascending=False)
        n = len(grp)
        if n < 3:
            continue

        probs = grp["pred_prob"].tolist()
        frames = grp["frame_no"].astype(int).tolist()
        line_groups = grp["line_group"].tolist()

        gap12 = probs[0] - probs[1]
        top3_sum = probs[0] + probs[1] + probs[2]

        # フィルタ: gap12 ≥ MIN_GAP12
        if gap12 < MIN_GAP12:
            continue

        # 欠車セット（finish_order=0）
        dns = set(grp[grp["finish_order"] == 0]["frame_no"].astype(int).tolist())

        races.append({
            "race_key": rk,
            "date": grp["race_date"].iloc[0],
            "n": n,
            "gap12": gap12,
            "top3_sum": top3_sum,
            "is_upset_q4": top3_sum < UPSET_Q4_CUT,
            "frames": frames,
            "line_groups": line_groups,
            "top3": top3,
            "dns": dns,
            "bd": bd,
        })

    print(f"    {len(races):,} races (gap12≥{MIN_GAP12})", flush=True)
    return races


# ──────────────────────────────────────────────────────────────────────────────
# ROI 計算
# ──────────────────────────────────────────────────────────────────────────────
def calc_roi(races: list, formation_fn) -> tuple[dict, int]:
    """フォーメーション関数を受け取り (roi_summary結果, レース数) を返す。

    ガミ帯回避: コンボ最安オッズ ≥ MIN_ODDS でなければスキップ。
    欠車処理: DNS 車を含むコンボはスキップ。全コンボがスキップ → このレースは不計上。
    """
    pays, bets = [], []
    n_races = 0

    for r in races:
        combos = formation_fn(r["frames"], r["line_groups"])
        if not combos:
            continue

        # ③ DNS 車を含むコンボはスキップ
        valid = [c for c in combos if not c.intersection(r["dns"])]
        if not valid:
            continue

        # trio オッズが存在するコンボのみ（オッズなし=未発売=スキップ）
        valid = [(c, r["bd"].get(c)) for c in valid if c in r["bd"]]
        if not valid:
            continue

        # ガミ帯回避: 全コンボのうち最安オッズ ≥ MIN_ODDS
        min_odds = min(ov for _, ov in valid)
        if min_odds < MIN_ODDS:
            continue

        n_races += 1
        total_bet = len(valid) * 100  # 1コンボ100円

        # 的中判定
        total_pay = 0
        for c, ov in valid:
            if c == r["top3"]:
                total_pay = int(round(ov * 100))
                break

        pays.append(float(total_pay))
        bets.append(float(total_bet))

    return roi_summary(pays, bets), n_races


# ──────────────────────────────────────────────────────────────────────────────
# 出力
# ──────────────────────────────────────────────────────────────────────────────
def fmt_cell(s: dict, n: int) -> str:
    if n == 0:
        return f"   0R   ---         ---       "
    ci_note = f"[{s['ci_lo']:>5.0%},{s['ci_hi']:>6.0%}]"
    return (f"{n:>4}R {s['roi']:>6.0%} {ci_note} ex:{s['roi_ex_max']:>5.0%} "
            f"hit:{s['hit_rate']:>4.0%}")


def print_table(results: dict):
    """
    results = {
        (formation_name, period_name, filter_label): (roi_summary_dict, n)
    }
    """
    formations = [f for f, _ in FORMATIONS]
    periods = ["TRAIN", "VAL", "HOLD"]
    filters = ["ALL", "Q4(upset)"]

    W_NAME = 10
    W_CELL = 53

    print(f"\n{'='*130}")
    print("  波乱Q4 × 代替買い目設計  ROI 比較 (doc18: eval model / 最終オッズ上限値)")
    print(f"  upset Q4 閾値: top3_sum < {UPSET_Q4_CUT:.4f}  |  MIN_ODDS≥{MIN_ODDS}  |  gap12≥{MIN_GAP12}")
    print(f"{'='*130}")
    for filt in filters:
        print(f"\n  ■ フィルタ: {filt}")
        header = f"  {'フォーメーション':<{W_NAME}}"
        for per in periods:
            header += f"  {per:<{W_CELL}}"
        print(header)
        print(f"  {'-'*128}")
        for fname in formations:
            row = f"  {fname:<{W_NAME}}"
            for per in periods:
                s, n = results[(fname, per, filt)]
                row += f"  {fmt_cell(s, n)}"
            print(row)
        print(f"  {'-'*128}")

    print(f"\n{'='*130}")
    print("  ROI = Σ払戻/Σ投資  |  CI = bootstrap 95%CI  |  ex = 最大払戻1件除去  |  hit = 的中率")
    print("  ※ 合格基準: VAL・HOLD 両方で CI 下限 > 100%")


def print_avg_pts(avg_pts: dict):
    """平均点数の出力"""
    print(f"\n  平均購入点数（参考）")
    print(f"  {'フォーメーション':<10}  {'ALL':<8}  {'Q4(upset)':<8}")
    for fname, _ in FORMATIONS:
        a = avg_pts.get((fname, "ALL"), 0)
        q = avg_pts.get((fname, "Q4(upset)"), 0)
        print(f"  {fname:<10}  {a:.1f}点  {q:.1f}点")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  W02: 波乱Q4 × 代替買い目設計  (doc28)")
    print(f"  モデル: lgbm_wt_eval (TRAIN 2023-07〜2025-06 学習)")
    print(f"  doc18 セマンティクス: ランキング全エントリー / 出走表≤6車 / 欠車③処理")
    print("=" * 80)

    model = load_model("lgbm_wt_eval")
    print("lgbm_wt_eval ロード完了")

    period_defs = {
        "TRAIN": TRAIN,
        "VAL":   VAL,
        "HOLD":  HOLD,
    }

    results = {}
    avg_pts_all = {}

    for per_name, (d_from, d_to) in period_defs.items():
        print(f"\n--- {per_name} ({d_from}〜{d_to}) ---")
        races_all = collect(d_from, d_to, model)
        races_q4  = [r for r in races_all if r["is_upset_q4"]]
        print(f"  Q4(upset) レース数: {len(races_q4):,} / {len(races_all):,} "
              f"({len(races_q4)/len(races_all)*100:.1f}%)" if races_all else "  データなし")

        for fname, ffn in FORMATIONS:
            for filt_label, races in [("ALL", races_all), ("Q4(upset)", races_q4)]:
                s, n = calc_roi(races, ffn)
                results[(fname, per_name, filt_label)] = (s, n)

        # 平均点数（ALL）
        for fname, ffn in FORMATIONS:
            total_pts = 0
            n_races_used = 0
            for r in races_all:
                combos = ffn(r["frames"], r["line_groups"])
                if combos:
                    valid = [c for c in combos if not c.intersection(r["dns"]) and c in r["bd"]]
                    if valid:
                        min_ov = min(r["bd"][c] for c in valid)
                        if min_ov >= MIN_ODDS:
                            total_pts += len(valid)
                            n_races_used += 1
            avg_pts_all[(fname, "ALL")] = total_pts / n_races_used if n_races_used else 0.0
            # Q4
            total_pts_q4 = 0
            n_races_q4_used = 0
            for r in races_q4:
                combos = ffn(r["frames"], r["line_groups"])
                if combos:
                    valid = [c for c in combos if not c.intersection(r["dns"]) and c in r["bd"]]
                    if valid:
                        min_ov = min(r["bd"][c] for c in valid)
                        if min_ov >= MIN_ODDS:
                            total_pts_q4 += len(valid)
                            n_races_q4_used += 1
            avg_pts_all[(fname, "Q4(upset)")] = (total_pts_q4 / n_races_q4_used
                                                  if n_races_q4_used else 0.0)

    # 結果表示
    print_table(results)
    print_avg_pts(avg_pts_all)

    # 合格判定
    print(f"\n  合格判定 (VAL & HOLD の CI 下限 > 100%):")
    for fname, _ in FORMATIONS:
        for filt in ["ALL", "Q4(upset)"]:
            val_s, val_n = results[(fname, "VAL", filt)]
            hld_s, hld_n = results[(fname, "HOLD", filt)]
            val_ok = val_n >= 10 and val_s["ci_lo"] > 1.0
            hld_ok = hld_n >= 10 and hld_s["ci_lo"] > 1.0
            both_ok = val_ok and hld_ok
            marker = "★PASS" if both_ok else ("VAL○" if val_ok else ("HOLD○" if hld_ok else "×"))
            print(f"  {fname:<10} {filt:<12}: {marker}"
                  f"  VAL ROI={val_s['roi']:.0%}[CI_lo={val_s['ci_lo']:.0%}] n={val_n}"
                  f"  HOLD ROI={hld_s['roi']:.0%}[CI_lo={hld_s['ci_lo']:.0%}] n={hld_n}")

    print(f"\n{'='*80}")
    print("  ※ 最終オッズ上限値 = 実運用上限（朝→確定ドリフトで実測は下振れ）")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
