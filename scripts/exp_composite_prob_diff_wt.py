"""Phase B follow-up: M(S3)ゲートの複合シグナル探索（正規プロトコル）。

[[keirin_phase_b_win_model]] の「次のステップ（未着手）」課題:
  「win_rank以外の複合特徴（例: 1着モデルと3着内モデルの確率差）も未探索」

現行M(S3)ゲート（m_axis_gate）は gap12>=0.10 OR win_rank>=3（win_rankは
1着モデル内の順位という離散量）。本スクリプトは win_rank を連続量に
置き換えた版を試す:
  diff = p_top3[axis] - p_win[axis]
    （3着内モデルは自信があるが1着モデルは自信がない度合い。
      win_rankが大きい≒diffが大きい傾向のはずだが、離散化で失われる
      情報がないか確認する）
  ratio = p_win[axis] / p_top3[axis]
    （0に近いほど「勝ちきれない」）

全て「不一致（WT◎≠システム◎）」を前提条件とし（現行M仕様と同一）、
軸=3着内モデル1位・相方=同ライン「逃」・三連複・目>=15倍のみ（現行と同一設定）。

正規プロトコル: 学習=〜2025-03-31・検証=2025-04-01〜2026-03-31（条件選定）・
テスト=2026-04-01〜07-15（選択条件のみ1回評価）。
選定基準: 検証ROI>=95 ∧ n>=100 で的中率最大（exp_win_axis_sweep_wt.pyと同一基準）。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import lightgbm as lgb

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

STAKE = 100
TRAIN_FROM, TRAIN_TO = "2022-12-01", "2025-03-31"
VAL_FROM, VAL_TO = "2025-04-01", "2026-03-31"
TEST_FROM, TEST_TO = "2026-04-01", "2026-07-15"
SEED = 42

# 現行本番ゲート（比較対象・M_GAP12_MIN / M_WIN_RANK_MIN と同値）
GAP12_MIN = 0.10
WIN_RANK_MIN = 3
LEG_MIN_ODDS = 15.0


def train_models():
    print("学習データ読み込み...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=TRAIN_TO))
    df = df[df["finish_order"].notna()]
    X = prepare_X(df)
    win_y = (df["finish_order"] == 1).astype(int)
    top3_y = df["finish_order"].between(1, 3).astype(int)

    def _fit(y):
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(X, y)
        return m

    print("1着モデル学習...", flush=True)
    win_model = _fit(win_y)
    print("3着内モデル学習...", flush=True)
    top3_model = _fit(top3_y)
    return win_model, top3_model


def collect(tf, tt, win_model, top3_model):
    """不一致(WT◎≠システム◎)の7車レースを収集し、gap12/win_rank/diff/ratioを付与する。"""
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    X = prepare_X(df)
    df["p_win"] = win_model.predict_proba(X)[:, 1]
    df["p_top3"] = top3_model.predict_proba(X)[:, 1]
    rks_all = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rks = [rk for rk in rks_all if ne_map.get(rk) == 7]
        marks, fins = {}, {}
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, f, pmv, fo in c.execute(q, ch):
                marks.setdefault(rk, {})[int(f)] = pmv
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(f)))
        trio_bd = defaultdict(dict)
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, bt, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0 or fv >= 9000:
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio_bd[rk][frozenset(parts)] = fv
    pm = _load_payouts_wt(rks)

    def _iv(v):
        return None if v is None or pd.isna(v) else int(v)

    races = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk, {})
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue

        g_top3 = g.sort_values("p_top3", ascending=False).reset_index(drop=True)
        g_win = g.sort_values("p_win", ascending=False).reset_index(drop=True)
        rows_top3 = list(g_top3.itertuples(index=False))

        wt_marks = [fno for fno, v in marks.get(rk, {}).items() if v == 1]
        wt_top = min(wt_marks) if wt_marks else None
        if wt_top is None:
            continue

        top3_m1 = int(rows_top3[0].frame_no)
        if wt_top == top3_m1:
            continue  # 不一致のみ対象（現行M仕様）

        probs_top3 = [float(x) for x in g_top3["p_top3"].tolist()]
        gap12_top3 = probs_top3[0] - probs_top3[1]

        win_rank_map = {int(r.frame_no): i + 1 for i, r in enumerate(g_win.itertuples(index=False))}
        top3_m1_win_rank = win_rank_map.get(top3_m1)

        p_top3_axis = float(rows_top3[0].p_top3)
        p_win_axis_row = g[g["frame_no"] == top3_m1]
        p_win_axis = float(p_win_axis_row["p_win"].iloc[0])
        diff = p_top3_axis - p_win_axis
        ratio = p_win_axis / p_top3_axis if p_top3_axis > 0 else None

        def _mate_same_line_nige(rows, axis_fno):
            ax_row = next((r for r in rows if int(r.frame_no) == axis_fno), None)
            if ax_row is None:
                return None
            lg = _iv(getattr(ax_row, "line_group", None))
            if lg is None:
                return None
            lp = _iv(getattr(ax_row, "line_pos", None))
            want = 1 if lp == 2 else 2
            cands = []
            for r in rows:
                fno = int(r.frame_no)
                st = r.style if isinstance(getattr(r, "style", None), str) else ""
                if fno == axis_fno or _iv(getattr(r, "line_group", None)) != lg or st != "逃":
                    continue
                cands.append((fno, _iv(getattr(r, "line_pos", None))))
            if not cands:
                return None
            cands.sort()
            return next((f for f, lp2 in cands if lp2 == want), cands[0][0])

        mate = _mate_same_line_nige(rows_top3, top3_m1)

        races.append({
            "trio": trio, "board": board,
            "top3": frozenset(fno for _, fno in f[:3]),
            "trio_pay": pm.get(rk, {}).get(("trio", frozenset(fno for _, fno in f[:3])), 0),
            "gap12": gap12_top3,
            "win_rank": top3_m1_win_rank,
            "diff": diff, "ratio": ratio,
            "axis": top3_m1, "mate": mate,
        })
    return races


def settle_trio(races, gate_fn, leg=LEG_MIN_ODDS):
    n = hits = bet = pay = 0
    for r in races:
        if not gate_fn(r):
            continue
        a, b = r["axis"], r["mate"]
        if a is None or b is None or a == b:
            continue
        buy = [frozenset({a, b, t}) for t in r["board"] - {a, b}
               if (r["trio"].get(frozenset({a, b, t})) or 0) >= leg]
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["top3"] in buy:
            hits += 1
            pay += r["trio_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def build_gates():
    """現行(比較対象) + 連続量の複合ゲート候補群。"""
    gates = [
        ("現行: gap12>=.10 OR win_rank>=3",
         lambda r: r["gap12"] >= GAP12_MIN or (r["win_rank"] is not None and r["win_rank"] >= WIN_RANK_MIN)),
        ("gap12単独(参考)", lambda r: r["gap12"] >= GAP12_MIN),
        ("win_rank単独(参考)", lambda r: r["win_rank"] is not None and r["win_rank"] >= WIN_RANK_MIN),
    ]
    # diff 単独しきい値
    for th in (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        gates.append((f"diff>={th:.2f}単独", (lambda r, th=th: r["diff"] >= th)))
    # gap12 OR diff（win_rankの連続量版に置換）
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        gates.append((f"gap12>=.10 OR diff>={th:.2f}",
                       (lambda r, th=th: r["gap12"] >= GAP12_MIN or r["diff"] >= th)))
    # ratio 単独しきい値（低いほど「勝ちきれない」）
    for th in (0.30, 0.40, 0.50, 0.60):
        gates.append((f"ratio<={th:.2f}単独",
                       (lambda r, th=th: r["ratio"] is not None and r["ratio"] <= th)))
    for th in (0.30, 0.40, 0.50, 0.60):
        gates.append((f"gap12>=.10 OR ratio<={th:.2f}",
                       (lambda r, th=th: r["gap12"] >= GAP12_MIN
                        or (r["ratio"] is not None and r["ratio"] <= th))))
    # ratio の微調整（0.30近傍）
    for th in (0.20, 0.25, 0.35):
        gates.append((f"ratio<={th:.2f}単独",
                       (lambda r, th=th: r["ratio"] is not None and r["ratio"] <= th)))
        gates.append((f"gap12>=.10 OR ratio<={th:.2f}",
                       (lambda r, th=th: r["gap12"] >= GAP12_MIN
                        or (r["ratio"] is not None and r["ratio"] <= th))))
    # 現行ゲートに ratio<=0.30 を第3項として追加（拡張案）
    gates.append(("現行(gap12 OR win_rank) OR ratio<=0.30",
                   lambda r: r["gap12"] >= GAP12_MIN
                   or (r["win_rank"] is not None and r["win_rank"] >= WIN_RANK_MIN)
                   or (r["ratio"] is not None and r["ratio"] <= 0.30)))
    return gates


def main():
    win_model, top3_model = train_models()
    print("\n検証データ構築...", flush=True)
    val = collect(VAL_FROM, VAL_TO, win_model, top3_model)
    print("テストデータ構築...", flush=True)
    test = collect(TEST_FROM, TEST_TO, win_model, top3_model)
    print(f"不一致7車レース 検証 {len(val)}R / テスト {len(test)}R", flush=True)

    gates = build_gates()
    print("\n===== 検証期間（条件選定） =====", flush=True)
    results = []
    for name, gf in gates:
        n, h, roi = settle_trio(val, gf)
        results.append((name, gf, n, h, roi))
        flag = "*" if n >= 100 and roi >= 95 else " "
        print(f"  {flag} {name:38s} n={n:4d} 的中={h/n*100 if n else 0:5.1f}% ROI={roi:6.1f}%")

    frontier = sorted([x for x in results if x[2] >= 100 and x[4] >= 95],
                       key=lambda x: -(x[3] / x[2]))

    print("\n===== テスト期間（比較用: 現行ゲート+拡張案を常に評価） =====", flush=True)
    always_eval = [x for x in results if x[0] in (
        "現行: gap12>=.10 OR win_rank>=3",
        "現行(gap12 OR win_rank) OR ratio<=0.30",
    )]
    for name, gf, n, h, roi in always_eval:
        tn, th, troi = settle_trio(test, gf)
        print(f"  {name:38s} 検証(的中{h/n*100:.1f}%/ROI{roi:.1f}%/n={n}) "
              f"→ テスト n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")

    if not frontier:
        print("\n検証ROI>=95% ∧ n>=100 の候補なし。")
        return

    print("\n===== テスト期間（的中率上位の選定条件を1回評価） =====", flush=True)
    for name, gf, n, h, roi in frontier[:6]:
        tn, th, troi = settle_trio(test, gf)
        marker = " <== 検証的中率最大" if (name, gf, n, h, roi) == frontier[0] else ""
        print(f"  {name:38s} 検証(的中{h/n*100:.1f}%/ROI{roi:.1f}%/n={n}) "
              f"→ テスト n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%{marker}")

    print("\n===== テスト期間（検証ROI上位の選定条件を1回評価） =====", flush=True)
    frontier_by_roi = sorted([x for x in results if x[2] >= 100 and x[4] >= 95],
                              key=lambda x: -x[4])
    for name, gf, n, h, roi in frontier_by_roi[:6]:
        tn, th, troi = settle_trio(test, gf)
        print(f"  {name:38s} 検証(的中{h/n*100:.1f}%/ROI{roi:.1f}%/n={n}) "
              f"→ テスト n={tn} 的中={th/tn*100 if tn else 0:.1f}% ROI={troi:.1f}%")


if __name__ == "__main__":
    main()
