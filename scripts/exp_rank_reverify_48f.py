"""48特徴モデル（sb_dyn 4特徴追加後）での S2/S3 現行条件の再検証。

モデル更新で pred_prob の分布が変わるため、凍結閾値（ent≥1.84・mto≥4.3・
gap12≥0.10）の意味も変わる。現行条件と近傍を新モデル（lgbm_wt_val25・48特徴）で
再評価し、条件維持/微調整を判断する。

正規プロトコル: 検証 2025-04-01〜2026-03-31（選定・比較の場）／
テスト 2026-04-01〜07-15（現行条件と採用候補のみ確認）。
S2/S3 の定義は backfill_um_rank_wt.build_rows と同一（S2優先の重複排除も同じ）。
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X

STAKE = 100
MODEL = "lgbm_wt_val25"
VAL = ("2025-04-01", "2026-03-31")
TEST = ("2026-04-01", "2026-07-15")


def _entropy(probs):
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum(max(p / total, 1e-9) * math.log(max(p / total, 1e-9)) for p in probs)


def collect(tf, tt, model):
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
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
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or not (0 < fv < 9000):
                    continue
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    trio_bd[rk][parts] = fv
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
        if len(board) != 7 or not trio:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue
        g = g.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        probs = [float(x) for x in g["pred_prob"].tolist()]
        rows_g = list(g.itertuples(index=False))
        q = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        mkt = {fo_: i + 1 for i, fo_ in enumerate(sorted(board, key=lambda x: (-q[x], x)))}
        wt_top = min((fno for fno, v in marks.get(rk, {}).items() if v == 1), default=None)
        # S2の穴ペア列挙（backfill_um_rank_wt と同一の決定則）
        eligible = []
        for rank_idx, r in enumerate(rows_g[:3], start=1):
            lg = _iv(getattr(r, "line_group", None))
            ls = _iv(getattr(r, "line_size", None))
            lp = _iv(getattr(r, "line_pos", None))
            if not (ls == 1 or lp in (1, 2)) or lg is None:
                continue
            dark = int(r.frame_no)
            if not (4 <= mkt.get(dark, 8) <= 7):
                continue
            for m in rows_g:
                m_fno = int(m.frame_no)
                m_lg = _iv(getattr(m, "line_group", None))
                m_style = m.style if isinstance(getattr(m, "style", None), str) else ""
                if m_fno == dark or m_lg is None or m_lg != lg or m_style != "逃":
                    continue
                eligible.append((rank_idx, dark, m_fno))
        eligible.sort()
        u_pair = (eligible[0][1], eligible[0][2]) if eligible else None
        # S3の相方（m1と同L逃・lp相補優先→車番最小）
        r1 = rows_g[0]
        m1 = int(r1.frame_no)
        lg1 = _iv(getattr(r1, "line_group", None))
        lp1 = _iv(getattr(r1, "line_pos", None))
        mate_m = None
        if lg1 is not None:
            want = 1 if lp1 == 2 else 2
            mates = sorted(
                (int(r.frame_no), _iv(getattr(r, "line_pos", None))) for r in rows_g
                if int(r.frame_no) != m1 and _iv(getattr(r, "line_group", None)) == lg1
                and (r.style if isinstance(getattr(r, "style", None), str) else "") == "逃")
            if mates:
                mate_m = next((f for f, lp in mates if lp == want), mates[0][0])
        races.append({
            "trio": trio, "board": board,
            "top3": frozenset(fno for _, fno in f[:3]),
            "trio_pay": pm.get(rk, {}).get(("trio", frozenset(fno for _, fno in f[:3])), 0),
            "gap12": probs[0] - probs[1], "ent": _entropy(probs),
            "mto": min(trio.values()),
            "m1": m1, "wt_top": wt_top, "u_pair": u_pair, "mate_m": mate_m,
        })
    return races


def _settle(races, sel_fn):
    """sel_fn(r) -> (a, b) or None。三連複2車軸×目>=15（現行共通）。"""
    n = hits = bet = pay = 0
    for r in races:
        ab = sel_fn(r)
        if ab is None:
            continue
        a, b = ab
        buy = [frozenset({a, b, t}) for t in r["board"] - {a, b}
               if (r["trio"].get(frozenset({a, b, t})) or 0) >= 15.0]
        if not buy:
            continue
        n += 1
        bet += len(buy) * STAKE
        if r["top3"] in buy:
            hits += 1
            pay += r["trio_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def s2_sel(ent_min, mto_min):
    def f(r):
        if r["ent"] < ent_min or r["mto"] < mto_min or r["u_pair"] is None:
            return None
        return r["u_pair"]
    return f


def s3_sel(gap_min, dedup=True):
    def f(r):
        if r["wt_top"] is None or r["m1"] == r["wt_top"]:
            return None
        if r["gap12"] < gap_min or r["mate_m"] is None:
            return None
        if dedup and r["u_pair"] is not None and set(r["u_pair"]) == {r["m1"], r["mate_m"]}:
            # S2条件成立レースでの同一ペアは S2 優先（現行運用と同じ）
            if r["ent"] >= 1.84 and r["mto"] >= 4.3:
                return None
        return (r["m1"], r["mate_m"])
    return f


def main():
    model = load_model(MODEL)
    val = collect(*VAL, model)
    test = collect(*TEST, model)
    print(f"7車 検証 {len(val)}R / テスト {len(test)}R", flush=True)

    print("\n== 新モデルでの分布シフト確認（検証期間・7車） ==")
    import numpy as np
    g = np.array([r["gap12"] for r in val])
    e = np.array([r["ent"] for r in val])
    print(f"  gap12: mean={g.mean():.4f} p50={np.percentile(g,50):.4f} "
          f"p75={np.percentile(g,75):.4f}  >=0.10 率={np.mean(g>=0.10)*100:.1f}%")
    print(f"  ent  : mean={e.mean():.4f} p75={np.percentile(e,75):.4f} "
          f" >=1.84 率={np.mean(e>=1.84)*100:.1f}%")

    print("\n== S2（現行 ent>=1.84 ∧ mto>=4.3 + 近傍）検証期間 ==")
    s2_cells = []
    for ent_min in (1.80, 1.84, 1.88):
        for mto_min in (4.0, 4.3, 5.0):
            n, h, roi = _settle(val, s2_sel(ent_min, mto_min))
            tag = " ←現行" if (ent_min, mto_min) == (1.84, 4.3) else ""
            print(f"  ent>={ent_min:.2f}∧mto>={mto_min:.1f}: n={n} "
                  f"的中={h/n*100 if n else 0:.1f}% ROI={roi:.1f}%{tag}")
            s2_cells.append((roi, n, ent_min, mto_min))
    n, h, roi = _settle(test, s2_sel(1.84, 4.3))
    print(f"  【テスト・現行条件】n={n} 的中={h/n*100 if n else 0:.1f}% ROI={roi:.1f}%")

    print("\n== S3（現行 不一致∧gap12>=0.10 + 近傍）検証期間 ==")
    for gap_min in (0.08, 0.10, 0.12):
        n, h, roi = _settle(val, s3_sel(gap_min))
        tag = " ←現行" if gap_min == 0.10 else ""
        print(f"  gap12>={gap_min:.2f}: n={n} 的中={h/n*100 if n else 0:.1f}% ROI={roi:.1f}%{tag}")
    n, h, roi = _settle(test, s3_sel(0.10))
    print(f"  【テスト・現行条件】n={n} 的中={h/n*100 if n else 0:.1f}% ROI={roi:.1f}%")


if __name__ == "__main__":
    main()
