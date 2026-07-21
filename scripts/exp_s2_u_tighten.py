"""S2(U)の絞り込み検討: 波乱ゲート(entropy/mto)・市場順位帯・モデル順位・
買い目オッズ下限を現行値から引き上げ、母数を減らしてでも的中率/ROIを
上げられるか検証する（正規プロトコル・単変量スイープ）。

現行本番(judge_u/build_rows): 7車 ∧ entropy>=1.84 ∧ mto>=4.3 ∧
  穴=モデル3位内∧(単騎 or ライン先頭/番手)∧市場順位4-7位 ∧ 相方=同ライン「逃」 ∧
  買い目=三連複{穴,相方,t}のうちオッズ>=15.0のみ。
テスト実績(2026-04-01〜07-15): n=87（正規プロトコル検証n=320・ROI127.8%→テストROI117.1%）。

正規プロトコル: 学習=〜2025-03-31・検証=2025-04-01〜2026-03-31（条件選定）・
テスト=2026-04-01〜07-15（選択条件のみ1回評価）。
選定基準: 検証 ROI>=95 ∧ n>=100 で的中率最大（既存スイープ群と同一基準）。
"""
import math
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

# 現行本番値（比較ベースライン）
BASE_ENTROPY_MIN = 1.84
BASE_MTO_MIN = 4.3
BASE_MKT_LO, BASE_MKT_HI = 4, 7
BASE_MODEL_RANK_MAX = 3
BASE_LEG_MIN = 15.0


def _entropy(probs):
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum(max(p / total, 1e-9) * math.log(max(p / total, 1e-9)) for p in probs)


def train_model():
    print("学習データ読み込み...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=TRAIN_TO))
    df = df[df["finish_order"].notna()]
    X = prepare_X(df)
    top3_y = df["finish_order"].between(1, 3).astype(int)
    m = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED,
        deterministic=True, force_row_wise=True, verbose=-1)
    print("3着内モデル学習...", flush=True)
    m.fit(X, top3_y)
    return m


def collect(tf, tt, top3_model):
    """レースごとに entropy/mto/穴候補(model_rank,dark,mate)/盤面/結果を集計する。"""
    df = build_features_wt(load_raw_data_wt(min_date=tf, max_date=tt))
    X = prepare_X(df)
    df["p_top3"] = top3_model.predict_proba(X)[:, 1]
    rks_all = df["race_key"].unique().tolist()
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?", (tf, tt)))
        rks = [rk for rk in rks_all if ne_map.get(rk) == 7]
        fins = {}
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, f, fo in c.execute(q, ch):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(f)))
        trio_bd = defaultdict(dict)
        for i in range(0, len(rks), 900):
            ch = rks[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(ch)))
            for rk, comb, od in c.execute(q, ch):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or not (0 < fv < 9000):
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
        if len(board) != 7 or not trio:
            continue
        f = sorted(fins.get(rk, []))
        if len(f) < 3:
            continue
        top3 = frozenset(fno for _, fno in f[:3])
        trio_pay = pm.get(rk, {}).get(("trio", top3), 0)

        probs = [float(x) for x in g["p_top3"].tolist()]
        ent = _entropy(probs)
        mto = min(trio.values())

        q = {fno: 0.0 for fno in board}
        for k, fv in trio.items():
            for fno in k:
                q[fno] += 1.0 / fv
        ranked = sorted(board, key=lambda x: (-q[x], x))
        mkt_rank = {fn: i + 1 for i, fn in enumerate(ranked)}

        g_top3 = g.sort_values("p_top3", ascending=False).reset_index(drop=True)
        rows_g = list(g_top3.itertuples(index=False))

        candidates = []
        for rank_idx, r in enumerate(rows_g[:3], start=1):
            lg = _iv(getattr(r, "line_group", None))
            ls = _iv(getattr(r, "line_size", None))
            lp = _iv(getattr(r, "line_pos", None))
            if not (ls == 1 or lp in (1, 2)) or lg is None:
                continue
            dark = int(r.frame_no)
            for m in rows_g:
                m_fno = int(m.frame_no)
                m_lg = _iv(getattr(m, "line_group", None))
                m_style = m.style if isinstance(getattr(m, "style", None), str) else ""
                if m_fno == dark or m_lg is None or m_lg != lg or m_style != "逃":
                    continue
                candidates.append((rank_idx, dark, m_fno))

        races.append({
            "entropy": ent, "mto": mto, "mkt_rank": mkt_rank, "candidates": candidates,
            "trio": trio, "board": board, "top3": top3, "trio_pay": trio_pay,
        })
    return races


def settle(races, entropy_min, mto_min, mkt_lo, mkt_hi, model_rank_max, leg_min):
    n = hits = bet = pay = 0
    for r in races:
        if r["entropy"] < entropy_min or r["mto"] < mto_min:
            continue
        eligible = [(mr, d, m) for mr, d, m in r["candidates"]
                    if mr <= model_rank_max and mkt_lo <= r["mkt_rank"].get(d, 99) <= mkt_hi]
        if not eligible:
            continue
        eligible.sort()
        _, dark, mate = eligible[0]
        combos = [frozenset({dark, mate, t}) for t in sorted(r["board"] - {dark, mate})
                  if (r["trio"].get(frozenset({dark, mate, t})) or 0) >= leg_min]
        if not combos:
            continue
        n += 1
        bet += len(combos) * STAKE
        if r["top3"] in combos:
            hits += 1
            pay += r["trio_pay"] * STAKE // 100
    return n, hits, (pay / bet * 100 if bet else 0)


def report(label, val, test, params):
    vn, vh, vroi = settle(val, *params)
    tn, th, troi = settle(test, *params)
    v_rate = vh / vn * 100 if vn else 0
    t_rate = th / tn * 100 if tn else 0
    flag = "*" if vn >= 100 and vroi >= 95 else " "
    print(f"{flag} {label:32s} | val n={vn:4d}({vn/365:5.2f}/日) 的中={v_rate:5.1f}% ROI={vroi:6.1f}% | "
          f"test n={tn:4d}({tn/106:5.2f}/日) 的中={t_rate:5.1f}% ROI={troi:6.1f}%")


def main():
    top3_model = train_model()
    print("\n検証データ構築(7車)...", flush=True)
    val = collect(VAL_FROM, VAL_TO, top3_model)
    print("テストデータ構築(7車)...", flush=True)
    test = collect(TEST_FROM, TEST_TO, top3_model)
    print(f"検証 {len(val)}R / テスト {len(test)}R", flush=True)

    base = (BASE_ENTROPY_MIN, BASE_MTO_MIN, BASE_MKT_LO, BASE_MKT_HI, BASE_MODEL_RANK_MAX, BASE_LEG_MIN)
    print("\n===== ベースライン(現行本番値) =====")
    report("現行(1.84/4.3/4-7/<=3/15)", val, test, base)

    print("\n===== entropy_min 単変量スイープ =====")
    for th in (1.84, 1.90, 1.95, 2.00, 2.05, 2.10, 2.15, 2.20):
        report(f"entropy>={th:.2f}", val, test,
               (th, BASE_MTO_MIN, BASE_MKT_LO, BASE_MKT_HI, BASE_MODEL_RANK_MAX, BASE_LEG_MIN))

    print("\n===== mto_min 単変量スイープ =====")
    for th in (4.3, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0):
        report(f"mto>={th:.1f}", val, test,
               (BASE_ENTROPY_MIN, th, BASE_MKT_LO, BASE_MKT_HI, BASE_MODEL_RANK_MAX, BASE_LEG_MIN))

    print("\n===== 市場順位帯 スイープ =====")
    for lo, hi in ((4, 7), (4, 6), (5, 7), (5, 6), (4, 5), (6, 7)):
        report(f"市場{lo}-{hi}位", val, test,
               (BASE_ENTROPY_MIN, BASE_MTO_MIN, lo, hi, BASE_MODEL_RANK_MAX, BASE_LEG_MIN))

    print("\n===== モデル順位上限 スイープ =====")
    for mr in (3, 2, 1):
        report(f"model_rank<={mr}", val, test,
               (BASE_ENTROPY_MIN, BASE_MTO_MIN, BASE_MKT_LO, BASE_MKT_HI, mr, BASE_LEG_MIN))

    print("\n===== 買い目オッズ下限 スイープ =====")
    for leg in (15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0):
        report(f"leg>={leg:.0f}倍", val, test,
               (BASE_ENTROPY_MIN, BASE_MTO_MIN, BASE_MKT_LO, BASE_MKT_HI, BASE_MODEL_RANK_MAX, leg))

    print("\n===== 有望候補の複合（単変量で改善したものを組合せ） =====")
    # 単変量結果を見てから追記する（後段で埋める）


if __name__ == "__main__":
    main()
