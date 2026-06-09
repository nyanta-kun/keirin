"""中間オッズ集中: ≤6車で買い目を三連複オッズ帯別に分解し、
中間帯が高オッズ帯より「再現度(train→test ROI一致)」「的中率」で上回るか検証。

高オッズ(大穴)＝低的中率・高分散・再現困難。中間オッズ＝市場非効率(favorite-longshot bias)が
残りやすく的中率・再現度を上げられる可能性。逆に低オッズ＝鉄板=ガミ。

3観点で測る:
  ① 市場効率(モデル無し): 全三連複組合せをオッズ帯別 → 帯ごとの天井ROI/的中
  ② モデル+オッズ帯: モデル2軸流し(std3点/wide全点)の買い目を帯別に分け、
     中間帯のみに絞ると再現黒字+的中率↑になるか
  ③ 「中間オッズ集中」合成: wide流しのうちオッズ∈[L,H]のみ購入
pooled lgbm_wt・≤6車・finish_order≥1・train(2023-07〜2026-02)→test(2026-03〜)。
払戻=wt_odds(最終オッズ)=上限値。再現=TR/TE>100% かつ TE十分R。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import itertools
from collections import defaultdict
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.database import get_connection
from roi_robustness_wt import roi_summary

model = load_model("lgbm_wt")

# 三連複オッズ帯（円→倍）。<5鉄板 / 5-10 やや堅 / 10-20,20-40 中間 / 40-80,80-200 穴 / >=200 大穴
BANDS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 200), (200, 1e9)]
BAND_LABEL = ["<5", "5-10", "10-20", "20-40", "40-80", "80-200", ">=200"]


def band_of(o):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= o < hi:
            return i
    return len(BANDS) - 1


def load_trio_board(race_keys):
    """{race_key: {frozenset(combo_ints): odds_value}} を wt_odds から構築。"""
    board = defaultdict(dict)
    CH = 900
    with get_connection() as c:
        for i in range(0, len(race_keys), CH):
            chunk = race_keys[i:i + CH]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is None:
                    continue
                board[rk][frozenset(int(x) for x in comb.split("="))] = od
    return board


def collect(f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz <= 6].index)].copy()   # ≤6車
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks = df["race_key"].unique().tolist()
    board = load_trio_board(rks)
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 4:
            continue
        bd = board.get(rk, {})
        if not bd:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        win = frozenset(int(x) for x in fin["frame_no"])   # 実着 top3
        win_odds = bd.get(win, 0.0)
        # モデル買い目候補
        std3 = [frozenset((fr[0], fr[1], x)) for x in fr[2:5]]        # 2軸流し3点
        wide = [frozenset((fr[0], fr[1], x)) for x in fr[2:]]         # 2軸流し全点(n-2)
        allc = [frozenset(c) for c in itertools.combinations(fr, 3)]  # 全C(n,3)
        rows.append({
            "rk": rk, "n": n, "gap12": p[0] - p[1], "top3_sum": p[0] + p[1] + p[2],
            "win": win, "win_odds": win_odds, "board": bd,
            "std3": std3, "wide": wide, "allc": allc,
        })
    return rows


def band_roi(rows, combo_key, band_idx=None, lo=None, hi=None):
    """各レースで combo_key の買い目のうち、オッズが [指定帯/範囲] のものを購入。
    per-race (payout, bet) を roi_summary に渡す。"""
    pays, bets = [], []
    for r in rows:
        bd = r["board"]
        sub = []
        for c in r[combo_key]:
            o = bd.get(c)
            if o is None:
                continue
            if band_idx is not None and band_of(o) != band_idx:
                continue
            if lo is not None and not (lo <= o < hi):
                continue
            sub.append(c)
        if not sub:
            continue
        bet = len(sub) * 100
        pay = r["win_odds"] * 100 if r["win"] in sub else 0.0
        pays.append(pay)
        bets.append(bet)
    return roi_summary(pays, bets), len(pays)


def fmt(s, n):
    return f"{n:>5}R {s['roi']:>5.0%} 的中{s['hit_rate']:>4.0%} [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]"


def report_market(tr, te):
    print(f"\n{'='*100}\n  ① 市場効率: ≤6車 全三連複(C(n,3))をオッズ帯別に全点買い  TR {len(tr)}R / TE {len(te)}R（最終オッズ上限値）\n{'='*100}")
    print(f"  {'帯(倍)':<9}{'TRAIN':<34}{'TEST':<34}{'再現':>5}")
    for i, lab in enumerate(BAND_LABEL):
        s1, n1 = band_roi(tr, "allc", band_idx=i)
        s2, n2 = band_roi(te, "allc", band_idx=i)
        flag = "★" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ""
        print(f"  {lab:<9}{fmt(s1,n1):<34}{fmt(s2,n2):<34}{flag:>5}")
    print("  ※ モデル無し純市場。控除率(~25%)で大半<100%。FL biasなら高オッズ帯ほど低ROIのはず。")


def report_model(tr, te):
    for ck, nm in [("std3", "2軸流し3点(現行)"), ("wide", "2軸流し全点")]:
        print(f"\n{'='*100}\n  ② モデル {nm}: 買い目をオッズ帯別に分けて購入  TR {len(tr)}R / TE {len(te)}R（上限値）\n{'='*100}")
        print(f"  {'帯(倍)':<9}{'TRAIN':<34}{'TEST':<34}{'再現':>5}")
        # 全帯まとめ（基準）
        s1, n1 = band_roi(tr, ck); s2, n2 = band_roi(te, ck)
        print(f"  {'(全帯)':<9}{fmt(s1,n1):<34}{fmt(s2,n2):<34}")
        for i, lab in enumerate(BAND_LABEL):
            s1, n1 = band_roi(tr, ck, band_idx=i)
            s2, n2 = band_roi(te, ck, band_idx=i)
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
            print(f"  {lab:<9}{fmt(s1,n1):<34}{fmt(s2,n2):<34}{flag:>5}")


def report_concentrate(tr, te):
    print(f"\n{'='*100}\n  ③ 中間オッズ集中: 2軸流し全点のうちオッズ∈[L,H]のみ購入  TR {len(tr)}R / TE {len(te)}R（上限値）\n{'='*100}")
    print(f"  {'範囲(倍)':<12}{'TRAIN':<34}{'TEST':<34}{'再現':>5}")
    RANGES = [(5, 20), (5, 40), (10, 40), (10, 80), (20, 80), (10, 200), (20, 200)]
    for lo, hi in RANGES:
        s1, n1 = band_roi(tr, "wide", lo=lo, hi=hi)
        s2, n2 = band_roi(te, "wide", lo=lo, hi=hi)
        flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else ("小標本" if n2 < 30 else "")
        print(f"  {f'{lo}-{hi}':<12}{fmt(s1,n1):<34}{fmt(s2,n2):<34}{flag:>5}")
    print("\n  ※ 高オッズ(大穴)を捨て中間帯に集中＝的中率↑・再現度↑が狙い。★再現が出るかが焦点。")


if __name__ == "__main__":
    tr = collect("2023-07-01", "2026-02-28")
    te = collect("2026-03-01", "2026-06-08")
    report_market(tr, te)
    report_model(tr, te)
    report_concentrate(tr, te)
