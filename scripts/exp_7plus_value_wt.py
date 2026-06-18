"""7車以上: 決定実験 — 較正済み確率 × 条件付き同時確率(Plackett-Luce) × value-bet。

これは18軸(精度最大化)の続きではない。**公開オッズ内に7+のエッジが在るかを問う最後の1実験**。
核心ロジック:
  EV = P_model × オッズ。エッジは「P_modelが高い」ことからは生まれない(それは的中-オッズが1:1連動)。
  エッジは「P_model > 市場示唆確率(1/オッズ)」= 市場が間違っている所からのみ生まれる。
  18軸はすべて P_model を最大化していた。P_model > P_market の value-bet を7+で回した記録は無い。

正しく回すための2つの修正(既存 run_value_backtest_wt の欠陥):
  (1) 確率の較正: pred_prob はランキング用の生出力。EV=P×odds は P が較正されないと無意味。
      → 三連複の同時スコアを TRAIN の実的中で isotonic 較正(どの同時近似でも単調なら系統誤差を吸収)。
  (2) 同時確率: 周辺積は相関/ライン構造を無視。Plackett-Luce(逐次=「1着が決まった後の2着…」)で
      ユーザーのフェーズ2「1位の結果で条件付けた再算出」を構造として内包。周辺積も交差検証で併記。

決定的判定:
  - 較正後 P_model ≈ 市場示唆確率(回帰slope≈1/相関→1)なら市場効率=エッジ無し → 7+撤退をEV原理で確定。
  - EV≥ev_min の組合せが TRAIN/TEST 両方で ROI>100% を再現するなら → 7+復活。
  - 較正後は EV≥1 の組合せ自体が TRAIN でほぼ空になるはず(P≈1/(odds·overround)<1)。空でなく黒字再現なら本物。

払戻/オッズ=wt_odds最終=上限値(実運用は下振れ)。train 2023-07〜2026-02 / test 2026-03〜。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import itertools
from collections import defaultdict
import numpy as np
from sklearn.isotonic import IsotonicRegression
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.database import get_connection
from roi_robustness_wt import roi_summary

model = load_model("lgbm_wt")


def load_trio_board(race_keys):
    board = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type='trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                if od is not None and od > 0:
                    board[rk][frozenset(int(x) for x in comb.split("="))] = float(od)
    return board


def pl_set_prob(strength: dict, trio: frozenset) -> float:
    """Plackett-Luce: 強さ strength から「この3頭が(順不同で)top3」になる確率。
    逐次抽出= 1着∝強さ → 残りから2着∝強さ → 3着。6順序の総和。
    フェーズ2「1着の結果で条件付けた2着以下」の生成構造そのもの。"""
    W = sum(strength.values())
    s = 0.0
    for x, y, z in itertools.permutations(trio):
        wx, wy, wz = strength[x], strength[y], strength[z]
        d1 = W
        d2 = W - wx
        d3 = W - wx - wy
        if d2 <= 0 or d3 <= 0:
            continue
        s += (wx / d1) * (wy / d2) * (wz / d3)
    return s


def collect(f, t):
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz >= 7].index)].copy()      # 7車以上
    df = df[df["finish_order"] >= 1].copy()
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    rks = df["race_key"].unique().tolist()
    board = load_trio_board(rks)
    races = []
    for rk, g in df.groupby("race_key"):
        n = len(g)
        if n < 7:
            continue
        bd = board.get(rk, {})
        if not bd:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        win = frozenset(int(x) for x in fin["frame_no"])
        if win not in bd:        # 的中組合せのオッズが無い=払戻計算不能、除外
            continue
        # 強さ=pred_prob(top3確率)。PLの逐次抽出強さとして使用(単調・isotonicが再較正)
        strength = {int(fr): max(float(p), 1e-6)
                    for fr, p in zip(g["frame_no"], g["pred_prob"])}
        trios = []   # (combo, odds, pl, prod, hit)
        for combo, odds in bd.items():
            if len(combo) != 3 or not combo.issubset(strength.keys()):
                continue
            prod = 1.0
            for fr in combo:
                prod *= strength[fr]
            pl = pl_set_prob(strength, combo)
            trios.append((combo, odds, pl, prod, 1 if combo == win else 0))
        if trios:
            races.append({"trios": trios, "win": win})
    return races


def flat(races, idx):
    return np.array([tr[idx] for r in races for tr in r["trios"]], dtype=float)


def fit_iso(races, score_idx):
    x = flat(races, score_idx)
    y = flat(races, 4)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, y)
    return iso


def build_cal(races, iso, score_idx):
    """全trioの較正済みP・odds・hit・市場確率を一括計算(ベクトル化)。
    返り値: per-race の (odds[], p_cal[], hit[], p_market[]) リスト。"""
    flat_scores = flat(races, score_idx)
    flat_p = iso.transform(flat_scores)           # 一括transform(数百万を1回)
    out, pos = [], 0
    for r in races:
        m = len(r["trios"])
        odds = np.array([tr[1] for tr in r["trios"]], dtype=float)
        hit = np.array([tr[4] for tr in r["trios"]], dtype=float)
        p_cal = flat_p[pos:pos + m]
        pos += m
        ovr = (1.0 / odds).sum()
        out.append((odds, p_cal, hit, (1.0 / odds) / ovr))
    return out


def value_roi(cal, ev_min, max_per_race=None):
    pays, bets = [], []
    for odds, p_cal, hit, _ in cal:
        ev = p_cal * odds
        mask = ev >= ev_min
        if not mask.any():
            continue
        sel_odds = odds[mask]; sel_hit = hit[mask]; sel_ev = ev[mask]
        if max_per_race and mask.sum() > max_per_race:
            order = np.argsort(-sel_ev)[:max_per_race]
            sel_odds = sel_odds[order]; sel_hit = sel_hit[order]
        pays.append(float((sel_odds * 100 * sel_hit).sum()))
        bets.append(int(len(sel_odds)) * 100)
    return roi_summary(pays, bets), len(pays)


def market_diag(cal, label):
    """決定的診断: 較正後 P_model が市場示唆確率を超えるか/単に再現するか。"""
    pmod = np.concatenate([c[1] for c in cal])
    hits = np.concatenate([c[2] for c in cal])
    pm = np.concatenate([c[3] for c in cal])
    # 回帰 P_model = a + b*market_p
    b, a = np.polyfit(pm, pmod, 1)
    corr = np.corrcoef(pm, pmod)[0, 1]
    # 較正の健全性(TEST): P_model 十分位の実的中
    print(f"\n  ▼ 市場効率診断 [{label}]  ({len(pm):,} trios)")
    print(f"    P_model = {a:+.4f} + {b:.3f}·P_market   corr={corr:.3f}")
    print(f"    (slope≈1・corr→1 ⇒ モデルは市場を再現するだけ=エッジ無し)")
    # value側: P_model>P_market の組で「実的中 vs 市場確率」
    edge = pmod > pm
    if edge.sum() > 0:
        realized = hits[edge].mean()
        claimed = pmod[edge].mean()
        mkt = pm[edge].mean()
        print(f"    P_model>P_market の組({edge.sum():,}件): モデル主張{claimed:.4f} / 実的中{realized:.4f} / 市場示唆{mkt:.4f}")
        print(f"    (実的中≈市場示唆 ⇒ モデルの'割安'は錯覚=市場が正しい / 実的中≈モデル主張 ⇒ 本物のエッジ)")


def overround_report(races):
    ovrs = [sum(1.0 / odds for _, odds, _, _, _ in r["trios"]) for r in races]
    ovrs = np.array(ovrs)
    print(f"  三連複オッズ盤面 overround 中央値 {np.median(ovrs):.3f} "
          f"(=Σ1/odds・控除込み)  → 市場ROI上限 ≈ {1/np.median(ovrs):.0%}")


def report(tr, te):
    print(f"\n{'='*100}")
    print(f"  7+ value-bet 決定実験 (較正済みP × Plackett-Luce同時確率)  TR {len(tr)}R / TE {len(te)}R")
    print(f"{'='*100}")
    overround_report(tr)

    for score_idx, name in [(2, "Plackett-Luce(条件付き同時)"), (3, "周辺積(独立近似)")]:
        print(f"\n{'─'*100}\n  ◆ 同時確率モデル: {name}\n{'─'*100}")
        iso = fit_iso(tr, score_idx)
        cal_tr = build_cal(tr, iso, score_idx)
        cal_te = build_cal(te, iso, score_idx)
        market_diag(cal_te, f"{name}・TEST")

        print(f"\n  ▼ value-bet ROIスイープ (EV=P_cal×odds ≥ ev_min を購入・最終オッズ上限値)")
        print(f"    {'ev_min':<8}{'TRAIN':<40}{'TEST':<40}{'再現':>6}")
        for ev_min in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
            s1, n1 = value_roi(cal_tr, ev_min)
            s2, n2 = value_roi(cal_te, ev_min)
            def f(s, n):
                return (f"{n:>5}R {s['roi']:>5.0%} 的中{s['hit_rate']:>4.0%} "
                        f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]")
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else \
                   ("空" if n1 == 0 else ("小標本" if n2 < 30 else ""))
            print(f"    {ev_min:<8}{f(s1,n1):<40}{f(s2,n2):<40}{flag:>6}")

    print(f"\n{'='*100}")
    print("  判定: TRAINで★再現(>100%)が出れば7+復活。出なければ較正済みvalue-betでも壁=公開オッズ内エッジ無しをEV原理で確定。")
    print("        市場効率診断の slope≈1/corr→1/実的中≈市場示唆 が揃えば、的中-オッズ1:1連動の正体=市場効率を直接証明。")


if __name__ == "__main__":
    print("collecting TRAIN (2023-07〜2026-02)...", flush=True)
    tr = collect("2023-07-01", "2026-02-28")
    print(f"  TRAIN {len(tr)} races", flush=True)
    print("collecting TEST (2026-03〜)...", flush=True)
    te = collect("2026-03-01", "2026-06-08")
    print(f"  TEST {len(te)} races", flush=True)
    report(tr, te)
