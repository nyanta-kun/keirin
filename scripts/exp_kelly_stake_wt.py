"""≤6車: 較正確率 × Kellyステーク傾斜（ロードマップ #4・docs/analysis/08）

目的: 固定3点フラットでなく「較正勝率に比例した賭け金(=Kelly)」で**資金成長率**を最適化できるか。
転用元: exp_7plus_value_wt.py(isotonic較正+Plackett-Luce) / exp_stake_tilt_wt.py(≤6車collect+top3_sum傾斜) /
        roi_robustness_wt.py(bootstrap CI)。再発明しない。

検証する2仮説:
  H-A (Kelly-on-value): ≤6車には 較正P > 市場示唆P の真の誤価格があり、Kellyサイズで成長率↑。
       (7+ では P_cal≈P_market でエッジ無し=否定済。≤6車=非効率ポケット有り、で再検証)
  H-B (Kelly-on-selection): 値の誤価格は無いが、top3_sum傾斜の連続版としてKellyが離散ティアを成長率で上回る。

判定の鍵:
  (1) 較正診断: 較正後 P_cal を市場示唆確率と回帰。slope≈1/corr→1/(P_cal>P_market 組の実的中≈市場示唆)
      なら ≤6車でも市場効率=value無し → H-A 否定。
  (2) 資金成長率 E[ln m_r] (Kellyが最大化する目的関数そのもの)を flat/top3_sum傾斜/Kelly で TRAIN→TEST 比較。
      各手法は重みベクトル w_i のみ異なり、レバレッジ s* は TRAIN で最適化(=各手法のベスト成長)→ TEST OOS 評価。
      重み形状(均一 vs ティア vs エッジ比例)の優劣だけを切り分ける。

スコープ: S/A 層(三連複2軸流し3点)。SS(三連単・N<100)は別券種のため本実験から除外(将来)。
モデル: lgbm_wt_eval(holdout・test>=2026-03 は OOS)。較正は TRAIN のみで fit(test 漏洩なし)。
払戻/オッズ = wt_odds 最終 = 上限値(実運用は下振れ)。train 2023-07〜2026-02 / test 2026-03〜。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import itertools
from collections import defaultdict
import numpy as np
from sklearn.isotonic import IsotonicRegression
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.database import get_connection
from src.strategy_wt import upset_tier, STAKE_TILT_DEFAULT
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _filter_by_n_riders, _assign_tier
from roi_robustness_wt import roi_summary

MODEL = "lgbm_wt_eval"   # holdout評価モデル（test>=2026-03 はOOS）


def load_trio_board(race_keys):
    """{race_key: {frozenset(3車): odds}}。exp_7plus_value_wt と同型。"""
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
    """Plackett-Luce: 強さから3頭が(順不同で)top3になる確率（exp_7plus_value_wt と同一）。"""
    W = sum(strength.values())
    s = 0.0
    for x, y, z in itertools.permutations(trio):
        wx, wy, wz = strength[x], strength[y], strength[z]
        d1 = W; d2 = W - wx; d3 = W - wx - wy
        if d2 <= 0 or d3 <= 0:
            continue
        s += (wx / d1) * (wy / d2) * (wz / d3)
    return s


def collect(f, t):
    """S/A レースを集める。
    各レース: 戦略3点(2軸流し trio)= (combo, odds, pl, prod, hit) と
              全board trio(較正/value用)= 同型、utier(top3_sum帯)を保持。"""
    model = load_model(MODEL)
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    df = _filter_by_n_riders(df, 6)
    board = load_trio_board(df["race_key"].unique().tolist())
    races = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 3:
            continue
        p = g["pred_prob"].tolist()
        tier = _assign_tier(p[0] - p[1], p[0] / (3.0 / n))
        if tier not in ("S", "A"):       # S/A のみ(SS=三連単は別券種・除外)
            continue
        bd = board.get(rk, {})
        if not bd:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        win = frozenset(int(x) for x in fin["frame_no"])
        if win not in bd:                # 的中組のオッズ欠 → 払戻計算不能、除外
            continue
        strength = {int(fr): max(float(pp), 1e-6)
                    for fr, pp in zip(g["frame_no"], g["pred_prob"])}
        # 全board trio（較正の母集団 + value-bet sweep 用）
        board_trios = []
        for combo, odds in bd.items():
            if len(combo) != 3 or not combo.issubset(strength.keys()):
                continue
            prod = 1.0
            for fr in combo:
                prod *= strength[fr]
            board_trios.append((combo, odds, pl_set_prob(strength, combo), prod,
                                1 if combo == win else 0))
        if not board_trios:
            continue
        # 戦略3点: 2軸流し pred1,pred2 × thirds(pred3-5)
        fr = g["frame_no"].astype(int).tolist()
        p1, p2, thirds = fr[0], fr[1], fr[2:5]
        strat = []
        for x in thirds:
            combo = frozenset((p1, p2, x))
            if combo in bd:
                strat.append((combo, bd[combo], pl_set_prob(strength, combo),
                              strength[p1] * strength[p2] * strength[x],
                              1 if combo == win else 0))
        if not strat:
            continue
        races.append({"strat": strat, "board": board_trios,
                      "utier": upset_tier(p[0] + p[1] + p[2]), "tier": tier})
    return races


# ---- 較正 (TRAIN board trio で isotonic fit) -------------------------------

def _flat_board(races, idx):
    return np.array([tr[idx] for r in races for tr in r["board"]], dtype=float)


def fit_iso(train_races, score_idx):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(_flat_board(train_races, score_idx), _flat_board(train_races, 4))
    return iso


def market_diag(races, iso, score_idx, label):
    """較正後 P_cal が市場示唆確率を超えるか/再現するか（H-A の決定診断）。"""
    pmod, pm, hits = [], [], []
    for r in races:
        odds = np.array([tr[1] for tr in r["board"]], dtype=float)
        sc = np.array([tr[score_idx] for tr in r["board"]], dtype=float)
        h = np.array([tr[4] for tr in r["board"]], dtype=float)
        ovr = (1.0 / odds).sum()
        pmod.append(iso.transform(sc)); pm.append((1.0 / odds) / ovr); hits.append(h)
    pmod = np.concatenate(pmod); pm = np.concatenate(pm); hits = np.concatenate(hits)
    b, a = np.polyfit(pm, pmod, 1)
    corr = np.corrcoef(pm, pmod)[0, 1]
    print(f"\n  ▼ 市場効率診断 [{label}]  ({len(pm):,} trio)")
    print(f"    P_cal = {a:+.4f} + {b:.3f}·P_market   corr={corr:.3f}  "
          f"(slope≈1・corr→1 ⇒ モデルは市場を再現=value無し)")
    edge = pmod > pm
    if edge.sum():
        print(f"    P_cal>P_market の組({edge.sum():,}件): モデル主張{pmod[edge].mean():.4f} / "
              f"実的中{hits[edge].mean():.4f} / 市場示唆{pm[edge].mean():.4f}")
        print(f"    (実的中≈市場示唆 ⇒ '割安'は錯覚=市場が正しい / 実的中≈モデル主張 ⇒ 本物のエッジ)")
    # 較正信頼性(十分位)
    sc_all = _flat_board(races, score_idx)
    p_all = iso.transform(sc_all); h_all = _flat_board(races, 4)
    qs = np.quantile(p_all, np.linspace(0, 1, 11))
    print(f"    較正信頼性(十分位 予測P→実的中): ", end="")
    for k in range(10):
        m = (p_all >= qs[k]) & (p_all <= qs[k + 1] if k == 9 else p_all < qs[k + 1])
        if m.sum():
            print(f"{p_all[m].mean():.3f}/{h_all[m].mean():.3f}", end=" ")
    print()


# ---- 重みベクトル(各手法) ---------------------------------------------------

def weights(race, method, p_cal_strat):
    """戦略3点 each の賭け金重み w_i を返す。"""
    n = len(race["strat"])
    if method == "flat":
        return np.ones(n)
    if method == "tilt":                       # top3_sum 帯倍率(レース一律)
        return np.full(n, float(STAKE_TILT_DEFAULT.get(race["utier"], 1)))
    if method in ("kelly", "kelly_x_tilt"):     # 単一賭けKelly: (p·o-1)/(o-1)=(EV-1)/(o-1)
        o = np.array([tr[1] for tr in race["strat"]], dtype=float)
        safe = o > 1.0                          # o≤1 は利得ゼロ/不能 → 賭けない
        f = np.zeros(n)
        f[safe] = np.clip((p_cal_strat[safe] * o[safe] - 1.0) / (o[safe] - 1.0), 0.0, None)
        if method == "kelly_x_tilt":            # Kelly形状 × top3_sum帯ゲート(Q3/Q4=0)
            f = f * float(STAKE_TILT_DEFAULT.get(race["utier"], 1))
        return f
    raise ValueError(method)


def per_race_X(races, method, iso, score_idx):
    """各レースの X_r = Σ_i w_i·(o_i·hit_i − 1)（フラクショナル賭けの相対損益）と
       unit-ROI 用 (payout, bet) を返す。"""
    X, pays, bets = [], [], []
    for r in races:
        sc = np.array([tr[score_idx] for tr in r["strat"]], dtype=float)
        p_cal = iso.transform(sc)
        w = weights(r, method, p_cal)
        if w.sum() <= 0:
            continue                            # 賭けないレース(ゲート見送り)
        o = np.array([tr[1] for tr in r["strat"]], dtype=float)
        hit = np.array([tr[4] for tr in r["strat"]], dtype=float)
        X.append(float((w * (o * hit - 1.0)).sum()))
        pays.append(float((w * o * hit).sum() * 100))
        bets.append(float(w.sum() * 100))
    return np.array(X), pays, bets


def opt_scale(X):
    """TRAIN で資金成長 Σln(1+s·X_r) を最大化する レバレッジ s* を1次元探索。"""
    X = X[np.abs(X) > 1e-12]
    if len(X) == 0:
        return 0.0, 0.0
    xmin = X.min()
    s_cap = (0.999 / (-xmin)) if xmin < 0 else 5.0   # 1+s·X>0 を保証
    best_s, best_g = 0.0, 0.0
    for s in np.linspace(0.0, s_cap, 400)[1:]:
        g = np.mean(np.log1p(s * X))
        if g > best_g:
            best_g, best_s = g, s
    return best_s, best_g


def growth_ci(X, s, n_boot=2000, seed=42):
    """TEST 成長率 E[ln(1+s·X)] と bootstrap 95%CI（レース再標本化）。"""
    g = np.log1p(np.clip(s * X, -0.999999, None))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boot = g[idx].mean(axis=1)
    return g.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ---- value-bet sweep (全board trio・H-A 補強) -------------------------------

def value_sweep(train, test, iso, score_idx):
    """較正P×odds≥ev_min の全board trioを買う(Kellyでなく等額)。7+と同型の対照。"""
    def roi_at(races, ev_min):
        pays, bets = [], []
        for r in races:
            sc = np.array([tr[score_idx] for tr in r["board"]], dtype=float)
            p_cal = iso.transform(sc)
            o = np.array([tr[1] for tr in r["board"]], dtype=float)
            h = np.array([tr[4] for tr in r["board"]], dtype=float)
            mask = (p_cal * o) >= ev_min
            if not mask.any():
                continue
            pays.append(float((o[mask] * h[mask]).sum() * 100))
            bets.append(int(mask.sum()) * 100)
        return roi_summary(pays, bets), len(pays)
    print(f"\n  ▼ value-bet 対照(較正P×odds≥ev_min の全trio等額・7+と同型)")
    print(f"    {'ev_min':<8}{'TRAIN':<38}{'TEST(OOS)':<38}{'再現':>6}")
    for ev_min in [1.0, 1.1, 1.3, 1.5]:
        s1, n1 = roi_at(train, ev_min); s2, n2 = roi_at(test, ev_min)
        def fmt(s, n):
            return f"{n:>5}R {s['roi']:>5.0%} 的中{s['hit_rate']:>4.1%} [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]"
        flag = "★再現" if (s1["roi"] > 1 and s2["roi"] > 1 and n2 >= 30) else ("空" if n1 == 0 else "")
        print(f"    {ev_min:<8}{fmt(s1,n1):<38}{fmt(s2,n2):<38}{flag:>6}")


# ---- レポート ---------------------------------------------------------------

def report(train, test):
    print(f"\n{'='*100}")
    print(f"  ≤6車 較正確率×Kellyステーク傾斜  S/A trio 2軸流し  TR {len(train)}R / TE {len(test)}R  (model={MODEL})")
    print(f"{'='*100}")

    for score_idx, sname in [(2, "Plackett-Luce(条件付き同時)"), (3, "周辺積(独立近似)")]:
        print(f"\n{'─'*100}\n  ◆ 同時確率モデル: {sname}\n{'─'*100}")
        iso = fit_iso(train, score_idx)
        market_diag(test, iso, score_idx, f"{sname}・TEST")
        value_sweep(train, test, iso, score_idx)

        print(f"\n  ▼ 資金成長率比較（重み形状の優劣・レバレッジs*はTRAIN最適化→TEST OOS評価）")
        print(f"    {'手法':<16}{'s*':>7}{'TRAIN g':>10}{'TEST g':>10}{'TEST 95%CI':>22}"
              f"{'TEST ROI(unit)':>16}{'購入R':>7}")
        print(f"    {'-'*96}")
        for method in ["flat", "tilt", "kelly", "kelly_x_tilt"]:
            Xtr, _, _ = per_race_X(train, method, iso, score_idx)
            Xte, pte, bte = per_race_X(test, method, iso, score_idx)
            s_star, g_tr = opt_scale(Xtr)
            g_te, lo, hi = growth_ci(Xte, s_star)
            roi = roi_summary(pte, bte)
            # 成長率は1レースあたり対数成長。年率換算でなく相対比較用。
            print(f"    {method:<16}{s_star:>7.3f}{g_tr:>10.4f}{g_te:>10.4f}"
                  f"[{lo:>+8.4f},{hi:>+8.4f}]{roi['roi']:>13.0%}  {len(pte):>6}")
        print(f"    (g=E[ln m_r] 1レース対数成長。Kellyが最大化する目的。g大ほど資金成長速い。"
              f"flatを上回れば傾斜の価値あり)")

    print(f"\n{'='*100}")
    print("  判定:")
    print("   ・市場診断 slope≈1/corr→1/実的中≈市場示唆 ⇒ ≤6車でも value無し=H-A否定 → Kellyの源泉は選別(top3_sum)のみ。")
    print("   ・成長率 kelly>flat なら傾斜採用余地。kelly≈tilt なら離散top3_sum傾斜で十分(較正の手間不要)。")
    print("   ・kelly_x_tilt が最大なら『波乱帯ゲート × エッジ比例サイズ』が最良 → wave-picks-wt 実装候補。")


if __name__ == "__main__":
    print("collecting TRAIN (2023-07〜2026-02)...", flush=True)
    train = collect("2023-07-01", "2026-02-28")
    print(f"  TRAIN {len(train)} races", flush=True)
    print("collecting TEST (2026-03〜)...", flush=True)
    test = collect("2026-03-01", "2026-06-08")
    print(f"  TEST {len(test)} races", flush=True)
    report(train, test)
