"""≤6車: 展開シナリオ・ミクスチャ実験 — 複数展開の条件付き確率合成で組み合わせ確率は改善するか。

仮説:
  現行モデルは選手単位top3確率=「平均的な展開1本」。実レースは複数展開（逃げ切り/捲り/番手差し…）の
  混合で、真の組み合わせ確率分布は多峰性。シナリオ条件付き分解 P(combo)=Σ_s P(s|race)·P(combo|s) が
  単一モデルのPL同時確率より較正された確率を与えるなら、EV買い目・Kelly・ガミ判定すべてが底上げされる。

過去の壁（必読の前提）:
  - docs/analysis/09: per-combo value は≤6車でも較正P≈市場示唆=エッジ無しで DEFER。
    本実験が勝つ条件=「シナリオ分解が単一モデルにも市場にも無い情報を加える」こと。
  - docs/analysis/07/13: ライン明示ルール・is_lone_nige特徴は不採用。レース構造の明示利用は負けてきた。

シナリオ定義（≤6車・勝者の決まり手×ライン位置。wt_entries.factor）:
  nige=逃(33%) / makuri=捲(22%) / sashi_f=差・非ラインリーダー=番手差し(34%) / sashi_l=差・リーダー(11%)

ゲート方式（撤退ラインを先に切る）:
  Phase1: シナリオ4クラス分類器が base rate logloss を TEST で有意に下回るか。下回らなければ撤退。
  Phase2: ミクスチャP(combo) が同一条件プール単一モデルのPLを TEST trio logloss /
          的中組合せ -logP（オッズ非依存指標）で上回るか。上回らなければ #4 と同じ壁=撤退。
          ランダム分割ミクスチャ(コントロール)も併記=「展開情報」と「アンサンブル容量」の切り分け。
  Phase3: Phase2通過時のみ意味を持つ value ROIスイープ（TRAIN=TR-cal / TEST 両方>100%・bootstrap CI）。

公平性:
  本番 lgbm_wt は TEST 期間を学習に含む可能性があるためベースラインに使わない（参考表示のみ）。
  プール単一 vs シナリオ条件付き×4 vs ランダム分割×4 は同一ハイパラ・同一特徴・同一期間(TR-fit)で学習。
  較正(isotonic)は TR-cal（両アームとも学習外）で fit、判定は TEST。

期間: TR-fit 2023-01〜2025-08 (≤6車 ~5.4kR) / TR-cal 2025-09〜2026-02 (~800R) / TEST 2026-03〜06-10 (~510R)
払戻/オッズ=wt_odds最終=上限値（実運用は下振れ）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT)
from src.models.trainer import load_model
from src.database import get_connection
from exp_7plus_value_wt import pl_set_prob, load_trio_board, market_diag, value_roi
from roi_robustness_wt import roi_summary

SCN = ["nige", "makuri", "sashi_f", "sashi_l"]
FIT_END, CAL_END, TEST_END = "2025-08-31", "2026-02-28", "2026-06-10"
MIN_DATE = "2023-01-01"
RNG = np.random.RandomState(42)

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbose=-1)


# ──────────────────────────────────────────────────────────────────────
# シナリオラベル（勝者の決まり手×ライン位置）
# ──────────────────────────────────────────────────────────────────────
def load_scenario_labels() -> dict:
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, factor, is_line_leader FROM wt_entries "
            "WHERE finish_order=1").fetchall()
    lab = {}
    for rk, f, ill in rows:
        if f == "逃":
            lab[rk] = 0
        elif f == "捲":
            lab[rk] = 1
        elif f in ("差", "マ"):
            lab[rk] = 3 if ill else 2
    return lab


# ──────────────────────────────────────────────────────────────────────
# Phase1: レースレベル特徴 → シナリオ4クラス分類
# ──────────────────────────────────────────────────────────────────────
RACE_FEATS = [
    "n_entries", "n_lines", "n_senko", "n_ryo", "n_oikomi",
    "max_line_size", "score_std", "rp_gap12",
    "senko_best_z", "senko_best_rank",
    "mean_spurt", "mean_thrust", "sum_s", "sum_b", "max_b",
    "top_style", "top_is_leader", "top_line_size",
    "bank_length_enc", "is_indoor", "grade_enc",
]


def build_race_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("race_key")
    R = pd.DataFrame(index=g.size().index)
    R["n_entries"] = g.size()
    for c in ["n_lines", "n_senko", "score_std", "bank_length_enc",
              "is_indoor", "grade_enc"]:
        R[c] = g[c].first()
    R["n_ryo"] = g["style_enc"].apply(lambda s: int((s == 1).sum()))
    R["n_oikomi"] = g["style_enc"].apply(lambda s: int((s == 2).sum()))
    R["max_line_size"] = g["line_size"].max()
    R["rp_gap12"] = g["race_point"].apply(
        lambda s: float(np.diff(np.sort(s.values)[-2:])[0]) if len(s) > 1 else 0.0)
    sen = df[df["style_enc"] == 0]
    R["senko_best_z"] = sen.groupby("race_key")["score_z"].max().reindex(R.index).fillna(-3.0)
    R["senko_best_rank"] = sen.groupby("race_key")["score_rank"].min().reindex(R.index).fillna(7.0)
    R["mean_spurt"] = g["ex_spurt_pct"].mean()
    R["mean_thrust"] = g["ex_thrust_pct"].mean()
    R["sum_s"] = g["s_count"].sum()
    R["sum_b"] = g["b_count"].sum()
    R["max_b"] = g["b_count"].max()
    top = df[df["score_rank"] == 1].drop_duplicates("race_key").set_index("race_key")
    R["top_style"] = top["style_enc"].reindex(R.index).fillna(-1)
    R["top_is_leader"] = top["is_line_leader"].reindex(R.index).fillna(0)
    R["top_line_size"] = top["line_size"].reindex(R.index).fillna(1)
    return R.fillna(0)


def phase1(R: pd.DataFrame, scn: pd.Series, split: pd.Series):
    Xf, yf = R[split == "fit"], scn[split == "fit"]
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=4, **LGB_PARAMS)
    clf.fit(Xf[RACE_FEATS], yf)

    base_p = np.bincount(yf, minlength=4) / len(yf)   # TR-fit の周辺分布
    print(f"\n{'='*100}\n  Phase1: 展開シナリオ4クラス分類（{'/'.join(SCN)}）\n{'='*100}")
    print(f"  TR-fit 分布: " + "  ".join(f"{n}={p:.1%}" for n, p in zip(SCN, base_p)))
    ok = True
    for sp in ["cal", "test"]:
        X, y = R[split == sp], scn[split == sp].values
        proba = clf.predict_proba(X[RACE_FEATS])
        eps = 1e-9
        ll = -np.log(np.clip(proba[np.arange(len(y)), y], eps, 1)).mean()
        ll_base = -np.log(np.clip(base_p[y], eps, 1)).mean()
        acc = (proba.argmax(1) == y).mean()
        acc_base = base_p.max()
        print(f"  [{sp.upper():4}] {len(y):>4}R  logloss {ll:.4f} (base {ll_base:.4f}, "
              f"改善 {ll_base-ll:+.4f})  acc {acc:.1%} (base {acc_base:.1%})")
        if sp == "test" and ll >= ll_base:
            ok = False
    print(f"  → Phase1 {'通過: 展開は事前特徴から base rate 超えで予測可能' if ok else '不通過: 展開は事前に予測不能 → 撤退'}")
    return clf, ok


# ──────────────────────────────────────────────────────────────────────
# Phase2: 条件付きモデル×ミクスチャ vs プール単一（同一条件）
# ──────────────────────────────────────────────────────────────────────
def train_binary(df_rows: pd.DataFrame) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    m.fit(prepare_X(df_rows), df_rows["top3_flag"])
    return m


def collect_races(df, scn_map, clf_scn, R, board, strength_cols):
    """race毎に board上の全trioの (odds, pl per strength-set, hit) を組む。"""
    races = []
    proba_all = pd.DataFrame(
        clf_scn.predict_proba(R[RACE_FEATS]), index=R.index, columns=range(4))
    for rk, g in df.groupby("race_key"):
        bd = board.get(rk)
        if not bd or rk not in scn_map or rk not in proba_all.index:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        win = frozenset(int(x) for x in fin["frame_no"])
        if win not in bd:
            continue
        strengths = {}
        for name, col in strength_cols.items():
            strengths[name] = {int(fr): max(float(p), 1e-6)
                               for fr, p in zip(g["frame_no"], g[col])}
        p_scn = proba_all.loc[rk].values
        trios = []
        for combo, odds in bd.items():
            if len(combo) != 3 or not combo.issubset(strengths["pool"].keys()):
                continue
            pls = {}
            pls["pool"] = pl_set_prob(strengths["pool"], combo)
            pls["mix"] = sum(p_scn[k] * pl_set_prob(strengths[f"s{k}"], combo)
                             for k in range(4))
            pls["rnd"] = sum(p_scn_rnd * pl_set_prob(strengths[f"r{k}"], combo)
                             for k, p_scn_rnd in enumerate(RND_W))
            pls["prod"] = pl_set_prob(strengths["prod"], combo)
            trios.append((combo, float(odds), pls, 1 if combo == win else 0))
        if trios:
            races.append({"rk": rk, "trios": trios, "win": win,
                          "date": g["race_date"].iloc[0]})
    return races


def eval_arm(races_cal, races_test, key, label):
    """isotonic を TR-cal で fit → TEST の trio logloss / 的中-logP / Brier。"""
    x_cal = np.array([t[2][key] for r in races_cal for t in r["trios"]])
    y_cal = np.array([t[3] for r in races_cal for t in r["trios"]], dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x_cal, y_cal)

    eps = 1e-6
    out = {"label": label, "iso": iso}
    for sp, races in [("cal", races_cal), ("test", races_test)]:
        p = np.clip(iso.transform(
            np.array([t[2][key] for r in races for t in r["trios"]])), eps, 1 - eps)
        y = np.array([t[3] for r in races for t in r["trios"]], dtype=float)
        out[f"{sp}_ll"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out[f"{sp}_brier"] = float(((p - y) ** 2).mean())
        # 的中組合せの -logP（レース内で正規化した多項logloss）
        nll, pos = [], 0
        for r in races:
            m = len(r["trios"])
            pr = p[pos:pos + m]
            hit = y[pos:pos + m]
            pos += m
            pn = pr / pr.sum() if pr.sum() > 0 else np.full(m, 1 / m)
            nll.append(-np.log(max(float(pn[hit == 1][0]), eps)) if hit.any() else np.nan)
        out[f"{sp}_race_nll"] = float(np.nanmean(nll))
        out[f"{sp}_nll_arr"] = np.array(nll, dtype=float)
    return out


def market_arm(races_cal, races_test):
    """市場示唆確率（1/odds をレース内 overround 正規化）の同指標。"""
    eps = 1e-6
    out = {"label": "市場示唆 (1/odds 正規化)"}
    for sp, races in [("cal", races_cal), ("test", races_test)]:
        ps, ys, nll = [], [], []
        for r in races:
            odds = np.array([t[1] for t in r["trios"]])
            hit = np.array([t[3] for t in r["trios"]], dtype=float)
            pm = (1 / odds) / (1 / odds).sum()
            ps.append(pm); ys.append(hit)
            nll.append(-np.log(max(float(pm[hit == 1][0]), eps)) if hit.any() else np.nan)
        p = np.clip(np.concatenate(ps), eps, 1 - eps)
        y = np.concatenate(ys)
        out[f"{sp}_ll"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out[f"{sp}_brier"] = float(((p - y) ** 2).mean())
        out[f"{sp}_race_nll"] = float(np.nanmean(nll))
        out[f"{sp}_nll_arr"] = np.array(nll, dtype=float)
    return out


def paired_diff(a, b, la, lb):
    """同一レース対の -logP 差のbootstrap CI（負=前者が良い）。"""
    d = a - b
    d = d[~np.isnan(d)]
    boots = np.array([d[RNG.randint(0, len(d), len(d))].mean() for _ in range(4000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig = (hi < 0) or (lo > 0)
    print(f"    {la} − {lb:<24}: Δ-logP {d.mean():+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]"
          f"  → {'有意' if sig else '有意でない（ノイズ範囲）'}")
    return sig


def build_cal_list(races, iso, key):
    """exp_7plus の market_diag / value_roi 形式 (odds, p_cal, hit, p_market)。"""
    out = []
    for r in races:
        odds = np.array([t[1] for t in r["trios"]])
        hit = np.array([t[3] for t in r["trios"]], dtype=float)
        p = iso.transform(np.array([t[2][key] for t in r["trios"]]))
        ovr = (1 / odds).sum()
        out.append((odds, p, hit, (1 / odds) / ovr))
    return out


# ──────────────────────────────────────────────────────────────────────
RND_W = None  # ランダム分割の混合重み（グループサイズ比・main で設定）

def main():
    global RND_W
    print("loading & building features (2023-01〜2026-06-10, 全車→≤6車抽出)...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=MIN_DATE, max_date=TEST_END))
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz <= 6].index)].copy()
    df = df[df["finish_order"] >= 1].copy()        # 結果確定レースのみ・欠車除外

    scn_map = load_scenario_labels()
    df = df[df["race_key"].isin(scn_map.keys())].copy()
    df["scn"] = df["race_key"].map(scn_map)
    df["split"] = np.where(df["race_date"] <= FIT_END, "fit",
                  np.where(df["race_date"] <= CAL_END, "cal", "test"))
    n_r = df.groupby("split")["race_key"].nunique()
    print(f"  races: fit={n_r.get('fit',0)} cal={n_r.get('cal',0)} test={n_r.get('test',0)}")

    # ── Phase1
    R = build_race_features(df)
    meta = df.groupby("race_key")[["scn", "split"]].first()
    R = R.loc[meta.index]
    clf_scn, ok1 = phase1(R, meta["scn"], meta["split"])
    if not ok1:
        print("\n  Phase1 不通過のため Phase2 以降は参考値（撤退前提で続行表示）。")

    # ── Phase2: モデル学習（同一条件）
    fit_rows = df[df["split"] == "fit"]
    print(f"\n{'='*100}\n  Phase2: ミクスチャ vs プール単一（同一ハイパラ・同一特徴・TR-fit学習）\n{'='*100}")
    print("  training pooled / 4 conditional / 4 random-control models...", flush=True)
    m_pool = train_binary(fit_rows)
    cond, rnd = {}, {}
    sizes = []
    for k in range(4):
        sub = fit_rows[fit_rows["scn"] == k]
        sizes.append(len(sub))
        cond[k] = train_binary(sub)
    # コントロール: 同サイズのランダムレース分割（展開情報なしのアンサンブル）
    rks = fit_rows["race_key"].unique()
    perm = RNG.permutation(rks)
    race_sizes = [fit_rows[fit_rows["scn"] == k]["race_key"].nunique() for k in range(4)]
    bounds = np.cumsum(race_sizes)[:-1]
    groups = np.split(perm, bounds)
    for k in range(4):
        rnd[k] = train_binary(fit_rows[fit_rows["race_key"].isin(groups[k])])
    RND_W = np.array(race_sizes, dtype=float) / sum(race_sizes)
    print("  cond rows: " + "  ".join(f"{SCN[k]}={sizes[k]}" for k in range(4)))

    # 予測列（全行・各強度セット）
    X_all = prepare_X(df)
    df["p_pool"] = m_pool.predict_proba(X_all)[:, 1]
    for k in range(4):
        df[f"p_s{k}"] = cond[k].predict_proba(X_all)[:, 1]
        df[f"p_r{k}"] = rnd[k].predict_proba(X_all)[:, 1]
    prod_model = load_model("lgbm_wt")               # 参考: 本番モデル(期間リーク可能性あり)
    df["p_prod"] = prod_model.predict_proba(X_all)[:, 1]

    # trio盤面と race 構造
    print("  building trio boards & PL probabilities...", flush=True)
    strength_cols = {"pool": "p_pool", "prod": "p_prod",
                     **{f"s{k}": f"p_s{k}" for k in range(4)},
                     **{f"r{k}": f"p_r{k}" for k in range(4)}}
    board = load_trio_board(df["race_key"].unique().tolist())
    races_cal = collect_races(df[df["split"] == "cal"], scn_map, clf_scn, R, board, strength_cols)
    races_test = collect_races(df[df["split"] == "test"], scn_map, clf_scn, R, board, strength_cols)
    print(f"  cal {len(races_cal)}R / test {len(races_test)}R (trio盤面・的中オッズあり)")

    arms = [
        eval_arm(races_cal, races_test, "pool", "プール単一 PL（公平ベースライン）"),
        eval_arm(races_cal, races_test, "mix",  "シナリオ・ミクスチャ Σ P(s)·PL_s"),
        eval_arm(races_cal, races_test, "rnd",  "ランダム分割ミクスチャ（コントロール）"),
        eval_arm(races_cal, races_test, "prod", "本番 lgbm_wt PL（参考・リーク可能性）"),
    ]
    mkt = market_arm(races_cal, races_test)

    print(f"\n  ▼ 較正後の確率品質（isotonic は TR-cal で fit・判定は TEST・オッズ非依存）")
    hdr = f"    {'アーム':<38}{'TEST trio-logloss':>18}{'TEST race -logP':>16}{'TEST Brier':>12}"
    print(hdr); print("    " + "─" * (len(hdr)))
    for a in arms + [mkt]:
        print(f"    {a['label']:<38}{a['test_ll']:>18.5f}{a['test_race_nll']:>16.4f}{a['test_brier']:>12.6f}")

    pool_a = arms[0]; mix_a = arms[1]; rnd_a = arms[2]; prod_a = arms[3]
    print(f"\n  ▼ 対比較の有意性（TEST・同一レース対の -logP 差・bootstrap 4000）")
    sig_pool = paired_diff(mix_a["test_nll_arr"], pool_a["test_nll_arr"],
                           "ミクスチャ", "プール単一")
    sig_rnd = paired_diff(mix_a["test_nll_arr"], rnd_a["test_nll_arr"],
                          "ミクスチャ", "ランダム分割(容量補正)")
    paired_diff(prod_a["test_nll_arr"], mkt["test_nll_arr"],
                "本番lgbm_wt", "市場示唆")
    gate2 = sig_pool and sig_rnd
    print(f"\n  → Phase2 {'通過' if gate2 else '不通過'}: "
          f"改善が{'有意かつ展開情報由来' if gate2 else '有意でない/容量で説明可能'}"
          f" (mix {mix_a['test_race_nll']:.4f} / pool {pool_a['test_race_nll']:.4f} / "
          f"rnd {rnd_a['test_race_nll']:.4f} / 市場 {mkt['test_race_nll']:.4f})")

    # ── Phase3: value ROI（Phase2不通過なら参考値）
    print(f"\n{'='*100}\n  Phase3: value-bet ROI スイープ（EV=P_cal×odds・最終オッズ上限値）"
          f"{'' if gate2 else '  ※Phase2不通過のため参考値'}\n{'='*100}")
    for a, key in [(pool_a, "pool"), (mix_a, "mix")]:
        cal_c = build_cal_list(races_cal, a["iso"], key)
        cal_t = build_cal_list(races_test, a["iso"], key)
        market_diag(cal_t, f"{a['label']}・TEST")
        print(f"\n    {'ev_min':<8}{'TR-cal':<40}{'TEST':<40}{'再現':>6}")
        for ev_min in [1.0, 1.1, 1.2, 1.5, 2.0]:
            s1, n1 = value_roi(cal_c, ev_min)
            s2, n2 = value_roi(cal_t, ev_min)
            def f(s, n):
                return (f"{n:>5}R {s['roi']:>5.0%} 的中{s['hit_rate']:>4.0%} "
                        f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]")
            flag = "★再現" if (s1["roi"] > 1.0 and s2["roi"] > 1.0 and n2 >= 30) else \
                   ("空" if n1 == 0 else ("小標本" if n2 < 30 else ""))
            print(f"    {ev_min:<8}{f(s1,n1):<40}{f(s2,n2):<40}{flag:>6}")

    print(f"\n{'='*100}")
    print(f"  総合判定: Phase1={'通過' if ok1 else '不通過'} / Phase2={'通過' if gate2 else '不通過'}")
    if not gate2:
        print("  → 撤退基準該当: シナリオ分解の改善は有意でない/容量で説明可能（#4 と同型の壁）。")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
