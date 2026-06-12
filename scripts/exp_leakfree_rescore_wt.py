"""既存レバーのリーク無し再採点 — 本番モデルの週次再学習リークによる上振れの定量化。

背景（docs/analysis/17 の副発見）:
  本番 lgbm_wt は週次再学習で「バックテスト評価期間」を学習済み＝過去の全バックテスト数字に
  リーク上振れが乗っている可能性。本実験は同一レース・同一レバー定義で
    arm A: リーク無しモデル（TRAIN期間 2023-07〜2025-06 のみで学習・VAL/HOLDOUTは真に未知）
    arm B: 本番 lgbm_wt（全期間学習済み）
  を並走させ、レバーごとに「上振れ幅 = B − A」を期間別に測る。

再採点対象（元の分析の定義に忠実・各armのモデル予測でランキング/層/シグナルを再計算）:
  C0 現行3層戦略: _assign_tier(SS/S/A)・SS=3連単p1→p2→x 3点 / S,A=3連複3点・3点最安≥5倍(ガミ帯)
  C1 C0 ∩ 波乱ゲート: top3_sum ≤ TRAIN四分位Q1（armごとにTRAINで閾値固定・roadmap A）
  C2 C0 ∩ fav_mismatch: モデル1位 ≠ 市場本命(trio盤面逆算)（docs/analysis/13）
  C3 中間オッズ[20,80]: 2軸流し全点(n-2点)のうちtrioオッズ∈[20,80]のみ・全≤6車（docs/analysis/06）
  C4 同[10,80]

読み方:
  - arm B の TRAIN/VAL/HOLDOUT は全て in-sample（従来のバックテストと同じ条件）
  - arm A の VAL/HOLDOUT が「リーク無しの実力値」。B−A が上振れ幅。
  - 実運用はさらに朝→確定オッズドリフトで下振れ（本表は最終オッズ上限値）。
期間: TRAIN 2023-07〜2025-06 / VAL 2025-07〜2026-02 / HOLDOUT 2026-03〜2026-06-12。≤6車・n≥4。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import lightgbm as lgb

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _assign_tier
from exp_segment_first_wt import load_boards, market_fav, LGB_PARAMS, TRAIN, VAL, HOLD
from roi_robustness_wt import roi_summary

ARMS = ["free", "prod"]


def collect():
    print("loading & building features (2023-07〜2026-06-12, 全車)...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    print(f"  arm A: fresh LGBM on TRAIN only ({len(fit):,} rows)...", flush=True)
    m_free = lgb.LGBMClassifier(**LGB_PARAMS)
    m_free.fit(prepare_X(fit), fit["top3_flag"])
    m_prod = load_model("lgbm_wt")

    X = prepare_X(df)
    df["p_free"] = m_free.predict_proba(X)[:, 1]
    df["p_prod"] = m_prod.predict_proba(X)[:, 1]

    # 本番忠実: 出走表(エントリー)基準で≤6車を判定し、ランキングも全エントリーで行う。
    # 旧バックテストは「完走者≤6人」(7車立て混入33%)＋欠車事後除外の生存バイアスがあった。
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz <= 6].index)].copy()
    done = df.groupby("race_key")["finish_order"].apply(
        lambda s: (s >= 1).sum() >= 3)               # 結果確定レースのみ
    df = df[df["race_key"].isin(done[done].index)]
    trio_b, tf_b, _ = load_boards(df["race_key"].unique().tolist())

    races = []
    for rk, g0 in df.groupby("race_key"):
        n = len(g0)                                   # エントリー数（欠車含む・本番と同一）
        if n < 4:
            continue
        bd = trio_b.get(rk, {})
        if not bd:
            continue
        fin = g0[g0["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        dns = set(g0[g0["finish_order"] == 0]["frame_no"].astype(int).tolist())  # 欠車
        mf = market_fav(bd)
        tfb = tf_b.get(rk, {})
        race = {"date": g0["race_date"].iloc[0]}
        for arm, col in (("free", "p_free"), ("prod", "p_prod")):
            g = g0.sort_values(col, ascending=False)  # 全エントリーでランキング（事前情報のみ）
            p = g[col].tolist()
            fr = g["frame_no"].astype(int).tolist()
            p1, p2, thirds = fr[0], fr[1], fr[2:5]
            tier = _assign_tier(p[0] - p[1], p[0] / (3.0 / n))
            # 欠車の返還処理（notify_results_wt._void_by_dns と同一規則）:
            #   軸(p1/p2)欠車 → レース無効(賭け不成立)。相手(third)欠車 → その点のみ除外。
            axis_void = (p1 in dns) or (p2 in dns)
            trio3, tf3, tf6 = [], [], []
            if not axis_void:
                for x in thirds:
                    if x in dns:
                        continue
                    c = frozenset((p1, p2, x))
                    if c in bd:
                        trio3.append((bd[c], c == top3))
                    o = tfb.get((p1, p2, x))
                    if o:
                        tf3.append((o, order == (p1, p2, x)))
                    for a, b in ((p1, p2), (p2, p1)):     # 1-2着BOX6点（本番SS）
                        ob = tfb.get((a, b, x))
                        if ob:
                            tf6.append((ob, order == (a, b, x)))
            widepts = []
            if not axis_void:
                for x in fr[2:]:
                    if x in dns:
                        continue
                    c = frozenset((p1, p2, x))
                    if c in bd:
                        widepts.append((bd[c], c == top3))
            race[arm] = {
                "tier": tier,
                "t3s": p[0] + p[1] + p[2],
                "mismatch": (mf != p1) if mf is not None else None,
                "trio3": trio3, "tf3": tf3, "tf6": tf6, "widepts": widepts,
                "min3": min((o for o, _ in (tf3 if tier == "SS" else trio3)), default=None),
            }
        races.append(race)
    return races


def base_legs(a):
    """C0: tier成立・該当賭式3点・最安≥5倍。不成立はNone。"""
    if a["tier"] is None:
        return None
    legs = a["tf3"] if a["tier"] == "SS" else a["trio3"]
    if not legs or a["min3"] is None or a["min3"] < 5.0:
        return None
    return legs


def make_cells(q1):
    def c0(a):
        return base_legs(a)

    def c1(a):
        legs = base_legs(a)
        return legs if (legs and a["t3s"] <= q1[id(a)]) else None

    def c2(a):
        legs = base_legs(a)
        return legs if (legs and a["mismatch"] is True) else None

    def c3(a):
        sub = [(o, h) for o, h in a["widepts"] if 20 <= o < 80]
        return sub or None

    def c4(a):
        sub = [(o, h) for o, h in a["widepts"] if 10 <= o < 80]
        return sub or None

    return [("C0 現行3層×ガミ≥5", c0), ("C1 +波乱Q1ゲート", c1),
            ("C2 +fav_mismatch", c2), ("C3 中間[20,80]全R", c3),
            ("C4 中間[10,80]全R", c4)]


def cell_roi(races, arm, fn):
    pays, bets = [], []
    for r in races:
        legs = fn(r[arm])
        if not legs:
            continue
        pays.append(sum(o * 100 for o, hit in legs if hit))
        bets.append(len(legs) * 100)
    return roi_summary(pays, bets), len(pays)


def fmt(s, n):
    if n == 0:
        return f"{0:>4}R  --"
    return f"{n:>4}R {s['roi']:>5.0%} [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}] 除{s['roi_ex_max']:>4.0%}"


def main():
    races = collect()
    by = {"TRAIN": [r for r in races if r["date"] <= TRAIN[1]],
          "VAL":   [r for r in races if TRAIN[1] < r["date"] <= VAL[1]],
          "HOLD":  [r for r in races if r["date"] > VAL[1]]}
    print(f"  races: TRAIN {len(by['TRAIN'])} / VAL {len(by['VAL'])} / HOLDOUT {len(by['HOLD'])}")

    # 波乱ゲート閾値: armごとにTRAIN期間のtier成立レースで四分位Q1を固定
    q1 = {}
    q1_val = {}
    for arm in ARMS:
        t3s = [r[arm]["t3s"] for r in by["TRAIN"] if r[arm]["tier"] is not None]
        q1_val[arm] = float(np.percentile(t3s, 25))
    print(f"  top3_sum Q1閾値: free={q1_val['free']:.3f} / prod={q1_val['prod']:.3f}")

    # id(a)で引けるよう全レースのarm dictに閾値を展開
    q1_map = {}
    for r in races:
        for arm in ARMS:
            q1_map[id(r[arm])] = q1_val[arm]
    cells = make_cells(q1_map)

    print(f"\n{'='*120}")
    print(f"  リーク無し再採点  arm A=free(TRAIN限定学習・VAL/HOLDは真のOOS) vs arm B=prod(lgbm_wt・全期間in-sample)")
    print(f"  払戻=最終オッズ上限値。リーク上振れ = B − A（VAL/HOLD列で見る）")
    print(f"{'='*120}")
    for name, fn in cells:
        print(f"\n  ◆ {name}")
        print(f"    {'arm':<6}{'TRAIN':<40}{'VAL':<40}{'HOLDOUT':<40}")
        res = {}
        for arm, lab in (("free", "A:無"), ("prod", "B:本番")):
            cols = []
            for per in ("TRAIN", "VAL", "HOLD"):
                s, nn = cell_roi(by[per], arm, fn)
                res[(arm, per)] = (s, nn)
                cols.append(fmt(s, nn))
            print(f"    {lab:<6}{cols[0]:<40}{cols[1]:<40}{cols[2]:<40}")
        for per in ("VAL", "HOLD"):
            sa, na = res[("free", per)]
            sb, nb = res[("prod", per)]
            if na and nb:
                print(f"      → {per}: 上振れ {sb['roi']-sa['roi']:+.0%}"
                      f"  (B {sb['roi']:.0%} − A {sa['roi']:.0%})")

    print(f"\n{'='*120}")
    print("  判定: arm A の VAL/HOLDOUT がレバーの実力値（さらに朝ドリフトで下振れ前提）。")
    print("        A で >100% を維持できないレバーは、過去数字がリーク上振れだった可能性が高い。")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
