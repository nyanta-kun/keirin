"""条件先行（segment-first）アプローチの全面再検討 — 3期間プロトコル。

ユーザーの問い:
  現行は「予想ありき→条件で絞る」。逆に「初めから条件（レース属性）でセグメントを分け、
  セグメントごとに予想アプローチを変える」方式でエッジは無いか、フラットに検討する。
  合格基準 = TRAIN>100% かつ VAL>100% かつ 完全独立HOLDOUTで>100%。

プロトコル（事前登録・撤退ライン先決め）:
  - モデル: 本番lgbm_wtは週次再学習でHOLDOUT期間を学習済み（リーク）のため使わない。
    TRAIN期間のみで新規学習したLGBM（同特徴・同パラメタ）を全期間の予測に使用。
  - 期間: TRAIN 2023-07〜2025-06 / VAL 2025-07〜2026-02（セル選別はここまでで完結）
          HOLDOUT 2026-03〜2026-06-12（選別に一切使わず、生存セルのみ最後に1回評価）
  - セル = セグメント(16) × アプローチ(5) = 80。多重比較は開示し、
    選別 = TRAIN ROI>100% & VAL ROI>100% & VAL n≥30 & VAL 最大払戻除去ROI>100%。
  - 注意: アプローチ自体（≥5帯・BOX6・top3_sum・W12・fav_mismatch）は過去分析の知見であり、
    その意味で2026-03〜06は「人間側の選択」を通じ部分的に既知。真の最終確認は前向きlive。

セグメント（レース属性・モデル出力を使わない条件）:
  ALL / 車立て(5,6) / 発走時間帯(〜10時=モーニング,10-16,16-20,20時〜=ミッドナイト)
  / グレード(A級,S級,L級=ガールズ) / 開催日次(1,2,3+) / ライン数(≤2,3,≥4)

アプローチ（買い方メニュー・各レースへの適用条件込み）:
  A1 trio-val   : 3連複2軸流し3点・3点の最安オッズ≥5倍（現行推奨帯）
  A2 tf-box6    : 3連単 pred1,pred2 1-2着BOX→3-5位 6点・無条件
  A3 trio-upset : 3連複3点・top3_sum ≤ TRAIN四分位Q1（波乱ゲート）
  A4 wide-val   : 指数1-2位ワイド1点・オッズ≥2.5倍
  A5 trio-favmis: 3連複3点・モデル1位≠市場本命(trio盤面逆算)のみ

払戻=wt_odds最終オッズ=上限値（朝→確定ドリフトで実運用は下振れ）。≤6車・n≥4・結果確定のみ。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
from collections import defaultdict
import numpy as np
import lightgbm as lgb

from src.preprocessing.feature_wt import (
    load_raw_data_wt, build_features_wt, prepare_X, FEATURE_COLS_WT)
from src.database import get_connection
from roi_robustness_wt import roi_summary

TRAIN = ("2023-07-01", "2025-06-30")
VAL   = ("2025-07-01", "2026-02-28")
HOLD  = ("2026-03-01", "2026-06-12")

LGB_PARAMS = dict(objective="binary", n_estimators=500, learning_rate=0.05,
                  num_leaves=31, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, random_state=42, verbose=-1)


# ──────────────────────────────────────────────────────────────────────
def load_boards(race_keys):
    """trio / trifecta / quinellaPlace(ワイド) の盤面を一括ロード。"""
    trio, tf, wd = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    CH = 900
    with get_connection() as c:
        for i in range(0, len(race_keys), CH):
            chunk = race_keys[i:i + CH]
            ph = ",".join("?" * len(chunk))
            for rk, bt, comb, ov in c.execute(
                    f"SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                    f"WHERE bet_type IN ('trio','trifecta','quinellaPlace') "
                    f"AND race_key IN ({ph})", chunk):
                if ov is None or ov <= 0:
                    continue
                try:
                    fr = [int(x) for x in re.split(r"[-=]", str(comb))]
                except ValueError:
                    continue
                if bt == "trio" and len(fr) == 3:
                    trio[rk][frozenset(fr)] = float(ov)
                elif bt == "trifecta" and len(fr) == 3:
                    tf[rk][tuple(fr)] = float(ov)
                elif bt == "quinellaPlace" and len(fr) == 2:
                    wd[rk][frozenset(fr)] = float(ov)
    return trio, tf, wd


def market_fav(trio_board):
    """trio盤面から市場P(top3)を逆算し、市場本命frameを返す（盤面不足はNone）。"""
    q = {}
    n_combo = 0
    for combo, ov in trio_board.items():
        if ov >= 9000:
            continue
        n_combo += 1
        for f in combo:
            q[f] = q.get(f, 0.0) + 1.0 / ov
    if n_combo < 4 or not q:
        return None
    return max(q.items(), key=lambda x: x[1])[0]


def collect():
    """全期間を一括ロード→TRAIN期間のみでモデル学習→全レース構造体を構築。"""
    print("loading & building features (2023-07〜2026-06-12, 全車)...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    # TRAIN期間のみで新規学習（HOLDOUTリーク排除）。学習はプール（全車立て・本番同型）
    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    print(f"  training fresh LGBM on TRAIN period only ({len(fit):,} rows)...", flush=True)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(prepare_X(fit), fit["top3_flag"])
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]

    # ≤6車・結果確定のみ
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz <= 6].index)]
    df = df[df["finish_order"] >= 1].copy()

    with get_connection() as c:
        day_idx = dict(c.execute("SELECT race_key, day_index FROM wt_races"))
    trio_b, tf_b, wd_b = load_boards(df["race_key"].unique().tolist())

    races = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 4:
            continue
        bd = trio_b.get(rk, {})
        if not bd:
            continue
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        top3 = frozenset(fin["frame_no"].astype(int).tolist())
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        p1, p2, thirds = fr[0], fr[1], fr[2:5]
        trio3 = [(bd[frozenset((p1, p2, x))], frozenset((p1, p2, x)) == top3)
                 for x in thirds if frozenset((p1, p2, x)) in bd]
        tfb = tf_b.get(rk, {})
        tf6 = []
        for a, b in ((p1, p2), (p2, p1)):
            for x in thirds:
                o = tfb.get((a, b, x))
                if o:
                    tf6.append((o, order == (a, b, x)))
        w12 = wd_b.get(rk, {}).get(frozenset((p1, p2)))
        mf = market_fav(bd)
        sa = g["start_at"].iloc[0]
        hour = int((int(sa) + 9 * 3600) // 3600 % 24) if sa == sa and sa is not None else -1
        races.append({
            "date": g["race_date"].iloc[0], "n": n, "hour": hour,
            "grade": g["grade"].iloc[0] or "?", "day": day_idx.get(rk, 0),
            "n_lines": int(g["n_lines"].iloc[0]),
            "t3s": p[0] + p[1] + p[2],
            "same_fav": (mf == p1) if mf is not None else None,
            "trio3": trio3, "tf6": tf6,
            "w12": (w12, (p1 in top3) and (p2 in top3)) if w12 else None,
        })
    return races


# ──────────────────────────────────────────────────────────────────────
def seg_of(r):
    segs = ["ALL", f"n={r['n']}"]
    h = r["hour"]
    segs.append("時間:~10" if 0 <= h < 10 else "時間:10-16" if h < 16 else
                "時間:16-20" if h < 20 else "時間:20~" if h >= 20 else "時間:?")
    segs.append(f"grade:{r['grade']}")
    segs.append(f"day:{min(r['day'], 3)}" if r["day"] >= 1 else "day:?")
    nl = r["n_lines"]
    segs.append("lines:<=2" if nl <= 2 else "lines:3" if nl == 3 else "lines:>=4")
    return segs


def make_approaches(t3s_q1):
    def a1(r):
        if not r["trio3"] or min(o for o, _ in r["trio3"]) < 5.0:
            return None
        return r["trio3"]

    def a2(r):
        return r["tf6"] or None

    def a3(r):
        if r["t3s"] > t3s_q1 or not r["trio3"]:
            return None
        return r["trio3"]

    def a4(r):
        if r["w12"] is None or r["w12"][0] < 2.5:
            return None
        return [r["w12"]]

    def a5(r):
        if r["same_fav"] is not False or not r["trio3"]:
            return None
        return r["trio3"]

    return [("A1 trio最安>=5", a1), ("A2 3連単BOX6", a2),
            ("A3 trio波乱Q1", a3), ("A4 ワイドval", a4), ("A5 trio市場不一致", a5)]


def cell(races, fn):
    pays, bets, dates = [], [], set()
    for r in races:
        legs = fn(r)
        if not legs:
            continue
        pays.append(sum(o * 100 for o, hit in legs if hit))
        bets.append(len(legs) * 100)
        dates.add(r["date"])
    return roi_summary(pays, bets), len(pays), len(dates)


def fmt(s, n):
    if n == 0:
        return f"{0:>4}R  --"
    return (f"{n:>4}R {s['roi']:>5.0%} 的{s['hit_rate']:>4.0%} "
            f"[{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}] 除{s['roi_ex_max']:>4.0%}")


def main():
    races = collect()
    by = {"TR": [r for r in races if r["date"] <= TRAIN[1]],
          "VA": [r for r in races if TRAIN[1] < r["date"] <= VAL[1]],
          "HO": [r for r in races if r["date"] > VAL[1]]}
    print(f"  races: TRAIN {len(by['TR'])} / VAL {len(by['VA'])} / HOLDOUT {len(by['HO'])}")

    t3s_q1 = float(np.percentile([r["t3s"] for r in by["TR"]], 25))
    print(f"  top3_sum Q1 (TRAIN固定閾値) = {t3s_q1:.3f}")
    approaches = make_approaches(t3s_q1)

    seg_index = defaultdict(lambda: {k: [] for k in by})
    for k, rs in by.items():
        for r in rs:
            for s in seg_of(r):
                seg_index[s][k].append(r)
    seg_order = sorted(seg_index.keys(), key=lambda s: (s != "ALL", s))

    print(f"\n{'='*120}")
    print(f"  条件先行スキャン: セグメント×アプローチ（TRAIN/VALのみ・HOLDOUTは生存セルだけ最後に1回）")
    print(f"  セル数 {len(seg_order)*len(approaches)}（多重比較注意）  "
          f"選別=TRAIN>100% & VAL>100% & VALn≥30 & VAL除最大>100%")
    print(f"{'='*120}")

    survivors = []
    for s in seg_order:
        d = seg_index[s]
        print(f"\n  ◆ セグメント {s}  (TR {len(d['TR'])}R / VA {len(d['VA'])}R)")
        print(f"    {'アプローチ':<16}{'TRAIN':<42}{'VAL':<42}{'判定':>6}")
        for name, fn in approaches:
            s1, n1, _ = cell(d["TR"], fn)
            s2, n2, _ = cell(d["VA"], fn)
            ok = (n1 > 0 and n2 >= 30 and s1["roi"] > 1.0 and s2["roi"] > 1.0
                  and s2["roi_ex_max"] > 1.0)
            flag = "→HO" if ok else ("小標本" if 0 < n2 < 30 else "")
            print(f"    {name:<16}{fmt(s1,n1):<42}{fmt(s2,n2):<42}{flag:>6}")
            if ok:
                survivors.append((s, name, fn))

    print(f"\n{'='*120}")
    print(f"  ★ HOLDOUT 一発評価（2026-03〜06-12・選別不使用期間）  生存セル {len(survivors)}個")
    print(f"  ※開示: {len(seg_order)*len(approaches)}セル走査のため偶然の生存が混入しうる。"
          f"HOLDOUT>100%かつ除最大>100%のみ合格。最終確認は前向きlive。")
    print(f"{'='*120}")
    if not survivors:
        print("  生存セルなし（TRAIN/VAL両立の時点で全滅）")
    for s, name, fn in survivors:
        sh, nh, dh = cell(seg_index[s]["HO"], fn)
        verdict = "✅合格" if (nh >= 20 and sh["roi"] > 1.0 and sh["roi_ex_max"] > 1.0) else \
                  ("△ROI>100だが除最大<100" if (nh >= 20 and sh["roi"] > 1.0) else
                   ("小標本" if nh < 20 else "✗"))
        print(f"  {s:<14}{name:<16}{fmt(sh,nh):<44}購入日{dh:>4}日  {verdict}")


if __name__ == "__main__":
    main()
