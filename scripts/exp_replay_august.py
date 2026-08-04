"""新モデル・3ヘッド軸選定で 2026-08-01〜04 を再現し、実際の推奨と比較する（2026-08-04）。

ユーザー要望:
  ①「このモデルにおいて 2026-08-01〜04 の算出をした場合、推奨レース・買い目・結果・
      的中率・ROI がどうなるか確認できますか？」
  ②「〜2026-07-31 で3ヘッド軸選定の学習を行い、8月のレースにおいて推奨レース・
      買い目の選出・結果と付き合わせ、件数・ROI など集計して下さい」

## ⚠️ in-sample を避けるための設計

本番モデル（lgbm_wt / lgbm_wt_win / lgbm_wt_bad）は **full-refit で 8/1〜8/4 を
学習データに含む**ため、そのまま予測させると in-sample 評価になり過大評価になる。
（CLAUDE.md「バックフィルした過去分は in-sample」と同型の罠）

そこで **2026-07-31 までで学習した重み**だけを使う:
    p3   = lgbm_wt_eval_m2608   月次凍結 vintage（train 2022-12-01〜2026-07-31）
    pw   = lgbm_wt_win_m2608    同上
    pbad = 本スクリプト内で学習   train 2022-12-01〜2026-07-31・target bad6
           （`train-wt --target` は top3/win のみで bad の vintage を作れないため。
             exp_three_head_axis.py と同じ 5seed アンサンブルで学習し
             data/exp_cache/ にキャッシュする）

## 比較する案

  A 実績（picks_history）  … 当日 48特徴＋現行軸選定で実際に出した推奨（実精算）
  A' 実績と同一の買い目を最終オッズで再計算（B/C と同じオッズ源に揃えた対照）
  B 新モデル（60特徴）×現行軸選定       … 特徴量追加の効果だけを見る
  C 新モデル（60特徴）×3ヘッド軸選定    … 大敗ヘッドを加えた軸選定
        軸1 = argmax z(pw) − w1·z(pbad)   … 勝ちそうで大きく負けなそう
        軸2 = argmax z(p3) − w2·z(pbad)   … 3着内は堅いが大敗しにくい（軸1を除く）

⚠️ **w1/w2 は未確定**（`exp_three_head_axis.py` で掃引中）。本スクリプトは
   グリッドを全部出すが、**この4日間を見て w を選んではいけない**（評価窓での
   選択＝過学習）。w の決定は 8月より前の窓で行うこと。

⚠️ 4日・7車立てのみの小標本。統計的な結論は出せず、
   「実際に何が変わるか」を目で見るための再現。

ランク判定・買い目（相手の絞り方）は本番と同一（7S/7A は総流し・7B は WT△除外）。

⚠️ **ガミ見送りは適用しない方を本線とする**（ユーザー確認・2026-08-04）。
   ガミ判定は「予想家として朝に出した推奨」ではなく **実購入時点の判定**であり、
   推奨レース選出の巧拙を比べる本スクリプトの目的とは層が違うため。
   ガミ適用版も参考として併記する（最終オッズ近似なので発走前実測とは一致しない）。

DB書き込みなし。

使い方:
    python scripts/exp_replay_august.py [--from 2026-08-01] [--to 2026-08-04]
                                        [--refresh-cache]
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import (
    FEATURE_COLS_WT, build_features_wt, load_raw_data_wt, prepare_X,
)
from src.strategy_wt import (
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
    rank_7b_order_disagree, rank_7b_select_legs, rank_7s_field_entropy,
    rank_7s_select_axis, rank_7s_wt_overlap_n,
)

STAKE = 100
MARK = {1: "◎", 2: "◯", 3: "△", 4: "×"}
GAMI_THRESHOLD = 7.0   # src/cli/main.py と同値。買い目の最安オッズがこれ未満なら見送り
TRAIN_FROM = "2022-12-01"          # vintage と揃える
SEEDS = [42, 101, 202, 303, 404]   # exp_three_head_axis.py と同じ
W_GRID = [(0.0, 0.0), (0.0, 0.3), (0.0, 0.6),
          (0.3, 0.3), (0.3, 0.6), (0.6, 0.6), (0.6, 1.0)]
CACHE_DIR = REPO / "data" / "exp_cache"


def _q(sql: str) -> list[dict]:
    from sqlalchemy import create_engine, text
    eng = create_engine(os.environ["KEIRIN_DB_URL"])
    with eng.connect() as c:
        rows = [dict(r._mapping) for r in c.execute(text(sql))]
    eng.dispose()
    return rows


def _z(d: dict[int, float]) -> dict[int, float]:
    """レース内 z 化（全車同値なら 0）。"""
    v = np.array(list(d.values()), dtype=float)
    m, s = v.mean(), v.std()
    if s <= 0:
        return {k: 0.0 for k in d}
    return {k: (x - m) / s for k, x in d.items()}


def classify(p3, pw, mark, a1, a2):
    """本番と同じランク判定（7車立て）。"""
    hon = next((f for f, m in mark.items() if m == 1), None)
    tai = next((f for f, m in mark.items() if m == 2), None)
    ov = rank_7s_wt_overlap_n(a1, a2, hon, tai)
    ent = rank_7s_field_entropy(p3)
    if ov == 2:
        return "7B" if rank_7b_order_disagree(pw, hon) is True else None
    if ov not in (0, 1):
        return None
    n_fail = ((p3[a1] + p3[a2] > RANK_7S_AXIS_SUM_MAX) + (ent > RANK_7S_ENTROPY_MAX))
    return "7S" if n_fail == 0 else ("7A" if n_fail == 1 else None)


def sel_current(r):
    sel = rank_7s_select_axis(r["pw"], r["p3"])
    return (sel[0], sel[1]) if sel else None


def sel_three_head(r, w1: float, w2: float):
    zw, zp, zb = _z(r["pw"]), _z(r["p3"]), _z(r["bad"])
    s1 = {f: zw[f] - w1 * zb[f] for f in zw}
    a1 = max(s1, key=lambda f: s1[f])
    s2 = {f: zp[f] - w2 * zb[f] for f in zp if f != a1}
    if not s2:
        return None
    return a1, max(s2, key=lambda f: s2[f])


def summarize(rows, label, width: int = 30) -> None:
    n = len(rows)
    if not n:
        print(f"  {label:<{width}} 推奨0件")
        return
    hit = sum(1 for r in rows if r["hit"])
    bet = sum(r["bet"] for r in rows)
    ret = sum(r["ret"] for r in rows)
    both = sum(1 for r in rows if r["both"])
    a1win = sum(1 for r in rows if r["fo"].get(r["a1"]) == 1)
    a2bad = sum(1 for r in rows if r["fo"].get(r["a2"], 0) >= 6)
    pays = [r["ret"] for r in rows if r["hit"]]
    print(f"  {label:<{width}} {n:3d}件 軸両方3着内{100*both/n:5.1f}% "
          f"軸1が1着{100*a1win/n:5.1f}% 軸2が6着以下{100*a2bad/n:5.1f}% "
          f"的中{100*hit/n:5.1f}%({hit:2d}) 投資{bet:6d} 払戻{ret:6d} "
          f"ROI{100*ret/bet if bet else 0:6.1f}% 的中中央値{statistics.median(pays)/100 if pays else 0:5.1f}倍")


def load_bad_head(d_from: str, d_to: str, refresh: bool) -> pd.DataFrame:
    """8月分の pbad を返す（race_key, frame_no, bad）。

    train は race_date < d_from（＝2026-07-31まで）のみ。全車立てで学習し、
    評価は 7車立てに絞る（vintage と同じ方針）。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"replay_pbad_{d_from}_{d_to}.pkl"
    if cache.exists() and not refresh:
        print(f"[pbad] キャッシュ利用: {cache.name}")
        return pd.read_pickle(cache)

    print(f"[pbad] 学習データ読み込み {TRAIN_FROM}〜{d_to} ...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN_FROM, max_date=d_to))
    fo = pd.to_numeric(df["finish_order"], errors="coerce")
    df["bad6"] = ((fo >= 6) & (fo >= 1)).astype(int)   # 完走して6着以下（DNFは0）

    train = df[df["race_date"] < d_from]
    test = df[(df["race_date"] >= d_from) & (df["race_date"] <= d_to)]
    print(f"[pbad] train {len(train):,}行 / predict {len(test):,}行  "
          f"（train終端 {train['race_date'].max()}）", flush=True)

    Xtr, ytr, Xte = prepare_X(train), train["bad6"], prepare_X(test)
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, random_state=seed,
            deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(Xtr, ytr)
        preds.append(m.predict_proba(Xte)[:, 1])
        print(f"[pbad] seed={seed} 完了", flush=True)
    out = test[["race_key", "frame_no"]].copy()
    out["bad"] = np.mean(preds, axis=0)
    out.to_pickle(cache)
    print(f"[pbad] 保存: {cache}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="2026-08-01")
    ap.add_argument("--to", dest="d_to", default="2026-08-04")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    print(f"=== {args.d_from}〜{args.d_to} の再現 "
          f"（honest: 学習は全て 2026-07-31 まで）===\n")

    pbad = load_bad_head(args.d_from, args.d_to, args.refresh_cache)

    df = build_features_wt(load_raw_data_wt(min_date=args.d_from, max_date=args.d_to))
    with get_connection() as conn:
        ne = dict(conn.execute("SELECT race_key, n_entries FROM wt_races"))
    df = df[df["race_key"].map(ne) == 7].copy()
    X = prepare_X(df)
    df["p3"] = load_model("lgbm_wt_eval_m2608").predict_proba(X)[:, 1]
    df["pw"] = load_model("lgbm_wt_win_m2608").predict_proba(X)[:, 1]
    df = df.merge(pbad, on=["race_key", "frame_no"], how="left")
    if df["bad"].isna().any():
        raise SystemExit(f"pbad 欠損 {int(df['bad'].isna().sum())}行")

    keys = sorted(df["race_key"].unique())
    inq = ",".join(f"'{k}'" for k in keys)
    meta = {r["race_key"]: r for r in _q(
        f"SELECT r.race_key, r.race_no, r.race_date, "
        f"COALESCE(v.name, r.venue_id) AS venue, r.start_at "
        f"FROM keirin.wt_races r LEFT JOIN keirin.venue_info v "
        f"ON v.venue_code = r.venue_id WHERE r.race_key IN ({inq})")}
    trio = defaultdict(dict)
    for o in _q(f"SELECT race_key, combination, odds_value FROM keirin.wt_odds "
                f"WHERE bet_type='trio' AND race_key IN ({inq})"):
        try:
            v = float(o["odds_value"])
        except (TypeError, ValueError):
            continue
        if 0 < v < 9999:
            trio[o["race_key"]][frozenset(
                int(x) for x in re.split(r"[-=→]", str(o["combination"])))] = v
    actual = {r["base"]: r for r in _q(
        f"SELECT split_part(race_key,'#',1) AS base, race_date, rank, pred_combo, "
        f"bet_amount, payout, hit FROM keirin.picks_history "
        f"WHERE race_date BETWEEN '{args.d_from}' AND '{args.d_to}' "
        f"AND NOT COALESCE(miwokuri, false)")}

    # ---- レース単位の素材を作る ------------------------------------------
    races = []
    for rk, g in df.groupby("race_key"):
        if len(g) != 7:
            continue
        fo = {int(r.frame_no): (int(r.finish_order)
                                if r.finish_order is not None
                                and r.finish_order == r.finish_order else 0)
              for r in g.itertuples(index=False)}
        top3 = {f for f, v in fo.items() if 1 <= v <= 3}
        if len(top3) != 3:
            continue                      # 未確定・欠車で3着内が確定しないもの
        races.append({
            "rk": rk, "fo": fo, "top3": top3,
            "p3": {int(r.frame_no): float(r.p3) for r in g.itertuples(index=False)},
            "pw": {int(r.frame_no): float(r.pw) for r in g.itertuples(index=False)},
            "bad": {int(r.frame_no): float(r.bad) for r in g.itertuples(index=False)},
            "mark": {int(r.frame_no): int(r.prediction_mark)
                     for r in g.itertuples(index=False)},
        })
    print(f"評価対象: 7車立て・結果確定 {len(races)} レース\n")

    def build(selector, gami: bool = False):
        out = []
        for r in races:
            sel = selector(r)
            if not sel:
                continue
            a1, a2 = sel
            rank = classify(r["p3"], r["pw"], r["mark"], a1, a2)
            if rank is None:
                continue
            others = [f for f in r["p3"] if f not in (a1, a2)]
            if rank == "7B":
                ana = next((f for f, m in r["mark"].items() if m == 3), None)
                legs = rank_7b_select_legs(others, r["p3"], ana)
            else:
                legs = others
            board = trio.get(r["rk"], {})
            legs = [x for x in legs if frozenset({a1, a2, x}) in board]
            if not legs:
                continue
            min_odds = min(board[frozenset({a1, a2, x})] for x in legs)
            if gami and min_odds < GAMI_THRESHOLD:
                continue                      # 本番の発走前ガミ見送りに相当
            rest = r["top3"] - {a1, a2}
            hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 \
                and next(iter(rest)) in legs
            out.append({
                "rk": r["rk"], "rank": rank, "a1": a1, "a2": a2, "legs": legs,
                "bet": len(legs) * STAKE,
                "ret": (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0,
                "hit": hit, "both": {a1, a2} <= r["top3"],
                "top3": r["top3"], "fo": r["fo"], "mark": r["mark"]})
        return out

    # ---- A: 実績（picks_history） -----------------------------------------
    by_rk = {r["rk"]: r for r in races}
    a_rows, a_bet, a_pay, a_hits = [], 0, 0, 0
    for base, p in actual.items():
        if p["bet_amount"]:
            a_bet += p["bet_amount"]
            a_pay += p["payout"] or 0
            a_hits += int(bool(p["hit"]))
        r = by_rk.get(base)
        if r is None:
            continue
        combo = (p["pred_combo"] or "").split(" ")[0]
        if "-" not in combo:
            continue
        ax, lg = combo.split("-", 1)
        axes = [int(x) for x in ax.replace("=", ",").split(",") if x.strip().isdigit()]
        legs = [int(x) for x in lg.split(",") if x.strip().isdigit()]
        if len(axes) < 2:
            continue
        a1, a2 = axes[0], axes[1]
        board = trio.get(base, {})
        legs = [x for x in legs if frozenset({a1, a2, x}) in board]
        if not legs:
            continue
        rest = r["top3"] - {a1, a2}
        hit = len(r["top3"] & {a1, a2}) == 2 and len(rest) == 1 \
            and next(iter(rest)) in legs
        a_rows.append({
            "rk": base, "rank": p["rank"].replace("RANK_", ""), "a1": a1, "a2": a2,
            "legs": legs, "bet": len(legs) * STAKE,
            "ret": (round(board[frozenset(r["top3"])] * 100) // 10 * 10) if hit else 0,
            "hit": hit, "both": {a1, a2} <= r["top3"],
            "top3": r["top3"], "fo": r["fo"], "mark": r["mark"]})

    b_rows = build(sel_current)
    c_rows = {(w1, w2): build(lambda r, a=w1, b=w2: sel_three_head(r, a, b))
              for w1, w2 in W_GRID}
    b_rows_g = build(sel_current, gami=True)
    c_rows_g = {(w1, w2): build(lambda r, a=w1, b=w2: sel_three_head(r, a, b), gami=True)
                for w1, w2 in W_GRID}

    print("【全体】")
    print(f"  {'A 実績(実精算・picks_history)':<30} "
          f"{len(actual):3d}件 " + " " * 56 +
          f"的中{100*a_hits/len(actual) if actual else 0:5.1f}%({a_hits:2d}) "
          f"投資{a_bet:6d} 払戻{a_pay:6d} ROI{100*a_pay/a_bet if a_bet else 0:6.1f}%")
    summarize(a_rows, "A' 実績の買い目×最終オッズ")
    summarize(b_rows, "B 新60特徴×現行軸選定")
    for (w1, w2), rows in c_rows.items():
        summarize(rows, f"C 3ヘッド w1={w1} w2={w2}")
    print()

    print("【ガミ見送り適用後（本番同等・最安 <7.0倍 は見送り／最終オッズ近似）】")
    summarize(b_rows_g, "B 新60特徴×現行軸選定")
    for (w1, w2), rows in c_rows_g.items():
        summarize(rows, f"C 3ヘッド w1={w1} w2={w2}")
    print()

    print("【ランク別 件数/的中】")
    def rank_line(lbl, rows):
        by = defaultdict(list)
        for r in rows:
            by[r["rank"]].append(r)
        parts = []
        for k in ("7S", "7A", "7B"):
            v = by.get(k, [])
            parts.append(f"{k} {len(v):2d}件/的中{sum(1 for x in v if x['hit'])}"
                         if v else f"{k}  0件      ")
        print(f"  {lbl:<30} " + "  ".join(parts))
    rank_line("A' 実績の買い目", a_rows)
    rank_line("B 新60特徴×現行軸選定", b_rows)
    for (w1, w2), rows in c_rows.items():
        rank_line(f"C 3ヘッド w1={w1} w2={w2}", rows)
    print()

    print("【日別】件数 / 的中 / 投資 / 払戻 / ROI")
    days = sorted({str(meta[r["rk"]]["race_date"]) for r in races})

    def day_block(lbl, rows):
        by = defaultdict(list)
        for r in rows:
            by[str(meta[r["rk"]]["race_date"])].append(r)
        print(f"  ── {lbl}")
        tn = th = tb = tr = 0
        for d in days:
            v = by.get(d, [])
            n = len(v)
            h = sum(1 for x in v if x["hit"])
            bet = sum(x["bet"] for x in v)
            ret = sum(x["ret"] for x in v)
            tn += n; th += h; tb += bet; tr += ret
            print(f"     {d}  {n:3d}件  的中{h:2d} ({100*h/n if n else 0:5.1f}%)  "
                  f"投資{bet:6d}  払戻{ret:6d}  ROI{100*ret/bet if bet else 0:7.1f}%")
        print(f"     {'合計':10}  {tn:3d}件  的中{th:2d} ({100*th/tn if tn else 0:5.1f}%)  "
              f"投資{tb:6d}  払戻{tr:6d}  ROI{100*tr/tb if tb else 0:7.1f}%")

    # A 実績（実精算）
    print("  ── A 実績（実精算・picks_history）")
    ad = defaultdict(lambda: [0, 0, 0, 0])
    for p in actual.values():
        c = ad[str(p["race_date"])]
        c[0] += 1
        c[1] += int(bool(p["hit"]))
        c[2] += p["bet_amount"] or 0
        c[3] += p["payout"] or 0
    tn = th = tb = tr = 0
    for d in sorted(ad):
        n, h, bet, ret = ad[d]
        tn += n; th += h; tb += bet; tr += ret
        print(f"     {d}  {n:3d}件  的中{h:2d} ({100*h/n if n else 0:5.1f}%)  "
              f"投資{bet:6d}  払戻{ret:6d}  ROI{100*ret/bet if bet else 0:7.1f}%")
    print(f"     {'合計':10}  {tn:3d}件  的中{th:2d} ({100*th/tn if tn else 0:5.1f}%)  "
          f"投資{tb:6d}  払戻{tr:6d}  ROI{100*tr/tb if tb else 0:7.1f}%")

    day_block("A' 実績の買い目×最終オッズ", a_rows)
    day_block("B 新60特徴×現行軸選定", b_rows)
    for (w1, w2), rows in c_rows.items():
        day_block(f"C 3ヘッド w1={w1} w2={w2}", rows)
    print()

    ref = (0.3, 0.6)
    print(f"【C w1={ref[0]} w2={ref[1]} の推奨明細】"
          f"（w は未確定・この4日で選んではいけない）")
    for r in sorted(c_rows[ref], key=lambda x: meta[x["rk"]].get("start_at") or 0):
        m = meta[r["rk"]]
        order = "-".join(str(f) for f in sorted(r["top3"], key=lambda f: r["fo"][f]))
        mk = lambda f: MARK.get(r["mark"].get(f), "—")     # noqa: E731
        res = "🎯的中" if r["hit"] else ("軸的中" if r["both"] else "×")
        print(f"  {str(m['race_date'])[5:]} {m.get('venue','')}{m.get('race_no','')}R "
              f"{r['rank']:3} 軸{r['a1']}({mk(r['a1'])})+{r['a2']}({mk(r['a2'])}) "
              f"相手{','.join(map(str, r['legs']))} → {order} {res} "
              f"{'+' + str(r['ret']) + '円' if r['hit'] else ''}")


if __name__ == "__main__":
    main()
