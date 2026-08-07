"""「着地オッズ想定」で配分できるか（2026-08-07・ユーザー依頼）。

ユーザー: 「1/オッズ比例としたいが、朝時点のオッズ精度は現状ない見込み。
            **着地オッズを想定して入稿**できないか」

dutch 配分で効くのは**買う点どうしの相対的な 1/オッズ**だけなので、
必要なのは「最終オッズの水準」ではなく **最終時点の相対人気（implied prob）** の推定。
一律の割引（朝の中央値 0.85）は比率を変えないので**何の役にも立たない**
（`dutch_adj ≡ dutch` で実証済み）。効くとすれば個別誤差の縮小だけ。

三連複・軸2車固定なら買う点の差は3列目の車だけなので、
**モデルの相対確率 ∝ 3列目の pred_top3** という極めて単純な推定量が使える。

比較する重み（stake ∝ w、払戻を揃える向き）:
    equal        w = 1                       （現行）
    morning      w = 1/朝オッズ               （前回の dutch）
    model        w = 3列目の p3               （オッズを一切使わない）
    blend(λ)     w = (1/朝オッズ)^λ × p3^(1-λ)
    oracle       w = 1/最終オッズ              （実装不能・上限）

⚠️ p3 は `axis_detail_7car.pkl` の vintage walk-forward 予測（honest）。
⚠️ 読み取り専用。
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.stake_allocation import allocate_budget  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_7C_LEGS_MIN, RANK_7C_P3_SUM_MIN,
    rank_7c_is_lowpay_pattern, rank_7c_select_legs,
)

BUDGET, UNIT = 10_000, 100
N_UNITS = BUDGET // UNIT
_SEP_RE = re.compile(r"[-=]")
DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
TRIO_RANKS = ("RANK_7SS", "RANK_7S", "RANK_7A", "RANK_7B")
MORNING_FROM = "2026-06-08"   # 朝オッズが存在する最初の日
FROM = os.environ.get("EXP_FROM", MORNING_FROM)
TO = os.environ.get("EXP_TO", "2026-08-06")


# ------------------------------------------------------------------ 読み込み
def load_odds(conn, keys, snap=None):
    out = defaultdict(dict)
    keys = list(keys)
    for i in range(0, len(keys), 800):
        inlist = ",".join(f"'{k}'" for k in keys[i:i + 800])
        tbl = ("keirin.wt_odds" if snap is None else "keirin.wt_odds_snapshot")
        cond = "" if snap is None else f" AND snapshot_type='{snap}'"
        for rk, comb, od in conn.execute(
            f"SELECT race_key, combination, odds_value FROM {tbl} "
            f"WHERE bet_type='trio'{cond} AND race_key IN ({inlist})"
        ):
            if od is None or not (0 < float(od) < 9000):
                continue
            try:
                key = frozenset(int(x) for x in _SEP_RE.split(str(comb)))
            except ValueError:
                continue
            if len(key) == 3:
                out[rk][key] = float(od)
    return out


def load_top3(conn, keys):
    out = {}
    keys = list(keys)
    for i in range(0, len(keys), 800):
        inlist = ",".join(f"'{k}'" for k in keys[i:i + 800])
        rows = defaultdict(dict)
        for rk, fno, fo in conn.execute(
            f"SELECT race_key, frame_no, finish_order FROM keirin.wt_entries "
            f"WHERE race_key IN ({inlist})"
        ):
            if fo is not None and 1 <= int(fo) <= 3:
                rows[rk][int(fo)] = int(fno)
        for rk, d in rows.items():
            if set(d) == {1, 2, 3}:
                out[rk] = frozenset(d.values())
    return out


def parse(s):
    head = str(s or "").split(" ")[0]
    if "-" not in head:
        return [], []
    a, l = head.split("-", 1)
    return ([int(x) for x in a.replace("=", ",").split(",") if x.strip().isdigit()],
            [int(x) for x in l.split(",") if x.strip().isdigit()])


def build(conn, p3map):
    """(date, base, rank, axes, legs, p3) のレコード列。7車のみ。"""
    recs = []
    rows = list(conn.execute(f"""
        SELECT p.race_date, p.race_key, p.rank, p.pred_combo
        FROM keirin.picks_history p
        JOIN keirin.wt_races r ON r.race_key = split_part(p.race_key,'#',1)
        WHERE p.rank IN ({",".join("'" + r + "'" for r in TRIO_RANKS)})
          AND p.race_date >= '{FROM}' AND p.race_date <= '{TO}'
          AND COALESCE(r.cancel,0)=0 AND r.n_entries=7
    """))
    for r in rows:
        base = r["race_key"].split("#")[0]
        axes, legs = parse(r["pred_combo"])
        if len(axes) != 2 or len(legs) < 2 or base not in p3map:
            continue
        recs.append(dict(date=r["race_date"], base=base,
                         rank=r["rank"].replace("RANK_", ""),
                         axes=axes, legs=legs, p3=p3map[base]))
    # 7C は picks_history 未 backfill のため確定仕様で再現
    for r in pickle.load(open(DETAIL, "rb")):
        if not (FROM <= r["date"] <= TO) or len(r["p3"]) != 7:
            continue
        p3 = r["p3"]
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        a1, a2 = ranked[0], ranked[1]
        if p3[a1] + p3[a2] < RANK_7C_P3_SUM_MIN:
            continue
        legs = rank_7c_select_legs(ranked[2:], p3)
        if len(legs) < RANK_7C_LEGS_MIN:
            continue
        if rank_7c_is_lowpay_pattern(p3, r.get("line") or {}):
            continue
        recs.append(dict(date=r["date"], base=r["rk"], rank="7C",
                         axes=[a1, a2], legs=legs, p3=p3))
    return recs


# ------------------------------------------------------------------ 配分
def allocate(weights, ref_payout=None):
    """**本番と同じ配分関数**（`src.stake_allocation.allocate_budget`）へ委譲する。

    検証と本番で別実装を持つと必ず食い違う。`ref_payout` は互換のため
    受け取るだけで使わない（端数の寄せ先にオッズを使うと先読みになる）。
    """
    return allocate_budget(weights, BUDGET, UNIT)


def weights_for(scheme, legs, morning, p3, lam=0.5):
    """legs（3列目の車番）ごとの重み。"""
    if scheme == "equal":
        return {t: 1.0 for t in legs}
    if scheme == "morning":
        return {t: 1.0 / max(morning[t], 1e-9) for t in legs}
    if scheme == "model":
        return {t: max(p3.get(t, 1e-6), 1e-6) for t in legs}
    if scheme == "blend":
        return {t: (1.0 / max(morning[t], 1e-9)) ** lam
                   * max(p3.get(t, 1e-6), 1e-6) ** (1 - lam) for t in legs}
    if scheme == "oracle":
        return {t: 1.0 / max(morning[t], 1e-9) for t in legs}  # 呼び出し側で最終を渡す
    raise ValueError(scheme)


def run(recs, board, morning_board, top3, label):
    print(f"\n{'='*92}\n{label}\n{'='*92}")
    print(f"{'配分':<16s} {'n':>5s} {'的中%':>7s} {'実質的中%':>10s} {'ガミ率%':>8s} "
          f"{'ROI%':>7s} {'相対確率の誤差':>14s}")

    schemes = [("equal（現行）", "equal", None),
               ("model（p3のみ）", "model", None),
               ("blend λ=0.25", "blend", 0.25),
               ("blend λ=0.5", "blend", 0.5),
               ("blend λ=0.75", "blend", 0.75),
               ("morning（λ=1）", "morning", None),
               ("oracle（上限）", "oracle", None)]

    for name, scheme, lam in schemes:
        n = hit = rhit = 0
        bet_t = ret_t = 0.0
        err_sum = err_n = 0.0
        for r in recs:
            fo, mo = board.get(r["base"], {}), morning_board.get(r["base"], {})
            win = top3.get(r["base"])
            if win is None:
                continue
            pts = {t: frozenset({*r["axes"], t}) for t in r["legs"]}
            f = {t: fo.get(p) for t, p in pts.items()}
            m = {t: mo.get(p) for t, p in pts.items()}
            if any(v is None for v in f.values()):
                continue
            has_m = all(v is not None for v in m.values())
            if scheme in ("morning", "blend") and not has_m:
                # 朝オッズが欠けたら **モデル** へフォールバックする。
                # ⚠️ ここを均等にすると λ→0 が model 単独に収束せず、
                #    ブレンドの λ 比較が「35%が均等の混ぜ物」同士の比較になる
                #    （最初の実装がこれで、λ=0.25 が model より悪いという
                #      あり得ない非単調が出た）。
                w = weights_for("model", r["legs"], m, r["p3"])
            elif scheme == "oracle":
                w = {t: 1.0 / f[t] for t in r["legs"]}
            else:
                w = weights_for(scheme, r["legs"], m, r["p3"], lam or 0.5)
            stakes = allocate(w, f)
            bet = sum(stakes.values())
            wt = next((t for t, p in pts.items() if p == win), None)
            ret = int(stakes[wt] * f[wt]) // 10 * 10 if wt is not None else 0
            n += 1
            bet_t += bet
            ret_t += ret
            if ret:
                hit += 1
                if ret >= bet:
                    rhit += 1
            # 推定した相対確率と最終の相対確率のずれ（L1・レース内）
            tot_w = sum(w.values())
            qf = {t: (1 / f[t]) for t in r["legs"]}
            tot_f = sum(qf.values())
            err_sum += sum(abs(w[t] / tot_w - qf[t] / tot_f) for t in r["legs"])
            err_n += 1
        if not n:
            continue
        print(f"{name:<16s} {n:>5,d} {100*hit/n:>7.2f} {100*rhit/n:>10.2f} "
              f"{100*(1-rhit/max(hit,1)):>8.1f} {100*ret_t/bet_t:>7.1f} "
              f"{err_sum/max(err_n,1):>14.4f}")


def paired(recs, board, morning_board, top3, a, b, la=None, lb=None, n_boot=2000):
    """同一レースで配分 a と b の実質的中率・ROI を比べる。"""
    import random

    def outcomes(scheme, lam):
        out = {}
        for r in recs:
            fo, mo = board.get(r["base"], {}), morning_board.get(r["base"], {})
            win = top3.get(r["base"])
            if win is None:
                continue
            pts = {t: frozenset({*r["axes"], t}) for t in r["legs"]}
            f = {t: fo.get(p) for t, p in pts.items()}
            if any(v is None for v in f.values()):
                continue
            m = {t: mo.get(p) for t, p in pts.items()}
            has_m = all(v is not None for v in m.values())
            sc = scheme
            if scheme in ("morning", "blend") and not has_m:
                sc = "model"
            w = ({t: 1.0 / f[t] for t in r["legs"]} if scheme == "oracle"
                 else weights_for(sc, r["legs"], m, r["p3"], lam or 0.5))
            stakes = allocate(w, f)
            bet = sum(stakes.values())
            wt = next((t for t, p in pts.items() if p == win), None)
            ret = int(stakes[wt] * f[wt]) // 10 * 10 if wt is not None else 0
            out[r["base"] + r["rank"]] = (int(ret >= bet and ret > 0), bet, ret)
        return out

    ra, rb = outcomes(a, la), outcomes(b, lb)
    keys = sorted(set(ra) & set(rb))
    rnd = random.Random(0)
    dh, dr = [], []
    for _ in range(n_boot):
        s = [keys[rnd.randrange(len(keys))] for _ in range(len(keys))]
        dh.append(sum(rb[k][0] - ra[k][0] for k in s) / len(s))
        dr.append(sum(rb[k][2] for k in s) / sum(rb[k][1] for k in s)
                  - sum(ra[k][2] for k in s) / sum(ra[k][1] for k in s))
    dh.sort()
    dr.sort()
    lo, hi = dh[int(0.025 * n_boot)], dh[int(0.975 * n_boot)]
    rlo, rhi = dr[int(0.025 * n_boot)], dr[int(0.975 * n_boot)]
    d = (sum(rb[k][0] for k in keys) - sum(ra[k][0] for k in keys)) / len(keys)
    droi = (sum(rb[k][2] for k in keys) / sum(rb[k][1] for k in keys)
            - sum(ra[k][2] for k in keys) / sum(ra[k][1] for k in keys))
    print(f"  {b}{lb or ''} − {a}: 実質的中 {100*d:+6.2f}pt [{100*lo:+6.2f}, {100*hi:+6.2f}] "
          f"P={100*sum(1 for x in dh if x > 0)/n_boot:5.1f}%  |  "
          f"ROI {100*droi:+6.2f}pt [{100*rlo:+6.2f}, {100*rhi:+6.2f}]")


def main():
    p3map = {r["rk"]: r["p3"] for r in pickle.load(open(DETAIL, "rb"))}
    with get_connection() as conn:
        recs = build(conn, p3map)
        bases = sorted({r["base"] for r in recs})
        board = load_odds(conn, bases, None)
        morning = load_odds(conn, bases, "morning")
        top3 = load_top3(conn, bases)

    if os.environ.get("EXP_ALL_DAYS"):
        pass   # 全期間モード: 朝スナップショットの有無で母集団を絞らない
    else:
        snap_days = {r["date"] for r in recs if r["base"] in morning}
        recs = [r for r in recs if r["date"] in snap_days]
    print(f"7車 {len(recs):,} レース（{FROM}〜{TO}・朝スナップショットのある日）")
    nfull = sum(1 for r in recs
                if all(frozenset({*r["axes"], t}) in morning.get(r["base"], {})
                       for t in r["legs"]))
    print(f"朝オッズが買う点すべてに揃う: {nfull:,}（{100*nfull/max(len(recs),1):.1f}%）")

    run(recs, board, morning, top3, "全体（本番相当・朝オッズ欠損はモデルへフォールバック）")
    print("\n-- 対比較（同一レース・2,000回ブートストラップ）--")
    for b, lb in (("model", None), ("morning", None), ("blend", 0.75)):
        paired(recs, board, morning, top3, "equal", b, None, lb)
    paired(recs, board, morning, top3, "morning", "blend", None, 0.75)
    for lo, hi in (("2026-06-08", "2026-06-30"), ("2026-07-01", "2026-08-06")):
        sub = [r for r in recs if lo <= r["date"] <= hi]
        run(sub, board, morning, top3, f"窓 {lo}〜{hi}")
    for rank in sorted({r["rank"] for r in recs}):
        sub = [r for r in recs if r["rank"] == rank]
        if len(sub) >= 80:
            run(sub, board, morning, top3, f"ランク {rank}")


if __name__ == "__main__":
    main()
