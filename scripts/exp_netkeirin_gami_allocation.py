"""netkeirin の「ガミ＝不的中」に対する傾斜配分とレース除外の検証（2026-08-07）。

ユーザー依頼:
  「netkeirin の的中率はガミが不的中扱いになる。ある程度の低オッズの場合、傾斜配分で
    購入資金を上げておかないといけない。均等買いの入稿について **10倍以下の買い目は
    他の倍の配分** にできるか。また **2点以上 朝の時点で10倍以下があるレースは
    推奨から除外** するのが良さそう。この条件での検証・実証を進めて」

## 測る指標

netkeirin の表示は「払戻 >= 投資」でないと的中扱いにならない。よって主指標は

    実質的中率 = P(払戻 >= 1レース予算10,000円)

であり、従来の 的中率（＝軸2車が3着内）とは別物。ROI も併せて見る
（配分を変えても市場が効率的なら ROI は動かないはず＝[[keirin_stake_allocation_rejected]]）。

## 比較する配分

    equal      現行。予算 10,000円 を点数で均等割り（100円単位）
    tilt(T,m)  参照オッズ <= T の点に m 倍の重み（ユーザー案は T=10, m=2）
    dutch(g)   全点の払戻が揃うよう 1/odds に比例配分（払戻目標 = 10,000×(1+g)）

## 2つの評価モード

    IDEAL  参照オッズ = **最終オッズ**。全期間で n が大きい。実装不能な上限値
    REAL   参照オッズ = **朝オッズ**（wt_odds_snapshot・2026-06-08〜のみ）。実装可能

決済は常に最終オッズ（wt_odds）。REAL と IDEAL の差が「朝→直前のオッズ変動で
どれだけ取り逃すか」＝実装時に払うコスト。

⚠️ 読み取り専用。DB へは一切書き込まない。
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_7C_LEGS_MIN, RANK_7C_P3_SUM_MIN,
    rank_7c_is_lowpay_pattern, rank_7c_select_legs,
)

BUDGET = 10_000
UNIT = 100                      # 賭け金の最小単位
N_UNITS = BUDGET // UNIT        # 100 口
DETAIL = REPO / "data" / "exp_cache" / "axis_detail_7car.pkl"
_SEP_RE = re.compile(r"[-=]")

# picks_history から復元するランク（三連複・軸2車ながし）。
# 7H1 は三連単+三連複の併せ買いで構造が違うため別扱い（本スクリプトの対象外）。
TRIO_RANKS = ("RANK_7SS", "RANK_7S", "RANK_7A", "RANK_7B", "RANK_9S", "RANK_9A")


# ---------------------------------------------------------------- 配分ロジック
def _alloc_from_weights(points, weights, ref_odds):
    """重み → 100円単位の賭け金 dict。端数は「払戻が最も低い点」へ寄せる。

    端数の寄せ先を払戻順にするのは、どの配分方式でも同じ規則にして
    比較を公平にするため（ガミ回避に最も効く方向でもある）。
    """
    total_w = sum(weights[p] for p in points)
    units = {p: int(N_UNITS * weights[p] / total_w) for p in points}
    left = N_UNITS - sum(units.values())
    # 現時点の払戻が低い順に1口ずつ配る
    for _ in range(left):
        tgt = min(points, key=lambda p: units[p] * UNIT * ref_odds.get(p, 1e9))
        units[tgt] += 1
    return {p: units[p] * UNIT for p in points}


def alloc_equal(points, ref_odds):
    return _alloc_from_weights(points, {p: 1.0 for p in points}, ref_odds)


def alloc_tilt(points, ref_odds, thr, mult):
    """参照オッズ <= thr の点に mult 倍の重みを与える（ユーザー案）。"""
    w = {p: (mult if ref_odds.get(p, 1e9) <= thr else 1.0) for p in points}
    return _alloc_from_weights(points, w, ref_odds)


def alloc_dutch(points, ref_odds):
    """全点の払戻が揃う配分（1/odds に比例）。"""
    w = {p: 1.0 / max(ref_odds.get(p, 1e9), 1e-9) for p in points}
    return _alloc_from_weights(points, w, ref_odds)


# 朝→最終の下振れ（実測中央値・drift_report の帯別集計から）。
# 朝オッズをそのまま使うと dutch の払戻目標が丸ごと約15%上振れした前提になる。
DRIFT_MEDIAN = ((3.0, 1.00), (5.0, 0.85), (10.0, 0.85), (20.0, 0.86),
                (50.0, 0.88), (float("inf"), 0.68))


def _shrink(o):
    for hi, k in DRIFT_MEDIAN:
        if o < hi:
            return o * k
    return o


def alloc_dutch_adj(points, ref_odds):
    """朝→最終の下振れ中央値で朝オッズを割り引いてから dutch する。"""
    w = {p: 1.0 / max(_shrink(ref_odds.get(p, 1e9)), 1e-9) for p in points}
    return _alloc_from_weights(points, w, ref_odds)


def alloc_power(points, ref_odds, alpha):
    """equal(α=0) と dutch(α=1) の中間。α を上げるほど低オッズ点へ寄せる。

    朝オッズは誤差が大きいので、そこへ完全に従う（α=1）より
    鈍らせた方が良い可能性がある。それを測るための族。
    """
    w = {p: max(ref_odds.get(p, 1e9), 1e-9) ** (-alpha) for p in points}
    return _alloc_from_weights(points, w, ref_odds)


def alloc_guard(points, ref_odds, margin):
    """**元返し保証+余剰は穴目へ**。

    各点に「参照オッズで払戻が予算×(1+margin) になる額」を確保し、
    余った予算は**最もオッズの高い点**へ全部乗せる。

    - dutch は全点の払戻を揃えるので **大的中が構造的に消える**。こちらは
      下限だけ揃えて余剰を穴目に置くので上振れの目が残る
    - 必要額の合計が予算を超えるレース（＝margin 込みのブックが1超）では
      保証できない。その場合は dutch へ落とす（払戻を最大限そろえる）
    """
    need = {}
    for p in points:
        o = max(ref_odds.get(p, 1e9), 1e-9)
        units = -(-int(BUDGET * (1 + margin)) // int(o * UNIT))  # 切り上げ
        need[p] = max(units, 1)
    if sum(need.values()) > N_UNITS:
        return alloc_dutch(points, ref_odds)
    top = max(points, key=lambda p: ref_odds.get(p, 0.0))
    need[top] += N_UNITS - sum(need.values())
    return {p: need[p] * UNIT for p in points}


ALLOCATORS = {
    "equal":      lambda pts, ref: alloc_equal(pts, ref),
    "tilt10x2":   lambda pts, ref: alloc_tilt(pts, ref, 10.0, 2.0),
    "tilt10x3":   lambda pts, ref: alloc_tilt(pts, ref, 10.0, 3.0),
    "pow0.5":     lambda pts, ref: alloc_power(pts, ref, 0.5),
    "pow0.75":    lambda pts, ref: alloc_power(pts, ref, 0.75),
    "dutch":      lambda pts, ref: alloc_dutch(pts, ref),
    "guard0.0":   lambda pts, ref: alloc_guard(pts, ref, 0.0),
    "guard0.3":   lambda pts, ref: alloc_guard(pts, ref, 0.3),
    "guard0.6":   lambda pts, ref: alloc_guard(pts, ref, 0.6),
    "guard1.0":   lambda pts, ref: alloc_guard(pts, ref, 1.0),
}


# ---------------------------------------------------------------- データ構築
def _parse_pred_combo(s):
    """'1=7-5,3,4 (axis_sum=1.5)' → ([1,7], [5,3,4])。

    ⚠️ 末尾の ' (axis_sum=…)' を落としてからでないと最後の相手を取りこぼす。
    """
    if not s:
        return [], []
    head = str(s).split(" ")[0]
    if "-" not in head:
        return [], []
    axis_part, legs_part = head.split("-", 1)
    axes = [int(x) for x in axis_part.replace("=", ",").split(",") if x.strip().isdigit()]
    legs = [int(x) for x in legs_part.split(",") if x.strip().isdigit()]
    return axes, legs


def _load_odds(conn, race_keys, snapshot_type=None):
    """race_key -> {frozenset(3車): odds} を返す。"""
    if not race_keys:
        return {}
    out = defaultdict(dict)
    keys = list(race_keys)
    for i in range(0, len(keys), 800):
        chunk = keys[i:i + 800]
        inlist = ",".join(f"'{k}'" for k in chunk)
        if snapshot_type is None:
            sql = (f"SELECT race_key, combination, odds_value FROM keirin.wt_odds "
                   f"WHERE bet_type='trio' AND race_key IN ({inlist})")
        else:
            sql = (f"SELECT race_key, combination, odds_value FROM keirin.wt_odds_snapshot "
                   f"WHERE bet_type='trio' AND snapshot_type='{snapshot_type}' "
                   f"AND race_key IN ({inlist})")
        for rk, comb, od in conn.execute(sql):
            if od is None or not (0 < float(od) < 9000):
                continue
            # ⚠️ 区切りは表で違う（wt_odds は '1=2=3' / wt_odds_snapshot は '1-2-3'）。
            # 片方だけを想定すると盤面が丸ごと空になり、無言でサンプルが消える。
            try:
                key = frozenset(int(x) for x in _SEP_RE.split(str(comb)))
            except ValueError:
                continue
            if len(key) == 3:
                out[rk][key] = float(od)
    return out


def _load_top3(conn, race_keys):
    out = {}
    keys = list(race_keys)
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


def build_from_picks(conn, date_from, date_to):
    """picks_history（本番＝walk-forward 再構築済み）から買い目を復元する。"""
    rows = list(conn.execute(f"""
        SELECT p.race_date, p.race_key, p.rank, p.pred_combo, r.n_entries
        FROM keirin.picks_history p
        JOIN keirin.wt_races r ON r.race_key = split_part(p.race_key,'#',1)
        WHERE p.rank IN ({",".join("'" + r + "'" for r in TRIO_RANKS)})
          AND p.race_date BETWEEN '{date_from}' AND '{date_to}'
          AND COALESCE(r.cancel,0) = 0
        ORDER BY p.race_date
    """))
    recs = []
    for r in rows:
        axes, legs = _parse_pred_combo(r["pred_combo"])
        if len(axes) != 2 or not legs:
            continue
        pts = [frozenset({axes[0], axes[1], t}) for t in legs]
        if len(set(pts)) != len(pts):
            continue
        recs.append(dict(date=r["race_date"], base=r["race_key"].split("#")[0],
                         rank=r["rank"].replace("RANK_", ""), points=pts))
    return recs


def build_7c(date_from, date_to):
    """7C は picks_history 未 backfill のため確定仕様を pkl 上で再現する。

    p3 は vintage walk-forward 予測（axis_detail_7car.pkl）＝ honest。
    """
    if not DETAIL.exists():
        return []
    recs = []
    for r in pickle.load(open(DETAIL, "rb")):
        d = r["date"]
        if not (date_from <= d <= date_to):
            continue
        p3 = r["p3"]
        if len(p3) != 7:
            continue
        ranked = sorted(p3, key=lambda f: (-p3[f], f))
        a1, a2 = ranked[0], ranked[1]
        if p3[a1] + p3[a2] < RANK_7C_P3_SUM_MIN:
            continue
        legs = rank_7c_select_legs(ranked[2:], p3)
        if len(legs) < RANK_7C_LEGS_MIN:
            continue
        if rank_7c_is_lowpay_pattern(p3, r.get("line") or {}):
            continue
        recs.append(dict(date=d, base=r["rk"], rank="7C",
                         points=[frozenset({a1, a2, t}) for t in legs]))
    return recs


# ---------------------------------------------------------------- 評価
def settle(points, stakes, final_odds, win):
    bet = sum(stakes.values())
    if win is None or win not in stakes:
        return bet, 0
    # 実際の払戻は10円単位で切り捨て（既存スクリプトと同じ扱い）
    return bet, int(stakes[win] * final_odds[win]) // 10 * 10


def prepare(recs, final_map, ref_map):
    """盤面（最終・参照の両方）が全点そろっているレースだけを残す。

    ⚠️ **母集団は必ずここで一度だけ確定させる**。REAL（朝オッズ参照）と
    IDEAL（最終オッズ参照）を別々に絞ると母集団が食い違い、
    「オッズ変動のコスト」と「母集団の入れ替え」を取り違える。
    """
    prepared = []
    for r in recs:
        fo = final_map.get(r["base"], {})
        ref = ref_map.get(r["base"], {})
        pts = [p for p in r["points"] if p in fo and p in ref]
        if len(pts) != len(r["points"]) or not pts:
            continue
        prepared.append(dict(r, points=pts, fo=fo, ref=ref))
    return prepared


def prepare_prod(recs, final_map, ref_map):
    """本番相当。**朝の盤面が欠けているレースも母集団に残す**（ref=None）。

    朝8:00 の板は後半レースほど薄く、買う点のオッズが1つでも無いことがある。
    そのレースを分析から落とすと「朝オッズで判断できたレースだけ」の
    都合のよい母集団になる。本番では均等割りへフォールバックするので
    ここでもそう扱う。
    """
    out = []
    for r in recs:
        fo = final_map.get(r["base"], {})
        if any(p not in fo for p in r["points"]):
            continue                      # 決済不能（＝評価不能）なので除く
        ref = ref_map.get(r["base"], {})
        full = all(p in ref for p in r["points"])
        out.append(dict(r, fo=fo, ref=(ref if full else None)))
    return out


def evaluate(prepared, allocators, exclusions, use_final_as_ref=False):
    """prepared を配分×除外規則の全組み合わせで精算し、集計行を返す。

    r["ref"] が None のレースは「朝に判断材料が無かった」＝均等割りへ
    フォールバックし、除外規則も適用しない（本番の挙動と同じ）。
    """
    if use_final_as_ref:
        prepared = [dict(r, ref=r["fo"]) for r in prepared]
    out = []
    for ex_name, ex_fn in exclusions.items():
        kept = [r for r in prepared
                if r["ref"] is None or not ex_fn(r["points"], r["ref"])]
        for al_name, al_fn in allocators.items():
            n = len(kept)
            hit = rhit = 0
            bet_t = ret_t = 0
            for r in kept:
                stakes = (alloc_equal(r["points"], r["fo"]) if r["ref"] is None
                          else al_fn(r["points"], r["ref"]))
                bet, ret = settle(r["points"], stakes, r["fo"], r.get("win"))
                bet_t += bet
                ret_t += ret
                if ret > 0:
                    hit += 1
                    if ret >= bet:
                        rhit += 1
            days = len({r["date"] for r in kept}) or 1
            out.append(dict(
                exclusion=ex_name, alloc=al_name, n=n, per_day=n / days,
                hit=100 * hit / n if n else 0.0,
                real_hit=100 * rhit / n if n else 0.0,
                gami_of_hit=100 * (1 - rhit / hit) if hit else 0.0,
                roi=100 * ret_t / bet_t if bet_t else 0.0,
            ))
    return out


# ---------------------------------------------------------------- 除外規則
def ex_none(points, ref):
    return False


def make_ex_lowcount(thr, k):
    """朝オッズ <= thr の点が k 点以上ならレースごと見送り（ユーザー案は thr=10, k=2）。"""
    def f(points, ref):
        return sum(1 for p in points if ref.get(p, 1e9) <= thr) >= k
    return f


def make_ex_book(theta):
    """買う点の合成ブック Σ(1/odds) > theta なら見送り。

    Σ(1/odds) <= 1 は「dutch 配分にすればどの点が来ても元返し以上」と同値。
    """
    def f(points, ref):
        return sum(1.0 / max(ref.get(p, 1e9), 1e-9) for p in points) > theta
    return f


EXCLUSIONS = {
    "なし": ex_none,
    "10倍以下3点以上を除外": make_ex_lowcount(10.0, 3),   # ← ユーザー修正案
    "10倍以下2点以上を除外": make_ex_lowcount(10.0, 2),   # ← 当初案
    "10倍以下4点以上を除外": make_ex_lowcount(10.0, 4),
    "book>1.00を除外": make_ex_book(1.00),
    "book>0.85を除外": make_ex_book(0.85),
    "book>0.75を除外": make_ex_book(0.75),
}


# ---------------------------------------------------------------- 有意性
def _outcomes(prepared, alloc_name, ex_name):
    """(race_key, 実質的中0/1, bet, ret) の列を返す。除外されたレースは含めない。"""
    al_fn, ex_fn = ALLOCATORS[alloc_name], EXCLUSIONS[ex_name]
    rows = []
    for r in prepared:
        if r["ref"] is not None and ex_fn(r["points"], r["ref"]):
            continue
        stakes = (alloc_equal(r["points"], r["fo"]) if r["ref"] is None
                  else al_fn(r["points"], r["ref"]))
        bet, ret = settle(r["points"], stakes, r["fo"], r.get("win"))
        rows.append((r["base"] + r["rank"], int(ret >= bet), bet, ret))
    return rows


def paired_test(prepared, a, b, ex="なし", n_boot=2000):
    """同一レース上で配分 a と b の実質的中率を比べる（McNemar + ブートストラップ）。"""
    import random
    ra = {k: v for k, v, _, _ in _outcomes(prepared, a, ex)}
    rb = {k: v for k, v, _, _ in _outcomes(prepared, b, ex)}
    keys = sorted(set(ra) & set(rb))
    b01 = sum(1 for k in keys if ra[k] == 0 and rb[k] == 1)
    b10 = sum(1 for k in keys if ra[k] == 1 and rb[k] == 0)
    diffs = []
    rnd = random.Random(0)
    for _ in range(n_boot):
        s = [keys[rnd.randrange(len(keys))] for _ in range(len(keys))]
        diffs.append(sum(rb[k] - ra[k] for k in s) / len(s))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    print(f"  {b:>9s} − {a:<9s}: 差 {100*(sum(rb[k] for k in keys)-sum(ra[k] for k in keys))/len(keys):+6.2f}pt "
          f"[{100*lo:+6.2f}, {100*hi:+6.2f}]  "
          f"改善{b01:>4d}件 / 悪化{b10:>4d}件  P(差>0)={100*sum(1 for d in diffs if d > 0)/n_boot:5.1f}%")


# ---------------------------------------------------------------- ドリフト
def drift_report(prepared):
    """朝オッズ → 最終オッズ の変化を帯別に見る。

    配分は朝に決めるのに決済は最終オッズなので、**朝より下がった点が
    そのままガミの取りこぼし**になる。どれだけ余裕（マージン）を積めば
    足りるのかをここで決める。
    """
    print(f"\n{'='*104}\n【朝→最終オッズのドリフト】買った点すべて\n{'='*104}")
    bands = [(0, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
    print(f"{'朝オッズ帯':>12s} {'点数':>7s} {'中央値 最終/朝':>14s} "
          f"{'下振れ率(<1.0)':>14s} {'>10%下振れ':>11s} {'>25%下振れ':>11s}")
    per_band = defaultdict(list)
    for r in prepared:
        for p in r["points"]:
            m, f = r["ref"].get(p), r["fo"].get(p)
            if m and f:
                per_band[next(i for i, (lo, hi) in enumerate(bands) if lo <= m < hi)].append(f / m)
    for i, (lo, hi) in enumerate(bands):
        v = sorted(per_band.get(i, []))
        if not v:
            continue
        med = v[len(v) // 2]
        lab = f"{lo:.0f}〜{'∞' if hi > 1e8 else f'{hi:.0f}'}倍"
        print(f"{lab:>12s} {len(v):>7,d} {med:>14.3f} "
              f"{100*sum(1 for x in v if x < 1.0)/len(v):>13.1f}% "
              f"{100*sum(1 for x in v if x < 0.90)/len(v):>10.1f}% "
              f"{100*sum(1 for x in v if x < 0.75)/len(v):>10.1f}%")

    # 「その日いちばん最初の発走まで時間があるか」で分けたいが start_at が無いので
    # レース番号を代理指標にする（後半レースほど朝8:00の板が薄い）。
    print("\n-- 参考: 買った点の最低オッズが朝→最終で下がった割合（レース単位） --")
    worse = sum(1 for r in prepared
                if min(r["fo"][p] for p in r["points"]) < min(r["ref"][p] for p in r["points"]))
    print(f"   最低オッズが下がったレース: {100*worse/max(len(prepared),1):.1f}%")


# ---------------------------------------------------------------- 出力
def show(title, rows, ranks_note=""):
    print(f"\n{'='*104}\n{title}  {ranks_note}\n{'='*104}")
    print(f"{'除外規則':<24s} {'配分':<10s} {'n':>6s} {'件/日':>7s} "
          f"{'的中%':>7s} {'実質的中%':>10s} {'ガミ率%':>8s} {'ROI%':>7s}")
    prev = None
    for r in rows:
        if prev is not None and r["exclusion"] != prev:
            print("-" * 104)
        prev = r["exclusion"]
        print(f"{r['exclusion']:<24s} {r['alloc']:<10s} {r['n']:>6,d} {r['per_day']:>7.2f} "
              f"{r['hit']:>7.2f} {r['real_hit']:>10.2f} {r['gami_of_hit']:>8.1f} "
              f"{r['roi']:>7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default="2024-07-01")
    ap.add_argument("--to", dest="dto", default="2026-08-06")
    ap.add_argument("--morning-from", default="2026-06-08")
    ap.add_argument("--ranks", default="")
    args = ap.parse_args()

    with get_connection() as conn:
        recs = build_from_picks(conn, args.dfrom, args.dto)
        recs += build_7c(args.dfrom, args.dto)
        if args.ranks:
            want = set(args.ranks.split(","))
            recs = [r for r in recs if r["rank"] in want]
        bases = sorted({r["base"] for r in recs})
        print(f"対象 {len(recs):,} レース（{args.dfrom}〜{args.dto}）")
        final_map = _load_odds(conn, bases, None)
        morning_map = _load_odds(conn, bases, "morning")
        top3 = _load_top3(conn, bases)

    for r in recs:
        r["win"] = top3.get(r["base"])

    by_rank = defaultdict(list)
    for r in recs:
        by_rank[r["rank"]].append(r)
    print("ランク別: " + " / ".join(f"{k} {len(v):,}" for k, v in sorted(by_rank.items())))
    print(f"最終オッズ盤面あり: {len(final_map):,} レース / "
          f"朝オッズ盤面あり: {len(morning_map):,} レース")

    # ---- IDEAL: 参照 = 最終オッズ（全期間・実装不能な上限）
    ideal_all = prepare(recs, final_map, final_map)
    show("【IDEAL】参照オッズ＝最終オッズ（実装不能・配分の上限値）",
         evaluate(ideal_all, ALLOCATORS, EXCLUSIONS), f"{args.dfrom}〜{args.dto}")

    # ---- REAL: 参照 = 朝オッズ。母集団は一度だけ確定させ IDEAL と共有する
    mrecs = [r for r in recs if r["date"] >= args.morning_from]
    pair = prepare(mrecs, final_map, morning_map)
    n_win = len([r for r in mrecs if r["base"] in morning_map])
    print(f"\n朝オッズ窓 {args.morning_from}〜{args.dto}: 対象 {len(mrecs):,} レース / "
          f"朝の盤面あり {n_win:,} / **全点そろう {len(pair):,}**"
          f"（{100*len(pair)/max(len(mrecs),1):.1f}%）")

    show("【REAL】参照＝朝オッズ（実装可能・決済は最終オッズ）",
         evaluate(pair, ALLOCATORS, EXCLUSIONS), f"n={len(pair)}")
    show("【同一母集団の IDEAL】上との差＝朝→直前のオッズ変動で失う分",
         evaluate(pair, ALLOCATORS, EXCLUSIONS, use_final_as_ref=True), f"n={len(pair)}")

    # ---- 本番相当: 朝の盤面が欠けたレースは均等割りへフォールバックして残す
    snap_days = {r["date"] for r in recs if r["base"] in morning_map}
    prod = prepare_prod([r for r in mrecs if r["date"] in snap_days],
                        final_map, morning_map)
    nfb = sum(1 for r in prod if r["ref"] is None)
    print(f"\n本番相当の母集団: {len(prod):,} レース（朝スナップショットのある日のみ）"
          f" / うち朝オッズ欠損で均等へフォールバック {nfb:,}"
          f"（{100*nfb/max(len(prod),1):.1f}%）")
    show("【本番相当】朝オッズ欠損レースは均等割りのまま残す",
         evaluate(prod, ALLOCATORS, EXCLUSIONS), f"n={len(prod)}")

    print("\n-- 実質的中率の対比較（本番相当・同一レース・2,000回ブートストラップ）--")
    for b in ("tilt10x2", "pow0.75", "dutch", "guard0.6"):
        paired_test(prod, "equal", b)
    print("-- 同じ比較を『10倍以下3点以上を除外』の母集団で --")
    for b in ("tilt10x2", "dutch"):
        paired_test(prod, "equal", b, ex="10倍以下3点以上を除外")

    # 朝オッズ窓は連続していない（6/8-6/18 と 7/16-8/6）。2ブロックに割って
    # 符号が揃うかを見る。揃わなければ単一窓の数字を信用しない。
    print("-- 窓を割った確認（朝スナップショットの2ブロック）--")
    for lo, hi in (("2026-06-08", "2026-06-30"), ("2026-07-01", "2026-08-06")):
        blk = [r for r in prod if lo <= r["date"] <= hi]
        print(f"  [{lo}〜{hi}] n={len(blk)}")
        for b in ("tilt10x2", "dutch"):
            paired_test(blk, "equal", b)

    drift_report(pair)

    # ---- ランク別（REAL）
    for rank in sorted({r["rank"] for r in pair}):
        sub = [r for r in pair if r["rank"] == rank]
        if len(sub) < 60:
            continue
        show(f"【REAL・{rank}】", evaluate(sub, ALLOCATORS, EXCLUSIONS), f"n={len(sub)}")


if __name__ == "__main__":
    main()
