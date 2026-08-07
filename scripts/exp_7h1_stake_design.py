"""7H1 の金額配分を見直す（2026-08-07・ユーザー設計）。

現状（picks_history 2,425R）: 的中 18.47% に対し **実質的中 10.06%（ガミ率45.5%）**。
netkeirin は払戻 < 投資を不的中として数えるので、表示は 18% でなく 10% になる。

ユーザー指示:
  「三連複のみの的中でガミを無くしたいが安い目もあり困難と予想される。
    **三連複だけの的中は半分（5,000円戻ればよし）**とし、**三連単は均等買い**、
    **端数を三連複の安い目へ追加**」

## 現行の買い方

    三連単 7,500円 / 三連複 2,500円（`RANK_7H1_BUDGET_TF/TRIO`）をそれぞれ
    点数で均等割り・100円単位で切り捨て。切り捨て分は**使われず捨てられている**。

## 提案する買い方

    1. 三連単は 7,500円 を点数で均等割り（現行どおり）
    2. 三連複は **各目の払戻が 5,000円** になるよう 1/オッズ比例で配分
    3. 余った端数（三連単の切り捨て＋三連複の余り）は**三連複の安い目**
       （＝低オッズ＝最も当たりやすい目）へ寄せる

まず「5,000円戻し」が予算内で成立するのかを数える。成立しないレースが多ければ
設計を見直す必要がある。

⚠️ 配分にオッズが要る。本番は朝の板しか無く、7H1 は穴狙いで薄い目を買うため
   欠損しやすい。ここでは**最終オッズ**で設計の成立性を測る（先読みなので
   ROI/的中の数値は上限として読むこと）。
⚠️ 読み取り専用。
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.stake_allocation import allocate_budget  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_7H1_BUDGET_CAP, RANK_7H1_BUDGET_TF, RANK_7H1_UNIT, rank_7h1_unit,
)

# 2026-08-07 ユーザー指定: 三連単は1点500円の均等（7車）。残りを三連複へ回し、
# **入稿時点のオッズで払戻が等しくなるよう配分**し、端数は最も払戻が低い目へ。
TF_UNIT_7CAR = 500

TRIO_TARGET = 5000        # 三連複だけ当たったときに戻したい額（＝投資の半分）
_SEP_RE = re.compile(r"[-=]")


def load_board(conn, keys, bet_type):
    out = defaultdict(dict)
    keys = list(keys)
    for i in range(0, len(keys), 500):
        inlist = ",".join(f"'{k}'" for k in keys[i:i + 500])
        for rk, comb, od in conn.execute(
            f"SELECT race_key, combination, odds_value FROM keirin.wt_odds "
            f"WHERE bet_type='{bet_type}' AND race_key IN ({inlist})"
        ):
            if od is None or not (0 < float(od) < 9000):
                continue
            if bet_type == "trio":
                try:
                    key = frozenset(int(x) for x in _SEP_RE.split(str(comb)))
                except ValueError:
                    continue
                if len(key) == 3:
                    out[rk][key] = float(od)
            else:
                out[rk][str(comb).replace("=", "-")] = float(od)
    return out


def parse_7h1(pred_combo: str):
    """'三複:1=2=6,… / 三単:1-2-3,…' → (trio目のリスト, 三単目のリスト)。"""
    trio, tf = [], []
    for part in str(pred_combo or "").split("/"):
        part = part.strip()
        if part.startswith("三複:"):
            for c in part[3:].split(","):
                cars = [int(x) for x in _SEP_RE.split(c.strip()) if x.strip().isdigit()]
                if len(cars) == 3:
                    trio.append(frozenset(cars))
        elif part.startswith("三単:"):
            for c in part[3:].split(","):
                s = c.strip()
                if len(s.split("-")) == 3:
                    tf.append(s)
    return trio, tf


def allocate_flat_tf(trio_pts, trio_odds, n_tf):
    """ユーザー指定の新方式: 三連単は 500円/点 均等、残りを三連複へ払戻均等配分。

    端数の寄せ先（払戻が最小の点）は `allocate_budget` が持っている規則をそのまま使う。
    """
    uf = TF_UNIT_7CAR
    rest = RANK_7H1_BUDGET_CAP - uf * n_tf
    if not trio_pts or rest < RANK_7H1_UNIT * len(trio_pts):
        return uf, {p: RANK_7H1_UNIT for p in trio_pts}
    w = {p: 1.0 / trio_odds[p] for p in trio_pts}
    return uf, allocate_budget(w, budget=rest, unit=RANK_7H1_UNIT)


def allocate_proposed(trio_pts, trio_odds, n_tf):
    """提案どおりに配分する。returns (三単1点あたり, {三連複の目: 金額})。"""
    uf = rank_7h1_unit(RANK_7H1_BUDGET_TF, n_tf)
    rest = RANK_7H1_BUDGET_CAP - uf * n_tf          # 三連複へ回せる額（切り捨て分を含む）
    if not trio_pts:
        return uf, {}
    # 各目 TRIO_TARGET 円が戻る額（100円単位・切り上げ）
    need = {p: max(RANK_7H1_UNIT,
                   -(-int(TRIO_TARGET) // int(trio_odds[p] * RANK_7H1_UNIT))
                   * RANK_7H1_UNIT)
            for p in trio_pts}
    if sum(need.values()) <= rest:
        stakes = dict(need)
        left = rest - sum(stakes.values())
        # 端数は**安い目**（低オッズ＝最も当たりやすい目）へ寄せる
        order = sorted(trio_pts, key=lambda p: trio_odds[p])
        i = 0
        while left >= RANK_7H1_UNIT and order:
            stakes[order[i % len(order)]] += RANK_7H1_UNIT
            left -= RANK_7H1_UNIT
            i += 1
        return uf, stakes
    # 予算内に収まらない: 全点の払戻をそろえる形（1/オッズ比例）で最善を尽くす
    w = {p: 1.0 / trio_odds[p] for p in trio_pts}
    tot = sum(w.values())
    units = {p: max(1, int((rest // RANK_7H1_UNIT) * w[p] / tot)) for p in trio_pts}
    while sum(units.values()) * RANK_7H1_UNIT > rest and max(units.values()) > 1:
        units[max(units, key=lambda p: units[p])] -= 1
    return uf, {p: units[p] * RANK_7H1_UNIT for p in trio_pts}


def main():
    with get_connection() as conn:
        rows = list(conn.execute("""
            SELECT race_key, race_date, pred_combo FROM keirin.picks_history
            WHERE rank = 'RANK_7H1' ORDER BY race_date
        """))
        recs = []
        for r in rows:
            base = r["race_key"].split("#")[0]
            trio, tf = parse_7h1(r["pred_combo"])
            if trio:
                recs.append((base, r["race_date"], trio, tf))
        bases = sorted({x[0] for x in recs})
        tb = load_board(conn, bases, "trio")
        mb = defaultdict(dict)
        for i in range(0, len(bases), 500):
            inlist = ",".join(f"'{k}'" for k in bases[i:i + 500])
            for rk, comb, od in conn.execute(
                f"SELECT race_key, combination, odds_value FROM keirin.wt_odds_snapshot "
                f"WHERE snapshot_type='morning' AND bet_type='trio' "
                f"AND race_key IN ({inlist})"
            ):
                if od is None or not (0 < float(od) < 9000):
                    continue
                try:
                    key = frozenset(int(x) for x in _SEP_RE.split(str(comb)))
                except ValueError:
                    continue
                if len(key) == 3:
                    mb[rk][key] = float(od)
        fb = load_board(conn, bases, "trifecta")
        fin = defaultdict(dict)
        for i in range(0, len(bases), 500):
            inlist = ",".join(f"'{k}'" for k in bases[i:i + 500])
            for rk, fno, fo in conn.execute(
                f"SELECT race_key, frame_no, finish_order FROM keirin.wt_entries "
                f"WHERE race_key IN ({inlist})"
            ):
                if fo is not None and 1 <= int(fo) <= 3:
                    fin[rk][int(fo)] = int(fno)

    feas = tot = 0
    agg = {k: dict(n=0, bet=0, ret=0, hit=0, rhit=0, trio_only=0, trio_only_ok=0)
           for k in ("現行", "5000円戻し", "新方式(三単500円)", "新方式(朝オッズ)")}
    morning_ok = morning_win = 0
    for base, _d, trio, tf in recs:
        to, fo_ = tb.get(base, {}), fb.get(base, {})
        f = fin.get(base, {})
        if set(f) != {1, 2, 3} or any(p not in to for p in trio):
            continue
        top3 = frozenset(f.values())
        exact = f"{f[1]}-{f[2]}-{f[3]}"
        tot += 1
        need = sum(max(RANK_7H1_UNIT,
                       -(-TRIO_TARGET // int(to[p] * RANK_7H1_UNIT)) * RANK_7H1_UNIT)
                   for p in trio)
        uf0 = rank_7h1_unit(RANK_7H1_BUDGET_TF, len(tf))
        if need <= RANK_7H1_BUDGET_CAP - uf0 * len(tf):
            feas += 1

        mo = mb.get(base, {})
        has_m = bool(mo) and all(p in mo for p in trio)
        if base in mb:
            morning_win += 1
            if has_m:
                morning_ok += 1
        for name in ("現行", "5000円戻し", "新方式(三単500円)", "新方式(朝オッズ)"):
            if name == "新方式(朝オッズ)" and not has_m:
                continue          # 朝オッズが無い期間は母集団外（別集計）
            if name == "現行":
                ut = rank_7h1_unit(2500, len(trio))
                uf = uf0
                st = {p: ut for p in trio}
            elif name == "5000円戻し":
                uf, st = allocate_proposed(trio, to, len(tf))
            elif name == "新方式(三単500円)":
                uf, st = allocate_flat_tf(trio, to, len(tf))
            else:
                uf, st = allocate_flat_tf(trio, mo, len(tf))    # 配分は朝・精算は最終
            bet = sum(st.values()) + uf * len(tf)
            ret = 0
            if top3 in st:
                ret += int(st[top3] * to[top3]) // 10 * 10
            if exact in tf and exact in fo_:
                ret += int(uf * fo_[exact]) // 10 * 10
            a = agg[name]
            a["n"] += 1
            a["bet"] += bet
            a["ret"] += ret
            a["max"] = max(a.get("max", 0), ret)
            if ret:
                a["hit"] += 1
                if ret >= bet:
                    a["rhit"] += 1
            if top3 in st and exact not in tf:        # 三連複だけ当たった
                a["trio_only"] += 1
                if ret >= TRIO_TARGET:
                    a["trio_only_ok"] += 1

    print(f"7H1 {tot:,} レース（三連複盤面あり・結果確定）")
    print(f"  朝オッズのある日: {morning_win:,} / うち買う目すべてに値がある: "
          f"{morning_ok:,}（{100*morning_ok/max(morning_win,1):.1f}%）")
    print(f"  **「三連複の各目 5,000円戻し」が予算内で成立: {feas:,} "
          f"（{100*feas/max(tot,1):.1f}%）**\n")
    print(f"{'配分':<6s} {'n':>6s} {'的中%':>7s} {'実質的中%':>10s} {'ROI%':>7s} "
          f"{'平均投資':>9s} | {'三複のみ的中':>11s} {'うち5000円以上':>13s}")
    for name, a in agg.items():
        n = a["n"] or 1
        print(f"{name:<6s} {a['n']:>6,d} {100*a['hit']/n:>7.2f} {100*a['rhit']/n:>10.2f} "
              f"{100*a['ret']/max(a['bet'],1):>7.2f} {a['bet']/n:>9,.0f} | "
              f"{a['trio_only']:>11,d} "
              f"{100*a['trio_only_ok']/max(a['trio_only'],1):>12.1f}% "
              f"| 最高払戻 {a.get('max',0):>9,d}円")


if __name__ == "__main__":
    main()
