"""9車ランク（9S/9A）の相手足切り検証（2026-08-07・ユーザー指摘が起点）。

きっかけ: 和歌山9R 2026-08-07（9S・三連複 1=9-5,3,2,8,7,4,6 の7点総流し）。
  1-5-9 で的中したが三連複 7.0倍 ＝ 7点×1,400円 に対し払戻 9,800円 で **ちょうど100%**。
  「車6（複勝率3.4%・競走得点95.2）のように明らかに力不足の選手がいるので、
    事前にカットできれば ROI を上げられたはず」

7C では同じ足切り（3着内率15%以上・`rank_7c_select_legs`）が採用済みだが、
**9車ランクには入っていない**（9S/9A は相手7車の総流し）。7SS/7S/7A への
適用は既に検証して不採用（commit e600abb）だが、9車は未検証。

⚠️ **予測は必ず vintage walk-forward を使う**（`wf_preds9_*.pkl`）。
   `wt_entries.pred_top3_pct` は 2026-07-19 に追加された列で、過去分は後から
   backfill されている＝そのレースより未来を知っているモデルの出力。
   これで足切りを測ると必ず良く見える（model-vintage look-ahead）。
⚠️ 読み取り専用。
"""
from __future__ import annotations

import glob
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402

BUDGET = 10_000
UNIT = 100
_SEP_RE = re.compile(r"[-=]")
RANKS = ("RANK_9S", "RANK_9A")


def load_pp3():
    """race_key -> {frame_no: pp3}（honest walk-forward）。"""
    out = defaultdict(dict)
    for f in sorted(glob.glob(str(REPO / "data" / "exp_cache" / "wf_preds9_*.pkl"))):
        df = pickle.load(open(f, "rb"))
        for rk, fn, p in zip(df["race_key"], df["frame_no"], df["pp3"]):
            out[rk][int(fn)] = float(p)
    return out


def load_board(conn, keys):
    out = defaultdict(dict)
    keys = list(keys)
    for i in range(0, len(keys), 800):
        inlist = ",".join(f"'{k}'" for k in keys[i:i + 800])
        for rk, comb, od in conn.execute(
            f"SELECT race_key, combination, odds_value FROM keirin.wt_odds "
            f"WHERE bet_type='trio' AND race_key IN ({inlist})"
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


CUTS = [
    ("総流し（現行）",  lambda legs, p3: legs),
    ("p3>=8%",   lambda legs, p3: [x for x in legs if p3.get(x, 0) >= 0.08]),
    ("p3>=10%",  lambda legs, p3: [x for x in legs if p3.get(x, 0) >= 0.10]),
    ("p3>=15%",  lambda legs, p3: [x for x in legs if p3.get(x, 0) >= 0.15]),
    ("p3>=20%",  lambda legs, p3: [x for x in legs if p3.get(x, 0) >= 0.20]),
    ("上位6車",   lambda legs, p3: sorted(legs, key=lambda x: -p3.get(x, 0))[:6]),
    ("上位5車",   lambda legs, p3: sorted(legs, key=lambda x: -p3.get(x, 0))[:5]),
    ("上位4車",   lambda legs, p3: sorted(legs, key=lambda x: -p3.get(x, 0))[:4]),
]


def main():
    pp3 = load_pp3()
    with get_connection() as conn:
        rows = list(conn.execute(f"""
            SELECT p.race_date, p.race_key, p.rank, p.pred_combo
            FROM keirin.picks_history p
            JOIN keirin.wt_races r ON r.race_key = split_part(p.race_key,'#',1)
            WHERE p.rank IN ({",".join("'" + r + "'" for r in RANKS)})
              AND p.race_date >= '2024-07-01' AND p.race_date <= '2026-08-06'
              AND COALESCE(r.cancel,0)=0 AND r.n_entries=9
        """))
        recs = []
        for r in rows:
            base = r["race_key"].split("#")[0]
            axes, legs = parse(r["pred_combo"])
            if len(axes) != 2 or len(legs) < 3 or base not in pp3:
                continue
            recs.append(dict(date=r["race_date"], base=base, rank=r["rank"],
                             axes=axes, legs=legs, p3=pp3[base]))
        bases = sorted({r["base"] for r in recs})
        board = load_board(conn, bases)
        top3 = load_top3(conn, bases)

    print(f"9車ランク {len(recs):,} レース（walk-forward 予測あり）"
          f" / 盤面あり {len(board):,} / 結果あり {len(top3):,}")

    # n が薄い（500台）ので窓を割って符号の一貫性まで見る。
    # 単一窓の ROI 差は当たり方の裾で簡単に±10pt動く（memory の再三の教訓）。
    for wname, wfilter in (("全期間", lambda d: True),
                           ("〜2025-06", lambda d: d <= "2025-06-30"),
                           ("2025-07〜", lambda d: d > "2025-06-30")):
        run(recs, board, top3, wname, wfilter)


def run(recs, board, top3, wname, wfilter):
    recs = [r for r in recs if wfilter(r["date"])]
    print(f"\n■ {wname}")
    print(f"{'足切り':<16s} {'n':>5s} {'平均点':>7s} {'的中%':>7s} {'実質的中%':>10s} "
          f"{'ROI%':>7s} {'取りこぼし%':>11s} {'的中時 平均倍率':>15s}")
    for name, fn in CUTS:
        n = hit = rhit = 0
        bet_t = ret_t = 0
        pts_t = 0
        miss_by_cut = 0          # 総流しなら当たっていたのに足切りで落とした
        odds_sum = 0.0
        for r in recs:
            fo = board.get(r["base"], {})
            win = top3.get(r["base"])
            if not fo or win is None:
                continue
            all_pts = [frozenset({*r["axes"], t}) for t in r["legs"]]
            if any(p not in fo for p in all_pts):
                continue
            kept = fn(r["legs"], r["p3"])
            if len(kept) < 2:
                continue          # 相手が1点未満まで削れる設計は比較対象外
            pts = [frozenset({*r["axes"], t}) for t in kept]
            n += 1
            pts_t += len(pts)
            stake = BUDGET // len(pts) // UNIT * UNIT
            bet = stake * len(pts)
            ret = int(stake * fo[win]) // 10 * 10 if win in pts else 0
            bet_t += bet
            ret_t += ret
            if ret:
                hit += 1
                odds_sum += fo[win]
                if ret >= bet:
                    rhit += 1
            elif win in all_pts:
                miss_by_cut += 1
        if not n:
            continue
        print(f"{name:<16s} {n:>5,d} {pts_t/n:>7.2f} {100*hit/n:>7.2f} {100*rhit/n:>10.2f} "
              f"{100*ret_t/bet_t:>7.1f} {100*miss_by_cut/n:>11.2f} "
              f"{odds_sum/max(hit,1):>15.2f}")

    print("\n※ 取りこぼし% = 総流しなら的中していたのに足切りで落としたレースの割合")
    print("※ 配分は均等（100円単位切り捨て）。配分の効果は exp_netkeirin_gami_allocation.py 側で測る")


if __name__ == "__main__":
    main()
