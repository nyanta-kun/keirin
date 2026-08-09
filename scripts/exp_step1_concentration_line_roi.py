"""STEP1 / STEP1B: 相手(3着候補)の並べ替えと集中買いの実測比較。

STEP1  = 7A（--rank RANK_7A --points 1,2,5）
STEP1B = 7C（--rank RANK_7C --points 1,2,3）

相手を「pred_top3_pct順」と「自ライン優先」の2通りで並べ、上位k点へ集中買いしたときの
的中率・ガミ率・ROI を現行運用(実績)と比較する。

順序の定義
    prob: 相手を pred_top3_pct 降順（同値は車番昇順）
    line: 軸2車のいずれかと同じ line_group の相手を前に置き、その中で pred_top3_pct 降順
          → 残りを pred_top3_pct 降順

確率のソース（2系統を併記する）
    db: keirin.wt_entries.pred_top3_pct（本番テーブルの現行値。過去分は後日の
        バックフィルで書き換わっている可能性があり model-vintage look-ahead を含みうる）
    wf: data/exp_cache/wf_preds_*_f60_*.pkl（四半期ごとに再学習した walk-forward の
        pp3。当該レースより未来のデータを含まないため honest）

払戻
    勝ち三連複の最終払戻は keirin.picks_history.trio_payout（100円あたり）を使う。
    仕様書は keirin.wt_race_payouts を指すが、同テーブルは 2026-06-19〜07-04 の
    1,115レースしか無く全期間には使えない。trio_payout は対象行で非NULL・正値。
"""

from __future__ import annotations

import argparse
import csv
import glob
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from database import get_connection  # noqa: E402

BUDGET = 10_000
"""1レースあたりの予算(円)。"""

PER_POINT_CAP = 5_000
"""1点あたりの上限(円)。"""

_COMBO_RE = re.compile(r"^\s*(\d+)\s*=\s*(\d+)\s*-\s*([\d,\s]+)")


def parse_combo(pred_combo: str) -> tuple[int, int, list[int]] | None:
    """pred_combo("軸1=軸2-相手1,相手2,...") を (軸1, 軸2, 相手リスト) へ分解する。"""
    m = _COMBO_RE.match(pred_combo or "")
    if not m:
        return None
    axis1, axis2 = int(m.group(1)), int(m.group(2))
    partners = [int(x) for x in m.group(3).replace(" ", "").split(",") if x]
    return axis1, axis2, partners


def load_races(rank: str) -> list[dict]:
    """指定ランクの全レースを picks_history + wt_entries から取得する。

    race_key は "<wt_key>#<suffix>" 形式なので '#' の手前で wt_entries と突合する。
    """
    sql = """
        SELECT p.race_key,
               p.race_date,
               p.pred_combo,
               p.n_combos,
               p.hit,
               p.payout,
               p.bet_amount,
               p.trio_payout,
               e.frame_no,
               e.line_group,
               e.finish_order,
               e.pred_top3_pct
        FROM keirin.picks_history p
        JOIN keirin.wt_entries e
          ON e.race_key = split_part(p.race_key, '#', 1)
        WHERE p.rank = ?
        ORDER BY p.race_date, p.race_key, e.frame_no
    """
    by_race: dict[str, dict] = {}
    with get_connection() as con:
        cur = con.cursor()
        cur.execute(sql, (rank,))
        for r in cur.fetchall():
            key = r["race_key"]
            race = by_race.get(key)
            if race is None:
                race = by_race[key] = {
                    "race_key": key,
                    "wt_key": key.split("#", 1)[0],
                    "race_date": r["race_date"],
                    "pred_combo": r["pred_combo"],
                    "n_combos": int(r["n_combos"] or 0),
                    "hit": int(r["hit"] or 0),
                    "payout": int(r["payout"] or 0),
                    "bet_amount": int(r["bet_amount"] or 0),
                    "trio_payout": int(r["trio_payout"] or 0),
                    "entries": {},
                }
            race["entries"][int(r["frame_no"])] = {
                "line_group": r["line_group"],
                "finish_order": r["finish_order"],
                "p3_db": float(r["pred_top3_pct"]) if r["pred_top3_pct"] is not None else None,
            }
    return list(by_race.values())


def load_wf_pp3() -> dict[tuple[str, int], float]:
    """walk-forward の top3 確率 pp3 を (race_key, frame_no) -> pp3 で返す。"""
    out: dict[tuple[str, int], float] = {}
    for path in sorted(glob.glob(str(ROOT / "data" / "exp_cache" / "wf_preds_*_f60_*.pkl"))):
        with open(path, "rb") as fh:
            df = pickle.load(fh)
        for rk, fn, pp3 in zip(df["race_key"], df["frame_no"], df["pp3"], strict=True):
            out[(str(rk), int(fn))] = float(pp3)
    return out


def order_partners(
    partners: list[int],
    probs: dict[int, float],
    line_of: dict[int, int | None],
    axis_lines: set[int],
    mode: str,
) -> list[int]:
    """相手を指定の順序で並べる。

    mode="prob" は確率降順のみ。mode="line" は軸と同ラインを先頭ブロックへ寄せる。
    """
    def prob_key(fn: int) -> tuple[float, int]:
        return (-probs[fn], fn)

    if mode == "prob":
        return sorted(partners, key=prob_key)
    same = [fn for fn in partners if line_of.get(fn) is not None and line_of[fn] in axis_lines]
    rest = [fn for fn in partners if fn not in set(same)]
    return sorted(same, key=prob_key) + sorted(rest, key=prob_key)


def payouts_per_race(
    races: list[dict], prob_key: str, mode: str, n_points: int | None
) -> list[tuple[int, int]]:
    """レースごとの (払戻額, 賭け金) を並びで返す。

    n_points=None は「相手を全点買う」= 現行の点数（7C は 4 or 5 でレースごとに違う）。
    races は事前に共通母集団へ絞ってあること。
    """
    out: list[tuple[int, int]] = []
    for race in races:
        axis1, axis2, partners = race["parsed"]
        k = len(partners) if n_points is None else n_points
        stake_per_point = min(BUDGET // k, PER_POINT_CAP)
        bet = stake_per_point * k
        ordered = order_partners(
            partners, race["probs"][prob_key], race["line_of"], race["axis_lines"], mode
        )
        selected = set(ordered[:k])
        winners = race["winners"]
        won = 0
        # 軸2車が両方3着内 かつ 3頭目が採用済みの相手 なら的中
        if {axis1, axis2} <= winners:
            third = (winners - {axis1, axis2}).pop()
            if third in selected:
                won = race["trio_payout"] * stake_per_point // 100
        out.append((won, bet))
    return out


def simulate(
    races: list[dict], rank: str, prob_key: str, mode: str, n_points: int | None
) -> dict:
    """指定の確率ソース・順序・点数でシミュレーションし集計値を返す。"""
    pairs = payouts_per_race(races, prob_key, mode, n_points)
    n = len(pairs)
    total_bet = sum(b for _, b in pairs)
    total_payout = sum(w for w, _ in pairs)
    n_hit = sum(1 for w, _ in pairs if w)
    n_gami_stake = sum(1 for w, b in pairs if w and w < b)
    n_gami_budget = sum(1 for w, _ in pairs if w and w < BUDGET)
    stake_per_point = min(BUDGET // n_points, PER_POINT_CAP) if n_points else 0

    return {
        "rank": rank,
        "prob_source": prob_key.replace("p3_", ""),
        "order": mode,
        "n_points": n_points if n_points else "full(現行点数)",
        "n": n,
        "stake_per_point": stake_per_point,
        "bet_per_race": round(total_bet / n) if n else 0,
        "hit_rate": round(n_hit / n, 4) if n else 0.0,
        "gami_rate_vs_stake": round(n_gami_stake / n_hit, 4) if n_hit else 0.0,
        "gami_rate_vs_budget": round(n_gami_budget / n_hit, 4) if n_hit else 0.0,
        "roi": round(total_payout / total_bet, 4) if total_bet else 0.0,
        "total_bet": total_bet,
        "total_payout": total_payout,
    }


def baseline(races: list[dict], rank: str) -> dict:
    """現行運用(総流し・傾斜配分の実績値)を picks_history の実績から集計する。"""
    n = n_hit = n_gami = 0
    total_bet = total_payout = 0
    for race in races:
        if not race["bet_amount"]:
            continue
        n += 1
        total_bet += race["bet_amount"]
        total_payout += race["payout"]
        if race["hit"]:
            n_hit += 1
            if race["payout"] < race["bet_amount"]:
                n_gami += 1
    return {
        "rank": rank,
        "prob_source": "actual",
        "order": "current(総流し・傾斜配分)",
        "n_points": "full(現行点数)",
        "n": n,
        "stake_per_point": 0,
        "bet_per_race": round(total_bet / n) if n else 0,
        "hit_rate": round(n_hit / n, 4) if n else 0.0,
        "gami_rate_vs_stake": round(n_gami / n_hit, 4) if n_hit else 0.0,
        "gami_rate_vs_budget": round(n_gami / n_hit, 4) if n_hit else 0.0,
        "roi": round(total_payout / total_bet, 4) if total_bet else 0.0,
        "total_bet": total_bet,
        "total_payout": total_payout,
    }


def third_position_stats(races: list[dict], prob_key: str, mode: str) -> dict[int, int]:
    """軸2車が的中したレースで、勝ち目の3頭目が相手の何番目かの分布を返す。"""
    dist: dict[int, int] = defaultdict(int)
    for race in races:
        axis1, axis2, partners = race["parsed"]
        winners = race["winners"]
        if not {axis1, axis2} <= winners:
            continue
        ordered = order_partners(
            partners, race["probs"][prob_key], race["line_of"], race["axis_lines"], mode
        )
        third = (winners - {axis1, axis2}).pop()
        if third in ordered:
            dist[ordered.index(third) + 1] += 1
    return dict(dist)


def build_population(races: list[dict], prob_keys: tuple[str, ...]) -> list[dict]:
    """全ての確率ソースが揃い、1〜3着が確定し、pred_combo が解釈できるレースだけ残す。

    ソース間を同一母集団で比較するために必須。母集団が違うと ROI 差が順序の効果か
    期間の違いか区別できない。
    """
    out: list[dict] = []
    for race in races:
        parsed = parse_combo(race["pred_combo"])
        if parsed is None:
            continue
        axis1, axis2, partners = parsed
        winners = {fn for fn, e in race["entries"].items() if e["finish_order"] in (1, 2, 3)}
        if len(winners) != 3:
            continue  # 落車・失格等で1〜3着が確定しないレースは除外
        probs: dict[str, dict[int, float]] = {}
        ok = True
        for key in prob_keys:
            p = {fn: e[key] for fn, e in race["entries"].items() if e[key] is not None}
            if len(p) != len(race["entries"]) or not set(partners) <= p.keys():
                ok = False
                break
            probs[key] = p
        if not ok:
            continue
        line_of = {fn: e["line_group"] for fn, e in race["entries"].items()}
        race["parsed"] = parsed
        race["winners"] = winners
        race["probs"] = probs
        race["line_of"] = line_of
        race["axis_lines"] = {line_of[a] for a in (axis1, axis2) if line_of.get(a) is not None}
        out.append(race)
    return out


def paired_bootstrap(
    races: list[dict],
    prob_key: str,
    n_points: int | None,
    n_boot: int = 2000,
    seed: int = 20260809,
) -> tuple[float, float, float]:
    """line − prob の ROI 差をレース単位ペアードブートストラップし (差, lo, hi) を返す。"""
    import random

    a = [w for w, _ in payouts_per_race(races, prob_key, "prob", n_points)]
    pairs_b = payouts_per_race(races, prob_key, "line", n_points)
    b = [w for w, _ in pairs_b]
    bets = [bt for _, bt in pairs_b]
    n = len(a)
    if not n:
        return 0.0, 0.0, 0.0
    point = (sum(b) - sum(a)) / sum(bets)

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        num = sum(b[i] - a[i] for i in idx)
        den = sum(bets[i] for i in idx)
        diffs.append(num / den if den else 0.0)
    diffs.sort()
    return point, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def bootstrap_vs_baseline(
    races: list[dict],
    prob_key: str,
    mode: str,
    n_points: int | None,
    n_boot: int = 2000,
    seed: int = 20260809,
) -> tuple[float, float, float]:
    """変種 − 現行実績 の ROI 差をレース単位ペアードブートストラップし (差, lo, hi) を返す。

    現行は picks_history の実績（傾斜配分込み）なので、これが採否の決定的な比較になる。
    """
    import random

    pairs = payouts_per_race(races, prob_key, mode, n_points)
    var_pay = [w for w, _ in pairs]
    var_bet = [b for _, b in pairs]
    base_pay = [r["payout"] for r in races]
    base_bet = [r["bet_amount"] for r in races]
    n = len(races)
    if not n:
        return 0.0, 0.0, 0.0

    def roi_diff(idx: list[int]) -> float:
        vb = sum(var_bet[i] for i in idx)
        bb = sum(base_bet[i] for i in idx)
        if not vb or not bb:
            return 0.0
        return sum(var_pay[i] for i in idx) / vb - sum(base_pay[i] for i in idx) / bb

    point = roi_diff(list(range(n)))
    rng = random.Random(seed)
    diffs = sorted(roi_diff([rng.randrange(n) for _ in range(n)]) for _ in range(n_boot))
    return point, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def main() -> None:
    """全パターンを実測して CSV へ書き出し、要点を表示する。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", default="RANK_7A", help="picks_history.rank")
    ap.add_argument("--points", default="1,2,5", help="集中買いの点数(カンマ区切り)")
    ap.add_argument("--out", default="step1_7a_line_roi.csv", help="data/exports 配下の出力名")
    args = ap.parse_args()

    point_counts: list[int | None] = [int(x) for x in args.points.split(",") if x.strip()]
    point_counts.append(None)  # 相手を全点買う=現行点数（等分配分）

    raw = load_races(args.rank)
    wf = load_wf_pp3()
    for race in raw:
        for fn, e in race["entries"].items():
            e["p3_wf"] = wf.get((race["wt_key"], fn))

    prob_keys = ("p3_db", "p3_wf")
    races = build_population(raw, prob_keys)
    dates = sorted(r["race_date"] for r in races)
    print(f"母集団: 全{args.rank} {len(raw)}R → 共通母集団 {len(races)}R ({dates[0]}〜{dates[-1]})")
    print("  ※wf(walk-forward)確率が存在するレースに合わせて全条件を同一母集団に揃えている")
    combos = defaultdict(int)
    for r in races:
        combos[len(r["parsed"][2])] += 1
    print(f"  相手点数の内訳: {dict(sorted(combos.items()))}\n")

    rows = [baseline(races, args.rank)]
    for prob_key in prob_keys:
        for mode in ("prob", "line"):
            for k in point_counts:
                rows.append(simulate(races, args.rank, prob_key, mode, k))

    out_path = ROOT / "data" / "exports" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'src':>6} {'order':>5} {'pts':>4} {'n':>6} {'hit%':>7} {'gami%':>7} {'ROI':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        pts = r["n_points"] if isinstance(r["n_points"], int) else "full"
        print(
            f"{r['prob_source']:>6} {r['order'][:5]:>5} {pts:>4} {r['n']:>6} "
            f"{r['hit_rate'] * 100:>6.2f}% {r['gami_rate_vs_stake'] * 100:>6.2f}% "
            f"{r['roi'] * 100:>6.2f}%"
        )

    print("\n[line − prob の ROI 差・レース単位ペアードブートストラップ 95%CI]")
    for prob_key in prob_keys:
        for k in point_counts:
            d, lo, hi = paired_bootstrap(races, prob_key, k)
            sig = "有意" if (lo > 0 or hi < 0) else "ns"
            label = f"{k}点" if k else "full"
            print(
                f"  {prob_key.replace('p3_', ''):>2}/{label:<5} {d * 100:+6.2f}pt  "
                f"[{lo * 100:+6.2f}, {hi * 100:+6.2f}]  {sig}"
            )

    print("\n[各変種 − 現行実績 の ROI 差・95%CI（採否の決定的な比較）]")
    for prob_key in prob_keys:
        for mode in ("prob", "line"):
            for k in point_counts:
                d, lo, hi = bootstrap_vs_baseline(races, prob_key, mode, k)
                sig = "有意" if (lo > 0 or hi < 0) else "ns"
                label = f"{k}点" if k else "full"
                print(
                    f"  {prob_key.replace('p3_', ''):>2}/{mode:<4}/{label:<5} {d * 100:+6.2f}pt  "
                    f"[{lo * 100:+6.2f}, {hi * 100:+6.2f}]  {sig}"
                )

    print("\n[半期別の安定性（best cell が窓をまたいで持つか）]")
    windows: dict[str, list[dict]] = defaultdict(list)
    for r in races:
        y, m = r["race_date"][:4], int(r["race_date"][5:7])
        windows[f"{y}H{1 if m <= 6 else 2}"].append(r)
    for prob_key in prob_keys:
        for k in point_counts:
            if k is None:
                continue
            cells = []
            for wname in sorted(windows):
                sub = windows[wname]
                d, lo, hi = bootstrap_vs_baseline(sub, prob_key, "line", k, n_boot=800)
                cells.append(f"{wname} {d * 100:+6.2f}")
            src = prob_key.replace("p3_", "")
            print(f"  {src:>2}/line/{k}点 vs 現行: " + " | ".join(cells))

    print("\n[勝ち目の3頭目が相手の何番目か]")
    for prob_key in prob_keys:
        for mode in ("prob", "line"):
            dist = third_position_stats(races, prob_key, mode)
            tot = sum(dist.values())
            share = " ".join(
                f"{i}位 {dist.get(i, 0) / tot * 100:4.1f}%" for i in range(1, 6)
            ) if tot else "n/a"
            print(f"  {prob_key.replace('p3_', ''):>2}/{mode:<4} n={tot:<5} {share}")

    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
