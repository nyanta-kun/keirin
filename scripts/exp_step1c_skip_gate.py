"""STEP1C: 的中率ランク(7C/7A/7S)の「低配当レース見送り」ゲート検証。

ゲート候補
    (a) top2_sum 上限 — 軸2車の pred_top3_pct 合計。軸が強すぎる=低配当と見て切る。
    (b) 朝オッズ最人気三連複オッズ 下限 — レース最人気が安すぎる=低配当と見て切る。

見送り後に残ったレースを「現行の総流し・傾斜配分のまま」買った場合の実績を
picks_history の payout / bet_amount からそのまま集計する。

スケールについての注記
    仕様書の (a) 掃引例 0.70/0.75/0.80/0.85/0.90 は **データのスケールと合わない**。
    keirin.wt_entries.pred_top3_pct は 0〜100(%) で格納されており（実測 0.7〜99.6・平均42.7)、
    軸2車の合計は概ね 85〜170 になる。さらに 7C は選抜条件として既に top2_sum>=144 を
    要求している。そのためランクごとの実分布の分位点で掃引し、仕様書の literal 値が
    どこに当たるかは分布表で確認できるようにした。

確率ソース
    wf: data/exp_cache/wf_preds_*_f60_*.pkl の pp3（walk-forward・honest）。ゲート判定は
        本来「朝の時点で得られる予測」なので、後日バックフィルで書き換わった DB 値で
        測ると採否が変わる（STEP1 で実証済み）。wf を主・db を参考として併記する。

掃引窓と確認窓
    閾値を選ぶ窓と効果を確認する窓を分ける（本プロジェクトで確立済みの手順）。
    既定は sweep=〜2025-12-31 / confirm=2026-01-01〜。
"""

from __future__ import annotations

import argparse
import csv
import glob
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from database import get_connection  # noqa: E402

RANKS = ("RANK_7C", "RANK_7A", "RANK_7S")

SWEEP_END = "2025-12-31"
"""掃引窓の終わり（この日までで閾値を選ぶ）。"""

ODDS_FLOORS = (3.0, 4.0, 5.0, 6.0)
"""仕様書指定の朝オッズ下限スイープ。"""

_COMBO_RE = re.compile(r"^\s*(\d+)\s*=\s*(\d+)\s*-\s*([\d,\s]+)")


def parse_axes(pred_combo: str) -> tuple[int, int] | None:
    """pred_combo から軸2車の車番を取り出す。"""
    m = _COMBO_RE.match(pred_combo or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_wf_pp3() -> dict[tuple[str, int], float]:
    """walk-forward の top3 確率 pp3 を (race_key, frame_no) -> pp3(0〜1) で返す。"""
    out: dict[tuple[str, int], float] = {}
    for path in sorted(glob.glob(str(ROOT / "data" / "exp_cache" / "wf_preds_*_f60_*.pkl"))):
        with open(path, "rb") as fh:
            df = pickle.load(fh)
        for rk, fn, pp3 in zip(df["race_key"], df["frame_no"], df["pp3"], strict=True):
            out[(str(rk), int(fn))] = float(pp3)
    return out


def load_races(rank: str, wf: dict[tuple[str, int], float]) -> list[dict]:
    """指定ランクのレースを、ゲート判定に必要な指標つきで返す。"""
    sql = """
        SELECT p.race_key,
               p.race_date,
               p.pred_combo,
               p.hit,
               p.payout,
               p.bet_amount,
               e.frame_no,
               e.pred_top3_pct
        FROM keirin.picks_history p
        JOIN keirin.wt_entries e
          ON e.race_key = split_part(p.race_key, '#', 1)
        WHERE p.rank = ?
        ORDER BY p.race_date, p.race_key
    """
    odds_sql = """
        SELECT split_part(p.race_key, '#', 1) AS wt_key,
               MIN(s.odds_value) AS fav_odds
        FROM keirin.picks_history p
        JOIN keirin.wt_odds_snapshot s
          ON s.race_key = split_part(p.race_key, '#', 1)
        WHERE p.rank = ?
          AND s.snapshot_type = 'morning'
          AND s.bet_type = 'trio'
          AND s.odds_value > 0
        GROUP BY 1
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
                    "hit": int(r["hit"] or 0),
                    "payout": int(r["payout"] or 0),
                    "bet_amount": int(r["bet_amount"] or 0),
                    "p3": {},
                }
            if r["pred_top3_pct"] is not None:
                race["p3"][int(r["frame_no"])] = float(r["pred_top3_pct"])
        cur.execute(odds_sql, (rank,))
        fav = {r["wt_key"]: float(r["fav_odds"]) for r in cur.fetchall()}

    out: list[dict] = []
    for race in by_race.values():
        axes = parse_axes(race["pred_combo"])
        if axes is None or not race["bet_amount"]:
            continue
        a1, a2 = axes
        if a1 not in race["p3"] or a2 not in race["p3"]:
            continue
        race["top2_db"] = race["p3"][a1] + race["p3"][a2]
        w1 = wf.get((race["wt_key"], a1))
        w2 = wf.get((race["wt_key"], a2))
        # wf は 0〜1 なので db(0〜100) と同じ尺度へ揃える
        race["top2_wf"] = (w1 + w2) * 100 if (w1 is not None and w2 is not None) else None
        race["fav_odds"] = fav.get(race["wt_key"])
        out.append(race)
    return out


def agg(races: list[dict], all_dates: int) -> dict:
    """残存レース群の実績（現行の総流し・傾斜配分そのまま）を集計する。

    roi_drop1 は最大払戻の1件を除いた ROI。少数の高額配当への依存を見るため
    （本ランクの払戻分布は少数の万車券に偏る）。
    """
    n = len(races)
    bet = sum(r["bet_amount"] for r in races)
    pay = sum(r["payout"] for r in races)
    hits = [r for r in races if r["hit"]]
    gami = sum(1 for r in hits if r["payout"] < r["bet_amount"])
    top = max((r["payout"] for r in races), default=0)
    top_bet = next((r["bet_amount"] for r in races if r["payout"] == top), 0)
    return {
        "n": n,
        "per_day": round(n / all_dates, 2) if all_dates else 0.0,
        "hit_rate": round(len(hits) / n, 4) if n else 0.0,
        "gami_rate": round(gami / len(hits), 4) if hits else 0.0,
        "roi": round(pay / bet, 4) if bet else 0.0,
        "roi_drop1": round((pay - top) / (bet - top_bet), 4) if bet - top_bet > 0 else 0.0,
    }


def roi_ci(races: list[dict], n_boot: int = 2000, seed: int = 20260809) -> tuple[float, float]:
    """残存レース群の ROI をレース単位ブートストラップして 95%CI を返す。"""
    n = len(races)
    if n < 2:
        return 0.0, 0.0
    pay = [r["payout"] for r in races]
    bet = [r["bet_amount"] for r in races]
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(bet[i] for i in idx)
        vals.append(sum(pay[i] for i in idx) / b if b else 0.0)
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def quantiles(values: list[float], qs: tuple[float, ...]) -> list[float]:
    """単純な分位点（線形補間なし・下側）。"""
    s = sorted(values)
    return [s[min(int(q * len(s)), len(s) - 1)] for q in qs]


def evaluate(
    rank: str,
    races: list[dict],
    window: str,
    rows: list[dict],
    fixed_thr: dict[str, list[tuple[float, str]]] | None = None,
) -> dict[str, list[tuple[float, str]]]:
    """1ランク・1窓について全ゲート・全閾値を評価し rows へ積む。

    fixed_thr を渡すと分位点を取り直さず、その絶対閾値で評価する（確認窓の正しい使い方）。
    返り値は本窓で得た絶対閾値（掃引窓から確認窓へ引き渡すため）。
    """
    dates = len({r["race_date"] for r in races})
    base = agg(races, dates)
    lo, hi = roi_ci(races)
    rows.append(
        {
            "rank": rank, "window": window, "gate": "none(現行全件)", "source": "-",
            "threshold": "-", "keep_share": 1.0, **base,
            "roi_lo": round(lo, 4), "roi_hi": round(hi, 4),
        }
    )

    qs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    derived: dict[str, list[tuple[float, str]]] = {}
    for src in ("wf", "db"):
        key = f"top2_{src}"
        pool = [r for r in races if r[key] is not None]
        if not pool:
            continue
        if fixed_thr is not None:
            thrs = fixed_thr.get(src, [])
        else:
            thrs = [
                (t, f"q{int(q * 100)}")
                for q, t in zip(qs, quantiles([r[key] for r in pool], qs), strict=True)
            ]
        derived[src] = thrs
        # ゲート行は当該ソースが存在するレースだけを母集団にするので、
        # 比較対象の baseline も同じ母集団に揃える（wf は 2024-07-01 以降しか無い）
        a = agg(pool, dates)
        lo, hi = roi_ci(pool)
        rows.append(
            {
                "rank": rank, "window": window, "gate": f"none({src}母集団)", "source": src,
                "threshold": "-", "keep_share": 1.0, **a,
                "roi_lo": round(lo, 4), "roi_hi": round(hi, 4),
            }
        )
        for thr, label in thrs:
            keep = [r for r in pool if r[key] <= thr]
            if not keep:
                continue
            a = agg(keep, dates)
            lo, hi = roi_ci(keep)
            rows.append(
                {
                    "rank": rank, "window": window, "gate": "top2_sum<=", "source": src,
                    "threshold": f"{thr:.1f} ({label})",
                    "keep_share": round(len(keep) / len(pool), 4), **a,
                    "roi_lo": round(lo, 4), "roi_hi": round(hi, 4),
                }
            )

    pool = [r for r in races if r["fav_odds"] is not None]
    for thr in ODDS_FLOORS:
        keep = [r for r in pool if r["fav_odds"] >= thr]
        if not keep:
            continue
        a = agg(keep, dates)
        lo, hi = roi_ci(keep)
        rows.append(
            {
                "rank": rank, "window": window, "gate": "morning_fav_odds>=", "source": "morning",
                "threshold": f"{thr:.1f}", "keep_share": round(len(keep) / len(pool), 4), **a,
                "roi_lo": round(lo, 4), "roi_hi": round(hi, 4),
            }
        )
    return derived


def main() -> None:
    """全ランク・全窓を評価して CSV へ書き出し、要点を表示する。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="step1c_skip_gate.csv")
    args = ap.parse_args()

    wf = load_wf_pp3()
    rows: list[dict] = []
    for rank in RANKS:
        races = load_races(rank, wf)
        sweep = [r for r in races if r["race_date"] <= SWEEP_END]
        confirm = [r for r in races if r["race_date"] > SWEEP_END]
        n_odds = sum(1 for r in races if r["fav_odds"] is not None)
        n_wf = sum(1 for r in races if r["top2_wf"] is not None)
        print(f"{rank}: {len(races)}R  wf有 {n_wf}R  朝オッズ有 {n_odds}R")
        evaluate(rank, races, "ALL", rows)
        thrs = evaluate(rank, sweep, "sweep(〜2025-12-31)", rows) if sweep else {}
        # 確認窓は掃引窓で決めた「絶対閾値」で評価する（分位点を取り直すと別の閾値になる）
        if confirm:
            evaluate(rank, confirm, "confirm(2026-01-01〜)", rows, fixed_thr=thrs)

    out_path = ROOT / "data" / "exports" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for rank in RANKS:
        print(f"\n===== {rank} =====")
        for wname in ("sweep(〜2025-12-31)", "confirm(2026-01-01〜)"):
            sel = [r for r in rows if r["rank"] == rank and r["window"] == wname]
            if not sel:
                continue
            print(f"-- {wname}")
            hdr = (
                f"  {'gate':<20}{'src':<6}{'thr':<15}{'残存':>7}{'n':>7}{'/日':>6}"
                f"{'的中':>7}{'ROI':>8}{'drop1':>8}  95%CI"
            )
            print(hdr)
            for r in sel:
                print(
                    f"  {r['gate']:<20}{r['source']:<6}{str(r['threshold']):<15}"
                    f"{r['keep_share'] * 100:>6.1f}%{r['n']:>7}{r['per_day']:>6.2f}"
                    f"{r['hit_rate'] * 100:>6.1f}%{r['roi'] * 100:>7.2f}%"
                    f"{r['roi_drop1'] * 100:>7.2f}%"
                    f"  [{r['roi_lo'] * 100:5.1f}, {r['roi_hi'] * 100:5.1f}]"
                )

    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
