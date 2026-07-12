"""ライン得点格差 × SS/S 購入成績の検証（2026-07-12）

仮説: ライン単位の競走得点合計に大差があるレース（1本のラインが突出）や、
軸選手の所属ラインが得点面で優位なレースは、軸の信頼度や配当構造に傾向が
あるのではないか。

検証対象（7車立て限定）:
  - SS購入セット: 三連複 軸2車-全（min全目>=7, gap12>=0.10, gap23>=1pt）均等100円/点
  - S 購入セット: 三連単 1着固定F p1->(p2,r3)->全（gap12>=0.15, min全目>=10）100円/点

説明変数（レース単位・ライン構造）:
  1. ライン得点格差: line_group ごとの race_point 合計/平均の (1位 - 2位)
  2. 軸ライン優位性: p1所属ラインが合計/平均1位か、p1とp2が同一ラインか
  3. ライン構成: ライン数 / 最大line_size / 単騎数
  4. 交互作用: 軸2車同一ライン × ライン平均得点格差

窓とモデル:
  - in-sample参考: lgbm_wt_2026h1_eval  2025-07-01〜2026-03-31
  - クリーンOOS  : lgbm_wt_2026h1_eval  2026-04-01〜2026-06-30
  - フォワード   : lgbm_wt_2026h1       2026-07-01〜2026-07-10
  （OOS 2窓は合算して評価。in-sample と OOS で方向一致するもののみ採用）

使い方:
  .venv/bin/python scripts/exp_ss_line_score_wt.py
  （collect 結果は data/exp_line_cache/ にキャッシュされ再実行は高速）
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from exp_ss_trifecta_budget_wt import ss_races  # noqa: E402
from eval_clean_split_wt import (  # noqa: E402
    ST_GAMI, ST_GAP12, collect,
)
from src.database import get_connection  # noqa: E402
from src.models.trainer import load_model  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_line_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    # (label, model_name, from, to)
    ("IS", "lgbm_wt_2026h1_eval", "2025-07-01", "2026-03-31"),
    ("OOS1", "lgbm_wt_2026h1_eval", "2026-04-01", "2026-06-30"),
    ("FWD", "lgbm_wt_2026h1", "2026-07-01", "2026-07-10"),
]

N_BOOT = 10_000
SEED = 42


# ─── データ取得 ──────────────────────────────────────────────────────

def collect_cached(model_name: str, date_from: str, date_to: str) -> list[dict]:
    """collect() の結果を pickle キャッシュ（決定論的・再実行高速化）。"""
    cache = CACHE_DIR / f"collect_{model_name}_{date_from}_{date_to}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    rows = collect(load_model(model_name), date_from, date_to)
    with open(cache, "wb") as f:
        pickle.dump(rows, f)
    return rows


def load_n_entries() -> dict[str, int]:
    with get_connection() as c:
        return dict(c.execute("SELECT race_key, n_entries FROM wt_races"))


def load_line_features() -> tuple[dict[str, dict], int]:
    """wt_entries からレース単位のライン特徴を構築。

    Returns:
        (race_key -> 特徴dict, line情報欠損で除外したレース数)
    """
    per_race: dict[str, list[tuple]] = defaultdict(list)
    with get_connection() as c:
        q = ("SELECT race_key, frame_no, race_point, line_group, line_size, "
             "line_pos, is_line_leader, n_lines FROM wt_entries")
        for row in c.execute(q):
            per_race[row[0]].append(row[1:])

    feats: dict[str, dict] = {}
    n_excluded = 0
    for rk, ents in per_race.items():
        if any(e[2] is None or e[1] is None for e in ents):
            n_excluded += 1  # line_group / race_point 欠損
            continue
        groups: dict[int, list[tuple]] = defaultdict(list)
        for e in ents:
            groups[int(e[2])].append(e)
        if len(groups) < 2:
            n_excluded += 1  # ライン1本のみは格差定義不能
            continue
        sums = {g: sum(float(e[1]) for e in m) for g, m in groups.items()}
        avgs = {g: sums[g] / len(m) for g, m in groups.items()}
        s_sorted = sorted(sums.values(), reverse=True)
        a_sorted = sorted(avgs.values(), reverse=True)
        frame_line = {int(e[0]): int(e[2]) for e in ents}
        sizes = {g: len(m) for g, m in groups.items()}
        feats[rk] = {
            "sum_gap": s_sorted[0] - s_sorted[1],
            "avg_gap": a_sorted[0] - a_sorted[1],
            "line_sums": sums,
            "line_avgs": avgs,
            "frame_line": frame_line,
            "n_lines": len(groups),
            "max_line_size": max(sizes.values()),
            "n_solo": sum(1 for v in sizes.values() if v == 1),
        }
    return feats, n_excluded


def attach_features(races: list[dict], feats: dict[str, dict]) -> tuple[list[dict], int]:
    """レースにライン特徴を付与。特徴の無いレースは除外して件数を返す。"""
    out, n_missing = [], 0
    for r in races:
        f = feats.get(r["rk"])
        if f is None:
            n_missing += 1
            continue
        r = dict(r)
        p1_line = f["frame_line"].get(r["p1"])
        p2_line = f["frame_line"].get(r["p2"])
        if p1_line is None or p2_line is None:
            n_missing += 1
            continue
        max_sum = max(f["line_sums"].values())
        max_avg = max(f["line_avgs"].values())
        r.update(
            sum_gap=f["sum_gap"], avg_gap=f["avg_gap"],
            n_lines=f["n_lines"], max_line_size=f["max_line_size"],
            n_solo=f["n_solo"],
            p1_line_sum_top=f["line_sums"][p1_line] >= max_sum - 1e-9,
            p1_line_avg_top=f["line_avgs"][p1_line] >= max_avg - 1e-9,
            p1p2_same_line=p1_line == p2_line,
        )
        out.append(r)
    return out, n_missing


# ─── 購入セットと損益 ────────────────────────────────────────────────

def ss_bet_pay(r: dict) -> tuple[int, int]:
    """SS: 三連複 軸2車-全 均等100円/点。(bet, pay) を返す。"""
    bet = len(r["ss_legs"]) * 100
    pay = 0
    for t, o in r["ss_legs"].items():
        if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
            pay = int(o * 100)
            break
    return bet, pay


def s_set(rows: list[dict]) -> list[dict]:
    """S購入セット: gap12>=0.15 かつ 1着固定F 全目 min>=10。"""
    out = []
    for r in rows:
        if r["gap12"] < ST_GAP12:
            continue
        combos = {}
        for s in (r["p2"], r["r3"]):
            for t in r["frames"]:
                if t in (r["p1"], s):
                    continue
                ov = r["tri"].get((r["p1"], s, t))
                if ov:
                    combos[(r["p1"], s, t)] = ov
        if not combos or min(combos.values()) < ST_GAMI:
            continue
        r = dict(r)
        r["s_combos"] = combos
        out.append(r)
    return out


def s_bet_pay(r: dict) -> tuple[int, int]:
    """S: 三連単1着固定F 均等100円/点。(bet, pay) を返す。"""
    bet = len(r["s_combos"]) * 100
    pay = int(r["s_combos"][r["order"]] * 100) if r["order"] in r["s_combos"] else 0
    return bet, pay


def summarize(races: list[dict], bp_fn) -> tuple[int, float, float]:
    """(n, 的中率, ROI)"""
    n = len(races)
    if n == 0:
        return 0, float("nan"), float("nan")
    bets_pays = [bp_fn(r) for r in races]
    bet = sum(b for b, _ in bets_pays)
    pay = sum(p for _, p in bets_pays)
    hits = sum(1 for _, p in bets_pays if p > 0)
    return n, hits / n, pay / bet if bet else float("nan")


# ─── 集計・表出力 ────────────────────────────────────────────────────

def quartile_edges(values: list[float]) -> list[float]:
    return [float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75)]


def band_label(v: float, edges: list[float], name: str) -> str:
    if v < edges[0]:
        return f"Q1(<{edges[0]:.1f})"
    if v < edges[1]:
        return f"Q2({edges[0]:.1f}-{edges[1]:.1f})"
    if v < edges[2]:
        return f"Q3({edges[1]:.1f}-{edges[2]:.1f})"
    return f"Q4(>={edges[2]:.1f})"


def print_table(title: str, windows: dict[str, list[dict]], key_fn, bp_fn,
                order: list[str] | None = None) -> dict:
    """窓別×カテゴリ別に n/的中率/ROI を表出力。結果 dict も返す。"""
    result: dict[str, dict[str, tuple]] = {}
    cats: list[str] = []
    for wlabel, races in windows.items():
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in races:
            by_cat[str(key_fn(r))].append(r)
        result[wlabel] = {c: summarize(rs, bp_fn) for c, rs in by_cat.items()}
        for c in by_cat:
            if c not in cats:
                cats.append(c)
    cats = order if order is not None else sorted(cats)
    print(f"\n--- {title} ---")
    header = f"{'カテゴリ':<24}"
    for w in windows:
        header += f" | {w+' n':>6} {'的中率':>6} {'ROI':>6}"
    print(header)
    for c in cats:
        line = f"{c:<24}"
        for w in windows:
            n, hr, roi = result[w].get(c, (0, float('nan'), float('nan')))
            if n == 0:
                line += f" | {0:>6} {'-':>7} {'-':>6}"
            else:
                line += f" | {n:>6} {hr:>7.1%} {roi:>6.2f}"
        print(line)
    return result


def bootstrap_roi_diff(flag_races: list[dict], rest_races: list[dict],
                       bp_fn, rng: np.random.Generator) -> tuple[float, float, float]:
    """レース単位2標本ブートストラップで ROI差(flag - rest) の点推定と95%CI。"""
    fb = np.array([bp_fn(r) for r in flag_races], dtype=float)
    rb = np.array([bp_fn(r) for r in rest_races], dtype=float)
    point = fb[:, 1].sum() / fb[:, 0].sum() - rb[:, 1].sum() / rb[:, 0].sum()
    diffs = np.empty(N_BOOT)
    nf, nr = len(fb), len(rb)
    for i in range(N_BOOT):
        f = fb[rng.integers(0, nf, nf)]
        r = rb[rng.integers(0, nr, nr)]
        diffs[i] = f[:, 1].sum() / f[:, 0].sum() - r[:, 1].sum() / r[:, 0].sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


# ─── メイン ──────────────────────────────────────────────────────────

def main() -> None:
    feats, n_line_excl = load_line_features()
    ne_map = load_n_entries()
    print(f"ライン情報欠損/1本のみで特徴構築から除外: {n_line_excl}レース")

    # collect（キャッシュ利用）→ 7車立て限定
    raw: dict[str, list[dict]] = {}
    for label, model, f, t in WINDOWS:
        rows = collect_cached(model, f, t)
        rows7 = [r for r in rows if ne_map.get(r["rk"]) == 7]
        print(f"{label} {f}〜{t}: 候補{len(rows)}R → 7車立て{len(rows7)}R")
        raw[label] = rows7

    # 購入セット構築（IS / OOS=OOS1+FWD 合算）
    sets: dict[str, dict[str, list[dict]]] = {}
    for bet_label, builder in (("SS", ss_races), ("S", s_set)):
        is_races, miss_is = attach_features(builder(raw["IS"]), feats)
        oos_races, miss_oos = attach_features(
            builder(raw["OOS1"]) + builder(raw["FWD"]), feats)
        print(f"\n{bet_label}購入セット: IS {len(is_races)}R (特徴欠損除外{miss_is}) / "
              f"OOS {len(oos_races)}R (特徴欠損除外{miss_oos})")
        sets[bet_label] = {"IS": is_races, "OOS": oos_races}

    rng = np.random.default_rng(SEED)

    for bet_label, bp_fn in (("SS", ss_bet_pay), ("S", s_bet_pay)):
        windows = sets[bet_label]
        is_races, oos_races = windows["IS"], windows["OOS"]
        base_is = summarize(is_races, bp_fn)
        base_oos = summarize(oos_races, bp_fn)
        print(f"\n{'='*90}")
        print(f"■ {bet_label}購入セット 全体: "
              f"IS n={base_is[0]} 的中{base_is[1]:.1%} ROI{base_is[2]:.2f} / "
              f"OOS n={base_oos[0]} 的中{base_oos[1]:.1%} ROI{base_oos[2]:.2f}")

        # 1. ライン得点格差（四分位・エッジはISで固定）
        for metric, name in (("sum_gap", "ライン得点[合計]格差(1位-2位)"),
                             ("avg_gap", "ライン得点[平均]格差(1位-2位)")):
            edges = quartile_edges([r[metric] for r in is_races])
            labels = [band_label(v, edges, metric)
                      for v in (edges[0] - 1, edges[0], edges[1], edges[2])]
            print_table(f"{bet_label}: {name}  [四分位エッジ(IS基準): "
                        f"{edges[0]:.1f}/{edges[1]:.1f}/{edges[2]:.1f}]",
                        windows, lambda r, e=edges, m=metric: band_label(r[m], e, m),
                        bp_fn, order=labels)

        # 2. 軸ラインの優位性
        print_table(f"{bet_label}: p1所属ラインが得点[合計]1位か", windows,
                    lambda r: "合計1位" if r["p1_line_sum_top"] else "合計2位以下",
                    bp_fn, order=["合計1位", "合計2位以下"])
        print_table(f"{bet_label}: p1所属ラインが得点[平均]1位か", windows,
                    lambda r: "平均1位" if r["p1_line_avg_top"] else "平均2位以下",
                    bp_fn, order=["平均1位", "平均2位以下"])
        print_table(f"{bet_label}: p1とp2が同一ラインか", windows,
                    lambda r: "同一ライン" if r["p1p2_same_line"] else "別ライン",
                    bp_fn, order=["同一ライン", "別ライン"])

        # 3. ライン構成
        print_table(f"{bet_label}: ライン数", windows,
                    lambda r: f"{r['n_lines']}本", bp_fn)
        print_table(f"{bet_label}: 最大line_size", windows,
                    lambda r: f"最大{r['max_line_size']}車", bp_fn)
        print_table(f"{bet_label}: 単騎数", windows,
                    lambda r: f"単騎{r['n_solo']}", bp_fn)

        # 4. 交互作用: 軸2車同一ライン × 平均格差大（IS中央値以上）
        med = float(np.median([r["avg_gap"] for r in is_races]))
        print_table(
            f"{bet_label}: 軸2車同一ライン × 平均格差(IS中央値{med:.1f}で二分)",
            windows,
            lambda r: ("同一×格差大" if r["avg_gap"] >= med else "同一×格差小")
            if r["p1p2_same_line"]
            else ("別×格差大" if r["avg_gap"] >= med else "別×格差小"),
            bp_fn,
            order=["同一×格差大", "同一×格差小", "別×格差大", "別×格差小"])

        # 5. 主要二値カットのブートストラップ（OOS・レース単位1万回）
        print(f"\n--- {bet_label}: OOSブートストラップ ROI差(該当-非該当) 95%CI ---")
        q3 = float(np.quantile([r["avg_gap"] for r in is_races], 0.75))
        cuts = [
            ("p1ライン合計1位", lambda r: r["p1_line_sum_top"]),
            ("p1ライン平均1位", lambda r: r["p1_line_avg_top"]),
            ("軸2車同一ライン", lambda r: r["p1p2_same_line"]),
            ("平均格差大(IS中央値以上)", lambda r: r["avg_gap"] >= med),
            ("平均格差Q4(IS75%点以上)", lambda r: r["avg_gap"] >= q3),
            ("同一ライン×格差大", lambda r: r["p1p2_same_line"] and r["avg_gap"] >= med),
            ("単騎0", lambda r: r["n_solo"] == 0),
            ("ライン数3本以下", lambda r: r["n_lines"] <= 3),
            ("ライン数4本以上(全単騎除く)",
             lambda r: r["n_lines"] >= 4 and r["n_solo"] < 7),
        ]
        for name, pred in cuts:
            flag = [r for r in oos_races if pred(r)]
            rest = [r for r in oos_races if not pred(r)]
            if len(flag) < 30 or len(rest) < 30:
                print(f"{name:<28} n={len(flag)}/{len(rest)} → サンプル不足のためスキップ")
                continue
            point, lo, hi = bootstrap_roi_diff(flag, rest, bp_fn, rng)
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"{name:<28} n={len(flag):>3}/{len(rest):>3} "
                  f"ROI差={point:+.3f} [95%CI {lo:+.3f}, {hi:+.3f}]{sig}")


if __name__ == "__main__":
    main()
