"""レース構造特徴（グレード・競走得点分散・トップ差）× SS/S 成績の検証（2026-07-12）

購入対象（7車立て限定）の SS（三連複 軸2車-全）/ S（三連単1着固定F）について、
以下のレース単位特徴で的中率・ROI に傾向があるかを in-sample / OOS で検証する:

  1. レースランク: wt_races.grade × race_type（バケット化）
  2. 競走得点の分散: レース内 race_point の標準偏差（四分位バンド）
  3. トップと平均の差: max(race_point) - mean(race_point)（四分位バンド）
  4. 参考: 指数1位選手の race_point 順位（得点順位1位/2位/3位以下）

バンド境界は in-sample 購入セットの四分位で固定し、OOS にも同じ境界を適用する
（窓ごとに境界を引き直すと「方向一致」の判定ができないため）。

使い方:
  cd /Users/ysuzuki/GitHub/keirin
  .venv/bin/python scripts/exp_ss_structure_grade_wt.py            # キャッシュ利用
  .venv/bin/python scripts/exp_ss_structure_grade_wt.py --refresh  # collect し直し

決定論性: collect はモデル予測+DBのみで決定論的。ブートストラップは seed 固定。
"""
import argparse
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from exp_ss_trifecta_budget_wt import ss_races  # noqa: E402
from eval_clean_split_wt import ST_GAMI, ST_GAP12, collect  # noqa: E402
from src.database import get_connection  # noqa: E402
from src.models.trainer import load_model  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache"
BOOT_N = 10_000
BOOT_SEED = 20260712

# 検証ウィンドウ（モデルとの組は固定・変更禁止）
WINDOWS = [
    # (window_id, model_name, from, to)
    ("IN", "lgbm_wt_2026h1_eval", "2025-07-01", "2026-03-31"),
    ("OOS1", "lgbm_wt_2026h1_eval", "2026-04-01", "2026-06-30"),
    ("FWD", "lgbm_wt_2026h1", "2026-07-01", "2026-07-10"),
]


# ─── データ収集 ─────────────────────────────────────────────────────

def collect_cached(model_name: str, date_from: str, date_to: str, refresh: bool):
    """collect() の結果をディスクキャッシュ（1窓数分かかるため）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"exp_ss_structure_{model_name}_{date_from}_{date_to}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    rows = collect(load_model(model_name), date_from, date_to)
    with open(path, "wb") as f:
        pickle.dump(rows, f)
    return rows


def load_race_meta(race_keys):
    """race_key -> (grade, race_type, n_entries) と race_key -> {frame: point}。"""
    meta, points = {}, defaultdict(dict)
    keys = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            ph = ",".join("?" * len(chunk))
            for rk, g, rt, ne in c.execute(
                    f"SELECT race_key, grade, race_type, n_entries "
                    f"FROM wt_races WHERE race_key IN ({ph})", chunk):
                meta[rk] = (g or "?", rt or "?", ne)
            for rk, fr, pt in c.execute(
                    f"SELECT race_key, frame_no, race_point "
                    f"FROM wt_entries WHERE race_key IN ({ph})", chunk):
                if pt is not None:
                    points[rk][int(fr)] = float(pt)
    return meta, points


# ─── 特徴量構築 ─────────────────────────────────────────────────────

def type_bucket(race_type: str) -> str:
    """race_type を主要バケットへ正規化（準決勝は決勝より先に判定）。"""
    if "準決勝" in race_type:
        return "準決勝"
    if "決勝" in race_type:
        return "決勝"
    if "予選" in race_type:
        return "予選"
    if "特選" in race_type:
        return "特選"
    if "選抜" in race_type:
        return "選抜"
    if "一般" in race_type:
        return "一般"
    return "その他"


def grade_cat(grade: str, race_type: str) -> str:
    """A級チャレンジは A級 と分ける（実質別階級のため）。"""
    if "チャレンジ" in (race_type or ""):
        return "A級チャレンジ"
    return grade


def annotate(rows, meta, points):
    """各購入レースに構造特徴を付与。特徴が構築できないレースは除外。"""
    out = []
    for r in rows:
        rk = r["rk"]
        if rk not in meta:
            continue
        g, rt, _ = meta[rk]
        pts = points.get(rk, {})
        vals = [pts[f] for f in r["frames"] if f in pts]
        if len(vals) < 7 or pts.get(r["p1"]) is None:
            continue  # 得点欠損レースは母集団から外す（全窓とも同基準）
        mean_pt = statistics.fmean(vals)
        r = dict(r)
        r["grade_cat"] = grade_cat(g, rt)
        r["tbucket"] = type_bucket(rt)
        r["cat"] = f"{r['grade_cat']}×{r['tbucket']}"
        r["score_sd"] = statistics.pstdev(vals)
        r["top_gap"] = max(vals) - mean_pt
        p1_pt = pts[r["p1"]]
        r["p1_pt_rank"] = 1 + sum(1 for v in vals if v > p1_pt)
        out.append(r)
    return out


# ─── 購入セット・損益 ───────────────────────────────────────────────

def s_races(rows):
    """S購入セット（三連単1着固定F・gap12>=0.15・全目min>=10）。"""
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


def ss_pnl(r):
    """SS: (bet, pay) 100円/点均等。"""
    bet = len(r["ss_legs"]) * 100
    pay = 0
    for t, o in r["ss_legs"].items():
        if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
            pay = int(o * 100)
            break
    return bet, pay


def s_pnl(r):
    """S: (bet, pay) 100円/点均等。"""
    bet = len(r["s_combos"]) * 100
    pay = int(r["s_combos"][r["order"]] * 100) if r["order"] in r["s_combos"] else 0
    return bet, pay


# ─── 集計 ───────────────────────────────────────────────────────────

def agg(races, pnl_fn):
    n = h = b = p = 0
    for r in races:
        bet, pay = pnl_fn(r)
        n += 1
        h += 1 if pay > 0 else 0
        b += bet
        p += pay
    roi = p / b if b else 0.0
    hit = h / n if n else 0.0
    return n, hit, roi


def quartile_edges(values):
    """in-sample 購入セットで四分位境界（Q1/Q2/Q3）を返す。"""
    a = np.asarray(sorted(values))
    return [float(np.quantile(a, q)) for q in (0.25, 0.50, 0.75)]


def band_of(v, edges):
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


BAND_LABELS = ["Q1(小)", "Q2", "Q3", "Q4(大)"]


def print_band_table(title, var, races_by_win, pnl_fn, edges):
    print(f"\n### {title}  境界(in-sample四分位): "
          + " / ".join(f"{e:.3f}" for e in edges))
    header = f"{'バンド':<8}"
    for win in races_by_win:
        header += f" | {win:>4}: {'n':>4} {'hit':>6} {'ROI':>6}"
    print(header)
    for bi, lab in enumerate(BAND_LABELS):
        line = f"{lab:<8}"
        for win, races in races_by_win.items():
            sub = [r for r in races if band_of(r[var], edges) == bi]
            n, hit, roi = agg(sub, pnl_fn)
            line += f" | {'':>4}  {n:>4} {hit:>6.1%} {roi:>6.2f}"
        print(line)


def print_cat_table(title, key, races_by_win, pnl_fn, min_n=5):
    print(f"\n### {title}")
    cats = sorted({r[key] for races in races_by_win.values() for r in races})
    header = f"{'カテゴリ':<16}"
    for win in races_by_win:
        header += f" | {win:>4}: {'n':>4} {'hit':>6} {'ROI':>6}"
    print(header)
    for cat in cats:
        counts = [sum(1 for r in races if r[key] == cat)
                  for races in races_by_win.values()]
        if max(counts) < min_n:
            continue
        line = f"{cat:<16}"
        for win, races in races_by_win.items():
            sub = [r for r in races if r[key] == cat]
            n, hit, roi = agg(sub, pnl_fn)
            line += f" | {'':>4}  {n:>4} {hit:>6.1%} {roi:>6.2f}"
        print(line)


def bootstrap_roi_diff(races_a, races_b, pnl_fn, seed=BOOT_SEED):
    """レース単位リサンプルで ROI(a) - ROI(b) の95%CI。"""
    rng = np.random.default_rng(seed)
    pa = np.array([pnl_fn(r) for r in races_a], dtype=float)  # (n,2) bet,pay
    pb = np.array([pnl_fn(r) for r in races_b], dtype=float)
    diffs = np.empty(BOOT_N)
    for i in range(BOOT_N):
        ia = rng.integers(0, len(pa), len(pa))
        ib = rng.integers(0, len(pb), len(pb))
        sa, sb = pa[ia].sum(axis=0), pb[ib].sum(axis=0)
        diffs[i] = sa[1] / sa[0] - sb[1] / sb[0]
    return (float(np.quantile(diffs, 0.025)),
            float(np.quantile(diffs, 0.5)),
            float(np.quantile(diffs, 0.975)))


def bootstrap_roi_ci(races, pnl_fn, seed=BOOT_SEED):
    """レース単位リサンプルで ROI の95%CI。"""
    rng = np.random.default_rng(seed)
    p = np.array([pnl_fn(r) for r in races], dtype=float)
    rois = np.empty(BOOT_N)
    for i in range(BOOT_N):
        s = p[rng.integers(0, len(p), len(p))].sum(axis=0)
        rois[i] = s[1] / s[0]
    return (float(np.quantile(rois, 0.025)),
            float(np.quantile(rois, 0.5)),
            float(np.quantile(rois, 0.975)))


# ─── メイン ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="collect し直す")
    args = ap.parse_args()

    # 1) 候補レース収集（窓ごとにキャッシュ）
    raw = {}
    for wid, model_name, f, t in WINDOWS:
        raw[wid] = collect_cached(model_name, f, t, args.refresh)
        print(f"[collect] {wid} {f}〜{t}: {len(raw[wid])}R (候補・n_entries>=7)")

    # 2) 7車立て限定 + 構造特徴付与
    all_keys = {r["rk"] for rows in raw.values() for r in rows}
    meta, points = load_race_meta(all_keys)
    rows7 = {}
    for wid, rows in raw.items():
        f7 = [r for r in rows if meta.get(r["rk"], (None, None, 0))[2] == 7]
        rows7[wid] = annotate(f7, meta, points)
        print(f"[7car ] {wid}: {len(f7)}R -> 特徴付与後 {len(rows7[wid])}R")

    # 3) 購入セット構築（OOS1+FWD を合算して OOS とする）
    oos_rows = rows7["OOS1"] + rows7["FWD"]
    sets = {
        "SS": {"IN": ss_races(rows7["IN"]), "OOS": ss_races(oos_rows)},
        "S": {"IN": s_races(rows7["IN"]), "OOS": s_races(oos_rows)},
    }
    pnl_fns = {"SS": ss_pnl, "S": s_pnl}

    for sysname, by_win in sets.items():
        pnl = pnl_fns[sysname]
        print(f"\n{'=' * 72}\n== {sysname} 購入セット ==")
        for win, races in by_win.items():
            n, hit, roi = agg(races, pnl)
            print(f"  {win}: n={n}  hit={hit:.1%}  ROI={roi:.3f}")

        # レースランク（グレード / 種別 / 組合せ）
        print_cat_table(f"[{sysname}] グレード別", "grade_cat", by_win, pnl)
        print_cat_table(f"[{sysname}] レース種別別", "tbucket", by_win, pnl)
        print_cat_table(f"[{sysname}] グレード×種別", "cat", by_win, pnl, min_n=10)

        # 分散・トップ差（in-sample 購入セットで境界固定）
        for var, title in (("score_sd", "競走得点SD"),
                           ("top_gap", "トップ-平均差")):
            edges = quartile_edges([r[var] for r in by_win["IN"]])
            print_band_table(f"[{sysname}] {title}", var, by_win, pnl, edges)

        # 参考: 指数1位の得点順位
        print(f"\n### [{sysname}] 指数1位選手の競走得点順位")
        header = f"{'得点順位':<8}"
        for win in by_win:
            header += f" | {win:>4}: {'n':>4} {'hit':>6} {'ROI':>6}"
        print(header)
        for lab, cond in (("1位", lambda r: r["p1_pt_rank"] == 1),
                          ("2位", lambda r: r["p1_pt_rank"] == 2),
                          ("3位以下", lambda r: r["p1_pt_rank"] >= 3)):
            line = f"{lab:<8}"
            for win, races in by_win.items():
                sub = [r for r in races if cond(r)]
                n, hit, roi = agg(sub, pnl)
                line += f" | {'':>4}  {n:>4} {hit:>6.1%} {roi:>6.2f}"
            print(line)

    # 4) ブートストラップ（方向一致が見られた候補に対して実行）
    #    候補の定義はテーブル出力を見て決めるが、機械的に
    #    「INとOOSの両方で バンドROI - 全体ROI が同符号かつ|差|>=0.10」
    #    のバンドを自動抽出して CI を付ける。
    print(f"\n{'=' * 72}\n== ブートストラップ（自動抽出候補: IN/OOS 同方向 |ΔROI|>=0.10, n>=30） ==")
    for sysname, by_win in sets.items():
        pnl = pnl_fns[sysname]
        candidates = []
        # 数値バンド
        for var in ("score_sd", "top_gap"):
            edges = quartile_edges([r[var] for r in by_win["IN"]])
            for bi, lab in enumerate(BAND_LABELS):
                sel = {w: [r for r in races if band_of(r[var], edges) == bi]
                       for w, races in by_win.items()}
                candidates.append((f"{var} {lab}", sel))
        # カテゴリ
        for key in ("grade_cat", "tbucket"):
            for cat in sorted({r[key] for rs in by_win.values() for r in rs}):
                sel = {w: [r for r in races if r[key] == cat]
                       for w, races in by_win.items()}
                candidates.append((f"{key}={cat}", sel))
        # p1得点順位
        for lab, cond in (("p1_pt_rank=1", lambda r: r["p1_pt_rank"] == 1),
                          ("p1_pt_rank>=2", lambda r: r["p1_pt_rank"] >= 2)):
            sel = {w: [r for r in races if cond(r)] for w, races in by_win.items()}
            candidates.append((lab, sel))

        found = False
        for name, sel in candidates:
            if min(len(sel["IN"]), len(sel["OOS"])) < 30:
                continue
            deltas = {}
            for w in ("IN", "OOS"):
                _, _, roi_all = agg(by_win[w], pnl)
                _, _, roi_sub = agg(sel[w], pnl)
                deltas[w] = roi_sub - roi_all
            if deltas["IN"] * deltas["OOS"] <= 0:
                continue
            if min(abs(deltas["IN"]), abs(deltas["OOS"])) < 0.10:
                continue
            found = True
            # OOS でのバンド内 vs バンド外 ROI差 CI + バンド内 ROI CI
            inb = sel["OOS"]
            in_keys = {r["rk"] for r in inb}
            outb = [r for r in by_win["OOS"] if r["rk"] not in in_keys]
            lo, md, hi = bootstrap_roi_diff(inb, outb, pnl)
            rlo, rmd, rhi = bootstrap_roi_ci(inb, pnl)
            print(f"[{sysname}] {name:<24} ΔROI(IN)={deltas['IN']:+.2f} "
                  f"ΔROI(OOS)={deltas['OOS']:+.2f} "
                  f"| OOS n={len(inb)} ROI95%CI=[{rlo:.2f},{rhi:.2f}] "
                  f"(内-外)diff95%CI=[{lo:.2f},{hi:.2f}]")
        if not found:
            print(f"[{sysname}] 自動抽出基準を満たす候補なし")


if __name__ == "__main__":
    main()
