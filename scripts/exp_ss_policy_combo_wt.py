"""SS購入 統合ポリシー（選抜カット×ライン数×ライン格差増額）の同時適用検証（2026-07-12）

前段検証（exp_ss_structure_grade_wt / exp_ss_line_score_wt）で個別に有望と
判定された3条件を同時適用した場合の重複・干渉を検証する:

  1. 種別「選抜」（race_typeに「選抜」を含む）カット
  2. ライン平均得点格差 avg_gap >= 1.5 で増額（1.5x/2x/3xスイープ）
  3. ライン数4本以上（全単騎レース除く）で見送り or 減額(0.5x)

ポリシー定義（ベース100円/点・SS=三連複 軸2車-全 ss_legs均等）:
  P0: 現行（全SS均等）
  P1: 選抜カットのみ
  P2: P1 + ライン数>=4(全単騎除く) 見送り
  P3a/b/c: P2 + avg_gap>=1.5 で 1.5x/2x/3x 増額
  P4: P1 + ライン数>=4(全単騎除く) 減額0.5x + avg_gap>=1.5 で 2x 増額
      （両方該当時は乗算 0.5*2=1.0x）

注意（前段からの引き継ぎ）:
  - 母集団は公式（欠車レース込み）。ss_legs にある目だけ買う扱い
    （実運用と実払戻の突合で一致確認済み）。
  - exp_ss_structure_grade_wt.annotate は完走者ベース len(vals)<7 判定で
    欠車レースを除外するバグ的挙動があるため流用しない。
    種別は wt_races から、得点・ラインは wt_entries（出走全車）から取る。
  - ライン特徴が構築できないレース（line_group/race_point欠損）は
    「補正なし（ベース100円）」として扱い件数を報告する。

窓:
  IS  = lgbm_wt_2026h1_eval 2025-07-01〜2026-03-31
  OOS = lgbm_wt_2026h1_eval 2026-04-01〜2026-06-30
        + lgbm_wt_2026h1    2026-07-01〜2026-07-10（合算）

決定論性: collect はモデル+DBのみで決定論的（data/exp_line_cache/ に
キャッシュ）。ブートストラップは seed 固定・レース単位1万回。
ポリシー間比較は同一リサンプルを共有するペアドブートストラップ。

使い方:
  cd /Users/ysuzuki/GitHub/keirin
  .venv/bin/python scripts/exp_ss_policy_combo_wt.py
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
from eval_clean_split_wt import collect  # noqa: E402
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

BASE_STAKE = 100          # 円/点
AVG_GAP_TH = 1.5          # 増額発動しきい値
N_BOOT = 10_000
SEED = 20260712


# ─── データ取得 ──────────────────────────────────────────────────────

def collect_cached(model_name: str, date_from: str, date_to: str) -> list[dict]:
    """collect() の結果を pickle キャッシュ（exp_ss_line_score_wt と共有）。"""
    cache = CACHE_DIR / f"collect_{model_name}_{date_from}_{date_to}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    rows = collect(load_model(model_name), date_from, date_to)
    with open(cache, "wb") as f:
        pickle.dump(rows, f)
    return rows


def load_race_meta() -> dict[str, tuple]:
    """race_key -> (grade, race_type, n_entries)。"""
    with get_connection() as c:
        return {rk: (g or "?", rt or "?", ne) for rk, g, rt, ne in c.execute(
            "SELECT race_key, grade, race_type, n_entries FROM wt_races")}


def load_line_features() -> dict[str, dict]:
    """wt_entries（出走全車）からレース単位のライン特徴を構築。

    avg_gap: line_group ごとの race_point 平均の (1位 - 2位)。
             ライン1本のみのレースは None（格差定義不能）。
    構築不能（line_group / race_point 欠損）のレースは辞書に含めない。
    """
    per_race: dict[str, list[tuple]] = defaultdict(list)
    with get_connection() as c:
        q = "SELECT race_key, frame_no, race_point, line_group FROM wt_entries"
        for rk, fn, pt, lg in c.execute(q):
            per_race[rk].append((fn, pt, lg))

    feats: dict[str, dict] = {}
    for rk, ents in per_race.items():
        if any(e[1] is None or e[2] is None for e in ents):
            continue  # 特徴構築不能
        groups: dict[int, list[float]] = defaultdict(list)
        for _, pt, lg in ents:
            groups[int(lg)].append(float(pt))
        avgs = sorted((sum(v) / len(v) for v in groups.values()), reverse=True)
        feats[rk] = {
            "avg_gap": avgs[0] - avgs[1] if len(avgs) >= 2 else None,
            "n_lines": len(groups),
            "n_solo": sum(1 for v in groups.values() if len(v) == 1),
        }
    return feats


def annotate(races: list[dict], meta: dict, line_feats: dict) -> list[dict]:
    """SSレースに 選抜フラグ・ライン特徴を付与（レースは除外しない）。"""
    out = []
    for r in races:
        r = dict(r)
        g, rt, _ = meta.get(r["rk"], ("?", "?", None))
        r["grade"] = g
        r["race_type"] = rt
        r["is_senbatsu"] = "選抜" in rt
        f = line_feats.get(r["rk"])
        r["avg_gap"] = f["avg_gap"] if f else None
        r["n_lines"] = f["n_lines"] if f else None
        r["n_solo"] = f["n_solo"] if f else None
        r["line_missing"] = f is None
        out.append(r)
    return out


# ─── 条件フラグ・ポリシー ────────────────────────────────────────────

def flag_senbatsu(r: dict) -> bool:
    return r["is_senbatsu"]


def flag_line4(r: dict) -> bool:
    """ライン数4本以上（全単騎レース除く）。特徴欠損は非該当扱い。"""
    return (r["n_lines"] is not None
            and r["n_lines"] >= 4 and r["n_solo"] < 7)


def flag_gap_big(r: dict) -> bool:
    """ライン平均得点格差 avg_gap >= 1.5。欠損/定義不能は非該当扱い。"""
    return r["avg_gap"] is not None and r["avg_gap"] >= AVG_GAP_TH


def make_policies() -> list[tuple[str, callable]]:
    """(名称, race -> 賭け金倍率(0=見送り)) のリスト。"""

    def p0(r):
        return 1.0

    def p1(r):
        return 0.0 if flag_senbatsu(r) else 1.0

    def p2(r):
        if flag_senbatsu(r) or flag_line4(r):
            return 0.0
        return 1.0

    def p3(mult):
        def f(r):
            base = p2(r)
            if base == 0.0:
                return 0.0
            return mult if flag_gap_big(r) else 1.0
        return f

    def p4(r):
        if flag_senbatsu(r):
            return 0.0
        m = 0.5 if flag_line4(r) else 1.0
        if flag_gap_big(r):
            m *= 2.0
        return m

    return [
        ("P0 現行(全SS均等)", p0),
        ("P1 選抜カット", p1),
        ("P2 P1+4分戦見送り", p2),
        ("P3a P2+格差1.5x増額", p3(1.5)),
        ("P3b P2+格差2x増額", p3(2.0)),
        ("P3c P2+格差3x増額", p3(3.0)),
        ("P4 P1+4分戦0.5x+格差2x", p4),
    ]


# ─── 損益計算 ────────────────────────────────────────────────────────

def bet_pay(r: dict, mult: float) -> tuple[float, float]:
    """SS: 三連複 軸2車-全 均等 (BASE_STAKE*mult)円/点。(bet, pay)。"""
    if mult <= 0.0:
        return 0.0, 0.0
    stake = BASE_STAKE * mult
    bet = len(r["ss_legs"]) * stake
    pay = 0.0
    for t, o in r["ss_legs"].items():
        if frozenset({r["p1"], r["p2"], t}) == r["top3"]:
            pay = o * stake
            break
    return bet, pay


def policy_stats(races: list[dict], mult_fn) -> dict:
    """ポリシーの購入R数・的中率・投資・払戻・ROI・利益・月別ROI・最大DD。"""
    races = sorted(races, key=lambda r: r["rk"])  # 時系列
    n = hits = 0
    bet = pay = 0.0
    monthly: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    cum = peak = max_dd = 0.0
    for r in races:
        b, p = bet_pay(r, mult_fn(r))
        if b <= 0:
            continue
        n += 1
        hits += 1 if p > 0 else 0
        bet += b
        pay += p
        m = f"{r['rk'][:4]}-{r['rk'][4:6]}"
        monthly[m][0] += b
        monthly[m][1] += p
        cum += p - b
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return {
        "n": n, "hit": hits / n if n else float("nan"),
        "bet": bet, "pay": pay,
        "roi": pay / bet if bet else float("nan"),
        "profit": pay - bet,
        "monthly": dict(monthly), "max_dd": max_dd,
    }


# ─── ブートストラップ ────────────────────────────────────────────────

def two_sample_roi_diff(flag: list[dict], rest: list[dict],
                        rng: np.random.Generator) -> tuple[float, float, float]:
    """レース単位2標本ブートストラップで ROI差(flag - rest) 点推定+95%CI。"""
    fa = np.array([bet_pay(r, 1.0) for r in flag], dtype=float)
    ra = np.array([bet_pay(r, 1.0) for r in rest], dtype=float)
    point = fa[:, 1].sum() / fa[:, 0].sum() - ra[:, 1].sum() / ra[:, 0].sum()
    diffs = np.empty(N_BOOT)
    nf, nr = len(fa), len(ra)
    for i in range(N_BOOT):
        f = fa[rng.integers(0, nf, nf)]
        r = ra[rng.integers(0, nr, nr)]
        diffs[i] = f[:, 1].sum() / f[:, 0].sum() - r[:, 1].sum() / r[:, 0].sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_policy_bootstrap(races: list[dict], policies: list[tuple],
                            rng: np.random.Generator) -> dict[str, tuple]:
    """同一リサンプルを全ポリシーで共有し ROI(policy) - ROI(P0) の95%CI。

    Returns:
        名称 -> (点推定, lo, hi)
    """
    # 各ポリシーの (bet, pay) をレースごとに前計算
    mats = {name: np.array([bet_pay(r, fn(r)) for r in races], dtype=float)
            for name, fn in policies}
    base_name = policies[0][0]
    n = len(races)
    diffs = {name: np.empty(N_BOOT) for name, _ in policies[1:]}
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        b0 = mats[base_name][idx].sum(axis=0)
        roi0 = b0[1] / b0[0]
        for name, _ in policies[1:]:
            s = mats[name][idx].sum(axis=0)
            diffs[name][i] = (s[1] / s[0] if s[0] > 0 else np.nan) - roi0
    out = {}
    for name, _ in policies[1:]:
        full = mats[name].sum(axis=0)
        full0 = mats[base_name].sum(axis=0)
        point = (full[1] / full[0] if full[0] > 0 else float("nan")) \
            - full0[1] / full0[0]
        d = diffs[name][~np.isnan(diffs[name])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = (point, float(lo), float(hi))
    return out


# ─── 表出力 ──────────────────────────────────────────────────────────

def print_split_table(title: str, windows: dict[str, list[dict]], pred,
                      labels: tuple[str, str]) -> None:
    """条件該当/非該当の n/的中率/ROI を窓別に出力。"""
    print(f"\n--- {title} ---")
    header = f"{'区分':<16}"
    for w in windows:
        header += f" | {w:>3}: {'n':>4} {'hit':>6} {'ROI':>6}"
    print(header)
    for lab, want in ((labels[0], True), (labels[1], False)):
        line = f"{lab:<16}"
        for w, races in windows.items():
            sub = [r for r in races if pred(r) == want]
            st = policy_stats(sub, lambda r: 1.0)
            if st["n"] == 0:
                line += f" | {'':>3}  {0:>4} {'-':>6} {'-':>6}"
            else:
                line += (f" | {'':>3}  {st['n']:>4} {st['hit']:>6.1%} "
                         f"{st['roi']:>6.2f}")
        print(line)


def fmt_monthly(monthly: dict[str, list[float]]) -> str:
    return " ".join(f"{m[2:]}:{v[1] / v[0]:.2f}"
                    for m, v in sorted(monthly.items()))


# ─── メイン ──────────────────────────────────────────────────────────

def main() -> None:
    meta = load_race_meta()
    line_feats = load_line_features()

    # collect（キャッシュ）→ 7車立て限定（キャッシュが旧collect由来でも安全）
    raw: dict[str, list[dict]] = {}
    for label, model, f, t in WINDOWS:
        rows = collect_cached(model, f, t)
        rows7 = [r for r in rows
                 if meta.get(r["rk"], (None, None, 0))[2] == 7]
        print(f"[collect] {label} {f}〜{t}: 候補{len(rows)}R → 7車立て{len(rows7)}R")
        raw[label] = rows7

    windows = {
        "IS": annotate(ss_races(raw["IS"]), meta, line_feats),
        "OOS": annotate(ss_races(raw["OOS1"]) + ss_races(raw["FWD"]),
                        meta, line_feats),
    }
    for w, races in windows.items():
        n_miss = sum(1 for r in races if r["line_missing"])
        n_nogap = sum(1 for r in races
                      if not r["line_missing"] and r["avg_gap"] is None)
        print(f"[SSセット] {w}: {len(races)}R "
              f"(ライン特徴欠損{n_miss}R=補正なし扱い / "
              f"ライン1本のみ{n_nogap}R=格差増額なし扱い)")

    rng = np.random.default_rng(SEED)

    # ── 検証1: 選抜カットの公式母集団追試 ──────────────────────────
    print(f"\n{'=' * 78}")
    print("■ 検証1: 種別「選抜」カットの公式母集団（欠車込み）追試")
    print_split_table("選抜（race_typeに「選抜」含む） vs 非選抜",
                      windows, flag_senbatsu, ("選抜", "非選抜"))
    print("\n選抜レースの race_type 内訳:")
    for w, races in windows.items():
        cnt = defaultdict(int)
        for r in races:
            if r["is_senbatsu"]:
                cnt[r["race_type"]] += 1
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        print(f"  {w}: {detail or 'なし'}")
    print("\nブートストラップ ROI差(選抜 - 非選抜) レース単位1万回:")
    for w, races in windows.items():
        flag = [r for r in races if flag_senbatsu(r)]
        rest = [r for r in races if not flag_senbatsu(r)]
        if len(flag) < 10:
            print(f"  {w}: 選抜n={len(flag)} → サンプル不足のためCIスキップ")
            continue
        point, lo, hi = two_sample_roi_diff(flag, rest, rng)
        sig = " *" if hi < 0 or lo > 0 else ""
        print(f"  {w}: 選抜n={len(flag)}/非選抜n={len(rest)} "
              f"ROI差={point:+.3f} [95%CI {lo:+.3f}, {hi:+.3f}]{sig}")
    print("\nカット後の全体ROI変化（非選抜のみ購入 = P1 相当）:")
    for w, races in windows.items():
        st_all = policy_stats(races, lambda r: 1.0)
        st_cut = policy_stats([r for r in races if not flag_senbatsu(r)],
                              lambda r: 1.0)
        print(f"  {w}: 全体ROI {st_all['roi']:.3f} → カット後 {st_cut['roi']:.3f} "
              f"({st_cut['roi'] - st_all['roi']:+.3f}) / "
              f"n {st_all['n']} → {st_cut['n']}")

    # ── 検証3: 条件の重複マトリクス ─────────────────────────────────
    print(f"\n{'=' * 78}")
    print("■ 検証3: 3条件の重複マトリクス（該当レース数と重なり）")
    combos = [
        ("選抜のみ", lambda r: flag_senbatsu(r) and not flag_line4(r)
         and not flag_gap_big(r)),
        ("4分戦のみ", lambda r: flag_line4(r) and not flag_senbatsu(r)
         and not flag_gap_big(r)),
        ("格差大のみ", lambda r: flag_gap_big(r) and not flag_senbatsu(r)
         and not flag_line4(r)),
        ("選抜∧4分戦", lambda r: flag_senbatsu(r) and flag_line4(r)
         and not flag_gap_big(r)),
        ("選抜∧格差大", lambda r: flag_senbatsu(r) and flag_gap_big(r)
         and not flag_line4(r)),
        ("4分戦∧格差大", lambda r: flag_line4(r) and flag_gap_big(r)
         and not flag_senbatsu(r)),
        ("3条件すべて", lambda r: flag_senbatsu(r) and flag_line4(r)
         and flag_gap_big(r)),
        ("いずれも非該当", lambda r: not flag_senbatsu(r) and not flag_line4(r)
         and not flag_gap_big(r)),
    ]
    header = f"{'組合せ':<16}"
    for w in windows:
        header += f" | {w:>3}: {'n':>4} {'ROI':>6}"
    print(header)
    for name, pred in combos:
        line = f"{name:<16}"
        for w, races in windows.items():
            sub = [r for r in races if pred(r)]
            st = policy_stats(sub, lambda r: 1.0)
            roi = f"{st['roi']:.2f}" if st["n"] else "-"
            line += f" | {'':>3}  {st['n']:>4} {roi:>6}"
        print(line)
    print("\n条件別 単独該当数（参考: 干渉の規模感）:")
    for name, pred in (("選抜", flag_senbatsu), ("4分戦(全単騎除く)", flag_line4),
                       ("格差大(avg_gap>=1.5)", flag_gap_big)):
        line = f"  {name:<20}"
        for w, races in windows.items():
            n = sum(1 for r in races if pred(r))
            line += f" {w}:{n}/{len(races)} "
        print(line)

    # ── 検証2+4: 統合ポリシーのシミュレーション ─────────────────────
    policies = make_policies()
    print(f"\n{'=' * 78}")
    print("■ 検証2: 統合ポリシー比較（ベース100円/点・SS三連複軸2車-全）")
    stats: dict[str, dict[str, dict]] = {}
    for w, races in windows.items():
        print(f"\n### {w} 窓 ({len(races)}R)")
        print(f"{'ポリシー':<26} {'購入R':>5} {'的中率':>6} {'投資':>10} "
              f"{'払戻':>10} {'ROI':>6} {'利益':>9} {'最大DD':>8}")
        stats[w] = {}
        for name, fn in policies:
            st = policy_stats(races, fn)
            stats[w][name] = st
            print(f"{name:<26} {st['n']:>5} {st['hit']:>6.1%} "
                  f"{st['bet']:>10,.0f} {st['pay']:>10,.0f} {st['roi']:>6.3f} "
                  f"{st['profit']:>+9,.0f} {st['max_dd']:>8,.0f}")
        print("\n月別ROI:")
        for name, _ in policies:
            print(f"  {name:<26} {fmt_monthly(stats[w][name]['monthly'])}")

    # ── OOSペアドブートストラップ（P0比） ───────────────────────────
    print(f"\n{'=' * 78}")
    print("■ OOSペアドブートストラップ: ROI差(ポリシー - P0) レース単位1万回")
    boot = paired_policy_bootstrap(windows["OOS"], policies, rng)
    for name, (point, lo, hi) in boot.items():
        sig = " *" if lo > 0 or hi < 0 else ""
        print(f"  {name:<26} ROI差={point:+.3f} [95%CI {lo:+.3f}, {hi:+.3f}]{sig}")

    # ── 検証5: IS/OOS 方向一致判定 ──────────────────────────────────
    print(f"\n{'=' * 78}")
    print("■ 検証5: IS/OOS 方向一致（ROI差=ポリシー-P0 が同符号か）+ 絶対利益")
    p0_name = policies[0][0]
    print(f"{'ポリシー':<26} {'ΔROI(IS)':>9} {'ΔROI(OOS)':>10} {'一致':>4} "
          f"{'Δ利益(IS)':>10} {'Δ利益(OOS)':>11}")
    for name, _ in policies[1:]:
        d_is = stats["IS"][name]["roi"] - stats["IS"][p0_name]["roi"]
        d_oos = stats["OOS"][name]["roi"] - stats["OOS"][p0_name]["roi"]
        dp_is = stats["IS"][name]["profit"] - stats["IS"][p0_name]["profit"]
        dp_oos = stats["OOS"][name]["profit"] - stats["OOS"][p0_name]["profit"]
        ok = "○" if d_is * d_oos > 0 else "×"
        print(f"{name:<26} {d_is:>+9.3f} {d_oos:>+10.3f} {ok:>4} "
              f"{dp_is:>+10,.0f} {dp_oos:>+11,.0f}")


if __name__ == "__main__":
    main()
