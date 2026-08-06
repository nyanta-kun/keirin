#!/usr/bin/env python3
"""高額払い戻し（1万円 → 30万円以上）を狙う三連単構成の設計と検証。

## 目的関数（ROIではない）

    P(高額) = P( 1レース1万円の購入に対する払い戻しが 30万円以上 )

N点等分なら 1点 10000/N 円なので、的中目のオッズ o に対して

    払い戻し = (10000/N) * o >= 300000  ⇔  **o >= 30N**

## 事前に確定している算術（実験前に必ず読む）

市場が効率的（1円あたりの期待回収 = 1 - 控除率 = 0.75）なら、オッズ o の1点の
真の確率は p = 0.75 / o。したがって N 点すべてを o_i >= 30N で買ったとき

    P(高額) = Σ 0.75 / o_i  <=  N * 0.75 / (30N) = **2.5%**

- **N に依存しない。** 点数を増やしても要求オッズが同じ比率で上がるため相殺する。
- **等号は o_i がちょうど 30N のときだけ。** オッズが要求ラインより高いほど損。
  ⇒ 狙うべきは「最大配当」ではなく **30N 倍をわずかに超える帯**。
- 実測の帯ROIが 0.75 を下回るならその比率だけ上限も下がる
  （`exp_highpay_band_roi.py`: 7車三連単は 30〜600倍帯で 71〜76%＝ほぼ控除率どおり、
   600倍超で 64.5%、1200倍超で 51.6% と崩れる）。

## したがって検証すべき唯一の問いは

    **「要求オッズ帯の中で、モデルは市場より良い点を選べるか」**

帯を固定すると市場価格はほぼ揃うので、市場の順位付け情報は帯内では弱くなる。
全通りでの順位付けではモデルは市場に負けると実証済み
（[[keirin_clean_baseline_market_efficiency_2026_07_30]]）だが、
**帯内での順位付けは別問題**でまだ一度も測っていない。

## 事前宣言（掃引窓で作り、確認窓で一度きり検証する）

- 主要指標: **P(高額)**。副次: ROI・的中時配当中央値
- 検定対象: `model`（帯内モデル順） vs `low`（帯内オッズ昇順＝算術上の最適）
  vs `rand`（帯内ランダム＝対照）
- 掃引窓 2025-07-01〜2026-07-15 / 確認窓 2024-07-01〜2025-06-30 /
  未使用 2026-07-16〜2026-08-04

⚠️ 最終オッズを使う（＝最良ケース）。朝の入稿時点のオッズとの乖離は別途監査する。
   ここで成立しなければ朝オッズでは確実に成立しない。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import glob
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache"
ODDS_CACHE = CACHE_DIR / "highpay_tf_odds_{n_car}.pkl"

WINDOWS = {
    "掃引窓": ("2025-07-01", "2026-07-15"),
    "確認窓": ("2024-07-01", "2025-06-30"),
    "未使用": ("2026-07-16", "2026-08-04"),
}
STAKE = 10_000
HIGHPAY = 300_000
# 帯の上限倍率。30N 〜 30N*BAND_TOP を「適格」とする
BAND_TOP = 2.0


# ---------------------------------------------------------------- データ読み込み
def load_preds() -> pd.DataFrame:
    """月次/四半期の walk-forward 予測キャッシュを連結する（honest）。"""
    frames = []
    for f in sorted(glob.glob(str(CACHE_DIR / "wf_preds_*.pkl"))):
        frames.append(pd.read_pickle(f))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["race_key", "frame_no"], keep="last")


def load_races(n_car: int) -> pd.DataFrame:
    q = """
    WITH res AS (
      SELECT race_key,
             MAX(CASE WHEN finish_order = 1 THEN frame_no END) AS f1,
             MAX(CASE WHEN finish_order = 2 THEN frame_no END) AS f2,
             MAX(CASE WHEN finish_order = 3 THEN frame_no END) AS f3
      FROM keirin.wt_entries GROUP BY race_key
    )
    SELECT r.race_key, r.race_date, r.race_type, r.grade, r.venue_id,
           res.f1, res.f2, res.f3
    FROM res JOIN keirin.wt_races r ON r.race_key = res.race_key
    WHERE r.race_date >= '2024-07-01' AND r.n_entries = ?
      AND res.f1 IS NOT NULL AND res.f2 IS NOT NULL AND res.f3 IS NOT NULL
    """
    with get_connection() as c:
        return pd.DataFrame([dict(r) for r in c.execute(q, (n_car,)).fetchall()])


def load_tf_odds(race_keys: list[str], n_car: int, lo: float, hi: float) -> dict:
    """三連単オッズを [lo, hi] に絞って取得（キャッシュあり）。

    全通り(7車=210)を引くと数千万行になるため、検証で使う帯だけを引く。
    キャッシュは帯の範囲もキーに含める（範囲を広げたら別ファイルになる）。
    """
    path = Path(str(ODDS_CACHE).format(n_car=n_car)).with_name(
        f"highpay_tf_odds_{n_car}_{lo:g}_{hi:g}.pkl")
    if path.exists():
        with path.open("rb") as f:
            print(f"  [cache] {path.name}", flush=True)
            return pickle.load(f)

    out: dict[str, dict[str, float]] = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 400):
            ch = race_keys[i:i + 400]
            ph = ",".join("?" * len(ch))
            rows = c.execute(
                "SELECT race_key, combination, odds_value FROM keirin.wt_odds "
                f"WHERE bet_type='trifecta' AND race_key IN ({ph}) "
                "AND odds_value >= ? AND odds_value <= ?",
                ch + [lo, hi]).fetchall()
            for r in rows:
                out[r["race_key"]][r["combination"]] = float(r["odds_value"])
            if (i // 400) % 10 == 0:
                print(f"  odds {i + len(ch)}/{len(race_keys)}", flush=True)
    out = dict(out)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:               # 保存失敗は握り潰さない
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    print(f"  [built] {path.name} ({len(out)} レース)", flush=True)
    return out


# ---------------------------------------------------------------- モデル確率
def plackett_luce(frames: list[int], w: dict[int, float]) -> dict[str, float]:
    """車番ごとの強さ w から三連単各組の確率を作る（Plackett–Luce）。

    w には 1着率モデル(ppw) を使う。2・3着の相対強さも同じ w で近似する。
    """
    total = sum(w.values())
    out: dict[str, float] = {}
    for a in frames:
        pa = w[a] / total
        r1 = total - w[a]
        if r1 <= 0:
            continue
        for b in frames:
            if b == a:
                continue
            pb = w[b] / r1
            r2 = r1 - w[b]
            if r2 <= 0:
                continue
            for c in frames:
                if c in (a, b):
                    continue
                out[f"{a}-{b}-{c}"] = pa * pb * (w[c] / r2)
    return out


# ---------------------------------------------------------------- 戦略
def pick(elig: list[tuple[str, float]], n_pt: int, mode: str,
         mprob: dict[str, float], rng: random.Random) -> list[tuple[str, float]]:
    """適格な (組, オッズ) から n_pt 点を選ぶ。"""
    if len(elig) < n_pt:
        return []
    if mode == "low":                       # 要求ラインに最も近い＝算術上の最適
        return sorted(elig, key=lambda x: x[1])[:n_pt]
    if mode == "model":
        return sorted(elig, key=lambda x: -mprob.get(x[0], 0.0))[:n_pt]
    if mode == "model_ev":                  # モデル確率 × オッズ（期待値順）
        return sorted(elig, key=lambda x: -mprob.get(x[0], 0.0) * x[1])[:n_pt]
    if mode == "rand":
        return rng.sample(elig, n_pt)
    raise ValueError(mode)


def evaluate(races: pd.DataFrame, preds: pd.DataFrame, odds: dict,
             n_pts: list[int], modes: list[str], seed: int = 7) -> pd.DataFrame:
    by_race = {k: g for k, g in preds.groupby("race_key")}
    rng = random.Random(seed)
    acc: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "hit": 0, "ret": 0.0, "pays": []})

    for row in races.itertuples():
        g = by_race.get(row.race_key)
        od = odds.get(row.race_key)
        if g is None or not od or len(g) < 3:
            continue
        frames = [int(x) for x in g["frame_no"]]
        w = {int(f): max(float(p), 1e-6)
             for f, p in zip(g["frame_no"], g["ppw"])}
        mprob = plackett_luce(frames, w)
        win = f"{int(row.f1)}-{int(row.f2)}-{int(row.f3)}"

        for n_pt in n_pts:
            thr = 30.0 * n_pt
            elig = [(k, v) for k, v in od.items()
                    if thr <= v <= thr * BAND_TOP]
            for mode in modes:
                sel = pick(elig, n_pt, mode, mprob, rng)
                if not sel:
                    continue
                a = acc[(n_pt, mode)]
                a["n"] += 1
                for comb, o in sel:
                    if comb == win:
                        pay = (STAKE / n_pt) * o
                        a["hit"] += 1
                        a["ret"] += pay
                        a["pays"].append(pay)

    out = []
    for (n_pt, mode), a in sorted(acc.items()):
        if not a["n"]:
            continue
        pays = a["pays"]
        big = sum(1 for p in pays if p >= HIGHPAY)
        out.append({
            "N": n_pt, "mode": mode, "races": a["n"],
            "hit%": a["hit"] / a["n"] * 100,
            "高額%": big / a["n"] * 100,
            "高額/100R": big / a["n"] * 100,
            "ROI%": a["ret"] / (a["n"] * STAKE) * 100,
            "配当中央": float(np.median(pays)) if pays else 0.0,
            "n高額": big,
        })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-car", type=int, default=7)
    ap.add_argument("--windows", default="掃引窓")
    ap.add_argument("--modes", default="low,model,model_ev,rand")
    ap.add_argument("--n-pts", default="1,2,3,5,8")
    args = ap.parse_args()

    n_pts = [int(x) for x in args.n_pts.split(",")]
    modes = args.modes.split(",")
    lo = 30.0 * min(n_pts)
    hi = 30.0 * max(n_pts) * BAND_TOP

    print("予測キャッシュ読込...", flush=True)
    preds = load_preds()
    print(f"  {len(preds):,} エントリ", flush=True)
    print("レース読込...", flush=True)
    races = load_races(args.n_car)
    races = races[races["race_key"].isin(set(preds["race_key"]))]
    print(f"  {len(races):,} レース ({args.n_car}車・予測あり)", flush=True)
    print(f"三連単オッズ読込 [{lo:g}, {hi:g}]倍 ...", flush=True)
    odds = load_tf_odds(sorted(races["race_key"]), args.n_car, lo, hi)

    for wname in args.windows.split(","):
        d_from, d_to = WINDOWS[wname]
        sub = races[(races["race_date"] >= d_from) & (races["race_date"] <= d_to)]
        print(f"\n{'=' * 78}\n=== {wname} {d_from}〜{d_to}  {len(sub):,}レース ===")
        res = evaluate(sub, preds, odds, n_pts, modes)
        if res.empty:
            print("  該当なし")
            continue
        for n_pt in n_pts:
            block = res[res["N"] == n_pt]
            if block.empty:
                continue
            print(f"\n  --- N={n_pt}点 (要求 {30 * n_pt}倍以上 / "
                  f"帯 {30 * n_pt}〜{int(30 * n_pt * BAND_TOP)}倍) ---")
            print("    mode        レース  的中%   高額%  高額数  ROI%   配当中央")
            for r in block.to_dict("records"):
                print(f"    {r['mode']:<11} {r['races']:6}  {r['hit%']:5.2f}  "
                      f"{r['高額%']:5.2f}  {r['n高額']:5}  "
                      f"{r['ROI%']:5.1f}  {r['配当中央']:9.0f}")


if __name__ == "__main__":
    main()
