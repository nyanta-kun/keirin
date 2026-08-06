#!/usr/bin/env python3
"""高配当が発生するレースの「選別」で高額払い戻し率が上がるかを測る。

## 位置づけ

ユーザー方針（2026-08-06）:
  「まずは個別のレースではなく、**高配当の発生するレースの選別モデル**の検討が始め。
   三連複であれば1点・2点での的中で30〜50倍以上、三連単は10点くらいまで」

`exp_highpay_trifecta_design.py` で、**買い目の中身**（帯内でどの目を選ぶか）は
モデルでは改善できないと確定した（帯内モデル順 < オッズ昇順 ≒ ランダム）。
残っているレバーは **どのレースに賭けるか** だけ。本スクリプトはそれを測る。

## 何を測るのか（構成は固定・レース選別だけを動かす）

1レース1万円・払い戻し30万円以上を「高額」とする。等分なら要求オッズは 30×点数:

| 構成 | 点数 | 1点あたり | 要求オッズ | 帯ROI（実測・7車） |
|---|---|---|---|---|
| `trio1`  | 三連複1点 | 10,000円 | **30倍**  | 30-60倍帯 77.8% |
| `trio2`  | 三連複2点 |  5,000円 | **60倍**  | 60-90倍帯 58.3% |
| `tf10`   | 三連単10点|  1,000円 | **300倍** | 300-600倍帯 73.2% |
| `tf_budget` | 三連単・不等分 | 300000/o_i | 各点が単独で30万 | — |

いずれも「要求ラインに最も近い目から順に」買う（＝算術上の最適）。

## 事前に確定している上限（レース選別でも破れない部分）

市場効率が保たれる限り、オッズ o の1点の真の確率は p ≒ 帯ROI / o。全点が
「単独で30万円を返す」ように張るとき、投資1万円に対して

    P(高額) = Σ p_i = 帯ROI × Σ (1/o_i) = 帯ROI × (10000/300000) = **帯ROI / 30**

**点数・券種・配分に一切依存しない。** よってレース選別が効くのは
**「その帯の帯ROIが素より高いレース群」が存在するときだけ**で、それは
市場エッジの存在と同義。ここで探しているのはまさにそれ。

## セグメント

`race_type`（予選/準決勝/決勝…）/ `grade` / 分戦数 `n_lines` / 単騎の人数 /
級班の混在 / ライン最大サイズ / モデル予測のエントロピー（＝荒れ予測）/
競走得点の標準偏差 `rp_std`（既存の波乱度モデルで最重要特徴）/ 日中・ナイター。

⚠️ 掃引窓で候補を作り、確認窓で一度きり検証する。最終オッズを使う（最良ケース。
   朝オッズでは成立しないことは `exp_highpay_odds_drift.py` で確定済み）。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import glob
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache"
STAKE = 10_000
HIGHPAY = 300_000

WINDOWS = {
    "掃引窓": ("2025-07-01", "2026-07-15"),
    "確認窓": ("2024-07-01", "2025-06-30"),
    "未使用": ("2026-07-16", "2026-08-04"),
}


# ---------------------------------------------------------------- 読み込み
def load_preds() -> pd.DataFrame:
    frames = [pd.read_pickle(f)
              for f in sorted(glob.glob(str(CACHE_DIR / "wf_preds_*.pkl")))]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["race_key", "frame_no"], keep="last")


def load_race_meta(n_car: int) -> pd.DataFrame:
    """レース属性 + 結果 + 盤面構成（ライン・級班）をまとめて取る。"""
    q = """
    WITH res AS (
      SELECT race_key,
             MAX(CASE WHEN finish_order = 1 THEN frame_no END) AS f1,
             MAX(CASE WHEN finish_order = 2 THEN frame_no END) AS f2,
             MAX(CASE WHEN finish_order = 3 THEN frame_no END) AS f3
      FROM keirin.wt_entries GROUP BY race_key
    ), ent AS (
      SELECT race_key,
             MAX(n_lines) AS n_lines,
             MAX(line_size) AS max_line_size,
             COUNT(DISTINCT player_class) AS n_classes,
             SUM(CASE WHEN line_size = 1 THEN 1 ELSE 0 END) AS n_solo,
             STDDEV_POP(race_point) AS rp_std,
             AVG(race_point) AS rp_mean,
             SUM(CASE WHEN front_runner = 1 THEN 1 ELSE 0 END) AS n_front
      FROM keirin.wt_entries GROUP BY race_key
    )
    SELECT r.race_key, r.race_date, r.race_type, r.grade, r.venue_id,
           r.start_at, res.f1, res.f2, res.f3,
           ent.n_lines, ent.max_line_size, ent.n_classes, ent.n_solo,
           ent.rp_std, ent.rp_mean, ent.n_front
    FROM res
    JOIN keirin.wt_races r ON r.race_key = res.race_key
    JOIN ent ON ent.race_key = res.race_key
    WHERE r.race_date >= '2024-07-01' AND r.n_entries = ?
      AND res.f1 IS NOT NULL AND res.f2 IS NOT NULL AND res.f3 IS NOT NULL
    """
    with get_connection() as c:
        return pd.DataFrame([dict(r) for r in c.execute(q, (n_car,)).fetchall()])


def load_odds(race_keys: list[str], bet_type: str, n_car: int,
              lo: float, hi: float) -> dict:
    tag = f"highpay_{bet_type}_{n_car}_{lo:g}_{hi:g}.pkl"
    path = CACHE_DIR / tag
    if path.exists():
        with path.open("rb") as f:
            print(f"  [cache] {tag}", flush=True)
            return pickle.load(f)
    out: dict[str, dict] = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 400):
            ch = race_keys[i:i + 400]
            ph = ",".join("?" * len(ch))
            rows = c.execute(
                "SELECT race_key, combination, odds_value FROM keirin.wt_odds "
                f"WHERE bet_type=? AND race_key IN ({ph}) "
                "AND odds_value >= ? AND odds_value <= ?",
                [bet_type] + ch + [lo, hi]).fetchall()
            for r in rows:
                key = (r["combination"] if bet_type == "trifecta" else
                       frozenset(int(x) for x in re.split(r"[-=→]", r["combination"])))
                if bet_type == "trio" and len(key) != 3:
                    continue
                out[r["race_key"]][key] = float(r["odds_value"])
            if (i // 400) % 15 == 0:
                print(f"  {bet_type} {i + len(ch)}/{len(race_keys)}", flush=True)
    out = dict(out)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    print(f"  [built] {tag} ({len(out)} レース)", flush=True)
    return out


# ---------------------------------------------------------------- 構成
def structures(trio_od: dict, tf_od: dict) -> dict[str, list[tuple]]:
    """(組, オッズ, 賭け金) のリストを構成ごとに返す。買えない構成は空。"""
    out: dict[str, list[tuple]] = {}

    def lowest(od: dict, thr: float, n: int):
        el = sorted(((k, v) for k, v in od.items() if v >= thr), key=lambda x: x[1])
        return el[:n] if len(el) >= n else []

    for name, od, n_pt in (("trio1", trio_od, 1), ("trio2", trio_od, 2),
                           ("tf10", tf_od, 10)):
        sel = lowest(od, 30.0 * n_pt, n_pt)
        out[name] = [(k, o, STAKE / n_pt) for k, o in sel]

    # 不等分: 各点が単独で30万を返すよう s_i = 300000/o_i を割り当て、
    # 要求ラインに近い（＝確率の高い）目から予算1万円を使い切るまで積む。
    sel, spent = [], 0.0
    for k, o in sorted(((k, v) for k, v in tf_od.items() if v >= 30.0),
                       key=lambda x: x[1]):
        s = HIGHPAY / o
        if spent + s > STAKE + 1e-9:
            continue
        sel.append((k, o, s))
        spent += s
        if STAKE - spent < HIGHPAY / max(tf_od.values(), default=1e9):
            break
    out["tf_budget"] = sel
    return out


# ---------------------------------------------------------------- セグメント
def seg_values(row, ent: float) -> dict[str, str]:
    hour = None
    try:
        import datetime as dt
        hour = dt.datetime.fromtimestamp(int(row.start_at)).hour
    except Exception:
        pass
    return {
        "race_type": str(row.race_type),
        "grade": str(row.grade),
        "n_lines": f"分戦{int(row.n_lines)}" if row.n_lines else "不明",
        "n_solo": f"単騎{int(row.n_solo)}" if row.n_solo is not None else "不明",
        "n_classes": f"級班{int(row.n_classes)}種" if row.n_classes else "不明",
        "max_line": f"最大ライン{int(row.max_line_size)}" if row.max_line_size else "不明",
        "n_front": f"逃型{int(row.n_front)}人" if row.n_front is not None else "不明",
        "rp_std": _bucket(row.rp_std, [1.0, 2.0, 3.0, 5.0], "rp_std"),
        "entropy": _bucket(ent, [1.55, 1.70, 1.80, 1.88], "ent"),
        "time": ("ナイター" if hour is not None and hour >= 17
                 else "日中" if hour is not None else "不明"),
    }


def _bucket(v, edges, name) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{name}:不明"
    v = float(v)
    for i, e in enumerate(edges):
        if v < e:
            return f"{name}:Q{i + 1}(<{e})"
    return f"{name}:Q{len(edges) + 1}(>={edges[-1]})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-car", type=int, default=7)
    ap.add_argument("--windows", default="掃引窓")
    ap.add_argument("--min-n", type=int, default=400,
                    help="セグメントの最小レース数")
    args = ap.parse_args()

    preds = load_preds()
    meta = load_race_meta(args.n_car)
    meta = meta[meta["race_key"].isin(set(preds["race_key"]))]
    print(f"{args.n_car}車 {len(meta):,} レース（予測あり）", flush=True)
    keys = sorted(meta["race_key"])
    trio_all = load_odds(keys, "trio", args.n_car, 25.0, 1e6)
    tf_all = load_odds(keys, "trifecta", args.n_car, 25.0, 1e6)

    ppw = {k: g for k, g in preds.groupby("race_key")}

    for wname in args.windows.split(","):
        d_from, d_to = WINDOWS[wname]
        sub = meta[(meta["race_date"] >= d_from) & (meta["race_date"] <= d_to)]
        print(f"\n{'=' * 90}\n=== {wname} {d_from}〜{d_to}  {len(sub):,}レース ===")

        # 構成ごとの全体成績 + セグメント別
        overall = defaultdict(lambda: {"n": 0, "big": 0, "ret": 0.0, "hit": 0})
        seg = defaultdict(lambda: defaultdict(
            lambda: {"n": 0, "big": 0, "ret": 0.0, "hit": 0}))

        for row in sub.itertuples():
            g = ppw.get(row.race_key)
            if g is None:
                continue
            p = np.asarray(g["ppw"], dtype=float)
            p = p / p.sum() if p.sum() > 0 else p
            ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
            trio_od = trio_all.get(row.race_key) or {}
            tf_od = tf_all.get(row.race_key) or {}
            if not trio_od and not tf_od:
                continue
            f1, f2, f3 = int(row.f1), int(row.f2), int(row.f3)
            win_tf = f"{f1}-{f2}-{f3}"
            win_tr = frozenset((f1, f2, f3))
            segs = seg_values(row, ent)

            for name, sel in structures(trio_od, tf_od).items():
                if not sel:
                    continue
                win = win_tr if name.startswith("trio") else win_tf
                ret = 0.0
                hit = big = 0
                for k, o, s in sel:
                    if k == win:
                        pay = s * o
                        ret += pay
                        hit += 1
                        if pay >= HIGHPAY - 1:
                            big += 1
                cost = sum(s for _, _, s in sel)
                for tgt in (overall[name],) + tuple(
                        seg[(name, dim)][val] for dim, val in segs.items()):
                    tgt["n"] += 1
                    tgt["big"] += big
                    tgt["hit"] += hit
                    tgt["ret"] += ret
                    tgt["cost"] = tgt.get("cost", 0.0) + cost

        print("\n  --- 構成別 全体 ---")
        print("    構成         レース   点数  的中%   高額%   高額数   ROI%")
        for name, a in overall.items():
            if not a["n"]:
                continue
            print(f"    {name:<11} {a['n']:7} {'':6} {a['hit'] / a['n'] * 100:6.2f} "
                  f"{a['big'] / a['n'] * 100:6.2f} {a['big']:8} "
                  f"{a['ret'] / a['cost'] * 100:6.1f}")

        for name in overall:
            base = overall[name]
            if not base["n"]:
                continue
            base_rate = base["big"] / base["n"] * 100
            print(f"\n  --- {name} セグメント別（全体 高額 {base_rate:.2f}% ・"
                  f"n>={args.min_n} のみ）---")
            rows = []
            for (nm, dim), vals in seg.items():
                if nm != name:
                    continue
                for val, a in vals.items():
                    if a["n"] < args.min_n:
                        continue
                    rows.append((a["big"] / a["n"] * 100, dim, val, a))
            for rate, dim, val, a in sorted(rows, reverse=True)[:12]:
                print(f"    {dim:<10} {val:<18} n={a['n']:6}  高額 {rate:5.2f}% "
                      f"(基準比 {rate - base_rate:+5.2f}) 高額数{a['big']:5} "
                      f"ROI {a['ret'] / a['cost'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
