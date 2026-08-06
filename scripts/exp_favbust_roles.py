#!/usr/bin/env python3
"""P-A: 本命を除いた6車のうち、誰が1着／3着以内に来るのか（役割別の傾向）。

## ユーザー指定（2026-08-06）

> 「P-Aにおいて、6車とした場合、**1着に傾向があるか3着以内の傾向があるのか**検証して。
>  除外した1車が含まれる**ライン2番手**、**別ライン1番手**などで1車が1着、
>  もしくは3着以内なら精度高いのであれば、そこから組み立てができるはず」

## 母集団（Phase 6 で固定済み）

    7車 ∧ 軸1(モデル1着率最上位) == WINTICKET◎
      ∧ 抜け度（1着率の1位−2位差） >= 20pt ∧ バスト確率 上位10%

## 役割の定義（除外した本命 fav との構造的関係）

| 役割 | 定義 |
|---|---|
| `本命ライン番手` | fav と同一 `line_group` で `line_pos` が fav の直後 |
| `本命ライン3番手以降` | fav と同一ラインで、番手より後ろ |
| `別ライン先頭(最強)` | fav と別ラインの `is_line_leader`。ライン得点合計が最大のもの |
| `別ライン先頭(その他)` | 上記以外の別ライン先頭 |
| `別ライン番手` | fav と別ラインの非先頭（単騎を除く） |
| `単騎` | `line_size == 1` |

**⚠️ fav が番手や3番手のこともある**（本命＝マーク屋）。その場合「本命ライン番手」は
fav の直後の車を指す。fav がライン最後尾なら該当なしになる。

## 出す指標

- **選別レース全体**（実運用で賭ける母集団）と
  **本命が実際に飛んだレース**（前提が成立したケース）の両方で
  役割ごとの **1着率 / 3着内率 / 出現率**
- モデル3着内率順 r1..r6 の同指標（役割が順位を超える情報を持つかの対照）
- 役割 × モデル順位のクロス

DB は読み取りのみ（キャッシュ利用）。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_highpay_fav_bust import load_preds3  # noqa: E402
from scripts.exp_highpay_race_model import load_entries, load_races  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache"
SCORED = CACHE_DIR / "favbust_scored.pkl"
PAYCACHE = CACHE_DIR / "favbust_payouts.pkl"
ENTCACHE = CACHE_DIR / "favbust_entries.pkl"

ROLES = ["本命ライン番手", "本命ライン3番手以降", "別ライン先頭(最強)",
         "別ライン先頭(その他)", "別ライン番手", "単騎"]


def load_ents() -> dict:
    if ENTCACHE.exists():
        with ENTCACHE.open("rb") as f:
            print(f"[cache] {ENTCACHE.name}", flush=True)
            return pickle.load(f)
    races = load_races(7)
    ents = load_entries(sorted(races))
    slim = {}
    for rk, lst in ents.items():
        slim[rk] = [{k: e[k] for k in
                     ("frame_no", "line_group", "line_size", "line_pos",
                      "is_line_leader", "race_point", "style", "prediction_mark")}
                    for e in lst]
    tmp = ENTCACHE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(ENTCACHE)
    return slim


def role_of(ents: list[dict], fav: int) -> dict[int, str]:
    """本命 fav から見た各車の役割を返す（fav 自身は除く）。"""
    by_f = {int(e["frame_no"]): e for e in ents}
    fav_e = by_f[fav]
    fav_lg = fav_e["line_group"]
    fav_pos = fav_e["line_pos"] or 0

    # ライン得点合計（別ライン先頭の強弱判定用）
    line_rp: dict = defaultdict(float)
    for e in ents:
        if e["line_group"] is not None:
            line_rp[e["line_group"]] += float(e["race_point"] or 0)
    other_leads = [int(e["frame_no"]) for e in ents
                   if int(e["frame_no"]) != fav and (e["is_line_leader"] or 0) == 1
                   and e["line_group"] is not None and e["line_group"] != fav_lg
                   and (e["line_size"] or 1) > 1]
    strongest = (max(other_leads,
                     key=lambda f: line_rp[by_f[f]["line_group"]])
                 if other_leads else None)

    out: dict[int, str] = {}
    for e in ents:
        f = int(e["frame_no"])
        if f == fav:
            continue
        if (e["line_size"] or 1) == 1:
            out[f] = "単騎"
        elif fav_lg is not None and e["line_group"] == fav_lg:
            pos = e["line_pos"] or 0
            out[f] = "本命ライン番手" if pos == fav_pos + 1 else "本命ライン3番手以降"
        elif (e["is_line_leader"] or 0) == 1:
            out[f] = "別ライン先頭(最強)" if f == strongest else "別ライン先頭(その他)"
        else:
            out[f] = "別ライン番手"
    return out


def tabulate(rows: list[tuple], title: str, n_race: int) -> None:
    """rows: (key, is_1st, is_top3) のリスト。"""
    agg = defaultdict(lambda: {"n": 0, "w": 0, "t": 0})
    for k, w, t in rows:
        a = agg[k]
        a["n"] += 1
        a["w"] += w
        a["t"] += t
    print(f"\n  ── {title}（{n_race:,}レース）──")
    print("    区分                    出現数  1レースあたり   1着率    3着内率")
    for k in sorted(agg, key=lambda x: -(agg[x]["w"] / max(agg[x]["n"], 1))):
        a = agg[k]
        if a["n"] < 30:
            continue
        print(f"    {k:<22} {a['n']:7} {a['n'] / n_race:9.2f}人  "
              f"{a['w'] / a['n'] * 100:7.2f}%  {a['t'] / a['n'] * 100:8.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.20)
    ap.add_argument("--top-frac", type=float, default=0.10)
    args = ap.parse_args()

    with SCORED.open("rb") as f:
        data = pickle.load(f)
    with PAYCACHE.open("rb") as f:
        pay = pickle.load(f)
    ents_all = load_ents()
    pr_all = load_preds3()

    data = [d for d in data if d["race_key"] in pay]
    strat = [d for d in data if d["fav_ppw_gap12"] >= args.gap]
    thr = np.quantile([d["score"] for d in strat], 1 - args.top_frac)
    sel = [d for d in strat if d["score"] >= thr]
    bust = [d for d in sel if d["bust"] == 1]
    print(f"選別レース {len(sel):,} / うち本命が実際に飛んだ {len(bust):,} "
          f"({len(bust) / len(sel) * 100:.2f}%)")

    def collect(group: list[dict], key_fn):
        rows = []
        for d in group:
            rk = d["race_key"]
            pr = pr_all.get(rk)
            ents = ents_all.get(rk)
            if not pr or not ents:
                continue
            fav = max(pr, key=lambda f: pr[f][1])
            roles = role_of(ents, fav)
            others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
            rank = {f: i + 1 for i, f in enumerate(others)}
            top3 = pay[rk]["order"]
            for f in others:
                rows.append((key_fn(f, roles, rank), 1 if top3[0] == f else 0,
                             1 if f in top3 else 0))
        return rows

    for gname, grp in (("【選別レース全体】＝実運用で賭ける母集団", sel),
                       ("【本命が実際に飛んだレース】＝前提が成立したケース", bust)):
        print(f"\n{'=' * 88}\n=== {gname} ===")
        tabulate(collect(grp, lambda f, ro, rk: ro.get(f, "?")),
                 "役割別", len(grp))
        tabulate(collect(grp, lambda f, ro, rk: f"モデル順 r{rk[f]}"),
                 "モデル3着内率順 別", len(grp))

    # --- 役割 × モデル順位のクロス（前提成立ケース）---
    print(f"\n{'=' * 88}\n=== 役割 × モデル順位 クロス（本命が飛んだレース {len(bust):,}R）===")
    rows = collect(bust, lambda f, ro, rk: (ro.get(f, "?"), min(rk[f], 4)))
    agg = defaultdict(lambda: {"n": 0, "w": 0, "t": 0})
    for k, w, t in rows:
        a = agg[k]
        a["n"] += 1
        a["w"] += w
        a["t"] += t
    print("    役割                   モデル順   出現数    1着率    3着内率")
    for k in sorted(agg, key=lambda x: -(agg[x]["w"] / max(agg[x]["n"], 1))):
        a = agg[k]
        if a["n"] < 50:
            continue
        lab = f"r{k[1]}" + ("以下" if k[1] == 4 else "")
        print(f"    {k[0]:<22} {lab:<8} {a['n']:6}  {a['w'] / a['n'] * 100:7.2f}%  "
              f"{a['t'] / a['n'] * 100:8.2f}%")

    # --- 1着が誰だったかの内訳 ---
    print(f"\n{'=' * 88}\n=== 本命が飛んだレースで「1着になったのは誰か」の内訳 ===")
    cnt_role = defaultdict(int)
    cnt_rank = defaultdict(int)
    for d in bust:
        rk = d["race_key"]
        pr, ents = pr_all.get(rk), ents_all.get(rk)
        if not pr or not ents:
            continue
        fav = max(pr, key=lambda f: pr[f][1])
        roles = role_of(ents, fav)
        others = sorted((f for f in pr if f != fav), key=lambda f: -pr[f][0])
        w = pay[rk]["order"][0]
        if w == fav:
            continue
        cnt_role[roles.get(w, "?")] += 1
        cnt_rank[f"r{others.index(w) + 1}"] += 1
    tot = sum(cnt_role.values())
    print(f"    役割別（n={tot}）")
    for k, v in sorted(cnt_role.items(), key=lambda x: -x[1]):
        print(f"      {k:<22} {v:5} ({v / tot * 100:5.1f}%)")
    print(f"    モデル順別（n={sum(cnt_rank.values())}）")
    for k in sorted(cnt_rank):
        v = cnt_rank[k]
        print(f"      {k:<22} {v:5} ({v / tot * 100:5.1f}%)")


if __name__ == "__main__":
    main()
