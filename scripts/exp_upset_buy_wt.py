"""波乱傾向レースの買い目方針分析（2026-07-15）

Q1: 波乱傾向レース(競争得点が拮抗=rp_std下位)の1着は
    「人気ライン頭」か「人気薄ライン頭」か（or 番手/非頭か）。
Q2: 波乱傾向レースに絞った上で、候補買い目の的中率が 20〜30% 以上を
    満たすものを探す（10倍割れ一発頼みは不可）。

市場人気: 二車単(exacta)オッズから各選手の暗黙勝率 p_win(h) ∝ Σ_j 1/exa(h→j) を復元。
ライン頭: is_line_leader==1。
前提: perhorse_n7.pkl（クリーン7車・落車失格欠車なし）
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.database import get_connection  # noqa: E402
from exp_stable_top2_wt import seg, DISC, CONF, CACHE_DIR  # noqa: E402

PH = CACHE_DIR / "perhorse_n7.pkl"


def load_odds(race_keys):
    exa, tri, wide = {}, {}, {}
    rks = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(rks), 900):
            chunk = rks[i:i + 900]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('exacta','trifecta','quinellaPlace') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                if od is None or not (0 < float(od) < 90000):
                    continue
                try:
                    parts = tuple(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if bt == "exacta" and len(parts) == 2:
                    exa.setdefault(rk, {})[parts] = float(od)
                elif bt == "trifecta" and len(parts) == 3:
                    tri.setdefault(rk, {})[parts] = float(od)
                elif bt == "quinellaPlace" and len(parts) == 2:
                    wide.setdefault(rk, {})[frozenset(parts)] = float(od)
    return exa, tri, wide


def market_winscore(exa_r, frames):
    """二車単から各選手の暗黙勝率スコア。"""
    sc = {f: 0.0 for f in frames}
    for (i, j), od in exa_r.items():
        if i in sc and od > 0:
            sc[i] += 1.0 / od
    return sc


def build(ph, exa):
    recs = []
    for rk, g in ph.groupby("race_key"):
        g = g.sort_values("model_rank")
        if len(g) != 7:
            continue
        frames = g["frame_no"].astype(int).tolist()
        rp = dict(zip(frames, g["race_point"].astype(float)))
        lg = dict(zip(frames, g["line_group"].astype(int)))
        head = dict(zip(frames, g["is_line_leader"].astype(int)))
        fo = {int(f): (int(o) if pd.notna(o) else 99)
              for f, o in zip(frames, g["finish_order"])}
        pos = {v: k for k, v in fo.items()}
        if not all(p in pos for p in (1, 2, 3)):
            continue
        er = exa.get(rk)
        if not er:
            continue
        ws = market_winscore(er, frames)
        # ライン人気度 = ライン内メンバーの市場勝率スコア合計
        line_ws = defaultdict(float)
        for f in frames:
            line_ws[lg[f]] += ws[f]
        line_rank = {l: r for r, (l, _) in enumerate(
            sorted(line_ws.items(), key=lambda x: -x[1]), start=1)}
        rp_arr = np.array(list(rp.values()))

        def classify(frame):
            """1着馬などの分類: (ライン人気順位, 頭か)"""
            return line_rank[lg[frame]], head[frame]

        w = pos[1]
        wl_rank, w_head = classify(w)
        recs.append({
            "race_key": rk, "race_date": g["race_date"].iloc[0],
            "rp_std": rp_arr.std(),
            "pos1": w, "pos2": pos[2], "pos3": pos[3],
            "win_line_rank": wl_rank, "win_is_head": w_head,
            # 人気ライン(rank1)頭 か / 人気薄ライン(rank>=2)頭 か / 非頭(番手)か
            "win_type": ("人気L頭" if (wl_rank == 1 and w_head) else
                         ("人気薄L頭" if (wl_rank >= 2 and w_head) else
                          ("人気L番手" if wl_rank == 1 else "人気薄L番手"))),
            "frames": frames, "lg": lg, "head": head, "line_rank": line_rank,
            "top3": {pos[1], pos[2], pos[3]},
        })
    return pd.DataFrame(recs)


def main():
    ph = pd.read_pickle(PH)
    exa, tri, wide = load_odds(ph["race_key"].unique().tolist())
    df = build(ph, exa)
    print(f"クリーン7車・オッズ有: {len(df):,}レース")

    # 波乱傾向レース = rp_std 下位1/3（拮抗）
    th = df["rp_std"].quantile(1 / 3)
    upset = df[df["rp_std"] <= th].copy()
    calm = df[df["rp_std"] > df["rp_std"].quantile(2 / 3)].copy()
    print(f"波乱傾向(rp_std<= {th:.2f}, 拮抗): {len(upset):,} / 平穏(rp_std上位): {len(calm):,}")

    # === Q1: 1着の型分布 ===
    print("\n=== Q1: 1着の型（人気ライン=市場勝率でrank1） ===")
    for name, d in (("波乱傾向(拮抗)", upset), ("平穏(実力差)", calm), ("全体", df)):
        vc = d["win_type"].value_counts(normalize=True)
        print(f"  [{name}] " + " / ".join(f"{k}:{vc.get(k,0):.1%}"
              for k in ["人気L頭", "人気L番手", "人気薄L頭", "人気薄L番手"]))
        print(f"     └ 1着が頭(先頭)である率={d['win_is_head'].mean():.1%}  "
              f"1着ラインの人気順位分布: " +
              " ".join(f"{r}位{(d['win_line_rank']==r).mean():.0%}" for r in (1, 2, 3, 4)))

    # === Q2: 波乱傾向レースでの候補買い目 的中率 ===
    print("\n" + "=" * 66)
    print("=== Q2: 波乱傾向レースでの候補買い目 的中率（DISC/CONF）===")

    def eval_bets(pop):
        for wl, w in (("DISC", DISC), ("CONF", CONF)):
            s = seg(pop, w)
            days = s["race_date"].nunique() or 1
            n = len(s)
            if not n:
                continue
            cnt = defaultdict(int); has = defaultdict(int)
            bet = defaultdict(int); pay = defaultdict(int)   # ROI用（ワイド/三連複）
            for _, r in s.iterrows():
                wr = wide.get(r["race_key"], {})
                tr = tri.get(r["race_key"], {})
                heads = [f for f in r["frames"] if r["head"][f]]
                heads_by_pop = sorted(heads, key=lambda f: r["line_rank"][r["lg"][f]])
                underdogs = [f for f in heads_by_pop if r["line_rank"][r["lg"][f]] >= 2]
                top3 = r["top3"]

                def wide_bet(key, pairs):
                    legs = {frozenset(p): wr.get(frozenset(p)) for p in pairs}
                    legs = {k: v for k, v in legs.items() if v}
                    if not legs:
                        return
                    has[key] += 1; bet[key] += len(legs) * 100
                    won = False
                    for pr, od in legs.items():
                        if pr <= top3:
                            pay[key] += int(od * 100); won = True
                    if won:
                        cnt[key] += 1

                # B1 頭box ワイド（全頭ペア）
                if len(heads) >= 2:
                    wide_bet("B1 頭box ワイド",
                             [(heads[i], heads[j]) for i in range(len(heads))
                              for j in range(i + 1, len(heads))])
                # B4 人気L頭 × 人気薄L頭 ワイド1点
                if heads_by_pop and underdogs:
                    wide_bet("B4 人気L頭×薄L頭 ワイド", [(heads_by_pop[0], underdogs[0])])
                # B6 人気薄L頭 軸 → 全頭ワイド流し
                if underdogs:
                    u = underdogs[0]
                    wide_bet("B6 人気薄L頭×全頭 ワイド流し",
                             [(u, f) for f in heads if f != u])
                # B2/B3 複勝(オッズ無し・的中率のみ)
                if heads_by_pop:
                    has["B2 人気L頭 複勝(的中率のみ)"] += 1
                    if heads_by_pop[0] in top3:
                        cnt["B2 人気L頭 複勝(的中率のみ)"] += 1
                if underdogs:
                    has["B3 人気薄L頭 複勝(的中率のみ)"] += 1
                    if underdogs[0] in top3:
                        cnt["B3 人気薄L頭 複勝(的中率のみ)"] += 1
            print(f"  --- {wl} (n={n}, {n/days:.1f}R/日) ---")
            for k in ["B1 頭box ワイド", "B4 人気L頭×薄L頭 ワイド",
                      "B6 人気薄L頭×全頭 ワイド流し",
                      "B2 人気L頭 複勝(的中率のみ)", "B3 人気薄L頭 複勝(的中率のみ)"]:
                if has[k]:
                    roi = (pay[k] / bet[k]) if bet[k] else None
                    roi_s = f" ROI={roi:5.1%}" if roi is not None else ""
                    print(f"     {k:<28} 的中率={cnt[k]/has[k]:5.1%}{roi_s} (対象{has[k]})")
    eval_bets(upset)


if __name__ == "__main__":
    main()
