"""7S/7A: 的中率 × 平均配当 × ROI のフロンティアと、軸選択の精度（2026-08-06・探索）。

`exp_7s7a_payout_structure.py` が作ったデータセットを使う（先に実行しておくこと）。

測るもの:
  【5】候補セルの頑健性（四半期別 ROI・裾依存・既存ゲートとの相関）
  【6】軸選択の精度分解（軸1/軸2/両方の3着内率）と、代替の軸2規則
  【7】配当帯を狙う方向（高配当セグメント）の実力

⚠️ オッズは wt_odds（最終）。DB 書き込みなし。掃引窓/確認窓を必ず分けて出す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(REPO))

DATASET = REPO / "data" / "exp_cache" / "payout_structure_7car.pkl"
CONFIRM_END = "2025-06-30"


def load():
    from scripts.exp_7s7a_payout_structure import rank_of  # noqa
    df = pd.read_pickle(DATASET)
    df["rank"] = [rank_of(r) for r in df.itertuples(index=False)]
    df["win"] = df["race_date"].apply(lambda d: "確認" if d <= CONFIRM_END else "掃引")
    df["q"] = pd.PeriodIndex(pd.to_datetime(df["race_date"]), freq="Q").astype(str)
    return df


def line(label, s, w=34):
    """掃引/確認を並べた1行。"""
    txt = f"  {label:<{w}}"
    for win in ("掃引", "確認"):
        t = s[s["win"] == win]
        if len(t) < 20:
            txt += f"{'—':>7}{'—':>7}{'—':>8}{'—':>7}{'—':>7}"
            continue
        h = t[t.hit == 1]
        txt += (f"{len(t):>7}{100*t.hit.mean():>6.1f}%{100*t.ret.sum()/t.bet.sum():>7.1f}%"
                f"{(h.res_odds.mean() if len(h) else 0):>7.1f}{(100*(h.res_odds>=10).mean() if len(h) else 0):>6.0f}%")
    print(txt)


HDR = f"  {'':<34}" + "".join([f"{'n':>7}{'的中':>7}{'ROI':>8}{'平均':>7}{'≥10':>7}" for _ in range(2)])


def sec5(df):
    print("=" * 104)
    print("【5】候補セルの頑健性 — 掃引窓で光ったものを確認窓・四半期・裾で崩す")
    a = df[df["rank"] == "7A"]
    print("\n  ── 左=掃引窓(2025-07〜2026-08) 右=確認窓(2024-07〜2025-06)")
    print(HDR)
    line("7A 基準", a)
    for q, lab in [(0.75, "上位25%"), (0.5, "上位50%")]:
        thr = a[a["win"] == "掃引"]["line_p_hhi"].quantile(q)
        line(f"7A ∧ line_p_hhi>={thr:.3f} ({lab})", a[a.line_p_hhi >= thr])
    thr_std = a[a["win"] == "掃引"]["p3_std"].quantile(0.75)
    line(f"7A ∧ p3_std>={thr_std:.3f} (上位25%)", a[a.p3_std >= thr_std])
    thr_lg = a[a["win"] == "掃引"]["line_p_g12"].quantile(0.75)
    line(f"7A ∧ line_p_g12>={thr_lg:.3f} (上位25%)", a[a.line_p_g12 >= thr_lg])

    print("\n  ── 四半期別 ROI（掃引窓で光った 7A∧line_p_hhi上位25% を分解）")
    thr = a[a["win"] == "掃引"]["line_p_hhi"].quantile(0.75)
    s = a[a.line_p_hhi >= thr]
    print(f"     {'四半期':<10}{'n':>6}{'的中':>8}{'ROI':>8}   基準ROI(同四半期の7A全体)")
    for q in sorted(s["q"].unique()):
        t = s[s["q"] == q]
        b = a[a["q"] == q]
        if len(t) < 15:
            continue
        print(f"     {q:<10}{len(t):>6}{100*t.hit.mean():>7.1f}%{100*t.ret.sum()/t.bet.sum():>7.1f}%"
              f"      {100*b.ret.sum()/b.bet.sum():>6.1f}%")

    print("\n  ── 裾依存（回収額の上位n件を除いた ROI）")
    for lab, s in [("7A 基準", a), (f"7A ∧ line_p_hhi>={thr:.3f}", a[a.line_p_hhi >= thr])]:
        for win in ("掃引", "確認"):
            t = s[s["win"] == win]
            r = np.sort(t.ret.values)[::-1]
            bet = t.bet.sum()
            base = 100 * r.sum() / bet
            d5 = 100 * r[5:].sum() / bet
            d10 = 100 * r[10:].sum() / bet
            top5 = 100 * r[:5].sum() / r.sum()
            print(f"     {lab:<28}{win}  ROI {base:5.1f}% → 上位5件除 {d5:5.1f}% "
                  f"→ 上位10件除 {d10:5.1f}%（上位5件が回収の {top5:4.1f}%）")

    print("\n  ── 既存ゲートとの相関（掃引窓・7A内）")
    sw = a[a["win"] == "掃引"]
    for f in ("line_p_hhi", "p3_std", "line_p_g12"):
        c1 = sw[[f, "asum"]].corr().iloc[0, 1]
        c2 = sw[[f, "ent"]].corr().iloc[0, 1]
        print(f"     {f:<14} vs asum {c1:+.3f}   vs entropy {c2:+.3f}")


def sec6(df):
    print("\n" + "=" * 104)
    print("【6】軸選択の精度 — 二軸総流しの的中は『軸2車がともに3着内』と同値")
    print("     ※ 相手は残り5車すべてなので、相手選択は的中に一切関与しない")
    for rk in ("7S", "7A", "7SS", "空白E"):
        s = df[df["rank"] == rk]
        for win in ("掃引", "確認"):
            t = s[s["win"] == win]
            if t.empty:
                continue
            a1in = np.mean([r.a1 in _top3(r) for r in t.itertuples(index=False)])
            a2in = np.mean([r.a2 in _top3(r) for r in t.itertuples(index=False)])
            both = t.hit.mean()
            print(f"  {rk:<6}{win}  軸1の3着内 {100*a1in:5.1f}%  軸2の3着内 {100*a2in:5.1f}%  "
                  f"両方 {100*both:5.1f}%  （独立仮定なら {100*a1in*a2in:5.1f}%）")


_TOP3_CACHE: dict[str, set] = {}


def _top3(r):
    return _TOP3_CACHE[r.race_key]


def sec7(df):
    print("\n" + "=" * 104)
    print("【7】ユーザー提案の方向（高配当レースを選ぶ）を素直に測る")
    print("     ※『結果が高配当』は事前に分からないので、事前に読める代理指標で選ぶ")
    pool = df[df["rank"].isin(["7S", "7A"])]
    print(HDR)
    line("7S+7A 基準", pool)
    sw = pool[pool["win"] == "掃引"]
    for f, direction in [("ent", "hi"), ("p3_std", "lo"), ("asum", "lo"),
                         ("line_p_hhi", "lo"), ("p3_top2", "lo")]:
        for q in (0.25, 0.5):
            if direction == "lo":
                thr = sw[f].quantile(q)
                s = pool[pool[f] <= thr]
                lab = f"{f} <= {thr:.3f}（下位{int(q*100)}%）"
            else:
                thr = sw[f].quantile(1 - q)
                s = pool[pool[f] >= thr]
                lab = f"{f} >= {thr:.3f}（上位{int(q*100)}%）"
            line(lab, s)

    print("\n  ── 参考: 現状どのランクにも入らない『空白E』（entropy不合格 ∧ 軸が別ライン）")
    print(HDR)
    line("空白E 全体", df[df["rank"] == "空白E"])
    print("\n  ── 配当と的中の交換レート（7S+7A を p3_std 十分位で分解・掃引窓）")
    sw = pool[pool["win"] == "掃引"]
    sw = sw.assign(d=pd.qcut(sw["p3_std"], 10, labels=False, duplicates="drop"))
    print(f"     {'十分位':<8}{'n':>6}{'的中':>8}{'平均配当':>10}{'ROI':>8}{'≥10倍率':>9}")
    for d in sorted(sw["d"].dropna().unique()):
        t = sw[sw["d"] == d]
        h = t[t.hit == 1]
        print(f"     D{int(d)+1:<7}{len(t):>6}{100*t.hit.mean():>7.1f}%{h.res_odds.mean():>9.1f}倍"
              f"{100*t.ret.sum()/t.bet.sum():>7.1f}%{100*(h.res_odds>=10).mean():>8.1f}%")


def main():
    df = load()
    # top3 を復元（データセットには入れていないので odds 由来では作れない → 再取得）
    import os
    import psycopg2
    keys = df["race_key"].tolist()
    with psycopg2.connect(os.environ["KEIRIN_DB_URL"]) as c:
        cur = c.cursor()
        cur.execute("SELECT race_key, frame_no, finish_order FROM keirin.wt_entries "
                    "WHERE race_key = ANY(%s) AND finish_order BETWEEN 1 AND 3", (keys,))
        for rk, fn, fo in cur.fetchall():
            _TOP3_CACHE.setdefault(rk, set()).add(int(fn))
    sec5(df)
    sec6(df)
    sec7(df)


if __name__ == "__main__":
    main()
