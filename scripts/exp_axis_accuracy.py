"""【読み取り専用】軸1・軸2の的中精度の実測（2026-08-04）。

ユーザー質問「現在のモデルにおいて、1軸目・2軸目の的中精度はどれくらいか」への
回答を作るためのスクリプト。ベース予想精度の現在地を把握するのが目的で、
特定の戦略の採否を判断するものではない。

測定内容:
  ① 軸1 / 軸2 それぞれの 1着率・2着内率・3着内率
  ② 比較対象: WT公式印◎/◯、モデルの pred_prob 最上位/2位、pred_win 最上位
  ③ ペア指標: 軸2車がともに3着内（＝三連複総流しの的中率）／1-2着独占
  ④ pred_prob のキャリブレーション（予測確率帯 × 実測3着内率）
  ⑤ 月次推移（精度が劣化しているかの確認）
  ⑥ WT印との重なり（overlap）別の層別

honest: 月次凍結vintageモデルで生成したキャッシュ（scripts/exp_7c_cache.py）を
使うため、各レースは「そのレースより前のデータだけで学習したモデル」で
スコアされている。DB書き込みなし。

使い方:
    python scripts/exp_axis_accuracy.py data/exp_7c_cache
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def nth_by(c: dict, key: str, n: int) -> int | None:
    """レース内で key（top3_probs / win_probs）の n 番目に高い車番。"""
    r = sorted(c[key], key=lambda f: -c[key][f])
    return r[n] if len(r) > n else None


def hit_stats(cands: list[dict], pick) -> dict:
    """pick(c) が返す1車について 1着/2着内/3着内 の率を測る。"""
    n = win = top2 = top3 = 0
    for c in cands:
        f = pick(c)
        if f is None:
            continue
        n += 1
        o = c["order3"]
        if f == o[0]:
            win += 1
        if f in o[:2]:
            top2 += 1
        if f in o:
            top3 += 1
    return {"n": n,
            "win": 100.0 * win / n if n else 0.0,
            "top2": 100.0 * top2 / n if n else 0.0,
            "top3": 100.0 * top3 / n if n else 0.0}


def pair_stats(cands: list[dict]) -> dict:
    """軸2車のペアとしての精度。"""
    n = both3 = any3 = ex12 = one_win = 0
    for c in cands:
        n += 1
        a = {c["axis1"], c["axis2"]}
        o = c["order3"]
        n_in = len(a & set(o))
        if n_in == 2:
            both3 += 1
        if n_in >= 1:
            any3 += 1
        if a == set(o[:2]):
            ex12 += 1
        if o[0] in a:
            one_win += 1
    return {"n": n,
            "both3": 100.0 * both3 / n if n else 0.0,
            "any3": 100.0 * any3 / n if n else 0.0,
            "ex12": 100.0 * ex12 / n if n else 0.0,
            "win": 100.0 * one_win / n if n else 0.0}


def prow(label: str, s: dict) -> str:
    return (f"{label:32} n={s['n']:6d}  1着 {s['win']:5.1f}%  "
            f"2着内 {s['top2']:5.1f}%  3着内 {s['top3']:5.1f}%")


def main() -> None:
    rows = load(Path(sys.argv[1]))
    if not rows:
        print("キャッシュが空です")
        return
    days = sorted({c["race_date"] for c in rows})
    print(f"母集団: 7車立て・軸選定成功レース {len(rows)}件 / {len(days)}日 "
          f"({days[0]}〜{days[-1]})")
    print("※ honest（月次凍結vintageモデル）・出走取消等で3着まで確定した"
          "レースのみ\n")

    # ---------------------------------------------------------------- 単体精度
    print("【① 軸1・軸2 の単体精度】")
    print(prow("軸1（axis1）", hit_stats(rows, lambda c: c["axis1"])))
    print(prow("軸2（axis2）", hit_stats(rows, lambda c: c["axis2"])))
    print()

    print("【② 比較対象】")
    print(prow("モデル pred_prob 1位", hit_stats(rows, lambda c: nth_by(c, "top3_probs", 0))))
    print(prow("モデル pred_prob 2位", hit_stats(rows, lambda c: nth_by(c, "top3_probs", 1))))
    print(prow("モデル pred_prob 3位", hit_stats(rows, lambda c: nth_by(c, "top3_probs", 2))))
    print(prow("モデル pred_win 1位", hit_stats(rows, lambda c: nth_by(c, "win_probs", 0))))
    print(prow("WT公式 ◎(honmei)", hit_stats(rows, lambda c: c.get("wt_honmei"))))
    print(prow("WT公式 ◯(taikou)", hit_stats(rows, lambda c: c.get("wt_taikou"))))
    print(prow("WT公式 △(ana)", hit_stats(rows, lambda c: c.get("wt_ana"))))
    print()

    # ---------------------------------------------------------------- ペア精度
    print("【③ 軸2車ペアの精度】")
    hdr = f"{'母集団':28} {'n':>6} {'両方3着内':>9} {'どちらか':>9} {'1-2着独占':>9} {'どちらか1着':>11}"
    print(hdr)

    def prow2(label: str, s: dict) -> str:
        return (f"{label:28} {s['n']:6d} {s['both3']:8.1f}% {s['any3']:8.1f}% "
                f"{s['ex12']:8.1f}% {s['win']:10.1f}%")

    print(prow2("全体", pair_stats(rows)))
    for ov in (0, 1, 2):
        sub = [c for c in rows if c["wt_overlap_n"] == ov]
        if sub:
            print(prow2(f"  WT◎◯との重なり={ov}", pair_stats(sub)))
    none_ov = [c for c in rows if c["wt_overlap_n"] is None]
    if none_ov:
        print(prow2("  WT印欠損", pair_stats(none_ov)))
    print()

    # ---------------------------------------------------------------- 較正
    print("【④ pred_prob のキャリブレーション（出走全車・7車×レース）】")
    print(f"{'予測確率帯':16} {'n':>8} {'予測平均':>9} {'実測3着内率':>11} {'乖離':>8}")
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c in rows:
        o = set(c["order3"])
        for f, p in c["top3_probs"].items():
            buckets[min(int(p * 10), 9)].append((p, f in o))
    for b in sorted(buckets):
        vals = buckets[b]
        pm = sum(p for p, _ in vals) / len(vals)
        am = sum(1 for _, h in vals if h) / len(vals)
        print(f"{b*10:3d}〜{b*10+10:3d}%      {len(vals):8d} {100*pm:8.1f}% "
              f"{100*am:10.1f}% {100*(am-pm):+7.1f}pt")
    print()

    # ---------------------------------------------------------------- 月次
    print("【⑤ 月次推移（劣化の有無）】")
    by_ym: dict[str, list[dict]] = defaultdict(list)
    for c in rows:
        by_ym[str(c["race_date"])[:7]].append(c)
    print(f"{'月':9} {'n':>6} {'軸1 3着内':>10} {'軸2 3着内':>10} {'両方3着内':>10} "
          f"{'軸1 1着':>9}")
    for ym in sorted(by_ym):
        sub = by_ym[ym]
        a1 = hit_stats(sub, lambda c: c["axis1"])
        a2 = hit_stats(sub, lambda c: c["axis2"])
        pr = pair_stats(sub)
        print(f"{ym:9} {len(sub):6d} {a1['top3']:9.1f}% {a2['top3']:9.1f}% "
              f"{pr['both3']:9.1f}% {a1['win']:8.1f}%")
    print()

    # ---------------------------------------------------------------- 現行ランク別
    print("【⑥ 現行ランクの母集団での軸精度】")
    from src.strategy_wt import RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX
    ov01 = [c for c in rows if c["wt_overlap_n"] in (0, 1)]
    seg = {
        "7S（2ゲート合格）": [c for c in ov01 if c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX
                              and c["entropy"] <= RANK_7S_ENTROPY_MAX],
        "7A（1ゲート不合格）": [c for c in ov01
                                if (c["axis_sum"] > RANK_7S_AXIS_SUM_MAX)
                                != (c["entropy"] > RANK_7S_ENTROPY_MAX)],
        "overlap2（◎◯一致）": [c for c in rows if c["wt_overlap_n"] == 2],
    }
    print(hdr)
    for name, sub in seg.items():
        if sub:
            print(prow2(name, pair_stats(sub)))
    print()
    for name, sub in seg.items():
        if sub:
            print(f"-- {name} --")
            print(prow("  軸1", hit_stats(sub, lambda c: c["axis1"])))
            print(prow("  軸2", hit_stats(sub, lambda c: c["axis2"])))


if __name__ == "__main__":
    main()
