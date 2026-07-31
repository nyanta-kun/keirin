"""【C-7残り: 上がり順位 − 着順】の市場ミスプライシング検証（2026-07-30）。

`inputs/情報源候補_ABC分類と検証プロトコル.md` C-7 のうち、
既に本番採用済みの部分（`fh_rel_90` / `fh_best_rate_90`）を除いた
**未検証の残り1項目**だけを測定する。

## 仮説（メモ C-7）

「着順は悪いが上がりタイムは優秀」＝展開に恵まれなかっただけの選手を検出できれば、
市場が織り込みにくい（着順は誰でも見るが、上がりタイムとの乖離は見られていない）。

指標の定義（符号を明示する。メモの符号注記は逆だったので本スクリプトで確定させる）:

    unlucky = 前走の着順 − 前走の上がり順位

  例) 上がり1位（最速）なのに7着 → 7 - 1 = +6 → **正が大きいほど展開に恵まれなかった**
  例) 上がり7位（最遅）なのに1着 → 1 - 7 = -6 → 負が大きいほど展開に恵まれた

`unlucky` が正に大きい選手は「脚は使えていたのに着順が出なかった」＝次走で
見直せる候補であり、市場が着順に引きずられて過小評価していれば妙味になる。

## データ上の注意

- `wt_entries.final_half` は**上がり200mタイム（秒）**。小さいほど速い。充填率96.3%
- **`final_half = 0.0` の無効値が存在する**ため 8.0〜20.0 秒の範囲外を除外する
- 上がり順位は「そのレースで完走かつ有効な上がりタイムを持つ選手」の中で算出
- 前走は**直近の有効レース**（完走かつ有効な上がりを持つレース）を使う

## 測定（プロトコル準拠）

    比 = 実測3着内率 ÷ 市場の3着内含意確率（三連複35通りから周辺化・Σ=3.0 検算）

- ROI = 0.75 × 比。**ROI100%超には比 ≥ 1.333**
- **人気帯で層別**して交絡を確認する（プロトコル追加チェック5）。
  `unlucky` は前走着順と強く相関し、前走着順は人気に反映されるため、
  層別しないと「人気の再発見」と区別できない
- TRAIN/TEST 両窓で符号・効果量が一致することを要求

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_c_candidates_market_test import (  # noqa: E402
    MEAS_FROM, MEAS_TO, MIN_BOARD, ROI_BREAKEVEN, TAKEOUT_RETURN, TRAIN_TO,
    Acc, load_all, load_trio_odds,
)

FH_MIN, FH_MAX = 8.0, 20.0      # 有効な上がりタイムの範囲（0.0等の無効値を除外）
MIN_SEG = 1500
MIN_STRAT = 800


def valid_fh(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if FH_MIN <= f <= FH_MAX else None


def build_prev_agari(races, entries_by_race):
    """選手ごとに「直近の有効レース」の (着順, 上がり順位, 母数) を point-in-time で持つ。"""
    order = sorted(races, key=lambda rk: (races[rk]["date"], races[rk]["race_no"]))
    prev = {}                    # pid -> (finish_order, agari_rank, n_ranked)
    out = {}                     # (race_key, pid) -> 同上（このレース開始前の状態）
    n_valid_races = 0

    for rk in order:
        ents = entries_by_race.get(rk)
        if not ents:
            continue

        # このレース開始前の状態を確定
        for e in ents:
            p = prev.get(e["player_id"])
            if p is not None:
                out[(rk, e["player_id"])] = p

        # 結果でstateを更新: 完走かつ有効な上がりを持つ選手のみで上がり順位を作る
        ranked = []
        for e in ents:
            fo = e["finish_order"]
            fh = valid_fh(e["final_half"])
            if fo is not None and int(fo) >= 1 and fh is not None:
                ranked.append((fh, int(fo), e["player_id"]))
        if len(ranked) < 3:
            continue
        n_valid_races += 1
        ranked.sort()                                  # 上がりタイム昇順 = 速い順
        n = len(ranked)
        for idx, (_fh, fo, pid) in enumerate(ranked):
            prev[pid] = (fo, idx + 1, n)

    print(f"[prev] 有効レース {n_valid_races} / 前走情報を持つ (race,player) {len(out)}",
          flush=True)
    return out


def band_unlucky(u):
    if u is None:
        return None
    if u <= -3:
        return "前走 着順-上がり順位 ≤-3（展開に恵まれた）"
    if u <= -1:
        return "前走 着順-上がり順位 -2〜-1"
    if u == 0:
        return "前走 着順-上がり順位 0（一致）"
    if u <= 2:
        return "前走 着順-上がり順位 +1〜+2"
    return "前走 着順-上がり順位 ≥+3（展開に恵まれず＝見直し候補）"


def main():
    races, entries = load_all()
    prev = build_prev_agari(races, entries)

    targets = [rk for rk, m in races.items()
               if m["n_entries"] == 7 and MEAS_FROM <= m["date"] <= MEAS_TO]
    print(f"[meas] 対象レース(7車立て): {len(targets)}", flush=True)

    acc = defaultdict(lambda: defaultdict(Acc))
    strat = defaultdict(lambda: defaultdict(Acc))
    sum_check = []

    by_month = defaultdict(list)
    for rk in targets:
        by_month[races[rk]["date"][:7]].append(rk)

    for ym in sorted(by_month):
        rks = by_month[ym]
        boards = load_trio_odds(rks)
        for rk in rks:
            meta = races[rk]
            ents = entries.get(rk)
            board = boards.get(rk)
            if not ents or len(ents) != 7 or not board or len(board) < MIN_BOARD:
                continue
            if any(e["pred_top3_pct"] is None for e in ents):
                continue
            fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
                   if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
            if len(fin) < 3:
                continue
            fin.sort()
            tm = 0
            for _, fr in fin[:3]:
                tm |= 1 << fr

            mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if o > 0}
            tot = sum(mk_raw.values())
            if tot <= 0:
                continue
            frames = sorted(int(e["frame_no"]) for e in ents)
            marg = {fr: 0.0 for fr in frames}
            for m, v in mk_raw.items():
                p = v / tot
                for fr in frames:
                    if (m >> fr) & 1:
                        marg[fr] += p
            if len(sum_check) < 3000:
                sum_check.append(sum(marg.values()))

            w = "TRAIN" if meta["date"] <= TRAIN_TO else "TEST"

            for e in ents:
                fr = int(e["frame_no"])
                mp = marg[fr]
                if mp <= 0:
                    continue
                y = 1.0 if (tm >> fr) & 1 else 0.0
                p = prev.get((rk, e["player_id"]))
                if p is None:
                    acc["C7_unlucky"][(w, "前走情報なし")].add(y, mp)
                    continue
                fo, ar, n = p
                u = fo - ar

                b = band_unlucky(u)
                acc["C7_unlucky"][(w, b)].add(y, mp)
                acc["C7_prev_agari_rank"][
                    (w, f"前走上がり{ar}位" if ar <= 3 else "前走上がり4位以下")].add(y, mp)
                acc["C7_prev_finish"][
                    (w, f"前走{fo}着" if fo <= 3 else "前走4着以下")].add(y, mp)
                # 「前走上がり最速なのに着外」＝仮説の最も純粋な形
                if ar == 1:
                    acc["C7_best_agari"][
                        (w, "前走上がり最速×3着内" if fo <= 3 else
                         "前走上がり最速×着外（★見直し候補）")].add(y, mp)

                band = ("人気帯1(低)" if mp < 0.25 else "人気帯2" if mp < 0.40 else
                        "人気帯3" if mp < 0.55 else "人気帯4(高)")
                strat[("C7_unlucky", band)][(w, b)].add(y, mp)

        print(f"  {ym}: {len(rks)}R", flush=True)

    print("\n" + "=" * 112)
    print("[検算] Σ_i market_P(3着内) = 3.0（プロトコル項目9）: "
          f"平均 {statistics.mean(sum_check):.4f} (n={len(sum_check)})")
    print("=" * 112)

    TITLES = [
        ("C7_unlucky", "★C-7本命: 前走の 着順 − 上がり順位（正＝展開に恵まれず）"),
        ("C7_best_agari", "C-7 前走で上がり最速だった選手（着順別）"),
        ("C7_prev_agari_rank", "【参考】前走の上がり順位そのもの"),
        ("C7_prev_finish", "【参考】前走の着順そのもの"),
    ]
    for dim, title in TITLES:
        if dim not in acc:
            continue
        print("\n" + "-" * 112)
        print(title)
        print("-" * 112)
        print(f"{'セグメント':<42}{'窓':<6}{'車数':>9}{'実測%':>8}{'市場%':>8}"
              f"{'比':>8}{'t値':>8}{'→ROI%':>9}{'判定':>16}")
        for seg in sorted({k[1] for k in acc[dim]}):
            printed = False
            for w in ("TRAIN", "TEST"):
                a = acc[dim].get((w, seg))
                if not a or a.n < MIN_SEG:
                    continue
                r = a.report()
                v = ""
                if r["ratio"] >= ROI_BREAKEVEN and r["t"] > 3:
                    v = "★ROI100%超"
                elif r["ratio"] > 1.0 and r["t"] > 3:
                    v = "市場が過小評価"
                elif r["ratio"] < 1.0 and r["t"] < -3:
                    v = "除外候補"
                print(f"{seg if not printed else '':<42}{w:<6}{r['n']:>9}{r['act']:>8.2f}"
                      f"{r['mkt']:>8.2f}{r['ratio']:>8.3f}{r['t']:>+8.2f}{r['roi']:>9.1f}{v:>16}")
                printed = True
            if printed:
                print()

    print("\n" + "-" * 112)
    print("【交絡確認】人気帯で層別（前走着順は人気に反映されるため必須）")
    print("-" * 112)
    for band in ("人気帯1(低)", "人気帯2", "人気帯3", "人気帯4(高)"):
        key = ("C7_unlucky", band)
        if key not in strat:
            continue
        print(f"\n  [{band}]")
        for seg in sorted({k[1] for k in strat[key]}):
            for w in ("TRAIN", "TEST"):
                a = strat[key].get((w, seg))
                if not a or a.n < MIN_STRAT:
                    continue
                r = a.report()
                print(f"    {seg:<42}{w:<6}n={r['n']:>7} 実測{r['act']:>6.2f}% "
                      f"市場{r['mkt']:>6.2f}% 比{r['ratio']:>7.3f} t{r['t']:>+7.2f}")

    print("\n" + "=" * 112)
    print(f"【結論】両窓で 比 ≥ {ROI_BREAKEVEN:.3f} かつ TEST t>3")
    print("=" * 112)
    hits, excl = [], []
    for dim in acc:
        for seg in {k[1] for k in acc[dim]}:
            a, b = acc[dim].get(("TRAIN", seg)), acc[dim].get(("TEST", seg))
            if not a or not b or a.n < MIN_SEG or b.n < MIN_SEG:
                continue
            ra, rb = a.report(), b.report()
            if ra["ratio"] >= ROI_BREAKEVEN and rb["ratio"] >= ROI_BREAKEVEN and rb["t"] > 3:
                hits.append((dim, seg, ra, rb))
            if ra["ratio"] < 1.0 and rb["ratio"] < 1.0 and ra["t"] < -3 and rb["t"] < -3:
                excl.append((dim, seg, ra, rb))
    print("  " + ("該当なし。" if not hits else ""))
    for dim, seg, ra, rb in sorted(hits, key=lambda x: -x[3]["ratio"]):
        print(f"  ★[{dim}] {seg}: TRAIN {ra['ratio']:.3f} / TEST {rb['ratio']:.3f} "
              f"(t={rb['t']:+.2f}) ROI={rb['roi']:.1f}%")
    print("\n【除外ルール候補】両窓で 比 < 1.0 かつ |t|>3")
    print("  " + ("該当なし。" if not excl else ""))
    for dim, seg, ra, rb in sorted(excl, key=lambda x: x[3]["ratio"]):
        print(f"  ▼[{dim}] {seg}: TRAIN {ra['ratio']:.3f}(t{ra['t']:+.1f}) / "
              f"TEST {rb['ratio']:.3f}(t{rb['t']:+.1f})")


if __name__ == "__main__":
    main()
