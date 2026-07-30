"""7車立てレースを「単勝順位1-2位差(g12)/2-3位差(g23)/3-4位差(g34)」の
最大ギャップ位置でパターン分類し、パターン別に6種類の三連複軸2車構成の
honest ROI・的中率をTRAIN(2024-01-01〜2025-12-31)/TEST(2026-01-01〜2026-07-30)
で比較するマトリクス分析。

DB格納値 (`wt_entries.pred_win_pct` / `pred_top3_pct`) は2026-07-30に月次凍結
vintageモデルで再計算済み・書き込み保護済みのためそのまま使う（モデル再ロード
不要・[[keirin_s7_foundational_rethink_2026_07_29]] 系の高速版）。

読み取り専用。DBへの書き込みは一切行わない。

## パターン分類の定義
各レースで7頭を pred_win_pct 降順ソートして w1..w7 (frame_no)、
pred_top3_pct 降順で t1..t7 とする。
  g12 = win_pct(w1) - win_pct(w2)
  g23 = win_pct(w2) - win_pct(w3)
  g34 = win_pct(w3) - win_pct(w4)
  max_gap = max(g12, g23, g34)

1. まず「拮抗」判定: max_gap が **TRAIN期間の max_gap 分布の下位25%点
   (percentile 25, TRAINのみで算出しTESTにも同じ閾値を固定適用 = リーク防止)**
   未満なら「全体拮抗」に分類する。
2. 拮抗でない残りのレースは、g12/g23/g34 のうち最大のものの位置で分類:
   - g12が最大 → 「1車突出」(1位が2位を大きく引き離す)
   - g23が最大 → 「2車突出」(1-2位が拮抗しつつ2-3位で差)
   - g34が最大 → 「3車突出」(1-3位が拮抗しつつ3-4位で差)

## 軸構成6種
1. w1+w2 (単勝上位2頭)
2. w1+w3 (1位+3位、2車突出時の共倒れ回避狙い)
3. w1 + t2以降(t2,t3,...,t7の順)で最初に現れる非w1車 (仕様書の指定通りt1は除外して
   t2から探索する)
4. t1+t2 (複勝上位2頭)
5. w1 + (◎◯のうちw1でない方) — w1が◎でも◯でもない場合は非適用(N/A)としてスキップ
6. (◎◯のうちpred_win_pct高い方) + 非マーク5車のうちpred_top3_pct最上位
   (既存S7新設計と同一ロジック・比較用ベースライン)

## 買い目・投資額
各軸構成につき、軸2車+残り5車のいずれか1車＝5点、1点100円＝計500円。
combo(3頭の frozenset)が wt_odds(trio) に存在する分だけ bet を計上する
(存在しないcomboは投資額に含めない。的中判定は着順から独立に決まる)。

Usage:
    cd /Users/ysuzuki/GitHub/keirin
    .venv/bin/python3 scripts/exp_pattern_axis_roi_matrix.py
"""
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
FULL_FROM, FULL_TO = TRAIN_FROM, TEST_TO

STAKE = 100
BOX_N = 5  # 残り5車のいずれか1車

PATTERNS = ["1車突出", "2車突出", "3車突出", "全体拮抗"]
AXIS_NAMES = [
    "1:w1+w2",
    "2:w1+w3",
    "3:w1+t2以降非w1",
    "4:t1+t2",
    "5:w1+markの他方",
    "6:baseline(mark高+非mark top3)",
]


def _split_combo(combo: str):
    try:
        return frozenset(int(x) for x in re.split(r"[-=→]", str(combo)))
    except ValueError:
        return None


def load_races_and_entries():
    """n_entries=7 & cancel=0 のレースについて、pred_win_pct/pred_top3_pct が
    7頭全て非NULLのレースのみ有効レースとして返す。"""
    with get_connection() as c:
        print("[load] wt_races ...", flush=True)
        race_rows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (FULL_FROM, FULL_TO),
        ).fetchall()
        race_date = {r["race_key"]: str(r["race_date"]) for r in race_rows}
        print(f"[load]   races(n_entries=7,cancel=0): {len(race_date)}", flush=True)

        print("[load] wt_entries ...", flush=True)
        ent_rows = c.execute(
            "SELECT we.race_key, we.frame_no, we.pred_win_pct, we.pred_top3_pct, "
            "we.prediction_mark, we.finish_order "
            "FROM wt_entries we JOIN wt_races wr ON we.race_key = wr.race_key "
            "WHERE wr.n_entries = 7 AND wr.cancel = 0 AND wr.race_date BETWEEN ? AND ?",
            (FULL_FROM, FULL_TO),
        ).fetchall()
        print(f"[load]   entry rows: {len(ent_rows)}", flush=True)

    by_race = defaultdict(dict)
    for r in ent_rows:
        by_race[r["race_key"]][int(r["frame_no"])] = {
            "win": r["pred_win_pct"],
            "top3": r["pred_top3_pct"],
            "mark": r["prediction_mark"],
            "fo": r["finish_order"],
        }

    valid = {}
    for rk, frames in by_race.items():
        if len(frames) != 7:
            continue
        if any(v["win"] is None or v["top3"] is None for v in frames.values()):
            continue
        valid[rk] = {
            "race_date": race_date[rk],
            "frames": {f: {"win": float(v["win"]), "top3": float(v["top3"]),
                           "mark": (int(v["mark"]) if v["mark"] is not None else None),
                           "fo": (int(v["fo"]) if v["fo"] is not None else None)}
                       for f, v in frames.items()},
        }
    print(f"[load]   valid races (7頭ともpred値あり): {len(valid)}", flush=True)
    return valid


def load_trio_boards(race_keys):
    trio = defaultdict(dict)
    with get_connection() as c:
        keys = list(race_keys)
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                parts = _split_combo(comb)
                if parts is not None and len(parts) == 3:
                    trio[rk][parts] = fv
            if (i // 900) % 20 == 0:
                print(f"[load]   trio boards progress: {i + len(chunk)}/{len(keys)}", flush=True)
    return trio


def build_axis_pairs(frames: dict):
    """frames: {frame_no: {win, top3, mark, fo}} -> dict axis_name -> (a1,a2) or None"""
    w_order = sorted(frames.keys(), key=lambda f: (-frames[f]["win"], f))
    t_order = sorted(frames.keys(), key=lambda f: (-frames[f]["top3"], f))
    w1, w2, w3 = w_order[0], w_order[1], w_order[2]

    axes = {}
    axes["1:w1+w2"] = (w1, w2)
    axes["2:w1+w3"] = (w1, w3)

    ac3_partner = next((f for f in t_order[1:] if f != w1), None)
    axes["3:w1+t2以降非w1"] = (w1, ac3_partner) if ac3_partner is not None else None

    axes["4:t1+t2"] = (t_order[0], t_order[1])

    honmei = next((f for f, v in frames.items() if v["mark"] == 1), None)
    taikou = next((f for f, v in frames.items() if v["mark"] == 2), None)

    if honmei is not None and taikou is not None:
        if w1 == honmei:
            other = taikou
        elif w1 == taikou:
            other = honmei
        else:
            other = None
        axes["5:w1+markの他方"] = (w1, other) if other is not None else None

        axis1_6 = honmei if frames[honmei]["win"] >= frames[taikou]["win"] else taikou
        others6 = [f for f in frames if f not in (honmei, taikou)]
        axis2_6 = max(others6, key=lambda f: frames[f]["top3"]) if len(others6) == 5 else None
        axes["6:baseline(mark高+非mark top3)"] = (
            (axis1_6, axis2_6) if axis2_6 is not None and axis1_6 != axis2_6 else None
        )
    else:
        axes["5:w1+markの他方"] = None
        axes["6:baseline(mark高+非mark top3)"] = None

    gaps = {
        "g12": frames[w1]["win"] - frames[w2]["win"],
        "g23": frames[w2]["win"] - frames[w3]["win"],
        "g34": frames[w3]["win"] - frames[w_order[3]]["win"],
    }
    return axes, gaps, w_order


def actual_top3(frames: dict):
    fin = sorted((v["fo"], f) for f, v in frames.items() if v["fo"] is not None and v["fo"] >= 1)
    if len(fin) < 3:
        return None
    return frozenset(f for _, f in fin[:3])


def evaluate_axis(frames, axis_pair, trio_board, act_top3):
    if axis_pair is None:
        return None
    a1, a2 = axis_pair
    if a1 == a2:
        return None
    box = sorted(set(frames.keys()) - {a1, a2})
    if len(box) != BOX_N:
        return None
    combo_odds = {}
    for x in box:
        key = frozenset({a1, a2, x})
        if key in trio_board:
            combo_odds[key] = trio_board[key]
    if not combo_odds:
        return None
    hit = act_top3 in combo_odds
    odds = combo_odds.get(act_top3, 0) if hit else 0
    pay = int(odds * STAKE) if hit else 0
    bet = len(combo_odds) * STAKE
    return {"hit": int(hit), "bet": bet, "payout": pay, "odds": odds if hit else None}


def main():
    valid = load_races_and_entries()
    trio = load_trio_boards(valid.keys())

    # --- 各レースのパターン用ギャップ・軸構成・結果を計算 ---
    records = []  # each: dict(race_key, race_date, period, gaps, axes_results{name: eval or None})
    n_no_result = 0
    n_no_trio_at_all = 0
    for rk, rec in valid.items():
        frames = rec["frames"]
        axes, gaps, w_order = build_axis_pairs(frames)
        act = actual_top3(frames)
        if act is None:
            n_no_result += 1
            continue
        board = trio.get(rk)
        if not board:
            n_no_trio_at_all += 1
            continue
        axis_eval = {}
        for name in AXIS_NAMES:
            axis_eval[name] = evaluate_axis(frames, axes.get(name), board, act)
        records.append({
            "race_key": rk,
            "race_date": rec["race_date"],
            "gaps": gaps,
            "axis_eval": axis_eval,
        })

    print(f"\n[main] 有効レース(結果あり&trioデータあり): {len(records)}"
          f" (着順不明で除外: {n_no_result} / trioデータ皆無で除外: {n_no_trio_at_all})")

    for r in records:
        r["date"] = r["race_date"]
    train_recs = [r for r in records if TRAIN_FROM <= r["date"] <= TRAIN_TO]
    test_recs = [r for r in records if TEST_FROM <= r["date"] <= TEST_TO]
    print(f"TRAIN: {len(train_recs)}件 / TEST: {len(test_recs)}件")

    # --- 拮抗閾値をTRAINのmax_gap分布のp25で決定し、TEST含め固定適用 ---
    train_max_gaps = sorted(max(r["gaps"]["g12"], r["gaps"]["g23"], r["gaps"]["g34"])
                             for r in train_recs)

    def percentile(sorted_vals, q):
        if not sorted_vals:
            return 0.0
        idx = (len(sorted_vals) - 1) * q / 100.0
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    congestion_th = percentile(train_max_gaps, 25)
    print(f"\n拮抗閾値(TRAIN max_gap p25, 固定してTESTにも適用): {congestion_th:.3f}pt")

    def classify(gaps):
        g12, g23, g34 = gaps["g12"], gaps["g23"], gaps["g34"]
        mg = max(g12, g23, g34)
        if mg < congestion_th:
            return "全体拮抗"
        if g12 >= g23 and g12 >= g34:
            return "1車突出"
        if g23 >= g12 and g23 >= g34:
            return "2車突出"
        return "3車突出"

    for r in records:
        r["pattern"] = classify(r["gaps"])

    # パターン分布
    print("\nパターン分布 (TRAIN / TEST):")
    for p in PATTERNS:
        ntr = sum(1 for r in train_recs if r["pattern"] == p)
        nte = sum(1 for r in test_recs if r["pattern"] == p)
        print(f"  {p}: TRAIN {ntr}件 ({ntr/len(train_recs)*100:.1f}%) / "
              f"TEST {nte}件 ({nte/len(test_recs)*100:.1f}%)")

    def summarize(evals):
        evals = [e for e in evals if e is not None]
        n = len(evals)
        if n == 0:
            return dict(n=0, hits=0, hitrate=0.0, bet=0, pay=0, roi=0.0)
        hits = sum(e["hit"] for e in evals)
        bet = sum(e["bet"] for e in evals)
        pay = sum(e["payout"] for e in evals)
        return dict(n=n, hits=hits, hitrate=hits / n * 100, bet=bet, pay=pay,
                    roi=(pay / bet * 100 if bet else 0.0))

    def cell(recs, pattern, axis_name):
        if pattern is not None:
            recs = [r for r in recs if r["pattern"] == pattern]
        return summarize([r["axis_eval"][axis_name] for r in recs])

    # --- 日数(母集団の稼働日数) ---
    train_days = len(set(r["date"] for r in train_recs))
    test_days = len(set(r["date"] for r in test_recs))
    print(f"\n稼働日数: TRAIN {train_days}日 / TEST {test_days}日")

    # --- マトリクス出力 ---
    print("\n" + "=" * 100)
    print("パターン × 軸構成 ROIマトリクス  [表記: TRAIN(n,hit%,ROI%) / TEST(n,hit%,ROI%)]")
    print("=" * 100)
    rows_for_recommend = []
    for pattern in PATTERNS + [None]:
        label = pattern if pattern is not None else "全体(パターン混合)"
        print(f"\n--- {label} ---")
        for axis_name in AXIS_NAMES:
            tr = cell(train_recs, pattern, axis_name)
            te = cell(test_recs, pattern, axis_name)
            tr_day = tr["n"] / train_days if train_days else 0
            te_day = te["n"] / test_days if test_days else 0
            flag = ""
            if tr["n"] >= 30 and te["n"] >= 30 and tr["roi"] > 100 and tr["hitrate"] > 30 \
                    and te["roi"] > 100 and te["hitrate"] > 30:
                flag = "  ★★TRAIN/TEST両方でROI100%超&的中30%超"
            elif tr["n"] < 30 or te["n"] < 30:
                flag = "  (n<30: 判断不能)"
            elif tr["n"] < 100 or te["n"] < 100:
                flag = "  (n<100: 要注意)"
            print(f"  {axis_name:32s} "
                  f"TRAIN n={tr['n']:5d}({tr_day:4.1f}/日) hit={tr['hitrate']:5.1f}% ROI={tr['roi']:6.1f}%  |  "
                  f"TEST n={te['n']:5d}({te_day:4.1f}/日) hit={te['hitrate']:5.1f}% ROI={te['roi']:6.1f}%{flag}")
            rows_for_recommend.append({
                "pattern": label, "axis": axis_name,
                "tr": tr, "te": te, "tr_day": tr_day, "te_day": te_day,
            })

    # --- パターンごとのベスト軸構成 (TRAIN基準 / TEST基準) ---
    print("\n" + "=" * 100)
    print("パターン別ベスト軸構成 (TRAIN基準ROI最大 vs TEST基準ROI最大の一致確認)")
    print("=" * 100)
    for pattern in PATTERNS:
        cells = [r for r in rows_for_recommend if r["pattern"] == pattern]
        best_tr = max(cells, key=lambda r: r["tr"]["roi"])
        best_te = max(cells, key=lambda r: r["te"]["roi"])
        match = "一致" if best_tr["axis"] == best_te["axis"] else "不一致"
        print(f"  [{pattern}] TRAINベスト={best_tr['axis']} (ROI={best_tr['tr']['roi']:.1f}%, n={best_tr['tr']['n']}) / "
              f"TESTベスト={best_te['axis']} (ROI={best_te['te']['roi']:.1f}%, n={best_te['te']['n']})  -> {match}")
        # 参考: TRAINベストのTEST側成績も併記
        tr_axis_on_test = next(r for r in cells if r["axis"] == best_tr["axis"])
        print(f"      (参考: TRAINベスト軸『{best_tr['axis']}』のTEST成績: "
              f"n={tr_axis_on_test['te']['n']}, hit={tr_axis_on_test['te']['hitrate']:.1f}%, "
              f"ROI={tr_axis_on_test['te']['roi']:.1f}%)")

    # --- 目標(ROI>100 & hit>30%)を両期間で満たす組み合わせ一覧 ---
    print("\n" + "=" * 100)
    print("目標判定: ROI100%超 かつ 的中率30%超 を TRAIN/TEST両方で満たす組み合わせ")
    print("=" * 100)
    qualifying = [r for r in rows_for_recommend
                  if r["tr"]["n"] >= 30 and r["te"]["n"] >= 30
                  and r["tr"]["roi"] > 100 and r["tr"]["hitrate"] > 30
                  and r["te"]["roi"] > 100 and r["te"]["hitrate"] > 30]
    if not qualifying:
        print("  該当なし。TRAIN/TEST双方でROI100%超&的中率30%超を同時に満たす"
              "「パターン×軸構成」の組み合わせは存在しない。")
    else:
        for r in qualifying:
            print(f"  [{r['pattern']} × {r['axis']}] "
                  f"TRAIN n={r['tr']['n']} hit={r['tr']['hitrate']:.1f}% ROI={r['tr']['roi']:.1f}% / "
                  f"TEST n={r['te']['n']} hit={r['te']['hitrate']:.1f}% ROI={r['te']['roi']:.1f}%")

    # --- 最もROIが高かった組み合わせ(n>=30 かつ TRAIN/TESTどちらもROI>0を要求)について月次推移・配当分布 ---
    print("\n" + "=" * 100)
    print("最有望候補(qualifying優先、無ければTRAIN/TEST合算n最大でROI最高のセル)の月次ROI推移・配当分布")
    print("=" * 100)
    candidates = qualifying if qualifying else [
        r for r in rows_for_recommend if r["tr"]["n"] >= 30 and r["te"]["n"] >= 30
    ]
    if not candidates:
        print("  月次分析可能な候補が無い(n>=30の組み合わせが無い)")
    else:
        best = max(candidates, key=lambda r: min(r["tr"]["roi"], r["te"]["roi"]))
        print(f"  選定: [{best['pattern']} × {best['axis']}] "
              f"TRAIN ROI={best['tr']['roi']:.1f}% / TEST ROI={best['te']['roi']:.1f}%")

        pattern_sel = None if best["pattern"] == "全体(パターン混合)" else best["pattern"]
        axis_sel = best["axis"]
        sel_recs = [r for r in records if (pattern_sel is None or r["pattern"] == pattern_sel)]
        sel_evals = [(r["date"][:7], r["axis_eval"][axis_sel]) for r in sel_recs
                     if r["axis_eval"][axis_sel] is not None]

        print("\n  月次ROI推移:")
        by_month = defaultdict(list)
        for ym, e in sel_evals:
            by_month[ym].append(e)
        all_odds_hits = []
        for ym in sorted(by_month.keys()):
            evs = by_month[ym]
            n = len(evs)
            hits = sum(e["hit"] for e in evs)
            bet = sum(e["bet"] for e in evs)
            pay = sum(e["payout"] for e in evs)
            roi = pay / bet * 100 if bet else 0.0
            hitrate = hits / n * 100 if n else 0.0
            period_tag = "TRAIN" if ym <= "2025-12" else "TEST"
            mark = " <=judged" if roi > 100 else ""
            print(f"    {ym} [{period_tag}] n={n:4d} hit={hitrate:5.1f}% ROI={roi:6.1f}%{mark}")
            for e in evs:
                if e["hit"]:
                    all_odds_hits.append(e["odds"])

        print("\n  的中時配当(オッズ倍率)分布:")
        if all_odds_hits:
            all_odds_hits_sorted = sorted(all_odds_hits)
            med = median(all_odds_hits_sorted)
            p90 = percentile(all_odds_hits_sorted, 90)
            share30 = sum(1 for o in all_odds_hits_sorted if o >= 30) / len(all_odds_hits_sorted) * 100
            print(f"    的中件数={len(all_odds_hits_sorted)} 中央値={med:.1f}倍 "
                  f"p90={p90:.1f}倍 30倍以上率={share30:.1f}%")
        else:
            print("    的中データなし")

    print("\n完了。")


if __name__ == "__main__":
    main()
