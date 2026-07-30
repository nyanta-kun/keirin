"""WT印(◎◯)関連の課題1〜5を、月次凍結vintageモデル（2026-07-29構築・書き込み
保護済み・唯一の正本`src.wt_vintage_config`）で再計算する
（[[keirin_s7_foundational_rethink_2026_07_29]] / [[keirin_wt_foundational_audit_2026_07_29]]）。

【重要】以前報告していた以下の数値は全て`wt_entries.pred_top3_pct`/
`pred_win_pct`というDB格納値（2026-07-19バックフィル時点の四半期vintageモデル
出力・その後2026-07-28に該当モデルファイルが無断上書きされ再現不能と判明済み）
を直接参照して計算したものであり、信頼性が確認できていなかった:
  - 課題1: neitherの事前識別（entropy/mark_sum、precision≈ベースレート）
  - 課題2: 軸1選定精度（pred_win_pct高い方、69.1-70.8%/67.4%）
  - 課題3: 軸2選定精度（非マーク最上位、64.4-65.2%/60.9-62.3%）
  - 課題4: 軸1・軸2同時的中率（46.2%/40.9%、独立仮定とほぼ一致）
  - 課題5: both除外のmark_sum precision/recall（mark_sum<=130で precision58%等）

本スクリプトは、DB格納値を一切参照せず、月次vintageモデルをその都度ロードして
`predict_proba`し直すことで、上記5項目を全てクリーンな状態から再計算する。
"""
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.wt_vintage_config import monthly_windows

TRAIN_FROM, TRAIN_TO = "20240101", "20251231"
TEST_FROM, TEST_TO = "20260101", "20261231"


def build_month(eval_model_name, date_from, date_to, win_model_name):
    """月内の全7車立てレースについて、honmei/taikou/非マーク5車のpred値と
    実際の3着内結果を持つdictのリストを返す（DB格納値は一切使わない）。
    """
    model = load_model(eval_model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins, marks = {}, {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_top3"] = model.predict_proba(X)[:, 1] * 100
    df["pred_win"] = win_model.predict_proba(X)[:, 1] * 100

    out = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        if wt_honmei is None or wt_taikou is None:
            continue
        row_by_frame = {int(r.frame_no): r for r in g.itertuples(index=False)}
        if wt_honmei not in row_by_frame or wt_taikou not in row_by_frame:
            continue
        fin = [(fo, fno) for fo, fno in fins.get(rk, []) if fno in row_by_frame]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        h_hit = wt_honmei in winners
        t_hit = wt_taikou in winners

        h_top3 = float(row_by_frame[wt_honmei].pred_top3)
        t_top3 = float(row_by_frame[wt_taikou].pred_top3)
        h_win = float(row_by_frame[wt_honmei].pred_win)
        t_win = float(row_by_frame[wt_taikou].pred_win)
        mark_sum = h_top3 + t_top3

        others_frames = [f for f in row_by_frame if f not in (wt_honmei, wt_taikou)]
        if len(others_frames) != 5:
            continue
        others_sorted = sorted(others_frames, key=lambda f: -float(row_by_frame[f].pred_top3))
        other_hits = frozenset(f for f in others_frames if f in winners)

        all_pcts = [float(row_by_frame[f].pred_top3) for f in row_by_frame]
        import math
        total = sum(all_pcts)
        entropy = 0.0
        if total > 0:
            for v in all_pcts:
                s = max(v / total, 1e-9)
                entropy -= s * math.log(s)

        out.append({
            "race_key": rk, "race_date": date_from[:0] + rk[:8],
            "h_hit": h_hit, "t_hit": t_hit,
            "h_top3": h_top3, "t_top3": t_top3, "h_win": h_win, "t_win": t_win,
            "mark_sum": mark_sum, "entropy": entropy,
            "gap1": abs(h_win - t_win),
            "top3_pick": others_sorted[0], "top3_pick2": others_sorted[1],
            "gap2": (float(row_by_frame[others_sorted[0]].pred_top3)
                     - float(row_by_frame[others_sorted[1]].pred_top3)),
            "other_hits": other_hits,
        })
    return out


def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    windows = monthly_windows()
    print(f"対象月数: {len(windows)}（{windows[0][0]}〜{windows[-1][1]}）")

    all_races = []
    for date_from, date_to, eval_model, win_model in windows:
        print(f"[build] {date_from}〜{date_to}  eval={eval_model} win={win_model}", flush=True)
        recs = build_month(eval_model, date_from, date_to, win_model)
        print(f"[build]   races: {len(recs)}", flush=True)
        all_races.extend(recs)

    print(f"\n[main] 全期間 races合計: {len(all_races)}")
    train = [r for r in all_races if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in all_races if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"TRAIN: {len(train)}件 / TEST: {len(test)}件")

    # exactly_one/both/neither の分類
    def categorize(r):
        if r["h_hit"] and r["t_hit"]:
            return "both"
        if r["h_hit"] or r["t_hit"]:
            return "exactly_one"
        return "neither"

    print("\n" + "=" * 78)
    print("カテゴリ内訳（クリーン再計算）")
    print("=" * 78)
    for label, data in (("全期間", all_races), ("TRAIN", train), ("TEST", test)):
        n = len(data)
        cnt = defaultdict(int)
        for r in data:
            cnt[categorize(r)] += 1
        print(f"[{label}] n={n}  both={cnt['both']}({cnt['both']/n*100:.1f}%)  "
              f"exactly_one={cnt['exactly_one']}({cnt['exactly_one']/n*100:.1f}%)  "
              f"neither={cnt['neither']}({cnt['neither']/n*100:.1f}%)")

    # ===== 課題2: 軸1選定精度 =====
    print("\n" + "=" * 78)
    print("課題2（再検証）: exactly_oneにおける軸1選定精度")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        eo = [r for r in data if categorize(r) == "exactly_one"]
        n = len(eo)
        correct_dyn = sum(1 for r in eo if (r["h_win"] >= r["t_win"]) == r["h_hit"])
        correct_always_h = sum(1 for r in eo if r["h_hit"])
        acc_dyn = correct_dyn / n * 100 if n else 0
        acc_h = correct_always_h / n * 100 if n else 0
        lo, hi = wilson_ci(correct_dyn, n)
        print(f"  [{label}] n={n}  動的選定(pred_win高い方)精度={acc_dyn:.1f}% "
              f"(95%CI {lo*100:.1f}-{hi*100:.1f}%)  常に◎固定={acc_h:.1f}%")

    # ===== 課題3: 軸2選定精度 =====
    print("\n" + "=" * 78)
    print("課題3（再検証）: exactly_oneにおける軸2選定精度（非マーク5車から1車）")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        eo = [r for r in data if categorize(r) == "exactly_one"]
        n = len(eo)
        correct = sum(1 for r in eo if r["top3_pick"] in r["other_hits"])
        acc = correct / n * 100 if n else 0
        lo, hi = wilson_ci(correct, n)
        print(f"  [{label}] n={n}  精度={acc:.1f}% (95%CI {lo*100:.1f}-{hi*100:.1f}%) "
              f"※ランダム基準=40.0%")

    # ===== 課題4: 軸1・軸2同時的中率 =====
    print("\n" + "=" * 78)
    print("課題4（再検証）: 軸1・軸2の同時的中率")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        eo = [r for r in data if categorize(r) == "exactly_one"]
        n = len(eo)
        a1 = sum(1 for r in eo if (r["h_win"] >= r["t_win"]) == r["h_hit"])
        a2 = sum(1 for r in eo if r["top3_pick"] in r["other_hits"])
        both = sum(1 for r in eo if (r["h_win"] >= r["t_win"]) == r["h_hit"]
                   and r["top3_pick"] in r["other_hits"])
        a1p, a2p = a1 / n * 100 if n else 0, a2 / n * 100 if n else 0
        bothp = both / n * 100 if n else 0
        indep = a1p / 100 * a2p / 100 * 100
        print(f"  [{label}] n={n} 軸1単独={a1p:.1f}% 軸2単独={a2p:.1f}% "
              f"同時的中(実測)={bothp:.1f}% 独立仮定の期待値={indep:.1f}%")

    # ===== 課題5: both除外のmark_sum precision/recall =====
    print("\n" + "=" * 78)
    print("課題5（再検証）: mark_sumによるboth除外の識別精度")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        for th in (110, 120, 130, 140, 150):
            flagged = [r for r in data if r["mark_sum"] >= th]
            n = len(flagged)
            total_both = sum(1 for r in data if categorize(r) == "both")
            if n == 0 or total_both == 0:
                continue
            prec = sum(1 for r in flagged if categorize(r) == "both") / n * 100
            recall = sum(1 for r in flagged if categorize(r) == "both") / total_both * 100
            print(f"  [{label}] mark_sum>={th}: n={n} precision={prec:.1f}% recall={recall:.1f}%")

    # ===== 課題1: neitherの事前識別（entropy/mark_sum） =====
    print("\n" + "=" * 78)
    print("課題1（再検証）: entropy/mark_sumによるneither事前識別")
    print("=" * 78)
    for label, data in (("TRAIN", train), ("TEST", test)):
        base_rate = sum(1 for r in data if categorize(r) == "neither") / len(data) * 100
        print(f"\n  [{label}] ベースレート(無条件neither率)={base_rate:.1f}%")
        for ent_th, mark_th in [(1.9, 100), (1.8, 120), (1.7, 130), (1.6, 130)]:
            flagged = [r for r in data if r["entropy"] <= ent_th and r["mark_sum"] >= mark_th]
            n = len(flagged)
            total_neither = sum(1 for r in data if categorize(r) == "neither")
            if n == 0 or total_neither == 0:
                continue
            prec = sum(1 for r in flagged if categorize(r) == "neither") / n * 100
            recall = sum(1 for r in flagged if categorize(r) == "neither") / total_neither * 100
            lift = prec / base_rate if base_rate else 0
            print(f"    ent<={ent_th},mark>={mark_th}: n={n} precision={prec:.1f}% "
                  f"(リフト{lift:.2f}倍) recall={recall:.1f}%")


if __name__ == "__main__":
    main()
