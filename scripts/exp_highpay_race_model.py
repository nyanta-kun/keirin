#!/usr/bin/env python3
"""高配当レース選別モデル（オッズ非使用・Phase 1）。

## ユーザー方針（2026-08-06）

> 「オッズもリアルオッズではなく、**過去の同じようなレースと競走得点構成、ライン構成**
>  などから大まかで良いです。毎日複数のレースが**50倍以上の払い戻し**となっていると
>  思います。こちらは現在稼働させている的中率重視のモデルと異なり、
>  **高額払い戻しを的中した実績を作るためのモデル**となります」

したがって:
- **リアルタイムオッズは一切使わない**（朝の入稿時点で確定している構造情報のみ）。
  [[keirin_highpay_payout_ceiling_2026_08_06]] で朝オッズが使えないと確定したため、
  そもそもオッズを条件に使わない設計にする。
- 目的は ROI ではなく **「50倍以上の払い戻しが出るレースを言い当てる」**こと。
  ⚠️ ROI は 76% 前後から動かない（恒等式）。それは狙いではないとユーザー判断済み。

## 目的変数

    y = 1 if 三連複の的中配当 >= THRESH倍 (既定 50)

「レース固有の量」であり買い目選択に依存しない（＝レース選別として正しい定義）。
三連単側の配当も併記する。

## 先行実装との違い

`exp_payout_expectation_model.py`（2026-07-30）は log(配当) の**回帰**で
TEST R² 0.109 / Spearman 0.352 / 十分位で配当中央値 15.9→2.9倍 を得ている。
本スクリプトは
  ① 目的を **50倍以上の二値**へ（＝ユーザーの評価軸に合わせる）
  ② 特徴量に **ライン形状・得点構成・脚質構成・級班混在・同県**を追加
  ③ **四半期ごとの walk-forward**（単一 TRAIN/TEST でなく全期間 honest）
  ④ 9車立てにも対応
  ⑤ 「1日あたり上位N レースを選ぶ」運用形での評価を出す

## 評価の読み方

**AUC ではなく「上位帯の 50倍+ 率が基準の何倍か（lift）」と「1日に何件出るか」**で読む。
単一特徴量（rp_std 単独など）に勝てなければモデルにする意味がない。

DB は読み取りのみ。
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.database import get_connection  # noqa: E402

JST = timezone(timedelta(hours=9))
DATE_FROM, DATE_TO = "2024-01-01", "2026-08-04"
GRADE_MAP = {"L級": 0, "A級": 1, "S級": 2, "SA混合": 3}

# walk-forward 窓（この開始日より前だけで学習する）
WF_WINDOWS = [
    ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
    ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30"),
    ("2026-07-01", "2026-08-04"),
]


def _entropy(vals: list[float]) -> float:
    total = sum(vals)
    if total <= 0:
        return 0.0
    return -sum(max(v / total, 1e-9) * math.log(max(v / total, 1e-9)) for v in vals)


# ---------------------------------------------------------------- 読み込み
def load_races(n_car: int) -> dict:
    with get_connection() as c:
        rows = c.execute(
            "SELECT r.race_key, r.race_date, r.grade, r.race_type, r.day_index, "
            "       r.start_at, r.distance, v.bank_length, v.is_indoor "
            "FROM keirin.wt_races r "
            "LEFT JOIN keirin.venue_info v ON r.venue_id = v.venue_code "
            "WHERE r.n_entries = ? AND r.cancel = 0 "
            "  AND r.race_date BETWEEN ? AND ?",
            (n_car, DATE_FROM, DATE_TO)).fetchall()
    out = {}
    for r in rows:
        is_night = 0
        try:
            dt = datetime.fromtimestamp(int(r["start_at"]), tz=timezone.utc).astimezone(JST)
            is_night = 1 if dt.hour >= 17 else 0
        except (TypeError, ValueError):
            pass
        out[r["race_key"]] = {
            "race_date": str(r["race_date"]), "grade": r["grade"],
            "race_type": r["race_type"], "day_index": r["day_index"],
            "bank_length": r["bank_length"], "is_indoor": r["is_indoor"],
            "distance": r["distance"], "is_night": is_night,
        }
    print(f"[load] races({n_car}車): {len(out)}", flush=True)
    return out


def load_entries(race_keys: list[str]) -> dict:
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(race_keys), 700):
            ch = race_keys[i:i + 700]
            q = ("SELECT race_key, frame_no, pred_win_pct, pred_top3_pct, prediction_mark, "
                 "       race_point, line_group, line_size, line_pos, is_line_leader, "
                 "       n_lines, finish_order, style, prefecture, player_class "
                 "FROM keirin.wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(ch)))
            for r in c.execute(q, ch):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load] entries: {len(by_race)} races", flush=True)
    return by_race


def load_win_payouts(race_keys: list[str], winners: dict) -> tuple[dict, dict]:
    """的中目の三連複／三連単 最終オッズ（＝配当/100）を返す。"""
    trio_out, tf_out = {}, {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 700):
            ch = race_keys[i:i + 700]
            ph = ",".join("?" * len(ch))
            tb, fb = defaultdict(dict), defaultdict(dict)
            for r in c.execute(
                    "SELECT race_key, bet_type, combination, odds_value FROM keirin.wt_odds "
                    f"WHERE bet_type IN ('trio','trifecta') AND race_key IN ({ph}) "
                    "AND odds_value > 0", ch):
                if r["bet_type"] == "trifecta":
                    fb[r["race_key"]][r["combination"]] = float(r["odds_value"])
                else:
                    p = frozenset(int(x) for x in re.split(r"[-=→]", r["combination"]))
                    if len(p) == 3:
                        tb[r["race_key"]][p] = float(r["odds_value"])
            for rk in ch:
                w = winners.get(rk)
                if not w:
                    continue
                v = tb.get(rk, {}).get(w["trio"])
                if v is not None:
                    trio_out[rk] = v
                v = fb.get(rk, {}).get(w["trifecta"])
                if v is not None:
                    tf_out[rk] = v
            if (i // 700) % 15 == 0:
                print(f"[load]   payout {i}/{len(race_keys)}", flush=True)
    print(f"[load] win payouts: trio {len(trio_out)} / trifecta {len(tf_out)}", flush=True)
    return trio_out, tf_out


# ---------------------------------------------------------------- 特徴量
def build_rows(races: dict, ents_by_race: dict, n_car: int) -> tuple[list, dict]:
    out, winners = [], {}
    for rk, meta in races.items():
        ents = ents_by_race.get(rk)
        if not ents or len(ents) != n_car:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners[rk] = {
            "trio": frozenset(f for _, f in fin[:3]),
            "trifecta": "-".join(str(f) for _, f in fin[:3]),
        }

        by_frame = {int(e["frame_no"]): e for e in ents}
        # --- WT 予想（発走前に確定・オッズではない） ---
        wv = sorted((float(e["pred_win_pct"] or 0) for e in ents), reverse=True)
        tv = sorted((float(e["pred_top3_pct"] or 0) for e in ents), reverse=True)
        has_pred = sum(wv) > 0
        w1 = max(by_frame, key=lambda f: float(by_frame[f]["pred_win_pct"] or 0))
        honmei = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 2), None)
        mark_t3 = mark_w = 0.0
        mark_same_line = -1
        if honmei is not None and taikou is not None:
            mark_t3 = (float(by_frame[honmei]["pred_top3_pct"] or 0)
                       + float(by_frame[taikou]["pred_top3_pct"] or 0))
            mark_w = (float(by_frame[honmei]["pred_win_pct"] or 0)
                      + float(by_frame[taikou]["pred_win_pct"] or 0))
            lh, lt = by_frame[honmei]["line_group"], by_frame[taikou]["line_group"]
            mark_same_line = 1 if (lh is not None and lh == lt) else 0

        # --- 競走得点の構成 ---
        rps = sorted((float(e["race_point"]) for e in ents
                      if e["race_point"] is not None), reverse=True)
        if len(rps) < 3:
            continue
        rp_std = float(np.std(rps))
        rp_mean = float(np.mean(rps))
        # 上位2名の得点が全体からどれだけ抜けているか＝「力の突出」
        rp_top2_edge = (rps[0] + rps[1]) / 2 - rp_mean

        # --- ライン構成 ---
        lines: dict = defaultdict(list)
        for e in ents:
            lines[e["line_group"] if e["line_group"] is not None
                  else f"solo{e['frame_no']}"].append(e)
        sizes = sorted((len(v) for v in lines.values()), reverse=True)
        line_rp = sorted((sum(float(x["race_point"] or 0) for x in v)
                          for v in lines.values()), reverse=True)
        # 最強ラインの得点シェアと2番目との差＝ライン間の力関係
        rp_total = sum(rps) or 1.0
        line_top_share = line_rp[0] / rp_total
        line_gap12 = (line_rp[0] - line_rp[1]) if len(line_rp) >= 2 else line_rp[0]

        pref = defaultdict(int)
        for e in ents:
            if e["prefecture"]:
                pref[e["prefecture"]] += 1
        n_samepref_pairs = sum(v * (v - 1) // 2 for v in pref.values())

        styles = defaultdict(int)
        for e in ents:
            styles[e["style"] or "?"] += 1
        # 逃型が同一ラインに複数＝先行争いが同居
        senko_lines = defaultdict(int)
        for e in ents:
            if e["style"] == "逃":
                senko_lines[e["line_group"]] += 1
        n_senko_clash = sum(1 for k, v in senko_lines.items() if k is not None and v >= 2)

        classes = {e["player_class"] for e in ents if e["player_class"]}

        rt = str(meta["race_type"] or "")
        row = {
            "race_key": rk, "race_date": meta["race_date"], "has_pred": has_pred,
            # WT 予想の拡散度＝荒れ予測の主軸
            "win_max": wv[0], "win_gap12": wv[0] - wv[1],
            "win_sum_top2": wv[0] + wv[1], "win_sum_top3": sum(wv[:3]),
            "win_entropy": _entropy(wv),
            "top3_max": tv[0], "top3_gap12": tv[0] - tv[1],
            "top3_sum_top2": tv[0] + tv[1], "top3_sum_top3": sum(tv[:3]),
            "top3_entropy": _entropy(tv),
            "mark_top3_sum": mark_t3, "mark_win_sum": mark_w,
            "honmei_is_w1": 1 if honmei == w1 else 0,
            "mark_same_line": mark_same_line,
            # 得点構成
            "rp_max": rps[0], "rp_min": rps[-1], "rp_mean": rp_mean, "rp_std": rp_std,
            "rp_gap12": rps[0] - rps[1], "rp_gap23": rps[1] - rps[2],
            "rp_range": rps[0] - rps[-1], "rp_top2_edge": rp_top2_edge,
            # ライン構成
            "n_lines": float(ents[0]["n_lines"] or len(lines)),
            "max_line_size": float(sizes[0]),
            "line_size_2nd": float(sizes[1]) if len(sizes) > 1 else 0.0,
            "n_solo": float(sum(1 for s in sizes if s == 1)),
            "line_top_share": line_top_share, "line_gap12": line_gap12,
            "n_samepref_pairs": float(n_samepref_pairs),
            # 脚質構成
            "n_senko": float(styles.get("逃", 0)),
            "n_oikomi": float(styles.get("追", 0)),
            "n_ryo": float(styles.get("両", 0)),
            "n_senko_clash": float(n_senko_clash),
            # 制度・会場
            "n_classes": float(len(classes)),
            "grade_enc": float(GRADE_MAP.get(meta["grade"], -1)),
            "day_index": float(meta["day_index"] or 0),
            "rt_final": 1.0 if "決勝" in rt else 0.0,
            "rt_semi": 1.0 if "準決" in rt else 0.0,
            "rt_heat": 1.0 if "予選" in rt else 0.0,
            "rt_senbatsu": 1.0 if "選抜" in rt else 0.0,
            "rt_tokusen": 1.0 if "特選" in rt else 0.0,
            "bank_length": float(meta["bank_length"] or 0),
            "is_indoor": float(meta["is_indoor"] or 0),
            "is_night": float(meta["is_night"]),
            "distance": float(meta["distance"] or 0),
        }
        out.append(row)
    print(f"[build] rows: {len(out)}", flush=True)
    return out, winners


FEATURES = [
    "win_max", "win_gap12", "win_sum_top2", "win_sum_top3", "win_entropy",
    "top3_max", "top3_gap12", "top3_sum_top2", "top3_sum_top3", "top3_entropy",
    "mark_top3_sum", "mark_win_sum", "honmei_is_w1", "mark_same_line",
    "rp_max", "rp_min", "rp_mean", "rp_std", "rp_gap12", "rp_gap23",
    "rp_range", "rp_top2_edge",
    "n_lines", "max_line_size", "line_size_2nd", "n_solo",
    "line_top_share", "line_gap12", "n_samepref_pairs",
    "n_senko", "n_oikomi", "n_ryo", "n_senko_clash",
    "n_classes", "grade_enc", "day_index",
    "rt_final", "rt_semi", "rt_heat", "rt_senbatsu", "rt_tokusen",
    "bank_length", "is_indoor", "is_night", "distance",
]


# ---------------------------------------------------------------- 評価
def decile_report(scores, y, payouts, thresh, title, n_bins=10):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(y)[order]
    pay = np.asarray(payouts)[order]
    n = len(y)
    base = y.mean()
    print(f"\n  {title}  (n={n} / 基準 {thresh}倍+率 {base * 100:.2f}%)")
    print("    分位   件数   {t}倍+率   lift   配当中央  100倍+率".format(t=thresh))
    for b in range(n_bins):
        lo, hi = n * b // n_bins, n * (b + 1) // n_bins
        yy, pp = y[lo:hi], pay[lo:hi]
        if len(yy) == 0:
            continue
        print(f"    D{b + 1:<2}  {len(yy):6}  {yy.mean() * 100:7.2f}%  "
              f"{yy.mean() / base:5.2f}  {np.median(pp):8.1f}  "
              f"{np.mean(pp >= 100) * 100:6.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-car", type=int, default=7)
    ap.add_argument("--thresh", type=float, default=50.0)
    ap.add_argument("--bet-type", default="trio", choices=["trio", "trifecta"])
    ap.add_argument("--per-day", default="1,2,3,5",
                    help="1日あたり上位N件を選ぶ運用の評価")
    args = ap.parse_args()

    races = load_races(args.n_car)
    keys = sorted(races)
    ents = load_entries(keys)
    rows, winners = build_rows(races, ents, args.n_car)
    trio_pay, tf_pay = load_win_payouts(sorted(winners), winners)
    pay_map = trio_pay if args.bet_type == "trio" else tf_pay

    rows = [r for r in rows if r["race_key"] in pay_map]
    for r in rows:
        r["payout"] = pay_map[r["race_key"]]
        r["y"] = 1 if r["payout"] >= args.thresh else 0
    n_pred = sum(1 for r in rows if r["has_pred"])
    print(f"[data] 評価対象 {len(rows)} レース / WT予想あり {n_pred} "
          f"({n_pred / max(len(rows), 1) * 100:.1f}%)")
    print(f"[data] {args.bet_type} {args.thresh}倍+ 発生率 "
          f"{np.mean([r['y'] for r in rows]) * 100:.2f}%")

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows])
    pay = np.array([r["payout"] for r in rows])
    dates = np.array([r["race_date"] for r in rows])

    # ---- walk-forward 予測 ----
    pred = np.full(len(rows), np.nan)
    for w_from, w_to in WF_WINDOWS:
        tr = dates < w_from
        te = (dates >= w_from) & (dates <= w_to)
        if te.sum() == 0 or tr.sum() < 3000:
            continue
        model = lgb.train(
            {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
             "min_data_in_leaf": 100, "feature_fraction": 0.8,
             "bagging_fraction": 0.8, "bagging_freq": 1,
             "verbose": -1, "seed": 42},
            lgb.Dataset(X[tr], label=y[tr]), num_boost_round=300)
        pred[te] = model.predict(X[te])
        print(f"  [wf] {w_from}〜{w_to}  train {tr.sum():6} / test {te.sum():5}  "
              f"AUC {roc_auc_score(y[te], pred[te]):.4f}", flush=True)

    ok = ~np.isnan(pred)
    print(f"\n=== honest walk-forward 全体 (n={ok.sum()}) ===")
    print(f"  AUC {roc_auc_score(y[ok], pred[ok]):.4f}")
    decile_report(pred[ok], y[ok], pay[ok], args.thresh, "モデル（45特徴）")

    # ---- 単一特徴量ベースライン ----
    for f in ("rp_std", "top3_entropy", "win_entropy", "n_solo", "rp_gap12"):
        i = FEATURES.index(f)
        v = X[ok][:, i]
        # 荒れる方向へ符号を合わせる
        auc = roc_auc_score(y[ok], v)
        sign = 1.0 if auc >= 0.5 else -1.0
        decile_report(sign * v, y[ok], pay[ok], args.thresh,
                      f"単一特徴 {f}（AUC {max(auc, 1 - auc):.4f}）")

    # ---- 「1日あたり上位N件」運用形 ----
    print(f"\n=== 1日あたり上位N件を選ぶ運用（{args.bet_type} {args.thresh}倍+）===")
    base = y[ok].mean()
    dd = dates[ok]
    pp, yy, paa = pred[ok], y[ok], pay[ok]
    print("    N/日   選定数  日数   {t}倍+率  lift   配当中央  最大配当".format(
        t=args.thresh))
    for n_top in [int(x) for x in args.per_day.split(",")]:
        sel = []
        for d in np.unique(dd):
            idx = np.where(dd == d)[0]
            idx = idx[np.argsort(-pp[idx])][:n_top]
            sel.extend(idx.tolist())
        sel = np.array(sel)
        print(f"    {n_top:4}  {len(sel):7} {len(np.unique(dd)):5}  "
              f"{yy[sel].mean() * 100:7.2f}%  {yy[sel].mean() / base:5.2f}  "
              f"{np.median(paa[sel]):8.1f}  {paa[sel].max():8.1f}")

    # ---- 重要度（最終窓のモデル） ----
    imp = sorted(zip(FEATURES, model.feature_importance("gain")),
                 key=lambda x: -x[1])[:15]
    print("\n=== 特徴量重要度 上位15（最終窓モデル・gain）===")
    for f, g in imp:
        print(f"    {f:<18} {g:12.0f}")


if __name__ == "__main__":
    main()
