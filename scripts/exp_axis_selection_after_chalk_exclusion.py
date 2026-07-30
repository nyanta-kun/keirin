"""【硬いレース除外後の軸選定方法の検討】（2026-07-30）。

[[keirin_dominance_pattern_verification_2026_07_30]] /
[[keirin_s7_foundational_rethink_2026_07_29]]の続き。ユーザー方針:
「硬いを除外し、残りのレースに対して軸の選別方法の検討を行う」。

## 主指標: P(軸2車がともに3着内)

三連複2軸流し（軸2車＋残り5車のいずれか1車＝5点）が的中する条件は
**「軸2車がともに3着以内に入ること」と厳密に同値**（3列目は総流しなので
軸2車が入れば3頭目は必ずカバーされる）。よって軸選定の品質は
`P(both axes in top3)` という単一指標で測れる。これを主指標とし、
配当統計・ROIは副次情報として併記する。

## 硬いレース除外の方法（2通りを比較）

`exp_chalk_vs_upset_discrimination.py`の知見:
- 硬い(<10倍)判別は TEST AUC 0.647（LightGBM 34特徴）
- ただし単一特徴 `top3_sum_top2` だけで AUC 0.631 とほぼ同等
  → LightGBMはTRAIN 0.801/TEST 0.647と過学習が明確なので、
    単純閾値の方が頑健と考えられる

本スクリプトでは両方を比較する:
  (a) `top3_sum_top2` 上位X%を除外（単純・頑健）
  (b) LightGBM硬い判別器スコア上位X%を除外（TRAINのみで学習）
除外率は 0%(除外なし)/10%/20%/30%/40%/50% で振る。閾値はすべて
**TRAIN分布で確定しTESTに固定適用**（リーク防止）。

## 軸選定方法（10種・ライン構成を使う新案を含む）

既存検証済み: w1+w2 / t1+t2 / w1+t2以降非w1 / w1+w3 / S7新設計
新規（未検証）: ライン構成を使う軸ペア選定（同ライン/別ライン）、◎固定系

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
STAKE = 100


def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date, grade FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: {"race_date": str(r["race_date"]), "grade": r["grade"]}
             for r in rrows}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_win_pct, pred_top3_pct, prediction_mark, "
                 "       line_group, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))

    print("[load] trio odds ...", flush=True)
    # 勝ち組合せのオッズと、5点流し用の全comboオッズ両方が必要
    boards = {}
    with get_connection() as c:
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
                try:
                    parts = frozenset(int(x) for x in re.split(r"[-=→]", str(comb)))
                except ValueError:
                    continue
                if len(parts) == 3:
                    boards.setdefault(rk, {})[parts] = fv
            if (i // 900) % 20 == 0:
                print(f"[load]   trio progress: {i}/{len(keys)}", flush=True)
    print(f"[load]   boards: {len(boards)}", flush=True)
    return races, by_race, boards


def build(races, entries_by_race, boards):
    print("[build] ...", flush=True)
    out = []
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_win_pct"] is None or e["pred_top3_pct"] is None for e in ents):
            continue
        board = boards.get(rk)
        if not board:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        win_pay = board.get(winners)
        if win_pay is None:
            continue

        by_frame = {int(e["frame_no"]): e for e in ents}
        frames = list(by_frame.keys())
        wsorted = sorted(frames, key=lambda f: -float(by_frame[f]["pred_win_pct"]))
        tsorted = sorted(frames, key=lambda f: -float(by_frame[f]["pred_top3_pct"]))
        wv = [float(by_frame[f]["pred_win_pct"]) for f in wsorted]
        tv = [float(by_frame[f]["pred_top3_pct"]) for f in tsorted]

        honmei = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 2), None)

        out.append({
            "race_key": rk, "race_date": meta["race_date"],
            "frames": frames, "by_frame": by_frame, "board": board,
            "winners": winners, "win_pay": win_pay,
            "wsorted": wsorted, "tsorted": tsorted,
            "top3_sum_top2": tv[0] + tv[1],
            "win_entropy": _entropy(wv), "top3_entropy": _entropy(tv),
            "win_max": wv[0], "win_gap12": wv[0] - wv[1],
            "honmei": honmei, "taikou": taikou,
        })
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


# ============ 軸選定方法（10種） ============
def _first_other_line(r, base):
    """baseと別ラインの車のうちpred_top3_pct最上位。無ければNone。"""
    bf = r["by_frame"]
    lg_base = bf[base]["line_group"]
    for f in r["tsorted"]:
        if f == base:
            continue
        if bf[f]["line_group"] is None or lg_base is None:
            continue
        if bf[f]["line_group"] != lg_base:
            return f
    return None


def _first_same_line(r, base):
    """baseと同ラインの車のうちpred_top3_pct最上位。無ければNone。"""
    bf = r["by_frame"]
    lg_base = bf[base]["line_group"]
    if lg_base is None:
        return None
    for f in r["tsorted"]:
        if f == base:
            continue
        if bf[f]["line_group"] == lg_base:
            return f
    return None


def axis_w1_w2(r):
    return r["wsorted"][0], r["wsorted"][1]


def axis_t1_t2(r):
    return r["tsorted"][0], r["tsorted"][1]


def axis_w1_t_next(r):
    w1 = r["wsorted"][0]
    for f in r["tsorted"]:
        if f != w1:
            return w1, f
    return None


def axis_w1_w3(r):
    return r["wsorted"][0], r["wsorted"][2]


def axis_s7_new(r):
    """S7新設計: ◎◯のwin高い方 + 非マークtop3最上位"""
    h, t = r["honmei"], r["taikou"]
    if h is None or t is None:
        return None
    bf = r["by_frame"]
    a1 = h if float(bf[h]["pred_win_pct"]) >= float(bf[t]["pred_win_pct"]) else t
    for f in r["tsorted"]:
        if f not in (h, t):
            return a1, f
    return None


def axis_w1_otherline(r):
    """【新】w1 + w1と別ラインのtop3最上位"""
    w1 = r["wsorted"][0]
    o = _first_other_line(r, w1)
    return (w1, o) if o is not None else None


def axis_w1_sameline(r):
    """【新】w1 + w1と同ラインのtop3最上位（ライン連携狙い）"""
    w1 = r["wsorted"][0]
    s = _first_same_line(r, w1)
    return (w1, s) if s is not None else None


def axis_t1_otherline(r):
    """【新】t1 + t1と別ラインのtop3最上位"""
    t1 = r["tsorted"][0]
    o = _first_other_line(r, t1)
    return (t1, o) if o is not None else None


def axis_honmei_nonmark(r):
    """【新】◎固定 + 非マークtop3最上位"""
    h, t = r["honmei"], r["taikou"]
    if h is None:
        return None
    for f in r["tsorted"]:
        if f != h and f != t:
            return h, f
    return None


def axis_t1_t3(r):
    return r["tsorted"][0], r["tsorted"][2]


AXIS_METHODS = [
    ("1:w1+w2", axis_w1_w2),
    ("2:t1+t2", axis_t1_t2),
    ("3:w1+t次点", axis_w1_t_next),
    ("4:w1+w3", axis_w1_w3),
    ("5:S7新設計", axis_s7_new),
    ("6:w1+別ラインtop3【新】", axis_w1_otherline),
    ("7:w1+同ラインtop3【新】", axis_w1_sameline),
    ("8:t1+別ラインtop3【新】", axis_t1_otherline),
    ("9:◎+非マークtop3【新】", axis_honmei_nonmark),
    ("10:t1+t3", axis_t1_t3),
]


def evaluate(rows, axis_fn):
    """P(軸2車ともに3着内)=的中率、および5点流しROI・配当統計を返す。"""
    n = hits = 0
    bet_total = pay_total = 0
    hit_pays = []
    for r in rows:
        sel = axis_fn(r)
        if sel is None:
            continue
        a1, a2 = sel
        if a1 == a2:
            continue
        others = [f for f in r["frames"] if f not in (a1, a2)]
        combos = {}
        for x in others:
            key = frozenset({a1, a2, x})
            if key in r["board"]:
                combos[key] = r["board"][key]
        if not combos:
            continue
        n += 1
        bet_total += len(combos) * STAKE
        # 軸2車がともに3着内 <=> winnersがcombosに含まれる
        if r["winners"] in combos:
            hits += 1
            pay = int(r["win_pay"] * STAKE)
            pay_total += pay
            hit_pays.append(r["win_pay"])
    hitrate = hits / n * 100 if n else 0.0
    roi = pay_total / bet_total * 100 if bet_total else 0.0
    hit_pays.sort()
    med = hit_pays[len(hit_pays) // 2] if hit_pays else 0.0
    over30 = sum(1 for p in hit_pays if p >= 30) / len(hit_pays) * 100 if hit_pays else 0.0
    return n, hitrate, roi, med, over30


def main():
    races, entries_by_race, boards = load_all()
    rows = build(races, entries_by_race, boards)

    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")

    # ===== 硬い除外 (a) top3_sum_top2 単純閾値 =====
    tr_vals = sorted((r["top3_sum_top2"] for r in train), reverse=True)

    # ===== 硬い除外 (b) LightGBM 硬い判別器（TRAINのみ学習） =====
    print("[chalk-lgb] 硬い判別器をTRAINで学習 ...", flush=True)
    import lightgbm as lgb
    FEATS = ["top3_sum_top2", "win_entropy", "top3_entropy", "win_max", "win_gap12"]
    Xtr = np.array([[r[f] for f in FEATS] for r in train], dtype=float)
    ytr = np.array([1 if r["win_pay"] < 10.0 else 0 for r in train])
    m = lgb.LGBMClassifier(objective="binary", metric="auc", n_estimators=300,
                            learning_rate=0.05, num_leaves=15, min_child_samples=60,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(Xtr, ytr)
    for r, p in zip(train, m.predict_proba(Xtr)[:, 1]):
        r["chalk_score"] = float(p)
    Xte = np.array([[r[f] for f in FEATS] for r in test], dtype=float)
    for r, p in zip(test, m.predict_proba(Xte)[:, 1]):
        r["chalk_score"] = float(p)
    tr_scores = sorted((r["chalk_score"] for r in train), reverse=True)

    EXCL = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    for mode, key, tr_sorted_vals in (
            ("(a)top3_sum_top2上位除外", "top3_sum_top2", tr_vals),
            ("(b)LGB硬いスコア上位除外", "chalk_score", tr_scores)):
        print(f"\n{'='*104}")
        print(f"硬い除外方式 {mode}")
        print(f"{'='*104}")
        for ex in EXCL:
            if ex == 0.0:
                thr = float("inf")
                tr_pool, te_pool = train, test
                label = "除外なし"
            else:
                # TRAIN分布の上位ex%点を閾値として確定（TESTにも固定適用）
                idx = max(0, int(len(tr_sorted_vals) * ex) - 1)
                thr = tr_sorted_vals[idx]
                tr_pool = [r for r in train if r[key] < thr]
                te_pool = [r for r in test if r[key] < thr]
                label = f"上位{int(ex*100)}%除外"
            base_med_tr = sorted(r["win_pay"] for r in tr_pool)
            base_med_te = sorted(r["win_pay"] for r in te_pool)
            mtr = base_med_tr[len(base_med_tr)//2] if base_med_tr else 0
            mte = base_med_te[len(base_med_te)//2] if base_med_te else 0
            print(f"\n--- {label}  (母集団 TRAIN n={len(tr_pool)} 配当中央値{mtr:.1f}倍 / "
                  f"TEST n={len(te_pool)} 配当中央値{mte:.1f}倍) ---")
            print(f"  {'軸選定':<26}{'TRAIN n':>9}{'的中%':>8}{'ROI%':>8}"
                  f"{'TEST n':>9}{'的中%':>8}{'ROI%':>8}{'的中中央値':>11}{'30倍+%':>9}")
            for name, fn in AXIS_METHODS:
                n1, h1, r1, _, _ = evaluate(tr_pool, fn)
                n2, h2, r2, med2, o2 = evaluate(te_pool, fn)
                flag = ""
                if h2 >= 30 and r2 >= 100:
                    flag = " ★目標達成"
                elif r2 >= 100:
                    flag = " ★ROI100+"
                print(f"  {name:<26}{n1:>9}{h1:>7.1f}%{r1:>7.1f}%"
                      f"{n2:>9}{h2:>7.1f}%{r2:>7.1f}%{med2:>10.1f}倍{o2:>8.1f}%{flag}")


if __name__ == "__main__":
    main()
