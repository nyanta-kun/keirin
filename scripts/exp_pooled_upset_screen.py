"""波乱スクリーンを 6/7/9車 で統合学習し、**9車だけを評価する**。

## なぜ統合するのか

[[keirin_9car_upset_screen_2026_08_08]] で、9車の波乱抽出(A)は帯ROIを 66%→77% へ
押し上げたが **Δratio +0.066 95%CI [-0.011, +0.147] で有意の手前**だった。
効果量は 7車で有意だった同じ現象と同じ大きさ（+0.097〜+0.107・n=23,000）で、
**違いは標本数だけ**（9車 4,738R vs 7車 60,000R）。
＝ 9車の中で工夫しても越えられない。壁は標本そのもの。

そこで **6/7/9車を1つの母集団にして学習し、評価は9車だけで行う**。

## 🔴 目的変数を車数別の分位で定義する（ここが設計の要点）

「決着オッズ >= 300倍」という**絶対閾値のまま統合すると、モデルは主に
『これは9車か』を当てるだけになる**（基準率 7車 9.3% / 9車 24.0%）。
車数は既知なので、それを当てても 9車の中の順位付けは1ミリも良くならない。

    --target quantile : 目的変数 = 「自分の車数コホートの中で上位 q の高配当か」
    --target abs      : 目的変数 = 「決着オッズ >= A_THR か」（比較用）

`quantile` は車数ごとに閾値を引き直すので、モデルは
**「同じ車数の中で相対的に荒れやすいレースはどれか」という転移する構造**だけを学ぶ。
評価は常に **9車の絶対閾値（A_THR）** で行うので、目的が変わることはない。

## 評価

    ratio = 実測発生率 ÷ 市場含意確率(Σ_{o>=A_THR} 0.75/o)      帯ROI = ratio × 75%

ratio 1.0 が控除率どおり。市場と同じ向きの分類器は精度が高くても ROI にならないので
lift だけでは判断しない（[[keirin_7h2_third_upset_rejected_2026_08_06]]）。

モデルは**半年ごとの walk-forward**（各 fold はそれ以前の全データだけで学習）。
9車単独版と**同じ fold・同じ特徴・同じ評価軸**で比較できるようにしてある。

⚠️ カテゴリ符号化に組み込みの `hash()` を使わないこと。文字列ハッシュはプロセスごとに
ランダム化され、同じ条件で結果が再現しなくなる（9車単独版で実際に踏んだ）。

## 結果（2026-08-08・86,977R = 6車5,087 / 7車77,152 / 9車4,738）

### 素の ratio は車数が多いほど高い（＝9車でやる意味はここ）

| 車数 | >=300倍 の基準率 | 市場含意 | ratio |
|---|---|---|---|
| 6車 | 4.82% | 6.50% | 0.741 |
| 7車 | 9.80% | 11.91% | 0.822 |
| **9車** | **24.04%** | 27.09% | **0.887** |

**同じ +0.13 を足しても 1.0 を超えるのは 9車だけ。**

### 統合学習は9車を改善する（評価は9車のみ・上位20%）

| 学習プール | A_THR | ratio | 期間分割 決定→評価 | 月次 | Δratio 95%CI |
|---|---|---|---|---|---|
| **6,7,9車** | 300 | **1.027** | 1.024 → **1.027** | **27/30 (90%)** | **+0.131 [+0.028, +0.230]** ✅ |
| 7,9車 | 300 | 0.974 | 0.951 → 0.981 | 19/30 (63%) | +0.070 [−0.033, +0.174] ❌ |
| 7,9車 | 500 | 1.037 | 1.063 → 0.979 | 23/30 (77%) | +0.141 [+0.018, +0.268] ✅ |
| 6,7,9車 | 500 | 1.035 | 1.071 → 0.978 | 23/30 (77%) | +0.141 [+0.006, +0.275] ✅ |
| **9車単独** | 300 | 0.996 | 0.917 → 1.062 | 24/32 (75%) | +0.090 [+0.001, +0.181] |

**統合すると 9車単独より Δratio が上がり（+0.090→+0.131）、月次一貫性も 75%→90%、
期間分割のブレも 0.917→1.062 が 1.024→1.027 へ収束する。**
帯ROI は 67.3% → **77.0%** で控除率の壁の上に出る。

### 効果の実体は 7車で確立済みの構造（9車はそれを借りている）

| 評価 | n | 上位20% ratio | Δratio 95%CI | 月次 |
|---|---|---|---|---|
| 7車 | 55,897 | 0.946 | **+0.142 [+0.103, +0.184]** | 30/32 (94%) |
| 9車 | 3,435 | 1.027 | +0.131 [+0.028, +0.230] | 27/30 (90%) |
| 6車 | 3,426 | 0.841 | +0.138 [−0.100, +0.399] | 13/31 (42%) |

Δratio は 3車数とも +0.13〜+0.14 でほぼ同じ。**違うのは素の水準だけ**で、
7車は 0.946（壁の下）、9車は 1.027（壁の上）、6車は 0.841 に着地する。
6車は月次42%で単独では成立しないが、**学習データとしては 9車の助けになる**
（外すと 9車の Δratio が +0.131→+0.070 に落ち有意でなくなる）。

### ⚠️ 読み方の注意

- **9車の CI は依然広い（±0.11）。学習プールの構成を変えるだけで ±0.06 動く。**
  「壁を超えた」と断言せず「**壁付近〜やや上**」と読むこと
- 帯ROI は**帯を丸ごと買った理論値**。9車の300倍+帯は約100点あり、1レース1万円だと
  1点100円で的中しても3〜5万円にしかならない。**実装は帯内の目選びが別途必要**で、
  帯内のモデル選択はランダムに負けると確定済み（オッズ昇順が算術上の最適）
- `--target abs` は「これは9車か」を当てるだけになるので使わないこと（比較用）

## 使い方

    .venv/bin/python scripts/exp_pooled_upset_screen.py --solo-9car
    .venv/bin/python scripts/exp_pooled_upset_screen.py --a-thr 500
    .venv/bin/python scripts/exp_pooled_upset_screen.py --pool 7,9 --eval-ne 7

DB へは書き込まない。
"""
from __future__ import annotations

import argparse
import collections
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import get_connection  # noqa: E402

ODDS_MAX = 9000.0

#: walk-forward の fold 境界（各 fold はこれ以前の全データで学習する）
FOLDS = [("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"),
         ("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"),
         ("2026-01-01", "2026-07-01"), ("2026-07-01", "2099-01-01")]

MODEL_COLS = ("pw_max", "pw_gap12", "pw_entropy", "p3_max", "p3_std", "p3_entropy")


def _load(n_list: list[int], a_thr: float, b_thr: float) -> list[dict]:
    """レース単位の (特徴, 決着オッズ, 帯の市場含意) を返す。

    ⚠️ 板は**レースあたり数百行**あるので Python へ持ってこない。必要なのは
    「決着オッズ」と「帯ごとの市場含意の和」だけなので **SQL で畳む**。
    7車を含めると素朴に持つと1,200万行になり、メモリと転送で詰まる。

    ⚠️ 絞り込みは `wt_races` への JOIN で書く。`race_key IN (SELECT ...)` にすると
    `wt_odds`（2,200万行）のプランが崩れて15秒→15分以上になる。
    """
    ne_in = ",".join(str(x) for x in n_list)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT e.race_key, r.race_date, r.n_entries, r.grade, r.race_type,
                   r.day_index, r.distance, r.start_at,
                   e.frame_no, e.race_point, e.line_group, e.line_size,
                   e.style, e.player_class, e.s_count, e.b_count,
                   e.first_rate, e.third_rate, e.prediction_mark,
                   e.pred_win_pct, e.pred_top3_pct, e.finish_order,
                   v.bank_length, v.is_indoor
            FROM wt_entries e
            JOIN wt_races r USING(race_key)
            LEFT JOIN venue_info v ON v.venue_code = r.venue_id
            WHERE r.cancel=0 AND r.n_entries IN ({ne_in})
        """)
        by_race: dict[str, list] = collections.defaultdict(list)
        for e in cur:
            by_race[e["race_key"]].append(e)

        # 決着した目のオッズと、帯ごとの市場含意（控除率25%を戻した値）を1クエリで
        cur.execute(f"""
            WITH fin AS (
              SELECT e.race_key,
                     concat(max(CASE WHEN e.finish_order=1 THEN e.frame_no END), '-',
                            max(CASE WHEN e.finish_order=2 THEN e.frame_no END), '-',
                            max(CASE WHEN e.finish_order=3 THEN e.frame_no END)) combo
              FROM wt_entries e JOIN wt_races r USING(race_key)
              WHERE r.cancel=0 AND r.n_entries IN ({ne_in}) GROUP BY e.race_key)
            SELECT o.race_key,
                   count(*) AS n_legs,
                   max(CASE WHEN o.combination=f.combo THEN o.odds_value END) AS win_odds,
                   sum(CASE WHEN o.odds_value>={a_thr} THEN 0.75/o.odds_value ELSE 0 END) AS imp_a,
                   sum(CASE WHEN o.odds_value<={b_thr} THEN 0.75/o.odds_value ELSE 0 END) AS imp_b
            FROM wt_odds o
            JOIN wt_races r USING(race_key)
            JOIN fin f ON f.race_key=o.race_key
            WHERE o.bet_type='trifecta' AND r.cancel=0 AND r.n_entries IN ({ne_in})
              AND o.odds_value>0 AND o.odds_value<{ODDS_MAX}
            GROUP BY o.race_key
        """)
        agg = {o["race_key"]: o for o in cur}

    out = []
    for rk, ents in by_race.items():
        ne = int(ents[0]["n_entries"] or 0)
        if len(ents) != ne:
            continue                    # 事前欠車のあるレースは行が消えるので対象外
        a = agg.get(rk)
        if not a or a["win_odds"] is None:
            continue
        if a["n_legs"] < ne * (ne - 1) * (ne - 2) * 0.9:
            continue                    # 板が欠けているレースは市場含意が測れない
        row = _features(ents)
        row.update(race_key=rk, date=ents[0]["race_date"], n_entries=ne,
                   win_odds=float(a["win_odds"]),
                   impA=float(a["imp_a"]), impB=float(a["imp_b"]))
        out.append(row)
    return out


def _features(ents: list) -> dict:
    """出走表だけから作れるレース単位の特徴（オッズ非依存）。

    **車数で意味が変わる量は割合に直す**（統合学習では 6/7/9車が混ざるため）。
    """
    ne = len(ents)
    rp = np.array([float(e["race_point"] or 0.0) for e in ents])
    rp_sorted = np.sort(rp)[::-1]
    lines: dict = collections.defaultdict(list)
    for e in ents:
        key = e["line_group"] if e["line_group"] is not None else f"solo{e['frame_no']}"
        lines[key].append(e)
    line_rp = sorted((sum(float(x["race_point"] or 0) for x in v) for v in lines.values()),
                     reverse=True)
    styles = collections.Counter((e["style"] or "")[:1] for e in ents)
    classes = collections.Counter(e["player_class"] or "" for e in ents)
    first = np.array([float(e["first_rate"] or 0.0) for e in ents])
    third = np.array([float(e["third_rate"] or 0.0) for e in ents])
    marks = {e["prediction_mark"]: e for e in ents if e["prediction_mark"] in (1, 2, 3)}
    rp_rank = {e["frame_no"]: i for i, e in
               enumerate(sorted(ents, key=lambda x: -float(x["race_point"] or 0)))}
    f = {
        # 競走得点の分布 — 「番組がフラットか」。車数に依らない量
        "rp_std": float(np.std(rp)), "rp_mean": float(np.mean(rp)),
        "rp_range": float(rp_sorted[0] - rp_sorted[-1]),
        "rp_gap12": float(rp_sorted[0] - rp_sorted[1]),
        "rp_gap23": float(rp_sorted[1] - rp_sorted[2]),
        "rp_top2_edge": float(np.mean(rp_sorted[:2]) - np.mean(rp_sorted[2:])),
        # ライン構成。**個数ではなく割合**にして車数間で比較可能にする
        "line_ratio": len(lines) / ne,
        "max_line_ratio": max(len(v) for v in lines.values()) / ne,
        "solo_ratio": sum(1 for v in lines.values() if len(v) == 1) / ne,
        "line_rp_gap12": float(line_rp[0] - line_rp[1]) if len(line_rp) > 1 else 0.0,
        "line_rp_std": float(np.std(line_rp)),
        # 脚質も割合で
        "nige_ratio": styles.get("逃", 0) / ne,
        "makuri_ratio": styles.get("捲", 0) / ne,
        "oikomi_ratio": styles.get("追", 0) / ne,
        # 実績のばらつき
        "first_max": float(first.max()), "first_std": float(first.std()),
        "third_std": float(third.std()),
        "s_mean": sum(int(e["s_count"] or 0) for e in ents) / ne,
        "b_mean": sum(int(e["b_count"] or 0) for e in ents) / ne,
        # 級班の混在度
        "class_ratio": len(classes) / ne,
        "top_class_share": max(classes.values()) / ne,
        # 番組
        "day_index": int(ents[0]["day_index"] or 0),
        "distance": int(ents[0]["distance"] or 0),
        "bank_length": float(ents[0]["bank_length"] or 0),
        "is_indoor": int(ents[0]["is_indoor"] or 0),
        "hour": _hour(ents[0]["start_at"]),
        "grade_enc": _enc(ents[0]["grade"]), "rtype_enc": _enc(ents[0]["race_type"]),
        # WT公式印（市場の代理変数だがオッズではない）。順位も割合に直す
        "mark1_rp_rank_ratio": (rp_rank[marks[1]["frame_no"]] / ne) if 1 in marks else -1.0,
        "mark1_line_size_ratio": (int(marks[1]["line_size"] or 1) / ne) if 1 in marks else -1.0,
    }
    pw = [e["pred_win_pct"] for e in ents]
    p3 = [e["pred_top3_pct"] for e in ents]
    if all(x is not None for x in pw) and all(x is not None for x in p3):
        pw_a = np.sort(np.array([float(x) for x in pw]))[::-1]
        p3_a = np.sort(np.array([float(x) for x in p3]))[::-1]
        f.update(pw_max=float(pw_a[0]), pw_gap12=float(pw_a[0] - pw_a[1]),
                 pw_entropy=_entropy(pw_a), p3_max=float(p3_a[0]),
                 p3_std=float(p3_a.std()), p3_entropy=_entropy(p3_a))
    else:
        f.update(dict.fromkeys(MODEL_COLS, np.nan))
    return f


def _entropy(v: np.ndarray) -> float:
    p = np.asarray(v, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _hour(start_at) -> int:
    try:
        import datetime as dt
        return dt.datetime.fromtimestamp(int(start_at), dt.UTC).astimezone(
            dt.timezone(dt.timedelta(hours=9))).hour
    except Exception:
        return -1


def _enc(v) -> int:
    """カテゴリの**決定的**ハッシュ。`hash()` は seed でぶれるので使わない。"""
    return zlib.crc32(str(v).encode()) % 1000 if v is not None else -1


def _make_target(rows: list[dict], mode: str, a_thr: float, q: float) -> np.ndarray:
    """学習用の目的変数。

    `quantile` は**車数ごとに閾値を引き直す**ので、モデルは「これは9車か」ではなく
    「同じ車数の中で相対的に荒れやすいか」を学ぶ。
    """
    if mode == "abs":
        return np.array([1.0 if r["win_odds"] >= a_thr else 0.0 for r in rows])
    y = np.zeros(len(rows))
    for ne in {r["n_entries"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["n_entries"] == ne]
        thr = np.quantile([rows[i]["win_odds"] for i in idx], 1 - q)
        for i in idx:
            y[i] = 1.0 if rows[i]["win_odds"] >= thr else 0.0
    return y


def _walk_forward(rows, cols, y_train) -> np.ndarray:
    import lightgbm as lgb

    scores = np.full(len(rows), np.nan)
    dates = np.array([r["date"] for r in rows])
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
    for lo, hi in FOLDS:
        te = (dates >= lo) & (dates < hi)
        tr = dates < lo
        if te.sum() < 50 or tr.sum() < 400 or y_train[tr].sum() < 30:
            continue
        m = lgb.train(
            {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15,
             "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.8,
             "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1, "seed": 0},
            lgb.Dataset(X[tr], label=y_train[tr]), num_boost_round=350)
        scores[te] = m.predict(X[te])
    return scores


def _report(tag, rows, s, yA, impA, ok, cut_split, best_q) -> None:
    """評価対象（9車など）に絞った層別・期間分割・bootstrap。"""
    if ok.sum() < 300:
        print(f"--- {tag}: 標本不足 ---")
        return
    print(f"\n--- {tag}（n={ok.sum()}）---")
    base_r = yA[ok].mean() / impA[ok].mean()
    print(f"  全件 実測={yA[ok].mean()*100:.2f}% 市場含意={impA[ok].mean()*100:.2f}% "
          f"ratio={base_r:.3f} 帯ROI={base_r*75:.1f}%")
    for q in (0.5, 0.3, 0.2, 0.1):
        m = ok & (s >= np.nanquantile(s[ok], 1 - q))
        if m.sum() < 100:
            continue
        r = yA[m].mean() / impA[m].mean()
        print(f"  上位{q:>4.0%} n={m.sum():>5} 実測={yA[m].mean()*100:>5.2f}% "
              f"lift={yA[m].mean()/yA[ok].mean():>4.2f} ratio={r:>5.3f} 帯ROI={r*75:>5.1f}%")

    dates = np.array([r["date"] for r in rows])
    dev = ok & (dates < cut_split)
    con = ok & (dates >= cut_split)
    if dev.sum() < 200 or con.sum() < 200:
        return
    print(f"  ---- 分位カットの期間分割（決定〜{cut_split} / 評価{cut_split}〜・一度きり）----")
    for q in (0.5, 0.3, 0.2, 0.1):
        t = np.nanquantile(s[dev], 1 - q)
        md, mc = dev & (s >= t), con & (s >= t)
        if md.sum() < 80 or mc.sum() < 80:
            continue
        rd = yA[md].mean() / impA[md].mean()
        rc = yA[mc].mean() / impA[mc].mean()
        print(f"    上位{q:>4.0%} | 決定期 n={md.sum():>5} ratio={rd:>5.3f} | "
              f"評価期 n={mc.sum():>5} 実測={yA[mc].mean()*100:>5.2f}% ratio={rc:>5.3f} "
              f"帯ROI={rc*75:>5.1f}%")

    # 採用候補の頑健性（月次と日ブロック bootstrap）
    t = np.nanquantile(s[dev], 1 - best_q)
    sel = ok & (s >= t)
    months: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
    for i, r in enumerate(rows):
        if not ok[i]:
            continue
        mm = months[r["date"][:7]]
        mm[0] += 1
        mm[2] += yA[i]
        if sel[i]:
            mm[1] += 1
            mm[3] += yA[i]
    win = tot = 0
    for _, v in sorted(months.items()):
        if v[1] < 5 or v[0] < 20:
            continue
        tot += 1
        win += int(v[3] / v[1] > v[2] / v[0])
    rng = np.random.default_rng(0)
    by_day: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if ok[i]:
            by_day[r["date"]].append(i)
    keys = list(by_day)
    deltas = []
    for _ in range(1000):
        idx = rng.choice(len(keys), size=len(keys), replace=True)
        b = [i for j in idx for i in by_day[keys[j]]]
        se = [i for i in b if sel[i]]
        if len(se) < 50:
            continue
        deltas.append(yA[se].mean() / impA[se].mean() - yA[b].mean() / impA[b].mean())
    print(f"  ---- 採用候補（上位{best_q:.0%}）の頑健性 ----")
    if tot:
        print(f"    月次で全体を上回った月 {win}/{tot} ({win/tot*100:.0f}%)")
    if deltas:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f"    Δratio = {np.mean(deltas):+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}]"
              f"  → {'有意' if lo > 0 else '有意差なし'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="6,7,9", help="学習に使う車数（カンマ区切り）")
    ap.add_argument("--eval-ne", type=int, default=9, help="評価する車数")
    ap.add_argument("--a-thr", type=float, default=300.0, help="評価する波乱の下限オッズ")
    ap.add_argument("--b-thr", type=float, default=50.0)
    ap.add_argument("--target", choices=("quantile", "abs"), default="quantile")
    ap.add_argument("--target-q", type=float, default=0.25,
                    help="quantile モードで「上位いくつを波乱とみなすか」")
    ap.add_argument("--features", choices=("card", "all"), default="card")
    ap.add_argument("--best-q", type=float, default=0.2, help="頑健性を見る採用候補の上位割合")
    ap.add_argument("--cut-split", default="2025-07-01")
    ap.add_argument("--solo-9car", action="store_true",
                    help="比較用に9車単独学習も同じ枠組みで走らせる")
    args = ap.parse_args()

    pool = [int(x) for x in args.pool.split(",")]
    rows = _load(pool, args.a_thr, args.b_thr)
    ne_arr = np.array([r["n_entries"] for r in rows])
    print(f"読み込み {len(rows)}R  内訳: "
          + " / ".join(f"{ne}車 {(ne_arr == ne).sum()}R" for ne in sorted(set(ne_arr))))

    yA = np.array([1.0 if r["win_odds"] >= args.a_thr else 0.0 for r in rows])
    impA = np.array([r["impA"] for r in rows])
    for ne in sorted(set(ne_arr)):
        m = ne_arr == ne
        print(f"  {ne}車: >={args.a_thr:.0f}倍 の基準率 {yA[m].mean()*100:5.2f}%  "
              f"市場含意 {impA[m].mean()*100:5.2f}%  ratio {yA[m].mean()/impA[m].mean():.3f}")

    drop = ("race_key", "date", "win_odds", "impA", "impB")
    cols = [c for c in rows[0] if c not in drop]     # n_entries は特徴として残す
    if args.features == "card":
        cols = [c for c in cols if c not in MODEL_COLS]
    y_train = _make_target(rows, args.target, args.a_thr, args.target_q)
    print(f"\n特徴 {len(cols)}本（{args.features}）/ 学習目的変数 {args.target}"
          f"{'(上位%.0f%%)' % (args.target_q*100) if args.target == 'quantile' else ''}"
          f" 陽性率 {y_train.mean()*100:.1f}%")

    s = _walk_forward(rows, cols, y_train)
    ok = ~np.isnan(s)
    print(f"\n=== 統合学習（{','.join(map(str, pool))}車）→ 評価 ===")
    _report(f"{args.eval_ne}車のみ", rows, s, yA, impA,
            ok & (ne_arr == args.eval_ne), args.cut_split, args.best_q)
    for ne in sorted(set(ne_arr)):
        if ne == args.eval_ne:
            continue
        _report(f"参考: {ne}車", rows, s, yA, impA, ok & (ne_arr == ne),
                args.cut_split, args.best_q)

    if args.solo_9car:
        idx = [i for i, r in enumerate(rows) if r["n_entries"] == args.eval_ne]
        sub = [rows[i] for i in idx]
        y_sub = _make_target(sub, args.target, args.a_thr, args.target_q)
        s2 = _walk_forward(sub, cols, y_sub)
        print(f"\n=== 比較: {args.eval_ne}車単独学習 ===")
        _report(f"{args.eval_ne}車のみ", sub, s2, yA[idx], impA[idx], ~np.isnan(s2),
                args.cut_split, args.best_q)


if __name__ == "__main__":
    main()
