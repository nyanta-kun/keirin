"""9車立て — 波乱レース(A)の抽出と人気決着レース(B)の除外、母集団 (A)−(B) の評価。

## 何をしているか（ユーザー方針・2026-08-08）

> 「ROI確保が困難と結論づけられたが、まずは**一定以上の波乱が発生する可能性がある
>   ものを絞り込めるか (A)**。逆に**一定以上の人気で決着する可能性があるレースを
>   除外できるか (B)**。**(A)−(B) を母集団として検討する**ところから始める。」

[[keirin_9car_highpay_rank_rejected_2026_08_08]] は「買い目の形」から入って失敗した。
今回は**母集団の選別だけ**を独立に測る。券種は三連単。

    目的変数 A : 決着した三連単オッズ >= A_THR（既定 300倍）  … 波乱
    目的変数 B : 決着した三連単オッズ <= B_THR（既定  50倍）  … 人気決着

## 🔴 lift だけでは採否を決められない（この検証の肝）

分類器が当たっても、**市場が既に同じことを知っていれば買い目の値段に織り込まれていて
回収率にならない**（[[keirin_7h2_third_upset_rejected_2026_08_06]]:「市場と同じ向きの
分類器は精度がどれだけ高くても ROI にならない」）。そこで各層で必ず

    ratio = 実測発生率 ÷ 市場含意確率(Σ_{該当オッズ} 0.75/o)

を併記する。**ratio が 1.0 に近いほど「市場が値段を付けきれていない」**＝妙味がある。
lift が高くても ratio が動かないなら、それは市場の写しでしかない。

## 検証の作り（前借りをしない）

モデルは **半年ごとの walk-forward**（各 fold はそれ以前のデータだけで学習）。
本番モデルを全期間で学習して過去へ遡らせる model-vintage look-ahead を避ける。

    fold = 2024H1 / 2024H2 / 2025H1 / 2025H2 / 2026H1 / 2026H2
    train = その fold より前の全期間（最初の学習は 2022-12〜2023-12）

`--features card` は出走表だけ（全期間そろう）、`--features all` は
`pred_win_pct` / `pred_top3_pct` 由来の特徴も足す（**2024-01 以降しか無い**ので
最初期の学習では欠損。LightGBM の欠損処理に委ねる）。

## 結果（2026-08-08・9車 4,738R・2022-12〜2026-08）

### (B) のほうが (A) より明確に予測しやすい

| 目的変数 | AUC(card) | AUC(all) |
|---|---|---|
| A: 決着>=300倍 | 0.5350 | 0.5435 |
| A: 決着>=500倍 | 0.5341 | — |
| **B: 決着<=50倍** | **0.5944** | **0.6011** |

**「荒れる」より「荒れない」ほうが当てられる**というのは設計上の重要事実。
ただし B の各層は ratio が 1.01〜1.15 と**市場より高く**出る＝人気決着はむしろ
市場が過小評価しており、B を当てても妙味にはならない。**B は除外にだけ使う。**

### (A)−(B) の母集団（カットは決定期 2024-01〜2025-06 で固定し評価期へ当てる）

**A_THR=300 / B_THR=50**（評価期 2025-07〜）

| 構成 | 決定期 ratio | 評価期 n | 300倍+率 | 評価期 ratio | 帯ROI |
|---|---|---|---|---|---|
| 全件 | 0.912 | 1,486 | 23.96% | 0.878 | 65.9% |
| B除外50% のみ | 0.951 | 703 | 26.60% | 0.920 | 69.0% |
| A上位30% のみ | 0.928 | 478 | 28.66% | 0.982 | 73.7% |
| **A上位20% のみ** | 0.953 | 295 | 30.51% | **1.027** | **77.0%** |
| **A上位20% ∧ B除外50%** | 0.979 | 249 | 31.33% | **1.037** | **77.8%** |

**A_THR=500 / B_THR=50** も同じ形（全件 0.845 → A上位20% で **1.016 / 76.2%**）。

- **主レバーは (A) の抽出**。全件 ratio 0.85〜0.88 が A上位20% で 1.02〜1.04 になる
  ＝帯ROI 63〜66% → 76〜78% で**控除率の壁の上に出る**
- **(B) の除外は単独では弱い**（0.874〜0.920）。A に重ねても +0.01 程度しか足さない。
  ただし**害は無い**ので、件数を絞りたいときの補助には使える
- A_THR は **300倍のほうが 500倍より安定**（発生率が 24% と 15% で標本が違う）
- B_THR は 50倍が最良（30倍にすると評価期 ratio が落ちる）
- モデル出力（`pred_*`）を足しても改善しない＝**出走表だけで完結する**

### ⚠️ ただし統計的には「有意の手前」

    A_THR=300 / A上位30% ∧ B除外50%
      Δratio(選別−全体) = +0.0660  95%CI [-0.0113, +0.1470]   → 有意差なし（下限がほぼ0）
      月次で全体を上回った月 23/31 (74%)
    A_THR=500 / 同条件
      Δratio            = +0.0677  95%CI [-0.0594, +0.1912]   → 有意差なし
      月次 21/32 (66%)

**効果量は 7車で有意だった同じ現象と同じ大きさ**
（[[keirin_highpay_tail_mispricing_2026_08_08]]: 7車 Δratio +0.097〜+0.107・n=23,000 で
有意・月次12〜13/13）なのに、9車は 4,738R しかないため CI が3〜5倍広い。
＝ **「9車で新しい効果を見つけた」のではなく「7車で確立した効果が9車でも同方向に
出ているが、9車単独では検出力が足りない」** と読むのが正しい。
実際、単一特徴でも `rp_std低半分` が lift 1.12 / ratio 0.952 と 7車と同じ向きに出る。

**→ 次にやるなら 9車だけで学習するのをやめ、7車と1つの母集団にして車数を特徴に
入れる**（検出力の壁は標本数そのものなので、9車の中で工夫しても越えられない）。

### 🔴 途中で踏んだ再現性のバグ（同型の再発を防ぐ）

初回は `_enc()` が組み込みの `hash(str(v))` を使っていた。**Python の文字列ハッシュは
プロセスごとにランダム化される**（PYTHONHASHSEED）ため `grade_enc` / `rtype_enc` が
実行のたびに変わり、木の分割が変わって**同じデータ・同じ条件で結果が再現しなかった**。
実測で評価期 ratio が **1.062 → 0.963** と動いた（最初の値は seed の当たりだった）。
`zlib.crc32` に置き換えて解決。**カテゴリのハッシュ符号化に `hash()` を使わないこと。**

## 使い方

    .venv/bin/python scripts/exp_9car_upset_screen.py
    .venv/bin/python scripts/exp_9car_upset_screen.py --a-thr 500 --b-thr 50
    .venv/bin/python scripts/exp_9car_upset_screen.py --features all --n-entries 7

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

#: モデル出力由来の特徴（2024-01 以降しか無い）
MODEL_COLS = ("pw_max", "pw_gap12", "pw_entropy", "p3_max", "p3_std", "p3_entropy")


def _load(n_entries: int) -> list[dict]:
    """レース単位の (オッズ非依存の特徴, 決着オッズ, 全オッズ) を返す。

    ⚠️ 絞り込みは `wt_races` への JOIN で書く。`race_key IN (SELECT ...)` にすると
    `wt_odds`（2,200万行）のプランが崩れて15秒→15分以上になる（2026-08-08 実測）。
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT e.race_key, r.race_date, r.venue_id, r.grade, r.race_type,
                   r.day_index, r.distance, r.start_at,
                   e.frame_no, e.race_point, e.line_group, e.line_size, e.is_line_leader,
                   e.style, e.player_class, e.n_lines, e.s_count, e.b_count,
                   e.first_rate, e.third_rate, e.prediction_mark,
                   e.pred_win_pct, e.pred_top3_pct, e.finish_order,
                   v.bank_length, v.is_indoor
            FROM wt_entries e
            JOIN wt_races r USING(race_key)
            LEFT JOIN venue_info v ON v.venue_code = r.venue_id
            WHERE r.cancel=0 AND r.n_entries={n_entries}
        """)
        by_race: dict[str, list] = collections.defaultdict(list)
        for e in cur:
            by_race[e["race_key"]].append(e)

        cur.execute(f"""
            SELECT o.race_key, o.combination, o.odds_value
            FROM wt_odds o JOIN wt_races r USING(race_key)
            WHERE o.bet_type='trifecta' AND r.cancel=0 AND r.n_entries={n_entries}
              AND o.odds_value>0 AND o.odds_value<{ODDS_MAX}
        """)
        odds: dict[str, dict] = collections.defaultdict(dict)
        for o in cur:
            odds[o["race_key"]][o["combination"]] = float(o["odds_value"])

    full = n_entries * (n_entries - 1) * (n_entries - 2)
    out = []
    for rk, ents in by_race.items():
        # 事前欠車は行自体が消える（CLAUDE.md「finish_order=0 の意味」）ので、
        # 行数がそろっていないレースは「その車数で発走していない」＝対象外。
        if len(ents) != n_entries:
            continue
        board = odds.get(rk)
        if not board or len(board) < full * 0.9:
            continue
        fin = {e["finish_order"]: e["frame_no"] for e in ents
               if e["finish_order"] in (1, 2, 3)}
        if len(fin) < 3:
            continue
        win = f"{fin[1]}-{fin[2]}-{fin[3]}"
        if win not in board:
            continue
        row = _features(ents)
        row.update(race_key=rk, date=ents[0]["race_date"], win_odds=board[win],
                   board=sorted(board.values()))
        out.append(row)
    return out


def _features(ents: list) -> dict:
    """出走表だけから作れるレース単位の特徴（オッズ非依存）。"""
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
        # 競走得点の分布 — 「番組がフラットか」
        "rp_std": float(np.std(rp)), "rp_mean": float(np.mean(rp)),
        "rp_range": float(rp_sorted[0] - rp_sorted[-1]),
        "rp_gap12": float(rp_sorted[0] - rp_sorted[1]),
        "rp_gap23": float(rp_sorted[1] - rp_sorted[2]),
        "rp_top2_edge": float(np.mean(rp_sorted[:2]) - np.mean(rp_sorted[2:])),
        # ライン構成 — 競輪はライン戦なので隊列の作りが決着形を決める
        "n_lines": len(lines), "max_line": max(len(v) for v in lines.values()),
        "n_solo": sum(1 for v in lines.values() if len(v) == 1),
        "line_rp_gap12": float(line_rp[0] - line_rp[1]) if len(line_rp) > 1 else 0.0,
        "line_rp_std": float(np.std(line_rp)),
        # 脚質 — 逃げが多いと潰し合いになり波乱が増える、という定番の仮説
        "n_nige": styles.get("逃", 0), "n_makuri": styles.get("捲", 0),
        "n_oikomi": styles.get("追", 0),
        # 実績のばらつき
        "first_max": float(first.max()), "first_std": float(first.std()),
        "third_std": float(third.std()),
        "s_sum": sum(int(e["s_count"] or 0) for e in ents),
        "b_sum": sum(int(e["b_count"] or 0) for e in ents),
        # 級班の混在度（同じ級班ばかりだと力が拮抗する）
        "n_class": len(classes),
        "top_class_share": max(classes.values()) / len(ents),
        # 番組
        "day_index": int(ents[0]["day_index"] or 0),
        "distance": int(ents[0]["distance"] or 0),
        "bank_length": float(ents[0]["bank_length"] or 0),
        "is_indoor": int(ents[0]["is_indoor"] or 0),
        "hour": _hour(ents[0]["start_at"]),
        "grade_enc": _enc(ents[0]["grade"]), "rtype_enc": _enc(ents[0]["race_type"]),
        # WT公式印 — 記者が本命をどこに置いたか（市場の代理変数だがオッズではない）
        "mark1_rp_rank": rp_rank.get(marks[1]["frame_no"], -1) if 1 in marks else -1,
        "mark1_line_size": int(marks[1]["line_size"] or 1) if 1 in marks else -1,
    }
    # モデル出力（2024-01 以降のみ。無い期間は NaN のまま LightGBM に渡す）
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
    return float(-(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))).sum())


def _hour(start_at) -> int:
    """発走時刻（JST の時）。ミッドナイト・モーニングの別を粗く表す。"""
    try:
        import datetime as dt
        return dt.datetime.fromtimestamp(int(start_at), dt.UTC).astimezone(
            dt.timezone(dt.timedelta(hours=9))).hour
    except Exception:
        return -1


def _enc(v) -> int:
    """カテゴリの**決定的**ハッシュ。順序に意味は無く、木が分割に使えればよい。

    ⚠️ 組み込みの `hash()` を使ってはいけない。文字列ハッシュはプロセスごとに
    ランダム化される（PYTHONHASHSEED）ため、**同じデータで実行するたびに特徴量が
    変わり、結果が再現しない**。2026-08-08 にこれを踏み、同一条件の再実行で
    評価期 ratio が 1.062 → 0.963 と大きく動いた（＝最初の値は seed の当たり）。
    """
    return zlib.crc32(str(v).encode()) % 1000 if v is not None else -1


def _implied(board: list[float], lo: float | None, hi: float | None) -> float:
    """帯 [lo, hi] を全部買ったときの的中確率の市場推定（控除率25%を戻した値）。"""
    return sum(0.75 / o for o in board
               if (lo is None or o >= lo) and (hi is None or o <= hi))


def _walk_forward(rows: list[dict], cols: list[str], target: str) -> np.ndarray:
    """半年ごとの walk-forward で out-of-fold スコアを返す（学習外は nan）。"""
    import lightgbm as lgb

    scores = np.full(len(rows), np.nan)
    dates = np.array([r["date"] for r in rows])
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
    y = np.array([r[target] for r in rows], dtype=float)
    for lo, hi in FOLDS:
        te = (dates >= lo) & (dates < hi)
        tr = dates < lo
        if te.sum() < 50 or tr.sum() < 400 or y[tr].sum() < 30:
            continue
        m = lgb.train(
            {"objective": "binary", "learning_rate": 0.03, "num_leaves": 7,
             "min_data_in_leaf": 60, "feature_fraction": 0.7, "bagging_fraction": 0.8,
             "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1, "seed": 0},
            lgb.Dataset(X[tr], label=y[tr]), num_boost_round=250)
        scores[te] = m.predict(X[te])
    return scores


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(s)
    if ok.sum() < 50 or len(set(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def _strata(score: np.ndarray, y: np.ndarray, implied: np.ndarray,
            label: str, n_bin: int = 5) -> None:
    ok = ~np.isnan(score)
    if ok.sum() < 200:
        print(f"  {label}: 標本不足")
        return
    s, yy, im = score[ok], y[ok], implied[ok]
    edges = np.quantile(s, np.linspace(0, 1, n_bin + 1))
    print(f"  {label}  （基準 実測={yy.mean()*100:.2f}% / 市場含意={im.mean()*100:.2f}% "
          f"/ ratio={yy.mean()/im.mean():.3f}）")
    for i in range(n_bin):
        m = ((s >= edges[i]) & (s <= edges[i + 1])) if i == n_bin - 1 else \
            ((s >= edges[i]) & (s < edges[i + 1]))
        if m.sum() < 30:
            continue
        print(f"    D{i+1} n={m.sum():>5}  実測={yy[m].mean()*100:>5.2f}%  "
              f"lift={yy[m].mean()/yy.mean():>4.2f}  市場含意={im[m].mean()*100:>5.2f}%  "
              f"ratio={yy[m].mean()/im[m].mean():>5.3f}")


def _population(rows, sA, sB, yA, yB, impA, ok, days) -> None:
    print("\n=== 母集団 (A高) − (B高) ===")
    print(f"{'A上位':>6} {'B除外':>6} {'n':>6} {'件/日':>6} "
          f"{'>=A_THR':>8} {'lift':>5} {'ratio':>6} {'<=B_THR':>8} {'帯ROI':>6}")
    for a_q in (1.0, 0.5, 0.3, 0.2):
        for b_q in (0.0, 0.2, 0.3, 0.5):
            m = ok.copy()
            if a_q < 1.0:
                m &= sA >= np.nanquantile(sA[ok], 1 - a_q)
            if b_q > 0.0:
                m &= sB <= np.nanquantile(sB[ok], 1 - b_q)
            if m.sum() < 150:
                continue
            r = yA[m].mean() / impA[m].mean()
            print(f"{a_q:>6.0%} {b_q:>6.0%} {m.sum():>6} {m.sum()/days:>6.2f} "
                  f"{yA[m].mean()*100:>7.2f}% {yA[m].mean()/yA[ok].mean():>5.2f} "
                  f"{r:>6.3f} {yB[m].mean()*100:>7.2f}% {r*75:>5.1f}%")


def _split_check(rows, sA, sB, yA, impA, ok, args) -> None:
    """分位カットの選び方そのものが多重比較になるので、期間で割って一度きり確認する。"""
    dates = np.array([r["date"] for r in rows])
    dev = ok & (dates < args.cut_split)
    con = ok & (dates >= args.cut_split)
    if dev.sum() < 400 or con.sum() < 400:
        return
    print(f"\n=== 分位カットの期間分割確認（決定={min(dates[dev])}〜{args.cut_split} / "
          f"評価={args.cut_split}〜・一度きり）===")
    print(f"{'A上位':>6} {'B除外':>6} | {'決定期 n':>8} {'ratio':>6} | "
          f"{'評価期 n':>8} {'>=A_THR':>8} {'ratio':>6} {'帯ROI':>6}")
    for a_q in (1.0, 0.5, 0.3, 0.2):
        for b_q in (0.0, 0.3, 0.5):
            # 閾値は**決定期の分布**から取り、評価期へはそのまま当てる（引き直さない）
            ta = np.nanquantile(sA[dev], 1 - a_q) if a_q < 1.0 else -np.inf
            tb = np.nanquantile(sB[dev], 1 - b_q) if b_q > 0.0 else np.inf
            md = dev & (sA >= ta) & (sB <= tb)
            mc = con & (sA >= ta) & (sB <= tb)
            if md.sum() < 100 or mc.sum() < 100:
                continue
            rd = yA[md].mean() / impA[md].mean()
            rc = yA[mc].mean() / impA[mc].mean()
            print(f"{a_q:>6.0%} {b_q:>6.0%} | {md.sum():>8} {rd:>6.3f} | "
                  f"{mc.sum():>8} {yA[mc].mean()*100:>7.2f}% {rc:>6.3f} {rc*75:>5.1f}%")

    # 平均は窓別の反転を隠すので、採用候補の月次符号一致と bootstrap CI を必ず見る
    ta = np.nanquantile(sA[dev], 1 - args.best_a)
    tb = np.nanquantile(sB[dev], 1 - args.best_b)
    sel = ok & (sA >= ta) & (sB <= tb)
    print(f"\n--- 採用候補（A上位{args.best_a:.0%} ∧ B除外{args.best_b:.0%}）の頑健性 ---")
    months: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
    for i, r in enumerate(rows):
        if not ok[i]:
            continue
        m = months[r["date"][:7]]
        m[0] += 1
        m[2] += yA[i]
        if sel[i]:
            m[1] += 1
            m[3] += yA[i]
    win = tot = 0
    for _, v in sorted(months.items()):
        if v[1] < 5 or v[0] < 20:
            continue
        tot += 1
        win += int(v[3] / v[1] > v[2] / v[0])
    if tot:
        print(f"  選別後の発生率が全体を上回った月 {win}/{tot} ({win/tot*100:.0f}%)")

    # 月次は1ヶ月あたり数件しか選ばれず検出力が無いので、日ブロック bootstrap も見る
    rng = np.random.default_rng(0)
    by_day: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if ok[i]:
            by_day[r["date"]].append(i)
    day_keys = list(by_day)
    deltas = []
    for _ in range(1000):
        idx = rng.choice(len(day_keys), size=len(day_keys), replace=True)
        b = [i for j in idx for i in by_day[day_keys[j]]]
        s = [i for i in b if sel[i]]
        if len(s) < 50:
            continue
        deltas.append(yA[s].mean() / impA[s].mean() - yA[b].mean() / impA[b].mean())
    if deltas:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f"  Δratio(選別−全体) = {np.mean(deltas):+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}]"
              f"  → {'有意' if lo > 0 else '有意差なし'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-entries", type=int, default=9)
    ap.add_argument("--a-thr", type=float, default=300.0, help="波乱の下限オッズ")
    ap.add_argument("--b-thr", type=float, default=50.0, help="人気決着の上限オッズ")
    ap.add_argument("--features", choices=("card", "all"), default="card")
    ap.add_argument("--best-a", type=float, default=0.3, help="頑健性を見る候補のA上位割合")
    ap.add_argument("--best-b", type=float, default=0.5, help="同・B除外割合")
    ap.add_argument("--cut-split", default="2025-07-01",
                    help="分位カットを決める期間と一度きり評価する期間の境目")
    args = ap.parse_args()

    rows = _load(args.n_entries)
    print(f"{args.n_entries}車 {len(rows)}R（{min(r['date'] for r in rows)}〜"
          f"{max(r['date'] for r in rows)}）")

    for r in rows:
        r["A"] = 1.0 if r["win_odds"] >= args.a_thr else 0.0
        r["B"] = 1.0 if r["win_odds"] <= args.b_thr else 0.0
        r["impA"] = _implied(r["board"], args.a_thr, None)
        r["impB"] = _implied(r["board"], None, args.b_thr)
    yA = np.array([r["A"] for r in rows])
    yB = np.array([r["B"] for r in rows])
    impA = np.array([r["impA"] for r in rows])
    impB = np.array([r["impB"] for r in rows])
    print(f"A: 決着>={args.a_thr:.0f}倍 の基準率 {yA.mean()*100:.2f}%  "
          f"B: 決着<={args.b_thr:.0f}倍 の基準率 {yB.mean()*100:.2f}%\n")

    drop = ("race_key", "date", "win_odds", "board", "A", "B", "impA", "impB")
    cols = [c for c in rows[0] if c not in drop]
    if args.features == "card":
        cols = [c for c in cols if c not in MODEL_COLS]
    print(f"特徴 {len(cols)}本（{args.features}）\n")

    sA = _walk_forward(rows, cols, "A")
    sB = _walk_forward(rows, cols, "B")
    print(f"=== (A) 波乱の予測  AUC={_auc(yA, sA):.4f} ===")
    _strata(sA, yA, impA, f"A スコア5分位（>={args.a_thr:.0f}倍）")
    print(f"\n=== (B) 人気決着の予測  AUC={_auc(yB, sB):.4f} ===")
    _strata(sB, yB, impB, f"B スコア5分位（<={args.b_thr:.0f}倍）")

    # 単一特徴のベースライン。45特徴モデルが rp_std 単体に負けた前例があるため必ず併記する
    print("\n=== 単一特徴ベースライン（Aの上位/下位半分での 実測 / ratio）===")
    ok = ~np.isnan(sA)
    for c in sorted(cols):
        v = np.array([r[c] for r in rows], dtype=float)
        if np.isnan(v[ok]).any() or len(set(v[ok])) < 5:
            continue
        for sign, tag in ((1, "高"), (-1, "低")):
            m = ok & (sign * v >= np.nanquantile(sign * v[ok], 0.5))
            if m.sum() < 300:
                continue
            r = yA[m].mean() / impA[m].mean()
            if yA[m].mean() / yA[ok].mean() >= 1.10 or r >= 0.95:
                print(f"    {c}{tag}半分 n={m.sum():>5} 実測={yA[m].mean()*100:>5.2f}% "
                      f"lift={yA[m].mean()/yA[ok].mean():>4.2f} ratio={r:>5.3f}")

    _population(rows, sA, sB, yA, yB, impA, ok, len({r["date"] for r in rows}))
    _split_check(rows, sA, sB, yA, impA, ok, args)


if __name__ == "__main__":
    main()
