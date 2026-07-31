"""三連複50倍以上(波乱)の傾向をTRAIN(2024年より前)で抽出し、TESTで再現性を検証する（2026-07-31）。

ユーザー依頼: 「まずは2024年より前のデータから波乱（三連複で50倍以上）したレース傾向、
ライン構成、競走得点分布、1着率、3着内率の構成から波乱が発生した傾向をピックアップ。
同条件で2024年以降のデータで再現性があるか的中率を算出して」

【重要】ここで使う first_rate/third_rate は選手の通算1着率・3着内率（公表値・
wt_entriesの生データ）であり、自社モデルの pred_win_pct/pred_top3_pct とは別物。
公表値は2024年より前でも100%充足しているため、モデルのvintage有無に依存せず
TRAIN/TESTを完全に独立して評価できる（既存の波乱度回帰モデル
exp_payout_expectation_model.py は逆にモデル予測値を特徴量にしており、
2024年以前のTRAINには使えない設計だった）。

honest分割:
  TRAIN = 2022-12-01(データ開始) 〜 2023-12-31（傾向抽出・閾値決定はここのみ）
  TEST  = 2024-01-01 〜 直近（再現性検証。TRAINで決めた閾値をそのまま適用するのみ）

対象: 7車立て・非中止レースで三連複オッズが取得できたもの。
上位馬定義: is_upset = 1 (勝ち三連複オッズ >= 50.0倍)

特徴量（すべてレース開催前に分かる情報のみ）:
  ライン構成: n_lines, max_line_size, n_solo(単騎数), line_entropy
  競走得点分布: rp_max, rp_std, rp_gap12
  1着率構成:   fr_max, fr_std, fr_gap12
  3着内率構成: tr_max, tr_std, tr_gap12

DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2022-01-01", "2023-12-31"
TEST_FROM, TEST_TO = "2024-01-01", "2026-07-30"
UPSET_ODDS = 50.0


def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def load_races(date_from, date_to):
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (date_from, date_to)).fetchall()
    return {r["race_key"]: str(r["race_date"]) for r in rows}


def load_entries(race_keys):
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, race_point, line_group, line_size, n_lines, "
                 "       first_rate, third_rate, finish_order "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    return by_race


def load_trio_win_odds(race_keys, winners_by_race):
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            boards = defaultdict(dict)
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
                    boards[rk][parts] = fv
            for rk in chunk:
                w = winners_by_race.get(rk)
                if w is None:
                    continue
                odds = boards.get(rk, {}).get(w)
                if odds is not None:
                    out[rk] = odds
    return out


def build_rows(races, entries_by_race):
    winners_by_race = {}
    prelim = {}
    for rk, race_date in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["race_point"] is None or e["first_rate"] is None or e["third_rate"] is None
               for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        winners_by_race[rk] = winners

        rps = sorted((float(e["race_point"]) for e in ents), reverse=True)
        frs = sorted((float(e["first_rate"]) for e in ents), reverse=True)
        trs = sorted((float(e["third_rate"]) for e in ents), reverse=True)

        line_sizes = defaultdict(int)
        for e in ents:
            if e["line_group"] is not None:
                line_sizes[e["line_group"]] += 1
        n_lines = float(ents[0]["n_lines"] or len(line_sizes) or 0)
        max_line_size = max(line_sizes.values()) if line_sizes else 0
        n_solo = sum(1 for v in line_sizes.values() if v == 1)
        line_entropy = _entropy(list(line_sizes.values())) if line_sizes else 0.0

        prelim[rk] = {
            "race_date": race_date,
            "rp_max": rps[0], "rp_std": float(np.std(rps)), "rp_gap12": rps[0] - rps[1],
            "fr_max": frs[0], "fr_std": float(np.std(frs)), "fr_gap12": frs[0] - frs[1],
            "tr_max": trs[0], "tr_std": float(np.std(trs)), "tr_gap12": trs[0] - trs[1],
            "n_lines": n_lines, "max_line_size": float(max_line_size),
            "n_solo": float(n_solo), "line_entropy": line_entropy,
        }
    return prelim, winners_by_race


FEATURES = [
    "rp_max", "rp_std", "rp_gap12",
    "fr_max", "fr_std", "fr_gap12",
    "tr_max", "tr_std", "tr_gap12",
    "n_lines", "max_line_size", "n_solo", "line_entropy",
]


def finalize(prelim, winners_by_race, trio_win_odds):
    rows = []
    for rk, feat in prelim.items():
        odds = trio_win_odds.get(rk)
        if odds is None:
            continue
        row = dict(feat)
        row["race_key"] = rk
        row["trio_win_odds"] = odds
        row["is_upset"] = 1 if odds >= UPSET_ODDS else 0
        rows.append(row)
    return rows


def load_period(date_from, date_to, label):
    print(f"[{label}] loading races {date_from}..{date_to} ...", flush=True)
    races = load_races(date_from, date_to)
    print(f"[{label}]   races: {len(races)}", flush=True)
    entries = load_entries(list(races.keys()))
    prelim, winners = build_rows(races, entries)
    print(f"[{label}]   complete-feature races: {len(prelim)}", flush=True)
    trio_odds = load_trio_win_odds(list(prelim.keys()), winners)
    print(f"[{label}]   races with trio win odds: {len(trio_odds)}", flush=True)
    rows = finalize(prelim, winners, trio_odds)
    n_upset = sum(r["is_upset"] for r in rows)
    print(f"[{label}]   final rows: {len(rows)} / upset(>=50x): {n_upset} "
          f"({100.0 * n_upset / len(rows):.2f}%)", flush=True)
    return rows


def decile_report(rows, feature, reverse, label):
    """featureで昇順(またはreverseなら降順)ソートし10分位ごとのupset率を表示。"""
    srows = sorted(rows, key=lambda r: r[feature], reverse=reverse)
    n = len(srows)
    print(f"  -- {label}: {feature} (reverse={reverse}) --")
    for d in range(10):
        lo = n * d // 10
        hi = n * (d + 1) // 10
        chunk = srows[lo:hi]
        if not chunk:
            continue
        rate = 100.0 * sum(r["is_upset"] for r in chunk) / len(chunk)
        print(f"    d{d+1:02d}: n={len(chunk):5d}  upset%={rate:5.2f}")


def corr_with_upset(rows, feature):
    xs = np.array([r[feature] for r in rows])
    ys = np.array([r["is_upset"] for r in rows], dtype=float)
    if np.std(xs) == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def composite_score(row, weights):
    return sum(weights[f] * row[f] for f in weights)


def main():
    train_rows = load_period(TRAIN_FROM, TRAIN_TO, "TRAIN")
    test_rows = load_period(TEST_FROM, TEST_TO, "TEST")

    base_train = 100.0 * sum(r["is_upset"] for r in train_rows) / len(train_rows)
    base_test = 100.0 * sum(r["is_upset"] for r in test_rows) / len(test_rows)
    print(f"\n[baseline] TRAIN upset率={base_train:.2f}%  TEST upset率={base_test:.2f}%")

    print("\n=== 単一特徴量 相関(pointbiserial) TRAIN ===")
    corrs = {}
    for f in FEATURES:
        c = corr_with_upset(train_rows, f)
        corrs[f] = c
        print(f"  {f:15s}: r={c:+.4f}")

    print("\n=== 十分位分析 (TRAINで方向確認 → TEST再現性チェック) ===")
    # 相関の符号で reverse を決める(正の相関なら降順=大きい方がupsetしやすい)
    for f in FEATURES:
        rev = corrs[f] >= 0
        decile_report(train_rows, f, rev, "TRAIN")
        decile_report(test_rows, f, rev, "TEST")
        print()

    # --- 複合スコア: TRAINの相関符号を使い、各特徴を標準化した上で符号付き合算 ---
    print("=== 複合「穴指数」スコア (TRAIN標準化パラメータをTESTにそのまま適用) ===")
    mu = {f: float(np.mean([r[f] for r in train_rows])) for f in FEATURES}
    sd = {f: float(np.std([r[f] for r in train_rows])) or 1.0 for f in FEATURES}
    sign = {f: (1.0 if corrs[f] >= 0 else -1.0) for f in FEATURES}

    def score(row):
        return sum(sign[f] * (row[f] - mu[f]) / sd[f] for f in FEATURES)

    for r in train_rows:
        r["_score"] = score(r)
    for r in test_rows:
        r["_score"] = score(r)

    decile_report(train_rows, "_score", True, "TRAIN複合スコア")
    decile_report(test_rows, "_score", True, "TEST複合スコア")

    # 上位X%閾値をTRAINで決定 → TESTにそのまま適用
    for pct in (10, 20, 25):
        thr = float(np.percentile([r["_score"] for r in train_rows], 100 - pct))
        tr_flag = [r for r in train_rows if r["_score"] >= thr]
        te_flag = [r for r in test_rows if r["_score"] >= thr]
        tr_rate = 100.0 * sum(r["is_upset"] for r in tr_flag) / len(tr_flag) if tr_flag else 0
        te_rate = 100.0 * sum(r["is_upset"] for r in te_flag) / len(te_flag) if te_flag else 0
        print(f"\n[上位{pct}%抽出・閾値はTRAIN上位{pct}%点をTESTに固定適用]")
        print(f"  TRAIN: n={len(tr_flag)} upset率={tr_rate:.2f}% (baseline {base_train:.2f}% "
              f"/ lift {tr_rate / base_train:.2f}x)")
        print(f"  TEST : n={len(te_flag)} upset率={te_rate:.2f}% (baseline {base_test:.2f}% "
              f"/ lift {te_rate / base_test:.2f}x)")


if __name__ == "__main__":
    main()
