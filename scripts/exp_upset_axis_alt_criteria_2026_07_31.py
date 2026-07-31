"""波乱予兆レース（拮抗度上位20%・exp_upset50_pattern_train_test_2026_07_31と同一定義）
に限定した場合、「実力上位(1着率+3着内率)」以外のどの基準が軸馬（3着内）選定に
向いているかをTRAINで比較しTESTで再現性を確認する（2026-07-31）。

背景: exp_upset50_axis_roi_2026_07_31.py で「実力上位2頭」軸は波乱予兆セグメントで
3着内率が59.9%/59.5%(TRAIN/TEST)まで低下（全体は71.6%/71.2%）することが判明。
定義上「際立った実力上位馬がいない」レースを拮抗＝波乱予兆と判定しているため、
実力上位軸との相性が悪いのは構造的。実力とは別の予備情報（WT公式印・ライン内
位置・脚質・先行有無）で同セグメント内の軸選定基準を探索する。

候補（単一基準、いずれも1頭選定→3着内率で比較）:
  cand_fr_tr      : 1着率+3着内率 平均上位1頭（ベースライン・旧axis1相当）
  cand_rp         : 競走得点(race_point)上位1頭
  cand_wtmark     : WT公式印 prediction_mark==1（◎）
  cand_lineleader : is_line_leader==1 かつ line_size最大のライン（不在なら該当なし扱い）
  cand_senko      : style=='逃' かつ front_runner過去実績最多（先行選手の中で最有力）

DB書き込みなし・読み取り専用。
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2022-01-01", "2023-12-31"
TEST_FROM, TEST_TO = "2024-01-01", "2026-07-30"


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
            q = ("SELECT race_key, frame_no, race_point, line_group, line_size, line_pos, "
                 "       is_line_leader, n_lines, first_rate, third_rate, finish_order, "
                 "       style, prediction_mark, front_runner "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    return by_race


FEATURES = [
    "rp_max", "rp_std", "rp_gap12",
    "fr_max", "fr_std", "fr_gap12",
    "tr_max", "tr_std", "tr_gap12",
    "n_lines", "max_line_size", "n_solo", "line_entropy",
]


def build_rows(races, entries_by_race):
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

        # --- 軸候補 ---
        by_frame = {int(e["frame_no"]): e for e in ents}

        cand_fr_tr = max(by_frame, key=lambda f: 0.5 * float(by_frame[f]["first_rate"])
                         + 0.5 * float(by_frame[f]["third_rate"]))

        cand_rp = max(by_frame, key=lambda f: float(by_frame[f]["race_point"] or -1))

        wtmark_frames = [f for f in by_frame if by_frame[f]["prediction_mark"] == 1]
        cand_wtmark = wtmark_frames[0] if wtmark_frames else None

        # 最大ラインの先頭(is_line_leader=1)を軸候補に
        biggest_line = max(line_sizes, key=lambda g: line_sizes[g]) if line_sizes else None
        leader_frames = [f for f in by_frame
                          if by_frame[f]["line_group"] == biggest_line
                          and by_frame[f]["is_line_leader"] == 1] if biggest_line else []
        cand_lineleader = leader_frames[0] if leader_frames else None

        senko_frames = [f for f in by_frame if by_frame[f]["style"] == "逃"]
        cand_senko = (max(senko_frames, key=lambda f: by_frame[f]["front_runner"] or 0)
                      if senko_frames else None)

        prelim[rk] = {
            "race_date": race_date, "winners": winners,
            "rp_max": rps[0], "rp_std": float(np.std(rps)), "rp_gap12": rps[0] - rps[1],
            "fr_max": frs[0], "fr_std": float(np.std(frs)), "fr_gap12": frs[0] - frs[1],
            "tr_max": trs[0], "tr_std": float(np.std(trs)), "tr_gap12": trs[0] - trs[1],
            "n_lines": n_lines, "max_line_size": float(max_line_size),
            "n_solo": float(n_solo), "line_entropy": line_entropy,
            "cand_fr_tr": cand_fr_tr, "cand_rp": cand_rp, "cand_wtmark": cand_wtmark,
            "cand_lineleader": cand_lineleader, "cand_senko": cand_senko,
        }
    return prelim


def load_period(date_from, date_to, label):
    print(f"[{label}] loading races {date_from}..{date_to} ...", flush=True)
    races = load_races(date_from, date_to)
    entries = load_entries(list(races.keys()))
    rows = build_rows(races, entries)
    print(f"[{label}]   final rows: {len(rows)}", flush=True)
    return list(rows.values())


CANDIDATES = ["cand_fr_tr", "cand_rp", "cand_wtmark", "cand_lineleader", "cand_senko"]


def recall_report(rows, cand, label):
    valid = [r for r in rows if r[cand] is not None]
    if not valid:
        print(f"  {label} / {cand:16s}: 該当なし")
        return
    hit = sum(1 for r in valid if r[cand] in r["winners"])
    cover = 100.0 * len(valid) / len(rows)
    rate = 100.0 * hit / len(valid)
    print(f"  {label} / {cand:16s}: n={len(valid):6d} (カバー率{cover:5.1f}%)  3着内率={rate:5.2f}%")


def main():
    train_rows = load_period(TRAIN_FROM, TRAIN_TO, "TRAIN")
    test_rows = load_period(TEST_FROM, TEST_TO, "TEST")

    mu = {f: float(np.mean([r[f] for r in train_rows])) for f in FEATURES}
    sd = {f: float(np.std([r[f] for r in train_rows])) or 1.0 for f in FEATURES}
    # is_upsetが無いのでrp_std等の符号は前スクリプトの確定結果をそのまま使う
    sign = {
        "rp_max": 1.0, "rp_std": -1.0, "rp_gap12": -1.0,
        "fr_max": -1.0, "fr_std": -1.0, "fr_gap12": -1.0,
        "tr_max": -1.0, "tr_std": -1.0, "tr_gap12": -1.0,
        "n_lines": -1.0, "max_line_size": -1.0, "n_solo": -1.0, "line_entropy": 1.0,
    }

    def score(row):
        return sum(sign[f] * (row[f] - mu[f]) / sd[f] for f in FEATURES)

    for r in train_rows:
        r["_score"] = score(r)
    for r in test_rows:
        r["_score"] = score(r)

    thr20 = float(np.percentile([r["_score"] for r in train_rows], 80))
    tr_up = [r for r in train_rows if r["_score"] >= thr20]
    te_up = [r for r in test_rows if r["_score"] >= thr20]

    print(f"\n=== 波乱予兆(穴指数上位20%) 軸候補比較: TRAIN n={len(tr_up)} / TEST n={len(te_up)} ===")
    for cand in CANDIDATES:
        recall_report(tr_up, cand, "TRAIN 波乱予兆")
    print()
    for cand in CANDIDATES:
        recall_report(te_up, cand, "TEST  波乱予兆")

    print("\n=== 参考: 全体(絞り込みなし) 軸候補比較 ===")
    for cand in CANDIDATES:
        recall_report(train_rows, cand, "TRAIN 全体")
    print()
    for cand in CANDIDATES:
        recall_report(test_rows, cand, "TEST  全体")


if __name__ == "__main__":
    main()
