"""7A の相手を削って点数を減らす条件を掃引窓で探索する（2026-08-06・ユーザー指示）。

## なぜ「相手を削る」なのか

7A（A群 = axis_sum だけ不合格 ＝ 軸2車が堅い）は **的中率が全ランク最高なのに
ROI は最下位**という構造にある。掃引窓の実測で**的中中央値 0.78倍**、
picks_history（旧定義）でも**的中してなお元本割れ（ガミ）が 47.3%**。
つまり 7A のボトルネックは的中率ではなく**配当**であり、的中率をさらに
上げても ROI は動かない。効きうるのは分母（点数）を削る方向。

先例は 7B の △除外（残り5車から WT△ を外して3点）。7B は的中が 20pt 落ちる
代わりに ROI が戻る。7A でも同型が成立するかを見る。

## 構造（設計時に必ず踏まえる）

7A の買い目は 三連複 軸2車 + 残り5車の総流し（5点）。軸2車が両方3着内なら
**必ず1点だけ**当たる。よって

    的中 ⟺ 軸2車が両方3着内 ∧ 3着目の車が残した相手に含まれる

相手を k 車削ると、コストは (5-k)/5 に減り、的中は「削った車が3着目だった」
分だけ落ちる。**削った車の平均回収が 100円/レース（その車の賭け金）を
下回っていれば得**という単純な損得勘定になる。

## 使う条件（オッズ非依存・朝の入稿時点で確定しているもののみ）

オッズは ROI の計測にだけ使い、選択条件には一切使わない
（ユーザー方針: 「オッズは入稿時に正確に判断できない」）。

- `mark`   : WT公式印（1=◎ 2=◯ 3=△ 4=×）。7B の △除外と同型
- `line`   : `wt_entries.line_group`。軸と同一ラインか別ラインか
             （7SS で「軸2車が同一ライン」が効いた。相手側でも見る）

⚠️ モデル確率（p3）による相手の順位づけは**意図的に入れていない**。
   同型の案は既に3回とも確認窓で否定されており（`p3[軸2]` での絞り込み）、
   配分レベルでも「モデル確率での傾斜は有害」と実測済み
   （memory: keirin_stake_allocation_rejected_2026_08_05）。

## 手順の約束

- 本スクリプトは**掃引窓（2025-07〜2026-07）で候補を作るだけ**。
  採否は確認窓（2024-07〜2025-06）で閾値・規則を固定して一度きり検証する。
- **窓別の符号一貫性を必ず見る**（平均は反転を隠す。2026-08-04/05 に複数回踏んだ）。
- **件数の減りも必ず見る**。
- 候補は本番 `build_rows` を通して取得する（判定ロジックを複製しない）。
  モデルは月次凍結 vintage、特徴量は月単位＝ rebuild / live と同じ条件。

DB書き込みなし（読み取りのみ）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_7a_leg_reduction.py
    PYTHONPATH=. .venv/bin/python scripts/exp_7a_leg_reduction.py --from 2024-07-01 --to 2025-06-30
"""
from __future__ import annotations

import argparse
import statistics
import sys

import numpy as np
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_wt_candidate_cache import month_candidates  # noqa: E402
from src.database import get_connection  # noqa: E402
from src.strategy_wt import (  # noqa: E402
    RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX,
)
from src.wt_vintage_config import bad_model_name, monthly_windows  # noqa: E402

SWEEP_FROM, SWEEP_TO = "2025-07-01", "2026-07-31"
STAKE = 100

# 掃引窓を4つの部分窓へ分ける（符号一貫性の確認用）。exp_axis_rule_decomposition
# の w1〜w4 と厳密に同じ境界ではない（あちらは日付境界・こちらは月境界）。
def subwindow(month: str) -> str:
    if month >= "2026-04":
        return "s4"
    if month >= "2026-01":
        return "s3"
    if month >= "2025-10":
        return "s2"
    return "s1"


def is_7a(c: dict) -> bool:
    """本番 rank_7a_daily_select と同じ条件（A群 = axis_sum だけ不合格）。"""
    return (c["wt_overlap_n"] in (0, 1)
            and c["axis_sum"] > RANK_7S_AXIS_SUM_MAX
            and c["entropy"] <= RANK_7S_ENTROPY_MAX)


def load_marks_and_lines(race_keys: list[str]) -> tuple[dict, dict]:
    """wt_entries から prediction_mark と line_group を引く（特徴量構築は不要）。"""
    marks: dict[str, dict[int, int]] = defaultdict(dict)
    lines: dict[str, dict[int, str]] = defaultdict(dict)
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, prediction_mark, line_group "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, mk, lg in c.execute(q, chunk):
                if mk is not None:
                    marks[rk][int(fno)] = int(mk)
                lines[rk][int(fno)] = None if lg is None else str(lg).strip()
    return marks, lines


# ---- 相手の削り方（レース → 残す相手のリスト）------------------------------
# いずれも「削れないときは総流しのまま」＝推奨を壊さない側に倒す。

def keep_all(c, mk, lg):
    return list(c["others"])


def _drop_by_mark(c, mk, target):
    drop = [f for f in c["others"] if mk.get(f) == target]
    keep = [f for f in c["others"] if f not in drop]
    return keep if keep else list(c["others"])


def drop_ana(c, mk, lg):        # △（WT印3位）を除外。7B と同型
    return _drop_by_mark(c, mk, 3)


def drop_batsu(c, mk, lg):      # ×（WT印4位）を除外
    return _drop_by_mark(c, mk, 4)


def drop_ana_batsu(c, mk, lg):  # △と×の両方を除外
    drop = {f for f in c["others"] if mk.get(f) in (3, 4)}
    keep = [f for f in c["others"] if f not in drop]
    return keep if keep else list(c["others"])


def drop_unmarked(c, mk, lg):   # 無印（◎◯△×のどれでもない）を除外
    keep = [f for f in c["others"] if mk.get(f) in (1, 2, 3, 4)]
    return keep if keep else list(c["others"])


def _axis_lines(c, lg):
    return {lg.get(c["axis1"]), lg.get(c["axis2"])} - {None, ""}


def drop_same_line(c, mk, lg):  # 軸と同じラインの相手を除外
    al = _axis_lines(c, lg)
    keep = [f for f in c["others"] if lg.get(f) not in al or not al]
    return keep if keep else list(c["others"])


def drop_other_line(c, mk, lg):  # 軸と別ラインの相手を除外（＝同ラインだけ残す）
    al = _axis_lines(c, lg)
    keep = [f for f in c["others"] if al and lg.get(f) in al]
    return keep if keep else list(c["others"])


RULES = [
    ("総流し5点(基準)", keep_all),
    ("△除外", drop_ana),
    ("×除外", drop_batsu),
    ("△×除外", drop_ana_batsu),
    ("無印除外", drop_unmarked),
    ("軸と同ラインの相手を除外", drop_same_line),
    ("軸と別ラインの相手を除外", drop_other_line),
]


def settle(c: dict, keep: list[int]):
    a1, a2 = c["axis1"], c["axis2"]
    combos = [frozenset({a1, a2, x}) for x in keep
              if frozenset({a1, a2, x}) in c["trio"]]
    if not combos:
        return None
    hit = c["actual_top3"] in combos
    pay = (round(c["trio"][c["actual_top3"]] * 100) // 10 * 10) if hit else 0
    return len(combos) * STAKE, pay, hit


def agg(rows: list):
    rows = [r for r in rows if r]
    if not rows:
        return None
    bet = sum(r[0] for r in rows)
    pay = sum(r[1] for r in rows)
    hits = [r for r in rows if r[2]]
    return dict(
        n=len(rows), pts=bet / len(rows) / STAKE,
        hit=100 * len(hits) / len(rows),
        roi=100 * pay / bet if bet else 0.0,
        med=statistics.median([r[1] / r[0] for r in hits]) if hits else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default=SWEEP_FROM)
    ap.add_argument("--to", dest="date_to", default=SWEEP_TO)
    args = ap.parse_args()

    windows = [w for w in monthly_windows()
               if w[0] >= args.date_from and w[1] <= args.date_to]
    print(f"掃引窓 {args.date_from}〜{args.date_to}  {len(windows)}ヶ月", flush=True)

    races: list[tuple[str, dict]] = []   # (subwindow, candidate)
    for date_from, date_to, eval_model, win_model in windows:
        cands = month_candidates(date_from, date_to, eval_model, win_model,
                                 bad_model_name(eval_model))
        sel = [c for c in cands if is_7a(c)]
        races.extend((subwindow(date_from[:7]), c) for c in sel)
        print(f"      → 7A {len(sel):4}", flush=True)

    keys = sorted({c["race_key"] for _, c in races})
    marks, lines = load_marks_and_lines(keys)
    print(f"\n7A母集団: {len(races)} レース\n")

    subs = sorted({s for s, _ in races})
    header = (f"{'規則':<26}{'n':>5}{'点数':>6}{'的中':>8}{'ROI':>8}{'中央値':>8}   "
              + "  ".join(f"{s}" for s in subs))
    print(header)
    print("-" * len(header))
    base_roi = None
    for name, rule in RULES:
        rows = [settle(c, rule(c, marks.get(c["race_key"], {}),
                               lines.get(c["race_key"], {}))) for _, c in races]
        a = agg(rows)
        if not a:
            continue
        per_sub = []
        for s in subs:
            sr = [settle(c, rule(c, marks.get(c["race_key"], {}),
                                 lines.get(c["race_key"], {})))
                  for w, c in races if w == s]
            sa = agg(sr)
            per_sub.append(f"{sa['roi']:5.1f}" if sa else "  —  ")
        if base_roi is None:
            base_roi = a["roi"]
        delta = a["roi"] - base_roi
        flag = "✓" if all(float(x) >= 75.0 for x in per_sub if x.strip() != "—") else " "
        print(f"{name:<26}{a['n']:>5}{a['pts']:>6.2f}{a['hit']:>7.1f}%"
              f"{a['roi']:>7.1f}%{a['med']:>7.2f}倍   "
              + "  ".join(per_sub) + f" {flag} ({delta:+.1f}pt)")

    print("\n✓ = 4部分窓すべてで ROI>=75%（控除率の壁）。"
          "\n⚠️ ここは掃引窓。採否は確認窓（2024-07〜2025-06）で規則を固定して一度きり検証する。")

    # ---- paired bootstrap と裾依存 -----------------------------------------
    # 「4窓すべてで改善」は有意性の代わりにならない（2026-08-05 の 7B で実証済み。
    # 4/4改善かつ掃引窓から縮まなかった候補が +8.2pt [−2.1,+19.1] で有意差なしだった）。
    # また改善の正体が少数の高配当なら人工物なので、上位k本を除いた ROI も見る。
    rng = np.random.default_rng(20260806)
    base_rows = [settle(c, keep_all(c, marks.get(c["race_key"], {}),
                                    lines.get(c["race_key"], {}))) for _, c in races]

    print(f"\n{'規則':<26}{'Δ ROI (基準比)':>22}{'除・上5':>9}{'除・上10':>10}{'上5シェア':>10}")
    print("-" * 78)
    for name, rule in RULES[1:]:
        rule_rows = [settle(c, rule(c, marks.get(c["race_key"], {}),
                                    lines.get(c["race_key"], {}))) for _, c in races]
        pairs = [(b, r) for b, r in zip(base_rows, rule_rows) if b and r]
        n = len(pairs)
        idx = rng.integers(0, n, size=(2000, n))
        diffs = []
        bb = np.array([[p[0][0], p[0][1]] for p in pairs], dtype=float)
        rr = np.array([[p[1][0], p[1][1]] for p in pairs], dtype=float)
        for row in idx:
            b_s, r_s = bb[row], rr[row]
            diffs.append(100 * (r_s[:, 1].sum() / r_s[:, 0].sum()
                                - b_s[:, 1].sum() / b_s[:, 0].sum()))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        point = float(np.mean(diffs))

        # 裾依存: 払戻の大きい順に上位k本を除いた ROI
        rows_ok = [r for r in rule_rows if r]
        bet_all = sum(r[0] for r in rows_ok)
        pay_all = sum(r[1] for r in rows_ok)
        top = sorted(rows_ok, key=lambda r: -r[1])
        def ex(k):
            b = bet_all - sum(r[0] for r in top[:k])
            p = pay_all - sum(r[1] for r in top[:k])
            return 100 * p / b if b else 0.0
        share5 = 100 * sum(r[1] for r in top[:5]) / pay_all if pay_all else 0.0
        sig = "有意" if lo > 0 else ("　　" if hi > 0 else "有意(悪)")
        print(f"{name:<26}{point:+7.1f} [{lo:+6.1f},{hi:+6.1f}] {sig}"
              f"{ex(5):>9.1f}{ex(10):>10.1f}{share5:>9.1f}%")

    b_ok = [r for r in base_rows if r]
    b_bet = sum(r[0] for r in b_ok); b_pay = sum(r[1] for r in b_ok)
    b_top = sorted(b_ok, key=lambda r: -r[1])
    def bex(k):
        return 100 * (b_pay - sum(r[1] for r in b_top[:k])) / (b_bet - sum(r[0] for r in b_top[:k]))
    print(f"{'（基準・総流し5点）':<26}{'—':>22}{bex(5):>9.1f}{bex(10):>10.1f}"
          f"{100*sum(r[1] for r in b_top[:5])/b_pay:>9.1f}%")


if __name__ == "__main__":
    main()
