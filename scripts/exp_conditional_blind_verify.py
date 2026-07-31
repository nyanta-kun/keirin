"""条件別勝率(周長/時間帯/グレード)の事後セグメントフィルタ・ブラインド検証（2026-07-29）。

[[keirin_roi_validation_crisis_2026_07_29]] の残タスク。`exp_conditional_wr_wt.py`で
周長別/時間帯別/グレード別 選手勝率は「モデル入力特徴としては符号不一致で不採用」との
判定のみ残っており、race_type同様「事後セグメントフィルタとしての検証」は未実施だった。

race_type・ライン連携と同じ設計で、S7本番pick(picks_history rank='RANK_7S'・D構成
そのもの・527件)を下記3軸それぞれで単純にセグメント分解し、TRAINで有望セグメントを
選定→TESTで一度だけ評価するブラインド検証を行う（選手個人のpoint-in-time条件別勝率
ではなく、レース自体の条件によるセグメント分解である点に注意。特徴量検証時は選手×条件の
複雑な計算だったが、事後フィルタとしてはレース自体の属性で十分単純に切れる）。

  - 周長  : venue_info.bank_length（250/333/400/500）
  - 時間帯: wt_races.start_at → JST 17時以降を night
  - グレード: wt_races.grade（S級/A級/L級/SA混合）

本番コード・モデルは一切変更しない（picks_historyの実績を読むだけ）。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone, timedelta

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"
MIN_N = 20
JST = timezone(timedelta(hours=9))


def load_s7_picks():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, race_date, hit, payout, bet_amount "
            "FROM picks_history WHERE rank = 'RANK_7S'").fetchall()
    picks = {}
    for r in rows:
        rk = r["race_key"].split("#")[0]
        picks[rk] = {
            "race_date": str(r["race_date"]), "hit": int(r["hit"]),
            "payout": int(r["payout"] or 0), "bet_amount": int(r["bet_amount"] or 0),
        }
    return picks


def load_race_conditions(race_keys):
    with get_connection() as c:
        q = ("SELECT r.race_key, r.grade, r.start_at, v.bank_length "
             "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
             "WHERE r.race_key IN (%s)" % ",".join("?" * len(race_keys)))
        rows = c.execute(q, race_keys).fetchall()
    out = {}
    for r in rows:
        st = r["start_at"]
        is_night = None
        if st is not None:
            try:
                dt = datetime.fromtimestamp(int(st), tz=timezone.utc).astimezone(JST)
                is_night = dt.hour >= 17
            except (TypeError, ValueError):
                pass
        out[r["race_key"]] = {
            "grade": r["grade"], "bank_length": r["bank_length"], "is_night": is_night,
        }
    return out


def summarize(cands):
    n = len(cands)
    hits = sum(c["hit"] for c in cands)
    bet = sum(c["bet_amount"] for c in cands)
    pay = sum(c["payout"] for c in cands)
    roi = pay / bet * 100 if bet else 0.0
    hitrate = hits / n * 100 if n else 0.0
    return n, hits, hitrate, bet, pay, roi


def analyze_axis(axis_name, key_fn, train, test):
    print(f"\n===== 軸: {axis_name} =====")
    by_seg_train = defaultdict(list)
    for r in train:
        by_seg_train[key_fn(r)].append(r)

    print(f"[TRAIN] {axis_name}別ROI (n>={MIN_N}のみ表示)")
    print(f"{'segment':<16}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    rows = []
    for seg, cs in by_seg_train.items():
        n, hits, hitrate, bet, pay, roi = summarize(cs)
        if n >= MIN_N:
            rows.append((seg, n, hitrate, bet, pay, roi))
    rows.sort(key=lambda r: -r[5])
    for seg, n, hitrate, bet, pay, roi in rows:
        mark = " ★100%超" if roi > 100 else ""
        print(f"{str(seg):<16}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")

    selected = [seg for seg, n, hitrate, bet, pay, roi in rows if roi > 100]
    print(f"[選定] TRAINでROI>100%: {selected if selected else '(なし)'}")
    if not selected:
        return

    by_seg_test = defaultdict(list)
    for r in test:
        by_seg_test[key_fn(r)].append(r)

    print(f"[TEST] 選定セグメントのみ評価（一度きり）")
    print(f"{'segment':<16}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    n_survive = 0
    for seg in selected:
        cs = by_seg_test.get(seg, [])
        n, hits, hitrate, bet, pay, roi = summarize(cs)
        mark = " ★100%超(再現)" if roi > 100 and n > 0 else ""
        if roi > 100 and n > 0:
            n_survive += 1
        print(f"{str(seg):<16}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")
    print(f"判定: {len(selected)}セグメント中、TESTでもROI>100%を維持したのは{n_survive}件。")


def main():
    print("S7 pick読み込み中...")
    picks = load_s7_picks()
    print(f"  S7 picks: {len(picks)}件")

    conds = load_race_conditions(list(picks.keys()))
    print(f"  条件情報が引けた件数: {len(conds)}件")

    merged = []
    for rk, p in picks.items():
        c = conds.get(rk)
        if c is None:
            continue
        merged.append({**p, "race_key": rk, **c})

    train = [r for r in merged if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in merged if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\nTRAIN({TRAIN_FROM}〜{TRAIN_TO}): {len(train)}件 / TEST({TEST_FROM}〜): {len(test)}件")

    n_a, hits_a, hr_a, bet_a, pay_a, roi_a = summarize(train)
    print(f"[TRAIN] 全体ROI: n={n_a} hit={hr_a:.1f}% ROI={roi_a:.1f}%")
    n_b, hits_b, hr_b, bet_b, pay_b, roi_b = summarize(test)
    print(f"[TEST]  全体ROI: n={n_b} hit={hr_b:.1f}% ROI={roi_b:.1f}%")

    analyze_axis("周長(bank_length)", lambda r: r["bank_length"], train, test)
    analyze_axis("時間帯(is_night)", lambda r: "night" if r["is_night"] else "day", train, test)
    analyze_axis("グレード(grade)", lambda r: r["grade"], train, test)


if __name__ == "__main__":
    main()
