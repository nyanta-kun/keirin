"""ライン連携実績の事後セグメントフィルタ・ブラインド検証（2026-07-29）。

[[keirin_roi_validation_crisis_2026_07_29]] の残タスク。`exp_elo_linecoop_wt.py`で
「ライン連携実績(lc_n_prev/lc_rate/lc_max_n)」はモデル入力特徴としては採否未確定
（Elo同様に一緒に検証されたのみで単独結論なし）のまま残っていた。本スクリプトは
race_typeの検証と同じ設計で、S7本番(picks_history rank='RANK_7S', D構成=現行本番
定義そのもの・527件)の軸2車(axis1/axis2)について「過去に同ラインを組んだ実績」を
事後セグメントフィルタとして評価する。

手順:
1. wt_entries+wt_races全履歴(2022-12-01〜)を時系列に走査し、同ラインペアの
   「回数(pair_n)」「両者3着内率(pair_hit/pair_n)」をpoint-in-timeで累積する
   （exp_elo_linecoop_wt.pyのcompute_elo_and_linecoopと同じ設計・Elo部分は不要なので省略）
2. S7本番pick(527件)それぞれについて、pred_comboから軸1/軸2のframe_noを取り出し、
   そのレース"時点"でのpair_n/pair_hit、および今回のレースで同ラインを組んでいるか
   (same_line)をマージする
3. TRAIN(2024-01-01〜2025-12-31)でセグメント別ROIを集計し、n>=20かつROI>100%の
   ものを選定 → TEST(2026-01-01〜2026-07-18、選定に一切使っていない期間)で
   一度だけ評価する

本番コード・モデルは一切変更しない（picks_historyの実績を読むだけ）。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

WARMUP_FROM = "2022-12-01"
TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"
MIN_N = 20

COMBO_RE = re.compile(r"^(\d+)=(\d+)-")


def load_s7_picks():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, race_date, pred_combo, hit, payout, bet_amount "
            "FROM picks_history WHERE rank = 'RANK_7S'").fetchall()
    picks = {}
    for r in rows:
        m = COMBO_RE.match(r["pred_combo"] or "")
        if not m:
            continue
        rk = r["race_key"].split("#")[0]
        picks[rk] = {
            "race_date": str(r["race_date"]),
            "axis1": int(m.group(1)),
            "axis2": int(m.group(2)),
            "hit": int(r["hit"]),
            "payout": int(r["payout"] or 0),
            "bet_amount": int(r["bet_amount"] or 0),
        }
    return picks


def load_entries():
    with get_connection() as c:
        rows = c.execute(
            "SELECT e.race_key, r.race_date, r.start_at, e.frame_no, e.player_id, "
            "e.line_group, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
            "WHERE r.race_date >= :from_date ORDER BY r.race_date, r.start_at, e.race_key",
            {"from_date": WARMUP_FROM}).fetchall()
    return rows


def main():
    print("S7 pick読み込み中...")
    picks = load_s7_picks()
    print(f"  S7 picks (pred_combo解析成功): {len(picks)}件")

    print("全履歴のライン構成読み込み中(2022-12-01〜)...")
    rows = load_entries()
    print(f"  entries行数: {len(rows)}")

    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_key"]].append(r)
    # race_key の時系列順（既にSQLでrace_date,start_at順に並んでいるので出現順を維持）
    race_order = list(dict.fromkeys(r["race_key"] for r in rows))

    pair_n: dict = defaultdict(int)
    pair_hit: dict = defaultdict(int)
    results = []

    for rk in race_order:
        g = by_race[rk]
        pick = picks.get(rk)
        if pick is not None:
            frame_to_player = {int(e["frame_no"]): e["player_id"] for e in g}
            frame_to_line = {int(e["frame_no"]): e["line_group"] for e in g}
            a1p = frame_to_player.get(pick["axis1"])
            a2p = frame_to_player.get(pick["axis2"])
            a1l = frame_to_line.get(pick["axis1"])
            a2l = frame_to_line.get(pick["axis2"])
            if a1p is not None and a2p is not None:
                key = (a1p, a2p) if a1p < a2p else (a2p, a1p)
                n_prev = pair_n[key]
                rate = (pair_hit[key] / n_prev) if n_prev > 0 else 0.0
                same_line = bool(a1l is not None and a1l == a2l)
                results.append({
                    "race_key": rk, "race_date": pick["race_date"],
                    "hit": pick["hit"], "payout": pick["payout"],
                    "bet_amount": pick["bet_amount"],
                    "lc_n_prev": n_prev, "lc_rate": rate, "same_line": same_line,
                })

        # 更新（このレースの同ラインペア実績を反映）
        pids = [e["player_id"] for e in g]
        lines = [e["line_group"] for e in g]
        fins = [e["finish_order"] for e in g]
        fin_map = dict(zip(pids, fins))
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if lines[i] is None or lines[i] != lines[j]:
                    continue
                key = (pids[i], pids[j]) if pids[i] < pids[j] else (pids[j], pids[i])
                pair_n[key] += 1
                fi, fj = fin_map[pids[i]], fin_map[pids[j]]
                if fi is not None and fj is not None and 1 <= fi <= 3 and 1 <= fj <= 3:
                    pair_hit[key] += 1

    print(f"\nマージ完了: S7 picks中 {len(results)}件 にライン連携特徴を付与"
          f"（picks全{len(picks)}件との差分はentries不整合等）")

    def bucket(r):
        # 過去の同ライン実績が「ある(>=1)」か「ない(0)」か、かつ今回同ラインを組むか
        prev = "実績あり" if r["lc_n_prev"] >= 1 else "実績なし"
        cur = "今回同ライン" if r["same_line"] else "今回別ライン"
        return f"{cur}×{prev}"

    def summarize(cands):
        n = len(cands)
        hits = sum(c["hit"] for c in cands)
        bet = sum(c["bet_amount"] for c in cands)
        pay = sum(c["payout"] for c in cands)
        roi = pay / bet * 100 if bet else 0.0
        hitrate = hits / n * 100 if n else 0.0
        return n, hits, hitrate, bet, pay, roi

    train = [r for r in results if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in results if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\nTRAIN({TRAIN_FROM}〜{TRAIN_TO}): {len(train)}件 / "
          f"TEST({TEST_FROM}〜): {len(test)}件")

    n_a, hits_a, hr_a, bet_a, pay_a, roi_a = summarize(train)
    print(f"[TRAIN] 全体ROI: n={n_a} hit={hr_a:.1f}% ROI={roi_a:.1f}%")

    by_seg_train = defaultdict(list)
    for r in train:
        by_seg_train[bucket(r)].append(r)

    print(f"\n[TRAIN] セグメント別ROI (n>={MIN_N}のみ表示)")
    print(f"{'segment':<24}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    train_rows = []
    for seg, cs in by_seg_train.items():
        n, hits, hitrate, bet, pay, roi = summarize(cs)
        if n >= MIN_N:
            train_rows.append((seg, n, hitrate, bet, pay, roi))
    train_rows.sort(key=lambda r: -r[5])
    for seg, n, hitrate, bet, pay, roi in train_rows:
        mark = " ★100%超" if roi > 100 else ""
        print(f"{seg:<24}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")

    selected = [seg for seg, n, hitrate, bet, pay, roi in train_rows if roi > 100]
    print(f"\n[選定] TRAINでROI>100%だったセグメント: {selected if selected else '(なし)'}")

    if not selected:
        print("\n判定: TRAINの時点で n>=20 かつ ROI>100% のセグメントが一つもない。"
              "\n      ライン連携実績の事後セグメントフィルタはこの粒度では見込みなし。")
        return

    by_seg_test = defaultdict(list)
    for r in test:
        by_seg_test[bucket(r)].append(r)

    print(f"\n[TEST] TRAINで選定したセグメントのみ評価（一度きり・{TEST_FROM}〜）")
    print(f"{'segment':<24}{'n':>8}{'hit%':>8}{'bet':>12}{'payout':>12}{'ROI':>10}")
    n_survive = 0
    for seg in selected:
        cs = by_seg_test.get(seg, [])
        n, hits, hitrate, bet, pay, roi = summarize(cs)
        mark = " ★100%超(再現)" if roi > 100 and n > 0 else ""
        if roi > 100 and n > 0:
            n_survive += 1
        print(f"{seg:<24}{n:>8}{hitrate:>7.1f}%{bet:>12,}{pay:>12,}{roi:>9.1f}%{mark}")

    n_t, hits_t, hr_t, bet_t, pay_t, roi_t = summarize(test)
    print(f"\n[参考] TEST期間 全体ROI: n={n_t} hit={hr_t:.1f}% ROI={roi_t:.1f}%")

    print(f"\n判定: TRAINで選定した{len(selected)}セグメント中、TESTでもROI>100%を"
          f"維持したのは{n_survive}件。")


if __name__ == "__main__":
    main()
