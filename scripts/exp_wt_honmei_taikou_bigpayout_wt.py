"""7車立て・三連複30倍以上レースにおけるWINTICKET◎◯の3着内率検証（2026-07-29）。

[[keirin_s7_foundational_rethink_2026_07_29]]。S7根幹見直しの一環。ユーザーの
ターゲット定義: 三連複2車軸総流し(5点=500円)でROI>100%に必要な最低配当ラインは
「25〜30倍以上」（1/4的中でも収支が合う水準）。この「30倍以上に決着する7車立て
レース」という母集団で、WINTICKET公式印◎(honmei)・◯(taikou)それぞれが実際に
3着内に入っている率を検証する。

もし◎◯が高倍率決着でも高い3着内率を維持しているなら「◎◯のどちらかを軸に
使い続けて問題ない」ことを支持し、逆に3着内率が大きく落ちるなら「高配当レースでは
◎◯を軸にする設計そのものが的外れ」という根本的な疑義になる。

対象: n_entries=7 の全レース（S7の他ゲートは一切適用しない・母集団を絞らない）。
「30倍以上」= 実際の勝ち三連複組み合わせのwt_odds odds_valueが30以上。
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-12-31"
PAYOUT_THRESHOLDS = [20, 25, 30, 40, 50]


def load_races_7():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE n_entries = 7").fetchall()
    return {r["race_key"]: str(r["race_date"]) for r in rows}


def load_entries(race_keys):
    out = defaultdict(dict)  # race_key -> frame_no -> {finish_order, prediction_mark}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pm in c.execute(q, chunk):
                out[rk][int(fno)] = {"finish_order": fo, "prediction_mark": pm}
    return out


def load_trio_win_odds(race_keys):
    """各レースの実際の勝ち三連複組み合わせのオッズを返す（finish_orderから決定）。"""
    import re
    out = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
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
                    out.setdefault(rk, {})[parts] = fv
    return out


def main():
    print("7車立てレース読み込み中...")
    races = load_races_7()
    race_keys = list(races.keys())
    print(f"  対象レース数: {len(race_keys)}")

    print("エントリー読み込み中...")
    entries = load_entries(race_keys)

    print("三連複オッズ読み込み中...")
    trio_odds = load_trio_win_odds(race_keys)

    rows = []
    for rk, race_date in races.items():
        ent = entries.get(rk)
        if not ent:
            continue
        fin = [(fo["finish_order"], fno) for fno, fo in ent.items()
               if fo["finish_order"] is not None and fo["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        winners = frozenset(fno for _, fno in fin[:3])
        trio = trio_odds.get(rk)
        if not trio:
            continue
        odds = trio.get(winners)
        if odds is None:
            continue
        honmei = next((fno for fno, v in ent.items() if v["prediction_mark"] == 1), None)
        taikou = next((fno for fno, v in ent.items() if v["prediction_mark"] == 2), None)
        ana = next((fno for fno, v in ent.items() if v["prediction_mark"] == 3), None)
        rows.append({
            "race_key": rk, "race_date": race_date, "trio_odds": odds,
            "honmei_top3": int(honmei in winners) if honmei is not None else None,
            "taikou_top3": int(taikou in winners) if taikou is not None else None,
            "ana_top3": int(ana in winners) if ana is not None else None,
            "honmei_missing": honmei is None, "taikou_missing": taikou is None,
        })

    print(f"\n三連複オッズ+印すべて確認できたレース数: {len(rows)}")

    def rate(rows_, key):
        vals = [r[key] for r in rows_ if r[key] is not None]
        n = len(vals)
        return n, (sum(vals) / n * 100 if n else 0.0)

    print(f"\n{'閾値':<10}{'n':>8}{'◎3着内率':>12}{'◯3着内率':>12}{'△3着内率':>12}"
          f"{'◎◯少なくとも一方3着内':>22}")
    for th in PAYOUT_THRESHOLDS:
        sub = [r for r in rows if r["trio_odds"] >= th]
        n = len(sub)
        n1, r1 = rate(sub, "honmei_top3")
        n2, r2 = rate(sub, "taikou_top3")
        n3, r3 = rate(sub, "ana_top3")
        either = sum(1 for r in sub
                     if (r["honmei_top3"] == 1) or (r["taikou_top3"] == 1))
        either_rate = either / n * 100 if n else 0.0
        print(f">={th}倍{'':<5}{n:>8}{r1:>11.1f}%{r2:>11.1f}%{r3:>11.1f}%{either_rate:>21.1f}%")

    print("\n--- 参考: 全体（オッズ閾値なし）---")
    n_all = len(rows)
    n1, r1 = rate(rows, "honmei_top3")
    n2, r2 = rate(rows, "taikou_top3")
    n3, r3 = rate(rows, "ana_top3")
    either_all = sum(1 for r in rows if (r["honmei_top3"] == 1) or (r["taikou_top3"] == 1))
    print(f"全体      n={n_all:>8} ◎={r1:.1f}% ◯={r2:.1f}% △={r3:.1f}% "
          f"◎◯少なくとも一方={either_all/n_all*100:.1f}%")

    print("\n--- TRAIN/TEST分割（30倍以上のみ）---")
    for label, frm, to in (("TRAIN", TRAIN_FROM, TRAIN_TO), ("TEST", TEST_FROM, TEST_TO)):
        sub = [r for r in rows if r["trio_odds"] >= 30 and frm <= r["race_date"] <= to]
        n = len(sub)
        n1, r1 = rate(sub, "honmei_top3")
        n2, r2 = rate(sub, "taikou_top3")
        either = sum(1 for r in sub if (r["honmei_top3"] == 1) or (r["taikou_top3"] == 1))
        either_rate = either / n * 100 if n else 0.0
        print(f"{label}({frm}〜{to}) 30倍以上: n={n:>6} ◎={r1:.1f}% ◯={r2:.1f}% "
              f"◎◯少なくとも一方={either_rate:.1f}%")


if __name__ == "__main__":
    main()
