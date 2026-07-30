"""【券種構造の横断比較】同一の軸2車選定に対し、券種・点数構成を変えてROIを比較する
（2026-07-30・[[keirin_dominance_pattern_verification_2026_07_30]]の続き）。

## 動機（外部AI予想家からの学び）

netkeirin掲載のAI予想家「Aiライン極」を確認したところ、使用データは競走得点・脚質・
AI予測勝利確率・ラインの並びと**我々とほぼ同じ**だが、**券種が2車単3点**だった
（我々は三連複5点）。データや指数ではなく「買い方の構造」が違う。

## 決定的な着眼点

三連複2軸流し5点が的中する条件は「軸2車がともに3着内」であり、これは
**ワイド1点の的中条件と数学的に完全に同一**。にもかかわらず:
  - 三連複5点流し: 500円投資
  - ワイド1点: 100円投資（5分の1）
同じ的中条件で投資額が1/5なら、ワイド配当が三連複配当の1/5を超えていれば
ワイドが構造的に有利。**5倍の金額を払って同じ条件を買っていた可能性がある。**

## 比較する券種・点数構成（すべて同一の軸2車 a1,a2 に対して）

| # | 構成 | 的中条件 | 投資 |
|---|---|---|---|
| 1 | 三連複2軸流し5点（現行） | a1,a2 がともに3着内 | 500 |
| 2 | **ワイド1点 (a1-a2)** | **a1,a2 がともに3着内（#1と同一）** | 100 |
| 3 | 二車複1点 (a1-a2) | a1,a2 が1-2着（順不同） | 100 |
| 4 | 2車単2点 (a1→a2, a2→a1) | a1,a2 が1-2着（順不同） | 200 |
| 5 | 2車単1点 (a1→a2) | その順で1-2着 | 100 |
| 6 | 2車単3点流し (a1→他3頭) | a1が1着 かつ 2着が指定3頭のいずれか | 300 |
| 7 | ワイド2点 (a1-a2, a1-a3) | a1と(a2 or a3)がともに3着内 | 200 |
| 8 | 三連単a1→a2→総流し5点 | a1→a2→残り5頭のいずれか | 500 |

注: wt_odds の quinellaPlace(ワイド) は `minOdds`（レンジ下限・保守的）を格納している
ため、ワイドのROIは**過小評価**側に出る（実際にはこれ以上になる）。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
軸選定は既存検証で的中率上位だった方式を複数使い、券種比較の頑健性を見る。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
STAKE = 100
ORDERED = {"exacta", "trifecta"}


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: str(r["race_date"]) for r in rrows}
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

    print("[load] odds (5券種) ...", flush=True)
    odds = defaultdict(lambda: defaultdict(dict))  # race_key -> bet_type -> key -> odds
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, bt, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    nums = [int(x) for x in re.split(r"[-=→]", str(comb)) if x != ""]
                except ValueError:
                    continue
                key = tuple(nums) if bt in ORDERED else frozenset(nums)
                odds[rk][bt][key] = fv
            if (i // 600) % 20 == 0:
                print(f"[load]   odds progress: {i}/{len(keys)}", flush=True)
    print(f"[load]   odds races: {len(odds)}", flush=True)
    return races, by_race, odds


def build(races, entries_by_race, odds):
    print("[build] ...", flush=True)
    out = []
    for rk, rdate in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_win_pct"] is None or e["pred_top3_pct"] is None for e in ents):
            continue
        od = odds.get(rk)
        if not od:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        order = [fno for _, fno in fin[:3]]   # 1着,2着,3着
        by_frame = {int(e["frame_no"]): e for e in ents}
        frames = list(by_frame.keys())
        wsorted = sorted(frames, key=lambda f: -float(by_frame[f]["pred_win_pct"]))
        tsorted = sorted(frames, key=lambda f: -float(by_frame[f]["pred_top3_pct"]))
        honmei = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 2), None)
        out.append({
            "race_key": rk, "race_date": rdate, "frames": frames, "by_frame": by_frame,
            "odds": od, "order": order, "top3": frozenset(order),
            "top2": frozenset(order[:2]), "exact12": (order[0], order[1]),
            "wsorted": wsorted, "tsorted": tsorted,
            "honmei": honmei, "taikou": taikou,
        })
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


# ---------- 軸選定 ----------
def axis_t1_t2(r):
    return r["tsorted"][0], r["tsorted"][1]


def axis_w1_w2(r):
    return r["wsorted"][0], r["wsorted"][1]


def axis_w1_sameline(r):
    w1 = r["wsorted"][0]
    bf = r["by_frame"]
    lg = bf[w1]["line_group"]
    if lg is None:
        return None
    for f in r["tsorted"]:
        if f != w1 and bf[f]["line_group"] == lg:
            return w1, f
    return None


def axis_s7_new(r):
    h, t = r["honmei"], r["taikou"]
    if h is None or t is None:
        return None
    bf = r["by_frame"]
    a1 = h if float(bf[h]["pred_win_pct"]) >= float(bf[t]["pred_win_pct"]) else t
    for f in r["tsorted"]:
        if f not in (h, t):
            return a1, f
    return None


AXES = [("t1+t2", axis_t1_t2), ("w1+w2", axis_w1_w2),
        ("w1+同ラインtop3", axis_w1_sameline), ("S7新設計", axis_s7_new)]


# ---------- 券種構成 ----------
def bet_trio_flow5(r, a1, a2):
    """三連複2軸+残り5頭流し（現行）。的中=a1,a2ともに3着内"""
    board = r["odds"].get("trio", {})
    others = [f for f in r["frames"] if f not in (a1, a2)]
    combos = {}
    for x in others:
        k = frozenset({a1, a2, x})
        if k in board:
            combos[k] = board[k]
    if not combos:
        return None
    bet = len(combos) * STAKE
    hit = r["top3"] in combos
    pay = int(board[r["top3"]] * STAKE) if hit else 0
    return bet, hit, pay


def bet_wide1(r, a1, a2):
    """ワイド1点。的中=a1,a2ともに3着内（三連複流しと同一条件）"""
    board = r["odds"].get("quinellaPlace", {})
    k = frozenset({a1, a2})
    if k not in board:
        return None
    hit = {a1, a2} <= r["top3"]
    return STAKE, hit, (int(board[k] * STAKE) if hit else 0)


def bet_quinella1(r, a1, a2):
    """二車複1点。的中=a1,a2が1-2着"""
    board = r["odds"].get("quinella", {})
    k = frozenset({a1, a2})
    if k not in board:
        return None
    hit = frozenset({a1, a2}) == r["top2"]
    return STAKE, hit, (int(board[k] * STAKE) if hit else 0)


def bet_exacta2(r, a1, a2):
    """2車単2点（両順）。的中=a1,a2が1-2着"""
    board = r["odds"].get("exacta", {})
    ks = [(a1, a2), (a2, a1)]
    ks = [k for k in ks if k in board]
    if not ks:
        return None
    bet = len(ks) * STAKE
    hit = r["exact12"] in ks
    return bet, hit, (int(board[r["exact12"]] * STAKE) if hit else 0)


def bet_exacta1(r, a1, a2):
    """2車単1点（a1→a2）。的中=その順で1-2着"""
    board = r["odds"].get("exacta", {})
    k = (a1, a2)
    if k not in board:
        return None
    hit = r["exact12"] == k
    return STAKE, hit, (int(board[k] * STAKE) if hit else 0)


def bet_exacta_flow3(r, a1, a2):
    """2車単3点流し a1→(top3順で上位3頭)。外部AI予想家の構成に相当"""
    board = r["odds"].get("exacta", {})
    partners = [f for f in r["tsorted"] if f != a1][:3]
    ks = [(a1, p) for p in partners if (a1, p) in board]
    if not ks:
        return None
    bet = len(ks) * STAKE
    hit = r["exact12"] in ks
    return bet, hit, (int(board[r["exact12"]] * STAKE) if hit else 0)


def bet_wide2(r, a1, a2):
    """ワイド2点 (a1-a2, a1-a3)。a3=top3順でa1,a2以外の最上位"""
    board = r["odds"].get("quinellaPlace", {})
    a3 = next((f for f in r["tsorted"] if f not in (a1, a2)), None)
    ks = [frozenset({a1, a2})]
    if a3 is not None:
        ks.append(frozenset({a1, a3}))
    ks = [k for k in ks if k in board]
    if not ks:
        return None
    bet = len(ks) * STAKE
    hitk = next((k for k in ks if k <= r["top3"]), None)
    return bet, hitk is not None, (int(board[hitk] * STAKE) if hitk else 0)


def bet_trifecta_flow5(r, a1, a2):
    """三連単 a1→a2→残り5頭流し5点"""
    board = r["odds"].get("trifecta", {})
    others = [f for f in r["frames"] if f not in (a1, a2)]
    ks = [(a1, a2, x) for x in others]
    ks = [k for k in ks if k in board]
    if not ks:
        return None
    bet = len(ks) * STAKE
    actual = tuple(r["order"])
    hit = actual in ks
    return bet, hit, (int(board[actual] * STAKE) if hit else 0)


BETS = [
    ("1:三連複5点流し(現行)", bet_trio_flow5),
    ("2:ワイド1点【同条件】", bet_wide1),
    ("3:二車複1点", bet_quinella1),
    ("4:2車単2点(両順)", bet_exacta2),
    ("5:2車単1点", bet_exacta1),
    ("6:2車単3点流し", bet_exacta_flow3),
    ("7:ワイド2点", bet_wide2),
    ("8:三連単5点流し", bet_trifecta_flow5),
]


def evaluate(rows, axis_fn, bet_fn):
    n = hits = 0
    bet_total = pay_total = 0
    for r in rows:
        sel = axis_fn(r)
        if sel is None:
            continue
        a1, a2 = sel
        if a1 == a2:
            continue
        res = bet_fn(r, a1, a2)
        if res is None:
            continue
        bet, hit, pay = res
        n += 1
        bet_total += bet
        pay_total += pay
        hits += 1 if hit else 0
    hitrate = hits / n * 100 if n else 0.0
    roi = pay_total / bet_total * 100 if bet_total else 0.0
    return n, hitrate, roi, bet_total, pay_total


def main():
    races, entries_by_race, odds = load_all()
    rows = build(races, entries_by_race, odds)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}\n")

    for axis_name, axis_fn in AXES:
        print("=" * 100)
        print(f"軸選定: {axis_name}")
        print("=" * 100)
        print(f"  {'券種構成':<26}{'TRAIN n':>9}{'的中%':>8}{'ROI%':>9}"
              f"{'TEST n':>9}{'的中%':>8}{'ROI%':>9}")
        for bet_name, bet_fn in BETS:
            n1, h1, r1, _, _ = evaluate(train, axis_fn, bet_fn)
            n2, h2, r2, _, _ = evaluate(test, axis_fn, bet_fn)
            flag = ""
            if r1 >= 100 and r2 >= 100:
                flag = " ★★両窓100%超"
            elif r2 >= 100:
                flag = " ★TEST100%超"
            elif r1 >= 100:
                flag = " ★TRAIN100%超"
            print(f"  {bet_name:<26}{n1:>9}{h1:>7.1f}%{r1:>8.1f}%"
                  f"{n2:>9}{h2:>7.1f}%{r2:>8.1f}%{flag}")
        print()


if __name__ == "__main__":
    main()
