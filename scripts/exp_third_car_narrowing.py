"""【3列目の絞り込み可否検証】軸2車は固定し、3列目を5点総流しから減らせるか
（2026-07-30・[[keirin_dominance_pattern_verification_2026_07_30]]の続き）。

## 制約（ユーザー方針・ブランド「二軸探偵」）
- **軸2車の選定は必須**（ズラせない）
- **券種は三連複・三連単のみ**（ワイド/2車単/二車複は対象外）
- 3列目は現在5点総流しだが、レースによって絞り込めれば点数削減は良い

## 検証の核心
現状は軸2車+残り5頭総流し=5点。もし「3列目になりにくい車」を発走前に
見分けられれば、4点/3点/2点に削って投資を減らしROIを改善できる。

ただし構造的なトレードオフがある: pred_top3_pctが最も低い車を切ると、
その車は**最も配当が大きい**組み合わせでもあるため、切ることで
「稀だが大きい的中」を失う。実際にどちらが勝つかは経験的に測るしかない。

## 測定内容
1. **3列目の予測可能性**: 軸2車がともに3着内だったレースで、実際の3着車が
   残り5頭のpred_top3_pct順位（o1..o5）のどこにいたかの分布。
   一様(各20%)なら絞り込み不可、上位に偏るなら絞り込み可能。
2. **点数別ROI**: o1..ok を買う場合（k=1..5）の的中率・ROI・投資額。
   k=5が現行。三連複・三連単の両方で算出。
3. **絞り込み基準の比較**: pred_top3_pct順位以外に
   (a) 軸と同ラインの車を優先 (b) 逃げ・追い込み等の脚質
   で絞った場合も比較する。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
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
                 "       line_group, style, finish_order FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))

    print("[load] odds (trio/trifecta) ...", flush=True)
    odds = defaultdict(lambda: defaultdict(dict))
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
            q = ("SELECT race_key, bet_type, combination, odds_value FROM wt_odds "
                 "WHERE bet_type IN ('trio','trifecta') AND race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
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
        if not od or "trio" not in od:
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        order = [fno for _, fno in fin[:3]]
        by_frame = {int(e["frame_no"]): e for e in ents}
        frames = list(by_frame.keys())
        tsorted = sorted(frames, key=lambda f: -float(by_frame[f]["pred_top3_pct"]))
        honmei = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 1), None)
        taikou = next((int(e["frame_no"]) for e in ents if e["prediction_mark"] == 2), None)
        out.append({
            "race_key": rk, "race_date": rdate, "frames": frames, "by_frame": by_frame,
            "odds": od, "order": order, "top3": frozenset(order),
            "tsorted": tsorted, "honmei": honmei, "taikou": taikou,
        })
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


def axis_t1_t2(r):
    return r["tsorted"][0], r["tsorted"][1]


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


AXES = [("t1+t2", axis_t1_t2), ("S7新設計", axis_s7_new)]


# ---------- 3列目の並べ替え基準 ----------
def rank_by_top3(r, a1, a2):
    """pred_top3_pct降順（既定）"""
    return [f for f in r["tsorted"] if f not in (a1, a2)]


def rank_sameline_first(r, a1, a2):
    """軸と同ラインの車を優先、その中はpred_top3_pct降順"""
    bf = r["by_frame"]
    axis_lines = {bf[a1]["line_group"], bf[a2]["line_group"]} - {None}
    others = [f for f in r["tsorted"] if f not in (a1, a2)]
    same = [f for f in others if bf[f]["line_group"] in axis_lines]
    diff = [f for f in others if bf[f]["line_group"] not in axis_lines]
    return same + diff


def rank_otherline_first(r, a1, a2):
    """軸と別ラインの車を優先（同ライン優先の逆・対照）"""
    bf = r["by_frame"]
    axis_lines = {bf[a1]["line_group"], bf[a2]["line_group"]} - {None}
    others = [f for f in r["tsorted"] if f not in (a1, a2)]
    same = [f for f in others if bf[f]["line_group"] in axis_lines]
    diff = [f for f in others if bf[f]["line_group"] not in axis_lines]
    return diff + same


def rank_senko_first(r, a1, a2):
    """逃げ(先行)脚質を優先、その中はpred_top3_pct降順"""
    bf = r["by_frame"]
    others = [f for f in r["tsorted"] if f not in (a1, a2)]
    senko = [f for f in others if bf[f]["style"] == "逃"]
    rest = [f for f in others if bf[f]["style"] != "逃"]
    return senko + rest


RANKERS = [
    ("A:pred_top3順", rank_by_top3),
    ("B:同ライン優先", rank_sameline_first),
    ("C:別ライン優先", rank_otherline_first),
    ("D:逃げ優先", rank_senko_first),
]


def main():
    races, entries_by_race, odds = load_all()
    rows = build(races, entries_by_race, odds)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}\n")

    # ========== 1. 3列目の予測可能性 ==========
    for axis_name, axis_fn in AXES:
        print("=" * 96)
        print(f"1. 3列目の予測可能性  軸選定={axis_name}")
        print("   （軸2車がともに3着内だったレースで、実際の3着車が残り5頭の")
        print("     pred_top3_pct順位のどこにいたか。一様なら各20%＝絞り込み不可）")
        print("=" * 96)
        for label, data in (("TRAIN", train), ("TEST", test)):
            dist = defaultdict(int)
            n_hit = 0
            pay_by_rank = defaultdict(list)
            for r in data:
                sel = axis_fn(r)
                if sel is None:
                    continue
                a1, a2 = sel
                if a1 == a2 or not ({a1, a2} <= r["top3"]):
                    continue
                third = next(iter(r["top3"] - {a1, a2}), None)
                if third is None:
                    continue
                others = rank_by_top3(r, a1, a2)
                if third not in others:
                    continue
                idx = others.index(third)
                dist[idx] += 1
                n_hit += 1
                pay = r["odds"]["trio"].get(r["top3"])
                if pay:
                    pay_by_rank[idx].append(pay)
            if n_hit == 0:
                continue
            print(f"\n  [{label}] 軸2車ともに3着内だったレース n={n_hit}")
            print(f"    {'3着車の順位':<14}{'件数':>8}{'構成比':>9}{'配当中央値':>12}")
            for i in range(5):
                c = dist.get(i, 0)
                pays = sorted(pay_by_rank.get(i, []))
                med = pays[len(pays) // 2] if pays else 0
                print(f"    o{i+1}{'':<12}{c:>8}{c/n_hit*100:>8.1f}%{med:>11.1f}倍")

    # ========== 2. 点数別ROI（三連複・三連単） ==========
    for axis_name, axis_fn in AXES:
        for bt, btlabel in (("trio", "三連複"), ("trifecta", "三連単")):
            print("\n" + "=" * 96)
            print(f"2. 点数別ROI  軸={axis_name}  券種={btlabel}")
            print("=" * 96)
            for rk_name, rk_fn in RANKERS:
                print(f"\n  --- 3列目の優先順位: {rk_name} ---")
                print(f"    {'点数':<8}{'TRAIN n':>9}{'的中%':>8}{'ROI%':>9}"
                      f"{'TEST n':>9}{'的中%':>8}{'ROI%':>9}")
                for k in (1, 2, 3, 4, 5):
                    res = []
                    for data in (train, test):
                        n = hits = 0
                        bet_total = pay_total = 0
                        for r in data:
                            sel = axis_fn(r)
                            if sel is None:
                                continue
                            a1, a2 = sel
                            if a1 == a2:
                                continue
                            board = r["odds"].get(bt, {})
                            others = rk_fn(r, a1, a2)[:k]
                            if bt == "trio":
                                combos = {}
                                for x in others:
                                    key = frozenset({a1, a2, x})
                                    if key in board:
                                        combos[key] = board[key]
                                if not combos:
                                    continue
                                n += 1
                                bet_total += len(combos) * STAKE
                                if r["top3"] in combos:
                                    hits += 1
                                    pay_total += int(board[r["top3"]] * STAKE)
                            else:
                                # 三連単: a1→a2→x と a2→a1→x の両順（3着は流し）
                                ks = []
                                for x in others:
                                    for t in ((a1, a2, x), (a2, a1, x)):
                                        if t in board:
                                            ks.append(t)
                                if not ks:
                                    continue
                                n += 1
                                bet_total += len(ks) * STAKE
                                actual = tuple(r["order"])
                                if actual in ks:
                                    hits += 1
                                    pay_total += int(board[actual] * STAKE)
                        hr = hits / n * 100 if n else 0.0
                        roi = pay_total / bet_total * 100 if bet_total else 0.0
                        res.append((n, hr, roi))
                    (n1, h1, r1), (n2, h2, r2) = res
                    flag = ""
                    if r1 >= 100 and r2 >= 100:
                        flag = " ★★両窓100%超"
                    elif r2 >= 100:
                        flag = " ★TEST100%超"
                    elif r1 >= 100:
                        flag = " ★TRAIN100%超"
                    tag = f"{k}点" + ("(現行)" if k == 5 and bt == "trio" else "")
                    print(f"    {tag:<8}{n1:>9}{h1:>7.1f}%{r1:>8.1f}%"
                          f"{n2:>9}{h2:>7.1f}%{r2:>8.1f}%{flag}")


if __name__ == "__main__":
    main()
