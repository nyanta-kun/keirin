"""【読み取り専用】race_type を推奨レース選定（信頼度ゲート）に使えるかの検証
（2026-08-04）。

ユーザー提案:
  「race_type は推奨レース選定における信頼度に使えないか」

背景:
  特徴量として追加した場合の効果は 1位3着内 +0.09〜0.18pt に留まった
  （scripts/exp_racetype_field_ab.py）。理由は race_type 別の生の精度差
  （ガールズ予選 95.2% 〜 初特選 68.4% ＝ 26.8pt）の大半をモデルが既に
  p1 へ織り込んでいるため。**モデルが誤っているのは較正誤差の分だけ**:
      初特選 −6.7pt / 選抜 −6.3pt / 決勝 −5.6pt（過大評価）
      ガールズ予選 +2.1〜+2.2pt（過小評価）
  この「モデルが自分で気づけないズレ」は、特徴量にするより
  **レース選定のゲート**として直接効かせる方が素直。

測定内容:
  ① 現行ランク（7S/7A/7B）の成績を race_type 別に集計
  ② 較正誤差が大きい種別（初特選・選抜・決勝）を deny した場合の効果
  ③ 年次で一貫しているか（多重比較の罠を避けるための最低限の確認）

⚠️ 多重比較に注意: race_type は17種類あり「最悪の種別を除外」すれば
   見かけ上は必ず改善する。本スクリプトは
   **事前に較正誤差という独立した根拠がある3種別のみ**を検証対象とし、
   年次の一貫性も併せて見る。

DB書き込みなし。

使い方:
    python scripts/exp_racetype_rank_gate.py data/exp_7c_cache
"""
import os
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import RANK_7S_AXIS_SUM_MAX, RANK_7S_ENTROPY_MAX

STAKE = 100
# 較正誤差が2SEを超えて過大評価だった種別（exp_axis1_miss_analysis.py の実測）
DENY_CANDIDATES = ("初特選", "選抜", "決勝")


def load(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def fetch_race_type(race_keys: list[str]) -> dict[str, str]:
    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        raise RuntimeError("KEIRIN_DB_URL が未設定です")
    from sqlalchemy import create_engine

    engine = create_engine(db_url)
    rt = pd.read_sql_query(
        "SELECT race_key, race_type FROM keirin.wt_races", engine)
    engine.dispose()
    if rt["race_key"].eq("race_key").any():
        raise RuntimeError("race_type の取得が壊れています")
    return dict(zip(rt["race_key"], rt["race_type"].fillna("不明")))


def order_disagree(c: dict) -> bool | None:
    if c.get("wt_honmei") is None or not c["win_probs"]:
        return None
    return max(c["win_probs"], key=lambda f: c["win_probs"][f]) != c["wt_honmei"]


def trio_bets(c: dict, k: int, drop_ana: bool) -> list[tuple[frozenset, int]]:
    ranked = sorted(c["others"], key=lambda x: -c["top3_probs"][x])
    if drop_ana and c.get("wt_ana") is not None:
        ranked = [x for x in ranked if x != c["wt_ana"]]
    out = []
    for x in ranked[:k]:
        od = c["trio_legs"].get(x)
        if od is not None:
            out.append((frozenset({c["axis1"], c["axis2"], x}), round(od * 100) // 10 * 10))
    return out


def score(cands: list[tuple[dict, int, bool]]) -> dict:
    """cands = [(候補, 買い目点数K, △除外か)]"""
    bet = ret = hit = n = 0
    pays = []
    days = set()
    a1_3 = both3 = 0
    for c, k, drop in cands:
        bs = trio_bets(c, k, drop)
        if not bs:
            continue
        n += 1
        days.add(c["race_date"])
        o = set(c["order3"])
        if c["axis1"] in o:
            a1_3 += 1
        if {c["axis1"], c["axis2"]} <= o:
            both3 += 1
        stake = len(bs) * STAKE
        bet += stake
        got = next((p for key, p in bs if key == frozenset(c["order3"])), 0)
        if got:
            hit += 1
            ret += got
            pays.append(got)
    return {"n": n, "per_day": n / len(days) if days else 0.0,
            "a1": 100.0 * a1_3 / n if n else 0.0,
            "both": 100.0 * both3 / n if n else 0.0,
            "hit": 100.0 * hit / n if n else 0.0,
            "roi": 100.0 * ret / bet if bet else 0.0,
            "med": statistics.median(pays) / 100 if pays else 0.0}


HDR = (f"{'race_type':22} {'n':>6} {'件/日':>6} {'軸1':>6} {'両方':>6} "
       f"{'的中':>6} {'ROI':>7} {'中央値':>7}")


def row(label: str, s: dict) -> str:
    return (f"{label:22} {s['n']:6d} {s['per_day']:6.2f} {s['a1']:5.1f}% "
            f"{s['both']:5.1f}% {s['hit']:5.1f}% {s['roi']:6.1f}% {s['med']:6.1f}倍")


def main() -> None:
    rows = load(Path(sys.argv[1]))
    rt_map = fetch_race_type([c["race_key"] for c in rows])
    for c in rows:
        c["race_type"] = rt_map.get(c["race_key"], "不明")

    ov01 = [c for c in rows if c["wt_overlap_n"] in (0, 1)]
    ranks: dict[str, list[tuple[dict, int, bool]]] = {
        "7S": [(c, 5, False) for c in ov01
               if c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX
               and c["entropy"] <= RANK_7S_ENTROPY_MAX],
        "7A": [(c, 5, False) for c in ov01
               if (c["axis_sum"] > RANK_7S_AXIS_SUM_MAX)
               != (c["entropy"] > RANK_7S_ENTROPY_MAX)],
        "7B": [(c, 3, True) for c in rows
               if c["wt_overlap_n"] == 2 and order_disagree(c) is True],
    }
    days = sorted({c["race_date"] for c in rows})
    print(f"母集団 {len(rows)}件 / {len(days)}日 ({days[0]}〜{days[-1]})")
    print(f"現行ランク: " + " / ".join(f"{k} {len(v)}件" for k, v in ranks.items()))
    print()

    # ---------------------------------------------------------------- ①種別別
    for name, lst in ranks.items():
        print(f"【① {name} の race_type 別成績】")
        print(HDR)
        by_rt: dict[str, list] = defaultdict(list)
        for item in lst:
            by_rt[item[0]["race_type"]].append(item)
        out = [(k, score(v)) for k, v in by_rt.items() if len(v) >= 80]
        out.sort(key=lambda kv: -kv[1]["roi"])
        for k, s in out:
            mark = " ←deny候補" if k in DENY_CANDIDATES else ""
            print(row(k, s) + mark)
        small = sum(len(v) for k, v in by_rt.items() if len(v) < 80)
        print(f"  （n<80 の種別 計 {small}件は省略）")
        print()

    # ---------------------------------------------------------------- ②deny効果
    print("【② 較正誤差が大きい3種別（初特選・選抜・決勝）を deny した場合】")
    print(f"  {'ランク':10} {'案':16} {'n':>6} {'件/日':>6} {'軸1':>6} {'両方':>6} "
          f"{'的中':>6} {'ROI':>7}")
    for name, lst in ranks.items():
        keep = [x for x in lst if x[0]["race_type"] not in DENY_CANDIDATES]
        drop = [x for x in lst if x[0]["race_type"] in DENY_CANDIDATES]
        for label, sub in (("現行(全種別)", lst), ("deny適用後", keep), ("除外分のみ", drop)):
            s = score(sub)
            print(f"  {name:10} {label:16} {s['n']:6d} {s['per_day']:6.2f} "
                  f"{s['a1']:5.1f}% {s['both']:5.1f}% {s['hit']:5.1f}% {s['roi']:6.1f}%")
        print()

    # 3ランク合算
    allr = [x for lst in ranks.values() for x in lst]
    keep = [x for x in allr if x[0]["race_type"] not in DENY_CANDIDATES]
    drop = [x for x in allr if x[0]["race_type"] in DENY_CANDIDATES]
    print("  -- 3ランク合算 --")
    for label, sub in (("現行(全種別)", allr), ("deny適用後", keep), ("除外分のみ", drop)):
        s = score(sub)
        print(f"  {'合算':10} {label:16} {s['n']:6d} {s['per_day']:6.2f} "
              f"{s['a1']:5.1f}% {s['both']:5.1f}% {s['hit']:5.1f}% {s['roi']:6.1f}%")
    print()

    # ---------------------------------------------------------------- ③一貫性
    print("【③ 年次の一貫性（多重比較の罠を避けるための確認）】")
    print(f"  {'年':6} {'区分':14} {'n':>6} {'軸1':>6} {'両方':>6} {'的中':>6} {'ROI':>7}")
    for y in sorted({str(c["race_date"])[:4] for c in rows}):
        for label, sub in (("deny適用後", keep), ("除外分のみ", drop)):
            yl = [x for x in sub if str(x[0]["race_date"]).startswith(y)]
            if not yl:
                continue
            s = score(yl)
            print(f"  {y:6} {label:14} {s['n']:6d} {s['a1']:5.1f}% {s['both']:5.1f}% "
                  f"{s['hit']:5.1f}% {s['roi']:6.1f}%")
    print()

    # ---------------------------------------------------------------- ④全種別の素の軸精度
    print("【④ 参考: 全候補（ランク問わず）の race_type 別 軸1精度と較正誤差】")
    print(f"  {'race_type':22} {'n':>7} {'予測p1':>8} {'実測':>8} {'乖離':>8} {'±2SE':>7}")
    by_rt2: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for c in rows:
        p = c["top3_probs"][c["axis1"]]
        by_rt2[c["race_type"]].append((p, c["axis1"] in set(c["order3"])))
    stat = []
    for k, v in by_rt2.items():
        if len(v) < 300:
            continue
        n = len(v)
        pm = sum(p for p, _ in v) / n
        am = sum(1 for _, h in v if h) / n
        se2 = 2 * (am * (1 - am) / n) ** 0.5
        stat.append((am - pm, k, n, pm, am, se2))
    stat.sort()
    for d, k, n, pm, am, se2 in stat:
        flag = " ★" if abs(d) > se2 else ""
        print(f"  {k:22} {n:7d} {100*pm:7.1f}% {100*am:7.1f}% {100*d:+7.1f}pt "
              f"{100*se2:6.1f}pt{flag}")


if __name__ == "__main__":
    main()
