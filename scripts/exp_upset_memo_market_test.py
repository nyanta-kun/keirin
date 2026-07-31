"""【堅調/波乱・穴メモの仮説を市場ミスプライシングとして検証】（2026-07-30）。

`inputs/競輪_堅調波乱_穴予測_調査メモ.md` の各仮説を、
[[keirin_pair_correlation_mispricing_2026_07_30]]で確立した枠組みで検証する。

## この検証の考え方（既存検証との決定的な違い）

同メモの仮説の多くは既に「**我々のモデルの特徴量として有効か**」で検証済み
（例: same-meet form は `exp_samemeet_form_wt.py`・day_index は `exp_day_index_wt.py`・
風は G06・みなし直線は netkeirin 調査）。しかしメモ4章の核心主張は違う:

    「初日に良い動きを見せても人気に反映されるのは翌日以降。
      そのタイムラグを突く」

これは「我々のモデルの精度が上がるか」ではなく「**市場が織り込むのが遅いか**」
という主張であり、まったく別の問いである。我々のAUCが上がらなくても
市場が遅ければ妙味は存在しうる。

[[keirin_segment_market_edge_closure_2026_07_30]]で「我々のモデルは市場に勝てない」
ことが確定した以上、**市場が構造的に遅い/歪んでいる領域**を探すのが唯一の道であり、
ラグ仮説はまさにその型。

## 測る指標

車単位:
  market_marg(i) = Σ_{iを含む三連複} 正規化(0.75/odds)   ← 市場の3着内確率
  ratio = 実測3着内率 ÷ market_marg                      ← >1なら市場が過小評価

ペア単位（S7型5点流しのROIに直結）:
  ROI = 0.75 × [実測ペア的中率 ÷ 市場ペア確率]  →  ROI100%超には ratio ≥ 1.333

## 検証する仮説（メモ対応）

A. 【最重要・ラグ仮説】同開催内の調子 × 開催日
   同一cup_id内で strictly 前日までの実績を point-in-time 集計
   （初日未走 / 今節1着あり / 今節3着内あり / 今節不振）× day_index
   → メモ4章「初日の好走が人気化するのは翌日以降」

B. 競輪場別（メモ2章の「荒れやすい競輪場」リストを自社データで再検証）
   荒れ度（勝ち三連複配当の中央値・30倍以上率）と 市場ミスプライシング の両方

C. 車番（メモ2章「4/6/8番車が絡むと高配当」※7車立てなので4/6が対象）

D. 級班 × 脚質（メモ4章「下位クラスでは追込を軸に」）

E. 競走得点トレンド（メモ4章・5章「調子の変化を明示的にモデル化」）
   前走時からの得点変化の符号

F. バンク周長（メモ2章「333mは荒れやすい」）

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import re
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_TO = "2025-12-31"
TAKEOUT_RETURN = 0.75
MIN_BOARD = 33
MIN_RIDERS = 1500
MIN_PAIRS = 800
ROI_BREAKEVEN = 1.0 / TAKEOUT_RETURN


def load_all():
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date, grade, cup_id, day_index, venue_id "
            "FROM wt_races WHERE n_entries = 7 AND cancel = 0 "
            "AND race_date BETWEEN ? AND ?", ("2024-01-01", "2026-07-30")).fetchall()
        venues = {str(r["venue_code"]): {"name": r["name"], "bank": r["bank_length"]}
                  for r in c.execute("SELECT venue_code, name, bank_length FROM venue_info").fetchall()}
    races = {}
    for r in rrows:
        races[r["race_key"]] = {
            "race_date": str(r["race_date"]), "grade": str(r["grade"] or "?"),
            "cup_id": r["cup_id"], "day_index": r["day_index"],
            "venue_id": str(r["venue_id"]),
        }
    print(f"[load] races: {len(races)}  venues: {len(venues)}", flush=True)

    keys = list(races)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, player_id, pred_top3_pct, player_class, "
                 "       style, race_point, line_group, line_pos, line_size, finish_order "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load] entries: {sum(len(v) for v in by_race.values())}", flush=True)
    return races, by_race, venues


def load_trio_odds(race_keys):
    out = {}
    keys = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
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
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if len(set(parts)) != 3:
                    continue
                m = 0
                for p in parts:
                    m |= 1 << p
                out.setdefault(rk, {})[m] = fv
    return out


def build_samemeet(races, entries_by_race):
    """同一cup_id内で strictly 前の day_index までの実績を point-in-time 集計。"""
    # (cup_id, player_id) -> [(day_index, finish_order), ...]
    hist = defaultdict(list)
    for rk, ents in entries_by_race.items():
        meta = races.get(rk)
        if not meta or meta["cup_id"] is None or meta["day_index"] is None:
            continue
        for e in ents:
            fo = e["finish_order"]
            if fo is None or int(fo) < 1:
                continue
            hist[(meta["cup_id"], e["player_id"])].append((int(meta["day_index"]), int(fo)))

    sm = {}   # (race_key, player_id) -> label
    for rk, ents in entries_by_race.items():
        meta = races.get(rk)
        if not meta or meta["cup_id"] is None or meta["day_index"] is None:
            continue
        di = int(meta["day_index"])
        for e in ents:
            prev = [fo for d, fo in hist.get((meta["cup_id"], e["player_id"]), []) if d < di]
            if not prev:
                lab = "今節未走(初日等)"
            elif any(f == 1 for f in prev):
                lab = "今節1着あり"
            elif any(f <= 3 for f in prev):
                lab = "今節3着内あり(1着なし)"
            else:
                lab = "今節着外のみ"
            sm[(rk, e["player_id"])] = lab
    return sm


def build_rp_delta(races, entries_by_race):
    """選手の前回出走時（異なるrace_date）からの競走得点変化の符号。"""
    seq = defaultdict(list)   # player_id -> [(date, rp)]
    for rk, ents in entries_by_race.items():
        meta = races.get(rk)
        if not meta:
            continue
        for e in ents:
            if e["race_point"] is None:
                continue
            seq[e["player_id"]].append((meta["race_date"], float(e["race_point"])))
    prev_rp = {}
    for pid, lst in seq.items():
        lst.sort()
        # 日付ごとに代表値（同日は同一得点）
        by_date = {}
        for d, rp in lst:
            by_date.setdefault(d, rp)
        dates = sorted(by_date)
        for n, d in enumerate(dates):
            prev_rp[(pid, d)] = by_date[dates[n - 1]] if n > 0 else None
    return prev_rp


class Acc:
    __slots__ = ("n", "obs", "mkt", "d", "d2", "odds")

    def __init__(self):
        self.n = 0
        self.obs = 0.0
        self.mkt = 0.0
        self.d = self.d2 = 0.0
        self.odds = []

    def add(self, y, mp, win_odds=None):
        self.n += 1
        self.obs += y
        self.mkt += mp
        d = y - mp
        self.d += d
        self.d2 += d * d
        if win_odds is not None and len(self.odds) < 200000:
            self.odds.append(win_odds)

    def report(self):
        n = self.n
        act, mkt = self.obs / n, self.mkt / n
        md = self.d / n
        var = max(self.d2 / n - md * md, 0.0)
        t = md / math.sqrt(var / n) if var > 0 else 0.0
        ratio = act / mkt if mkt > 0 else 0.0
        return {"n": n, "act": act * 100, "mkt": mkt * 100, "ratio": ratio, "t": t,
                "roi": TAKEOUT_RETURN * ratio * 100,
                "med": statistics.median(self.odds) if self.odds else 0.0,
                "p30": (sum(1 for o in self.odds if o >= 30) / len(self.odds) * 100)
                       if self.odds else 0.0}


def main():
    races, entries, venues = load_all()
    print("[prep] same-meet 実績を集計 ...", flush=True)
    sm = build_samemeet(races, entries)
    print("[prep] 競走得点トレンドを集計 ...", flush=True)
    prev_rp = build_rp_delta(races, entries)

    rider = defaultdict(lambda: defaultdict(Acc))   # dim -> (window, seg) -> Acc
    pair = defaultdict(lambda: defaultdict(Acc))
    venue_pay = defaultdict(lambda: defaultdict(Acc))   # 荒れ度用

    by_month = defaultdict(list)
    for rk, meta in races.items():
        by_month[meta["race_date"][:7]].append(rk)

    for ym in sorted(by_month):
        rks = by_month[ym]
        boards = load_trio_odds(rks)
        for rk in rks:
            meta = races[rk]
            ents = entries.get(rk)
            board = boards.get(rk)
            if not ents or len(ents) != 7 or not board or len(board) < MIN_BOARD:
                continue
            if any(e["pred_top3_pct"] is None for e in ents):
                continue
            fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
                   if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
            if len(fin) < 3:
                continue
            fin.sort()
            tm = 0
            for _, f in fin[:3]:
                tm |= 1 << f
            if tm not in board:
                continue
            win_odds = board[tm]

            w = "TRAIN" if meta["race_date"] <= TRAIN_TO else "TEST"
            bf = {int(e["frame_no"]): e for e in ents}
            frames = sorted(bf)

            mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if o > 0}
            tot = sum(mk_raw.values())
            if tot <= 0:
                continue
            mkt = {m: v / tot for m, v in mk_raw.items()}
            marg = {f: 0.0 for f in frames}
            pair_mkt = defaultdict(float)
            for m, p in mkt.items():
                fs = [f for f in frames if (m >> f) & 1]
                for f in fs:
                    marg[f] += p
                for a, b in combinations(fs, 2):
                    pair_mkt[(a, b)] += p

            vname = venues.get(meta["venue_id"], {}).get("name", meta["venue_id"])
            bank = venues.get(meta["venue_id"], {}).get("bank")
            di = meta["day_index"]
            di_lab = f"{di}日目" if di is not None else "日番不明"

            # 荒れ度（レース単位・場別）
            venue_pay[("venue", vname)][(w, vname)].add(0.0, 0.0, win_odds)
            venue_pay[("bank", f"周長{bank}")][(w, f"周長{bank}")].add(0.0, 0.0, win_odds)

            # ---- 車単位 ----
            for f in frames:
                e = bf[f]
                y = 1.0 if (tm >> f) & 1 else 0.0
                mp = marg[f]
                if mp <= 0:
                    continue
                smlab = sm.get((rk, e["player_id"]), "不明")
                pr = prev_rp.get((e["player_id"], meta["race_date"]))
                if pr is None or e["race_point"] is None:
                    rplab = "得点トレンド不明"
                else:
                    d = float(e["race_point"]) - pr
                    rplab = ("得点上昇(+1.0超)" if d > 1.0 else
                             "得点上昇(0〜1.0)" if d > 0 else
                             "得点横這い" if d == 0 else
                             "得点下降(0〜-1.0)" if d > -1.0 else "得点下降(-1.0超)")
                cls = str(e["player_class"] or "?")
                sty = str(e["style"] or "?")
                segs = [
                    ("A_samemeet", smlab),
                    ("A_samemeet_x_day", f"{smlab}×{di_lab}"),
                    ("B_venue", vname),
                    ("C_frame", f"{f}番車"),
                    ("D_class_style", f"{cls}×{sty}"),
                    ("E_rp_trend", rplab),
                    ("F_bank", f"周長{bank}"),
                ]
                for dim, seg in segs:
                    rider[dim][(w, seg)].add(y, mp)

            # ---- ペア単位（S7型ROI）----
            for i, j in combinations(frames, 2):
                mp = pair_mkt.get((i, j), 0.0)
                if mp <= 0:
                    continue
                y = 1.0 if ((tm >> i) & 1 and (tm >> j) & 1) else 0.0
                si = sm.get((rk, bf[i]["player_id"]), "不明")
                sj = sm.get((rk, bf[j]["player_id"]), "不明")
                # 「今節1着あり」を含むペアか
                lab = ("両方今節1着あり" if si == sj == "今節1着あり" else
                       "片方今節1着あり" if "今節1着あり" in (si, sj) else "どちらも1着なし")
                pair[("A_pair_samemeet", lab)][(w, lab)].add(y, mp)
                pair[("B_pair_venue", vname)][(w, vname)].add(y, mp)
        print(f"  {ym}: {len(rks)}R", flush=True)

    # ---------------- 出力 ----------------
    def dump_rider(dim, title, min_n=MIN_RIDERS, sort_by_ratio=False):
        print("\n" + "-" * 116)
        print(title)
        print("-" * 116)
        print(f"{'セグメント':<30}{'窓':<6}{'車数':>9}{'実測%':>8}{'市場%':>8}"
              f"{'実測/市場':>10}{'t値':>8}{'判定':>22}")
        segs = {k[1] for k in rider[dim]}
        if sort_by_ratio:
            def key(s):
                a = rider[dim].get(("TEST", s))
                return -(a.report()["ratio"] if a and a.n >= min_n else -9)
        else:
            def key(s):
                a = rider[dim].get(("TEST", s))
                return -(a.n if a else 0)
        for seg in sorted(segs, key=key):
            printed = False
            for w in ("TRAIN", "TEST"):
                a = rider[dim].get((w, seg))
                if not a or a.n < min_n:
                    continue
                p = a.report()
                v = ""
                if p["ratio"] > 1.0 and p["t"] > 3:
                    v = "市場が過小評価"
                elif p["ratio"] < 1.0 and p["t"] < -3:
                    v = "市場が過大評価"
                print(f"{seg if not printed else '':<30}{w:<6}{p['n']:>9}{p['act']:>8.2f}"
                      f"{p['mkt']:>8.2f}{p['ratio']:>10.3f}{p['t']:>+8.2f}{v:>22}")
                printed = True
            if printed:
                print()

    print("\n" + "=" * 116)
    print("堅調/波乱・穴メモの仮説を市場ミスプライシングとして検証")
    print("  車単位 ratio = 実測3着内率 ÷ 市場の3着内確率  →  >1 なら市場が過小評価")
    print("  ペア単位 ROI = 0.75 × ratio  →  ROI100%超には ratio ≥ 1.333")
    print("=" * 116)

    dump_rider("A_samemeet", "A【最重要・ラグ仮説】同開催内の調子（メモ4章「初日の好走は翌日以降に人気化」）")
    dump_rider("A_samemeet_x_day", "A' 同開催内の調子 × 開催日")
    dump_rider("E_rp_trend", "E 競走得点トレンド（メモ4章「調子の変化」）")
    dump_rider("C_frame", "C 車番（メモ2章「4/6/8番車が絡むと高配当」※7車立てなので4/6）")
    dump_rider("D_class_style", "D 級班×脚質（メモ4章「下位クラスでは追込を軸に」）")
    dump_rider("F_bank", "F バンク周長（メモ2章「333mは荒れやすい」）")
    dump_rider("B_venue", "B 競輪場別の市場ミスプライシング（ratio降順）", sort_by_ratio=True)

    # 荒れ度（配当）
    print("\n" + "-" * 116)
    print("B' 競輪場別の「荒れ度」＝勝ち三連複配当（メモ2章の荒れやすい場リストの再検証）")
    print("   メモの列挙: 小松島・大宮・大垣・弥彦・熊本・四日市・松戸・小田原・宇都宮・高知")
    print("-" * 116)
    MEMO = {"小松島", "大宮", "大垣", "弥彦", "熊本", "四日市", "松戸", "小田原", "宇都宮", "高知"}
    print(f"{'競輪場':<14}{'メモ':<6}{'TRAIN n':>9}{'TR中央値':>10}{'TR30倍+%':>10}"
          f"{'TEST n':>9}{'TE中央値':>10}{'TE30倍+%':>10}")
    rowsv = []
    for (dim, seg) in list(venue_pay):
        if dim != "venue":
            continue
        a = venue_pay[(dim, seg)].get(("TRAIN", seg))
        b = venue_pay[(dim, seg)].get(("TEST", seg))
        if not a or not b or a.n < 300 or b.n < 100:
            continue
        ra, rb = a.report(), b.report()
        rowsv.append((seg, ra, rb))
    for seg, ra, rb in sorted(rowsv, key=lambda x: -x[1]["med"]):
        print(f"{seg:<14}{'★' if seg in MEMO else '':<6}{ra['n']:>9}{ra['med']:>10.1f}"
              f"{ra['p30']:>10.1f}{rb['n']:>9}{rb['med']:>10.1f}{rb['p30']:>10.1f}")

    print("\n" + "-" * 116)
    print("F' バンク周長別の荒れ度")
    print("-" * 116)
    for (dim, seg) in list(venue_pay):
        if dim != "bank":
            continue
        for w in ("TRAIN", "TEST"):
            a = venue_pay[(dim, seg)].get((w, seg))
            if not a or a.n < 300:
                continue
            p = a.report()
            print(f"  {seg:<12}{w:<6} n={p['n']:>6}  配当中央値 {p['med']:>7.1f}倍  30倍以上 {p['p30']:>5.1f}%")

    # ペア単位ROI
    print("\n" + "-" * 116)
    print("A'' ペア単位ROI: 「今節1着あり」の選手を含む軸ペア（S7型5点流し）")
    print("-" * 116)
    print(f"{'セグメント':<24}{'窓':<6}{'ペア数':>10}{'実測%':>8}{'市場%':>8}"
          f"{'実測/市場':>10}{'t値':>8}{'→ROI%':>9}")
    for key in list(pair):
        if key[0] != "A_pair_samemeet":
            continue
        seg = key[1]
        printed = False
        for w in ("TRAIN", "TEST"):
            a = pair[key].get((w, seg))
            if not a or a.n < MIN_PAIRS:
                continue
            p = a.report()
            flag = "  ★" if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3 else ""
            print(f"{seg if not printed else '':<24}{w:<6}{p['n']:>10}{p['act']:>8.2f}"
                  f"{p['mkt']:>8.2f}{p['ratio']:>10.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}{flag}")
            printed = True
        if printed:
            print()

    # 結論判定
    print("\n" + "=" * 116)
    print(f"【結論】TRAIN/TESTともに ratio ≥ {ROI_BREAKEVEN:.3f} かつ TEST t>3（ROI100%超）")
    print("=" * 116)
    hits = []
    for store, minn, tag in ((rider, MIN_RIDERS, "車"), (pair, MIN_PAIRS, "ペア")):
        for dim in list(store):
            for seg in {k[1] for k in store[dim]}:
                a, b = store[dim].get(("TRAIN", seg)), store[dim].get(("TEST", seg))
                if not a or not b or a.n < minn or b.n < minn:
                    continue
                ra, rb = a.report(), b.report()
                if ra["ratio"] >= ROI_BREAKEVEN and rb["ratio"] >= ROI_BREAKEVEN and rb["t"] > 3:
                    hits.append((tag, dim, seg, ra, rb))
    if not hits:
        print("  該当なし。")
        print()
        print("  以下も併せて確認すること（本スクリプトの主眼）:")
        print("   - A（ラグ仮説）の ratio が 1.0 付近なら、市場は同開催内の調子を")
        print("     即座に織り込んでおり『翌日以降に人気化する』というタイムラグは存在しない。")
        print("   - ratio > 1 が有意でも 1.333 未満なら方向は正しいが控除率に届かない。")
    else:
        for tag, dim, seg, ra, rb in sorted(hits, key=lambda x: -x[4]["ratio"]):
            print(f"  ★[{tag}/{dim}] {seg}")
            print(f"      TRAIN n={ra['n']:>7} ratio={ra['ratio']:.3f} / "
                  f"TEST n={rb['n']:>7} ratio={rb['ratio']:.3f} (t={rb['t']:+.2f}) ROI={rb['roi']:.1f}%")


if __name__ == "__main__":
    main()
