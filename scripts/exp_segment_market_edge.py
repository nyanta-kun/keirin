"""【セグメント別 市場エッジ診断】我々の情報が市場に勝つレース領域は存在するか
（2026-07-30・ユーザー方針「レンジを絞り、レースを厳選することでROIを確保する」の前提検証）。

## 位置づけ（なぜこの検証が必要か）

[[keirin_clean_baseline_market_efficiency_2026_07_30]]で以下が確定している:
- S1 ROI 80.0% / S7 78.6%（控除率の壁75%に収束）
- モデルは市場に予測精度で負ける（Brier 0.0249 vs 0.0245・logloss 0.1025 vs 0.0996）
- パターン分類×軸選定30セル・硬い除外×軸選定120セル・3列目絞り込み等
  すべてROI100%未満

しかし**上記の市場効率の実証は全469,280組の「全体平均」で測られており、
セグメント別には測られていない**。ユーザー方針「レンジを絞れば勝てる領域がある」が
成立する必要十分条件は「**我々の情報が市場に勝つセグメントが存在すること**」であり、
本スクリプトはそれを直接測る。

## なぜROIではなく予測精度で測るのか

ROIは「的中」というレアイベント依存で分散が極端に大きく、n=100程度のセルでは
多重比較ノイズに埋もれる（3車突出のTEST 91.5%がまさにそれ。TRAIN/TESTで
ベスト軸が食い違い月次46.5-148.9%と乱高下した）。
一方 logloss/Brier は1レースあたり35組の密な情報を使うため、同じデータで
桁違いの検出力が得られる。しかも**paired**（同一レースでour vs marketを比較）
なのでレース難易度の交絡が自動的に消える。

**エッジがあるセグメントが存在しなければ、絞り込みの工夫でROI100%超を目指す
探索は打ち切ってよい**と確定できる。逆に存在すれば、そこで初めて条件別モデル・
買い目点数絞りが意味を持つ。

## 測る指標（すべてセグメント別・paired）

各レースの35組について:
  our_prob    = pred_top3_pct の積にライン相関lift適用 → レース内正規化
  market_prob = 0.75 / trio_odds → レース内正規化（控除率を戻した市場の含意確率）

per-race:
  ll     = -log(p[実際に来た組])         ← 主指標。小さいほど良い
  brier  = Σ_35組 (p - y)^2
  rank   = 実際に来た組の順位（1-35）
  d      = ll_market - ll_our            ← 正なら我々の勝ち（paired差分）

セグメント別に mean(d) と t統計量（= mean(d)/SE(d)）を出す。
多重比較を考慮し **|t|>3 かつ TRAIN/TEST で符号一致** を採用条件とする。

## セグメント軸（すべて発走前・オッズ非依存）

1. dominance pattern（1車/2車/3車突出・全体拮抗）※閾値はTRAIN p25固定
2. chalk_q     : top3_sum_top2 の四分位（硬さ）
3. rp_std_q    : 競走得点の標準偏差の四分位（波乱度の最重要特徴量）
4. entropy_q   : pred_top3_pct のエントロピー四分位
5. n_lines     : 分戦数（2/3/4/5+）
6. n_tanki     : 単騎数（0/1/2+）
7. max_line    : 最大ライン長（2/3/4+）
8. grade       : グレード
9. mark_agree  : 記者◎がモデルw1と一致するか（市場と我々のズレの代理指標）
10. 2D: pattern × chalk_q（絞り込みの実務形に最も近い組み合わせ）

副次的に各セグメントの参考ROI（軸w1+w2の三連複5点流し）と配当中央値も併記する。

honest分割: TRAIN 2024-01-01〜2025-12-31（lift推定・閾値決定のみ）/
            TEST 2026-01-01〜2026-07-30（評価）
閾値・liftはTRAINのみで確定しTESTへ固定適用する（リーク防止）。
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

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
TAKEOUT_RETURN = 0.75      # 控除率25% → 払戻率75%
MIN_BOARD = 33             # 35通りのうちこれ以上揃っているレースのみ採用
MIN_SEG_RACES = 150        # これ未満のセグメントは表示しない（ノイズ回避）


# ---------------------------------------------------------------- ロード

def load_races():
    with get_connection() as c:
        rows = c.execute(
            "SELECT race_key, race_date, grade FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    out = {}
    for r in rows:
        out[r["race_key"]] = {"race_date": str(r["race_date"]),
                              "grade": str(r["grade"]) if r["grade"] is not None else "?"}
    print(f"[load] races(7車・非中止): {len(out)}", flush=True)
    return out


def load_entries(race_keys):
    by_race = defaultdict(list)
    keys = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_win_pct, pred_top3_pct, prediction_mark, "
                 "       line_group, line_pos, line_size, n_lines, race_point, finish_order "
                 "FROM wt_entries WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load] entries: {sum(len(v) for v in by_race.values())} 行", flush=True)
    return by_race


def load_trio_odds(race_keys):
    """{race_key: {mask: odds}} を返す。組は車番のビットマスクで表現（メモリ節約）。"""
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
                mask = 0
                for p in parts:
                    mask |= 1 << p
                out.setdefault(rk, {})[mask] = fv
    return out


# ---------------------------------------------------------------- 前処理

def _entropy(vals):
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def build_rows(races, entries_by_race):
    """レース単位の特徴量・着順を組み立てる（オッズ非依存部分）。"""
    rows = []
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None or e["pred_win_pct"] is None for e in ents):
            continue
        fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        top3_frames = [f for _, f in fin[:3]]
        mask = 0
        for f in top3_frames:
            mask |= 1 << f

        by_frame = {int(e["frame_no"]): e for e in ents}
        win = sorted(((float(e["pred_win_pct"]), int(e["frame_no"])) for e in ents), reverse=True)
        top3p = sorted(((float(e["pred_top3_pct"]), int(e["frame_no"])) for e in ents), reverse=True)

        w = [v for v, _ in win]
        g12, g23, g34 = w[0] - w[1], w[1] - w[2], w[2] - w[3]

        # ライン構造
        sizes = defaultdict(int)
        for e in ents:
            lg = e["line_group"]
            sizes[lg if lg is not None else f"_solo{e['frame_no']}"] += 1
        n_lines = len(sizes)
        n_tanki = sum(1 for v in sizes.values() if v == 1)
        max_line = max(sizes.values())

        rps = [float(e["race_point"]) for e in ents if e["race_point"] is not None]
        rp_std = statistics.pstdev(rps) if len(rps) >= 7 else None

        # prediction_mark は数値コード（1=◎ 2=◯ 3=▲ 4=△ 0=無印）
        marks = {}
        for e in ents:
            pm = e["prediction_mark"]
            if pm is None:
                continue
            try:
                pmi = int(pm)
            except (TypeError, ValueError):
                continue
            if pmi >= 1:
                marks[pmi] = int(e["frame_no"])
        honmei = marks.get(1)
        mark_top2 = {marks[k] for k in (1, 2) if k in marks}
        model_top2 = {win[0][1], win[1][1]}
        n_overlap = len(mark_top2 & model_top2) if mark_top2 else None

        rows.append({
            "race_key": rk,
            "race_date": meta["race_date"],
            "grade": meta["grade"],
            "by_frame": by_frame,
            "win_order": [f for _, f in win],       # pred_win_pct 降順の車番
            "top3_order": [f for _, f in top3p],
            "win_max": w[0], "g12": g12, "g23": g23, "g34": g34,
            "top3_sum_top2": top3p[0][0] + top3p[1][0],
            "top3_entropy": _entropy([v for v, _ in top3p]),
            "n_lines": n_lines, "n_tanki": n_tanki, "max_line": max_line,
            "rp_std": rp_std,
            "mark_agree": (honmei is not None and honmei == win[0][1]),
            "mark_overlap": n_overlap,
            "top3_mask": mask,
        })
    print(f"[build] rows: {len(rows)}", flush=True)
    return rows


def pair_bucket(bf, i, j):
    li, lj = bf[i]["line_group"], bf[j]["line_group"]
    if li is None or lj is None:
        return "unknown"
    if li != lj:
        return "diff"
    pi, pj = bf[i]["line_pos"], bf[j]["line_pos"]
    if pi is None or pj is None:
        return "same_other"
    a, b = sorted([int(pi), int(pj)])
    if (a, b) == (1, 2):
        return "same_12"
    if (a, b) == (2, 3):
        return "same_23"
    if (a, b) == (1, 3):
        return "same_13"
    return "same_other"


def estimate_lifts(rows):
    """ライン相関lift（TRAINのみで推定）。"""
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        tm = r["top3_mask"]
        for i, j in combinations(sorted(bf.keys()), 2):
            b = pair_bucket(bf, i, j)
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if (tm >> i) & 1 and (tm >> j) & 1:
                a["obs"] += 1
    return {b: ((a["obs"] / a["n"]) / (a["exp"] / a["n"]) if a["n"] >= 100 and a["exp"] > 0 else 1.0)
            for b, a in agg.items()}


# ---------------------------------------------------------------- セグメント定義

def quartile_cuts(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return (0.0, 0.0, 0.0)
    return (v[len(v) // 4], v[len(v) // 2], v[3 * len(v) // 4])


def q_label(x, cuts, name):
    if x is None:
        return f"{name}=NA"
    if x < cuts[0]:
        return f"{name}Q1(低)"
    if x < cuts[1]:
        return f"{name}Q2"
    if x < cuts[2]:
        return f"{name}Q3"
    return f"{name}Q4(高)"


def pattern_of(r, tie_thr):
    """1車/2車/3車突出・全体拮抗（既存 exp_dominance_pattern_analysis.py と同一定義）。"""
    mx = max(r["g12"], r["g23"], r["g34"])
    if mx < tie_thr:
        return "全体拮抗"
    if r["g12"] == mx:
        return "1車突出"
    if r["g23"] == mx:
        return "2車突出"
    return "3車突出"


def segments_of(r, cuts, tie_thr):
    """このレースが属する全セグメントラベルを返す。"""
    pat = pattern_of(r, tie_thr)
    chalk = q_label(r["top3_sum_top2"], cuts["chalk"], "硬さ")
    labs = [
        ("ALL", "全体"),
        ("pattern", pat),
        ("chalk", chalk),
        ("rp_std", q_label(r["rp_std"], cuts["rp_std"], "rp_std")),
        ("entropy", q_label(r["top3_entropy"], cuts["entropy"], "entropy")),
        ("n_lines", f"分戦{min(r['n_lines'], 5)}{'+' if r['n_lines'] >= 5 else ''}"),
        ("n_tanki", f"単騎{min(r['n_tanki'], 2)}{'+' if r['n_tanki'] >= 2 else ''}"),
        ("max_line", f"最長ライン{min(r['max_line'], 4)}{'+' if r['max_line'] >= 4 else ''}"),
        ("grade", f"G:{r['grade']}"),
        ("mark_agree", "◎=w1一致" if r["mark_agree"] else "◎≠w1乖離"),
        ("mark_overlap", "記者◎◯×モデル上位2の一致数=NA" if r["mark_overlap"] is None
         else f"記者◎◯×モデル上位2 一致{r['mark_overlap']}車"),
        ("2D", f"{pat}×{chalk}"),
    ]
    return labs


# ---------------------------------------------------------------- 評価

def race_metrics(r, board, lifts):
    """1レースの our/market の logloss・brier・rank と参考ROIを返す。"""
    bf = r["by_frame"]
    frames = sorted(bf.keys())
    p = {f: float(bf[f]["pred_top3_pct"]) / 100.0 for f in frames}

    raw = {}
    for tri in combinations(frames, 3):
        s = p[tri[0]] * p[tri[1]] * p[tri[2]]
        for x, y in combinations(tri, 2):
            s *= lifts.get(pair_bucket(bf, x, y), 1.0)
        m = 0
        for f in tri:
            m |= 1 << f
        raw[m] = s

    mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if m in raw and o > 0}
    common = set(raw) & set(mk_raw)
    if len(common) < MIN_BOARD:
        return None
    win = r["top3_mask"]
    if win not in common:
        return None

    ot = sum(raw[m] for m in common)
    mt = sum(mk_raw[m] for m in common)
    if ot <= 0 or mt <= 0:
        return None
    our = {m: raw[m] / ot for m in common}
    mkt = {m: mk_raw[m] / mt for m in common}

    ll_our = -math.log(max(our[win], 1e-12))
    ll_mkt = -math.log(max(mkt[win], 1e-12))
    br_our = sum((our[m] - (1.0 if m == win else 0.0)) ** 2 for m in common)
    br_mkt = sum((mkt[m] - (1.0 if m == win else 0.0)) ** 2 for m in common)
    rk_our = 1 + sum(1 for m in common if our[m] > our[win])
    rk_mkt = 1 + sum(1 for m in common if mkt[m] > mkt[win])

    # 参考ROI: 軸=pred_win_pct上位2車、三連複5点流し
    a1, a2 = r["win_order"][0], r["win_order"][1]
    axis_mask = (1 << a1) | (1 << a2)
    hit = (win & axis_mask) == axis_mask
    payout = board[win] if hit else 0.0

    return {"ll_our": ll_our, "ll_mkt": ll_mkt, "br_our": br_our, "br_mkt": br_mkt,
            "rk_our": rk_our, "rk_mkt": rk_mkt,
            "bet": 5.0, "ret": payout, "hit": 1 if hit else 0,
            "win_odds": board[win]}


class Acc:
    __slots__ = ("n", "ll_our", "ll_mkt", "br_our", "br_mkt", "rk_our", "rk_mkt",
                 "d", "d2", "bet", "ret", "hit", "odds")

    def __init__(self):
        self.n = 0
        self.ll_our = self.ll_mkt = 0.0
        self.br_our = self.br_mkt = 0.0
        self.rk_our = self.rk_mkt = 0
        self.d = self.d2 = 0.0
        self.bet = self.ret = 0.0
        self.hit = 0
        self.odds = []

    def add(self, m):
        self.n += 1
        self.ll_our += m["ll_our"]
        self.ll_mkt += m["ll_mkt"]
        self.br_our += m["br_our"]
        self.br_mkt += m["br_mkt"]
        self.rk_our += m["rk_our"]
        self.rk_mkt += m["rk_mkt"]
        d = m["ll_mkt"] - m["ll_our"]        # 正なら我々の勝ち
        self.d += d
        self.d2 += d * d
        self.bet += m["bet"]
        self.ret += m["ret"]
        self.hit += m["hit"]
        if len(self.odds) < 200000:
            self.odds.append(m["win_odds"])

    def report(self):
        n = self.n
        mean_d = self.d / n
        var = max(self.d2 / n - mean_d * mean_d, 0.0)
        se = math.sqrt(var / n) if n > 1 else float("inf")
        t = mean_d / se if se > 0 else 0.0
        return {
            "n": n,
            "ll_our": self.ll_our / n, "ll_mkt": self.ll_mkt / n,
            "d": mean_d, "t": t,
            "br_our": self.br_our / n, "br_mkt": self.br_mkt / n,
            "rk_our": self.rk_our / n, "rk_mkt": self.rk_mkt / n,
            "roi": (self.ret / self.bet * 100.0) if self.bet > 0 else 0.0,
            "hit": self.hit / n * 100.0,
            "med_odds": statistics.median(self.odds) if self.odds else 0.0,
        }


def evaluate(rows, lifts, cuts, tie_thr, label):
    """rowsを月ごとにオッズをロードしながら評価（メモリ節約）。"""
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["race_date"][:7]].append(r)

    accs = defaultdict(lambda: defaultdict(Acc))    # dim -> seg -> Acc
    used = skipped = 0
    for ym in sorted(by_month):
        chunk = by_month[ym]
        boards = load_trio_odds([r["race_key"] for r in chunk])
        for r in chunk:
            board = boards.get(r["race_key"])
            if not board:
                skipped += 1
                continue
            m = race_metrics(r, board, lifts)
            if m is None:
                skipped += 1
                continue
            used += 1
            for dim, seg in segments_of(r, cuts, tie_thr):
                accs[dim][seg].add(m)
        print(f"  [{label}] {ym}: {len(chunk)}R (採用累計 {used} / 除外 {skipped})", flush=True)
    return accs, used, skipped


# ---------------------------------------------------------------- 出力

DIM_TITLES = [
    ("ALL", "全体（ベースライン）"),
    ("pattern", "① 突出パターン"),
    ("chalk", "② 硬さ（top3_sum_top2 四分位）"),
    ("rp_std", "③ 競走得点ばらつき rp_std 四分位"),
    ("entropy", "④ top3 エントロピー四分位"),
    ("n_lines", "⑤ 分戦数"),
    ("n_tanki", "⑥ 単騎数"),
    ("max_line", "⑦ 最長ライン長"),
    ("grade", "⑧ グレード"),
    ("mark_agree", "⑨ 記者◎とモデルw1の一致/乖離"),
    ("mark_overlap", "⑨' 記者◎◯とモデル上位2車の一致数（市場との読み違い度）"),
    ("2D", "⑩ パターン×硬さ（2次元）"),
]


def print_dim(dim, title, tr, te):
    print("\n" + "-" * 118)
    print(f"{title}")
    print("-" * 118)
    print(f"{'セグメント':<22}{'窓':<6}{'n':>7}{'ll_our':>9}{'ll_mkt':>9}"
          f"{'Δll':>9}{'t値':>8}{'rank_our':>10}{'rank_mkt':>10}{'ROI%':>8}{'的中%':>7}{'配当中央':>9}")
    segs = sorted(set(tr[dim]) | set(te[dim]),
                  key=lambda s: -(te[dim][s].n if s in te[dim] else 0))
    for seg in segs:
        lines = []
        for wl, acc_map in (("TRAIN", tr), ("TEST", te)):
            a = acc_map[dim].get(seg)
            if a is None or a.n < MIN_SEG_RACES:
                continue
            rp = a.report()
            flag = ""
            if rp["d"] > 0 and rp["t"] > 3:
                flag = "  ★我々優位"
            elif rp["d"] > 0:
                flag = "  (我々微優位)"
            lines.append(f"{seg if wl == 'TRAIN' else '':<22}{wl:<6}{rp['n']:>7}"
                         f"{rp['ll_our']:>9.4f}{rp['ll_mkt']:>9.4f}{rp['d']:>+9.4f}"
                         f"{rp['t']:>+8.2f}{rp['rk_our']:>10.2f}{rp['rk_mkt']:>10.2f}"
                         f"{rp['roi']:>8.1f}{rp['hit']:>7.1f}{rp['med_odds']:>9.1f}{flag}")
        for ln in lines:
            print(ln)
        if lines:
            print()


def main():
    races = load_races()
    entries = load_entries(races.keys())
    rows = build_rows(races, entries)
    del entries

    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[split] TRAIN={len(train)}R  TEST={len(test)}R")

    print("\n[lift] TRAINでライン相関liftを推定 ...", flush=True)
    lifts = estimate_lifts(train)
    for b in sorted(lifts, key=lambda x: -lifts[x]):
        print(f"    {b:<12} {lifts[b]:.4f}")

    # 閾値はTRAINのみで決定しTESTへ固定適用
    tie_thr = quartile_cuts([max(r["g12"], r["g23"], r["g34"]) for r in train])[0]
    cuts = {
        "chalk": quartile_cuts([r["top3_sum_top2"] for r in train]),
        "rp_std": quartile_cuts([r["rp_std"] for r in train]),
        "entropy": quartile_cuts([r["top3_entropy"] for r in train]),
    }
    print(f"\n[thr] 全体拮抗閾値(max_gap p25) = {tie_thr:.2f}pt")
    for k, v in cuts.items():
        print(f"[thr] {k:<8} 四分位カット = {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f}")

    print("\n[eval] TRAIN ...", flush=True)
    tr_acc, tr_used, tr_skip = evaluate(train, lifts, cuts, tie_thr, "TRAIN")
    print("\n[eval] TEST ...", flush=True)
    te_acc, te_used, te_skip = evaluate(test, lifts, cuts, tie_thr, "TEST")

    print("\n" + "=" * 118)
    print("セグメント別 市場エッジ診断")
    print("  Δll = ll_market - ll_our  →  正なら我々が市場より正確（＝真のエッジ）")
    print("  採用条件: Δll>0 かつ |t|>3 かつ TRAIN/TEST で符号一致")
    print(f"  評価レース数: TRAIN {tr_used}（除外{tr_skip}） / TEST {te_used}（除外{te_skip}）")
    print("=" * 118)

    for dim, title in DIM_TITLES:
        print_dim(dim, title, tr_acc, te_acc)

    # ---- 結論の自動判定 ----
    print("\n" + "=" * 118)
    print("【結論】TRAIN/TESTともにΔll>0 かつ TEST |t|>3 を満たすセグメント")
    print("=" * 118)
    found = []
    for dim, _ in DIM_TITLES:
        if dim == "ALL":
            continue
        for seg in set(tr_acc[dim]) & set(te_acc[dim]):
            a, b = tr_acc[dim][seg], te_acc[dim][seg]
            if a.n < MIN_SEG_RACES or b.n < MIN_SEG_RACES:
                continue
            ra, rb = a.report(), b.report()
            if ra["d"] > 0 and rb["d"] > 0 and rb["t"] > 3:
                found.append((dim, seg, ra, rb))
    if not found:
        print("  該当なし。")
        print("  → 発走前情報で定義できるどのレース領域でも、我々のモデルは市場より")
        print("     正確ではない。レース厳選・軸選定・点数絞りの工夫でROI100%超を")
        print("     目指す探索は、この特徴量セットでは打ち切って良いと確定する。")
    else:
        for dim, seg, ra, rb in sorted(found, key=lambda x: -x[3]["t"]):
            print(f"  ★ [{dim}] {seg}")
            print(f"      TRAIN n={ra['n']:>6} Δll={ra['d']:+.4f} t={ra['t']:+.2f} ROI={ra['roi']:.1f}%")
            print(f"      TEST  n={rb['n']:>6} Δll={rb['d']:+.4f} t={rb['t']:+.2f} ROI={rb['roi']:.1f}%")


if __name__ == "__main__":
    main()
