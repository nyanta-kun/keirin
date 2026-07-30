"""【ペア相関の網羅スキャン】競輪固有の相関構造を体系的に洗い出す
（2026-07-30・ユーザー指摘「他にも競輪として見落としている相関関係がないか」）。

## 測定指標
全レースの全ペア(i,j)（21通り/レース）について:
  lift = 観測された同時3着内率 / 独立仮定の期待値(p_i × p_j)
lift > 1 = 正の相関（独立仮定より一緒に来やすい）、< 1 = 負の相関（共倒れ）

現行モデルは各選手の周辺確率を独立に予測しているため、lift が 1 から
有意に離れるバケットは **モデルが構造的に取りこぼしている情報** を意味する。

## スキャンする軸（A〜E）
A. ライン構造: 同/別ライン・位置関係・番手の恩恵・単騎・別ライン先頭同士・分戦数
B. 脚質: 逃げ×逃げ(別ライン)＝先行争い共倒れ・逃げ×追込(同ライン)＝相補コンビ・全組合せ
C. 地縁: 同県・同地区・同期(term近接)
D. 装備: ギア倍数の近接性
E. 交互作用: バンク周長×ライン相関・グレード×ライン相関・開催日程×ライン相関

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
（TRAIN/TESTで同じ方向に出るかを必ず確認する。片方だけのliftは信用しない）
"""
import sys
from collections import defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"

REGION = {}
for pref, reg in [
    ("北海道", "北日本"), ("青森", "北日本"), ("岩手", "北日本"), ("宮城", "北日本"),
    ("秋田", "北日本"), ("山形", "北日本"), ("福島", "北日本"),
    ("茨城", "関東"), ("栃木", "関東"), ("群馬", "関東"), ("埼玉", "関東"),
    ("千葉", "南関東"), ("東京", "南関東"), ("神奈川", "南関東"), ("山梨", "南関東"),
    ("静岡", "南関東"),
    ("愛知", "中部"), ("岐阜", "中部"), ("三重", "中部"), ("長野", "中部"),
    ("富山", "中部"), ("石川", "中部"), ("福井", "中部"),
    ("滋賀", "近畿"), ("京都", "近畿"), ("大阪", "近畿"), ("兵庫", "近畿"),
    ("奈良", "近畿"), ("和歌山", "近畿"),
    ("鳥取", "中国"), ("島根", "中国"), ("岡山", "中国"), ("広島", "中国"), ("山口", "中国"),
    ("徳島", "四国"), ("香川", "四国"), ("愛媛", "四国"), ("高知", "四国"),
    ("福岡", "九州"), ("佐賀", "九州"), ("長崎", "九州"), ("熊本", "九州"),
    ("大分", "九州"), ("宮崎", "九州"), ("鹿児島", "九州"), ("沖縄", "九州"),
]:
    REGION[pref] = reg


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT r.race_key, r.race_date, r.grade, r.day_index, v.bank_length "
            "FROM wt_races r LEFT JOIN venue_info v ON r.venue_id = v.venue_code "
            "WHERE r.n_entries = 7 AND r.cancel = 0 AND r.race_date BETWEEN ? AND ?",
            (TRAIN_FROM, TEST_TO)).fetchall()
    races = {r["race_key"]: {"race_date": str(r["race_date"]), "grade": r["grade"],
                              "day_index": r["day_index"], "bank_length": r["bank_length"]}
             for r in rrows}
    keys = list(races.keys())
    print(f"[load]   races: {len(keys)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, pred_top3_pct, prefecture, term, gear_ratio, "
                 "       style, line_group, line_pos, line_size, is_line_leader, n_lines, "
                 "       finish_order FROM wt_entries WHERE race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load]   entries races: {len(by_race)}", flush=True)
    return races, by_race


def build(races, entries_by_race):
    out = []
    for rk, meta in races.items():
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        if any(e["pred_top3_pct"] is None for e in ents):
            continue
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        if len(fin) < 3:
            continue
        fin.sort()
        by_frame = {int(e["frame_no"]): e for e in ents}
        out.append({"race_key": rk, "race_date": meta["race_date"], "meta": meta,
                    "by_frame": by_frame,
                    "top3": frozenset(fno for _, fno in fin[:3])})
    print(f"[build]   rows: {len(out)}", flush=True)
    return out


def _same_line(bf, i, j):
    li, lj = bf[i]["line_group"], bf[j]["line_group"]
    if li is None or lj is None:
        return None
    return li == lj


def scan(rows, label, keyfn, min_n=300, title=""):
    agg = defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})
    for r in rows:
        bf = r["by_frame"]
        for i, j in combinations(bf.keys(), 2):
            b = keyfn(r, bf, i, j)
            if b is None:
                continue
            pi = float(bf[i]["pred_top3_pct"]) / 100.0
            pj = float(bf[j]["pred_top3_pct"]) / 100.0
            a = agg[b]
            a["n"] += 1
            a["exp"] += pi * pj
            if i in r["top3"] and j in r["top3"]:
                a["obs"] += 1
    res = {}
    for b, a in agg.items():
        if a["n"] < min_n:
            continue
        obs = a["obs"] / a["n"]
        exp = a["exp"] / a["n"]
        lift = obs / exp if exp > 0 else 0.0
        se = sqrt(max(obs * (1 - obs), 1e-12) / a["n"])
        res[b] = (a["n"], obs, exp, lift, se)
    return res


def print_compare(title, res_tr, res_te):
    print(f"\n--- {title} ---")
    print(f"  {'バケット':<26}{'TRAIN n':>9}{'lift':>8}"
          f"{'TEST n':>9}{'lift':>8}{'判定':>16}")
    keys = sorted(set(res_tr) | set(res_te),
                  key=lambda k: -(res_tr.get(k, (0,))[0]))
    for k in keys:
        tr = res_tr.get(k)
        te = res_te.get(k)
        if tr is None or te is None:
            continue
        n1, o1, e1, l1, s1 = tr
        n2, o2, e2, l2, s2 = te
        # TRAIN/TESTで同方向かつ両方1から離れているか
        same_dir = (l1 - 1) * (l2 - 1) > 0
        strong = min(abs(l1 - 1), abs(l2 - 1)) >= 0.05
        if same_dir and strong:
            verdict = "★正の相関" if l1 > 1 else "★負の相関"
        elif same_dir:
            verdict = "同方向(弱)"
        else:
            verdict = "不一致"
        print(f"  {str(k):<26}{n1:>9}{l1:>7.3f}x{n2:>9}{l2:>7.3f}x{verdict:>16}")


def main():
    races, entries_by_race = load_all()
    rows = build(races, entries_by_race)
    train = [r for r in rows if TRAIN_FROM <= r["race_date"] <= TRAIN_TO]
    test = [r for r in rows if TEST_FROM <= r["race_date"] <= TEST_TO]
    print(f"\n[main] TRAIN={len(train)} TEST={len(test)}")
    print("lift = 観測同時3着内率 / 独立仮定の期待値。1.0=独立、>1=一緒に来やすい\n")

    SCANS = []

    # ===== A. ライン構造 =====
    SCANS.append(("A1. 同ライン/別ライン",
                  lambda r, bf, i, j: (None if _same_line(bf, i, j) is None
                                        else ("同ライン" if _same_line(bf, i, j) else "別ライン"))))

    def a2(r, bf, i, j):
        s = _same_line(bf, i, j)
        if s is None or not s:
            return None
        pi, pj = bf[i]["line_pos"], bf[j]["line_pos"]
        if pi is None or pj is None:
            return None
        a, b = sorted([int(pi), int(pj)])
        return f"同ライン{a}-{b}番手"
    SCANS.append(("A2. 同ライン内の位置関係", a2))

    def a3(r, bf, i, j):
        """番手の恩恵: 先頭+番手(1-2番手)ペアか否か"""
        s = _same_line(bf, i, j)
        if s is None:
            return None
        if not s:
            return "別ライン"
        pi, pj = bf[i]["line_pos"], bf[j]["line_pos"]
        if pi is None or pj is None:
            return None
        a, b = sorted([int(pi), int(pj)])
        return "同ライン先頭+番手" if (a, b) == (1, 2) else "同ラインその他"
    SCANS.append(("A3. 番手の恩恵", a3))

    def a4(r, bf, i, j):
        """単騎(line_size==1)を含むペア"""
        si, sj = bf[i]["line_size"], bf[j]["line_size"]
        if si is None or sj is None:
            return None
        n_solo = (1 if int(si) == 1 else 0) + (1 if int(sj) == 1 else 0)
        return {0: "両者ライン所属", 1: "片方が単騎", 2: "両者単騎"}[n_solo]
    SCANS.append(("A4. 単騎を含むペア", a4))

    def a5(r, bf, i, j):
        """別ラインの先頭同士（先行争い）"""
        s = _same_line(bf, i, j)
        if s is None or s:
            return None
        li, lj = bf[i]["is_line_leader"], bf[j]["is_line_leader"]
        if li is None or lj is None:
            return None
        n = int(li) + int(lj)
        return {2: "別ライン先頭×先頭", 1: "別ライン先頭×非先頭", 0: "別ライン非先頭×非先頭"}[n]
    SCANS.append(("A5. 別ライン先頭同士(先行争い)", a5))

    def a6(r, bf, i, j):
        s = _same_line(bf, i, j)
        if s is None:
            return None
        nl = bf[i]["n_lines"]
        if nl is None:
            return None
        return f"{int(nl)}分戦/{'同' if s else '別'}ライン"
    SCANS.append(("A6. 分戦数 × ライン関係", a6))

    # ===== B. 脚質 =====
    def b1(r, bf, i, j):
        s = _same_line(bf, i, j)
        if s is None:
            return None
        si, sj = bf[i]["style"], bf[j]["style"]
        if not si or not sj:
            return None
        pair = "×".join(sorted([si, sj]))
        return f"{'同' if s else '別'}/{pair}"
    SCANS.append(("B1. 脚質組合せ × ライン関係", b1))

    def b2(r, bf, i, j):
        """先行争い: 別ラインの逃げ×逃げ"""
        s = _same_line(bf, i, j)
        if s is None or s:
            return None
        si, sj = bf[i]["style"], bf[j]["style"]
        if not si or not sj:
            return None
        if si == "逃" and sj == "逃":
            return "別ライン 逃×逃"
        if "逃" in (si, sj):
            return "別ライン 逃×非逃"
        return "別ライン 非逃×非逃"
    SCANS.append(("B2. 先行争い(別ライン逃×逃)", b2))

    # ===== C. 地縁 =====
    def c1(r, bf, i, j):
        pi, pj = bf[i]["prefecture"], bf[j]["prefecture"]
        s = _same_line(bf, i, j)
        if not pi or not pj or s is None:
            return None
        same_pref = pi == pj
        return f"{'同' if s else '別'}ライン/{'同県' if same_pref else '別県'}"
    SCANS.append(("C1. 同県 × ライン関係", c1))

    def c2(r, bf, i, j):
        pi, pj = bf[i]["prefecture"], bf[j]["prefecture"]
        s = _same_line(bf, i, j)
        if not pi or not pj or s is None:
            return None
        ri, rj = REGION.get(pi), REGION.get(pj)
        if ri is None or rj is None:
            return None
        return f"{'同' if s else '別'}ライン/{'同地区' if ri == rj else '別地区'}"
    SCANS.append(("C2. 同地区 × ライン関係", c2))

    def c3(r, bf, i, j):
        ti, tj = bf[i]["term"], bf[j]["term"]
        if ti is None or tj is None:
            return None
        d = abs(int(ti) - int(tj))
        if d == 0:
            return "同期"
        if d <= 4:
            return "期近接(1-4)"
        if d <= 12:
            return "期中距離(5-12)"
        return "期遠い(13+)"
    SCANS.append(("C3. 期(term)の近接性", c3))

    # ===== D. 装備 =====
    def d1(r, bf, i, j):
        gi, gj = bf[i]["gear_ratio"], bf[j]["gear_ratio"]
        if gi is None or gj is None:
            return None
        d = abs(float(gi) - float(gj))
        if d < 0.005:
            return "ギア同一"
        if d < 0.04:
            return "ギア近接"
        return "ギア差大"
    SCANS.append(("D1. ギア倍数の近接性", d1))

    # ===== E. 交互作用 =====
    def e1(r, bf, i, j):
        s = _same_line(bf, i, j)
        if s is None or not s:
            return None
        bl = r["meta"]["bank_length"]
        if bl is None:
            return None
        return f"周長{int(bl)}/同ライン"
    SCANS.append(("E1. バンク周長 × 同ライン相関", e1))

    def e2(r, bf, i, j):
        s = _same_line(bf, i, j)
        if s is None or not s:
            return None
        g = r["meta"]["grade"]
        if not g:
            return None
        return f"{g}/同ライン"
    SCANS.append(("E2. グレード × 同ライン相関", e2))

    def e3(r, bf, i, j):
        s = _same_line(bf, i, j)
        if s is None or not s:
            return None
        di = r["meta"]["day_index"]
        if di is None:
            return None
        d = int(di)
        tag = "初日" if d == 1 else ("2日目" if d == 2 else "3日目以降")
        return f"{tag}/同ライン"
    SCANS.append(("E3. 開催日程 × 同ライン相関", e3))

    for title, fn in SCANS:
        rtr = scan(train, "TRAIN", fn)
        rte = scan(test, "TEST", fn)
        print_compare(title, rtr, rte)

    print("\n" + "=" * 90)
    print("判定基準: TRAIN/TESTで同方向 かつ 両方でliftが1から0.05以上離れる → ★")
    print("=" * 90)


if __name__ == "__main__":
    main()
