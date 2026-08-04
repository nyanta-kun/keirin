"""【読み取り専用】1軸目（axis1）が3着内を外す要因の分解（2026-08-04）。

ユーザー要望:
  「軸の精度向上がやはり必須。5〜10%は落車・想定外の出来事があるとしても、
    メンバーを見て勝ち切れる1車を選ぶのがベース。現在の1軸目の外れ要因を分析して」

軸1の実測（honest 2025-01〜2026-08・36,831レース）は 3着内 79.3% / 1着 46.1%。
本スクリプトは残り 20.7% の外れを
  ① 欠車・失格（落車含む）による外れ
  ② 完走して4着以下（＝純粋に負けた）
に分け、②がどの条件で起きているかを観測可能な属性で層別する。
併せて「軸1が飛んだとき誰が来たのか」も見る。

honest: 月次凍結vintageモデルのキャッシュ（scripts/exp_7c_cache.py）+ DB属性。
DB書き込みなし（SELECTのみ）。

使い方:
    python scripts/exp_axis1_miss_analysis.py data/exp_7c_cache
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

ENTRY_COLS = [
    "player_class", "style", "race_point", "line_group", "line_size", "line_pos",
    "is_line_leader", "n_lines", "s_count", "b_count", "first_rate", "third_rate",
    "finish_order", "factor", "prediction_mark",
]
RACE_COLS = ["grade", "race_type", "distance", "venue_id"]


def load_cache(cache: Path) -> list[dict]:
    rows = []
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    return rows


def fetch_db(race_keys: list[str]) -> tuple[dict, dict]:
    """(entries[(rk, frame)] , races[rk]) を返す。"""
    entries: dict[tuple[str, int], dict] = {}
    races: dict[str, dict] = {}
    with get_connection() as c:
        for i in range(0, len(race_keys), 900):
            chunk = race_keys[i:i + 900]
            ph = ",".join("?" * len(chunk))
            q = (f"SELECT race_key, frame_no, {','.join(ENTRY_COLS)} "
                 f"FROM wt_entries WHERE race_key IN ({ph})")
            for r in c.execute(q, chunk):
                entries[(r["race_key"], int(r["frame_no"]))] = {
                    k: r[k] for k in ENTRY_COLS}
            q2 = (f"SELECT race_key, {','.join(RACE_COLS)} "
                  f"FROM wt_races WHERE race_key IN ({ph})")
            for r in c.execute(q2, chunk):
                races[r["race_key"]] = {k: r[k] for k in RACE_COLS}
    return entries, races


def bucket(label: str, sub: list[tuple]) -> str:
    """sub = [(3着内か, 1着か, 欠車か)] の集計行を作る。"""
    n = len(sub)
    if n == 0:
        return ""
    t3 = sum(1 for a, _, _ in sub if a)
    w = sum(1 for _, b, _ in sub if b)
    dnf = sum(1 for _, _, d in sub if d)
    return (f"  {label:26} {n:7d} ({100.0*n/BASE_N:5.1f}%) "
            f"3着内 {100.0*t3/n:5.1f}%  1着 {100.0*w/n:5.1f}%  "
            f"欠車失格 {100.0*dnf/n:4.1f}%")


BASE_N = 1


def report(title: str, groups: dict[str, list[tuple]], min_n: int = 200) -> None:
    print(f"【{title}】")
    rows = [(k, v) for k, v in groups.items() if len(v) >= min_n]
    rows.sort(key=lambda kv: -sum(1 for a, _, _ in kv[1] if a) / len(kv[1]))
    for k, v in rows:
        print(bucket(str(k), v))
    small = sum(len(v) for k, v in groups.items() if len(v) < min_n)
    if small:
        print(f"  （n<{min_n} の区分 計 {small}件は省略）")
    print()


def main() -> None:
    global BASE_N
    cands = load_cache(Path(sys.argv[1]))
    rks = sorted({c["race_key"] for c in cands})
    print(f"母集団: {len(cands)}レース（7車立て・軸選定成功）")
    print("DB属性を取得中...", flush=True)
    entries, races = fetch_db(rks)
    print("完了\n")

    recs = []
    for c in cands:
        e = entries.get((c["race_key"], c["axis1"]))
        r = races.get(c["race_key"])
        if e is None or r is None:
            continue
        fo = e["finish_order"]
        dnf = fo is None or int(fo) == 0
        in3 = (not dnf) and 1 <= int(fo) <= 3
        win = (not dnf) and int(fo) == 1
        recs.append({"c": c, "e": e, "r": r, "in3": in3, "win": win, "dnf": dnf,
                     "fo": None if dnf else int(fo)})
    BASE_N = len(recs)
    print(f"属性が揃ったレース: {BASE_N}件\n")

    # ---------------------------------------------------------------- ①内訳
    n_dnf = sum(1 for x in recs if x["dnf"])
    n_in3 = sum(1 for x in recs if x["in3"])
    n_lose = BASE_N - n_dnf - n_in3
    print("【① 軸1の結果の内訳】")
    print(f"  3着内            {n_in3:7d} ({100.0*n_in3/BASE_N:5.1f}%)")
    print(f"  完走4着以下       {n_lose:7d} ({100.0*n_lose/BASE_N:5.1f}%)  ← 外れの本体")
    print(f"  欠車・失格・落車   {n_dnf:7d} ({100.0*n_dnf/BASE_N:5.1f}%)")
    print(f"  → 外れ {100.0*(BASE_N-n_in3)/BASE_N:.1f}% のうち "
          f"欠車失格は {100.0*n_dnf/(BASE_N-n_in3):.1f}%、"
          f"残り {100.0*n_lose/(BASE_N-n_in3):.1f}% は走った上で負けている")
    print()
    fo_dist = defaultdict(int)
    for x in recs:
        fo_dist[x["fo"] if x["fo"] else 0] += 1
    print("  着順分布: " + "  ".join(
        f"{k if k else 'DNF'}着 {100.0*v/BASE_N:.1f}%" for k, v in sorted(fo_dist.items())))
    print()

    def g(keyfn) -> dict:
        out = defaultdict(list)
        for x in recs:
            k = keyfn(x)
            if k is not None:
                out[k].append((x["in3"], x["win"], x["dnf"]))
        return out

    # ---------------------------------------------------------------- ②予測確率
    report("② 予測確率 p1 帯別（較正の確認・どの帯で落としているか）",
           g(lambda x: f"p1 {int(x['c']['top3_probs'][x['c']['axis1']]*10)*10:2d}"
                       f"〜{int(x['c']['top3_probs'][x['c']['axis1']]*10)*10+10:3d}%"))

    # ---------------------------------------------------------------- ③脚質
    report("③ 軸1の脚質（style）別", g(lambda x: x["e"]["style"]))

    # ---------------------------------------------------------------- ④ライン
    report("④ 軸1のライン内位置別",
           g(lambda x: ("単騎" if (x["e"]["line_size"] or 1) == 1
                        else f"ライン{x['e']['line_size']}車の{x['e']['line_pos']}番手")))
    report("⑤ ライン本数（n_lines）別", g(lambda x: f"{x['e']['n_lines']}ライン"))

    # ---------------------------------------------------------------- ⑥級班
    report("⑥ 軸1の級班別", g(lambda x: x["e"]["player_class"]))

    # ---------------------------------------------------------------- ⑦レース属性
    report("⑦ グレード別", g(lambda x: x["r"]["grade"]))
    report("⑧ レース種別（上位10）", g(lambda x: x["r"]["race_type"]), min_n=500)
    report("⑨ バンク周長別", g(lambda x: f"{x['r']['distance']}m"))

    # ---------------------------------------------------------------- ⑩WT印
    report("⑩ 軸1とWT公式印の一致", g(lambda x: {
        1: "◎(honmei)", 2: "◯(taikou)", 3: "△(ana)"}.get(x["e"]["prediction_mark"], "無印")))

    # ---------------------------------------------------------------- ⑪展開
    print("【⑪ レースの決着（勝者の決まり手）別に見た軸1の成績】")
    win_factor: dict[str, list] = defaultdict(list)
    for x in recs:
        rk = x["c"]["race_key"]
        w = x["c"]["order3"][0]
        we = entries.get((rk, w))
        f = (we or {}).get("factor") or "不明"
        win_factor[f].append((x["in3"], x["win"], x["dnf"]))
    for k, v in sorted(win_factor.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 200:
            print(bucket(k, v))
    print()

    # ---------------------------------------------------------------- ⑫飛んだとき
    print("【⑫ 軸1が3着内を外したレースで、代わりに3着内に入った選手のモデル評価順位】")
    miss = [x for x in recs if not x["in3"]]
    rank_cnt = defaultdict(int)
    tot = 0
    for x in miss:
        c = x["c"]
        order = sorted(c["top3_probs"], key=lambda f: -c["top3_probs"][f])
        pos = {f: i for i, f in enumerate(order)}
        for f in c["order3"]:
            rank_cnt[pos.get(f, 99)] += 1
            tot += 1
    print(f"  対象 {len(miss)}レース・3着内枠 {tot}件")
    for i in sorted(rank_cnt):
        if i < 7:
            print(f"    モデル評価{i+1}位: {100.0*rank_cnt[i]/tot:5.1f}%")
    print()

    # ------------------------------------------------------ ⑭層別の較正誤差
    print("【⑭ 層別の較正誤差 — モデルが系統的に誤っている層はどこか】")
    print("  乖離 = 実測3着内率 − モデル予測p1の平均。")
    print("  全体の較正は正確（±1pt以内）なので、ここで大きく振れる層が"
          "『モデルが見落としている構造』の候補になる。")
    print(f"  {'層':30} {'n':>7} {'予測':>7} {'実測':>7} {'乖離':>8} {'±2SE':>7}")

    def calib(name: str, keyfn, min_n: int = 300) -> None:
        groups: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for x in recs:
            k = keyfn(x)
            if k is None:
                continue
            p = x["c"]["top3_probs"][x["c"]["axis1"]]
            groups[str(k)].append((p, x["in3"]))
        out = []
        for k, v in groups.items():
            if len(v) < min_n:
                continue
            n = len(v)
            pm = sum(p for p, _ in v) / n
            am = sum(1 for _, h in v if h) / n
            se = (am * (1 - am) / n) ** 0.5
            out.append((am - pm, k, n, pm, am, 2 * se))
        out.sort()
        if out:
            print(f"  -- {name} --")
        for d, k, n, pm, am, se2 in out:
            flag = " ★" if abs(d) > se2 else ""
            print(f"  {k:30} {n:7d} {100*pm:6.1f}% {100*am:6.1f}% "
                  f"{100*d:+7.1f}pt {100*se2:6.1f}pt{flag}")

    calib("脚質", lambda x: x["e"]["style"])
    calib("ライン内位置", lambda x: ("単騎" if (x["e"]["line_size"] or 1) == 1
                                     else f"ライン{x['e']['line_size']}車の"
                                          f"{x['e']['line_pos']}番手"))
    calib("ライン本数", lambda x: f"{x['e']['n_lines']}ライン")
    calib("級班", lambda x: x["e"]["player_class"])
    calib("バンク周長", lambda x: f"{x['r']['distance']}m")
    calib("グレード", lambda x: x["r"]["grade"])
    calib("WT印", lambda x: {1: "◎", 2: "◯", 3: "△"}.get(x["e"]["prediction_mark"], "無印"))
    calib("レース種別", lambda x: x["r"]["race_type"], min_n=1000)
    print("  ★ = 乖離が2標準誤差を超える（偶然では説明しにくい）")
    print()

    # ------------------------------------------------------ ⑮1着予測の較正
    print("【⑮ 1着予測（pred_win）の較正 — 「勝ち切れる1車」の観点】")
    print(f"  {'pred_win帯':16} {'n':>7} {'予測':>7} {'実測1着率':>10} {'乖離':>8}")
    wb: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for x in recs:
        c = x["c"]
        wp = c["win_probs"][c["axis1"]]
        wb[min(int(wp * 10), 9)].append((wp, x["win"]))
    for b in sorted(wb):
        v = wb[b]
        if len(v) < 100:
            continue
        pm = sum(p for p, _ in v) / len(v)
        am = sum(1 for _, h in v if h) / len(v)
        print(f"  {b*10:3d}〜{b*10+10:3d}%     {len(v):7d} {100*pm:6.1f}% "
              f"{100*am:9.1f}% {100*(am-pm):+7.1f}pt")
    print()

    print("【⑬ 軸1が3着内を外したレースの勝者は、事前に何番人気(モデル評価)だったか】")
    w_rank = defaultdict(int)
    for x in miss:
        c = x["c"]
        order = sorted(c["top3_probs"], key=lambda f: -c["top3_probs"][f])
        pos = {f: i for i, f in enumerate(order)}
        w_rank[pos.get(c["order3"][0], 99)] += 1
    for i in sorted(w_rank):
        if i < 7:
            print(f"    モデル評価{i+1}位: {100.0*w_rank[i]/len(miss):5.1f}%")


if __name__ == "__main__":
    main()
