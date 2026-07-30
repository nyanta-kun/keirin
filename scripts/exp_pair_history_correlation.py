"""【ペアの過去実績】を相関(lift)として検証する（2026-07-30）。

## 背景
既存検証（[[keirin_s7_foundational_rethink_2026_07_29]] 系）で「H2H（頭対戦成績）」
「ライン連携実績」は**モデルの周辺特徴量として**は不採用と判定済み。だが、
それは「pred_top3_pct等への入力特徴」としての検証であり、**ペア相関（2人が
一緒に3着内に来る確率の補正係数）としては未検証**。これを再検証する。

## 測定方法（既存 exp_pair_correlation_structure.py / broad_scan.py と同一の枠組み）
lift = 観測同時3着内率 / 独立仮定の期待値(p_top3(i) * p_top3(j))
lift > 1 = 独立仮定より一緒に来やすい（正の相関）、< 1 = 負の相関（共倒れ）

## point-in-time厳守
各レースの各ペア(i,j)について、**そのレースより前の履歴のみ**を使って
下記1〜5の特徴を計算する。レースを (race_date, race_key) 順に1本ずつ処理し、
特徴計算 → obs記録 → 履歴更新、の順で進める（同日内のレース間の順序は
無視できる: 同じ2選手が同日に複数回対戦する事はほぼ無い）。

履歴の蓄積開始点は 2022-12-01（データ開始日）。評価対象は pred_top3_pct が
格納されている 2024-01-01以降（TRAIN 2024-01-01〜2025-12-31 / TEST
2026-01-01〜2026-07-30）。

## 計算する5種類の特徴
1. 同ライン共演回数（0回/1回/2-3回/4回以上）
2. 同ライン共演時の両者3着内率（実績2回以上のペアのみ。0-33%/34-66%/67-100%）
3. 対戦回数H2H（同ラインか否か問わず同レース出走回数。0/1-2/3-5/6回以上）
4. H2H勝敗の偏り（決着済み対戦3回以上のペアのみ。pair内でplayer_id最小の方の
   勝率 rate_a を計算し <34%/34-66%/>66% で層別。<34%と>66%は同一現象の鏡像
   （どちらかが優勢）であり、34-66%が「互角」）
5. 1と現在の同/別ラインの交差（今回同ライン×過去共演有無、今回別ライン×過去共演有無）

判定: TRAIN/TESTで同方向 かつ 両方でliftが1から0.05以上離れる場合のみ有意と扱う。
n<300の層は非表示。
"""
import sys
from collections import defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

HIST_FROM = "2022-12-01"
TRAIN_FROM, TRAIN_TO = "2024-01-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-07-30"
MIN_N = 300


def load_all():
    print("[load] races ...", flush=True)
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date FROM wt_races "
            "WHERE n_entries = 7 AND cancel = 0 AND race_date BETWEEN ? AND ? "
            "ORDER BY race_date, race_key",
            (HIST_FROM, TEST_TO)).fetchall()
    race_order = [(r["race_key"], str(r["race_date"])) for r in rrows]
    print(f"[load]   races: {len(race_order)}", flush=True)

    print("[load] entries ...", flush=True)
    by_race = defaultdict(list)
    keys = [rk for rk, _ in race_order]
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, player_id, line_group, finish_order, "
                 "       pred_top3_pct FROM wt_entries WHERE race_key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load]   entries races: {len(by_race)}", flush=True)
    return race_order, by_race


def pk(a, b):
    """player_id ペアの正規化キー（昇順タプル）。"""
    return (a, b) if a < b else (b, a)


def bucket_count(n, edges_labels):
    for edge, label in edges_labels:
        if n <= edge:
            return label
    return edges_labels[-1][1]


def same_line_count_bucket(n):
    if n == 0:
        return "0回"
    if n == 1:
        return "1回"
    if n <= 3:
        return "2-3回"
    return "4回以上"


def h2h_count_bucket(n):
    if n == 0:
        return "0回"
    if n <= 2:
        return "1-2回"
    if n <= 5:
        return "3-5回"
    return "6回以上"


def rate_bucket_thirds(rate):
    """0-33/34-66/67-100 の三分割（rateは0-1）。"""
    pct = rate * 100.0
    if pct <= 33.999:
        return "0-33%"
    if pct <= 66.999:
        return "34-66%"
    return "67-100%"


def main():
    race_order, entries_by_race = load_all()

    # ---- 履歴状態（point-in-time で逐次更新） ----
    same_line_hist = {}   # pair_key -> {"n": int, "top3_both": int}
    h2h_hist = {}          # pair_key -> {"n_enc": int, "n_dec": int, "wins": {pid: int}}

    # ---- 評価用の蓄積（バケット別 lift 集計） ----
    def new_agg():
        return defaultdict(lambda: {"n": 0, "obs": 0, "exp": 0.0})

    agg = {
        "TRAIN": {f"m{i}": new_agg() for i in range(1, 6)},
        "TEST": {f"m{i}": new_agg() for i in range(1, 6)},
    }

    n_eval_races = {"TRAIN": 0, "TEST": 0}

    for rk, rdate in race_order:
        ents = entries_by_race.get(rk)
        if not ents or len(ents) != 7:
            continue
        by_frame = {int(e["frame_no"]): e for e in ents}
        frames = list(by_frame.keys())

        # このレースの着順（3着内判定用）
        fin = [(e["finish_order"], int(e["frame_no"])) for e in ents
               if e["finish_order"] is not None and e["finish_order"] >= 1]
        fin.sort()
        valid_top3 = len(fin) >= 3
        top3 = frozenset(fno for _, fno in fin[:3]) if valid_top3 else frozenset()

        # 評価対象期間か（pred_top3_pct 使用可否）
        in_train = TRAIN_FROM <= rdate <= TRAIN_TO
        in_test = TEST_FROM <= rdate <= TEST_TO
        do_eval = (in_train or in_test) and valid_top3 and \
            all(by_frame[f]["pred_top3_pct"] is not None for f in frames)
        split = "TRAIN" if in_train else ("TEST" if in_test else None)

        if do_eval:
            n_eval_races[split] += 1
            for i, j in combinations(frames, 2):
                ei, ej = by_frame[i], by_frame[j]
                pidi, pidj = ei["player_id"], ej["player_id"]
                if pidi is None or pidj is None:
                    continue
                key = pk(pidi, pidj)
                pi = float(ei["pred_top3_pct"]) / 100.0
                pj = float(ej["pred_top3_pct"]) / 100.0
                exp_ij = pi * pj
                obs_ij = 1 if (i in top3 and j in top3) else 0

                sl = same_line_hist.get(key)
                h2 = h2h_hist.get(key)

                # --- 1. 同ライン共演回数 ---
                sl_n = sl["n"] if sl else 0
                b1 = same_line_count_bucket(sl_n)
                a = agg[split]["m1"][b1]
                a["n"] += 1
                a["exp"] += exp_ij
                a["obs"] += obs_ij

                # --- 2. 同ライン共演時の両者3着内率（実績2回以上のみ） ---
                if sl_n >= 2:
                    rate = sl["top3_both"] / sl_n
                    b2 = rate_bucket_thirds(rate)
                    a = agg[split]["m2"][b2]
                    a["n"] += 1
                    a["exp"] += exp_ij
                    a["obs"] += obs_ij

                # --- 3. 対戦回数(H2H) ---
                h2_enc = h2["n_enc"] if h2 else 0
                b3 = h2h_count_bucket(h2_enc)
                a = agg[split]["m3"][b3]
                a["n"] += 1
                a["exp"] += exp_ij
                a["obs"] += obs_ij

                # --- 4. H2H勝敗の偏り（決着済み3回以上のみ） ---
                if h2 and h2["n_dec"] >= 3:
                    a_pid = key[0]  # player_idが小さい方
                    wins_a = h2["wins"].get(a_pid, 0)
                    rate_a = wins_a / h2["n_dec"]
                    b4 = rate_bucket_thirds(rate_a)
                    a = agg[split]["m4"][b4]
                    a["n"] += 1
                    a["exp"] += exp_ij
                    a["obs"] += obs_ij

                # --- 5. 今回ライン関係 × 過去共演有無 の交差 ---
                li, lj = ei["line_group"], ej["line_group"]
                if li is not None and lj is not None:
                    now_same = "今回同ライン" if li == lj else "今回別ライン"
                    hist_flag = "過去共演あり" if sl_n >= 1 else "過去共演なし"
                    b5 = f"{now_same}×{hist_flag}"
                    a = agg[split]["m5"][b5]
                    a["n"] += 1
                    a["exp"] += exp_ij
                    a["obs"] += obs_ij

        # ---- 履歴更新（このレースの実際の結果で） ----
        if valid_top3:
            for i, j in combinations(frames, 2):
                ei, ej = by_frame[i], by_frame[j]
                pidi, pidj = ei["player_id"], ej["player_id"]
                if pidi is None or pidj is None:
                    continue
                key = pk(pidi, pidj)

                li, lj = ei["line_group"], ej["line_group"]
                if li is not None and lj is not None and li == lj:
                    sl = same_line_hist.setdefault(key, {"n": 0, "top3_both": 0})
                    sl["n"] += 1
                    if i in top3 and j in top3:
                        sl["top3_both"] += 1

                h2 = h2h_hist.setdefault(key, {"n_enc": 0, "n_dec": 0, "wins": {}})
                h2["n_enc"] += 1
                foi, foj = ei["finish_order"], ej["finish_order"]
                if foi is not None and foj is not None:
                    h2["n_dec"] += 1
                    winner = pidi if foi < foj else pidj
                    h2["wins"][winner] = h2["wins"].get(winner, 0) + 1

    print(f"\n[main] 評価レース TRAIN={n_eval_races['TRAIN']} TEST={n_eval_races['TEST']}")
    print("lift = 観測同時3着内率 / 独立仮定の期待値。1.0=独立、>1=一緒に来やすい")

    titles = {
        "m1": "1. 同ライン共演回数",
        "m2": "2. 同ライン共演時の両者3着内率（実績2回以上）",
        "m3": "3. 対戦回数(H2H・ライン問わず)",
        "m4": "4. H2H勝敗の偏り（決着3回以上・player_id最小側の勝率）",
        "m5": "5. 今回ライン関係 × 過去共演有無 の交差",
    }

    def summarize(agg_bucket):
        res = {}
        for b, a in agg_bucket.items():
            if a["n"] < MIN_N:
                continue
            obs = a["obs"] / a["n"]
            exp = a["exp"] / a["n"]
            lift = obs / exp if exp > 0 else 0.0
            se = sqrt(max(obs * (1 - obs), 1e-12) / a["n"])
            res[b] = (a["n"], obs, exp, lift, se)
        return res

    verdicts = []
    for mkey, title in titles.items():
        res_tr = summarize(agg["TRAIN"][mkey])
        res_te = summarize(agg["TEST"][mkey])
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        print(f"  {'バケット':<26}{'TRAIN n':>9}{'obs':>8}{'exp':>8}{'lift':>8}"
              f"{'TEST n':>9}{'obs':>8}{'exp':>8}{'lift':>8}{'判定':>14}")
        keys = sorted(set(res_tr) | set(res_te),
                      key=lambda k: -(res_tr.get(k, (0,))[0] + res_te.get(k, (0,))[0]))
        any_row = False
        for b in keys:
            tr = res_tr.get(b)
            te = res_te.get(b)
            if tr is None or te is None:
                print(f"  {str(b):<26}{'n不足(片方のみ)':>60}")
                continue
            any_row = True
            n1, o1, e1, l1, s1 = tr
            n2, o2, e2, l2, s2 = te
            same_dir = (l1 - 1) * (l2 - 1) > 0
            strong = min(abs(l1 - 1), abs(l2 - 1)) >= 0.05
            if same_dir and strong:
                verdict = "★正の相関" if l1 > 1 else "★負の相関"
                verdicts.append((title, b, l1, l2))
            elif same_dir:
                verdict = "同方向(弱)"
            else:
                verdict = "不一致"
            print(f"  {str(b):<26}{n1:>9}{o1*100:>7.1f}%{e1*100:>7.1f}%{l1:>7.3f}x"
                  f"{n2:>9}{o2*100:>7.1f}%{e2*100:>7.1f}%{l2:>7.3f}x{verdict:>14}")
        if not any_row:
            print("  (n>=300のバケットなし)")

    print("\n" + "=" * 96)
    print("結論用サマリ: TRAIN/TESTで同方向かつ両方lift偏差>=0.05の層のみ")
    print("=" * 96)
    if verdicts:
        for title, b, l1, l2 in verdicts:
            print(f"  [{title}] {b}: TRAIN={l1:.3f}x / TEST={l2:.3f}x")
    else:
        print("  該当なし（有意な上乗せ情報は検出されず）")

    # ---- item5 の核心比較: 「今回ライン状態」を固定した上で「過去共演有無」が
    #      上乗せ情報を持つか(あり-なしのlift差・z検定つき) ----
    print("\n" + "=" * 96)
    print("5-核心: 今回のライン状態を固定した上での「過去共演あり/なし」比較")
    print("  （既知のライン効果を差し引いた後に、過去実績が追加情報を持つか）")
    print("=" * 96)
    res5_tr = summarize(agg["TRAIN"]["m5"])
    res5_te = summarize(agg["TEST"]["m5"])
    for now_label in ("今回同ライン", "今回別ライン"):
        b_have = f"{now_label}×過去共演あり"
        b_none = f"{now_label}×過去共演なし"
        rows = []
        for split_name, res in (("TRAIN", res5_tr), ("TEST", res5_te)):
            h = res.get(b_have)
            n_ = res.get(b_none)
            if h is None or n_ is None:
                rows.append((split_name, None))
                continue
            n1, o1, e1, l1, s1 = h
            n2, o2, e2, l2, s2 = n_
            diff_lift = l1 - l2
            diff_obs = o1 - o2
            se_diff = sqrt(s1 ** 2 + s2 ** 2)
            z = diff_obs / se_diff if se_diff > 0 else 0.0
            rows.append((split_name, (n1, l1, n2, l2, diff_lift, diff_obs, z)))
        print(f"\n  [{now_label}] あり vs なし")
        for split_name, vals in rows:
            if vals is None:
                print(f"    {split_name}: データ不足")
                continue
            n1, l1, n2, l2, diff_lift, diff_obs, z = vals
            print(f"    {split_name}: あり(n={n1}) lift={l1:.4f}x / なし(n={n2}) lift={l2:.4f}x "
                  f"/ lift差={diff_lift:+.4f} / 観測率差={diff_obs*100:+.2f}pt / z={z:+.2f}")


if __name__ == "__main__":
    main()
