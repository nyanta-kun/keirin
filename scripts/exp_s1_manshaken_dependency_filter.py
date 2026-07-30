"""S1: 現行ゲート通過母集団を対象に、セグメントごとの成績が「万車券
（三連単配当10,000円以上＝大穴）が出た回だけで黒字化している」のか、
それとも「万車券抜きでも軸選定の質だけで的中率・ROIが成立している」のかを
分離して分析する。

ユーザー依頼(2026-07-25): 「大穴的中を軸により不要だったレースを分析、
フィルタすることで的中率・ROIの向上ができないか」
→ 本スクリプトでの解釈: 万車券が出ないと当該セグメントのROIが崩壊する
（＝大穴のまぐれに依存している）セグメントを識別し、そうしたセグメントを
事前にdenyフィルタで除外することで、残りの母集団の的中率・ROIが向上するか
（かつ万車券自体の再現率をどれだけ犠牲にするか）を検証する。

exp_s1_segment_deny_analysis.py の cache(/tmp/exp_s1_segment_deny_cache.pkl)
をそのまま再利用する（現行本番ゲート: top3_gap>=0.15 AND 軸勝率<=0.50 通過の
的中+非的中全候補、2024-01-01〜2026-07-22）。

正規プロトコル踏襲: train+val(〜2026-03-31)でセグメント傾向を探索し、
test(2026-04-01〜)で一度だけ評価する。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/exp_s1_manshaken_dependency_filter.py
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_PATH = Path("/tmp/exp_s1_segment_deny_cache.pkl")

TRAIN_VAL_END = "2026-03-31"
MANSHAKEN_MIN = 10000
BUCKET_20X = 2000
BUCKET_30X = 3000
BUCKET_50X = 5000


def _stats(rows):
    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    bet = sum(r["bet"] for r in rows)
    pay = sum(r["pay"] for r in rows)
    pay_excl_man = sum(r["pay"] for r in rows if r["trifecta_pay"] < MANSHAKEN_MIN)
    man_hits = sum(1 for r in rows if r["hit"] and r["trifecta_pay"] >= MANSHAKEN_MIN)
    roi = pay / bet * 100 if bet else 0.0
    roi_excl_man = pay_excl_man / bet * 100 if bet else 0.0
    hit_rate = hits / n * 100 if n else 0.0
    return {
        "n": n, "hits": hits, "hit_rate": hit_rate, "bet": bet, "pay": pay,
        "roi": roi, "roi_excl_man": roi_excl_man, "man_hits": man_hits,
        "man_share_of_pay": (1 - pay_excl_man / pay) * 100 if pay else 0.0,
    }


def _bucket_recall(rows, all_rows, thresh, label):
    base = [r for r in all_rows if r["trifecta_pay"] >= thresh]
    kept = [r for r in rows if r["trifecta_pay"] >= thresh]
    n_base = len(base)
    n_kept = len(kept)
    recall = n_kept / n_base * 100 if n_base else 0.0
    return f"{label}: {n_kept}/{n_base}残存({recall:.1f}%)"


def _segment_table(rows, key_fn, label, min_n=15):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    print(f"\n--- セグメント別（{label}）train+val（〜{TRAIN_VAL_END}）---")
    print(f"{'segment':<16}{'n':>6}{'hit%':>7}{'ROI(実)':>9}{'ROI(万車券抜)':>14}"
          f"{'万車券依存幅':>12}{'万車券数':>8}{'万車券寄与%':>10}")
    segs = []
    for seg in sorted(groups.keys(), key=lambda k: -len(groups[k])):
        sub = groups[seg]
        if len(sub) < min_n:
            continue
        s = _stats(sub)
        gap = s["roi"] - s["roi_excl_man"]
        print(f"{str(seg):<16}{s['n']:>6}{s['hit_rate']:>6.1f}%{s['roi']:>8.1f}%"
              f"{s['roi_excl_man']:>13.1f}%{gap:>11.1f}pt{s['man_hits']:>8}"
              f"{s['man_share_of_pay']:>9.1f}%")
        segs.append((seg, s, gap))
    return segs


def main():
    if not CACHE_PATH.exists():
        print(f"[ERROR] cache not found: {CACHE_PATH}。"
              f"先に exp_s1_segment_deny_analysis.py を実行してcacheを作成してください。")
        return
    with open(CACHE_PATH, "rb") as f:
        all_races = pickle.load(f)

    train_val = [r for r in all_races if r["race_date"] <= TRAIN_VAL_END]
    test = [r for r in all_races if r["race_date"] > TRAIN_VAL_END]
    print(f"収集済みcacheロード: 全候補 {len(all_races)}R "
          f"(train+val={len(train_val)}R / test={len(test)}R)")
    print(f"date range: {min(r['race_date'] for r in all_races)} 〜 "
          f"{max(r['race_date'] for r in all_races)}\n")

    base = _stats(train_val)
    print("=" * 116)
    print(f"[train+val ベースライン] n={base['n']} 的中={base['hits']}({base['hit_rate']:.1f}%) "
          f"投資={base['bet']:,} 回収={base['pay']:,} ROI(実)={base['roi']:.1f}%  "
          f"ROI(万車券抜)={base['roi_excl_man']:.1f}%  "
          f"万車券={base['man_hits']}件(回収の{base['man_share_of_pay']:.1f}%を占める)")
    print("=" * 116)

    # ------------------------------------------------------------------
    # セグメント別: 万車券依存幅(ROI実 - ROI万車券抜)が大きい = そのセグメントは
    # 万車券が出ないとROIが崩壊する「大穴頼み」。逆に依存幅が小さく、かつ
    # ROI(万車券抜)自体がそこそこ高い = 「軸の質だけで成立している」セグメント。
    # ------------------------------------------------------------------
    dims = [
        (lambda r: r["venue_id"], "場(venue_id)"),
        (lambda r: r["grade"], "グレード"),
        (lambda r: r["distance"], "距離"),
        (lambda r: r["axis_n_lines"], "軸レースのライン数"),
        (lambda r: r["axis_line_size"], "軸ライン人数"),
        (lambda r: r["axis_line_pos"], "軸ライン内位置"),
        (lambda r: r["axis_player_class"], "軸級班"),
        (lambda r: r["axis_style"], "軸脚質"),
        (lambda r: r["n_senko"], "レース内逃げ人数"),
    ]

    all_seg_results = []  # (dim_label, seg_value, stats, gap)
    for key_fn, label in dims:
        segs = _segment_table(train_val, key_fn, label)
        for seg, s, gap in segs:
            all_seg_results.append((label, seg, s, gap))

    print(f"\n{'='*116}\n[万車券依存幅ランキング（train+val・n>=15）: 依存幅が大きい順]\n{'='*116}")
    print(f"{'dim':<20}{'segment':<14}{'n':>6}{'ROI(実)':>9}{'ROI(万車券抜)':>14}{'依存幅':>9}")
    for label, seg, s, gap in sorted(all_seg_results, key=lambda x: -x[3])[:15]:
        print(f"{label:<20}{str(seg):<14}{s['n']:>6}{s['roi']:>8.1f}%{s['roi_excl_man']:>13.1f}%{gap:>8.1f}pt")

    print(f"\n{'='*116}\n[万車券非依存(≒軸の質で成立)セグメント: ROI(万車券抜)が高い順・n>=15]\n{'='*116}")
    print(f"{'dim':<20}{'segment':<14}{'n':>6}{'ROI(実)':>9}{'ROI(万車券抜)':>14}{'依存幅':>9}")
    for label, seg, s, gap in sorted(all_seg_results, key=lambda x: -x[2]["roi_excl_man"])[:15]:
        print(f"{label:<20}{str(seg):<14}{s['n']:>6}{s['roi']:>8.1f}%{s['roi_excl_man']:>13.1f}%{gap:>8.1f}pt")

    # ------------------------------------------------------------------
    # denyフィルター候補: train+valで「ROI(万車券抜)が低い(<70%)AND n>=30」の
    # セグメントを機械的に抽出し、それらを除外するフィルターをtestで検証する。
    # ------------------------------------------------------------------
    deny_candidates = defaultdict(set)
    for label, seg, s, gap in all_seg_results:
        if s["n"] >= 30 and s["roi_excl_man"] < 70.0:
            deny_candidates[label].add(seg)

    print(f"\n{'='*116}\n[機械抽出されたdeny候補（train+val・n>=30 AND ROI(万車券抜)<70%）]\n{'='*116}")
    for label, segset in deny_candidates.items():
        print(f"  {label}: {sorted(segset, key=str)}")

    def _key_for_label(label):
        return {
            "場(venue_id)": lambda r: r["venue_id"],
            "グレード": lambda r: r["grade"],
            "距離": lambda r: r["distance"],
            "軸レースのライン数": lambda r: r["axis_n_lines"],
            "軸ライン人数": lambda r: r["axis_line_size"],
            "軸ライン内位置": lambda r: r["axis_line_pos"],
            "軸級班": lambda r: r["axis_player_class"],
            "軸脚質": lambda r: r["axis_style"],
            "レース内逃げ人数": lambda r: r["n_senko"],
        }[label]

    def make_filter(denymap):
        keyfns = {label: _key_for_label(label) for label in denymap}

        def _f(rows):
            out = []
            for r in rows:
                deny = False
                for label, segset in denymap.items():
                    if keyfns[label](r) in segset:
                        deny = True
                        break
                if not deny:
                    out.append(r)
            return out
        return _f

    combined_filter = make_filter(deny_candidates)

    def _report(label, base_rows, sub_rows, base_stats):
        s = _stats(sub_rows)
        if s["n"] == 0:
            print(f"\n■ {label}: 該当0件")
            return
        print(f"\n■ {label}")
        print(f"  n={s['n']}({s['n']/base_stats['n']*100:.1f}%)  "
              f"的中率={s['hit_rate']:.1f}%(元{base_stats['hit_rate']:.1f}%)  "
              f"投資={s['bet']:,}  回収={s['pay']:,}  "
              f"ROI={s['roi']:.1f}%(元{base_stats['roi']:.1f}%)")
        print(f"  再現率: "
              f"{_bucket_recall(sub_rows, base_rows, BUCKET_20X, '20倍+')} / "
              f"{_bucket_recall(sub_rows, base_rows, BUCKET_30X, '30倍+')} / "
              f"{_bucket_recall(sub_rows, base_rows, BUCKET_50X, '50倍+')} / "
              f"{_bucket_recall(sub_rows, base_rows, MANSHAKEN_MIN, '万車券')}")

    print("\n" + "-" * 50 + " train+val（選定用・参考） " + "-" * 50)
    _report("機械抽出フィルター(train+valで選定)", train_val, combined_filter(train_val), base)

    print("\n" + "-" * 50 + " ★ test（一度だけ評価） " + "-" * 50)
    test_base = _stats(test)
    print(f"[testベースライン(フィルターなし)] n={test_base['n']} "
          f"的中={test_base['hits']}({test_base['hit_rate']:.1f}%) "
          f"投資={test_base['bet']:,} 回収={test_base['pay']:,} ROI={test_base['roi']:.1f}%  "
          f"万車券={test_base['man_hits']}件")
    _report("機械抽出フィルター", test, combined_filter(test), test_base)

    # 既存の本番候補（軸級班S1/A1除外）との比較用に単体でも見る
    def f_deny_top_class(rows):
        return [r for r in rows if r["axis_player_class"] not in ("S1", "A1")]

    print("\n" + "-" * 50 + " 参考: 既存フィルター(軸級班S1/A1除外)単体・test " + "-" * 50)
    _report("軸級班S1/A1除外(既存)", test, f_deny_top_class(test), test_base)


if __name__ == "__main__":
    main()
