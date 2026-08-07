"""高配当帯（三連単 N倍以上）の市場ミスプライシングをレース構造で切る。

## 問いの立て方（ここが従来と違う）

[[keirin_highpay_payout_ceiling_2026_08_06]] は「高配当レースを**当てられるか**」を
測った（`exp_highpay_race_model.py`・AUC 0.5573 で失敗）。本スクリプトが測るのは
「高配当帯に**市場が正しい値段を付けているか**」＝相対量である。

    ratio = 実測発生率 ÷ 市場含意確率        市場含意 = Σ_{o_i >= THR} 0.75 / o_i
    帯ROI = ratio × 75%

ratio = 1.0 なら控除率どおり（＝市場は正しい）。1.0 未満なら**その帯が買われ過ぎ**。
モデルの絶対精度が低くても、市場がそれ以上に鈍ければ相対的な妙味は存在しうる
——[[keirin_upset_folklore_market_test_2026_07_30]] の ratio 枠組みと同じ発想を、
車単位ではなく**「レース全体の裾」**に当てたもの。

## 検証窓（宣言・2026-08-08）

| 窓 | 期間 | 使い方 |
|---|---|---|
| 掃引窓 | 2024-01-01 〜 | 閾値（中央値）を決める。探索で既に何度も見ている |
| **確認窓** | **〜2023-12-31** | **一度きり**。`pred_top3_pct` のバックフィルが 2024-01 以降しか無いため、探索では一度も触っていない |

⚠️ そのため確認窓で検査できるのは **`rp_std` / `max_line` / `n_lines` / `n_solo`
＝出走表だけから作れる特徴**に限る（`p3max` などモデル出力は掃引窓のみ）。
これは制約であると同時に、**モデル不要で成立するか**という一段強い問いでもある。

## 結果（2026-08-08・最終オッズ・規則は `rp_std低 ∧ max_line<=3`＝出走表のみ）

7車（確認窓 n≈23,000）。**3つの閾値すべてで、確認窓が掃引窓とほぼ同じ幅で再現した**:

| 帯 | 窓 | 全体 ratio | 選別後 ratio | 帯ROI | Δratio 95%CI | 月次一貫性 |
|---|---|---|---|---|---|---|
| 300倍+ | 掃引 | 0.804 | 0.901 | 60.3→67.6% | +0.098 [+0.076, +0.118] | 27/32 |
| 300倍+ | **確認** | 0.830 | **0.927** | 62.3→**69.5%** | **+0.097 [+0.062, +0.130]** | **13/13** |
| 500倍+ | 掃引 | 0.759 | 0.875 | 56.9→65.6% | +0.116 [+0.088, +0.147] | 27/32 |
| 500倍+ | **確認** | 0.772 | **0.879** | 57.9→**65.9%** | **+0.107 [+0.056, +0.158]** | 12/13 |
| 1000倍+ | 掃引 | 0.670 | 0.786 | 50.2→59.0% | +0.116 [+0.071, +0.167] | 25/32 |
| 1000倍+ | **確認** | 0.691 | **0.803** | 51.8→**60.2%** | **+0.113 [+0.041, +0.191]** | 9/13 |

9車（500倍+）は方向こそ一致するが確認窓 n=2,248 では有意にならない:
掃引 0.840→0.916（+0.075 [+0.022, +0.135]・25/31ヶ月）/
確認 0.820→0.873（+0.053 [−0.034, +0.144]・7/13ヶ月）。

🔴 **モデルを一切使っていない**。`race_point` の分散と最大ライン長＝出走表だけ。
`p3max`（モデル3着内率の最大値）を足すと掃引窓で ratio がさらに +0.03 上がるが、
確認窓ではバックフィルが無く検証できていない。

🔴 **機序**: 市場含意（imp）は選別してもほとんど動かない（500倍+ で 6.74%→6.73%）
のに、実測発生率だけが動く（5.11%→5.89%）。つまり
**市場は「番組がフラットかどうか」を高配当帯の値段に反映していない**。
[[keirin_highpay_payout_ceiling_2026_08_06]] で「高配当レースの予測は不能
（AUC 0.5573）」と結論したのと矛盾しない——**予測が弱いのはこちらも同じで、
市場はもっと弱い**というだけ。相対量で見て初めて出る。

⚠️ **深追いするほど悪くなる**。選別後でも 300倍+ 69.5% → 1000倍+ 60.2%。
「もっと高い配当を狙う」ことは、それ自体が -9pt の追加コスト。

🔴 **機序**: 市場含意（imp）は選別してもほとんど動かない（6.74%→6.73%）のに、
実測発生率だけが動く（5.11%→5.89%）。つまり
**市場は「番組がフラットかどうか」を高配当帯の値段に反映していない**。
[[keirin_highpay_payout_ceiling_2026_08_06]] で「高配当レースの予測は不能
（AUC 0.5573）」と結論したのと矛盾しない——**予測が弱いのはこちらも同じで、
市場はもっと弱い**というだけ。相対量で見て初めて出る。

⚠️ **黒字にはならない**。最良でも帯ROI 65〜72% で控除率75%の壁の下。
使い道は「高配当を狙う買い目に重ねるレース選別」「この条件を外したレースでは
高配当を狙わない足切り」であって、単独ランクには足りない。

⚠️ **これは帯を丸ごと買ったときの理論値**。実装可能な点数（1〜10点）へ落とす
には帯内でどの目を買うか決める必要があり、`exp_highpay_trifecta_design.py` で
**帯内のモデル選択はランダムにも負ける**と確定している。オッズ昇順（＝要求
ラインぎりぎり）で拾うのが算術上の最適。

## 使い方

    .venv/bin/python scripts/exp_highpay_tail_mispricing.py --n-entries 7 --thr 500
    .venv/bin/python scripts/exp_highpay_tail_mispricing.py --n-entries 9 --thr 300 --boot 2000

DB へは書き込まない。オッズは `wt_odds`（最終オッズ）を使う。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import get_connection  # noqa: E402

#: 掃引窓の開始日。これ以降が「閾値を決めるのに使ってよい」期間。
SWEEP_FROM = "2024-01-01"

#: winticket の「オッズ未確定」センチネル 9999.9 を捨てる上限（既存 exp 群と同一）。
ODDS_MAX = 9000.0


def _fetch(n_entries: int, thr: float) -> list[dict]:
    """レース単位の (構造特徴, 帯の市場含意確率, 帯で決着したか) を返す。

    `imp` は「オッズ THR 以上の目をすべて買ったときの的中確率の市場推定」で、
    控除率 25% を戻すために 0.75 を掛けている（＝ratio 1.0 が控除率どおり）。
    """
    # ⚠️ `wt_odds` は 2,200万行ある。**対象レースの絞り込みは `wt_races` への
    #    JOIN で書くこと**。`race_key IN (SELECT ... FROM CTE)` に書き換えると
    #    プランが崩れて15秒→15分以上になる（2026-08-08 実測）。
    sql = f"""
    WITH fin AS (
      SELECT e.race_key,
             max(CASE WHEN e.finish_order=1 THEN e.frame_no END) f1,
             max(CASE WHEN e.finish_order=2 THEN e.frame_no END) f2,
             max(CASE WHEN e.finish_order=3 THEN e.frame_no END) f3
      FROM wt_entries e JOIN wt_races r USING(race_key)
      WHERE r.cancel=0 AND r.n_entries={n_entries} GROUP BY e.race_key),
    st AS (
      SELECT e.race_key,
             count(DISTINCT e.line_group)                        n_lines,
             sum(CASE WHEN e.line_size=1 THEN 1 ELSE 0 END)      n_solo,
             max(e.line_size)                                    max_line,
             stddev_pop(e.race_point)                            rp_std,
             max(e.race_point) - min(e.race_point)               rp_range,
             max(e.pred_top3_pct)                                p3max,
             stddev_pop(e.pred_top3_pct)                         p3sd
      FROM wt_entries e JOIN wt_races r USING(race_key)
      WHERE r.cancel=0 AND r.n_entries={n_entries} GROUP BY e.race_key),
    imp AS (
      SELECT o.race_key, sum(0.75/o.odds_value) imp
      FROM wt_odds o JOIN wt_races r USING(race_key)
      WHERE o.bet_type='trifecta' AND o.odds_value>={thr} AND o.odds_value<{ODDS_MAX}
        AND r.cancel=0 AND r.n_entries={n_entries}
      GROUP BY o.race_key)
    SELECT r.race_key, r.race_date, r.venue_id,
           st.n_lines, st.n_solo, st.max_line, st.rp_std, st.rp_range,
           st.p3max, st.p3sd, i.imp, w.odds_value AS win_odds
    FROM wt_races r
    JOIN fin f USING(race_key) JOIN st USING(race_key) JOIN imp i USING(race_key)
    JOIN wt_odds w ON w.race_key=r.race_key AND w.bet_type='trifecta'
         AND w.combination=concat(f.f1,'-',f.f2,'-',f.f3)
         AND w.odds_value>0 AND w.odds_value<{ODDS_MAX}
    WHERE r.cancel=0 AND r.n_entries={n_entries}
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [dict(x) for x in cur.fetchall()]
    out = []
    for r in rows:
        if r["imp"] is None or r["rp_std"] is None:
            continue
        out.append({
            "date": r["race_date"], "venue": r["venue_id"],
            "n_lines": int(r["n_lines"] or 0), "n_solo": int(r["n_solo"] or 0),
            "max_line": int(r["max_line"] or 0),
            "rp_std": float(r["rp_std"]), "rp_range": float(r["rp_range"] or 0.0),
            "p3max": float(r["p3max"]) if r["p3max"] is not None else np.nan,
            "p3sd": float(r["p3sd"]) if r["p3sd"] is not None else np.nan,
            "imp": float(r["imp"]),
            "hit": 1.0 if float(r["win_odds"]) >= thr else 0.0,
        })
    return out


def _ratio(rows: list[dict]) -> tuple[float, float, float]:
    """(ratio, 実測発生率, 市場含意) を返す。rows が空なら nan。"""
    if not rows:
        return (np.nan, np.nan, np.nan)
    hit = float(np.mean([r["hit"] for r in rows]))
    imp = float(np.mean([r["imp"] for r in rows]))
    return (hit / imp if imp else np.nan, hit, imp)


def _boot_delta(sel: list[dict], base: list[dict], n_boot: int, seed: int = 0) -> tuple:
    """Δratio(選別 − 全体) の**日ブロック** bootstrap 95%CI。

    レースは同一開催日の中で強く相関する（同じ番組・同じ客層）ので、
    レース単位ではなく日単位で再標本する。既存の高配当検証と同じ扱い。
    """
    rng = np.random.default_rng(seed)
    by_day: dict[str, list[dict]] = {}
    for r in base:
        by_day.setdefault(r["date"], []).append(r)
    days = list(by_day)
    sel_keys = {id(r) for r in sel}
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(len(days), size=len(days), replace=True)
        b, s = [], []
        for i in pick:
            day = by_day[days[i]]
            b.extend(day)
            s.extend([r for r in day if id(r) in sel_keys])
        if len(s) < 30:
            continue
        deltas.append(_ratio(s)[0] - _ratio(b)[0])
    if not deltas:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(deltas)), float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


def _report(label: str, rows: list[dict], n_min: int = 150) -> None:
    ratio, hit, imp = _ratio(rows)
    if len(rows) < n_min:
        print(f"  {label:<34} n={len(rows):>6}  （標本不足）")
        return
    print(f"  {label:<34} n={len(rows):>6}  実測={hit*100:>5.2f}%  "
          f"市場含意={imp*100:>5.2f}%  ratio={ratio:>5.3f}  帯ROI={ratio*75:>5.1f}%")


def _monthly(rows: list[dict], sel: list[dict]) -> None:
    """月次で「選別が全体を上回った月」の割合。平均が反転を隠すのを防ぐ。"""
    sel_keys = {id(r) for r in sel}
    months: dict[str, list[dict]] = {}
    for r in rows:
        months.setdefault(r["date"][:7], []).append(r)
    win = tot = 0
    for _, mrows in sorted(months.items()):
        s = [r for r in mrows if id(r) in sel_keys]
        if len(s) < 30 or len(mrows) < 60:
            continue
        tot += 1
        if _ratio(s)[0] > _ratio(mrows)[0]:
            win += 1
    if tot:
        print(f"  月次一貫性: 選別が全体を上回った月 {win}/{tot} ({win/tot*100:.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-entries", type=int, default=7, choices=(6, 7, 9))
    ap.add_argument("--thr", type=float, default=500.0, help="高配当の下限オッズ")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap 反復数（0で省略）")
    args = ap.parse_args()

    rows = _fetch(args.n_entries, args.thr)
    sweep = [r for r in rows if r["date"] >= SWEEP_FROM]
    confirm = [r for r in rows if r["date"] < SWEEP_FROM]
    print(f"=== {args.n_entries}車 / 三連単 {args.thr:.0f}倍以上 ===")
    print(f"掃引窓 {SWEEP_FROM}〜 : {len(sweep)}R   "
          f"確認窓 〜{SWEEP_FROM} : {len(confirm)}R（探索で未使用）")

    # 閾値は掃引窓の中央値で固定し、確認窓へはそのまま当てる（再最適化しない）
    rp_cut = float(np.median([r["rp_std"] for r in sweep]))
    p3_cut = float(np.nanmedian([r["p3max"] for r in sweep]))
    print(f"掃引窓で固定した閾値: rp_std<={rp_cut:.3f} / p3max<={p3_cut:.2f} / max_line<=3\n")

    # 出走表だけで作れる規則（確認窓でも評価できる）と、モデル出力を使う規則を分ける
    card_only = {
        "全体": lambda r: True,
        "rp_std 低半分": lambda r: r["rp_std"] <= rp_cut,
        "max_line<=3": lambda r: r["max_line"] <= 3,
        "rp_std低 ∧ max_line<=3": lambda r: r["rp_std"] <= rp_cut and r["max_line"] <= 3,
    }
    model_based = {
        "p3max 低半分": lambda r: r["p3max"] <= p3_cut,
        "rp低 ∧ p3max低": lambda r: r["rp_std"] <= rp_cut and r["p3max"] <= p3_cut,
        "rp低 ∧ p3max低 ∧ line<=3":
            lambda r: (r["rp_std"] <= rp_cut and r["p3max"] <= p3_cut
                       and r["max_line"] <= 3),
    }

    for wname, wrows in (("掃引窓", sweep), ("確認窓（一度きり）", confirm)):
        if not wrows:
            continue
        print(f"--- {wname} ---")
        rules = dict(card_only)
        if not np.isnan(np.nanmax([r["p3max"] for r in wrows] or [np.nan])):
            rules.update(model_based)
        for label, fn in rules.items():
            _report(label, [r for r in wrows if fn(r)])
        print()

    # 主役の規則（出走表のみ）について、有意性と月次一貫性を掃引窓・確認窓の両方で見る
    key = "rp_std低 ∧ max_line<=3"
    fn = card_only[key]
    for wname, wrows in (("掃引窓", sweep), ("確認窓", confirm)):
        if len(wrows) < 500:
            continue
        sel = [r for r in wrows if fn(r)]
        print(f"--- {wname}: 「{key}」の頑健性 ---")
        _monthly(wrows, sel)
        if args.boot:
            m, lo, hi = _boot_delta(sel, wrows, args.boot)
            sig = "有意" if lo > 0 else "有意差なし"
            print(f"  Δratio(選別−全体) = {m:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  → {sig}")
        print()


if __name__ == "__main__":
    main()
