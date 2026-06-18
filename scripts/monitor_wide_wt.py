"""ワイド1点(W12=指数1-2位)の 朝オッズ→直前(確定)オッズ ドリフト監視。

ユーザー方針(2026-06-10): ワイドは朝≥2.5倍で「割安」判定しても確定で2.5未満に
ドリフトし得る(例 前橋1R 朝3.2→確定1.0で的中も±0)。朝と直前(確定)で確認し
しばらく監視する。本スクリプトは日次で各≤6車のW12について
  朝オッズ(wt_odds_snapshot morning) / 確定オッズ(wt_odds) / 的中 / 確定配当
を記録(data/logs/wide_monitor.jsonl 追記)し、ドリフトと「朝公開→確定配当」の
実質ROIを累積集計する。daily/evening 後に実行(picks_history非依存・W12をモデル再計算)。

使い方: python scripts/monitor_wide_wt.py [YYYY-MM-DD] [--min-odds 2.5]
"""
import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.models.trainer import load_model
from src.database import get_connection

LOG = Path(__file__).resolve().parent.parent / "data" / "logs" / "wide_monitor.jsonl"
PLACEHOLDER = 9000.0


def _pair_odds(rows, a, b):
    """[(combination, odds_value)] から順不同ペア {a,b} のオッズ。無ければ None。"""
    tgt = {str(a), str(b)}
    for combo, ov in rows:
        if set(re.split(r"[-=]", str(combo))) == tgt:
            return ov
    return None


def collect_day(target_date: str, min_odds: float):
    model = load_model("lgbm_wt")
    df = build_features_wt(load_raw_data_wt(min_date=target_date, max_date=target_date))
    if df.empty:
        return []
    df["pred_prob"] = model.predict_proba(prepare_X(df))[:, 1]
    recs = []
    with get_connection() as conn:
        for rk, g in df.groupby("race_key"):
            g = g.sort_values("pred_prob", ascending=False)
            n = len(g)
            if not (2 <= n <= 6):
                continue
            fr = g["frame_no"].astype(int).tolist()
            a, b = fr[0], fr[1]
            m_rows = conn.execute(
                "SELECT combination, odds_value FROM wt_odds_snapshot "
                "WHERE race_key=? AND bet_type='quinellaPlace' AND snapshot_type='morning'",
                (rk,)).fetchall()
            f_rows = conn.execute(
                "SELECT combination, odds_value FROM wt_odds "
                "WHERE race_key=? AND bet_type='quinellaPlace'", (rk,)).fetchall()
            morning = _pair_odds(m_rows, a, b)
            final = _pair_odds(f_rows, a, b)
            top3 = {int(r[0]) for r in conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3",
                (rk,)).fetchall()}
            hit = len(top3) == 3 and {a, b}.issubset(top3)
            recs.append({
                "date": target_date, "race_key": rk, "pair": f"{a}-{b}",
                "morning": morning, "final": final,
                "morning_ok": morning is not None and morning < PLACEHOLDER,
                "final_ok": final is not None and final < PLACEHOLDER,
                "resolved": len(top3) == 3,
                "hit": bool(hit),
                "dividend": int(round(final * 100)) if (hit and final and final < PLACEHOLDER) else 0,
            })
    # 朝≥min_odds(かつ朝オッズ確定)で推奨されたもののみ監視対象
    return [r for r in recs if r["morning_ok"] and r["morning"] >= min_odds]


def summarize(rows, label):
    res = [r for r in rows if r["resolved"] and r["final_ok"]]
    if not res:
        print(f"  {label}: 確定データなし"); return
    n = len(res)
    drift = [r["final"] / r["morning"] - 1.0 for r in res]
    stayed = sum(1 for r in res if r["final"] >= 2.5)
    hits = sum(1 for r in res if r["hit"])
    bet = n * 100
    pay = sum(r["dividend"] for r in res)
    avg_drift = sum(drift) / n
    print(f"  {label}: {n}件  的中{hits}({hits/n*100:.0f}%)  "
          f"朝→確定 平均{avg_drift*100:+.0f}%  確定≥2.5維持 {stayed}/{n}({stayed/n*100:.0f}%)  "
          f"朝公開→確定配当ROI {pay/bet*100:.0f}%")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    from datetime import date
    target_date = args[0] if args else date.today().strftime("%Y-%m-%d")
    min_odds = 2.5
    if "--min-odds" in sys.argv:
        min_odds = float(sys.argv[sys.argv.index("--min-odds") + 1])

    rows = collect_day(target_date, min_odds)
    # 追記（race_key重複は最新で置換）
    existing = {}
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line); existing[r["race_key"]] = r
            except Exception:
                pass
    for r in rows:
        existing[r["race_key"]] = r
    LOG.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in existing.values()) + "\n",
                   encoding="utf-8")

    print(f"\n=== ワイド朝→直前(確定)ドリフト監視  {target_date}（朝≥{min_odds}倍の推奨）===")
    for r in sorted(rows, key=lambda x: x["race_key"]):
        st = "的中" if r["hit"] else ("外" if r["resolved"] else "未確定")
        fo = f"{r['final']:.1f}" if r["final_ok"] else "未"
        div = f"(配当{r['dividend']}円)" if r["hit"] else ""
        print(f"  {r['race_key']} W{r['pair']}: 朝{r['morning']:.1f} → 確定{fo}  {st}{div}")
    print(f"  当日 推奨{len(rows)}件")
    print(f"\n--- 累積（{LOG.name}・全{len(existing)}件）---")
    allrows = list(existing.values())
    summarize(allrows, "全体")
    summarize([r for r in allrows if r["resolved"] and r["final_ok"] and r["final"] >= 2.5], "確定≥2.5のみ")


if __name__ == "__main__":
    main()
