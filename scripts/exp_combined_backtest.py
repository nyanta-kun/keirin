"""ks + wt 合算バックテスト（ユーザー要望・合算運用の採否判断用）。

両モデルを同一閾値(SS/S/A)・同一レース集合・同一払戻(wt_odds)・同一結果(wt_entries)で評価し、
  - wt単独 / ks単独 / 合算(union) のROI・的中率
  - 昼(〜19時) / 夜(19時〜) の分割（夜はwtライン未公開でksが頑健という仮説の検証）
を比較する。

合算ルール: 同一レースを両モデルが推奨 → 両者の買い目(券種,組合せ)を union（重複除去）して全部購入。
払戻=wt_odds最終オッズ×100＝上限値。結果=wt_entries finish_order between(1,3)（欠車は着外）。
モデル: wt=lgbm_wt_eval / ks=lgbm（いずれもOOS）。テスト期間 2026-03-01〜2026-06-07（ks稼働期）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from src.database import get_connection
from roi_robustness_wt import roi_summary

FROM, TO = "2026-03-01", "2026-06-07"


def _tickets(p, fr, n):
    """SS/S/A 本番ロジックで買い目チケット [(bet_type, key)] を返す。非該当は []。"""
    if n < 3 or len(fr) < 3:
        return []
    gap12 = p[0] - p[1]
    ratio = p[0] / (3.0 / n)
    p1, p2 = fr[0], fr[1]
    thirds = fr[2:5]
    if not thirds:
        return []
    if gap12 >= 0.15 and ratio < 1.3:            # SS: 3連単 1→2→{thirds}
        return [("trifecta", (p1, p2, t)) for t in thirds]
    if gap12 >= 0.15 and ratio < 1.6:            # S: 3連複 2軸流し
        return [("trio", frozenset((p1, p2, t))) for t in thirds]
    if 0.06 <= gap12 < 0.15:                      # A: 3連複 2軸流し
        return [("trio", frozenset((p1, p2, t))) for t in thirds]
    return []


def wt_picks():
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=FROM, max_date=TO))
    df = _apply_pred_prob_wt(model, df)
    out = {}
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n > 6:
            continue
        out[rk] = _tickets(g["pred_prob"].tolist(), g["frame_no"].astype(int).tolist(), n)
    return out


def ks_picks():
    model = load_model("lgbm")
    df = build_features(load_raw_data(min_date=FROM, max_date=TO))
    df = df[df["finish_position"].notna()].copy()
    X = df[FEATURE_COLS]
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    out = {}
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n > 6:
            continue
        out[rk] = _tickets(g["pred_prob"].tolist(), g["frame_no"].astype(int).tolist(), n)
    return out


def main():
    wt = wt_picks()
    ks = ks_picks()
    all_keys = sorted(set(wt) | set(ks))

    # 結果(top3 / 着順) と払戻
    pm = _load_payouts_wt(all_keys)
    results = {}
    with get_connection() as conn:
        # start_at（昼夜判定）
        starts = dict(conn.execute(
            "SELECT race_key, start_at FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (FROM, TO)).fetchall())
        for rk in all_keys:
            rows = conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                "ORDER BY finish_order", (rk,)).fetchall()
            order = [int(r[0]) for r in rows]
            if len(order) >= 3:
                results[rk] = (tuple(order[:3]), frozenset(order[:3]))

    from datetime import datetime, timezone, timedelta
    def is_night(rk):
        st = starts.get(rk)
        if st is None:
            return None
        try:
            h = datetime.fromtimestamp(int(st), tz=timezone(timedelta(hours=9))).hour
            return h >= 19
        except (ValueError, TypeError):
            return None

    def score(tickets, rk):
        """チケット群を採点。返り値 (payout, bet)。"""
        if not tickets:
            return None
        order, top3 = results[rk]
        rp = pm.get(rk, {})
        uniq = list(dict.fromkeys(tickets))   # 重複除去（同一券種・組合せ）
        pay = 0
        for bt, key in uniq:
            if bt == "trifecta" and tuple(key) == order:
                pay += rp.get(("trifecta", tuple(key)), 0)
            elif bt == "trio" and key == top3:
                pay += rp.get(("trio", key), 0)
        return pay, len(uniq) * 100

    # 集計: wt単独 / ks単独 / 合算
    def collect(mode, night_filter=None):
        pays, bets = [], []
        for rk in all_keys:
            if rk not in results:
                continue
            if night_filter is not None and is_night(rk) != night_filter:
                continue
            wt_t = wt.get(rk, []) or []
            ks_t = ks.get(rk, []) or []
            if mode == "wt":
                tk = wt_t
            elif mode == "ks":
                tk = ks_t
            else:
                tk = wt_t + ks_t
            s = score(tk, rk)
            if s is None:
                continue
            pays.append(s[0]); bets.append(s[1])
        return roi_summary(pays, bets), len(pays)

    print(f"\n{'='*84}\n  ks+wt 合算バックテスト  {FROM}〜{TO}（≤6車・払戻=最終オッズ上限値・OOS）\n{'='*84}")
    print(f"  {'区分':<8}{'モード':<8}{'購入R':>7}{'的中率':>8}{'ROI':>8}{'95%CI':>16}{'最大除ROI':>10}")
    for label, nf in [("全体", None), ("昼(〜19時)", False), ("夜(19時〜)", True)]:
        for mode in ["wt", "ks", "combined"]:
            s, n = collect(mode, nf)
            print(f"  {label:<8}{mode:<8}{n:>7}{s['hit_rate']:>8.1%}{s['roi']:>7.0%}"
                  f" [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]{s['roi_ex_max']:>9.0%}")
        print(f"  {'-'*80}")
    # 重複度: 両モデルが同一レースを推奨した割合
    both = sum(1 for rk in all_keys if rk in results and wt.get(rk) and ks.get(rk))
    wt_only = sum(1 for rk in all_keys if rk in results and wt.get(rk) and not ks.get(rk))
    ks_only = sum(1 for rk in all_keys if rk in results and ks.get(rk) and not wt.get(rk))
    print(f"  推奨レース重複: 両方={both}  wtのみ={wt_only}  ksのみ={ks_only}")
    print(f"{'='*84}")


if __name__ == "__main__":
    main()
