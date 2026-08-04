"""【読み取り専用】的中重視ランク（overlap2 の相手3点・A-2案）を設計するための
honest 候補キャッシュ生成（2026-08-04）。

`scripts/exp_7s7a_volume_cache.py` と同じ月次凍結vintageモデル方式だが、
overlap2 側の絞り込みを設計するのに必要なフィールドを追加で保存する:

  - top3_probs / win_probs  … 相手の並べ替え・順序不一致（7B）判定に必要
  - wt_honmei / wt_taikou / wt_ana … WT公式印。△除外の効果測定に必要
  - trio_legs               … 軸2車+相手x の三連複オッズ（stale=最終オッズ）
  - order3 / tri_perm       … 着順（1-2-3着）と三連単オッズ。ガミ回避のため
                              三連単へ切り替える案を評価するのに必要
  - wide_axis / q_axis / ex_axis … 軸2車のワイド・二車複・二車単オッズ

⚠️ オッズは全て wt_odds＝**最終オッズ**（本番 judge が見る発走15分前とは別）。
   選出条件をオッズ非依存（axis_sum/entropy/pred_prob）に保つ限り選択バイアスは
   入らないが、オッズを条件に使う案を評価する際は stale-odds バイアスに注意。

DB書き込みなし。既存キャッシュ（data/exp_7c_cache）は月単位で skip する。

使い方:
    python scripts/exp_7c_cache.py data/exp_7c_cache 2026-08-03 2025-01
        引数1: 出力ディレクトリ
        引数2: upto（この日までで当月を打ち切る）
        引数3: since（この月より前は生成しない・省略時は全月）
"""
import pickle
import sys
from datetime import date
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.evaluation.backtest_wt import _load_payouts_wt
from src.models.trainer import load_model
from src.preprocessing.feature_wt import build_features_wt, load_raw_data_wt, prepare_X
from src.strategy_wt import (
    rank_7s_field_entropy, rank_7s_select_axis,
    rank_7s_wt_mark3_overlap_n, rank_7s_wt_overlap_n,
)
from src.wt_vintage_config import monthly_windows

from scripts.backfill_7s_rank_wt import _load_trio_boards
from scripts.backfill_s1w_rank_wt import _load_trifecta_boards


def build_month(model_name: str, date_from: str, date_to: str, win_model_name: str) -> list[dict]:
    """`build_candidates_with_lineinfo` と同一ロジック＋保存フィールド拡張。"""
    model = load_model(model_name)
    win_model = load_model(win_model_name)
    df = build_features_wt(load_raw_data_wt(min_date=date_from, max_date=date_to))
    if df.empty:
        return []
    with get_connection() as c:
        ne_map = dict(c.execute(
            "SELECT race_key, n_entries FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        date_map = dict(c.execute(
            "SELECT race_key, race_date FROM wt_races WHERE race_date BETWEEN ? AND ?",
            (date_from, date_to)))
        rks7 = [rk for rk, ne in ne_map.items() if ne and int(ne) == 7]
        fins, marks = {}, {}
        for i in range(0, len(rks7), 900):
            chunk = rks7[i:i + 900]
            q = ("SELECT race_key, frame_no, finish_order, prediction_mark FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, fno, fo, pmv in c.execute(q, chunk):
                if fo is not None and fo >= 1:
                    fins.setdefault(rk, []).append((fo, int(fno)))
                if pmv is not None:
                    marks.setdefault(rk, {})[int(fno)] = int(pmv)
    df = df[df["race_key"].isin(set(rks7))].copy()
    if df.empty:
        return []
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    df["pred_win"] = win_model.predict_proba(X)[:, 1]
    rks = df["race_key"].unique().tolist()
    trio_bd = _load_trio_boards(rks)
    tri_bd = _load_trifecta_boards(rks)
    pm = _load_payouts_wt(rks)

    out = []
    for rk, g in df.groupby("race_key"):
        if ne_map.get(rk) != 7 or len(g) != 7:
            continue
        trio = trio_bd.get(rk)
        if not trio:
            continue
        board = set()
        for k in trio:
            board |= set(k)
        if len(board) != 7:
            continue
        fin = sorted(fins.get(rk, []))
        if len(fin) < 3:
            continue

        win_probs = {int(r.frame_no): float(r.pred_win) for r in g.itertuples(index=False)}
        top3_probs = {int(r.frame_no): float(r.pred_prob) for r in g.itertuples(index=False)}
        sel = rank_7s_select_axis(win_probs, top3_probs)
        if sel is None:
            continue
        axis1, axis2, axis_sum = sel
        entropy = rank_7s_field_entropy(top3_probs)
        if axis1 not in board or axis2 not in board:
            continue
        others = sorted(board - {axis1, axis2})
        if len(others) != 5:
            continue

        order3 = tuple(fno for _, fno in fin[:3])
        actual_top3 = frozenset(order3)
        mk = marks.get(rk, {})
        wt_honmei = next((fno for fno, v in mk.items() if v == 1), None)
        wt_taikou = next((fno for fno, v in mk.items() if v == 2), None)
        wt_ana = next((fno for fno, v in mk.items() if v == 3), None)

        # 三連単は「軸2車+相手x」の全順列（相手ごと6通り）だけ保存すれば足りる。
        tri = tri_bd.get(rk, {})
        tri_perm = {}
        for x in others:
            for p in permutations((axis1, axis2, x)):
                v = tri.get(p)
                if v is not None:
                    tri_perm[p] = v

        out.append({
            "race_key": rk,
            "race_date": date_map.get(rk, ""),
            "axis1": axis1, "axis2": axis2, "axis_sum": axis_sum, "entropy": entropy,
            "others": others,
            "top3_probs": top3_probs, "win_probs": win_probs,
            "trio_legs": {x: trio.get(frozenset({axis1, axis2, x})) for x in others},
            "tri_perm": tri_perm,
            "actual_top3": tuple(sorted(actual_top3)),
            "order3": order3,
            "trio_pay": pm.get(rk, {}).get(("trio", actual_top3), 0),
            "trifecta_pay": pm.get(rk, {}).get(("trifecta", order3), 0),
            "wide_axis": pm.get(rk, {}).get(("quinellaPlace", frozenset({axis1, axis2}))),
            "q_axis": pm.get(rk, {}).get(("quinella", frozenset({axis1, axis2}))),
            "ex_axis": {(axis1, axis2): pm.get(rk, {}).get(("exacta", (axis1, axis2))),
                        (axis2, axis1): pm.get(rk, {}).get(("exacta", (axis2, axis1)))},
            "wt_honmei": wt_honmei, "wt_taikou": wt_taikou, "wt_ana": wt_ana,
            "wt_overlap_n": rank_7s_wt_overlap_n(axis1, axis2, wt_honmei, wt_taikou),
            "wt_mark3_overlap_n": rank_7s_wt_mark3_overlap_n(
                axis1, axis2, wt_honmei, wt_taikou, wt_ana),
        })
    return out


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    upto = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    since = sys.argv[3] if len(sys.argv) > 3 else "0000-00"

    for date_from, date_to, eval_model, win_model in monthly_windows(upto):
        tag = date_from[:7]
        if tag < since:
            continue
        dst = out_dir / f"{tag}.pkl"
        if dst.exists():
            print(f"[skip] {tag}", flush=True)
            continue
        print(f"[build] {tag} {date_from}〜{date_to}", flush=True)
        try:
            rows = build_month(eval_model, date_from, date_to, win_model)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {tag}: {type(e).__name__}: {e}", flush=True)
            continue
        with open(dst, "wb") as f:
            pickle.dump(rows, f)
        print(f"[done] {tag}: {len(rows)}件", flush=True)


if __name__ == "__main__":
    main()
