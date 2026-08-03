"""【読み取り専用】7S/7A の選出規則を再設計するための honest 候補キャッシュ生成。

月次凍結vintageモデル（src.wt_vintage_config.monthly_windows）で全期間の
7車立て生候補（軸選定成功・axis_sum・entropy・wt_overlap_n・実着順・払戻）を
月ごとに pickle へ吐き出す。DB書き込みなし。
"""
import pickle
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wt_vintage_config import monthly_windows
from scripts.exp_s7_ev_threshold_staking_validation import build_candidates_with_lineinfo

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def main():
    upto = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    for date_from, date_to, eval_model, win_model in monthly_windows(upto):
        tag = date_from[:7]
        dst = OUT / f"{tag}.pkl"
        if dst.exists():
            print(f"[skip] {tag}", flush=True)
            continue
        print(f"[build] {tag} {date_from}〜{date_to}", flush=True)
        try:
            cands, pm = build_candidates_with_lineinfo(eval_model, date_from, date_to, win_model)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {tag}: {e}", flush=True)
            continue
        slim = []
        for c in cands:
            rk = c["race_key"]
            slim.append({
                "race_key": rk, "race_date": c["race_date"],
                "axis1": c["axis1"], "axis2": c["axis2"],
                "axis_sum": c["axis_sum"], "entropy": c["entropy"],
                "others": c["others"],
                "trio_legs": {x: c["trio"].get(frozenset({c["axis1"], c["axis2"], x}))
                              for x in c["others"]},
                "actual_top3": tuple(sorted(c["actual_top3"])),
                "trio_pay": pm.get(rk, {}).get(("trio", c["actual_top3"]), 0),
                "wt_overlap_n": c["wt_overlap_n"],
                "wt_mark3_overlap_n": c["wt_mark3_overlap_n"],
            })
        with open(dst, "wb") as f:
            pickle.dump(slim, f)
        print(f"[done] {tag}: {len(slim)}件", flush=True)


if __name__ == "__main__":
    main()
