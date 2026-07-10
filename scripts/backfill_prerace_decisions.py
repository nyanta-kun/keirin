#!/usr/bin/env python3
"""prerace.log から発走前判定を復元し prerace_decisions_*.json を生成する（一回限り）.

対象: 2026-06-27〜2026-07-07（prerace通知の本格稼働後、判定永続化の実装前）。
この期間は「SS/S昇格の通知済みレースが翌朝の採点で見送り誤記される」バグにより
picks_history の 58% が誤記されていた。本スクリプトで判定を復元後、
notify_results_wt.py {date} --silent を再実行して採点を修正する。

復元ルール:
  - rank / 点数: prerace.log の「live判定: 7PLUS_XX (N点)」行（レースごと最終行）
  - 購入レグ: candidates.json の thirds から最終オッズ >= 7.0 の目を採用。
    本数が点数と合わない場合は最終オッズ降順の上位N目（近似・プレフィックス APPROX 出力）。
    ※ 発走前オッズは未保存のため最終オッズで近似する。
  - 条件不成立: decision=skip

実行例（VPS）:
  python3 scripts/backfill_prerace_decisions.py --from 2026-06-27 --to 2026-07-07
  その後: for d in ...; do python3 scripts/notify_results_wt.py $d --silent; done
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

LOG_PATH = Path(__file__).parent.parent / "data" / "logs" / "prerace.log"
PICKS_DIR = Path(__file__).parent.parent / "data" / "picks"
DATA_DIR = Path(__file__).parent.parent / "data"
# 過去日（判定永続化導入前=2026-07-08以前）の再現専用スクリプトのため、当時の運用閾値 7.0 を維持する。
GAMI_THRESHOLD = 7.0

_LINE_RE = re.compile(
    r"\[prerace\] (20\d{6}_\d+_\d+) 候補 → live判定: "
    r"(?:(7PLUS_SS|7PLUS_S) \((\d+)点\)|(条件不成立))"
)


def _load_log_decisions() -> dict[str, tuple[str | None, int | None]]:
    """{race_key: (rank|None, n_points|None)}。None=条件不成立。最終行優先。"""
    out: dict[str, tuple[str | None, int | None]] = {}
    if not LOG_PATH.exists():
        print(f"log not found: {LOG_PATH}", file=sys.stderr)
        return out
    for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        rk = m.group(1)
        if m.group(4):
            out[rk] = (None, None)
        else:
            out[rk] = (m.group(2), int(m.group(3)))
    return out


def _load_candidates(target_date: str) -> dict[str, dict]:
    cands: dict[str, dict] = {}
    for fname in (f"wave_picks_wt_{target_date}_candidates.json",
                  f"wave_picks_wt_{target_date}_night_candidates.json"):
        p = PICKS_DIR / fname
        if p.exists():
            try:
                for c in json.loads(p.read_text(encoding="utf-8")):
                    if c.get("race_key"):
                        cands[c["race_key"]] = c
            except Exception as e:
                print(f"candidates 読み込み失敗 {fname}: {e}", file=sys.stderr)
    return cands


def _final_trio_odds(race_key: str) -> dict[int, float]:
    """{third候補番号は含まない: combo frozenset→odds} ではなく後で組む用に生を返す。"""
    combo_re = re.compile(r"\d+")
    out: dict[frozenset, float] = {}
    with get_connection() as conn:
        for row in conn.execute(
            "SELECT combination, odds_value FROM wt_odds "
            "WHERE bet_type='trio' AND race_key=?", (race_key,)):
            ov = row["odds_value"]
            if ov is None or float(ov) <= 0:
                continue
            parts = combo_re.findall(str(row["combination"]))
            if len(parts) == 3:
                out[frozenset(int(p) for p in parts)] = float(ov)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", default="2026-06-27")
    ap.add_argument("--to", dest="date_to", default="2026-07-07")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_dec = _load_log_decisions()
    d = date.fromisoformat(args.date_from)
    d_to = date.fromisoformat(args.date_to)
    while d <= d_to:
        target = d.isoformat()
        dc = target.replace("-", "")
        cands = _load_candidates(target)
        day_keys = [rk for rk in log_dec if rk.startswith(dc)]
        if not day_keys:
            d += timedelta(days=1)
            continue
        decisions: dict[str, dict] = {}
        n_buy = n_skip = n_approx = 0
        for rk in sorted(day_keys):
            rank, n_pts = log_dec[rk]
            cand = cands.get(rk)
            if rank is None:
                rec = {"decision": "skip", "backfilled": True}
                if cand:
                    rec["pivot1"] = cand.get("pivot1")
                    rec["pivot2"] = cand.get("pivot2")
                decisions[rk] = rec
                n_skip += 1
                continue
            if cand is None:
                print(f"  {rk}: candidates に無いためスキップ（buy {rank} {n_pts}点）",
                      file=sys.stderr)
                continue
            p1, p2 = cand.get("pivot1"), cand.get("pivot2")
            thirds = [int(t) for t in cand.get("thirds", [])]
            odds_map = _final_trio_odds(rk)
            legs = {t: odds_map.get(frozenset({int(p1), int(p2), t})) for t in thirds}
            legs = {t: o for t, o in legs.items() if o}
            ge = [t for t in thirds if legs.get(t, 0) >= GAMI_THRESHOLD]
            approx = False
            if n_pts is not None and len(ge) != n_pts and legs:
                # 最終オッズ降順の上位N目で近似
                ge = [t for t, _ in sorted(legs.items(), key=lambda x: -x[1])[:n_pts]]
                ge = [t for t in thirds if t in ge]  # モデル順を維持
                approx = True
                n_approx += 1
            if not ge:
                print(f"  {rk}: レグ復元不可（オッズ欠損）→ スキップ扱いにせず対象外",
                      file=sys.stderr)
                continue
            decisions[rk] = {
                "decision": "buy",
                "rank": rank,
                "pivot1": p1, "pivot2": p2,
                "thirds": ge,
                "leg_odds": {str(t): legs[t] for t in ge if t in legs},
                "backfilled": True,
                "approx_legs": approx,
            }
            n_buy += 1
        out_path = DATA_DIR / f"prerace_decisions_{target}.json"
        if out_path.exists():
            # 実運用で生成済みの判定は上書きしない
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            merged = {**decisions, **existing}
        else:
            merged = decisions
        print(f"{target}: buy={n_buy} skip={n_skip} approx={n_approx} → {out_path.name}")
        if not args.dry_run:
            out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        d += timedelta(days=1)


if __name__ == "__main__":
    main()
