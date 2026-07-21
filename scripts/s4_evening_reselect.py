#!/usr/bin/env python3
"""S4（重なり1=Sランク）の夕方最終選出。朝夜の生候補を統合し、既に買い判定済み
（ロック済み）のレースは維持しつつ、未判定分だけ日次合計 S4_DAILY_TOP_N 件へ
トリムする（2026-07-22新設計）。

背景: 朝(daily_picks_wt.sh)と夕(evening_picks_wt.sh)は別プロセスとして独立に
S4候補を生成する。夜レースのライン情報は午後まで公開されないための2段階構成
だが、朝が先着でaxis_sum上位から枠を埋めてしまうと、夜により優れた候補が
あっても取りこぼす（honest全期間検証で朝だけの部でS候補が10件に達する日が
57.2%あった）。本スクリプトは夕方バッチの最後に実行し、朝夜の生候補プール
（各wave-picks-wtが _s4_raw_candidates.json / _night_s4_raw_candidates.json に
保存済み）を合算して改めてaxis_sumランキングし直す。ただし既に買い判定済み
（picks_history に bet_amount>0 で記録済み）のレースは実購入を取り消せない
ため変更しない。

honest全期間バックテスト（exp_s4_6plus6_trim10_locked.py）: ROI120.8%
（朝夜統合の理論上限120.6%とほぼ同等・現行の朝夕別選出の117.7%より改善・
理論上限との選出一致率89.5%）。

evening_picks_wt.sh から wave-picks-wt（夜の部）の直後・write_candidates_wt.py
の前に呼ばれる。

使い方:
    python3 scripts/s4_evening_reselect.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.strategy_wt import s4_evening_reselect


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[s4_evening_reselect] {path.name} 読み込み失敗: {e}", flush=True)
        return []


def _locked_keys(target_date: str) -> set[str]:
    """当日、既に買い判定済み（bet_amount>0）のS4レース（base race_key）集合を返す。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key FROM picks_history "
            "WHERE race_date = ? AND rank = 'SEVEN_S4' AND bet_amount > 0",
            (target_date,),
        ).fetchall()
    return {r[0].split("#")[0] for r in rows}


def _delete_dropped_placeholders(target_date: str, dropped_keys: set[str]) -> None:
    """トリムで外れた未購入プレースホルダ行（bet_amount=0）をDBから削除する。"""
    if not dropped_keys:
        return
    store_keys = [f"{rk}#7S4" for rk in dropped_keys]
    try:
        with get_connection() as conn:
            for sk in store_keys:
                conn.execute(
                    "DELETE FROM picks_history WHERE race_key = ? AND bet_amount = 0",
                    (sk,),
                )
            conn.commit()
    except Exception as e:
        print(f"[s4_evening_reselect] プレースホルダ削除(SQLite) 失敗: {e}", flush=True)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    try:
        import psycopg2  # noqa: PLC0415
        with psycopg2.connect(db_url) as pg_conn:
            with pg_conn.cursor() as cur:
                for sk in store_keys:
                    cur.execute(
                        "DELETE FROM keirin.picks_history WHERE race_key = %s AND bet_amount = 0",
                        (sk,),
                    )
    except Exception as e:
        print(f"[s4_evening_reselect] プレースホルダ削除(VPS) 失敗: {e}", flush=True)


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")

    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    day_raw = _load_raw(picks_dir / f"wave_picks_wt_{target_date}_s4_raw_candidates.json")
    night_raw = _load_raw(picks_dir / f"wave_picks_wt_{target_date}_night_s4_raw_candidates.json")

    if not day_raw and not night_raw:
        print(f"[s4_evening_reselect] {target_date}: 朝夜とも生候補なし（スキップ）", flush=True)
        return

    locked = _locked_keys(target_date)
    final = s4_evening_reselect(day_raw, night_raw, locked)

    day_raw_keys = {c["race_key"] for c in day_raw}
    night_raw_keys = {c["race_key"] for c in night_raw}
    final_keys = {c["race_key"] for c in final}

    final_day = [c for c in final if c["race_key"] in day_raw_keys]
    final_night = [c for c in final if c["race_key"] in night_raw_keys]
    final_day.sort(key=lambda c: c["axis_sum"])
    final_night.sort(key=lambda c: c["axis_sum"])

    # 朝の一次選出済み(#s4_candidates.json)にあったが、今回のトリムで外れた
    # （かつ未購入の）候補を洗い出し、プレースホルダ行を削除する。
    day_selected_path = picks_dir / f"wave_picks_wt_{target_date}_s4_candidates.json"
    day_selected = _load_raw(day_selected_path)
    day_selected_keys = {c["race_key"] for c in day_selected}
    dropped = (day_selected_keys - final_keys) - locked
    if dropped:
        print(f"[s4_evening_reselect] トリムで除外(未購入分): {sorted(dropped)}", flush=True)
        _delete_dropped_placeholders(target_date, dropped)

    with open(day_selected_path, "w", encoding="utf-8") as f:
        json.dump(final_day, f, ensure_ascii=False, indent=2)
    night_path = picks_dir / f"wave_picks_wt_{target_date}_night_s4_candidates.json"
    with open(night_path, "w", encoding="utf-8") as f:
        json.dump(final_night, f, ensure_ascii=False, indent=2)

    print(f"[s4_evening_reselect] {target_date}: 朝{len(final_day)}件+夜{len(final_night)}件"
          f"={len(final)}件（ロック{len(locked)}件・トリム除外{len(dropped)}件）", flush=True)


if __name__ == "__main__":
    main()
