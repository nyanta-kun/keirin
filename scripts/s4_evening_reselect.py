#!/usr/bin/env python3
"""S4の朝夜統合選出（2026-07-26改定: 件数cap撤廃に伴い単純併合へ簡素化）。

背景: 朝(daily_picks_wt.sh)と夕(evening_picks_wt.sh)は別プロセスとして独立に
S4候補を生成する。夜レースのライン情報は午後まで公開されないための2段階構成。
2026-07-22時点では「重なり1(S)候補を1日合計 S4_DAILY_TOP_N 件に絞る」件数cap
があったため、朝が先着で枠を埋めると夜の優良候補を取りこぼす問題があり、本
スクリプトで朝夜の生候補プールを合算して改めてトリムし直していた。

2026-07-26に件数capそのものを撤廃（axis_sum/entropy の閾値ゲートのみで選出。
strategy_wt.S4_ENTROPY_MAX 参照）したため、朝の一次選出結果と夕方にこの
スクリプトを実行した結果は常に一致する（先着による取りこぼしが構造的に
発生しない）。既に買い判定済みレースが除外される事態も起こらないため、
旧来のロック考慮・プレースホルダ削除ロジックは不要になり削除した。
本スクリプトは朝夜の生プールを合算してゲートを再適用するだけの薄い処理。

evening_picks_wt.sh から wave-picks-wt（夜の部）の直後・write_candidates_wt.py
の前に呼ばれる。

使い方:
    python3 scripts/s4_evening_reselect.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy_wt import s4_evening_reselect


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[s4_evening_reselect] {path.name} 読み込み失敗: {e}", flush=True)
        return []


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")

    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    day_raw = _load_raw(picks_dir / f"wave_picks_wt_{target_date}_s4_raw_candidates.json")
    night_raw = _load_raw(picks_dir / f"wave_picks_wt_{target_date}_night_s4_raw_candidates.json")

    if not day_raw and not night_raw:
        print(f"[s4_evening_reselect] {target_date}: 朝夜とも生候補なし（スキップ）", flush=True)
        return

    final = s4_evening_reselect(day_raw, night_raw)

    day_raw_keys = {c["race_key"] for c in day_raw}
    night_raw_keys = {c["race_key"] for c in night_raw}

    final_day = [c for c in final if c["race_key"] in day_raw_keys]
    final_night = [c for c in final if c["race_key"] in night_raw_keys]
    final_day.sort(key=lambda c: c["axis_sum"])
    final_night.sort(key=lambda c: c["axis_sum"])

    day_selected_path = picks_dir / f"wave_picks_wt_{target_date}_s4_candidates.json"
    with open(day_selected_path, "w", encoding="utf-8") as f:
        json.dump(final_day, f, ensure_ascii=False, indent=2)
    night_path = picks_dir / f"wave_picks_wt_{target_date}_night_s4_candidates.json"
    with open(night_path, "w", encoding="utf-8") as f:
        json.dump(final_night, f, ensure_ascii=False, indent=2)

    print(f"[s4_evening_reselect] {target_date}: 朝{len(final_day)}件+夜{len(final_night)}件"
          f"={len(final)}件", flush=True)


if __name__ == "__main__":
    main()
