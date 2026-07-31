#!/usr/bin/env python3
"""S7の朝夜統合選出（2026-07-26改定: entropyゲート通過後、日次合計を
RANK_7S_DAILY_CAP件までentropy昇順でトリムする。ロック考慮あり）。

背景: 朝(daily_picks_wt.sh)と夕(evening_picks_wt.sh)は別プロセスとして独立に
S7候補を生成する。夜レースのライン情報は午後まで公開されないための2段階構成。

2026-07-26に件数capをentropyゲートへ置換した際、日次capも一旦撤廃したが、
デプロイ当日にentropyフィールドを持たない旧形式の生候補JSON経由でゲートが
実質無効化され1日26件という異常が発生（原因は rank_7s_daily_select() 側で修正済み）
したのを機に、ユーザー要望で「朝夕合わせて10レースちょっと」に収める日次cap
RANK_7S_DAILY_CAP（entropy昇順・信頼度の高い順）を再導入した。honest全期間(832日・
2024-01-01〜2026-07-25)ではentropyゲート通過が最大9件/日のため、通常運用では
ほぼ発火しない安全網。

既に買い判定済み（picks_history に bet_amount>0 で記録済み）のレースはトリムで
除外しない（実購入は取り消せないため）。

evening_picks_wt.sh から wave-picks-wt（夜の部）の直後・write_candidates_wt.py
の前に呼ばれる。

netkeirin入稿済み下書きの取り下げ漏れ対応（2026-07-31追加）:
netkeirin_submit_wt.py は「同一(race_key, rank_key)への再送信は上書きされる
だけなので朝夕で対象が重複しても無害」としているが、これは選出対象がずっと
残り続けるケースの話であり、**トリムで選出から完全に外れた場合は例外**。
未購入プレースホルダを削除する際、その race_key が既に netkeirin_submissions
に入稿済み（rank_key: 7SS/7S）であれば、下書きが「幽霊ピック」として残るため
Discordへ警告する（`netkeirin`チャンネル）。通知はベストエフォート
（失敗してもトリム処理自体は継続する）。

使い方:
    python3 scripts/reselect_7s_evening.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.netkeirin_client import RACE_AUTH_URL
from src.notify.discord import send
from src.strategy_wt import rank_7s_evening_reselect

# netkeirin_submit_wt.py の RANK_CONFIGS で file_key="s7" に紐づく表示ランク。
# rank_7s_gate_label() は現在 "S" のみ返す（"SS" は2026-07-31廃止）が、netkeirin側の
# ランク定義自体は残っているため、取り下げ漏れ検知では両方を対象にしておく。
_RANK_7S_NETKEIRIN_RANK_KEYS = ("7SS", "7S")


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[reselect_7s_evening] {path.name} 読み込み失敗: {e}", flush=True)
        return []


def _locked_keys(target_date: str) -> set[str]:
    """当日、既に買い判定済み（bet_amount>0）のS7レース（base race_key）集合を返す。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key FROM picks_history "
            "WHERE race_date = ? AND rank = 'RANK_7S' AND bet_amount > 0",
            (target_date,),
        ).fetchall()
    return {r[0].split("#")[0] for r in rows}


def _find_netkeirin_submitted(dropped_keys: set[str]) -> list[tuple[str, str]]:
    """dropped_keys のうち netkeirin へ既に入稿済み（netkeirin_submissions に
    存在）の (race_key, rank_key) 一覧を返す。DB接続失敗時は例外を送出する
    （呼び出し側の _notify_netkeirin_orphans で握りつぶす）。
    """
    if not dropped_keys:
        return []
    keys = sorted(dropped_keys)
    placeholders = ",".join("?" * len(keys))
    rank_placeholders = ",".join("?" * len(_RANK_7S_NETKEIRIN_RANK_KEYS))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key, rank_key FROM netkeirin_submissions "
            f"WHERE race_key IN ({placeholders}) AND rank_key IN ({rank_placeholders})",
            (*keys, *_RANK_7S_NETKEIRIN_RANK_KEYS),
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _notify_netkeirin_orphans(target_date: str, dropped_keys: set[str]) -> None:
    """netkeirin入稿済みのままトリムで選出から外れたレースをDiscordへ警告する。

    付加機能のため、DB問い合わせ・Discord送信いずれの失敗もログに残すのみで
    例外を外へ伝播させない（本体のトリム処理を止めないため）。
    """
    try:
        submitted = _find_netkeirin_submitted(dropped_keys)
    except Exception as e:
        print(f"[reselect_7s_evening] netkeirin_submissions確認 失敗: {e}", flush=True)
        return
    if not submitted:
        return

    lines = "\n".join(
        f"- {target_date} race_key={rk} rank={rank_key}"
        for rk, rank_key in sorted(submitted)
    )
    msg = (
        "⚠️ **[netkeirin取り下げ漏れ]** netkeirin に入稿済みの下書きが夕方のトリムで"
        "選出から外れました。手動で削除してください。\n"
        f"{lines}\n"
        f"確認: {RACE_AUTH_URL}"
    )
    try:
        send(msg, channel="netkeirin")
    except Exception as e:
        print(f"[reselect_7s_evening] netkeirin取り下げ漏れDiscord通知失敗: {e}", flush=True)


def _delete_dropped_placeholders(target_date: str, dropped_keys: set[str]) -> None:
    """トリムで外れた未購入プレースホルダ行（bet_amount=0）をDBから削除する。

    削除前に、対象レースが netkeirin へ既に入稿済みでないか確認し、入稿済みなら
    Discordへ警告する（`_notify_netkeirin_orphans` 参照・失敗してもここでの
    削除処理は継続する）。
    """
    if not dropped_keys:
        return
    _notify_netkeirin_orphans(target_date, dropped_keys)

    store_keys = [f"{rk}#7S" for rk in dropped_keys]
    try:
        with get_connection() as conn:
            for sk in store_keys:
                conn.execute(
                    "DELETE FROM picks_history WHERE race_key = ? AND bet_amount = 0",
                    (sk,),
                )
            conn.commit()
    except Exception as e:
        print(f"[reselect_7s_evening] プレースホルダ削除(SQLite) 失敗: {e}", flush=True)

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
        print(f"[reselect_7s_evening] プレースホルダ削除(VPS) 失敗: {e}", flush=True)


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")

    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    day_raw = _load_raw(picks_dir / f"wave_picks_wt_{target_date}_s7_raw_candidates.json")
    night_raw = _load_raw(picks_dir / f"wave_picks_wt_{target_date}_night_s7_raw_candidates.json")

    if not day_raw and not night_raw:
        print(f"[reselect_7s_evening] {target_date}: 朝夜とも生候補なし（スキップ）", flush=True)
        return

    locked = _locked_keys(target_date)
    final = rank_7s_evening_reselect(day_raw, night_raw, locked)

    day_raw_keys = {c["race_key"] for c in day_raw}
    night_raw_keys = {c["race_key"] for c in night_raw}
    final_keys = {c["race_key"] for c in final}

    final_day = [c for c in final if c["race_key"] in day_raw_keys]
    final_night = [c for c in final if c["race_key"] in night_raw_keys]
    final_day.sort(key=lambda c: c["axis_sum"])
    final_night.sort(key=lambda c: c["axis_sum"])

    # 朝の一次選出済み(_s7_candidates.json)にあったが、今回のトリムで外れた
    # （かつ未購入の）候補を洗い出し、プレースホルダ行を削除する。
    day_selected_path = picks_dir / f"wave_picks_wt_{target_date}_s7_candidates.json"
    day_selected = _load_raw(day_selected_path)
    day_selected_keys = {c["race_key"] for c in day_selected}
    dropped = (day_selected_keys - final_keys) - locked
    if dropped:
        print(f"[reselect_7s_evening] トリムで除外(未購入分): {sorted(dropped)}", flush=True)
        _delete_dropped_placeholders(target_date, dropped)

    with open(day_selected_path, "w", encoding="utf-8") as f:
        json.dump(final_day, f, ensure_ascii=False, indent=2)
    night_path = picks_dir / f"wave_picks_wt_{target_date}_night_s7_candidates.json"
    with open(night_path, "w", encoding="utf-8") as f:
        json.dump(final_night, f, ensure_ascii=False, indent=2)

    print(f"[reselect_7s_evening] {target_date}: 朝{len(final_day)}件+夜{len(final_night)}件"
          f"={len(final)}件（ロック{len(locked)}件・トリム除外{len(dropped)}件）", flush=True)


if __name__ == "__main__":
    main()
