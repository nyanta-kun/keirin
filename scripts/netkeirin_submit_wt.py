#!/usr/bin/env python3
"""S4（SS+/SS/S）候補をnetkeirin「ウマい車券」へ下書き自動入稿する（2026-07-23新設）。

朝バッチ(daily_picks_wt.sh)・夕バッチ(evening_picks_wt.sh)それぞれの候補生成
直後に呼ばれる。候補生成時点で確定しているgate_label（wt_overlap_n由来。
notify_prerace_wt.pyのT-15分判定を待たない）で SS+/SS/S を抽出し、
未入稿のレースのみ netkeirin へ下書き保存する。同一race_idへの再送信は
上書きされるだけなので、朝夕で対象が重複しても無害。

入稿完了後、新規に登録した件数が1件以上あれば1本のDiscordサマリーを送る。
公開は必ずユーザー本人が確認用URLから行う（本スクリプトは自動化しない）。

仕様の根拠: docs/netkeirin-input-api-spec.md

使い方:
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD morning
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD evening
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.netkeirin_client import NetkeirinClient, RACE_AUTH_URL
from src.notify.discord import send
from src.strategy_wt import s4_gate_label

TARGET_GATE_LABELS = ("SS+", "SS", "S")
SESSION_LABEL_JP = {"morning": "午前", "evening": "午後"}


def _load_candidates(target_date: str, session: str) -> list[dict]:
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    suffix = "_night_s4_candidates.json" if session == "evening" else "_s4_candidates.json"
    path = picks_dir / f"wave_picks_wt_{target_date}{suffix}"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[netkeirin_submit] {path.name} 読み込み失敗: {e}", flush=True)
        return []


def _already_submitted(race_keys: list[str]) -> set[str]:
    if not race_keys:
        return set()
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key FROM netkeirin_submissions WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    return {r[0] for r in rows}


def _record_submission(race_key: str, session: str, venue_name: str, race_no: int,
                        gate_label: str, axis1: int, axis2: int, netkeirin_race_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO netkeirin_submissions "
            "(race_key,session,venue_name,race_no,gate_label,axis1,axis2,netkeirin_race_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (race_key, session, venue_name, race_no, gate_label, axis1, axis2, netkeirin_race_id),
        )
        conn.commit()


def _build_title(venue_name: str, race_no: int) -> str:
    # プレースホルダ（タイトル/コメントのルールはユーザー方針で別途検討中）。
    return f"{venue_name}{race_no}R 二軸探偵"


def _build_comment() -> str:
    return ""


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print("使い方: netkeirin_submit_wt.py YYYY-MM-DD morning|evening", file=sys.stderr)
        sys.exit(1)
    target_date, session = args[0], args[1]
    if session not in ("morning", "evening"):
        print("session は morning または evening を指定してください", file=sys.stderr)
        sys.exit(1)
    race_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    candidates = _load_candidates(target_date, session)
    if not candidates:
        print(f"[netkeirin_submit] {target_date} {session}: 候補なし（スキップ）", flush=True)
        return

    targets = []
    for cand in candidates:
        gate_label = s4_gate_label(
            cand.get("wt_overlap_n"), cand.get("axis1_class"), cand.get("axis2_class"),
        )
        if gate_label in TARGET_GATE_LABELS:
            targets.append((cand, gate_label))

    if not targets:
        print(f"[netkeirin_submit] {target_date} {session}: SS+/SS/S該当なし（スキップ）", flush=True)
        return

    already = _already_submitted([c["race_key"] for c, _ in targets])
    pending = [(c, g) for c, g in targets if c["race_key"] not in already]
    if not pending:
        print(f"[netkeirin_submit] {target_date} {session}: 全件入稿済み（スキップ）", flush=True)
        return

    client = NetkeirinClient()
    submitted_counts: dict[str, int] = {"SS+": 0, "SS": 0, "S": 0}
    failures: list[str] = []

    for cand, gate_label in pending:
        race_key = cand["race_key"]
        venue_name = cand.get("venue_name", "?")
        race_no = cand.get("race_no")
        axis1, axis2 = cand.get("axis1"), cand.get("axis2")
        try:
            ok, msg = client.submit_pick(
                race_date=race_date,
                venue_name=venue_name,
                race_no=race_no,
                axis1=axis1,
                axis2=axis2,
                n_entries=7,
                gate_label=gate_label,
                title=_build_title(venue_name, race_no),
                comment=_build_comment(),
            )
        except Exception as e:
            ok, msg = False, f"例外: {e}"

        if ok:
            _record_submission(race_key, session, venue_name, race_no, gate_label, axis1, axis2, msg)
            submitted_counts[gate_label] += 1
            print(f"[netkeirin_submit] 入稿成功 {venue_name}{race_no}R ({gate_label}) → {msg}", flush=True)
        else:
            failures.append(f"{venue_name}{race_no}R({gate_label}): {msg}")
            print(f"[netkeirin_submit] 入稿失敗 {venue_name}{race_no}R ({gate_label}): {msg}", flush=True)

    total = sum(submitted_counts.values())
    session_jp = SESSION_LABEL_JP[session]
    if total > 0:
        breakdown = "・".join(f"{k}{v}件" for k, v in submitted_counts.items() if v > 0)
        msg = (
            f"📮 **[netkeirin入稿完了] {target_date}（{session_jp}）: "
            f"{breakdown}（計{total}件）**\n"
            f"確認: {RACE_AUTH_URL}\n"
            f"内容を確認の上、公開してください。"
        )
        if failures:
            msg += f"\n⚠️ 入稿失敗 {len(failures)}件: " + " / ".join(failures)
        try:
            send(msg, channel="netkeirin")
        except Exception as e:
            print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)
    elif failures:
        try:
            send(
                f"⚠️ **[netkeirin入稿] {target_date}（{session_jp}）: 全{len(failures)}件が入稿失敗**\n"
                + " / ".join(failures),
                channel="netkeirin",
            )
        except Exception as e:
            print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)

    print(f"[netkeirin_submit] {target_date} {session}: 完了（成功{total}件・失敗{len(failures)}件）",
          flush=True)


if __name__ == "__main__":
    main()
