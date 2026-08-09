"""推奨レースの結果を確定直後に取得し、レース単位で Discord 通知する（2026-08-04 新設）。

ユーザー要望:
  「レース確定後、レース単位で即時取得・discordへ結果通知できないか。
    ただしスクレイピングのため過負荷にならないようにする必要がある」
  → 「推奨レースのみ対応。その他は定期的に結果更新（現行 intraday のまま）」

## 過負荷にならない理由（むしろ現行より軽い）

現行 `intraday_results_wt.sh` は 15分ごとに `collect-wt --date <今日>` を実行し、
`_get_collected_keys`（finish_order>=1 をスキップ）以外の**未確定レースを全部**取得する。
つまり**まだ発走していないレースまで15分おきに叩いている**。

  現行  : 60回/日 × 平均40R × 2req ≒ 4,800 req/日
  本script: 推奨13R × 最大4回 × 2req ≒ 100 req/日（実際は初回で取れるものが多い）

本スクリプトは**発走時刻を過ぎた推奨レースだけ**を、経過時間が
CHECK_MINUTES のいずれかに一致した分だけ取得する（バックオフ）。
毎分 cron で起動しても、対象が無ければ1件も通信しない。

## 設計

- 対象 = 当日の picks_history に存在するレース（＝推奨を出したレース）で結果未確定のもの
- 発走からの経過分が CHECK_MINUTES に一致したときだけ fetch
  （競輪は発走から確定まで概ね5分前後。6分で大半が取れ、遅延分を10/15/25分で拾う）
- 取得できたら Discord（results チャンネル）へ 1レース1通知
- 二重通知は notified テーブルではなくログ用 JSON で抑止（DBスキーマを増やさない）

多重起動は flock（呼び出し側シェル）ではなく本体の PID ロックで防ぐ
（macOS に flock が無い前例があるため Python 側で完結させる）。

使い方:
    python scripts/notify_race_result_wt.py [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.notify.discord import send
from src.scraper.pipeline_wt import _save_batch
from src.scraper.winticket import WinticketScraper

JST = timezone(timedelta(hours=9))
# 発走からの経過分。競輪は発走〜確定が概ね5分前後なので6分で大半が取れる。
# 以降は遅延（写真判定・失格審議など）を拾うためのバックオフ。
CHECK_MINUTES = (6, 10, 15, 25)
STATE = Path(__file__).resolve().parent.parent / "data" / "notified_race_results.json"
LOCK = Path(__file__).resolve().parent.parent / "data" / "notify_race_result.lock"
MARK = {1: "◎", 2: "◯", 3: "△", 4: "×"}


def _acquire_lock() -> bool:
    """PIDロック。前回プロセスが生きていれば False。"""
    try:
        if LOCK.exists():
            pid = int(LOCK.read_text().strip() or 0)
            if pid and pid != os.getpid():
                try:
                    os.kill(pid, 0)
                    return False          # 生きている
                except OSError:
                    pass                  # 死んでいる → 奪う
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(str(os.getpid()))
        return True
    except Exception:
        return True                       # ロック機構の失敗で処理を止めない


def _load_state() -> set[str]:
    try:
        return set(json.loads(STATE.read_text()))
    except Exception:
        return set()


def _save_state(s: set[str]) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(sorted(s), ensure_ascii=False))
    except Exception as e:
        print(f"[warn] 状態保存に失敗: {e}", flush=True)


def _targets(date: str, now_ts: int) -> list[dict]:
    """当日の推奨レースのうち、発走からの経過分が CHECK_MINUTES に一致し
    かつ結果未確定のものを返す。"""
    with get_connection() as c:
        rows = c.execute(
            """
            SELECT DISTINCT split_part(p.race_key, '#', 1) AS base,
                   r.venue_id, r.race_no, r.start_at, r.cup_id, r.day_index,
                   COALESCE(v.name, r.venue_id) AS venue_name
            FROM picks_history p
            JOIN wt_races r ON r.race_key = split_part(p.race_key, '#', 1)
            LEFT JOIN venue_info v ON v.venue_code = r.venue_id
            WHERE p.race_date = ?
            """,
            (date,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r) if not isinstance(r, dict) else r
            try:
                start = int(d["start_at"])
            except (TypeError, ValueError):
                continue
            elapsed = (now_ts - start) // 60
            if elapsed not in CHECK_MINUTES:
                continue
            done = c.execute(
                "SELECT COUNT(*) AS n FROM wt_entries "
                "WHERE race_key = ? AND finish_order >= 1", (d["base"],)
            ).fetchone()
            n = (done["n"] if isinstance(done, dict) else done[0]) or 0
            if n > 0:
                continue                   # 既に確定済み
            d["elapsed"] = elapsed
            out.append(d)
    return out


def _fetch_one(scraper: WinticketScraper, t: dict, date: str) -> dict | None:
    data = scraper.fetch_race_data(
        t["venue_id"], date, int(t["race_no"]),
        cup_id=t["cup_id"], day_index=t["day_index"])
    if not data:
        return None
    if not any(e.get("finish_order") is not None for e in data.get("entries", [])):
        return None                        # まだ結果が載っていない
    try:
        data["odds"] = scraper.fetch_odds(
            t["venue_id"], date, int(t["race_no"]), t["cup_id"], t["day_index"]) or {}
    except Exception:
        data["odds"] = {}
    return data


def _build_message(t: dict, base: str) -> str:
    """着順と、そのレースに出していた推奨の的中可否をまとめる。"""
    with get_connection() as c:
        ents = c.execute(
            "SELECT frame_no, name, prediction_mark, finish_order "
            "FROM wt_entries WHERE race_key = ? ORDER BY finish_order", (base,)
        ).fetchall()
        picks = c.execute(
            "SELECT race_key, rank, pred_combo FROM picks_history "
            "WHERE split_part(race_key, '#', 1) = ?", (base,)
        ).fetchall()

    def _g(r, k):
        return r[k] if isinstance(r, dict) else r[list(r.keys()).index(k)]

    top3 = [(int(_g(e, "frame_no")), _g(e, "name"), _g(e, "prediction_mark"))
            for e in ents
            if _g(e, "finish_order") and 1 <= int(_g(e, "finish_order")) <= 3]
    order3 = tuple(f for f, _, _ in top3)   # 着順（1着,2着,3着）。三連単の判定に要る
    order = " − ".join(
        f"**{f}** {n}{MARK.get(m, '')}" for f, n, m in top3)
    top3_set = {f for f, _, _ in top3}

    lines = [f"🏁 **{t['venue_name']}{t['race_no']}R 確定**",
             f"着順: {order}"]
    for p in picks:
        rank = _g(p, "rank").replace("RANK_", "")
        combo = _g(p, "pred_combo") or ""
        # 🔴 `三単:` 付きは**着順まで当てて初めて的中**（7C の三連単切替・2026-08-09）。
        #    順不同で判定すると着順違いを的中として通知してしまう。
        #    表記は `三単:{1着}-{2着}-{3着候補,...}` で、軸の区切りが `=` ではなく `-`。
        is_tf = combo.startswith("三単:")
        body = combo.split(":", 1)[1] if ":" in combo else combo
        axis_part = body.split("-")[0] if "-" in body else ""
        legs = ([int(x) for x in body.split("-", 1)[1].split(",")
                 if x.strip().isdigit()] if "-" in body else [])
        if is_tf:
            head = [int(x) for x in body.split("-")[:2] if x.strip().isdigit()]
            legs = ([int(x) for x in body.split("-")[2].split(",")
                     if x.strip().isdigit()] if len(body.split("-")) >= 3 else [])
            axes = head
            hit = (len(order3) == 3 and len(head) == 2
                   and order3[0] == head[0] and order3[1] == head[1]
                   and order3[2] in legs)
            n_in = len(set(axes) & top3_set)
        else:
            axes = [int(x) for x in axis_part.replace("=", ",").split(",")
                    if x.strip().isdigit()]
            n_in = len(set(axes) & top3_set)
            third = list(top3_set - set(axes))
            hit = n_in == 2 and len(third) == 1 and third[0] in legs
        if hit:
            mark = "🎯 **的中**"
        elif is_tf and n_in == 2:
            mark = "😖 軸2車は3着内・着順/相手外し"
        elif n_in == 2:
            mark = "😖 軸的中・相手外し"
        else:
            mark = f"❌ 不的中（軸{n_in}/2）"
        lines.append(f"{rank}: {combo}  → {mark}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(JST).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not _acquire_lock():
        return
    now_ts = int(time.time())
    targets = _targets(args.date, now_ts)
    if not targets:
        return                              # 対象なし＝1件も通信しない

    print(f"[notify_race_result] 対象 {len(targets)}件", flush=True)
    done = _load_state()
    scraper = WinticketScraper()
    for t in targets:
        base = t["base"]
        try:
            data = _fetch_one(scraper, t, args.date)
        except Exception as e:
            print(f"[warn] {base} 取得失敗: {e}", flush=True)
            continue
        if not data:
            print(f"[info] {base} まだ結果なし（発走+{t['elapsed']}分）", flush=True)
            continue
        if args.dry_run:
            print(f"[dry-run] {base} 結果取得（保存・通知はしない）", flush=True)
            continue
        _save_batch([data])
        if base in done:
            continue                        # 二重通知の抑止
        try:
            send(_build_message(t, base), channel="results")
            done.add(base)
            _save_state(done)
            print(f"[ok] {base} 通知", flush=True)
        except Exception as e:
            print(f"[warn] {base} 通知失敗: {e}", flush=True)


if __name__ == "__main__":
    main()
