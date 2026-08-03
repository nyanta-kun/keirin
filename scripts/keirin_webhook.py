#!/usr/bin/env python3
"""keirin webhook trigger server（VPS 常駐・systemd: keirin-webhook.service）

kiseki バックエンド（galloplab-backend-1）からの POST を受けて
keirin ホスト側スクリプトをバックグラウンド起動する。

エンドポイント:
  POST /fetch-results : 当日結果の即時取得+採点 → scripts/intraday_results_wt.sh
  POST /fetch-odds    : 発走前ガミ判定の即時実行 → scripts/notify_prerace_wt.py
  POST /submit-race   : 指定レース1件のみをピンポイントでnetkeirinへ入稿
                        → scripts/netkeirin_submit_wt.py --race-key（通常入稿と同一ルール）
                        rank_key/axis1/axis2 も含む場合（推奨外レースの手動入稿・
                        2026-07-31新設）は --manual-rank-key/--axis1/--axis2 を付与し
                        候補JSON検索を経由しない手動入稿パスを起動する

kiseki 側の呼び出し元:
  backend/src/api/keirin_router.py の /api/keirin/fetch-results, /fetch-odds, /submit-race
  （_WEBHOOK_BASE = http://172.18.0.1:8010 → Docker bridge 経由でホストに到達）

レスポンスは {"ok": bool, "message": str} 固定（frontend api.ts が参照）。

systemd unit（/etc/systemd/system/keirin-webhook.service）:
  ExecStart=/home/ysuzuki/keirin/.venv/bin/python3 scripts/keirin_webhook.py
  EnvironmentFile=/home/ysuzuki/keirin/.env.webhook  (KEIRIN_DB_URL を供給)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("keirin_webhook")

KEIRIN_HOME = Path(os.environ.get("KEIRIN_HOME", str(Path(__file__).resolve().parent.parent)))
HOST = os.environ.get("WEBHOOK_HOST", "0.0.0.0")
PORT = int(os.environ.get("WEBHOOK_PORT", "8010"))
LOG_DIR = KEIRIN_HOME / "data" / "logs"

# エンドポイントごとに直近の子プロセスを保持し、多重起動を防ぐ
_running: dict[str, subprocess.Popen] = {}

_RACE_KEY_RE = re.compile(r"^\d{8}_\d{2}_\d{2}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# scripts/netkeirin_submit_wt.py の MANUAL_ALLOWED_RANKS と同一
# （S1・旧7SS/旧9SS は全廃済み。波乱軸選出の RANK_7SS も 2026-08-02 に全廃した
#   ため対象外。2026-08-02 時点で submit 側は既に ("7S","7A","9S","9A") だったが
#   こちらだけ旧値が残っており、_is_enabled() が fail-open のため webhook 経由の
#   手動入稿では廃止済みランクが通る状態になっていた＝同時に是正）
_MANUAL_ALLOWED_RANKS = ("7S", "7A", "7B", "9S", "9A")


def _spawn(name: str, cmd: list[str], log_file: Path, extra_env: dict[str, str] | None = None) -> tuple[bool, str]:
    prev = _running.get(name)
    if prev is not None and prev.poll() is None:
        return False, "前回の処理がまだ実行中です"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    with open(log_file, "ab") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(KEIRIN_HOME),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _running[name] = proc
    return True, f"{name} をバックグラウンド起動しました (pid={proc.pid})"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/fetch-results":
            log.info("triggered /fetch-results")
            ok, message = _spawn(
                "fetch-results",
                ["bash", "scripts/intraday_results_wt.sh"],
                LOG_DIR / "cron.log",
            )
        elif self.path == "/fetch-odds":
            log.info("triggered /fetch-odds")
            ok, message = _spawn(
                "fetch-odds",
                [str(KEIRIN_HOME / ".venv" / "bin" / "python3"), "scripts/notify_prerace_wt.py"],
                LOG_DIR / "prerace.log",
                extra_env={"PYTHONPATH": "."},
            )
        elif self.path == "/submit-race":
            ok, message, status = self._handle_submit_race()
            self._respond(status, {"ok": ok, "message": message})
            return
        else:
            self._respond(404, {"ok": False, "message": f"unknown path: {self.path}"})
            return
        self._respond(200, {"ok": ok, "message": message})

    def _handle_submit_race(self) -> tuple[bool, str, int]:
        """race_key/date/sessionを検証し、単一レースのみを対象にnetkeirin_submit_wt.pyを起動する。

        rank_key/axis1/axis2（任意項目）が揃っている場合は、推奨外レースの手動入稿
        （kiseki Webのランク選択ダイアログ由来・2026-07-31新設）として
        --manual-rank-key/--axis1/--axis2 を付与する。
        """
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return False, "invalid JSON body", 400

        race_key = str(body.get("race_key", ""))
        date = str(body.get("date", ""))
        session = str(body.get("session", ""))
        if not _RACE_KEY_RE.match(race_key):
            return False, f"invalid race_key: {race_key}", 400
        if not _DATE_RE.match(date):
            return False, f"invalid date: {date}", 400
        if session not in ("morning", "evening"):
            return False, f"invalid session: {session}", 400

        cmd = [
            str(KEIRIN_HOME / ".venv" / "bin" / "python3"), "scripts/netkeirin_submit_wt.py",
            date, session, "--race-key", race_key,
        ]

        rank_key = body.get("rank_key")
        axis1 = body.get("axis1")
        axis2 = body.get("axis2")
        if rank_key is not None or axis1 is not None or axis2 is not None:
            rank_key = str(rank_key)
            if rank_key not in _MANUAL_ALLOWED_RANKS:
                return False, f"invalid rank_key: {rank_key}", 400
            try:
                axis1_i, axis2_i = int(axis1), int(axis2)
            except (TypeError, ValueError):
                return False, f"invalid axis1/axis2: {axis1}/{axis2}", 400
            cmd += ["--manual-rank-key", rank_key, "--axis1", str(axis1_i), "--axis2", str(axis2_i)]

        log.info("triggered /submit-race race_key=%s date=%s session=%s rank_key=%s",
                  race_key, date, session, rank_key)
        ok, message = _spawn(
            f"submit-race-{race_key}",
            cmd,
            LOG_DIR / "netkeirin_submit.log",
            extra_env={"PYTHONPATH": "."},
        )
        return ok, message, 200

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._respond(200, {"ok": True, "message": "alive"})
        else:
            self._respond(404, {"ok": False, "message": "POST /fetch-results | /fetch-odds | /submit-race"})

    def _respond(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        log.info("%s %s", self.address_string(), format % args)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("keirin webhook listening on %s:%d (KEIRIN_HOME=%s)", HOST, PORT, KEIRIN_HOME)
    server.serve_forever()


if __name__ == "__main__":
    main()
