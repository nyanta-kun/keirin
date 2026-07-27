#!/usr/bin/env python3
"""全7予想ランク（S1/7SS/7S/7A/9SS/9S/9A）をnetkeirin「ウマい車券」へ下書き自動入稿する。

2026-07-23に7SS/7S専用スクリプトとして新設、2026-07-28に全ランク対応へ全面再構成。
朝バッチ(daily_picks_wt.sh)・夕バッチ(evening_picks_wt.sh)それぞれの候補生成
直後に呼ばれる。ランクごとの候補ファイル（候補生成時点で既にゲート適用済み）から
未入稿のレースのみ netkeirin へ下書き保存する。同一(race_key, rank_key)への
再送信は上書きされるだけなので、朝夕で対象が重複しても無害。

各ランクのON/OFF・タイトル/コメントのテンプレートは kiseki 側の入稿設定画面
（/keirin/settings）で編集された keirin.netkeirin_settings を読む。OFFのランクは
スキップする。コメント末尾には、そのレースの出走選手ごとの1着率・3着内率テーブルを
自動付加する（数値はkiseki Web(/keirin)と同じロジット空間シフトによる正規化値）。

入稿完了後、新規に登録した件数が1件以上あれば1本のDiscordサマリーを送る。
公開は必ずユーザー本人が確認用URLから行う（本スクリプトは自動化しない）。

仕様の根拠: docs/netkeirin-input-api-spec.md

使い方:
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD morning
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD evening
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD morning --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.netkeirin_client import (
    BET_KIND_TRIFECTA_AXIS1,
    BET_KIND_TRIO_AXIS2,
    NetkeirinClient,
    RACE_AUTH_URL,
)
from src.notify.discord import send
from src.strategy_wt import s7_gate_label

SESSION_LABEL_JP = {"morning": "午前", "evening": "午後"}

_DEFAULT_TITLE_TEMPLATE = "{venue}{race_no}R 二軸探偵"
_DEFAULT_COMMENT_TEMPLATE = (
    "本日の二軸をお届けします。\n\n"
    "買い目は三連複・軸2車流し（5点均等）です。独自の検証では、この5点のうち"
    "最終オッズが低い（目安5〜10倍以下）組み合わせを購入対象から外すと、"
    "的中率は下がる一方で回収率は上昇する傾向を確認しています。"
    "二軸探偵の入稿は発走前の最終オッズを確認できないタイミングで行っているため、"
    "この絞り込みは行っておりません。\n\n"
    "レース直前の最終オッズをご自身でご確認いただき、低倍率の目を外すなど、"
    "回収率を意識したアレンジにもぜひご活用ください。"
)

# ランク定義。file_key は候補JSON（wave_picks_wt_{date}[_night]_{file_key}_candidates.json）の
# サフィックス。gate_filter は None なら候補全件対象、'SS'/'S' なら s7_gate_label() で絞り込む
# （7SS/7S・9SS/9S は同じ候補ファイルを wt_overlap_n で分岐させたもの）。
RANK_CONFIGS: dict[str, dict[str, Any]] = {
    "S1":  {"file_key": "s1",  "n_cars": 7, "bet_kind": BET_KIND_TRIFECTA_AXIS1, "stake_per_line": 5000, "gate_filter": None},
    "7SS": {"file_key": "s7",  "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 2000, "gate_filter": "SS"},
    "7S":  {"file_key": "s7",  "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 2000, "gate_filter": "S"},
    "7A":  {"file_key": "s7a", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 2000, "gate_filter": None},
    "9SS": {"file_key": "s9",  "n_cars": 9, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 1400, "gate_filter": "SS"},
    "9S":  {"file_key": "s9",  "n_cars": 9, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 1400, "gate_filter": "S"},
    "9A":  {"file_key": "s9a", "n_cars": 9, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 1400, "gate_filter": None},
}
RANK_ORDER = ["S1", "7SS", "7S", "7A", "9SS", "9S", "9A"]


# ---------------------------------------------------------------------------
# 選手成績表（1着率・3着内率）
# frontend/src/app/keirin/page.tsx の sigmoid/logit/solveLogitShift と同一ロジックの
# Python移植。pred_win_pct/pred_top3_pct（選手ごと独立モデルの生確率）はレース内合計が
# 揃わないため、ロジット空間で一律シフトして単勝=100%・複勝=min(出走数,3)*100%に補正する。
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    eps = 1e-6
    c = min(max(p, eps), 1 - eps)
    return math.log(c / (1 - c))


def _solve_logit_shift(probs: list[float], target: float) -> float:
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        total = sum(_sigmoid(_logit(p) + mid) for p in probs)
        if total < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _build_entry_table(race_key: str, marks: dict[int, str]) -> str | None:
    """出走選手ごとの1着率・3着内率（正規化値）をプレーンテキスト表として返す。
    指数未算出（pred_win_pct が全件NULL）のレースは None を返し呼び出し側で省略する。
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT frame_no, name, pred_win_pct, pred_top3_pct FROM wt_entries "
            "WHERE race_key = ? ORDER BY frame_no",
            (race_key,),
        ).fetchall()
    entries = [dict(r) for r in rows]
    if not entries or all(e["pred_win_pct"] is None for e in entries):
        return None

    win_probs = [float(e["pred_win_pct"] or 0) / 100 for e in entries]
    top3_probs = [float(e["pred_top3_pct"] or 0) / 100 for e in entries]
    win_shift = _solve_logit_shift(win_probs, 1) if any(p > 0 for p in win_probs) else None
    top3_shift = (
        _solve_logit_shift(top3_probs, min(len(entries), 3)) if any(p > 0 for p in top3_probs) else None
    )

    lines = ["【出走選手 1着率・3着内率】"]
    for e, wp, tp in zip(entries, win_probs, top3_probs):
        frame_no = int(e["frame_no"])
        mark = marks.get(frame_no, "　")
        name = e["name"] or "―"
        win_pct = (
            100 * _sigmoid(_logit(wp) + win_shift)
            if win_shift is not None and e["pred_win_pct"] is not None else None
        )
        top3_pct = (
            100 * _sigmoid(_logit(tp) + top3_shift)
            if top3_shift is not None and e["pred_top3_pct"] is not None else None
        )
        win_str = f"{win_pct:.1f}%" if win_pct is not None else "―"
        top3_str = f"{top3_pct:.1f}%" if top3_pct is not None else "―"
        lines.append(f"{frame_no}番{mark} {name}　1着率{win_str}　3着内率{top3_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# テンプレート
# ---------------------------------------------------------------------------

def _apply_template(
    template: str, *, venue_name: str, race_no: int, rank_key: str, target_date: str,
    axis1: int, axis2: int,
) -> str:
    """{venue}{race_no}{rank}{date}{axis1}{axis2} を置換する。
    str.format ではなく固定辞書の逐次 str.replace を使う（未定義の{...}をユーザーが
    書いても例外にせず素通しするため）。
    """
    repl = {
        "{venue}": venue_name,
        "{race_no}": str(race_no),
        "{rank}": rank_key,
        "{date}": target_date,
        "{axis1}": str(axis1),
        "{axis2}": str(axis2),
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


# ---------------------------------------------------------------------------
# 候補・設定・送信済み記録の読み書き
# ---------------------------------------------------------------------------

def _load_candidates(target_date: str, session: str, file_key: str) -> list[dict]:
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    suffix = f"_night_{file_key}_candidates.json" if session == "evening" else f"_{file_key}_candidates.json"
    path = picks_dir / f"wave_picks_wt_{target_date}{suffix}"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[netkeirin_submit] {path.name} 読み込み失敗: {e}", flush=True)
        return []


def _load_settings() -> dict[str, dict]:
    """keirin.netkeirin_settings を読む。テーブル未取得（migration未適用等）や
    行が無いランクは全ON・デフォルトテンプレート扱いにする（フェイルオープン）。
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT rank_key, enabled, title_template, comment_template FROM netkeirin_settings"
            ).fetchall()
        return {r["rank_key"]: dict(r) for r in rows}
    except Exception as e:
        print(f"[netkeirin_submit] netkeirin_settings読み込み失敗（全ON既定で継続）: {e}", flush=True)
        return {}


def _is_enabled(settings: dict[str, dict], rank_key: str) -> bool:
    row = settings.get(rank_key)
    return True if row is None else bool(row["enabled"])


def _already_submitted(race_keys: list[str]) -> set[tuple[str, str]]:
    if not race_keys:
        return set()
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, rank_key FROM netkeirin_submissions WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    return {(r["race_key"], r["rank_key"]) for r in rows}


def _record_submission(
    race_key: str, rank_key: str, session: str, venue_name: str, race_no: int,
    gate_label: str | None, axis1: int, axis2: int, netkeirin_race_id: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO netkeirin_submissions "
            "(race_key,rank_key,session,venue_name,race_no,gate_label,axis1,axis2,netkeirin_race_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (race_key, rank_key, session, venue_name, race_no, gate_label, axis1, axis2, netkeirin_race_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# ランクごとの候補正規化（S1のキー構造は他ランクと異なるため吸収する）
# ---------------------------------------------------------------------------

def _normalize_candidate(cand: dict, cfg: dict) -> tuple[int, int, list[int], dict[int, str]]:
    """候補dictから (axis1, axis2, partners, marks) を返す。
    axis2 は trifecta_axis1 では submit_pick に渡さないが、テンプレート変数用に
    p1（相手1）を充てて返す。
    """
    if cfg["bet_kind"] == BET_KIND_TRIFECTA_AXIS1:
        axis1, p1, p2 = int(cand["axis"]), int(cand["p1"]), int(cand["p2"])
        return axis1, p1, [p1, p2], {axis1: "◎", p1: "○", p2: "▲"}
    axis1, axis2 = int(cand["axis1"]), int(cand["axis2"])
    partners = [c for c in range(1, cfg["n_cars"] + 1) if c not in (axis1, axis2)]
    return axis1, axis2, partners, {axis1: "◎", axis2: "○"}


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def _process_rank(
    rank_key: str, target_date: str, session: str, race_date, settings: dict[str, dict],
    already: set[tuple[str, str]], dry_run: bool,
) -> tuple[int, list[str]]:
    cfg = RANK_CONFIGS[rank_key]
    if not _is_enabled(settings, rank_key):
        return 0, []

    raw = _load_candidates(target_date, session, cfg["file_key"])
    if not raw:
        return 0, []

    targets: list[tuple[dict, str | None]] = []
    for cand in raw:
        gate_label = None
        if cfg["gate_filter"] is not None:
            gate_label = s7_gate_label(cand.get("wt_overlap_n"))
            if gate_label != cfg["gate_filter"]:
                continue
        targets.append((cand, gate_label))
    if not targets:
        return 0, []

    pending = [(c, g) for c, g in targets if (c["race_key"], rank_key) not in already]
    if not pending:
        return 0, []

    setting = settings.get(rank_key)
    title_template = (setting or {}).get("title_template") or _DEFAULT_TITLE_TEMPLATE
    comment_template = (setting or {}).get("comment_template") or _DEFAULT_COMMENT_TEMPLATE

    client = NetkeirinClient() if not dry_run else None
    n_submitted = 0
    failures: list[str] = []

    for cand, gate_label in pending:
        race_key = cand["race_key"]
        venue_name = cand.get("venue_name", "?")
        race_no = int(cand["race_no"])
        axis1, axis2_or_p1, partners, marks = _normalize_candidate(cand, cfg)

        title = _apply_template(
            title_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
            target_date=target_date, axis1=axis1, axis2=axis2_or_p1,
        )
        comment = _apply_template(
            comment_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
            target_date=target_date, axis1=axis1, axis2=axis2_or_p1,
        )
        entry_table = _build_entry_table(race_key, marks)
        if entry_table:
            comment = f"{comment}\n\n{entry_table}"

        if dry_run:
            print(
                f"[dry-run] {venue_name}{race_no}R ({rank_key}) "
                f"軸={axis1} 相手={partners} 賭け金={cfg['stake_per_line']}円/点\n"
                f"  タイトル: {title}\n"
                f"  コメント:\n{comment}\n",
                flush=True,
            )
            n_submitted += 1
            continue

        try:
            assert client is not None
            ok, msg = client.submit_pick(
                race_date=race_date, venue_name=venue_name, race_no=race_no,
                n_cars=cfg["n_cars"], bet_kind=cfg["bet_kind"],
                axis1=axis1,
                axis2=(axis2_or_p1 if cfg["bet_kind"] == BET_KIND_TRIO_AXIS2 else None),
                partners=partners, stake_per_line=cfg["stake_per_line"],
                title=title, comment=comment,
            )
        except Exception as e:
            ok, msg = False, f"例外: {e}"

        if ok:
            _record_submission(
                race_key, rank_key, session, venue_name, race_no, gate_label, axis1, axis2_or_p1, msg,
            )
            n_submitted += 1
            print(f"[netkeirin_submit] 入稿成功 {venue_name}{race_no}R ({rank_key}) → {msg}", flush=True)
        else:
            failures.append(f"{venue_name}{race_no}R({rank_key}): {msg}")
            print(f"[netkeirin_submit] 入稿失敗 {venue_name}{race_no}R ({rank_key}): {msg}", flush=True)

    return n_submitted, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_date")
    parser.add_argument("session", choices=("morning", "evening"))
    parser.add_argument("--dry-run", action="store_true", help="送信せず生成内容を標準出力に出す")
    args = parser.parse_args()

    target_date, session = args.target_date, args.session
    race_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    settings = _load_settings()
    if not _is_enabled(settings, "_global"):
        print(f"[netkeirin_submit] {target_date} {session}: 全体OFF（スキップ）", flush=True)
        return

    all_race_keys: set[str] = set()
    per_rank_raw: dict[str, list[dict]] = {}
    for rank_key in RANK_ORDER:
        if not _is_enabled(settings, rank_key):
            continue
        cfg = RANK_CONFIGS[rank_key]
        raw = _load_candidates(target_date, session, cfg["file_key"])
        per_rank_raw[rank_key] = raw
        all_race_keys.update(c["race_key"] for c in raw)

    already = _already_submitted(sorted(all_race_keys))

    submitted_counts: dict[str, int] = {r: 0 for r in RANK_ORDER}
    all_failures: list[str] = []
    for rank_key in RANK_ORDER:
        if rank_key not in per_rank_raw:
            continue
        n, failures = _process_rank(rank_key, target_date, session, race_date, settings, already, args.dry_run)
        submitted_counts[rank_key] = n
        all_failures.extend(failures)

    total = sum(submitted_counts.values())
    if args.dry_run:
        print(f"[dry-run] {target_date} {session}: 完了（生成{total}件）", flush=True)
        return

    session_jp = SESSION_LABEL_JP[session]
    if total > 0:
        breakdown = "・".join(f"{k}{v}件" for k, v in submitted_counts.items() if v > 0)
        msg = (
            f"📮 **[netkeirin入稿完了] {target_date}（{session_jp}）: "
            f"{breakdown}（計{total}件）**\n"
            f"確認: {RACE_AUTH_URL}\n"
            f"内容を確認の上、公開してください。"
        )
        if all_failures:
            msg += f"\n⚠️ 入稿失敗 {len(all_failures)}件: " + " / ".join(all_failures)
        try:
            send(msg, channel="netkeirin")
        except Exception as e:
            print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)
    elif all_failures:
        try:
            send(
                f"⚠️ **[netkeirin入稿] {target_date}（{session_jp}）: 全{len(all_failures)}件が入稿失敗**\n"
                + " / ".join(all_failures),
                channel="netkeirin",
            )
        except Exception as e:
            print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)
    else:
        print(f"[netkeirin_submit] {target_date} {session}: 対象なし（スキップ）", flush=True)

    print(
        f"[netkeirin_submit] {target_date} {session}: 完了（成功{total}件・失敗{len(all_failures)}件）",
        flush=True,
    )


if __name__ == "__main__":
    main()
