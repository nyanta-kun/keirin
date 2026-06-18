#!/usr/bin/env python3
"""
pre-race オッズ確認・Discord 通知（毎分 cron で実行）

発走 15 分前（購入締切 10 分前）に当日ピック済みレースの
現在オッズを winticket からリアルタイム取得し、Discord へ通知する。

cron 設定（crontab -e に追加）:
  * * * * * cd /Users/ysuzuki/GitHub/keirin && .venv/bin/python3 scripts/notify_prerace_wt.py >> /tmp/prerace.log 2>&1

通知ウィンドウ: start_at - 900秒 ≤ now < start_at - 840秒（1分間）
  競輪の購入締切は発走 5 分前 → 締切 10 分前 = 発走 15 分前（900秒前）に通知

通知内容:
  - 発走時刻・会場・レース番号・車数
  - 推奨ランク（SS/S/A/7+）と買い目
  - 現在の各組み合わせオッズ（三連複 or 三連単）
  - 最安目オッズ → ガミ(≥5倍)の充足確認
  - gap12 / ratio（朝時点の予想強度の参考値）
"""

from __future__ import annotations

import json
import re
import sys
import time
import logging
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.scraper.winticket import WinticketScraper
from src.notify.discord import send

logger = logging.getLogger(__name__)

# 日本標準時 (UTC+9)
_JST = timezone(timedelta(hours=9))

# 通知タイミング: 発走 N 秒前
NOTIFY_BEFORE_START_SEC = 15 * 60   # 15分前 = 締切（5分前）の10分前
NOTIFY_WINDOW_SEC       = 70        # 通知ウィンドウ幅（cron の遅延を吸収）

# ガミ足切り閾値
GAMI_THRESHOLD = 5.0

# 三連単を通知に含めるランク
TRIFECTA_RANKS = {"SS"}

# 7+車 gap12閾値（Sランク境界）
SEVEN_PLUS_S_GAP12 = 0.10


def _jst_now() -> datetime:
    return datetime.now(_JST)


def _now_unix() -> int:
    return int(time.time())


# ── 状態ファイル（通知済みレースを記録） ───────────────────────────────────────

def _state_path(today: str) -> Path:
    return Path(__file__).parent.parent / "data" / f"prerace_notified_{today}.json"


def _load_notified(today: str) -> set[str]:
    p = _state_path(today)
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def _save_notified(today: str, notified: set[str]) -> None:
    p = _state_path(today)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(notified), ensure_ascii=False, indent=2), encoding="utf-8")


# ── picks の読み込み ─────────────────────────────────────────────────────────

def _load_picks(today: str) -> list[dict]:
    """当日の candidates JSON (発走前再検証用・gamiフィルタなし) を優先して返す。
    candidates がなければ detail JSON（フィルタ済み）にフォールバック。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    # 候補ファイル優先（gap12のみ・gamiフィルタなし）
    for fname in (f"wave_picks_wt_{today}_candidates.json",
                  f"wave_picks_wt_{today}_detail.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                entries = json.loads(p.read_text(encoding="utf-8"))
                # night 候補は別ファイル (candidates は日中のみ)
                night_cands = picks_dir / f"wave_picks_wt_{today}_night_candidates.json"
                if night_cands.exists():
                    try:
                        entries += json.loads(night_cands.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                return entries
            except Exception as e:
                logger.warning("%s 読み込み失敗: %s", fname, e)
    return []


# ── wt_races から race 情報取得 ─────────────────────────────────────────────

def _load_race_info(race_keys: list[str]) -> dict[str, dict]:
    """wt_races から {race_key: {start_at, venue_id, cup_id, day_index, n_entries}} を返す。"""
    if not race_keys:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, start_at, venue_id, cup_id, day_index, n_entries, race_date "
            f"FROM wt_races WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    return {r["race_key"]: dict(r) for r in rows}


# ── 現在オッズ取得 ──────────────────────────────────────────────────────────

def _fetch_current_odds(race_info: dict) -> dict[str, list[dict]] | None:
    """WinticketScraper でリアルタイムオッズを取得する。"""
    try:
        scraper = WinticketScraper(request_interval=0.5)
        return scraper.fetch_odds(
            venue_id   = race_info["venue_id"],
            race_date  = race_info["race_date"],
            race_no    = race_info["race_no"],
            cup_id     = race_info["cup_id"],
            day_index  = race_info["day_index"],
        )
    except Exception as e:
        logger.warning("fetch_odds 失敗 %s: %s", race_info.get("race_key"), e)
        return None


def _parse_combo_key(combo_str: str, ordered: bool) -> tuple | frozenset | None:
    """combination 文字列 (例 '1-2-3' or '1=2=3') をキーに変換する。"""
    parts = re.split(r"[-=]", str(combo_str))
    try:
        nums = [int(p) for p in parts]
        return tuple(nums) if ordered else frozenset(nums)
    except Exception:
        return None


def _build_odds_lookup(odds_data: dict, bet_type: str) -> dict:
    """odds_data[bet_type] から {key: odds_value} 辞書を返す。"""
    ordered = (bet_type == "trifecta")
    lookup = {}
    for item in odds_data.get(bet_type, []):
        k = _parse_combo_key(str(item["combination"]), ordered)
        if k:
            lookup[k] = item["odds_value"]
    return lookup


# ── 候補レースのリアルタイムランク判定 ─────────────────────────────────────────

def _determine_live_rank(pick: dict, odds_data: dict | None) -> tuple[str, list]:
    """7PLUS_CANDレースの現在オッズでSS/S/A を判定する。
    returns (live_rank, valid_thirds)
      - "7PLUS_SS" / "7PLUS_S" / "7PLUS_A": 条件成立
      - "なし": 購入条件不成立（全目ガミ or S/A gami全不通過 and SS残り0 or >3）
    """
    p1 = pick.get("pivot1")
    p2 = pick.get("pivot2")
    thirds = pick.get("thirds", [])
    gap12 = pick.get("gap12", 0.0)

    if odds_data is None:
        return "不明", thirds

    lookup = _build_odds_lookup(odds_data, "trio")

    # 各目の現在オッズ
    combo_odds: dict[int, float] = {}
    for t in thirds:
        key = frozenset({int(p1), int(p2), int(t)})
        ov = lookup.get(key)
        if ov and float(ov) > 0:
            combo_odds[t] = float(ov)

    valid_ge5 = [t for t in thirds if combo_odds.get(t, 0.0) >= GAMI_THRESHOLD]

    # SSランク: gami≥5倍の目が1〜3残る
    if 1 <= len(valid_ge5) <= 3:
        return "7PLUS_SS", valid_ge5

    # S/Aランク: 全目がgami≥5倍
    if len(valid_ge5) == len(thirds) and valid_ge5:
        rank = "7PLUS_S" if gap12 >= SEVEN_PLUS_S_GAP12 else "7PLUS_A"
        return rank, thirds

    return "なし", []


# ── 通知メッセージ生成 ────────────────────────────────────────────────────────

def _get_min_trio_odds(pick: dict, odds_data: dict | None) -> float | None:
    """ピックの三連複全目の最安オッズを返す。オッズ取得失敗時は None。"""
    if odds_data is None:
        return None
    p1 = pick.get("pivot1") or pick.get("pred1")
    p2 = pick.get("pivot2") or pick.get("pred2")
    thirds = pick.get("thirds", [])
    if not thirds or p1 is None or p2 is None:
        return None
    lookup = _build_odds_lookup(odds_data, "trio")
    valid_odds = []
    for t in thirds:
        key = frozenset({int(p1), int(p2), int(t)})
        ov = lookup.get(key)
        if ov and float(ov) > 0:
            valid_odds.append(float(ov))
    return min(valid_odds) if valid_odds else None


def _save_prerace_gami(race_key: str, min_odds: float) -> None:
    """picks_history.prerace_gami を発走前実測値で更新する。"""
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE picks_history SET prerace_gami = ? WHERE race_key = ?",
                (round(min_odds, 2), race_key),
            )
            conn.commit()
    except Exception as e:
        logger.warning("prerace_gami 書き込み失敗 %s: %s", race_key, e)


def _build_message(pick: dict, race_info: dict, odds_data: dict | None) -> str:
    rank     = pick["rank"]
    venue    = pick["venue_name"]
    race_no  = pick["race_no"]
    start    = pick["start_time"]
    n        = race_info.get("n_entries", pick.get("n_riders", "?"))
    gap12    = pick.get("gap12", 0)
    ratio    = pick.get("ratio", 0)
    p1       = pick.get("pivot1", pick.get("pred1"))
    p2       = pick.get("pivot2", pick.get("pred2"))
    thirds   = pick.get("thirds", [])

    # ランク表示
    rank_icon = {
        "7PLUS_SS": "🚲⭐", "7PLUS_S": "🚲🔵", "7PLUS_A": "🚲🟢",
        "7PLUS": "🚲", "SS": "⭐", "S": "🔵", "A": "🟢",
    }.get(rank, "▪️")
    is_trifecta = rank in TRIFECTA_RANKS

    # 買い目文字列（全目）
    is_7plus = rank.startswith("7PLUS")
    if is_trifecta:
        thirds_str = ",".join(str(t) for t in thirds)
        combo_str  = f"{p1}→{p2}→{thirds_str}"
        bet_label  = "3連単"
        market     = "trifecta"
    else:
        thirds_str = ",".join(str(t) for t in thirds)
        combo_str  = f"{p1}-{p2}-{thirds_str}"
        bet_label  = "3連複"
        market     = "trio"

    n_pts = len(thirds)

    # ── 現在オッズ（全目チェック・gamiは全目の最安値） ──
    lines = []
    if odds_data:
        lookup = _build_odds_lookup(odds_data, market)
        odds_per_bet = []

        for t in thirds:
            if is_trifecta:
                key = (int(p1), int(p2), int(t))
            else:
                key = frozenset({int(p1), int(p2), int(t)})
            odds_val = lookup.get(key)
            odds_per_bet.append((t, odds_val))

        sep = "→" if is_trifecta else "-"
        for t, ov in odds_per_bet:
            if ov is None:
                lines.append(f"    {p1}{sep}{p2}{sep}{t}:  取得不可")
            else:
                gami_ng = " ⚠️" if ov < GAMI_THRESHOLD else ""
                lines.append(f"    {p1}{sep}{p2}{sep}{t}:  {ov:.1f}倍{gami_ng}")

        valid_odds = [ov for _, ov in odds_per_bet if ov is not None]
        if valid_odds:
            min_odds = min(valid_odds)
            # 合成オッズ = 1 / Σ(1/odds_i)  ← 全有効目の逆数和の逆数
            synth_odds = 1.0 / sum(1.0 / ov for ov in valid_odds)
            investment = n_pts * 100
            if min_odds >= GAMI_THRESHOLD:
                gami_mark = f"✅ ガミOK（全{n_pts}目 最安 {min_odds:.1f}倍）"
            else:
                if is_7plus:
                    surviving = [t for t, ov in odds_per_bet if ov is not None and ov >= GAMI_THRESHOLD]
                    if surviving:
                        gami_mark = (f"⚠️ ガミ注意（最安 {min_odds:.1f}倍）"
                                     f" → 5倍以上は{len(surviving)}目: {','.join(str(t) for t in surviving)}")
                    else:
                        gami_mark = f"❌ 全目ガミ（最安 {min_odds:.1f}倍）— 購入非推奨"
                        synth_odds = 0.0
                else:
                    gami_mark = f"⚠️ ガミ注意（最安 {min_odds:.1f}倍 < {GAMI_THRESHOLD}倍）"
        else:
            synth_odds = 0.0
            investment = n_pts * 100
            gami_mark = "⚠️ オッズ全取得不可（締切済みの可能性）"
    else:
        lines = ["    ⚠️ リアルタイムオッズ取得失敗（手動で確認してください）"]
        gami_mark = ""
        synth_odds = 0.0
        investment = n_pts * 100

    # 目数が多い場合は折り畳み表示（SSランクは少ないのでそのまま）
    MAX_DISPLAY = 5
    if len(lines) > MAX_DISPLAY:
        odds_block = "\n".join(lines[:MAX_DISPLAY]) + f"\n    … (全{n_pts}目)"
    else:
        odds_block = "\n".join(lines)

    synth_str = f"{synth_odds:.2f}倍" if synth_odds > 0 else "—"

    msg = (
        f"{rank_icon} **[{rank}]  {venue} {race_no}R  [{n}車]  発走 {start}**\n"
        f"  {bet_label}({n_pts}点 / {investment}円): `{combo_str}`\n"
        f"  **合成オッズ: {synth_str}**  gap12={gap12:.3f}\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        f"{odds_block}\n"
        f"  {gami_mark}"
    )
    return msg


# ── メイン ──────────────────────────────────────────────────────────────────

def main():
    today     = date.today().strftime("%Y-%m-%d")
    now_unix  = _now_unix()

    picks = _load_picks(today)
    if not picks:
        return   # 当日のピックなし → 何もしない

    race_keys = [p["race_key"] for p in picks if "race_key" in p]
    race_info_map = _load_race_info(race_keys)

    notified = _load_notified(today)
    to_notify = []

    for pick in picks:
        rk = pick.get("race_key")
        if rk is None or rk in notified:
            continue

        ri = race_info_map.get(rk)
        if ri is None:
            continue

        start_at_unix = int(ri["start_at"])
        notify_at     = start_at_unix - NOTIFY_BEFORE_START_SEC

        # 通知ウィンドウ内かチェック
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            to_notify.append((pick, ri))

    if not to_notify:
        return   # 今分は通知すべきレースなし

    # ── 推奨メッセージを収集してからまとめて送信 ──
    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []   # (race_key, message)
    newly_done: set[str] = set()           # 今回処理完了（条件不成立も含む）

    for pick, ri in to_notify:
        rk = pick["race_key"]
        rank = pick.get("rank", "?")

        # ライブオッズ取得
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗 %s: %s", rk, e)
            odds_data = None

        # 発走前の三連複最安オッズを picks_history に記録
        min_odds = _get_min_trio_odds(pick, odds_data)
        if min_odds is not None:
            _save_prerace_gami(rk, min_odds)

        # race_no をピックに付与
        pick_with_raceno = dict(pick)
        pick_with_raceno["race_no"] = ri["race_no"]
        pick_with_raceno["n_entries"] = ri["n_entries"]

        # 候補レース（7PLUS_CAND）は現在オッズで SS/S/A を再判定
        if rank == "7PLUS_CAND":
            live_rank, live_thirds = _determine_live_rank(pick, odds_data)
            if live_rank == "なし":
                print(f"[prerace] {rk} 候補 → live判定: 条件不成立（通知なし）", flush=True)
                newly_done.add(rk)
                time.sleep(0.3)
                continue
            if live_rank == "不明":
                # オッズ取得失敗 → 再試行の余地を残すため newly_done に追加しない
                print(f"[prerace] {rk} 候補 → オッズ取得不可（次回再試行）", flush=True)
                time.sleep(0.3)
                continue
            # 判定成立: ランクと買い目を上書き
            pick_with_raceno["rank"] = live_rank
            pick_with_raceno["thirds"] = live_thirds
            n_pts = len(live_thirds)
            pivot1 = pick.get("pivot1")
            pivot2 = pick.get("pivot2")
            pick_with_raceno["combo_str"] = f"{pivot1}-{pivot2}-{','.join(str(t) for t in live_thirds)}"
            pick_with_raceno["n_points"] = n_pts
            pick_with_raceno["stake"] = n_pts * 100
            print(f"[prerace] {rk} 候補 → live判定: {live_rank} ({n_pts}点)", flush=True)

        msg = _build_message(pick_with_raceno, ri, odds_data)
        messages.append((rk, msg))
        newly_done.add(rk)
        time.sleep(0.5)   # Discord レート制限対策

    # 推奨がある場合のみ Discord 送信
    if messages:
        jst_now_str = _jst_now().strftime("%H:%M")
        send(f"🚲 **レース直前推奨** — {today}  {jst_now_str} 時点  ({len(messages)}件)")
        for rk, msg in messages:
            send(msg)
            print(f"[prerace] {rk} → 通知送信完了", flush=True)
            time.sleep(0.5)
    else:
        print(f"[prerace] {today} 推奨なし（オッズ確認のみ・通知スキップ）", flush=True)

    notified |= newly_done
    _save_notified(today, notified)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
