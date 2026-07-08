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

import fcntl
import json
import os
import re
import shutil
import sys
import time
import logging
from contextlib import contextmanager
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
GAMI_THRESHOLD = 7.0

# 三連単を通知に含めるランク
TRIFECTA_RANKS = {"SS"}

# 7+車 gap12閾値（Sランク境界）
SEVEN_PLUS_S_GAP12 = 0.10

# 合成オッズ下限（SO < この値は通知しない）
SYNTH_ODDS_MIN = 8.0

# gap23 下限・%ポイント（2位-3位予測確率差 < この値は通知しない）
GAP23_MIN = 1.0


def _jst_now() -> datetime:
    return datetime.now(_JST)


def _now_unix() -> int:
    return int(time.time())


# ── 状態ファイル共通ヘルパー ──────────────────────────────────────────────────
# 毎分cronの並行実行で read-modify-write が交錯すると当日の全記録が消える
# （2026-07-08 に prerace_decisions/notified が同時消失し、採点フォールバックが
# 「幻の購入」を復活させる事故が発生）。flock 排他 + tmp→os.replace の
# アトミック書き込み + .bak フォールバックで構造的に防ぐ。

@contextmanager
def _file_lock(p: Path):
    lock_path = p.with_name(p.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_json_state(p: Path, default):
    """本体 → .bak の順に読む。両方読めない場合のみ default を返す。"""
    for cand in (p, p.with_name(p.name + ".bak")):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error("状態ファイル読み込み失敗 %s: %s", cand, e)
    return default


def _write_json_atomic(p: Path, obj) -> None:
    """tmp に書いて os.replace。現行本体が正常JSONなら .bak に退避してから置換する。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if p.exists():
        try:
            json.loads(p.read_text(encoding="utf-8"))
            shutil.copy2(p, p.with_name(p.name + ".bak"))
        except Exception:
            # 破損した本体は .bak を汚さず forensic 用に退避
            shutil.copy2(p, p.with_name(p.name + ".corrupt"))
    os.replace(tmp, p)


# ── 状態ファイル（通知済みレースを記録） ───────────────────────────────────────

def _state_path(today: str) -> Path:
    return Path(__file__).parent.parent / "data" / f"prerace_notified_{today}.json"


def _load_notified(today: str) -> set[str]:
    return set(_read_json_state(_state_path(today), []))


def _save_notified(today: str, notified: set[str]) -> None:
    p = _state_path(today)
    with _file_lock(p):
        # 並行実行の追記を失わないよう保存時に現ファイルとマージする（追記専用集合）
        merged = set(_read_json_state(p, [])) | set(notified)
        _write_json_atomic(p, sorted(merged))


# ── 発走前判定の永続化 ─────────────────────────────────────────────────────────
# 発走15分前の判定（推奨/見送り・ランク・購入買い目・レグ別オッズ）を確定記録する。
# notify_results_wt.py はこの記録を最優先で採点する（15分前判定を事後変更しない）。

def _decisions_path(today: str) -> Path:
    return Path(__file__).parent.parent / "data" / f"prerace_decisions_{today}.json"


def _load_decisions(today: str) -> dict:
    return _read_json_state(_decisions_path(today), {})


def _score_stats(pick: dict) -> dict:
    """競走得点の構造統計（軸信頼度の live 評価用・判定には未使用）。

    12ヶ月検証 (2026-07-07): 得点SD・上位2と残りの格差が大きいレースほど
    2軸(pivot)が堅く ROI が高い（sd>=Q1 で残すと 2.87→3.0-3.6 / 除外帯 1.5-1.7）。
    live 蓄積後に除外条件へ昇格するか判断する。
    """
    out: dict = {}
    scores = [r.get("racing_score") for r in pick.get("riders", [])
              if r.get("racing_score") is not None]
    if len(scores) >= 5:
        vs = sorted(scores, reverse=True)
        n = len(vs)
        mean = sum(vs) / n
        sd = (sum((x - mean) ** 2 for x in vs) / n) ** 0.5
        rest_mean = sum(vs[2:]) / (n - 2)
        out.update({
            "score_mean": round(mean, 2),
            "score_sd": round(sd, 3),
            "score_gap2r": round((vs[0] + vs[1]) / 2 - rest_mean, 3),
        })
    # 指数(モデル予測確率)の分散: 配当予測比較(2026-07-08)で最強の低配当予測子
    # (低配当<1000円 AUC 0.637 > 得点統計の最良 0.582)
    preds = [r.get("pred_prob_pct") for r in pick.get("riders", [])
             if r.get("pred_prob_pct") is not None]
    if len(preds) >= 5:
        pv = sorted((p / 100.0 for p in preds), reverse=True)
        pm = sum(pv) / len(pv)
        out["pred_sd"] = round((sum((x - pm) ** 2 for x in pv) / len(pv)) ** 0.5, 4)
        out["pred_top2sum"] = round(pv[0] + pv[1], 4)
    return out


def _save_decision(today: str, race_key: str, record: dict) -> None:
    p = _decisions_path(today)
    with _file_lock(p):
        decisions = _read_json_state(p, {})
        record["decided_at"] = _jst_now().strftime("%H:%M:%S")
        decisions[race_key] = record
        _write_json_atomic(p, decisions)


# ── picks の読み込み ─────────────────────────────────────────────────────────

def _load_picks(today: str) -> list[dict]:
    """当日の candidates JSON (発走前再検証用・gamiフィルタなし) を優先して返す。
    candidates がなければ detail JSON（フィルタ済み）にフォールバック。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    cands_path = picks_dir / f"wave_picks_wt_{today}_candidates.json"
    detail_path = picks_dir / f"wave_picks_wt_{today}_detail.json"
    night_path  = picks_dir / f"wave_picks_wt_{today}_night_candidates.json"

    if cands_path.exists():
        try:
            entries = json.loads(cands_path.read_text(encoding="utf-8"))
            # candidates は日中（〜19時）のみ → 夜候補を追記
            if night_path.exists():
                try:
                    entries += json.loads(night_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return entries
        except Exception as e:
            logger.warning("candidates JSON 読み込み失敗: %s", e)

    if detail_path.exists():
        try:
            # detail.json は日中・夜 両方含む確定ピック → night_candidates 追記不要
            # (追記すると同一 race_key が CAND と確定ランクの2重エントリになる)
            return json.loads(detail_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("detail JSON 読み込み失敗: %s", e)

    return []


# ── wt_races から race 情報取得 ─────────────────────────────────────────────

def _load_race_info(race_keys: list[str]) -> dict[str, dict]:
    """wt_races から {race_key: {start_at, venue_id, cup_id, day_index, n_entries}} を返す。"""
    if not race_keys:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, start_at, venue_id, cup_id, day_index, n_entries, race_date, race_no "
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

def _determine_live_rank(pick: dict, odds_data: dict | None) -> tuple[str, list, dict]:
    """7PLUS_CANDレースの現在オッズでSS/S/A を判定する。
    returns (live_rank, valid_thirds, combo_odds)
      - "7PLUS_SS" / "7PLUS_S": 条件成立
      - "なし": 購入条件不成立（全目ガミ or S/A gami全不通過 and SS残り0 or >3）
      combo_odds は {third: 現在オッズ}（判定に使った値・記録用）
    """
    p1 = pick.get("pivot1")
    p2 = pick.get("pivot2")
    thirds = pick.get("thirds", [])
    gap12 = pick.get("gap12", 0.0)

    if odds_data is None:
        return "不明", thirds, {}

    lookup = _build_odds_lookup(odds_data, "trio")

    # 各目の現在オッズ
    combo_odds: dict[int, float] = {}
    for t in thirds:
        key = frozenset({int(p1), int(p2), int(t)})
        ov = lookup.get(key)
        if ov and float(ov) > 0:
            combo_odds[t] = float(ov)

    valid_ge5 = [t for t in thirds if combo_odds.get(t, 0.0) >= GAMI_THRESHOLD]

    if not valid_ge5:
        return "なし", [], combo_odds  # 全目ガミ → 購入不可

    # SOフィルタ: gami通過目の合成オッズ < SYNTH_ODDS_MIN なら不成立
    _valid_ov = [combo_odds[t] for t in valid_ge5]
    _synth = 1.0 / sum(1.0 / ov for ov in _valid_ov)
    if _synth < SYNTH_ODDS_MIN:
        return "なし", [], combo_odds

    # gap23フィルタ: モデル予測確率 2位-3位差 < GAP23_MIN なら不成立
    _gap23 = _calc_gap23(pick)
    if _gap23 is not None and _gap23 < GAP23_MIN:
        return "なし", [], combo_odds

    # SSランク: ガミカット後の有効目が1〜3目
    if len(valid_ge5) <= 3:
        return "7PLUS_SS", valid_ge5, combo_odds

    # Sランク: 有効目が4目以上（ガミ目は除外して買う）。Aランク廃止（2026-06-28）
    if gap12 < SEVEN_PLUS_S_GAP12:
        return "なし", [], combo_odds
    return "7PLUS_S", valid_ge5, combo_odds


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


def _save_picks_history_state(
    race_key: str,
    miwokuri: bool,
    new_rank: str | None = None,
    new_pred: tuple[str, int] | None = None,
) -> None:
    """picks_history の miwokuri / rank / 買い目 を即時更新する（SQLite + VPS PG）。

    ガミ落ち確定（miwokuri=True）とランク昇格（new_rank='7PLUS_SS'/'7PLUS_S'）を
    当日中にkisekiへ反映させるために呼ぶ。new_pred（ガミ目カット後の買い目）を
    渡すと pred_combo / n_combos も更新し、Webページに購入買い目が正しく出る。
    翌朝の notify_results_wt.py が prerace_decisions_*.json に基づき最終確定する。
    """
    pattern = race_key + "#%"
    cand_key = race_key + "#CAND"
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE picks_history SET miwokuri = ? WHERE race_key LIKE ? AND route = 'wt'",
                (miwokuri, pattern),
            )
            if new_rank is not None:
                conn.execute(
                    "UPDATE picks_history SET rank = ? WHERE race_key = ? AND route = 'wt'",
                    (new_rank, cand_key),
                )
            if new_pred is not None:
                conn.execute(
                    "UPDATE picks_history SET pred_combo = ?, n_combos = ? "
                    "WHERE race_key = ? AND route = 'wt'",
                    (new_pred[0], new_pred[1], cand_key),
                )
            conn.commit()
    except Exception as e:
        logger.warning("picks_history SQLite 更新失敗 %s: %s", race_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "UPDATE keirin.picks_history SET miwokuri = %s"
                        " WHERE race_key LIKE %s AND route = 'wt'",
                        (miwokuri, pattern),
                    )
                    if new_rank is not None:
                        cur.execute(
                            "UPDATE keirin.picks_history SET rank = %s"
                            " WHERE race_key = %s AND route = 'wt'",
                            (new_rank, cand_key),
                        )
                    if new_pred is not None:
                        cur.execute(
                            "UPDATE keirin.picks_history SET pred_combo = %s, n_combos = %s"
                            " WHERE race_key = %s AND route = 'wt'",
                            (new_pred[0], new_pred[1], cand_key),
                        )
        except Exception as e:
            logger.warning("picks_history VPS 更新失敗 %s: %s", race_key, e)


def _save_prerace_gami(race_key: str, min_odds: float) -> None:
    """picks_history.prerace_gami を発走前実測値で更新する（SQLite + VPS）。

    picks_history の race_key は "{base_key}#7S" / "#7A" / "#7SS" / "#CAND" 等の
    サフィックス付き形式で保存されているため、LIKE で全サフィックスを一括更新する。

    #CAND エントリが存在しない（candidates.json のガミフィルタで除外された）レースは
    UPDATE が 0 件になる。その場合 #GAMI プレースホルダーを INSERT し、
    notify_results_wt.py が existing_gami を参照できるようにする。
    """
    rounded = round(min_odds, 2)
    pattern = race_key + "#%"
    gami_key = race_key + "#GAMI"
    # race_date を race_key から復元（例: 20260624_37_03 → 2026-06-24）
    _parts = race_key.split("_")
    _d = _parts[0] if _parts else ""
    race_date = f"{_d[:4]}-{_d[4:6]}-{_d[6:8]}" if len(_d) == 8 else date.today().strftime("%Y-%m-%d")

    # SQLite 更新
    try:
        with get_connection() as conn:
            cur = conn.execute(
                "UPDATE picks_history SET prerace_gami = ? WHERE race_key LIKE ?",
                (rounded, pattern),
            )
            if cur.rowcount == 0:
                # #CAND なし → プレースホルダーを INSERT
                conn.execute(
                    "INSERT OR IGNORE INTO picks_history "
                    "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,prerace_gami) "
                    "VALUES (?,?,'GAMI',NULL,0,0,0,0,0,'wt',True,?)",
                    (race_date, gami_key, rounded),
                )
            conn.commit()
    except Exception as e:
        logger.warning("prerace_gami SQLite 書き込み失敗 %s: %s", race_key, e)

    # VPS PostgreSQL 直接更新（KEIRIN_DB_URL 設定時のみ・hourly sync を待たずに即反映）
    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "UPDATE keirin.picks_history SET prerace_gami = %s WHERE race_key LIKE %s",
                        (rounded, pattern),
                    )
                    if cur.rowcount == 0:
                        cur.execute(
                            "INSERT INTO keirin.picks_history "
                            "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,prerace_gami) "
                            "VALUES (%s,%s,'GAMI',NULL,0,0,0,0,0,'wt',TRUE,%s) "
                            "ON CONFLICT DO NOTHING",
                            (race_date, gami_key, rounded),
                        )
        except Exception as e:
            logger.warning("prerace_gami VPS 書き込み失敗 %s: %s", race_key, e)



def _calc_gap23(pick: dict) -> float | None:
    """ピックのモデル予測確率から gap23（2位-3位差, パーセント点）を計算する。

    riders リストの ai_rank 順に並べ、2位と3位の pred_prob_pct の差を返す。
    3人以上いない場合は None を返す。
    """
    riders = pick.get("riders", [])
    sorted_riders = sorted(riders, key=lambda r: r.get("ai_rank", 99))
    if len(sorted_riders) < 3:
        return None
    p2 = sorted_riders[1].get("pred_prob_pct")
    p3 = sorted_riders[2].get("pred_prob_pct")
    if p2 is None or p3 is None:
        return None
    return round(float(p2) - float(p3), 3)


def _save_gap23(race_key: str, gap23: float) -> None:
    """picks_history.gap23 を発走前実測値で保存する（SQLite + VPS PG）。

    UPDATE が 0 件（#CAND エントリが存在しない）の場合はスキップする。
    """
    pattern = race_key + "#%"
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE picks_history SET gap23 = ? WHERE race_key LIKE ?",
                (gap23, pattern),
            )
            conn.commit()
    except Exception as e:
        logger.warning("gap23 SQLite 書き込み失敗 %s: %s", race_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "UPDATE keirin.picks_history SET gap23 = %s WHERE race_key LIKE %s",
                        (gap23, pattern),
                    )
        except Exception as e:
            logger.warning("gap23 VPS 書き込み失敗 %s: %s", race_key, e)


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
        "7PLUS_SS": "🚲⭐", "7PLUS_S": "🚲🔵",
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

        # 7車以外は推奨対象外（ROI 構造的に不利）
        if ri.get("n_entries") != 7:
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

        # gap23（モデル予測確率 2位-3位差）を保存
        gap23_val = _calc_gap23(pick)
        if gap23_val is not None:
            _save_gap23(rk, gap23_val)

        # race_no をピックに付与
        pick_with_raceno = dict(pick)
        pick_with_raceno["race_no"] = ri["race_no"]
        pick_with_raceno["n_entries"] = ri["n_entries"]

        # 候補レース（7PLUS_CAND）は現在オッズで SS/S/A を再判定
        if rank == "7PLUS_CAND":
            live_rank, live_thirds, live_odds = _determine_live_rank(pick, odds_data)
            if live_rank == "なし":
                # gami条件不成立 → kisekiに見送りを即時反映 + 判定を確定記録
                _save_picks_history_state(rk, True)
                _save_decision(today, rk, {
                    "decision": "skip",
                    "pivot1": pick.get("pivot1"), "pivot2": pick.get("pivot2"),
                    "all_min_odds": min_odds,
                    "leg_odds": {str(t): o for t, o in live_odds.items()},
                    **_score_stats(pick),
                })
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
            # prerace_gami を購入目（ガミカット後・全目 >= GAMI_THRESHOLD）の最安値で上書き。
            # 上記 _save_prerace_gami(rk, min_odds) は全thirds最安値を使っており、
            # カット済み低オッズ目が含まれると prerace_gami < 7.0 のまま残り
            # notify_results_wt.py / kiseki 側でガミ見送り扱いに誤記される（SS/S共通）。
            _buy_leg_odds = [live_odds[t] for t in live_thirds if t in live_odds]
            if _buy_leg_odds:
                _save_prerace_gami(rk, min(_buy_leg_odds))
            # ランク確定をkisekiに即時反映（SS/Sとも・カット後の買い目も更新）
            _save_picks_history_state(
                rk, False, live_rank,
                new_pred=(pick_with_raceno["combo_str"], n_pts),
            )
            # 判定を確定記録（翌朝の採点はこの内容で行う）
            _save_decision(today, rk, {
                "decision": "buy",
                "rank": live_rank,
                "pivot1": pivot1, "pivot2": pivot2,
                "thirds": [int(t) for t in live_thirds],
                "leg_odds": {str(t): o for t, o in live_odds.items()},
                "all_min_odds": min_odds,
                **_score_stats(pick),
            })
            print(f"[prerace] {rk} 候補 → live判定: {live_rank} ({n_pts}点)", flush=True)
        else:
            # 非候補（朝ガミ通過済み）: 直前でガミ落ちなら見送りを即時反映
            # SSはガミ目カット済みのため gami判定を適用しない（全目最安値でmiwokuri=Trueにしない）
            if rank != "7PLUS_SS" and min_odds is not None and min_odds < GAMI_THRESHOLD:
                _save_picks_history_state(rk, True)
                _save_decision(today, rk, {
                    "decision": "skip",
                    "pivot1": pick.get("pivot1"), "pivot2": pick.get("pivot2"),
                    "all_min_odds": min_odds,
                    **_score_stats(pick),
                })
            else:
                _save_decision(today, rk, {
                    "decision": "buy",
                    "rank": rank,
                    "pivot1": pick.get("pivot1"), "pivot2": pick.get("pivot2"),
                    "thirds": [int(t) for t in pick.get("thirds", [])],
                    "all_min_odds": min_odds,
                    **_score_stats(pick),
                })

        msg = _build_message(pick_with_raceno, ri, odds_data)
        messages.append((rk, msg))
        newly_done.add(rk)
        time.sleep(0.5)   # Discord レート制限対策

    # 推奨がある場合のみ Discord 送信（ヘッダーなし・詳細メッセージのみ）
    if messages:
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
