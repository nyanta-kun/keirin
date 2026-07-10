#!/usr/bin/env python3
"""
pre-race オッズ確認・Discord 通知（毎分 cron で実行）

発走 15 分前（購入締切 10 分前）に当日ピック済みレースの
現在オッズを winticket からリアルタイム取得し、Discord へ通知する。

cron 設定（crontab -e に追加）:
  * * * * * cd /Users/ysuzuki/GitHub/keirin && .venv/bin/python3 scripts/notify_prerace_wt.py >> /tmp/prerace.log 2>&1

通知ウィンドウ: start_at - 900秒 ≤ now < start_at - 840秒（1分間）
  競輪の購入締切は発走 5 分前 → 締切 10 分前 = 発走 15 分前（900秒前）に通知

通知内容（ランク体系 2026-07-10〜）:
  - SS（三連複・レース単位 min(全目)≥7倍 ∧ gap12≥0.10 ∧ gap23≥1pt・全目購入）
  - S/S+（三連単 1着固定F: 1位→2,3位→全・全目min≥10倍 ∧ gap12≥0.15・S+は200円/点増額）
  - 発走時刻・会場・レース番号・車数・買い目・各目の現在オッズ・ガミ充足確認
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

# ガミ閾値（レース単位: min(全目) < この値 → レース見送り。doc52）
# 2026-07-10 に買い目カット方式(SS/S)を廃止し doc48 のレース単位セマンティクスへ回帰。
# main.py / write_candidates_wt.py の GAMI_THRESHOLD と揃えること。
GAMI_THRESHOLD = 7.0

# 三連単を通知に含めるランク（SS廃止済み・現在該当なし）
TRIFECTA_RANKS = {"SS"}

# 7+車 gap12閾値（SS=旧Rランク成立条件）
SEVEN_PLUS_S_GAP12 = 0.10

# gap23 下限・%ポイント（2位-3位予測確率差 < この値は通知しない）
GAP23_MIN = 1.0

# ── Sランク（三連単 1着固定フォーメーション・doc52追記 2026-07-10）──
# 買い目: 1着=指数1位固定 / 2着=指数2,3位 / 3着=全通り（2×(n-2)点）
# S:  gap12>=0.15 ∧ 購入全目の三連単オッズ min>=10 → 100円/点
# S+: さらに gap12>=0.25 ∧ gap34>=0.04 → 200円/点（増額）
# 検証: 真OOSプール(2025-11〜12+2026-06) S=10.2R/日 的中19.5% ROI115% / S+帯 ROI150%
ST_GAP12 = 0.15
ST_GAMI = 10.0
STP_GAP12 = 0.25
STP_GAP34 = 0.04
ST_STAKE = 100
STP_STAKE = 200


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
    """7PLUS_CANDレースの現在オッズで R を判定する（doc52・2026-07-10 SS/S置き換え）。

    Rランク = レース単位セマンティクス:
      min(全目オッズ) >= GAMI_THRESHOLD ∧ gap12 >= 0.10 ∧ gap23 >= 1pt → 全目購入。
    買い目カット・SOフィルタは廃止（SOは全目合成だと構造的に8を超えないため）。
    的中条件は「軸2車が3着内」で、的中率はオッズに依存しない（モデル起因）。
    検証: 2025通年 的中率29.3%・ROI147.6%（真OOS 11月120%/12月140%/2026-06 299%）

    returns (live_rank, valid_thirds, combo_odds)
      - "7PLUS_R": 条件成立（全目購入）
      - "なし": 購入条件不成立
      - "不明": オッズ取得失敗
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

    if not combo_odds:
        return "なし", [], combo_odds
    if min(combo_odds.values()) < GAMI_THRESHOLD:
        return "なし", [], combo_odds
    if gap12 < SEVEN_PLUS_S_GAP12:
        return "なし", [], combo_odds
    _gap23 = _calc_gap23(pick)
    if _gap23 is not None and _gap23 < GAP23_MIN:
        return "なし", [], combo_odds

    return "7PLUS_R", [t for t in thirds if t in combo_odds], combo_odds


def _calc_gap34(pick: dict) -> float | None:
    """指数3位と4位の予測確率差（0-1スケール）。4人未満は None。"""
    riders = sorted(pick.get("riders", []), key=lambda r: r.get("ai_rank", 99))
    if len(riders) < 4:
        return None
    try:
        return (riders[2]["pred_prob_pct"] - riders[3]["pred_prob_pct"]) / 100.0
    except (KeyError, TypeError):
        return None


def _determine_st_rank(pick: dict, odds_data: dict | None) -> tuple[str, list, dict, int]:
    """三連単Sランク判定（1着=指数1位固定 / 2着=指数2,3位 / 3着=全通り）。

    returns (rank, combos, leg_odds, stake_per_pt)
      - rank: "7PLUS_ST"(S) / "7PLUS_STP"(S+増額) / "なし" / "不明"
      - combos: [(1着, 2着, 3着), ...] 購入目
      - leg_odds: {"a-b-c": odds} 判定に使った三連単オッズ（記録用）
    """
    p1 = pick.get("pivot1")
    p2 = pick.get("pivot2")
    thirds = pick.get("thirds", [])
    gap12 = pick.get("gap12", 0.0)

    if gap12 < ST_GAP12 or len(thirds) < 1:
        return "なし", [], {}, 0
    if odds_data is None:
        return "不明", [], {}, 0

    r3 = thirds[0]  # 指数3位（thirds は指数順）
    frames = [int(p1), int(p2)] + [int(t) for t in thirds]

    lookup = _build_odds_lookup(odds_data, "trifecta")
    combos: list[tuple[int, int, int]] = []
    leg_odds: dict[str, float] = {}
    for s in (int(p2), int(r3)):
        for t in frames:
            if t in (int(p1), s):
                continue
            key = (int(p1), s, t)
            ov = lookup.get(key)
            if ov and float(ov) > 0:
                combos.append(key)
                leg_odds[f"{key[0]}-{key[1]}-{key[2]}"] = float(ov)

    if not combos:
        return "不明", [], {}, 0
    # レース単位ガミ条件: 購入全目の三連単オッズ min >= ST_GAMI
    if min(leg_odds.values()) < ST_GAMI:
        return "なし", [], leg_odds, 0

    gap34 = _calc_gap34(pick)
    if gap12 >= STP_GAP12 and gap34 is not None and gap34 >= STP_GAP34:
        return "7PLUS_STP", combos, leg_odds, STP_STAKE
    return "7PLUS_ST", combos, leg_odds, ST_STAKE


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

    ガミ落ち確定（miwokuri=True）とランク昇格（new_rank='7PLUS_R'）を
    当日中にkisekiへ反映させるために呼ぶ。new_pred（購入買い目）を
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


def _st_combo_str(pick: dict) -> str:
    """三連単フォーメーションの表示文字列（例: 3連単F: 1→2,3→全）。"""
    p1 = pick.get("pivot1")
    p2 = pick.get("pivot2")
    thirds = pick.get("thirds", [])
    r3 = thirds[0] if thirds else "?"
    return f"3連単F: {p1}→{p2},{r3}→全"


def _insert_st_pick(race_key: str, race_date: str, st_rank: str,
                    pick: dict, combos: list, stake: int) -> None:
    """三連単Sランクの購入行 {base}#7ST を picks_history に即時反映する（SQLite + VPS）。

    翌朝の notify_results_wt.py が decisions（{rk}#ST）に基づき最終確定する。
    """
    store_key = race_key + "#7ST"
    combo_str = _st_combo_str(pick)
    n_pts = len(combos)
    bet = n_pts * stake
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, st_rank, combo_str, n_pts, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("ST pick SQLite 書き込み失敗 %s: %s", race_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO keirin.picks_history "
                        "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                        "VALUES (%s,%s,%s,%s,%s,0,0,0,%s,'wt',FALSE) "
                        "ON CONFLICT (race_key) DO UPDATE SET "
                        "rank=EXCLUDED.rank, pred_combo=EXCLUDED.pred_combo, "
                        "n_combos=EXCLUDED.n_combos, bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE",
                        (race_date, store_key, st_rank, combo_str, n_pts, bet),
                    )
        except Exception as e:
            logger.warning("ST pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_st_message(pick: dict, race_info: dict, st_rank: str,
                      combos: list, leg_odds: dict, stake: int) -> str:
    """三連単Sランクの15分前通知メッセージ。"""
    venue = pick.get("venue_name", "?")
    race_no = pick.get("race_no") or race_info.get("race_no", "?")
    start = pick.get("start_time", "--:--")
    n = race_info.get("n_entries", pick.get("n_riders", "?"))
    gap12 = pick.get("gap12", 0.0)
    n_pts = len(combos)
    investment = n_pts * stake
    min_odds = min(leg_odds.values()) if leg_odds else 0.0
    is_plus = st_rank == "7PLUS_STP"
    disp = "7+ S+" if is_plus else "7+ S"
    icon = "🚲💎" if is_plus else "🚲🔷"
    plus_note = f"  ※増額 {stake}円/点（gap12≥{STP_GAP12:.2f}∧gap34≥{STP_GAP34:.2f}）\n" if is_plus else ""
    return (
        f"{icon} **[{disp}]  {venue} {race_no}R  [{n}車]  発走 {start}**\n"
        f"  {_st_combo_str(pick)}（{n_pts}点 / {investment:,}円）\n"
        f"  **全目min {min_odds:.1f}倍**  gap12={gap12:.3f}\n"
        f"{plus_note}"
        f"  的中条件: 1着=指数1位 ∧ 2着=指数2,3位（3着は全通り）"
    )


def _save_prerace_gami(race_key: str, min_odds: float) -> None:
    """picks_history.prerace_gami を発走前実測値で更新する（SQLite + VPS）。

    picks_history の race_key は "{base_key}#7R" / "#CAND" 等のサフィックス付き形式で
    保存されているため、LIKE で一括更新する。ただし三連単行(#7ST)は三連複基準の
    この値と無関係（ガミ条件は三連単オッズ min>=10）のため更新対象から除外する。

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
                "UPDATE picks_history SET prerace_gami = ? WHERE race_key LIKE ? "
                "AND race_key NOT LIKE '%#7ST'",
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
                        "UPDATE keirin.picks_history SET prerace_gami = %s WHERE race_key LIKE %s"
                        " AND race_key NOT LIKE %s",
                        (rounded, pattern, "%#7ST"),
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

    # ランク表示（7PLUS_R = 新SS。内部rankは旧SSとの実績分離のため 7PLUS_R のまま）
    rank_icon = {
        "7PLUS_SS": "🚲⭐", "7PLUS_S": "🚲🔵", "7PLUS_R": "🚲⭐",
        "7PLUS": "🚲", "SS": "⭐", "S": "🔵", "A": "🟢",
    }.get(rank, "▪️")
    rank_disp = {"7PLUS_R": "7+ SS", "7PLUS_SS": "7+ SS", "7PLUS_S": "7+ S"}.get(rank, rank)
    # ガミ表示閾値（レース単位: min全目 >= GAMI_THRESHOLD）
    gami_thr = GAMI_THRESHOLD
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
                gami_ng = " ⚠️" if ov < gami_thr else ""
                lines.append(f"    {p1}{sep}{p2}{sep}{t}:  {ov:.1f}倍{gami_ng}")

        valid_odds = [ov for _, ov in odds_per_bet if ov is not None]
        if valid_odds:
            min_odds = min(valid_odds)
            # 合成オッズ = 1 / Σ(1/odds_i)  ← 全有効目の逆数和の逆数
            synth_odds = 1.0 / sum(1.0 / ov for ov in valid_odds)
            investment = n_pts * 100
            if min_odds >= gami_thr:
                gami_mark = f"✅ ガミOK（全{n_pts}目 最安 {min_odds:.1f}倍 ≥ {gami_thr:.0f}倍）"
            else:
                # レース単位セマンティクス（doc52）: min<閾値はレースごと見送り対象
                gami_mark = f"⚠️ ガミ条件割れ（最安 {min_odds:.1f}倍 < {gami_thr:.0f}倍）— レース見送り対象"
                synth_odds = 0.0
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
    _g23 = _calc_gap23(pick)
    g23_str = f"{_g23:.1f}pt" if _g23 is not None else "—"

    msg = (
        f"{rank_icon} **[{rank_disp}]  {venue} {race_no}R  [{n}車]  発走 {start}**\n"
        f"  {bet_label}({n_pts}点 / {investment}円): `{combo_str}`\n"
        f"  **条件: gap12={gap12:.3f}(≥{SEVEN_PLUS_S_GAP12:.2f}) gap23={g23_str}(≥{GAP23_MIN:.0f}pt)**"
        f"  参考SO:{synth_str}\n"
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
            if live_rank == "不明":
                # オッズ取得失敗 → 再試行の余地を残すため newly_done に追加しない
                print(f"[prerace] {rk} 候補 → オッズ取得不可（次回再試行）", flush=True)
                time.sleep(0.3)
                continue

            # ── 三連単Sランク判定（SS=三連複とは独立・同一レース併存購入可）──
            st_rank, st_combos, st_leg_odds, st_stake = _determine_st_rank(pick, odds_data)
            st_bought = st_rank in ("7PLUS_ST", "7PLUS_STP")
            _r3 = int(pick["thirds"][0]) if pick.get("thirds") else None
            st_record = {
                "decision": "buy" if st_bought else "skip",
                "pivot1": pick.get("pivot1"),
                "seconds": ([int(pick.get("pivot2")), _r3] if _r3 is not None else []),
                "stake": st_stake if st_bought else 0,
                "st_min_odds": round(min(st_leg_odds.values()), 2) if st_leg_odds else None,
            }
            if st_bought:
                st_record["rank"] = st_rank
                st_record["combos"] = [f"{a}-{b}-{c}" for a, b, c in st_combos]
                st_record["leg_odds"] = st_leg_odds
            _save_decision(today, f"{rk}#ST", st_record)

            def _emit_st_buy():
                _insert_st_pick(rk, ri["race_date"], st_rank, pick, st_combos, st_stake)
                messages.append((rk, _build_st_message(pick, ri, st_rank, st_combos, st_leg_odds, st_stake)))
                _label = "S+" if st_rank == "7PLUS_STP" else "S"
                print(f"[prerace] {rk} 候補 → 三連単{_label} ({len(st_combos)}点×{st_stake}円)", flush=True)

            if live_rank == "なし":
                # SS(三連複)不成立 → 見送りを即時反映 + 判定を確定記録
                # （ST購入がある場合はこの後の _insert_st_pick が #7ST 行を miwokuri=False で入れ直す）
                _save_picks_history_state(rk, True)
                _save_decision(today, rk, {
                    "decision": "skip",
                    "pivot1": pick.get("pivot1"), "pivot2": pick.get("pivot2"),
                    "all_min_odds": min_odds,
                    "leg_odds": {str(t): o for t, o in live_odds.items()},
                    **_score_stats(pick),
                })
                if st_bought:
                    _emit_st_buy()
                    print(f"[prerace] {rk} 候補 → 三連複は条件不成立（三連単{st_rank}のみ）", flush=True)
                else:
                    print(f"[prerace] {rk} 候補 → live判定: 条件不成立（通知なし）", flush=True)
                newly_done.add(rk)
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
            # prerace_gami を購入目の最安値で上書き（R は全目購入なので全目 min と一致）。
            _buy_leg_odds = [live_odds[t] for t in live_thirds if t in live_odds]
            if _buy_leg_odds:
                _save_prerace_gami(rk, min(_buy_leg_odds))
            # ランク確定をkisekiに即時反映（買い目も更新）
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
            if st_bought:
                _emit_st_buy()
            print(f"[prerace] {rk} 候補 → live判定: {live_rank} ({n_pts}点)", flush=True)
        else:
            # 非候補（朝ガミ通過済み・detail JSONフォールバック時のみ）: 直前でガミ落ちなら見送りを即時反映
            # SS（旧カット方式・過去日互換）はガミ目カット済みのため gami判定を適用しない
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
