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
  - 発走時刻・会場・レース番号・車数・買い目・各目の現在オッズ・ガミ充足確認

※ S/S+（三連単 1着固定F・7PLUS_ST/STP）は優位性なしのため 2026-07-15 に全廃。
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

from src.bet_display import fold_trifecta_formation, fold_trio_box
from src.database import get_connection
from src.scraper.winticket import WinticketScraper
from src.notify.discord import send
from src.strategy_wt import (
    rank_7h1_stakes,
    rank_9h1_stakes,
    RANK_9H1_NE,
    RANK_9H1_SCORE_MIN,
    S1W_STAKE, S1W_TOP3_GAP_MIN, RANK_7S_STAKE, RANK_7A_STAKE, RANK_7B_STAKE,
    RANK_7C_NE, RANK_7C_LEGS_MIN, rank_7c_select_legs, rank_7c_unit_stake,
    RANK_7C_TRIO_P3_SUM_MIN, rank_7c_cut_legs_by_gap,
    unit_stake,
    rank_7b_select_legs, RANK_9S_STAKE, RANK_9A_STAKE, SS_STAKE,
    RANK_7SS_STAKE,
    line_score_features, rank_7s_gate_label, ss_policy,
)

# 発走前個別通知の送信スイッチ（2026-08-07 ユーザー要望で廃止）。
# **判定と picks_history への記録は続ける**（下の送信箇所のコメント参照）。
PRERACE_NOTIFY_ENABLED = False

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
    ordered = bet_type in ("trifecta", "exacta")
    lookup = {}
    for item in odds_data.get(bet_type, []):
        k = _parse_combo_key(str(item["combination"]), ordered)
        if k:
            lookup[k] = item["odds_value"]
    return lookup


# ── 候補レースのリアルタイムランク判定 ─────────────────────────────────────────

def _policy_ctx(pick: dict) -> tuple[str | None, float | None, int | None, bool | None]:
    """doc53 統合ポリシーの判定コンテキスト (race_type, avg_gap, n_lines, all_solo)。

    candidates.json（2026-07-12以降の wave_picks_wt が出力）に埋め込まれた値を優先し、
    無ければ DB から再構築する（移行日・旧形式候補ファイルのフォールバック）。
    取得不能時は (None, None, None, None) → ポリシーは見送り・増額とも適用しない。
    """
    if "race_type" in pick or "line_avg_gap" in pick:
        return (pick.get("race_type"), pick.get("line_avg_gap"),
                pick.get("line_n_lines"), pick.get("line_all_solo"))
    rk = pick.get("race_key")
    if not rk:
        return None, None, None, None
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT race_type FROM wt_races WHERE race_key = ?", (rk,)).fetchone()
            race_type = row[0] if row else None
            pairs = [(lg, rp) for lg, rp in conn.execute(
                "SELECT line_group, race_point FROM wt_entries WHERE race_key = ?", (rk,))]
        avg_gap, n_lines, all_solo = line_score_features(pairs)
        return race_type, avg_gap, n_lines, all_solo
    except Exception as e:
        logger.warning("policy_ctx 取得失敗 %s: %s", rk, e)
        return None, None, None, None


def _determine_live_rank(
    pick: dict, odds_data: dict | None,
    ctx: tuple | None = None,
) -> tuple[str, list, dict, int, str | None]:
    """7PLUS_CANDレースの現在オッズで R を判定する。

    Rランク = レース単位セマンティクス:
      min(全目オッズ) >= GAMI_THRESHOLD ∧ gap12 >= 0.10 ∧ gap23 >= 1pt → 全目購入。
    ポリシー（2026-07-16〜）: 選抜レースのみ見送り。
    4分戦カット・格差増額（doc53）は実精算方式再検証で方向不一致のため廃止
    （exp_ss_policy_realistic_wt.py: 4分戦=テスト有効/VAL逆効果、格差帯=テスト110%/VAL56%）。
    的中条件は「軸2車が3着内」で、的中率はオッズに依存しない（モデル起因）。

    returns (live_rank, valid_thirds, combo_odds, stake_per_pt, skip_reason)
      - "7PLUS_R": 条件成立（全目購入・stake_per_pt=100）
      - "なし": 購入条件不成立（skip_reason: "選抜"/None=オッズ条件）
      - "不明": オッズ取得失敗
      combo_odds は {third: 現在オッズ}（判定に使った値・記録用）
    """
    p1 = pick.get("pivot1")
    p2 = pick.get("pivot2")
    thirds = pick.get("thirds", [])
    gap12 = pick.get("gap12", 0.0)

    if odds_data is None:
        return "不明", thirds, {}, 0, None

    # 選抜見送りはオッズ非依存 → オッズ判定より先に確定
    if ctx is None:
        ctx = _policy_ctx(pick)
    skip_reason, stake = ss_policy(*ctx)
    if skip_reason:
        return "なし", [], {}, 0, skip_reason

    lookup = _build_odds_lookup(odds_data, "trio")

    # 各目の現在オッズ
    combo_odds: dict[int, float] = {}
    for t in thirds:
        key = frozenset({int(p1), int(p2), int(t)})
        ov = lookup.get(key)
        if ov and float(ov) > 0:
            combo_odds[t] = float(ov)

    if not combo_odds:
        return "なし", [], combo_odds, 0, None
    if min(combo_odds.values()) < GAMI_THRESHOLD:
        return "なし", [], combo_odds, 0, None
    if gap12 < SEVEN_PLUS_S_GAP12:
        return "なし", [], combo_odds, 0, None
    _gap23 = _calc_gap23(pick)
    if _gap23 is not None and _gap23 < GAP23_MIN:
        return "なし", [], combo_odds, 0, None

    return "7PLUS_R", [t for t in thirds if t in combo_odds], combo_odds, stake, None


def _u_third_list(combos: list[str], dark: int, mate: int) -> list[int]:
    """買い目文字列（"a-b-c"）から3車目（軸2車以外）のリストを返す。

    関数名は旧U戦略に由来するが、現在はS7（_insert_rank_7s_pick）が共有利用する
    汎用ヘルパー（U/M戦略は2026-07-23に削除済み）。
    """
    thirds: list[int] = []
    for c in combos:
        try:
            rest = [int(x) for x in str(c).split("-") if int(x) not in (dark, mate)]
        except ValueError:
            continue
        if len(rest) == 1:
            thirds.append(rest[0])
    return sorted(thirds)


def judge_s1(cand: dict, trifecta_lookup: dict) -> tuple[str, dict]:
    """S1(新設計・win軸1着固定)の発走前ライブオッズ判定（純関数・DB非依存）。

    cand:            朝のS1候補JSON行（axis/p1/p2/top3_gap・朝時点で確定済み）
    trifecta_lookup: _build_odds_lookup(odds_data, "trifecta") が返す {tuple: odds} 辞書

    判定（重複排除なし・目オッズ下限なし＝S1W_TOP3_GAP_MINは朝の候補選定で確定済み）:
      ① 盤面（有効オッズ 0<ov<9000 の掲載車）が7車 — 欠車発生なら見送り
      ② 軸/相手が盤面に在籍しているか
      ③ 買い目 = 軸→p1→p2, 軸→p2→p1 の2点（常に両方買う。オッズ未取得の目は除外）

    returns (decision, detail)
      decision: "buy" / "skip" / "不明"（盤面なし→次分再試行）
      detail:   axis/p1/p2 / combos（"a-b-c" 形式） / leg_odds（対象2目のみ）/ skip_reason
    """
    detail: dict = {"axis": None, "p1": None, "p2": None,
                     "combos": [], "leg_odds": {}, "skip_reason": None}
    try:
        axis = int(cand["axis"])
        p1 = int(cand["p1"])
        p2 = int(cand["p2"])
    except (KeyError, TypeError, ValueError):
        detail["skip_reason"] = "候補情報不正"
        return "skip", detail
    detail["axis"], detail["p1"], detail["p2"] = axis, p1, p2

    if not trifecta_lookup:
        return "不明", detail

    valid: dict[tuple, float] = {}
    for k, ov in trifecta_lookup.items():
        try:
            fv = float(ov)
        except (TypeError, ValueError):
            continue
        if 0 < fv < 9000:
            valid[k] = fv
    if not valid:
        return "不明", detail

    board: set[int] = set()
    for k in valid:
        board |= set(k)

    # ① 盤面7車判定（欠車発生なら見送り記録）
    if len(board) != 7:
        detail["skip_reason"] = f"盤面{len(board)}車（欠車）"
        return "skip", detail

    # ② 軸/相手が盤面に在籍しているか
    if axis not in board or p1 not in board or p2 not in board:
        detail["skip_reason"] = "軸/相手が盤面に不在"
        return "skip", detail

    # ③ 買い目 = 軸→p1→p2, 軸→p2→p1 の2点（目オッズ下限なし）
    combo_a = (axis, p1, p2)
    combo_b = (axis, p2, p1)
    ov_a = valid.get(combo_a)
    ov_b = valid.get(combo_b)
    label_a = "-".join(map(str, combo_a))
    label_b = "-".join(map(str, combo_b))
    detail["leg_odds"] = {label_a: ov_a, label_b: ov_b}
    combos = []
    if ov_a is not None:
        combos.append(label_a)
    if ov_b is not None:
        combos.append(label_b)
    detail["combos"] = combos
    if not combos:
        detail["skip_reason"] = "対象2目のオッズなし"
        return "skip", detail

    return "buy", detail


def _load_s1_candidates(today: str) -> list[dict]:
    """当日のS1候補 JSON（昼 + 夜）を読み込む。"""
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s1_candidates.json",
                  f"wave_picks_wt_{today}_night_s1_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("S1候補 JSON 読み込み失敗 %s: %s", p.name, e)
    return out


def _insert_s1_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int) -> None:
    """S1（新設計・ペーパー）の記録行 {base}#7S1 を picks_history に即時反映する（SQLite + VPS PG）。

    実際の賭けはないが、集計・kiseki 表示互換のため bet_amount は名目値
    （n_combos × S1W_STAKE）で記録する。三連単のため trifecta_payout を使う
    （U/Mの trio_payout とは別列）。翌朝の notify_results_wt.py が
    decisions（{rk}#S1）に基づき最終確定（採点）する。
    """
    store_key = race_key + "#7S1"
    bet = n_combos * S1W_STAKE
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trifecta_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, "SEVEN_S1", pred_combo, n_combos, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("S1 pick SQLite 書き込み失敗 %s: %s", race_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO keirin.picks_history "
                        "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trifecta_payout,bet_amount,route,miwokuri) "
                        "VALUES (%s,%s,%s,%s,%s,0,0,0,%s,'wt',FALSE) "
                        "ON CONFLICT (race_key) DO UPDATE SET "
                        "rank=EXCLUDED.rank, pred_combo=EXCLUDED.pred_combo, "
                        "n_combos=EXCLUDED.n_combos, bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE",
                        (race_date, store_key, "SEVEN_S1", pred_combo, n_combos, bet),
                    )
        except Exception as e:
            logger.warning("S1 pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_s1_message(cand: dict, race_info: dict, detail: dict) -> str:
    """S1（新設計・ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis = detail.get("axis")
    p1 = detail.get("p1")
    p2 = detail.get("p2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    tg = cand.get("top3_gap")
    tg_str = f"{float(tg):.3f}" if tg is not None else "—"
    return (
        f"🎯 **[S1・win軸固定検証(記録のみ)]  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: 1着モデル1位 {axis}  相手: 3着内モデル上位2車 {p1}/{p2}\n"
        f"  3連単({n_pts}点 / 名目{n_pts * S1W_STAKE:,}円): "
        f"`{axis}→{p1}={p2}`\n"
        f"  **条件: top3_gap={tg_str}(≥{S1W_TOP3_GAP_MIN})**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_s1_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """S1候補の発走前判定・記録・通知メッセージ生成。

    returns (messages, newly_done)
      messages:   [(s1_key, msg)]（buy 成立分のみ）
      newly_done: 処理完了キー {race_key}#S1 の集合（オッズ取得失敗は含めない=再試行）
    """
    cands = _load_s1_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#S1" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 7:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        s1_key = f"{rk}#S1"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(S1) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} S1候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trifecta_lookup = _build_odds_lookup(odds_data, "trifecta")
        decision, detail = judge_s1(cand, trifecta_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} S1候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        # 判定を確定記録（翌朝の採点は notify_results_wt がこの内容で行う）
        _save_decision(today, s1_key, {
            "decision": decision,
            "rank": "SEVEN_S1",
            "paper": True,
            "stake": S1W_STAKE,
            "top3_gap": cand.get("top3_gap"),
            **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            # 表記: axis→p1=p2（2点とも成立時）。片方のみ成立時は該当目のみ明示。
            if len(combos) == 2:
                pred = f"{detail['axis']}→{detail['p1']}={detail['p2']}"
            else:
                pred = ",".join(combos)
            _insert_s1_pick(rk, today, pred, len(combos))
            messages.append((s1_key, _build_s1_message(cand, ri, detail)))
            print(f"[prerace] {rk} S1候補 → buy（ペーパー・{len(combos)}点）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#7S1")  # 候補行をオッズ見送り表示に更新
            print(f"[prerace] {rk} S1候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(s1_key)
        time.sleep(0.3)
    return messages, newly_done


def judge_rank_7s(cand: dict, trio_lookup: dict) -> tuple[str, dict]:
    """S7（単勝×複勝指数トップ3重なり軸×波乱度選出）の発走前ライブオッズ判定（純関数・DB非依存）。

    cand:        朝のS7候補JSON行（axis1/axis2・朝時点の rank_7s_daily_select() による
                 日次選出＝WT◎◯重なり考慮版で選出済み・2026-07-21〜）
    trio_lookup: _build_odds_lookup(odds_data, "trio") が返す {frozenset: odds} 辞書

    判定（オッズ下限なし＝レース選出自体が朝のaxis_sum日次ランキングで完了済みのため。
    S1との重複排除もなし＝独立戦略）:
      ① 盤面（有効オッズ 0<ov<9000 の掲載車）が7車 — 欠車発生なら見送り
      ② 軸2車が盤面に在籍しているか
      ③ 買い目 = {axis1, axis2, t}（t=残り5車）の三連複5点（オッズ未取得の目は除外）

    returns (decision, detail)
      decision: "buy" / "skip" / "不明"（盤面なし→次分再試行）
      detail:   axis1/axis2 / combos（"a-b-c" 昇順文字列）/ leg_odds（全5目）/ skip_reason
    """
    detail: dict = {"axis1": None, "axis2": None, "combos": [], "leg_odds": {}, "skip_reason": None}
    try:
        axis1 = int(cand["axis1"])
        axis2 = int(cand["axis2"])
    except (KeyError, TypeError, ValueError):
        detail["skip_reason"] = "候補情報不正"
        return "skip", detail
    detail["axis1"] = axis1
    detail["axis2"] = axis2

    if not trio_lookup:
        return "不明", detail

    valid: dict[frozenset, float] = {}
    for k, ov in trio_lookup.items():
        try:
            fv = float(ov)
        except (TypeError, ValueError):
            continue
        if 0 < fv < 9000:
            valid[k] = fv
    if not valid:
        return "不明", detail

    board: set[int] = set()
    for k in valid:
        board |= set(k)

    # ① 盤面7車判定（欠車発生なら見送り記録）
    if len(board) != 7:
        detail["skip_reason"] = f"盤面{len(board)}車（欠車）"
        return "skip", detail

    # ② 軸2車が盤面に在籍しているか
    if axis1 not in board or axis2 not in board:
        detail["skip_reason"] = "軸が盤面に不在"
        return "skip", detail

    # ③ 買い目 = {axis1, axis2, t} の三連複5点のうちオッズ取得できた目のみ（下限なし）
    leg_odds: dict[str, float | None] = {}
    combos: list[str] = []
    for t in sorted(board - {axis1, axis2}):
        label = "-".join(map(str, sorted((axis1, axis2, t))))
        ov = valid.get(frozenset({axis1, axis2, t}))
        leg_odds[label] = ov
        if ov is not None:
            combos.append(label)
    detail["leg_odds"] = leg_odds
    detail["combos"] = combos
    if not combos:
        detail["skip_reason"] = "対象目のオッズなし"
        return "skip", detail

    return "buy", detail


def _exclude_overlapping_races(
    loser_cands: list[dict], winner_cands: list[dict], *, loser: str, winner: str,
) -> list[dict]:
    """`loser_cands` から `winner_cands` と race_key が重複するものを除外する。

    S/A 系ランク（7S vs 7A・9S vs 9A）は選出条件が定義上排他だが、候補JSONが
    昼/夜の2ファイルに分かれるため判定の転びで両方に載りうる。詳細と優先順位の
    根拠は `_load_rank_7a_candidates()` の docstring を参照。
    """
    winner_keys = {c.get("race_key") for c in winner_cands if c.get("race_key")}
    if not winner_keys:
        return loser_cands
    kept, dropped = [], []
    for c in loser_cands:
        if c.get("race_key") in winner_keys:
            dropped.append(c.get("race_key"))
        else:
            kept.append(c)
    if dropped:
        logger.warning(
            "%s候補のうち%d件が%sと重複したため除外（%s優先）: %s",
            loser, len(dropped), winner, winner, ", ".join(sorted(set(dropped))))
    return kept


def _load_rank_7s_candidates(today: str) -> list[dict]:
    """当日のS7候補 JSON（昼 + 夜）を読み込む。"""
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s7_candidates.json",
                  f"wave_picks_wt_{today}_night_s7_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("S7候補 JSON 読み込み失敗 %s: %s", p.name, e)
    return out


def _insert_rank_7s_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int,
                     gate_label: str | None = None) -> None:
    """S7（波乱度選出・ペーパー）の記録行 {base}#7S を picks_history に即時反映する（SQLite + VPS PG）。

    実際の賭けはないが、集計・kiseki 表示互換のため bet_amount は名目値
    （n_combos × RANK_7S_STAKE）で記録する（三連複のため trio_payout を使う）。
    翌朝の notify_results_wt.py が decisions（{rk}#S7）に基づき最終確定（採点）する。

    gate_label: "SS"（軸2車がWT◎◯と全く重ならない＝wt_overlap_n=0）/
                "S"（片方だけ重なる＝wt_overlap_n=1）。2026-07-21〜。
    """
    store_key = race_key + "#7S"
    # 賭け金は1レース RACE_BUDGET 円を点数で均等割り（2026-08-07 全ランク統一）。
    # 固定単価を掛けると欠車で点数が減ったとき投資額が予算枠からずれる。
    bet = n_combos * unit_stake(n_combos)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,gate_label) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False,?)",
                (race_date, store_key, "RANK_7S", pred_combo, n_combos, bet, gate_label),
            )
            conn.commit()
    except Exception as e:
        logger.warning("S7 pick SQLite 書き込み失敗 %s: %s", race_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO keirin.picks_history "
                        "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,gate_label) "
                        "VALUES (%s,%s,%s,%s,%s,0,0,0,%s,'wt',FALSE,%s) "
                        "ON CONFLICT (race_key) DO UPDATE SET "
                        "rank=EXCLUDED.rank, pred_combo=EXCLUDED.pred_combo, "
                        "n_combos=EXCLUDED.n_combos, bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE, "
                        "gate_label=EXCLUDED.gate_label",
                        (race_date, store_key, "RANK_7S", pred_combo, n_combos, bet, gate_label),
                    )
        except Exception as e:
            logger.warning("S7 pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_rank_7s_message(cand: dict, race_info: dict, detail: dict, gate_label: str | None) -> str:
    """S7（波乱度選出・ペーパー）の15分前 Discord 通知メッセージ。

    gate_label: "SS"（軸2車がWT◎◯と全く重ならない）/ "S"（片方だけ重なる）。
    2026-07-21〜、軸2車とWT◎◯の重なりに応じてSS/Sの2段階でランク表示する
    （honest全期間検証で重なりが増えるほどROIが悪化すると判明したため）。
    2026-07-23〜2026-07-27の間、SSはさらに軸級班で分岐したSS+観察用サブランクを
    持っていたが、サンプル数不足のためユーザー判断で廃止・SSへ統合した。
    """
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis1 = detail.get("axis1")
    axis2 = detail.get("axis2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    axis_sum = cand.get("axis_sum")
    axis_sum_str = f"{float(axis_sum):.1f}" if axis_sum is not None else "—"
    label = f"7{gate_label}" if gate_label else "S7"
    label_desc = {
        "SS": "WT◎◯と軸2車が全く重ならない",
        "S": "WT◎◯と軸2車が片方だけ重なる",
    }.get(gate_label, "")
    return (
        f"🎲 **[{label}]  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: 単勝×複勝指数トップ3重なり {axis1}/{axis2}"
        + (f"（{label_desc}）" if label_desc else "") + "\n"
        f"  三連複2軸総流し({n_pts}点 × {unit_stake(n_pts):,}円 = "
        f"{n_pts * unit_stake(n_pts):,}円): "
        f"`{axis1}={axis2}流し`\n"
        f"  **軸合計複勝指数(波乱度)={axis_sum_str}**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_rank_7s_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """S7候補の発走前判定・記録・通知メッセージ生成。

    returns (messages, newly_done)
      messages:   [(rank_7s_key, msg)]（buy 成立分のみ）
      newly_done: 処理完了キー {race_key}#S7 の集合（オッズ取得失敗は含めない=再試行）
    """
    cands = _load_rank_7s_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#S7" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 7:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        rank_7s_key = f"{rk}#S7"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(S7) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} S7候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        decision, detail = judge_rank_7s(cand, trio_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} S7候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        wt_overlap_n = cand.get("wt_overlap_n")
        gate_label = rank_7s_gate_label(wt_overlap_n, cand.get("axis1_class"), cand.get("axis2_class"))

        # 判定を確定記録（翌朝の採点は notify_results_wt がこの内容で行う）
        _save_decision(today, rank_7s_key, {
            "decision": decision,
            "rank": "RANK_7S",
            "paper": True,
            "stake": unit_stake(len(detail.get("combos") or [])),
            "axis_sum": cand.get("axis_sum"),
            "wt_overlap_n": wt_overlap_n,
            "gate_label": gate_label,
            **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["axis1"], detail["axis2"])
            pred = (f"{detail['axis1']}={detail['axis2']}-"
                    + ",".join(map(str, thirds)))
            _insert_rank_7s_pick(rk, today, pred, len(combos), gate_label)
            messages.append((rank_7s_key, _build_rank_7s_message(cand, ri, detail, gate_label)))
            print(f"[prerace] {rk} S7候補 → buy（ペーパー・{len(combos)}点・{gate_label}）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#7S")  # 候補行をオッズ見送り表示に更新
            print(f"[prerace] {rk} S7候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(rank_7s_key)
        time.sleep(0.3)
    return messages, newly_done


# ── 旧7SS（波乱軸選出・穴レース検知）は 2026-08-02 に全廃し、判定ロジックごと
#    2026-08-05 に破棄した（同名を別戦略へ充てたため残置の意味が消えた）。
#    **現行の 7SS は entropy不合格 × 軸2車が同一ライン の別物**で、7S/7A と同じ
#    候補JSON方式（wave-picks-wt が生成）に乗る。処理は _process_rank_7ss_candidates()。

def judge_rank_9s(cand: dict, trio_lookup: dict) -> tuple[str, dict]:
    """S9（S7の9車立て版）の発走前ライブオッズ判定（純関数・DB非依存）。

    judge_rank_7s の9車版（盤面判定が9車・買い目が残り7車流し=7点になる点のみ異なる）。
    """
    detail: dict = {"axis1": None, "axis2": None, "combos": [], "leg_odds": {}, "skip_reason": None}
    try:
        axis1 = int(cand["axis1"])
        axis2 = int(cand["axis2"])
    except (KeyError, TypeError, ValueError):
        detail["skip_reason"] = "候補情報不正"
        return "skip", detail
    detail["axis1"] = axis1
    detail["axis2"] = axis2

    if not trio_lookup:
        return "不明", detail

    valid: dict[frozenset, float] = {}
    for k, ov in trio_lookup.items():
        try:
            fv = float(ov)
        except (TypeError, ValueError):
            continue
        if 0 < fv < 9000:
            valid[k] = fv
    if not valid:
        return "不明", detail

    board: set[int] = set()
    for k in valid:
        board |= set(k)

    if len(board) != 9:
        detail["skip_reason"] = f"盤面{len(board)}車（欠車）"
        return "skip", detail

    if axis1 not in board or axis2 not in board:
        detail["skip_reason"] = "軸が盤面に不在"
        return "skip", detail

    leg_odds: dict[str, float | None] = {}
    combos: list[str] = []
    for t in sorted(board - {axis1, axis2}):
        label = "-".join(map(str, sorted((axis1, axis2, t))))
        ov = valid.get(frozenset({axis1, axis2, t}))
        leg_odds[label] = ov
        if ov is not None:
            combos.append(label)
    detail["leg_odds"] = leg_odds
    detail["combos"] = combos
    if not combos:
        detail["skip_reason"] = "対象目のオッズなし"
        return "skip", detail

    return "buy", detail


def _load_rank_9s_candidates(today: str) -> list[dict]:
    """当日のS9候補 JSON（昼 + 夜）を読み込む。"""
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s9_candidates.json",
                  f"wave_picks_wt_{today}_night_s9_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("S9候補 JSON 読み込み失敗 %s: %s", p.name, e)
    return out


def _insert_rank_9s_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int,
                     gate_label: str | None = None) -> None:
    """S9（9車entropy選出・ペーパー）の記録行 {base}#9S を picks_history に即時反映する。

    _insert_rank_7s_pick の9車版（rank='RANK_9S'・race_key末尾#9S・RANK_9S_STAKE）。
    """
    store_key = race_key + "#9S"
    # 賭け金は1レース RACE_BUDGET 円を点数で均等割り（2026-08-07 全ランク統一）。
    # 固定単価を掛けると欠車で点数が減ったとき投資額が予算枠からずれる。
    bet = n_combos * unit_stake(n_combos)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,gate_label) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False,?)",
                (race_date, store_key, "RANK_9S", pred_combo, n_combos, bet, gate_label),
            )
            conn.commit()
    except Exception as e:
        logger.warning("S9 pick SQLite 書き込み失敗 %s: %s", race_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO keirin.picks_history "
                        "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,gate_label) "
                        "VALUES (%s,%s,%s,%s,%s,0,0,0,%s,'wt',FALSE,%s) "
                        "ON CONFLICT (race_key) DO UPDATE SET "
                        "rank=EXCLUDED.rank, pred_combo=EXCLUDED.pred_combo, "
                        "n_combos=EXCLUDED.n_combos, bet_amount=EXCLUDED.bet_amount, miwokuri=FALSE, "
                        "gate_label=EXCLUDED.gate_label",
                        (race_date, store_key, "RANK_9S", pred_combo, n_combos, bet, gate_label),
                    )
        except Exception as e:
            logger.warning("S9 pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_rank_9s_message(cand: dict, race_info: dict, detail: dict, gate_label: str | None) -> str:
    """S9（9車entropy選出・ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis1 = detail.get("axis1")
    axis2 = detail.get("axis2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    axis_sum = cand.get("axis_sum")
    axis_sum_str = f"{float(axis_sum):.1f}" if axis_sum is not None else "—"
    label = f"9{gate_label}" if gate_label else "S9"
    label_desc = {
        "SS": "WT◎◯と軸2車が全く重ならない",
        "S": "WT◎◯と軸2車が片方だけ重なる",
    }.get(gate_label, "")
    return (
        f"🎲 **[{label}]（9車立て）  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: 単勝×複勝指数トップ3重なり {axis1}/{axis2}"
        + (f"（{label_desc}）" if label_desc else "") + "\n"
        f"  三連複2軸総流し({n_pts}点 × {unit_stake(n_pts):,}円 = "
        f"{n_pts * unit_stake(n_pts):,}円): "
        f"`{axis1}={axis2}流し`\n"
        f"  **軸合計複勝指数(波乱度)={axis_sum_str}**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_rank_9s_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """S9候補の発走前判定・記録・通知メッセージ生成（_process_rank_7s_candidates の9車版）。

    returns (messages, newly_done)
      messages:   [(rank_9s_key, msg)]（buy 成立分のみ）
      newly_done: 処理完了キー {race_key}#S9 の集合（オッズ取得失敗は含めない=再試行）
    """
    cands = _load_rank_9s_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#S9" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 9:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        rank_9s_key = f"{rk}#S9"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(S9) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} S9候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        decision, detail = judge_rank_9s(cand, trio_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} S9候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        wt_overlap_n = cand.get("wt_overlap_n")
        gate_label = rank_7s_gate_label(wt_overlap_n, cand.get("axis1_class"), cand.get("axis2_class"))

        _save_decision(today, rank_9s_key, {
            "decision": decision,
            "rank": "RANK_9S",
            "paper": True,
            "stake": unit_stake(len(detail.get("combos") or [])),
            "axis_sum": cand.get("axis_sum"),
            "wt_overlap_n": wt_overlap_n,
            "gate_label": gate_label,
            **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["axis1"], detail["axis2"])
            pred = (f"{detail['axis1']}={detail['axis2']}-"
                    + ",".join(map(str, thirds)))
            _insert_rank_9s_pick(rk, today, pred, len(combos), gate_label)
            messages.append((rank_9s_key, _build_rank_9s_message(cand, ri, detail, gate_label)))
            print(f"[prerace] {rk} S9候補 → buy（ペーパー・{len(combos)}点・{gate_label}）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#9S")  # 候補行をオッズ見送り表示に更新
            print(f"[prerace] {rk} S9候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(rank_9s_key)
        time.sleep(0.3)
    return messages, newly_done


# ── 7A/9A（S7/S9の境界ランク・3ゲート/2ゲート中1つだけ不合格・2026-07-27導入） ──
# 盤面判定・買い目構成のロジックは車数依存部分のみで、S7/S9本体と全く同一
# （軸2車+残り流し）のため judge_rank_7s()/judge_rank_9s() をそのまま再利用する。

def _load_rank_7a_candidates(today: str) -> list[dict]:
    """当日の7A候補 JSON（昼 + 夜）を読み込む。7Sと重複するレースは除外する。

    7Sと7Aは `rank_7s_daily_select()`（2ゲート不合格0個）と
    `rank_7a_daily_select()`（ちょうど1個）で**定義上排他**だが、生成が
    昼バッチ・夜バッチの2回に分かれていた時期には同一レースが両方に載りえた:
    朝は1ゲート不合格で7A→夕方の再収集でライン情報が更新され entropy/axis_sum が
    変化して0個不合格＝7S、という具合に判定が転ぶと、`_s7a_candidates.json`（昼）と
    `_night_s7_candidates.json`（夜）の双方に同じレースが載り、本ローダは
    昼+夜を無条件に連結するため両方が判定・記録されて `#7A` と `#7S` の
    2行が picks_history に書かれてしまう（1レースに1,000円投資として二重計上。
    軸が変わらなければ買い目まで完全同一になる）。

    実測: 2026-07-28〜31 に6レースで発生（うち3件は買い目も完全一致）。7Aの導入が
    2026-07-27なので、発生しうる全期間で6件＝取りこぼしなく検出できている。

    2026-08-01 の8:00単一バッチ一本化（commit adec6aa）で夜バッチは cron から
    外れたため通常運用では再発しないが、`evening_picks_wt.sh` は手動/アドホック
    実行用に残置されており実行されれば夜ファイルが生成される。cron 構成に
    依存しない構造的なガードとしてここで排他を保証する。

    どちらを優先するか: **7S**。7Sは2ゲートとも合格＝7A（1つ不合格の「惜しい
    レース」）より厳しい条件を満たしており、実測でも単独時 ROI 80.6% / 的中率
    42.0% と 7A（78.0% / 45.0%）以上。判定が転んだ場合は新しい情報（夕方の
    ライン確定後）に基づく方を採るのが筋でもある。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s7a_candidates.json",
                  f"wave_picks_wt_{today}_night_s7a_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("7A候補 JSON 読み込み失敗 %s: %s", p.name, e)
    return _exclude_overlapping_races(
        out, _load_rank_7s_candidates(today), loser="7A", winner="7S")


def _insert_rank_7a_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int) -> None:
    """7A（境界ランク・ペーパー）の記録行 {base}#7A を picks_history に即時反映する。

    _insert_rank_7s_pick の7A版（rank='RANK_7A'・race_key末尾#7A・RANK_7A_STAKE・gate_labelなし）。
    """
    store_key = race_key + "#7A"
    # 賭け金は1レース RACE_BUDGET 円を点数で均等割り（2026-08-07 全ランク統一）。
    # 固定単価を掛けると欠車で点数が減ったとき投資額が予算枠からずれる。
    bet = n_combos * unit_stake(n_combos)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, "RANK_7A", pred_combo, n_combos, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("7A pick SQLite 書き込み失敗 %s: %s", race_key, e)

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
                        (race_date, store_key, "RANK_7A", pred_combo, n_combos, bet),
                    )
        except Exception as e:
            logger.warning("7A pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_rank_7a_message(cand: dict, race_info: dict, detail: dict) -> str:
    """7A（境界ランク・ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis1 = detail.get("axis1")
    axis2 = detail.get("axis2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    axis_sum = cand.get("axis_sum")
    axis_sum_str = f"{float(axis_sum):.1f}" if axis_sum is not None else "—"
    return (
        f"🎲 **[7A]  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: 単勝×複勝指数トップ3重なり {axis1}/{axis2}"
        f"（S7の境界ランク・3ゲート中1つだけ不合格）\n"
        f"  三連複2軸総流し({n_pts}点 × {unit_stake(n_pts):,}円 = "
        f"{n_pts * unit_stake(n_pts):,}円): "
        f"`{axis1}={axis2}流し`\n"
        f"  **軸合計複勝指数(波乱度)={axis_sum_str}**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_rank_7a_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """7A候補の発走前判定・記録・通知メッセージ生成（_process_rank_7s_candidates の7A版）。"""
    cands = _load_rank_7a_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#7A" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 7:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        rank_7a_key = f"{rk}#7A"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(7A) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} 7A候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        decision, detail = judge_rank_7s(cand, trio_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} 7A候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        _save_decision(today, rank_7a_key, {
            "decision": decision, "rank": "RANK_7A", "paper": True,
            "stake": unit_stake(len(detail.get("combos") or [])),
            "axis_sum": cand.get("axis_sum"),
            "wt_overlap_n": cand.get("wt_overlap_n"), **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["axis1"], detail["axis2"])
            pred = (f"{detail['axis1']}={detail['axis2']}-"
                    + ",".join(map(str, thirds)))
            _insert_rank_7a_pick(rk, today, pred, len(combos))
            messages.append((rank_7a_key, _build_rank_7a_message(cand, ri, detail)))
            print(f"[prerace] {rk} 7A候補 → buy（ペーパー・{len(combos)}点）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#7A")
            print(f"[prerace] {rk} 7A候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(rank_7a_key)
        time.sleep(0.3)
    return messages, newly_done



# ── 7SS（entropy不合格 × 軸2車が同一ライン・2026-08-05新設）─────────────────
#    ⚠️ 2026-08-02に全廃した旧RANK_7SS（波乱軸選出）とは無関係の別物。
#    7A から entropy 不合格群を分離し、さらに同一ライン条件で絞ったもの。
#    買い目・判定は 7S/7A と同一（三連複 軸2車＋総流し5点）。
#    根拠は strategy_wt.RANK_7SS_STAKE 定義部のコメント参照。

def _load_rank_7ss_candidates(today: str) -> list[dict]:
    """当日の7SS候補 JSON（昼 + 夜）を読み込む。7Sと重複するレースは除外する。

    ⚠️ この 7SS は 2026-08-05 新設の「entropy不合格 × 軸2車が同一ライン」。
       2026-08-02 に全廃した旧 RANK_7SS（波乱軸選出）とは**無関係の別物**。

    7SS(entropy不合格) と 7S(2ゲート合格) / 7A(axis_sumだけ不合格) は定義上排他だが、
    昼/夜の2バッチでライン情報や確率が更新されて判定が転ぶと同一レースが両方に
    載りうる（_load_rank_7a_candidates の docstring 参照・実測6件）。cron 構成に
    依存しない構造的ガードとしてここで排他を保証する。優先は 7S。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s7ss_candidates.json",
                  f"wave_picks_wt_{today}_night_s7ss_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("7SS候補 JSON 読み込み失敗 %s: %s", p.name, e)
    return _exclude_overlapping_races(
        out, _load_rank_7s_candidates(today), loser="7SS", winner="7S")


def _insert_rank_7ss_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int) -> None:
    """7SS（entropy不合格×同一ライン・ペーパー）の記録行 {base}#7SS を picks_history に即時反映する。

    _insert_rank_7s_pick の7SS版（rank='RANK_7SS'・race_key末尾#7SS・RANK_7SS_STAKE・gate_labelなし）。
    """
    store_key = race_key + "#7SS"
    # 賭け金は1レース RACE_BUDGET 円を点数で均等割り（2026-08-07 全ランク統一）。
    # 固定単価を掛けると欠車で点数が減ったとき投資額が予算枠からずれる。
    bet = n_combos * unit_stake(n_combos)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, "RANK_7SS", pred_combo, n_combos, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("7A pick SQLite 書き込み失敗 %s: %s", race_key, e)

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
                        (race_date, store_key, "RANK_7SS", pred_combo, n_combos, bet),
                    )
        except Exception as e:
            logger.warning("7A pick VPS 書き込み失敗 %s: %s", race_key, e)


# ── 7C（ベースモデル・終日の二軸・2026-08-07新設）─────────────────────────
#    選別も軸も「モデル3着内率」だけで決まる（オッズ不使用）。
#    7S/7A/7SS と違い **総流しではなく相手を足切りし、点数が可変**（4〜5点）で、
#    賭け金は 1レース RANK_7C_BUDGET 円の予算枠を点数で割る。
#    根拠は strategy_wt.RANK_7C_P3_SUM_MIN 定義部のセクションコメント参照。

def judge_rank_7c(cand: dict, trio_lookup: dict,
                  trifecta_lookup: dict | None = None) -> tuple[str, dict]:
    """7Cの発走前ライブオッズ判定（純関数・DB非依存）。

    cand:        朝の7C候補JSON行（axis1/axis2 は _load_rank_7c_candidates が
                 7C の軸に差し替え済み。相手は `legs_7c`・`top3_probs` を持つ）
    trio_lookup: _build_odds_lookup(odds_data, "trio") が返す {frozenset: odds}

    判定:
      ① 盤面（有効オッズの掲載車）が RANK_7C_NE 車 — 欠車なら見送り
         （7S/7A/7SS と同じ規約。7C は点数ゲートがあるので欠車で条件が変わりうる）
      ② 軸2車が盤面に在籍
      ③ 相手を**盤面から再計算**する（朝の legs_7c をそのまま使わない）。
         `top3_probs` があれば rank_7c_select_legs を通し、無い旧形式のみ
         legs_7c へフォールバックする（7B と同じ方針）
      ④ 相手が RANK_7C_LEGS_MIN 点未満なら見送り
         ＝「相手が絞れる＝実力差が大きい＝配当が付かない」ので買わない
      ⑤ 賭け金 = rank_7c_unit_stake(点数)（予算枠を点数で割る）

    returns (decision, detail)
      decision: "buy" / "skip" / "不明"（盤面なし→次分再試行）
      detail:   axis1/axis2 / combos / leg_odds / stake / skip_reason
    """
    detail: dict = {"axis1": None, "axis2": None, "combos": [], "leg_odds": {},
                    "stake": None, "skip_reason": None, "bet_kind": "trio"}
    try:
        axis1 = int(cand["axis1"])
        axis2 = int(cand["axis2"])
    except (KeyError, TypeError, ValueError):
        detail["skip_reason"] = "候補情報不正"
        return "skip", detail
    detail["axis1"], detail["axis2"] = axis1, axis2

    if not trio_lookup:
        return "不明", detail

    valid: dict[frozenset, float] = {}
    for k, ov in trio_lookup.items():
        try:
            fv = float(ov)
        except (TypeError, ValueError):
            continue
        if 0 < fv < 9000:
            valid[k] = fv
    if not valid:
        return "不明", detail

    board: set[int] = set()
    for k in valid:
        board |= set(k)

    if len(board) != RANK_7C_NE:
        detail["skip_reason"] = f"盤面{len(board)}車（欠車）"
        return "skip", detail
    if axis1 not in board or axis2 not in board:
        detail["skip_reason"] = "軸が盤面に不在"
        return "skip", detail

    others = sorted(board - {axis1, axis2})
    probs = {int(k): float(v) for k, v in (cand.get("top3_probs") or {}).items()}
    if probs:
        legs = rank_7c_select_legs(others, probs)
    else:
        legs = [x for x in (cand.get("legs_7c") or []) if x in board]

    if len(legs) < RANK_7C_LEGS_MIN:
        detail["skip_reason"] = f"相手{len(legs)}点（{RANK_7C_LEGS_MIN}点未満・低配当回避）"
        return "skip", detail

    # ⚠️ leg_odds のキーは**買い目の文字列**（"2-4-7"）にすること。
    #    7S/9S/7B と同じ規約で、Discord のメッセージ生成が combos の各要素で
    #    leg_odds を引くため。3列目の車番をキーにすると全件「取得不可」表示になる
    #    （2026-08-07 の 7C 初日に実際にそうなった）。
    # 🔴 三連単へ切り替えるレース（`trifecta_7c`）でも **点数のゲートは三連複の板で行う**。
    #    三連単の板は三連複より薄く、そちらで足切りすると 7C の 16.9% が
    #    「オッズ取得できた目が N点」で黙って見送りになる。買い方が変わっただけで
    #    母集団まで変わってはいけない（点数・賭け金も三連複と同一が採用根拠）。
    #    三連単オッズは**表示のためだけ**に引き、取れなくても見送らない。
    # 🔴 券種は**朝の生予測で確定した真偽値**を読む（段階1の設計）。ここで
    #    win_probs から再判定してはいけない（候補JSONは win_probs を持たない
    #    ので静かに全件 False になる／根拠の二重管理にもなる）。
    use_trifecta = bool(cand.get("trifecta_7c"))
    if not use_trifecta:
        # 三連複側だけの追加ゲート（2026-08-09）。三連単は絞らない。
        _p3sum = cand.get("p3_sum_top2")
        if _p3sum is None or float(_p3sum) < RANK_7C_TRIO_P3_SUM_MIN:
            detail["skip_reason"] = (
                f"二軸の3着内率合計が{RANK_7C_TRIO_P3_SUM_MIN}未満（三連複側のゲート）")
            return "skip", detail
        # 3着内率の落差で打ち切る（差が無ければ削らない）。盤面から再計算した
        # legs に対して掛ける（欠車で相手が変われば削る位置も変わるべきなので、
        # 朝の `legs_7c_buy` は使わない）。
        legs = rank_7c_cut_legs_by_gap(legs, probs) if probs else legs
    combos, leg_odds = [], {}
    for t in legs:
        key = frozenset({axis1, axis2, t})
        ov = valid.get(key)
        if ov is None:
            continue
        if use_trifecta:
            # 1着=軸1 / 2着=軸2 / 3着=相手 の順序付きラベル。
            label = f"{axis1}-{axis2}-{t}"
            tov = (trifecta_lookup or {}).get((axis1, axis2, t))
            try:
                fv = float(tov) if tov is not None else None
            except (TypeError, ValueError):
                fv = None
            if fv is not None and 0 < fv < 9000:
                leg_odds[label] = fv
        else:
            label = "-".join(map(str, sorted(key)))
            leg_odds[label] = ov
        combos.append(label)
    # 🔴 必要点数は**買う点数**で判定する。`RANK_7C_LEGS_MIN`(=4) は
    #    「相手が4点未満なら配当が付かないので見送る」という**選別**の閾値で、
    #    三連複側を上位2点に絞った後（2026-08-09）にここへ流用すると
    #    **常に見送りになる**。実際にそれで全件 skip になった。
    # 削った後は1点まで縮みうるので、必要点数も実際の買い目数に合わせる。
    min_pts = RANK_7C_LEGS_MIN if use_trifecta else len(legs)
    if len(combos) < min_pts:
        detail["skip_reason"] = f"オッズ取得できた目が{len(combos)}点"
        return "skip", detail

    detail["combos"] = combos
    detail["leg_odds"] = leg_odds
    detail["thirds"] = legs
    detail["stake"] = rank_7c_unit_stake(len(combos))
    # 🔴 採点側（`notify_results_wt.py`）はこの値で**着順を見るかどうか**を決める。
    #    欠けると三連単を三連複として採点し、着順違いを的中に数えてしまう。
    detail["bet_kind"] = "trifecta" if use_trifecta else "trio"
    return "buy", detail


def _load_rank_7c_candidates(today: str) -> list[dict]:
    """当日の7C候補 JSON（昼 + 夜）を読み込む。

    ⚠️ **他ランクとの排他ガードは掛けない。** 7C は wt_overlap_n を一切見ないので
       7S/7A/7SS/7B/7H1 と同一レースに併存するのが正常な設計であり、
       `_exclude_overlapping_races` を掛けると本来の母集団が削れる。
       1レース1商品の制約は netkeirin 入稿側だけで解決する。

    ⚠️ 7C の軸は候補JSONの `axis1_7c`/`axis2_7c`（pred_top3 上位2車）で、
       `axis1`/`axis2`（3ヘッド軸）とは**別物**。judge_rank_7s は `axis1`/`axis2`
       を読むため、ここで 7C の軸を差し込んだコピーを返す。元の3ヘッド軸は
       `axis1_3head`/`axis2_3head` に退避して調査時に追えるようにする。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    raw: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s7c_candidates.json",
                  f"wave_picks_wt_{today}_night_s7c_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                raw += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("7C候補 JSON 読み込み失敗 %s: %s", p.name, e)
    out: list[dict] = []
    for c in raw:
        if c.get("axis1_7c") is None or c.get("axis2_7c") is None:
            logger.warning("7C候補に軸がありません（スキップ）: %s", c.get("race_key"))
            continue
        d = dict(c)
        d["axis1_3head"], d["axis2_3head"] = c.get("axis1"), c.get("axis2")
        d["axis1"], d["axis2"] = c["axis1_7c"], c["axis2_7c"]
        out.append(d)
    return out


def _insert_rank_7c_pick(race_key: str, race_date: str, pred_combo: str,
                         n_combos: int, stake: int) -> None:
    """7C（ベースモデル・ペーパー）の記録行 {base}#7C を picks_history に即時反映する。

    ⚠️ 他ランクと違い **stake が可変**（予算枠 ÷ 点数）なので呼び出し側から渡す。
       固定 STAKE を掛けると投資額が実態とずれて ROI が壊れる。
    """
    store_key = race_key + "#7C"
    bet = n_combos * stake
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, "RANK_7C", pred_combo, n_combos, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("7C pick SQLite 書き込み失敗 %s: %s", race_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as vconn:
                with vconn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO keirin.picks_history "
                        "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                        "VALUES (%s,%s,%s,%s,%s,0,0,0,%s,'wt',False) "
                        "ON CONFLICT (race_key) DO UPDATE SET "
                        "rank=EXCLUDED.rank, pred_combo=EXCLUDED.pred_combo, "
                        "n_combos=EXCLUDED.n_combos, bet_amount=EXCLUDED.bet_amount",
                        (race_date, store_key, "RANK_7C", pred_combo, n_combos, bet),
                    )
        except Exception as e:
            logger.warning("7C pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_rank_7c_message(cand: dict, race_info: dict, detail: dict) -> str:
    """7C（ベースモデル・ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis1, axis2 = detail.get("axis1"), detail.get("axis2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    p3sum = cand.get("p3_sum_top2")
    p3_str = f"{100 * float(p3sum):.1f}%" if p3sum is not None else "—"
    stake = int(detail.get("stake") or 0)
    thirds = detail.get("thirds") or []
    return (
        f"🧭 **[7C]  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: 複勝率トップ2 {axis1}/{axis2}（ベースモデル・終日対象）\n"
        f"  三連複 軸2車－相手{n_pts}車({n_pts}点 × {stake:,}円 = "
        f"{n_pts * stake:,}円): "
        f"`{axis1}={axis2}-{','.join(map(str, thirds))}`\n"
        f"  **上位2車の複勝率合計={p3_str}**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_rank_7c_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """7C候補の発走前判定・記録・通知メッセージ生成（_process_rank_7ss_candidates の7C版）。"""
    cands = _load_rank_7c_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#7C" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != RANK_7C_NE:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        rank_7c_key = f"{rk}#7C"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(7C) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} 7C候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        # 三連単へ切り替えるレースの表示用。取れなくても見送りにはしない。
        trifecta_lookup = _build_odds_lookup(odds_data, "trifecta")
        decision, detail = judge_rank_7c(cand, trio_lookup, trifecta_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} 7C候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        _save_decision(today, rank_7c_key, {
            "decision": decision, "rank": "RANK_7C", "paper": True,
            "p3_sum_top2": cand.get("p3_sum_top2"),
            "wt_overlap_n": cand.get("wt_overlap_n"), **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["axis1"], detail["axis2"])
            # 🔴 pred_combo は「何を買ったか」の記録。三連単は **`三単:` を付け、
            #    軸を `=`(順不同) ではなく `-`(着順) で書く**。結果通知
            #    （notify_race_result_wt）に券種を伝える手段はこの表記だけ。
            if detail.get("bet_kind") == "trifecta":
                pred = (f"三単:{detail['axis1']}-{detail['axis2']}-"
                        + ",".join(map(str, thirds)))
            else:
                pred = (f"{detail['axis1']}={detail['axis2']}-"
                        + ",".join(map(str, thirds)))
            _insert_rank_7c_pick(rk, today, pred, len(combos), int(detail["stake"]))
            messages.append((rank_7c_key, _build_rank_7c_message(cand, ri, detail)))
            print(f"[prerace] {rk} 7C候補 → buy（ペーパー・{len(combos)}点）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#7C")
            print(f"[prerace] {rk} 7C候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(rank_7c_key)
        time.sleep(0.3)
    return messages, newly_done


def _build_rank_7ss_message(cand: dict, race_info: dict, detail: dict) -> str:
    """7A（境界ランク・ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis1 = detail.get("axis1")
    axis2 = detail.get("axis2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    axis_sum = cand.get("axis_sum")
    axis_sum_str = f"{float(axis_sum):.1f}" if axis_sum is not None else "—"
    return (
        f"🎲 **[7A]  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: 単勝×複勝指数トップ3重なり {axis1}/{axis2}"
        f"（S7の境界ランク・3ゲート中1つだけ不合格）\n"
        f"  三連複2軸総流し({n_pts}点 × {unit_stake(n_pts):,}円 = "
        f"{n_pts * unit_stake(n_pts):,}円): "
        f"`{axis1}={axis2}流し`\n"
        f"  **軸合計複勝指数(波乱度)={axis_sum_str}**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_rank_7ss_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """7SS候補の発走前判定・記録・通知メッセージ生成（_process_rank_7s_candidates の7SS版）。"""
    cands = _load_rank_7ss_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#7SS" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 7:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        rank_7ss_key = f"{rk}#7SS"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(7A) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} 7SS候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        decision, detail = judge_rank_7s(cand, trio_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} 7SS候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        _save_decision(today, rank_7ss_key, {
            "decision": decision, "rank": "RANK_7SS", "paper": True,
            "stake": unit_stake(len(detail.get("combos") or [])),
            "axis_sum": cand.get("axis_sum"),
            "wt_overlap_n": cand.get("wt_overlap_n"), **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["axis1"], detail["axis2"])
            pred = (f"{detail['axis1']}={detail['axis2']}-"
                    + ",".join(map(str, thirds)))
            _insert_rank_7ss_pick(rk, today, pred, len(combos))
            messages.append((rank_7ss_key, _build_rank_7ss_message(cand, ri, detail)))
            print(f"[prerace] {rk} 7SS候補 → buy（ペーパー・{len(combos)}点）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#7SS")
            print(f"[prerace] {rk} 7SS候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(rank_7ss_key)
        time.sleep(0.3)
    return messages, newly_done




def judge_rank_7b(cand: dict, trio_lookup: dict) -> tuple[str, dict]:
    """7B（◎◯一致だが順序・相手で不一致）の発走前ライブオッズ判定（純関数・DB非依存）。

    judge_rank_7s との違いは**買い目の作り方だけ**（盤面チェックは同一）:
      7S/7A: 軸2車 + 残り5車の総流し（5点）
      7B   : 軸2車 + 相手（残り5車から WT△(ana) を除外し pred_prob 上位
             RANK_7B_LEGS 車）＝ 3点

    相手は朝の候補JSONの `legs_7b` をそのまま使わず、**発走前の盤面から再計算する**
    （欠車で相手が盤面から消えた場合に、朝の3点のうち買える目だけが残って
    2点・1点に痩せるのを防ぐ。盤面7車チェックを通っている以上ここで相手が
    消えることは通常ないが、朝の出走表と盤面がズレた場合のフェイルセーフ）。
    `top3_probs` が候補JSONに無い旧形式の場合のみ `legs_7b` へフォールバックする。

    returns (decision, detail) — judge_rank_7s と同一の形式。
    """
    detail: dict = {"axis1": None, "axis2": None, "combos": [], "leg_odds": {},
                    "skip_reason": None, "dropped_ana": None}
    try:
        axis1 = int(cand["axis1"])
        axis2 = int(cand["axis2"])
    except (KeyError, TypeError, ValueError):
        detail["skip_reason"] = "候補情報不正"
        return "skip", detail
    detail["axis1"] = axis1
    detail["axis2"] = axis2

    if not trio_lookup:
        return "不明", detail

    valid: dict[frozenset, float] = {}
    for k, ov in trio_lookup.items():
        try:
            fv = float(ov)
        except (TypeError, ValueError):
            continue
        if 0 < fv < 9000:
            valid[k] = fv
    if not valid:
        return "不明", detail

    board: set[int] = set()
    for k in valid:
        board |= set(k)

    # ① 盤面7車判定（7S/7Aと同一。欠車発生なら見送り）
    if len(board) != 7:
        detail["skip_reason"] = f"盤面{len(board)}車（欠車）"
        return "skip", detail

    # ② 軸2車が盤面に在籍しているか
    if axis1 not in board or axis2 not in board:
        detail["skip_reason"] = "軸が盤面に不在"
        return "skip", detail

    # ③ 相手を盤面から再計算（△除外・pred_prob上位RANK_7B_LEGS車）
    wt_ana = cand.get("wt_ana")
    detail["dropped_ana"] = wt_ana
    raw_probs = cand.get("top3_probs") or {}
    others = sorted(board - {axis1, axis2})
    if raw_probs:
        probs = {int(k): float(v) for k, v in raw_probs.items()}
        legs = rank_7b_select_legs(others, probs, wt_ana)
    else:
        legs = [x for x in (cand.get("legs_7b") or []) if x in board]
    if not legs:
        detail["skip_reason"] = "相手が選定できない"
        return "skip", detail

    # ④ 買い目 = {axis1, axis2, t}（t=選定した相手）のうちオッズ取得できた目のみ
    leg_odds: dict[str, float | None] = {}
    combos: list[str] = []
    for t in legs:
        label = "-".join(map(str, sorted((axis1, axis2, t))))
        ov = valid.get(frozenset({axis1, axis2, t}))
        leg_odds[label] = ov
        if ov is not None:
            combos.append(label)
    detail["leg_odds"] = leg_odds
    detail["combos"] = combos
    if not combos:
        detail["skip_reason"] = "対象目のオッズなし"
        return "skip", detail

    return "buy", detail


def _load_rank_7b_candidates(today: str) -> list[dict]:
    """当日の7B候補 JSON（昼 + 夜）を読み込む。

    7B は `wt_overlap_n == 2` のみ、7S/7A は `wt_overlap_n ∈ {0,1}` のみを取るため
    **定義上完全に排他**（7A vs 7S のようなゲート個数の境界による転びも起こらない）。
    それでも 7S/7A と同じ重複ガードを掛けるのは、昼/夜の2ファイル間で
    prediction_mark の取得状況が変わり overlap 判定が転ぶ可能性が理論上あるため
    （フェイルセーフ。実際に発火したらログに警告が出る）。優先順位は 7S > 7A > 7B。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s7b_candidates.json",
                  f"wave_picks_wt_{today}_night_s7b_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("7B候補 JSON 読み込み失敗 %s: %s", p.name, e)
    out = _exclude_overlapping_races(
        out, _load_rank_7s_candidates(today), loser="7B", winner="7S")
    return _exclude_overlapping_races(
        out, _load_rank_7a_candidates(today), loser="7B", winner="7A")


def _insert_rank_7b_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int) -> None:
    """7B（ペーパー）の記録行 {base}#7B を picks_history に即時反映する。

    _insert_rank_7a_pick の7B版（rank='RANK_7B'・race_key末尾#7B・RANK_7B_STAKE・
    gate_labelなし）。点数は総流しではなく相手を絞った RANK_7B_LEGS 点が基本。
    """
    store_key = race_key + "#7B"
    # 賭け金は1レース RACE_BUDGET 円を点数で均等割り（2026-08-07 全ランク統一）。
    # 固定単価を掛けると欠車で点数が減ったとき投資額が予算枠からずれる。
    bet = n_combos * unit_stake(n_combos)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, "RANK_7B", pred_combo, n_combos, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("7B pick SQLite 書き込み失敗 %s: %s", race_key, e)

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
                        (race_date, store_key, "RANK_7B", pred_combo, n_combos, bet),
                    )
        except Exception as e:
            logger.warning("7B pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_rank_7b_message(cand: dict, race_info: dict, detail: dict) -> str:
    """7B（◎◯一致×順序/相手不一致・ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis1 = detail.get("axis1")
    axis2 = detail.get("axis2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    ana = detail.get("dropped_ana")
    ana_str = f"{ana}番(△)を相手から除外" if ana is not None else "△なし"
    axis_sum = cand.get("axis_sum")
    axis_sum_str = f"{float(axis_sum):.1f}" if axis_sum is not None else "—"
    return (
        f"🎯 **[7B]  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: {axis1}/{axis2}（WT公式印◎◯と一致・ただしモデル1位は◎ではない）\n"
        f"  三連複2軸・相手絞り({n_pts}点 × {unit_stake(n_pts):,}円 = "
        f"{n_pts * unit_stake(n_pts):,}円): "
        f"`{axis1}={axis2}-" + ",".join(str(c.split("-")[-1]) for c in combos) + "`\n"
        f"  **{ana_str}**／軸合計複勝指数(波乱度)={axis_sum_str}\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_rank_7b_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """7B候補の発走前判定・記録・通知メッセージ生成（_process_rank_7a_candidates の7B版）。"""
    cands = _load_rank_7b_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#7B" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 7:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        rank_7b_key = f"{rk}#7B"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(7B) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} 7B候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        decision, detail = judge_rank_7b(cand, trio_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} 7B候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        _save_decision(today, rank_7b_key, {
            "decision": decision, "rank": "RANK_7B", "paper": True,
            "stake": unit_stake(len(detail.get("combos") or [])),
            "axis_sum": cand.get("axis_sum"),
            "wt_overlap_n": cand.get("wt_overlap_n"),
            "order_disagree": cand.get("order_disagree"), **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["axis1"], detail["axis2"])
            pred = (f"{detail['axis1']}={detail['axis2']}-"
                    + ",".join(map(str, thirds)))
            _insert_rank_7b_pick(rk, today, pred, len(combos))
            messages.append((rank_7b_key, _build_rank_7b_message(cand, ri, detail)))
            print(f"[prerace] {rk} 7B候補 → buy（ペーパー・{len(combos)}点）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#7B")
            print(f"[prerace] {rk} 7B候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(rank_7b_key)
        time.sleep(0.3)
    return messages, newly_done



# ═══════════════════════════════════════════════════════════════════════════
# RANK_7H1（穴推奨・本命バスト型）— 2026-08-06 新設
#
# 既存6ランクと違い **三連単+三連複の2券種**を買う。候補JSONは
# `scripts/build_7h1_candidates.py` が朝に生成し、買い目（legs_trio / legs_tf）と
# 100円単位の賭け金（stake_trio / stake_tf）まで確定させてある。
# ここでは**盤面（欠車）だけを見て**買える目に絞り、賭け金を張り直す。
#
# 欠車の扱いは既存ランクの `void_by_dns` と同じ思想:
#   - **三連単の1着固定車（別ライン先頭）が盤面に無い → レース無効（見送り）**
#   - 相手が欠けた → その目だけ落として購入継続（賭け金は残った点数で再計算）
# ═══════════════════════════════════════════════════════════════════════════


def _load_rank_7h1_candidates(today: str) -> list[dict]:
    """当日の7H1候補 JSON（昼 + 夜）を読み込む。"""
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    seen: set[str] = set()
    for fname in (f"wave_picks_wt_{today}_s7h1_candidates.json",
                  f"wave_picks_wt_{today}_night_s7h1_candidates.json"):
        fp = picks_dir / fname
        if not fp.exists():
            continue
        try:
            for c in json.loads(fp.read_text(encoding="utf-8")):
                rk = c.get("race_key")
                if rk and rk not in seen:
                    seen.add(rk)
                    out.append(c)
        except Exception as e:
            logger.warning("7H1候補 JSON 読み込み失敗 %s: %s", fp.name, e)
    return out


def judge_rank_7h1(cand: dict, trio_lookup: dict, tf_lookup: dict) -> tuple[str, dict]:
    """7H1 の発走前判定（純関数・DB非依存）。

    returns (decision, detail)。decision は "buy" / "skip" / "不明"。
    "不明" は盤面が取れていない場合で、呼び出し側は次回に再試行する。
    """
    detail: dict = {"legs_trio": [], "legs_tf": [], "stake_trio": 0, "stake_tf": 0,
                    "bet_amount": 0, "fav": cand.get("fav"), "skip_reason": None,
                    "dropped_trio": 0, "dropped_tf": 0}
    if not trio_lookup or not tf_lookup:
        return "不明", detail

    legs_trio_all = list(cand.get("legs_trio") or [])
    legs_tf_all = list(cand.get("legs_tf") or [])
    if not legs_trio_all or not legs_tf_all:
        detail["skip_reason"] = "候補に買い目が無い"
        return "skip", detail

    # 三連単の1着固定車が盤面から消えていたらレース無効
    head = legs_tf_all[0].split("-")[0]
    if not any(k[0] == int(head) for k in tf_lookup if isinstance(k, tuple)):
        detail["skip_reason"] = f"1着固定{head}番が盤面に無い（欠車）"
        return "skip", detail

    # 🔴 本命（＝バストすると読んだ相手）自身が欠車したら見送る（2026-08-08 追加）。
    # 他ランク（7S/9S/7C/7B）は全て「軸が盤面に不在」を明示的に skip 扱いにしているが、
    # 7H1 だけこの防御が無かった。買い目は設計上 fav を一切含まない（本命ラインを
    # 丸ごと落とす）ので、fav が消えても組み合わせ自体は残り6車で成立してしまい、
    # そのまま "buy" になり得た。
    #
    # だが 7H1 の選別は「**7車ちょうどの盤面で本命1車が沈む**」というレース構造の
    # 予測（favbust モデル）に依存している。fav 自身が欠車した時点でその前提が
    # 崩れ、実質6車レースというモデルが想定していない状況になる。
    #
    # ⚠️ 盤面は三連複(frozenset キー)と三連単(tuple キー)の**両方**から作る。
    #   `_parse_combo_key` は ordered=False で frozenset を返すので、
    #   tuple だけを見ると三連複側を1件も拾えず board が空になり、
    #   このガードが**無言で素通り**する（実装時に実際に踏んだ）。
    fav = cand.get("fav")
    board: set[int] = set()
    for lookup in (trio_lookup, tf_lookup):
        for k in lookup:
            if isinstance(k, (tuple, frozenset)):
                board |= {int(x) for x in k}
    if fav is not None and board and int(fav) not in board:
        detail["skip_reason"] = f"本命{fav}番が盤面に無い（欠車）"
        return "skip", detail

    legs_trio = [t for t in legs_trio_all
                 if _parse_combo_key(t, False) in trio_lookup]
    legs_tf = [t for t in legs_tf_all if _parse_combo_key(t, True) in tf_lookup]
    detail["dropped_trio"] = len(legs_trio_all) - len(legs_trio)
    detail["dropped_tf"] = len(legs_tf_all) - len(legs_tf)
    if not legs_trio or not legs_tf:
        detail["skip_reason"] = "欠車により買い目が全滅"
        return "skip", detail

    u_trio, u_tf, total = rank_7h1_stakes(len(legs_trio), len(legs_tf))
    if not u_trio or not u_tf:
        detail["skip_reason"] = "点数過多で100円未満になる"
        return "skip", detail
    detail.update(legs_trio=legs_trio, legs_tf=legs_tf, stake_trio=u_trio,
                  stake_tf=u_tf, bet_amount=total)
    return "buy", detail


def _insert_rank_7h1_pick(race_key: str, race_date: str, detail: dict) -> None:
    """7H1 の記録行 {base}#7H1 を picks_history に反映する。

    **1レース1行で2券種を合算**する（ユーザー承認・2026-08-06）。
      pred_combo : "三連複: … / 三連単: …"
      n_combos   : 三連複点数 + 三連単点数
      bet_amount : 合算購入額（<= 10,000円）
    券種別の払戻は採点時に trio_payout / trifecta_payout へ入る。
    """
    store_key = race_key + "#7H1"
    pred = ("三複:" + ",".join(detail["legs_trio"])
            + " / 三単:" + ",".join(detail["legs_tf"]))
    n = len(detail["legs_trio"]) + len(detail["legs_tf"])
    bet = int(detail["bet_amount"])
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
                " trio_payout,trifecta_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,0,?,'wt',False)",
                (race_date, store_key, "RANK_7H1", pred, n, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("7H1 pick 書き込み失敗 %s: %s", race_key, e)


def _build_rank_7h1_message(cand: dict, ri: dict, detail: dict) -> str:
    """Discord 通知本文。"""
    fav = detail.get("fav")
    lines = [
        f"**【7H1 穴推奨・本命バスト型】{cand.get('venue_name')}{cand.get('race_no')}R**",
        f"本命 {fav}番（{cand.get('fav_name') or ''}）が飛ぶと読んだレースです。",
        f"抜け度 {float(cand.get('gap12') or 0) * 100:.1f}pt / "
        f"バスト確率 {float(cand.get('bust_prob') or 0) * 100:.1f}%",
        "",
        f"三連複 {len(detail['legs_trio'])}点 × {detail['stake_trio']}円",
        # 全目の列挙は読めないのでフォーメーション表記へ畳む。畳めない構造
        # （欠車でBOXが崩れた等）は元の列挙にフォールバックする（src/bet_display.py）。
        "　" + (fold_trio_box(detail["legs_trio"]) or " ".join(detail["legs_trio"])),
        f"三連単 {len(detail['legs_tf'])}点 × {detail['stake_tf']}円",
        "　" + (fold_trifecta_formation(detail["legs_tf"]) or " ".join(detail["legs_tf"])),
        f"合計 {detail['bet_amount']:,}円",
    ]
    if detail["dropped_trio"] or detail["dropped_tf"]:
        lines.append(f"（欠車により 三複{detail['dropped_trio']}点 / "
                     f"三単{detail['dropped_tf']}点 を除外）")
    return "\n".join(lines)


def _process_rank_7h1_candidates(today: str, now_unix: int,
                                 notified: set[str]) -> tuple[list, set]:
    """7H1候補の発走前判定・記録・通知メッセージ生成。"""
    cands = _load_rank_7h1_candidates(today)
    if not cands:
        return [], set()
    race_info_map = _load_race_info([c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#7H1" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 7:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        key = f"{rk}#7H1"
        try:
            odds_data = scraper.fetch_odds(
                venue_id=ri["venue_id"], race_date=ri["race_date"],
                race_no=ri["race_no"], cup_id=ri["cup_id"],
                day_index=ri["day_index"])
        except Exception as e:
            logger.warning("fetch_odds 失敗(7H1) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} 7H1候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        decision, detail = judge_rank_7h1(
            cand, _build_odds_lookup(odds_data, "trio"),
            _build_odds_lookup(odds_data, "trifecta"))
        if decision == "不明":
            print(f"[prerace] {rk} 7H1候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        # ⚠️ **detail は丸ごと保存する**。採点（notify_results_wt.py の
        #    _slot=="seven_7h1"）が legs_trio / legs_tf / stake_trio / stake_tf /
        #    bet_amount をここから読むため、1つでも間引くと黙って採点できなくなる。
        _save_decision(today, key, {
            "decision": decision, "rank": "RANK_7H1", "paper": True,
            "bust_prob": cand.get("bust_prob"), "gap12": cand.get("gap12"),
            **detail,
        })
        if decision == "buy":
            _insert_rank_7h1_pick(rk, today, detail)
            messages.append((key, _build_rank_7h1_message(cand, ri, detail)))
            print(f"[prerace] {rk} 7H1候補 → buy（三複{len(detail['legs_trio'])}点+"
                  f"三単{len(detail['legs_tf'])}点・{detail['bet_amount']}円）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#7H1")
            print(f"[prerace] {rk} 7H1候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(key)
        time.sleep(0.3)
    return messages, newly_done


# ═══════════════════════════════════════════════════════════════════════════
# RANK_9H1（穴推奨・9車・高配当狙い）— 2026-08-08 新設
#
# 7H1 と同じ穴推奨系だが **9車ちょうど**が対象で、券種は**三連単フォーメーション
# の単一券種**（6点）。候補JSONは `scripts/build_9h1_candidates.py` が朝に生成し、
# 買い目（legs）と賭け金（stake）まで確定させてある。
# ここでは**盤面（欠車）だけを見て**買える目に絞り、賭け金を張り直す。
#
# 欠車の扱いは 7H1 と同じ思想:
#   - **1着固定車（モデル3着内率5位）が盤面に無い → レース無効（見送り）**
#   - 相手が欠けた → その目だけ落として購入継続（賭け金は残った点数で再計算）
# ═══════════════════════════════════════════════════════════════════════════


def _load_rank_9h1_candidates(today: str) -> list[dict]:
    """当日の9H1候補 JSON（昼 + 夜）を読み込む。"""
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    seen: set[str] = set()
    for fname in (f"wave_picks_wt_{today}_s9h1_candidates.json",
                  f"wave_picks_wt_{today}_night_s9h1_candidates.json"):
        fp = picks_dir / fname
        if not fp.exists():
            continue
        try:
            for c in json.loads(fp.read_text(encoding="utf-8")):
                rk = c.get("race_key")
                if rk and rk not in seen:
                    seen.add(rk)
                    out.append(c)
        except Exception as e:
            logger.warning("9H1候補 JSON 読み込み失敗 %s: %s", fp.name, e)
    return out


def judge_rank_9h1(cand: dict, tf_lookup: dict) -> tuple[str, dict]:
    """9H1 の発走前判定（純関数・DB非依存）。

    returns (decision, detail)。decision は "buy" / "skip" / "不明"。
    "不明" は盤面が取れていない場合で、呼び出し側は次回に再試行する。
    """
    detail: dict = {"legs": [], "stake": 0, "bet_amount": 0,
                    "lead": cand.get("lead"), "skip_reason": None, "dropped": 0}
    if not tf_lookup:
        return "不明", detail

    legs_all = list(cand.get("legs") or [])
    if not legs_all:
        detail["skip_reason"] = "候補に買い目が無い"
        return "skip", detail

    # 1着固定車が盤面から消えていたらレース無効（残り5車で組み直したりしない）
    head = legs_all[0].split("-")[0]
    if not any(k[0] == int(head) for k in tf_lookup if isinstance(k, tuple)):
        detail["skip_reason"] = f"1着固定{head}番が盤面に無い（欠車）"
        return "skip", detail

    legs = [t for t in legs_all if _parse_combo_key(t, True) in tf_lookup]
    detail["dropped"] = len(legs_all) - len(legs)
    if not legs:
        detail["skip_reason"] = "欠車により買い目が全滅"
        return "skip", detail

    unit, total = rank_9h1_stakes(len(legs))
    if not unit:
        detail["skip_reason"] = "点数過多で100円未満になる"
        return "skip", detail
    detail.update(legs=legs, stake=unit, bet_amount=total)
    return "buy", detail


def _insert_rank_9h1_pick(race_key: str, race_date: str, detail: dict) -> None:
    """9H1 の記録行 {base}#9H1 を picks_history に反映する。"""
    store_key = race_key + "#9H1"
    pred = "三単:" + ",".join(detail["legs"])
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
                " trio_payout,trifecta_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,0,?,'wt',False)",
                (race_date, store_key, "RANK_9H1", pred, len(detail["legs"]),
                 int(detail["bet_amount"])),
            )
    except Exception as e:
        logger.warning("9H1 pick 書き込み失敗 %s: %s", race_key, e)


def _build_rank_9h1_message(cand: dict, ri: dict, detail: dict) -> str:
    """Discord 通知本文。"""
    lines = [
        f"**【9H1 穴推奨・9車高配当】{cand.get('venue_name')}{cand.get('race_no')}R**",
        f"波乱スコア {float(cand.get('upset_score') or 0):.4f}"
        f"（採用ライン {RANK_9H1_SCORE_MIN:.4f}）",
        f"1着固定 {detail.get('lead')}番（{cand.get('lead_name') or ''}）",
        "",
        f"三連単 {len(detail['legs'])}点 × {detail['stake']:,}円 "
        f"= {detail['bet_amount']:,}円",
        # 全目の列挙は読めないのでフォーメーション表記へ畳む。畳めない構造
        # （欠車で形が崩れた等）は元の列挙にフォールバックする。
        "　" + (fold_trifecta_formation(detail["legs"]) or " ".join(detail["legs"])),
    ]
    if detail["dropped"]:
        lines.append(f"（欠車により {detail['dropped']}点 を除外）")
    return "\n".join(lines)


def _process_rank_9h1_candidates(today: str, now_unix: int,
                                 notified: set[str]) -> tuple[list, set]:
    """9H1候補の発走前判定・記録・通知メッセージ生成。"""
    cands = _load_rank_9h1_candidates(today)
    if not cands:
        return [], set()
    race_info_map = _load_race_info([c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#9H1" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != RANK_9H1_NE:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        key = f"{rk}#9H1"
        try:
            odds_data = scraper.fetch_odds(
                venue_id=ri["venue_id"], race_date=ri["race_date"],
                race_no=ri["race_no"], cup_id=ri["cup_id"],
                day_index=ri["day_index"])
        except Exception as e:
            logger.warning("fetch_odds 失敗(9H1) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} 9H1候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        decision, detail = judge_rank_9h1(
            cand, _build_odds_lookup(odds_data, "trifecta"))
        if decision == "不明":
            print(f"[prerace] {rk} 9H1候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        # ⚠️ **detail は丸ごと保存する**。採点が legs / stake / bet_amount を
        #    ここから読むため、1つでも間引くと黙って採点できなくなる（7H1 と同型）。
        _save_decision(today, key, {
            "decision": decision, "rank": "RANK_9H1", "paper": True,
            "upset_score": cand.get("upset_score"),
            **detail,
        })
        if decision == "buy":
            _insert_rank_9h1_pick(rk, today, detail)
            messages.append((key, _build_rank_9h1_message(cand, ri, detail)))
            print(f"[prerace] {rk} 9H1候補 → buy（三単{len(detail['legs'])}点・"
                  f"{detail['bet_amount']}円）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#9H1")
            print(f"[prerace] {rk} 9H1候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(key)
        time.sleep(0.3)
    return messages, newly_done


def _load_rank_9a_candidates(today: str) -> list[dict]:
    """当日の9A候補 JSON（昼 + 夜）を読み込む。9Sと重複するレースは除外する。

    9S/9A は 7S/7A と同一構造（`rank_9s_daily_select()`＝不合格0個 /
    `rank_9a_daily_select()`＝ちょうど1個で定義上排他）のため同型の重複が
    起こりうる。実測では9S/9Aの重複は0件だったが、これは9Sが極めて希少
    （全期間100件・ゼロの月も多い）で母数が小さいためであり、構造としては
    7S/7Aと同じ穴が空いている。予防的に同じガードを掛ける。
    根拠と優先順位は `_load_rank_7a_candidates()` の docstring を参照。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_s9a_candidates.json",
                  f"wave_picks_wt_{today}_night_s9a_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("9A候補 JSON 読み込み失敗 %s: %s", p.name, e)
    return _exclude_overlapping_races(
        out, _load_rank_9s_candidates(today), loser="9A", winner="9S")


def _insert_rank_9a_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int) -> None:
    """9A（境界ランク・ペーパー）の記録行 {base}#9A を picks_history に即時反映する。

    _insert_rank_9s_pick の9A版（rank='RANK_9A'・race_key末尾#9A・RANK_9A_STAKE・gate_labelなし）。
    """
    store_key = race_key + "#9A"
    # 賭け金は1レース RACE_BUDGET 円を点数で均等割り（2026-08-07 全ランク統一）。
    # 固定単価を掛けると欠車で点数が減ったとき投資額が予算枠からずれる。
    bet = n_combos * unit_stake(n_combos)
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, "RANK_9A", pred_combo, n_combos, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("9A pick SQLite 書き込み失敗 %s: %s", race_key, e)

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
                        (race_date, store_key, "RANK_9A", pred_combo, n_combos, bet),
                    )
        except Exception as e:
            logger.warning("9A pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_rank_9a_message(cand: dict, race_info: dict, detail: dict) -> str:
    """9A（境界ランク・ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    axis1 = detail.get("axis1")
    axis2 = detail.get("axis2")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    axis_sum = cand.get("axis_sum")
    axis_sum_str = f"{float(axis_sum):.1f}" if axis_sum is not None else "—"
    return (
        f"🎲 **[9A]（9車立て）  {venue} {race_no}R  発走 {start}**\n"
        f"  軸: 単勝×複勝指数トップ3重なり {axis1}/{axis2}"
        f"（S9の境界ランク・2ゲート中1つだけ不合格）\n"
        f"  三連複2軸総流し({n_pts}点 × {unit_stake(n_pts):,}円 = "
        f"{n_pts * unit_stake(n_pts):,}円): "
        f"`{axis1}={axis2}流し`\n"
        f"  **軸合計複勝指数(波乱度)={axis_sum_str}**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前）:\n"
        + "\n".join(lines)
    )


def _process_rank_9a_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """9A候補の発走前判定・記録・通知メッセージ生成（_process_rank_9s_candidates の9A版）。"""
    cands = _load_rank_9a_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#9A" in notified:
            continue
        ri = race_info_map.get(rk)
        if ri is None or ri.get("n_entries") != 9:
            continue
        notify_at = int(ri["start_at"]) - NOTIFY_BEFORE_START_SEC
        if notify_at <= now_unix < notify_at + NOTIFY_WINDOW_SEC:
            in_window.append((cand, ri))
    if not in_window:
        return [], set()

    scraper = WinticketScraper(request_interval=1.0)
    messages: list[tuple[str, str]] = []
    newly_done: set[str] = set()
    for cand, ri in in_window:
        rk = cand["race_key"]
        rank_9a_key = f"{rk}#9A"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(9A) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            print(f"[prerace] {rk} 9A候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        decision, detail = judge_rank_9s(cand, trio_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} 9A候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        _save_decision(today, rank_9a_key, {
            "decision": decision, "rank": "RANK_9A", "paper": True,
            "stake": unit_stake(len(detail.get("combos") or [])),
            "axis_sum": cand.get("axis_sum"),
            "wt_overlap_n": cand.get("wt_overlap_n"), **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["axis1"], detail["axis2"])
            pred = (f"{detail['axis1']}={detail['axis2']}-"
                    + ",".join(map(str, thirds)))
            _insert_rank_9a_pick(rk, today, pred, len(combos))
            messages.append((rank_9a_key, _build_rank_9a_message(cand, ri, detail)))
            print(f"[prerace] {rk} 9A候補 → buy（ペーパー・{len(combos)}点）", flush=True)
        else:
            _mark_paper_miwokuri(rk, "#9A")
            print(f"[prerace] {rk} 9A候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(rank_9a_key)
        time.sleep(0.3)
    return messages, newly_done


def _mark_paper_miwokuri(race_key: str, suffix: str) -> None:
    """ペーパー候補行（{rk}#7U/#7M/#7A・bet_amount=0）をオッズ見送りに更新する。

    発走15分前判定が skip のとき、write_candidates_wt が朝に書いた候補行を
    miwokuri=True にして「オッズ見送り」として Web に表示する。
    buy 済み行（bet_amount>0）は対象外（上書きしない）。
    """
    store_key = race_key + suffix
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE picks_history SET miwokuri = True "
                "WHERE race_key = ? AND bet_amount = 0",
                (store_key,),
            )
            conn.commit()
    except Exception as e:
        logger.warning("ペーパー見送り更新 SQLite 失敗 %s: %s", store_key, e)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        try:
            import psycopg2  # noqa: PLC0415
            with psycopg2.connect(db_url) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "UPDATE keirin.picks_history SET miwokuri = TRUE "
                        "WHERE race_key = %s AND bet_amount = 0",
                        (store_key,),
                    )
        except Exception as e:
            logger.warning("ペーパー見送り更新 VPS 失敗 %s: %s", store_key, e)


# ── A（◎一致×波乱×別L先頭・二連単）／S1（6車三連単）は 2026-07-17 全廃 ─────────
# 正規プロトコル再検証で両者とも検証ROI100%超なし → judge/記録/通知を停止。
# 経緯と検証値は src/strategy_wt.py の各セクションコメントを参照。


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
            # ペーパー行（#7U/#7M/#7A/#6S1）は各自の15分前判定が miwokuri を管理するため
            # 旧S1系のガミ落ち/昇格の一括更新に巻き込まない（2026-07-16）
            conn.execute(
                "UPDATE picks_history SET miwokuri = ? WHERE race_key LIKE ? AND route = 'wt' "
                "AND race_key NOT LIKE '%#7U' AND race_key NOT LIKE '%#7M' "
                "AND race_key NOT LIKE '%#7A' AND race_key NOT LIKE '%#6S1'",
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
                        " WHERE race_key LIKE %s AND route = 'wt'"
                        " AND race_key NOT LIKE '%%#7U' AND race_key NOT LIKE '%%#7M'"
                        " AND race_key NOT LIKE '%%#7A' AND race_key NOT LIKE '%%#6S1'",
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
    gap23 は三連複(R)ランクの判定条件のため、三連単行(#7ST)には書き込まない
    （_save_prerace_gami と同じ除外規則）。
    """
    pattern = race_key + "#%"
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE picks_history SET gap23 = ? WHERE race_key LIKE ? "
                "AND race_key NOT LIKE '%#7ST'",
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
                        "UPDATE keirin.picks_history SET gap23 = %s WHERE race_key LIKE %s"
                        " AND race_key NOT LIKE %s",
                        (gap23, pattern, "%#7ST"),
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
    # 表示名は 2026-07-16 に SS→S1 へ改称（内部rankは 7PLUS_R のまま）
    rank_disp = {"7PLUS_R": "7+ S1", "7PLUS_SS": "7+ S1(旧SS)", "7PLUS_S": "7+ S"}.get(rank, rank)
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
    stake_pp = int(pick.get("stake_per_pt") or 100)  # doc53: ライン格差増額時 200

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
            investment = n_pts * stake_pp
            if min_odds >= gami_thr:
                gami_mark = f"✅ ガミOK（全{n_pts}目 最安 {min_odds:.1f}倍 ≥ {gami_thr:.0f}倍）"
            else:
                # レース単位セマンティクス（doc52）: min<閾値はレースごと見送り対象
                gami_mark = f"⚠️ ガミ条件割れ（最安 {min_odds:.1f}倍 < {gami_thr:.0f}倍）— レース見送り対象"
                synth_odds = 0.0
        else:
            synth_odds = 0.0
            investment = n_pts * stake_pp
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
    boost_note = ""  # 格差増額は2026-07-16廃止（常に100円/点）

    msg = (
        f"{rank_icon} **[{rank_disp}]  {venue} {race_no}R  [{n}車]  発走 {start}**\n"
        f"  {bet_label}({n_pts}点 / {investment}円): `{combo_str}`\n"
        f"{boost_note}"
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

    notified = _load_notified(today)
    messages: list[tuple[str, str]] = []   # (race_key, message)
    newly_done: set[str] = set()           # 今回処理完了（条件不成立も含む）
    to_notify = []

    picks = _load_picks(today)
    if picks:
        race_keys = [p["race_key"] for p in picks if "race_key" in p]
        race_info_map = _load_race_info(race_keys)

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

    # ── 推奨メッセージを収集してからまとめて送信 ──
    if to_notify:
        scraper = WinticketScraper(request_interval=1.0)

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
            _ctx = _policy_ctx(pick)  # doc53: (race_type, avg_gap, n_lines, all_solo)
            live_rank, live_thirds, live_odds, ss_stake, ss_skip_reason = \
                _determine_live_rank(pick, odds_data, _ctx)
            if live_rank == "不明":
                # オッズ取得失敗 → 再試行の余地を残すため newly_done に追加しない
                print(f"[prerace] {rk} 候補 → オッズ取得不可（次回再試行）", flush=True)
                time.sleep(0.3)
                continue

            if live_rank == "なし":
                # SS(三連複)不成立 → 見送りを即時反映 + 判定を確定記録
                _save_picks_history_state(rk, True)
                _save_decision(today, rk, {
                    "decision": "skip",
                    "skip_reason": ss_skip_reason or "オッズ条件",
                    "pivot1": pick.get("pivot1"), "pivot2": pick.get("pivot2"),
                    "all_min_odds": min_odds,
                    "leg_odds": {str(t): o for t, o in live_odds.items()},
                    **_score_stats(pick),
                })
                _skip_disp = ss_skip_reason or "オッズ条件"
                print(f"[prerace] {rk} 候補 → live判定: {_skip_disp}で条件不成立（通知なし）", flush=True)
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
            pick_with_raceno["stake"] = n_pts * ss_stake
            pick_with_raceno["stake_per_pt"] = ss_stake
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
                "stake": ss_stake,
                "pivot1": pivot1, "pivot2": pivot2,
                "thirds": [int(t) for t in live_thirds],
                "leg_odds": {str(t): o for t, o in live_odds.items()},
                "all_min_odds": min_odds,
                **_score_stats(pick),
            })
            print(f"[prerace] {rk} 候補 → live判定: {live_rank} ({n_pts}点)", flush=True)
        else:
            # 非候補（detail JSON フォールバック時のみ到達）: 直前オッズで再判定する。
            # candidates.json 欠損時の保険経路。判定・通知の安全側動作は主経路（7PLUS_CAND）と揃える。
            if odds_data is None:
                # オッズ取得失敗 → buy/skip を確定せず次分の実行で再試行
                print(f"[prerace] {rk} 非候補({rank}) → オッズ取得不可（次回再試行）", flush=True)
                time.sleep(0.3)
                continue

            _fb_ctx = _policy_ctx(pick)  # doc53 フォールバック経路もポリシー適用
            if rank in ("7PLUS_ST", "7PLUS_STP"):
                # S/S+（三連単F）は 2026-07-15 に全廃。旧detail JSONの残存行は見送り扱い。
                _save_picks_history_state(rk, True)
                print(f"[prerace] {rk} 非候補 → 三連単ランク廃止済み（見送り・通知なし）", flush=True)
                newly_done.add(rk)
                time.sleep(0.3)
                continue

            # 三連複行（7PLUS_R / 旧互換）: ガミ落ち・オッズ解決不能（欠車等で min_odds=None）は見送り
            # SS（旧カット方式・過去日互換）はガミ目カット済みのため gami判定を適用しない
            _fb_skip, _fb_stake = ss_policy(*_fb_ctx)  # 選抜のみ見送り（2026-07-16〜）
            if (rank != "7PLUS_SS" and (min_odds is None or min_odds < GAMI_THRESHOLD)) or _fb_skip:
                _save_picks_history_state(rk, True)
                _save_decision(today, rk, {
                    "decision": "skip",
                    "skip_reason": _fb_skip or "オッズ条件",
                    "pivot1": pick.get("pivot1"), "pivot2": pick.get("pivot2"),
                    "all_min_odds": min_odds,
                    **_score_stats(pick),
                })
                print(f"[prerace] {rk} 非候補({rank}) → {_fb_skip or '条件'}不成立（見送り・通知なし）", flush=True)
                newly_done.add(rk)
                time.sleep(0.3)
                continue
            pick_with_raceno["stake_per_pt"] = _fb_stake
            _save_decision(today, rk, {
                "decision": "buy",
                "rank": rank,
                "stake": _fb_stake,
                "pivot1": pick.get("pivot1"), "pivot2": pick.get("pivot2"),
                "thirds": [int(t) for t in pick.get("thirds", [])],
                "all_min_odds": min_odds,
                **_score_stats(pick),
            })

        msg = _build_message(pick_with_raceno, ri, odds_data)
        messages.append((rk, msg))
        newly_done.add(rk)
        time.sleep(0.5)   # Discord レート制限対策

    # ── U(S2)候補・M(S3)候補処理 は 2026-07-21 全廃・2026-07-23 コード削除済み ──
    # 対象レース数・的中率・期待値の観点で継続困難と判断し廃止。

    # ── S1候補処理 は 2026-07-31 全廃（呼び出し停止・関数は互換のため残置） ──
    # ユーザー判断により「現在有効なデータとは言えない」として過去分picks_history
    # （SEVEN_S1・1,504件・2024-01-02〜2026-07-30）を削除
    # （バックアップ: data/backup/picks_history_s1_discarded_20260731.csv）。
    # judge_s1/_process_s1_candidates はU/M同様、過去日再採点・分析スクリプト
    # 互換のため残置するが、日次生成の呼び出しは停止する。

    # ── S7候補（単勝×複勝指数重なり軸×波乱度選出・ペーパー）処理 ──────────────
    # U/M/S1との重複排除はない（独立戦略）。try/exceptで既存通知を阻害しない。
    try:
        rank_7s_messages, rank_7s_done = _process_rank_7s_candidates(today, now_unix, notified)
        messages += rank_7s_messages
        newly_done |= rank_7s_done
    except Exception as e:
        logger.exception("S7候補処理失敗（SS/U/M/S1通知には影響しない）: %s", e)

    # ── 7SS候補（波乱軸選出・穴レース検知）は 2026-08-02 に全廃 ──────────────
    # live実績 n=16,298・ROI73.5% と控除率75%を下回り続けたためユーザー判断で停止。
    # 候補生成（src/cli/main.py）も同時に止めているので候補JSONは生成されないが、
    # S1全廃時の教訓（CLAUDE.md: 候補生成/ライブ判定/欠損自動補完の3経路すべてを
    # 止める）に従い、ここの呼び出しもコードレベルで停止する
    # （残置ファイルが手元に残っていても picks_history へ書き戻らないようにする）。
    # （旧RANK_7SSの記述。2026-08-05に同名の別戦略を新設し上で処理している）

    # ── S9候補（S7の9車立て版・独立ランク・ペーパー）処理 ────────────────────
    # 2026-07-26導入。S7等との重複排除はない（独立戦略・車数も異なる）。
    try:
        rank_9s_messages, rank_9s_done = _process_rank_9s_candidates(today, now_unix, notified)
        messages += rank_9s_messages
        newly_done |= rank_9s_done
    except Exception as e:
        logger.exception("S9候補処理失敗（他ランク通知には影響しない）: %s", e)

    # ── 7A候補（S7の境界ランク・ペーパー）処理 ──────────────────────────────
    # 2026-07-27導入。S7とは論理的に排他（3ゲート中1つだけ不合格）。
    try:
        rank_7a_messages, rank_7a_done = _process_rank_7a_candidates(today, now_unix, notified)
        messages += rank_7a_messages
        newly_done |= rank_7a_done
    except Exception as e:
        logger.exception("7A候補処理失敗（他ランク通知には影響しない）: %s", e)

    # ── 7SS候補（entropy不合格 × 軸2車が同一ライン・ペーパー）処理 ─────────────
    # 2026-08-05新設。7A から entropy 不合格群を分離したもので、7S(2ゲート合格)・
    # 7A(axis_sumだけ不合格) とは定義上排他。⚠️旧RANK_7SS(波乱軸選出・全廃)とは別物。
    try:
        rank_7ss_messages, rank_7ss_done = _process_rank_7ss_candidates(today, now_unix, notified)
        messages += rank_7ss_messages
        newly_done |= rank_7ss_done
    except Exception as e:
        logger.exception("7SS候補処理失敗（他ランク通知には影響しない）: %s", e)

    # ── 7H1候補（穴推奨・本命バスト型・三連単+三連複の2券種）処理 ─────────────
    # 2026-08-06新設。既存6ランク（予想ベース）とは母集団も券種も異なる独立ランク。
    try:
        rank_7h1_messages, rank_7h1_done = _process_rank_7h1_candidates(
            today, now_unix, notified)
        messages += rank_7h1_messages
        newly_done |= rank_7h1_done
    except Exception as e:
        logger.exception("7H1候補処理失敗（他ランク通知には影響しない）: %s", e)

    # ── 9H1候補（穴推奨・9車高配当・三連単フォーメーション6点）処理 ───────────
    try:
        rank_9h1_messages, rank_9h1_done = _process_rank_9h1_candidates(
            today, now_unix, notified)
        messages += rank_9h1_messages
        newly_done |= rank_9h1_done
    except Exception as e:
        logger.exception("9H1候補処理失敗（他ランク通知には影響しない）: %s", e)

    # ── 7B候補（◎◯一致だが順序・相手で不一致・三連複3点・ペーパー）処理 ──────
    # 2026-08-03導入。7S/7Aとは wt_overlap_n（7B=2 / 7S・7A∈{0,1}）で定義上排他。
    try:
        rank_7b_messages, rank_7b_done = _process_rank_7b_candidates(today, now_unix, notified)
        messages += rank_7b_messages
        newly_done |= rank_7b_done
    except Exception as e:
        logger.exception("7B候補処理失敗（他ランク通知には影響しない）: %s", e)

    # ── 7C候補（ベースモデル・終日の二軸・三連複5点・ペーパー）処理 ────────────
    # 2026-08-07新設。**他ランクとは排他ではない**（wt_overlap_n を見ないため
    # 同一レースに併存しうる）。1レース1商品の制約は netkeirin 入稿側で解決する。
    try:
        rank_7c_messages, rank_7c_done = _process_rank_7c_candidates(today, now_unix, notified)
        messages += rank_7c_messages
        newly_done |= rank_7c_done
    except Exception as e:
        logger.exception("7C候補処理失敗（他ランク通知には影響しない）: %s", e)

    # ── 9A候補（S9の境界ランク・ペーパー）処理 ──────────────────────────────
    # 2026-07-27導入。S9とは論理的に排他（2ゲート中1つだけ不合格）。
    try:
        rank_9a_messages, rank_9a_done = _process_rank_9a_candidates(today, now_unix, notified)
        messages += rank_9a_messages
        newly_done |= rank_9a_done
    except Exception as e:
        logger.exception("9A候補処理失敗（他ランク通知には影響しない）: %s", e)

    # 旧A候補・旧S1候補（6車三連単）の処理は 2026-07-17 全廃

    # 🔴 **発走前個別通知は 2026-08-07 にユーザー要望で廃止**。
    #    ただし本スクリプトは通知だけの存在ではなく、発走15分前の判定結果を
    #    `picks_history` へ書き込む（`_insert_*_pick`）。**cron から外すと
    #    その書き込みごと止まる**ので、送信だけを落として実行は続ける。
    #    再開は `PRERACE_NOTIFY_ENABLED = True` の1行。
    if messages and PRERACE_NOTIFY_ENABLED:
        for rk, msg in messages:
            send(msg, channel="prerace")
            print(f"[prerace] {rk} → 通知送信完了", flush=True)
            time.sleep(0.5)
    elif messages:
        print(f"[prerace] {today} 推奨{len(messages)}件（通知は廃止・判定の記録のみ）",
              flush=True)
    elif to_notify:
        print(f"[prerace] {today} 推奨なし（オッズ確認のみ・通知スキップ）", flush=True)

    if to_notify or newly_done:
        notified |= newly_done
        _save_notified(today, notified)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
