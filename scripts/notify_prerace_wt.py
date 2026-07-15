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

from src.database import get_connection
from src.scraper.winticket import WinticketScraper
from src.notify.discord import send
from src.strategy_wt import (
    SS_STAKE, U_ENTROPY_MIN, U_LEG_MIN_ODDS, U_MTO_MIN, U_STAKE,
    line_score_features, ss_policy,
)

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
    ordered = (bet_type == "trifecta")
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


# ── U（波乱ライン連れ込み・ペーパートレード検証 2026-07-16〜）─────────────────
# 朝の wave-picks-wt が u_candidates JSON（entropy≥U_ENTROPY_MIN・ペア候補あり）を
# 出力し、ここで発走15分前のライブオッズにより最終判定する。
# 実際の賭けは行わない（記録 + Discord 通知のみ）。

def judge_u(pairs: list[dict], trio_lookup: dict) -> tuple[str, dict]:
    """U戦略の発走前ライブオッズ判定（純関数・DB非依存）。

    pairs:       朝のU候補 JSON の pairs（[{"dark","dark_model_rank","mate"}]）
    trio_lookup: _build_odds_lookup(odds_data, "trio") が返す {frozenset: odds} 辞書

    判定順:
      ① 盤面（有効オッズ 0<ov<9000 の掲載車）が7車 — 欠車発生なら見送り
      ② 盤面min三連複オッズ(mto) >= U_MTO_MIN
      ③ 穴の市場評価順位（q_i=Σ1/オッズ の降順順位）が 4〜7位
      ④ 成立ペアが複数なら「穴のモデル順位最小 → 車番最小」の1ペアに決定
      ⑤ 買い目 = {穴, 相方, t}（t=残り5車）のうちオッズ >= U_LEG_MIN_ODDS のみ

    returns (decision, detail)
      decision: "buy" / "skip" / "不明"（盤面なし→次分再試行）
      detail:   dark / mate / mto / mkt_rank / combos（"a-b-c" 昇順文字列）
                / leg_odds（全5目 {label: odds}）/ skip_reason
    """
    detail: dict = {"dark": None, "mate": None, "mto": None, "mkt_rank": None,
                    "combos": [], "leg_odds": {}, "skip_reason": None}
    if not trio_lookup:
        return "不明", detail

    # 有効オッズのみ採用（0以下=無効、9000以上=placeholder。_market_fav_frame と同基準）
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

    # ② 盤面min三連複オッズ
    mto = min(valid.values())
    detail["mto"] = round(mto, 2)
    if mto < U_MTO_MIN:
        detail["skip_reason"] = f"mto不足（{mto:.1f} < {U_MTO_MIN}）"
        return "skip", detail

    # ③ 市場評価順位: q_i = Σ(1/三連複オッズ)（その車を含む全組合せ）を降順順位
    q: dict[int, float] = {fno: 0.0 for fno in board}
    for k, fv in valid.items():
        for fno in k:
            q[fno] += 1.0 / fv
    ranked = sorted(board, key=lambda f: (-q[f], f))
    mkt_rank = {f: i + 1 for i, f in enumerate(ranked)}

    # ④ 成立ペア（穴の市場順位 4〜7位）→ モデル順位最小 → 車番最小 で1ペアに決定
    eligible: list[tuple[int, int, int]] = []
    for pr in pairs:
        try:
            d = int(pr["dark"])
            m = int(pr["mate"])
            mr = int(pr.get("dark_model_rank", 99))
        except (KeyError, TypeError, ValueError):
            continue
        if d not in board or m not in board:
            continue
        if 4 <= mkt_rank[d] <= 7:
            eligible.append((mr, d, m))
    if not eligible:
        detail["skip_reason"] = "穴の市場順位が4〜7位の成立ペアなし"
        return "skip", detail
    eligible.sort()
    _, dark, mate = eligible[0]
    detail["dark"] = dark
    detail["mate"] = mate
    detail["mkt_rank"] = mkt_rank[dark]

    # ⑤ 買い目 = {穴, 相方, t} の三連複のうちオッズ >= U_LEG_MIN_ODDS の目のみ
    leg_odds: dict[str, float | None] = {}
    combos: list[str] = []
    for t in sorted(board - {dark, mate}):
        label = "-".join(map(str, sorted((dark, mate, t))))
        ov = valid.get(frozenset({dark, mate, t}))
        leg_odds[label] = ov
        if ov is not None and ov >= U_LEG_MIN_ODDS:
            combos.append(label)
    detail["leg_odds"] = leg_odds
    detail["combos"] = combos
    if not combos:
        detail["skip_reason"] = f"{U_LEG_MIN_ODDS:.0f}倍以上の目なし"
        return "skip", detail

    return "buy", detail


def _load_u_candidates(today: str) -> list[dict]:
    """当日のU候補 JSON（昼 + 夜）を読み込む。"""
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    out: list[dict] = []
    for fname in (f"wave_picks_wt_{today}_u_candidates.json",
                  f"wave_picks_wt_{today}_night_u_candidates.json"):
        p = picks_dir / fname
        if p.exists():
            try:
                out += json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("U候補 JSON 読み込み失敗 %s: %s", p.name, e)
    return out


def _u_third_list(combos: list[str], dark: int, mate: int) -> list[int]:
    """買い目文字列（"a-b-c"）から3車目（軸2車以外）のリストを返す。"""
    thirds: list[int] = []
    for c in combos:
        try:
            rest = [int(x) for x in str(c).split("-") if int(x) not in (dark, mate)]
        except ValueError:
            continue
        if len(rest) == 1:
            thirds.append(rest[0])
    return sorted(thirds)


def _insert_u_pick(race_key: str, race_date: str, pred_combo: str, n_combos: int) -> None:
    """U（ペーパー）の記録行 {base}#7U を picks_history に即時反映する（SQLite + VPS PG）。

    実際の賭けはないが、集計・kiseki 表示互換のため bet_amount は名目値
    （n_combos × U_STAKE）で記録する。翌朝の notify_results_wt.py が
    decisions（{rk}#U）に基づき最終確定（採点）する。
    """
    store_key = race_key + "#7U"
    bet = n_combos * U_STAKE
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,0,0,0,?,'wt',False)",
                (race_date, store_key, "7PLUS_U", pred_combo, n_combos, bet),
            )
            conn.commit()
    except Exception as e:
        logger.warning("U pick SQLite 書き込み失敗 %s: %s", race_key, e)

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
                        (race_date, store_key, "7PLUS_U", pred_combo, n_combos, bet),
                    )
        except Exception as e:
            logger.warning("U pick VPS 書き込み失敗 %s: %s", race_key, e)


def _build_u_message(cand: dict, race_info: dict, detail: dict) -> str:
    """U（ペーパー）の15分前 Discord 通知メッセージ。"""
    venue = cand.get("venue_name", "?")
    race_no = race_info.get("race_no", cand.get("race_no", "?"))
    start = cand.get("start_time", "--:--")
    dark = detail.get("dark")
    mate = detail.get("mate")
    combos = detail.get("combos") or []
    leg_odds = detail.get("leg_odds") or {}
    n_pts = len(combos)
    lines = []
    for c in combos:
        ov = leg_odds.get(c)
        ov_str = f"{float(ov):.1f}倍" if ov is not None else "取得不可"
        lines.append(f"    {c}:  {ov_str}")
    ent = cand.get("entropy")
    ent_str = f"{float(ent):.2f}" if ent is not None else "—"
    mto = detail.get("mto")
    mto_str = f"{float(mto):.1f}" if mto is not None else "—"
    return (
        f"🌀 **[U・波乱検証(記録のみ)]  {venue} {race_no}R  発走 {start}**\n"
        f"  軸2車: 穴 {dark}（市場{detail.get('mkt_rank')}位）× 相方 {mate}（同ライン・逃）\n"
        f"  3連複({n_pts}点 / 名目{n_pts * U_STAKE:,}円): "
        f"`{dark}={mate} 流し（{U_LEG_MIN_ODDS:.0f}倍以上の目のみ）`\n"
        f"  **条件: entropy={ent_str}(≥{U_ENTROPY_MIN}) mto={mto_str}(≥{U_MTO_MIN})**\n"
        f"\n"
        f"  📊 現在オッズ（締切10分前・採用目のみ）:\n"
        + "\n".join(lines) + "\n"
        f"  ※ペーパートレード（賭金なし・検証記録のみ）"
    )


def _process_u_candidates(today: str, now_unix: int, notified: set[str]) -> tuple[list, set]:
    """U候補の発走前判定・記録・通知メッセージ生成。

    returns (messages, newly_done)
      messages:   [(u_key, msg)]（buy 成立分のみ）
      newly_done: 処理完了キー {race_key}#U の集合（オッズ取得失敗は含めない=再試行）
    """
    cands = _load_u_candidates(today)
    if not cands:
        return [], set()

    race_info_map = _load_race_info(
        [c["race_key"] for c in cands if "race_key" in c])

    in_window: list[tuple[dict, dict]] = []
    for cand in cands:
        rk = cand.get("race_key")
        if not rk or f"{rk}#U" in notified:
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
        u_key = f"{rk}#U"
        try:
            odds_data = scraper.fetch_odds(
                venue_id  = ri["venue_id"],
                race_date = ri["race_date"],
                race_no   = ri["race_no"],
                cup_id    = ri["cup_id"],
                day_index = ri["day_index"],
            )
        except Exception as e:
            logger.warning("fetch_odds 失敗(U) %s: %s", rk, e)
            odds_data = None
        if odds_data is None:
            # オッズ取得失敗 → notified に入れず次分の実行で再試行
            print(f"[prerace] {rk} U候補 → オッズ取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        trio_lookup = _build_odds_lookup(odds_data, "trio")
        decision, detail = judge_u(cand.get("pairs", []), trio_lookup)
        if decision == "不明":
            print(f"[prerace] {rk} U候補 → 盤面取得不可（次回再試行）", flush=True)
            time.sleep(0.3)
            continue

        # 判定を確定記録（翌朝の採点は notify_results_wt がこの内容で行う）
        _save_decision(today, u_key, {
            "decision": decision,
            "rank": "7PLUS_U",
            "paper": True,
            "stake": U_STAKE,
            "entropy": cand.get("entropy"),
            **detail,
        })

        if decision == "buy":
            combos = detail["combos"]
            thirds = _u_third_list(combos, detail["dark"], detail["mate"])
            pred = (f"{detail['dark']}-{detail['mate']}-"
                    + ",".join(map(str, thirds)))
            _insert_u_pick(rk, today, pred, len(combos))
            messages.append((u_key, _build_u_message(cand, ri, detail)))
            print(f"[prerace] {rk} U候補 → buy（ペーパー・{len(combos)}点）", flush=True)
        else:
            print(f"[prerace] {rk} U候補 → skip: {detail.get('skip_reason')}", flush=True)
        newly_done.add(u_key)
        time.sleep(0.3)
    return messages, newly_done


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

    # ── U候補（波乱ライン連れ込み・ペーパー）処理 ──
    # try/except で全体を包み、U処理の失敗が既存 SS 通知を絶対に阻害しないようにする。
    try:
        u_messages, u_done = _process_u_candidates(today, now_unix, notified)
        messages += u_messages
        newly_done |= u_done
    except Exception as e:
        logger.exception("U候補処理失敗（SS通知には影響しない）: %s", e)

    # 推奨がある場合のみ Discord 送信（ヘッダーなし・詳細メッセージのみ）
    if messages:
        for rk, msg in messages:
            send(msg)
            print(f"[prerace] {rk} → 通知送信完了", flush=True)
            time.sleep(0.5)
    elif to_notify:
        print(f"[prerace] {today} 推奨なし（オッズ確認のみ・通知スキップ）", flush=True)

    if to_notify or newly_done:
        notified |= newly_done
        _save_notified(today, notified)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
