"""netkeirin へ実際に入稿した賭け金を採点の正本として引く。

## なぜ必要か（2026-08-08 に見つかった実バグ）

入稿は**点ごとの傾斜配分**（`w=(1/朝オッズ)^0.5 × p3^0.5`）で金額を決めるのに、
採点側は**全点に同じ単価**を掛けていた:

    c7_pay = c7_trio_pay * c7_stake // 100     # c7_stake は1レース1個のスカラー

立川3R(2026-08-08) の実例: 的中した `3=5=7` の入稿額は 1,300円・三連複配当は
1,250円/100円なので払戻は **16,250円**。ところが記録は 10,000÷5点=2,000円で
計算した **25,000円** だった。

⚠️ **ずれ方に方向性がある。** 傾斜配分は高オッズの点ほど薄く張るので、
**高配当が当たったときほど過大に記録される**。`bet_detail` を持つ全13的中で
実測したところ 記録 219,330 / 正しい 186,810 ＝ **+17.4% 過大**で、
2日間のサマリー ROI は 0.750 → 0.698（約5pt甘い）だった。

さらに再構築（tail reconcile）は点ごとの配分を使うものの**入稿時ではない
オッズで配分し直す**ため、ライブ行・再構築行・実際に賭けた額の3つが併存していた。

## 方針

**`netkeirin_submissions.bet_detail` を賭け金の単一正本にする。**
実際に netkeirin へ出した金額そのものなので定義上ずれない。

⚠️ `bet_detail` は **2026-08-07 以降しか無い**（kiseki PR#77 で追加）。
それ以前と未入稿レースは復元不能なので、呼び出し側は None を受けたら
従来のスカラー単価へフォールバックする（過去の数字を作り変えない）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# bet_detail の bet_type 表記。netkeirin の商品名に合わせた日本語が入る。
TRIO = "3連複"
TRIFECTA = "3連単"

_SPLIT_RE = re.compile(r"[=\-]")


def _combo_key(bet_type: str, combo: str) -> frozenset[int] | tuple[int, ...] | None:
    """買い目文字列を照合キーへ変換する。

    三連複は順序を持たない（`frozenset`）、三連単は順序を持つ（`tuple`）。
    ここを取り違えると「当たっているのに拾えない」「別の点の金額を拾う」になる。
    """
    parts = [int(x) for x in _SPLIT_RE.split(combo) if x.isdigit()]
    if len(parts) != 3:
        return None
    if bet_type.startswith(TRIFECTA):
        return tuple(parts)
    return frozenset(parts)


def load_submitted_bet(conn: Any, race_key: str, rank_key: str) -> dict | None:
    """入稿記録（bet_detail）を読む。無ければ None。

    Args:
        race_key: サフィックス無しのレースキー（`20260808_28_03`）
        rank_key: `7C` / `7H1` など（picks_history の `#` 以降と同じ）
    """
    try:
        row = conn.execute(
            "SELECT bet_detail FROM netkeirin_submissions WHERE race_key=? AND rank_key=?",
            (race_key, rank_key),
        ).fetchone()
    except Exception as e:  # テーブルが無い環境（旧DB）でも採点自体は続ける
        logger.debug(f"netkeirin_submissions 参照失敗 {race_key}#{rank_key}: {e}")
        return None

    if not row or not row[0]:
        return None
    raw = row[0]
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as e:
        logger.warning(f"bet_detail のJSONが壊れている {race_key}#{rank_key}: {e}")
        return None


def submitted_stakes(
    conn: Any, race_key: str, rank_key: str
) -> tuple[dict[Any, int], int] | None:
    """実際に入稿した {買い目キー: 金額} と合計投資額を返す。

    Returns:
        ({frozenset|tuple: stake}, total) または None（入稿記録なし）
    """
    bet = load_submitted_bet(conn, race_key, rank_key)
    if not bet:
        return None

    stakes: dict[Any, int] = {}
    for line in bet.get("lines") or []:
        key = _combo_key(str(line.get("bet_type") or ""), str(line.get("combo") or ""))
        stake = line.get("stake")
        if key is None or not isinstance(stake, int):
            continue
        stakes[key] = stake

    if not stakes:
        return None

    total = bet.get("total")
    if not isinstance(total, int) or total <= 0:
        total = sum(stakes.values())
    return stakes, total


def resolve_payout(
    conn: Any,
    race_key: str,
    rank_key: str,
    *,
    hit: bool,
    winning_key: frozenset[int] | tuple[int, ...],
    odds_payout: int,
    fallback_stake: int,
    n_combos: int,
) -> tuple[int, int]:
    """(払戻, 投資額) を決める。採点各ランクの共通入口。

    入稿記録があればそれを使い、無ければ従来どおり単価×点数へフォールバックする。
    **不的中でも投資額は入稿記録の合計を優先する**（傾斜配分では端数の寄せ方で
    単価×点数と合計が一致しないため）。
    """
    found = submitted_stakes(conn, race_key, rank_key)
    if found is not None:
        stakes, total = found
        if not hit:
            return 0, total
        stake = stakes.get(winning_key)
        if stake is not None:
            return odds_payout * stake // 100, total
        # 入稿記録はあるが的中点が無い（欠車での組み替え等）。黙って0にしない。
        logger.warning(
            f"入稿記録に的中点が無い {race_key}#{rank_key} winning={sorted(winning_key)}"
        )

    pay = odds_payout * fallback_stake // 100 if hit else 0
    return pay, n_combos * fallback_stake


def payout_from_submitted(
    conn: Any,
    race_key: str,
    rank_key: str,
    winning_key: frozenset[int] | tuple[int, ...],
    odds_payout: int,
) -> tuple[int, int] | None:
    """入稿額で払戻と投資額を計算する。

    Args:
        winning_key: 的中した買い目（三連複=frozenset / 三連単=tuple）
        odds_payout: その券種の100円あたり確定払戻

    Returns:
        (払戻, 投資額) または None（入稿記録が無い＝呼び出し側でフォールバック）
    """
    found = submitted_stakes(conn, race_key, rank_key)
    if found is None:
        return None
    stakes, total = found

    stake = stakes.get(winning_key)
    if stake is None:
        # 入稿記録はあるが的中点が含まれない。欠車での組み替え等で起こりうるので、
        # 黙って0円にせずフォールバックさせる（取りこぼしを無言にしない）。
        logger.warning(
            f"入稿記録に的中点が無い {race_key}#{rank_key} winning={sorted(winning_key)}"
        )
        return None

    return odds_payout * stake // 100, total
