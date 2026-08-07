"""過去分の再構築で「入稿と同じ傾斜配分」を再現する（2026-08-07 新設）。

## なぜ要るのか

2026-08-07 に netkeirin 入稿を均等割りから**想定着地オッズに応じた傾斜配分**へ
変えた（`src/stake_allocation.py`）。picks_history の過去行は均等割りのままなので、
**新旧が混ざった記録**になっている（[[feedback_full_period_migration]]）。

## 🔴 最終オッズで配分してはいけない

再構築時には最終オッズが手元にあるので、それで配分すると数字は良くなる。
実測（7車 28,415R・2024-07〜2026-08）:

| 配分の基準 | 実質的中率 | 性質 |
|---|---|---|
| 均等（旧記録） | 25.86% | — |
| **モデル p3 のみ** | **33.10%** | 全期間で入手可能・先読みなし |
| 朝オッズ×モデル | 33.16% | 本番と同じ規則 |
| 最終オッズ | **47.67%** | **先読み。実運用では到達不能** |

最終オッズ版は本番より **14.5pt** 高く出る。これはモデルの実力ではなく
「発走時点のオッズを知っていたこと」による。記録に残すと後日
「以前は48%出ていた」と誤読する。**本番と同じ規則だけを使う。**

## 使う規則（本番 `stake_allocation.landing_weights` と同一）

    w = (1/朝オッズ)^0.5 × p3^0.5   … 買う点すべてに朝オッズがある場合
    w = p3                          … それ以外（本番の欠損時フォールバック）

朝オッズは 2026-06-08 以降しか無く、過去全体では 2.4% のレースにしか無い。
実測でも blend − model は **+0.01pt [−0.02, +0.05]** で区別できないが、
**本番と同じ規則を使うことで再構築行とライブ行の継ぎ目を作らない**ことに意味がある
（再構築はライブ行を上書きするので、規則が違うと上書きのたびに数字が動く）。
"""
from __future__ import annotations

import re

from src.database import get_connection
from src.stake_allocation import BUDGET_DEFAULT, tilted_stakes

_SEP_RE = re.compile(r"[-=]")


def load_morning_boards(race_keys: list[str]) -> dict[str, dict[frozenset[int], float]]:
    """race_key → {3車の組: 朝の三連複オッズ}。無い期間は空になる。

    ⚠️ **区切りは表で違う**（`wt_odds` は '1=2=3' / `wt_odds_snapshot` は '1-2-3'）。
    ⚠️ 9999.9 は winticket の「オッズ未確定」センチネルなので採らない。
    """
    out: dict[str, dict[frozenset[int], float]] = {}
    keys = sorted(set(race_keys))
    if not keys:
        return out
    with get_connection() as conn:
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT race_key, combination, odds_value FROM wt_odds_snapshot "
                f"WHERE snapshot_type = 'morning' AND bet_type = 'trio' "
                f"AND race_key IN ({ph})",
                chunk,
            ).fetchall()
            for r in rows:
                od = r["odds_value"]
                if od is None or not (0 < float(od) < 9000):
                    continue
                try:
                    key = frozenset(int(x) for x in _SEP_RE.split(str(r["combination"])))
                except ValueError:
                    continue
                if len(key) == 3:
                    out.setdefault(r["race_key"], {})[key] = float(od)
    return out


def stakes_for_combos(
    axis1: int,
    axis2: int,
    combos: list[frozenset[int]],
    top3_probs: dict[int, float],
    board: dict[frozenset[int], float] | None = None,
    budget: int = BUDGET_DEFAULT,
) -> dict[frozenset[int], int]:
    """買う目 → 賭け金。入稿時と同じ配分を再現する。

    combos は三連複・軸2車ながしの目（{axis1, axis2, 3列目}）。
    top3_probs は **その時点の vintage 予測**（0-1 スケール）。
    board は朝オッズ盤面（無ければ p3 単独へ落ちる＝本番と同じ）。
    """
    # 🔴 **p3 が無ければ落とす。黙って均等へ落としてはいけない。**
    #    本番の `landing_weights` は p3 欠損時に均等へフォールバックするが、
    #    それは「入稿時に予測が読めなかった」ための救済。再構築では p3 は
    #    必ず手元にあるので、**空なら呼び出し側のバグ**である。
    #    2026-08-07: 実際に7ランク中5つで候補dictへの `top3_probs` 登録が漏れ、
    #    11時間の再構築が丸ごと均等配分で走った（実質的中率が +0.22pt しか
    #    動かず気づいた）。無言のフォールバックは検知できない。
    if not top3_probs:
        raise ValueError(
            "top3_probs が空です。候補dictへ 'top3_probs' を載せ忘れていませんか"
            "（再構築では p3 は必ず取れるはずで、空なら呼び出し側のバグ）")
    thirds = [next(iter(c - {axis1, axis2})) for c in combos]
    missing = [t for t in thirds if not top3_probs.get(t)]
    if missing:
        raise ValueError(f"top3_probs に買う相手の値がありません: {missing}")
    morning = None
    if board:
        got = {t: board.get(frozenset({axis1, axis2, t})) for t in thirds}
        if all(v for v in got.values()):
            morning = got
    stakes, _ = tilted_stakes(thirds, morning, top3_probs, budget=budget)
    return {frozenset({axis1, axis2, t}): stakes[t] for t in thirds}
