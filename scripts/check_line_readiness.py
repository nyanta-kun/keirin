#!/usr/bin/env python3
"""winticket ライン予想（linePrediction）の収集充足度チェック（2026-08-01 新設）。

背景:
    ナイター(夜レース)はライン情報の公開が遅く、従来は
    daily_picks_wt.sh(朝・日中レースのみ推奨)→evening_picks_wt.sh(夕方・夜レース
    のみ推奨)の2段階構成で「公開が遅い夜レースは夕方に取りに行く」時間差を
    確保していた。ユーザー要望により朝を7:00・夕方(実質)を8:00へ前倒しする際、
    「ライン情報が想定より遅れて公開された場合」に備えたリトライを追加する。

判定方法:
    wt_entries.n_lines は winticket の linePrediction が未公開のときレース単位で
    0 になる（ライン未公開時は line_group=0/line_size=1/line_pos=1/n_lines=0が
    レコード生成時に既定値として書き込まれる。src/scraper/winticket.py
    _parse_lineup() 参照。NULLにはならない）。よって「n_lines=0の行が
    そのレースの全行（=収集済みentryが1件もない場合を含む）」を「ライン未公開」
    と判定する。

    対象レースは --start-from-hour / --start-to-hour で絞り込む。
    判定ロジックは src/cli/main.py の _hour_skip() と同じ意味論
    （hour不明は「to側のみ判定」＝朝の対象に含める）を反転させたもの。
    hour は wt_races.start_at（winticket仕様のJST unix秒文字列）から
    src/cli/main.py の _fmt_start() と同じ変換式で算出する
    （解釈がズレると本番の朝/夜レース切り分けと不整合になるため厳密に合わせる）。

閾値の根拠（2026-08-01 実測・PM調査に基づく判断）:
    - wt_entries のライン列は最新状態で UPSERT される実装のため、過去の
      「収集時点で何%が未公開だったか」を遡って検証することはできない
      （収集のたびに上書きされ、収集直後の状態は保存されない）。
    - 直近4日分（2026-07-28〜07-31・314レース）の「1日の最後に見た状態」は
      いずれもライン欠損0件（100%充足）だった。ただしこれは複数回の再収集後
      の最終状態であり、7:00/8:00 時点の状態を保証するものではない。
    - 上記の制約から、閾値は実測ベースの精密な値ではなく「なだらかな安全網」
      として保守的に設定した: RATIO_THRESHOLD=0.3 は、数レース程度が
      たまたま公開が遅れているだけの通常のばらつきでは発火せず、対象レースの
      概ね3割以上が未公開という broad な障害（WINTICKET側の広範な遅延・
      収集コードの不具合等）でのみ発火することを狙っている。
    - 運用開始後の実測（check_line_readinessの出力ログ）を見ながら
      閾値を再調整すべき。0%（1レースでも欠損したら不足）という厳格な閾値は
      ユーザー要望にもある通り誤検知（無駄なリトライ・遅延）を生みやすいため
      採用しない。

使い方:
    .venv/bin/python3 scripts/check_line_readiness.py --date 2026-08-01 --start-to-hour 19
    .venv/bin/python3 scripts/check_line_readiness.py --date 2026-08-01 --start-from-hour 19

終了コード: 0=充足(またはスキップ) / 1=不足
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

JST = timezone(timedelta(hours=9))

RATIO_THRESHOLD = 0.3   # 対象レースのうちこの割合を超えてライン未公開なら不足と判定
MIN_TARGET_RACES = 3    # 対象レースがこれ未満なら判定不能としてOK扱い（非開催日等）


def _hour_of(start_at: object) -> int | None:
    """winticket start_at（JST unix秒文字列）からJST発走時(hour)を返す。不明はNone。

    src/cli/main.py の _fmt_start()/_hour_of() と同じ変換ロジック
    （解釈が食い違うと本番の朝/夜レース切り分けとズレるため厳密に合わせる）。
    """
    if start_at is None:
        return None
    try:
        ts = int(start_at)
    except (ValueError, TypeError):
        return None
    try:
        return datetime.fromtimestamp(ts, tz=JST).hour
    except (OSError, OverflowError, ValueError):
        return None


def _in_target_window(
    hh: int | None, start_from_hour: int | None, start_to_hour: int | None
) -> bool:
    """src/cli/main.py の _hour_skip() を反転した対象判定（同じ意味論を再現）。

    hour不明のレースは「to側のみ判定」対象に含める（=start_from_hour指定時は除外、
    start_to_hour指定時は含める）。_hour_skip() のコメント参照。
    """
    if start_to_hour is not None and hh is not None and hh >= start_to_hour:
        return False
    if start_from_hour is not None and (hh is None or hh < start_from_hour):
        return False
    return True


def check(
    target_date: str,
    start_from_hour: int | None = None,
    start_to_hour: int | None = None,
) -> tuple[bool, str]:
    """対象時刻帯のライン情報充足度を判定する。returns (is_ok, message)。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT r.race_key AS race_key, r.start_at AS start_at, "
            "MAX(e.n_lines) AS n_lines "
            "FROM wt_races r LEFT JOIN wt_entries e ON r.race_key = e.race_key "
            "WHERE r.race_date = ? "
            "GROUP BY r.race_key, r.start_at",
            (target_date,),
        ).fetchall()

    target = [
        r for r in rows
        if _in_target_window(_hour_of(r["start_at"]), start_from_hour, start_to_hour)
    ]
    total = len(target)
    window_desc = f"(from={start_from_hour}, to={start_to_hour})"

    if total < MIN_TARGET_RACES:
        return True, (
            f"{target_date}: 対象レース数不足(n={total})のため判定スキップ {window_desc}"
        )

    no_line = sum(1 for r in target if (r["n_lines"] or 0) == 0)
    ratio = no_line / total

    if ratio > RATIO_THRESHOLD:
        return False, (
            f"{target_date}: ライン情報不足 — 対象{total}レース中{no_line}レースで"
            f"ライン未公開({ratio * 100:.0f}%、閾値{RATIO_THRESHOLD * 100:.0f}%超) "
            f"{window_desc}"
        )
    return True, (
        f"{target_date}: ライン情報充足 — 対象{total}レース中{no_line}レースで未公開"
        f"({ratio * 100:.0f}%) {window_desc}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="対象日 (YYYY-MM-DD)")
    ap.add_argument("--start-from-hour", type=int, default=None,
                     help="この時刻(JST・時)以降のレースのみ対象（夜の部用）")
    ap.add_argument("--start-to-hour", type=int, default=None,
                     help="この時刻(JST・時)未満のレースのみ対象（朝の部用）")
    args = ap.parse_args()

    is_ok, message = check(args.date, args.start_from_hour, args.start_to_hour)
    print(f"[check_line_readiness] {message}")
    sys.exit(0 if is_ok else 1)


if __name__ == "__main__":
    main()
