#!/usr/bin/env python3
"""朝8:00バッチで「情報不足のため候補にできなかった」レースを列挙する（2026-08-04・U-1）。

## 背景

8:00 単一バッチへ一本化した副作用で、**高知・武雄のミッドナイト開催が毎日まるごと
推奨から消えていた**。原因は開催情報の公開が遅いこと（実測: WINTICKET公式の印は
13:16 まで、ライン予想は 11:35 まで出揃わない）。夜開催だからではなく**場ごとの事情**で、
京王閣は同じ夜発走でも朝5時には公開済み。

判定に必要なのは公式の◎◯（`wt_entries.prediction_mark`）で、これが無いと
`rank_7s_wt_overlap_n()` が None を返し、7S/7A/7B のどのゲートにも載らない。
データ収集自体は `intraday_results_wt.sh` が15分毎に回しているので、
**足りないのは「もう一度選び直す処理」だけ**だった。

## 何を「不足分」とみなすか

朝バッチが書く生候補 `data/picks/wave_picks_wt_{date}_s7_raw_candidates.json` の
**`wt_overlap_n` が null の行**＝「朝の時点で◎◯が未公開だったレース」。

これを使うのが重要で、「候補に無いレース」を対象にしてはいけない。朝に情報が
揃っていてゲートで正しく落ちたレースまで16:00に作り直すと、**同じレースを
条件を変えて何度も引き直す**ことになり、朝の判定を事後変更しない運用
（prerace decisions の方針）と矛盾する。

⚠️ 生候補ファイルは7車立てのみ（9車は raw を書き出さない実装）。したがって本
スクリプトが拾えるのは 7S/7A/7B の母集団だけで、9S/9A の取りこぼしは対象外。
ミッドナイト開催は7車立てのため実用上の穴は無いが、9車で同じ問題が起きた場合は
別途 raw の書き出しが必要になる。

## 出力

1行1 race_key のテキスト。`wave-picks-wt --only-races-file` にそのまま渡す。
**既に発走した（または発走が近い）レースは除外**する（--min-lead-min 既定15分＝
発走15分前の prerace 判定に間に合わないものを追加しても意味がないため）。

該当0件なら空ファイルを書き exit 1 を返す（呼び出し側のシェルが後続処理を
スキップできるようにするため。エラーではない）。

使い方:
    python scripts/list_deferred_races_wt.py 2026-08-04 --out data/picks/deferred_2026-08-04.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection

PICKS_DIR = REPO / "data" / "picks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target_date", help="YYYY-MM-DD")
    ap.add_argument("--out", default=None, help="出力先（既定: data/picks/deferred_{date}.txt）")
    ap.add_argument("--min-lead-min", type=int, default=15,
                    help="発走までこの分数を切ったレースは除外する（既定15分）")
    args = ap.parse_args()

    raw_path = PICKS_DIR / f"wave_picks_wt_{args.target_date}_s7_raw_candidates.json"
    out_path = Path(args.out) if args.out else PICKS_DIR / f"deferred_{args.target_date}.txt"

    if not raw_path.exists():
        print(f"[deferred] 朝の生候補ファイルが見つかりません: {raw_path}")
        print("[deferred] 8:00バッチが未実行か失敗しています。対象0件として終了します。")
        out_path.write_text("", encoding="utf-8")
        sys.exit(1)

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    deferred = sorted({c["race_key"] for c in raw if c.get("wt_overlap_n") is None})
    print(f"[deferred] 朝の生候補 {len(raw)}件中、◎◯未公開だったのは {len(deferred)}レース")

    if not deferred:
        out_path.write_text("", encoding="utf-8")
        print("[deferred] 不足分なし。")
        sys.exit(1)

    # 発走済み・発走直前を除外する
    now = int(time.time())
    cutoff = now + args.min_lead_min * 60
    with get_connection() as conn:
        starts = dict(conn.execute(
            "SELECT race_key, start_at FROM wt_races WHERE race_date = ?",
            (args.target_date,),
        ))
    upcoming, passed = [], []
    for rk in deferred:
        s = starts.get(rk)
        (upcoming if s is not None and int(s) > cutoff else passed).append(rk)

    if passed:
        print(f"[deferred] 発走済み/直前のため除外: {len(passed)}レース")
    print(f"[deferred] 再算出の対象: {len(upcoming)}レース")
    for rk in upcoming:
        print(f"    {rk}")

    out_path.write_text("\n".join(upcoming) + ("\n" if upcoming else ""), encoding="utf-8")
    print(f"[deferred] 出力: {out_path}")
    if not upcoming:
        sys.exit(1)


if __name__ == "__main__":
    main()
