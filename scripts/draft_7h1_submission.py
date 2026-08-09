#!/usr/bin/env python3
"""RANK_7H1 の netkeirin 投稿下書きを出力する（自動入稿の代替・確認用）。

## なぜ下書きなのか

7H1 は**どちらの脚も既存の bet_kind で表現できない**:
  - 三連単 = 1着1車 × 2着2車 × 3着5車の**フォーメーション8点**
    （既存 `BET_KIND_TRIFECTA_AXIS1` は「1着ながし・相手2車＝2点」だけ）
  - 三連複 = プール上位5車**BOX**（既存 `BET_KIND_TRIO_AXIS2` は軸2頭ながしだけ）

netkeirin のフォーメーション/BOX の方式コードと bet_id 形式が未確認のため、
**推測で実装すると誤った買い目が外部へ入稿される**。実機確認が済むまでは
本スクリプトで下書きを出し、手動で投稿する。

## 出力

タイトル・本文・買い目を、レースごとに人が読める形とJSONの両方で出す。
タイトルと本文のテンプレは `netkeirin_settings`（rank_key='7H1'）があれば
そちらを優先し、無ければ本ファイルの既定値を使う（既存ランクと同じ方式）。

使い方:
    PYTHONPATH=. .venv/bin/python scripts/draft_7h1_submission.py --date 2026-08-06
    #   --night-only  … 17時以降の発走のみ
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.bet_display import (  # noqa: E402
    fold_trifecta_formation, fold_trio_box,
)
from src.database import get_connection  # noqa: E402
# 🔴 自動入稿と**同じ関数**でタイトルを組む。ここで独自に組むと、下書きで確認した
#    ものと本番の商品が食い違う（この下書きは手動投稿の原稿になるため致命的）。
from scripts.netkeirin_submit_wt import (  # noqa: E402
    RANK_CONFIGS, _apply_template, _normalize_multi_candidate, _shape_texts,
    _stake_note_for,
)

JST = timezone(timedelta(hours=9))
RANK_KEY = "7H1"

# 既定のテンプレ。タイトルは既存ランクの `〜の二軸｜{shape}` に揃える
# （{shape} は `src/race_shape.py` が返すレース構造の見立て）。
DEFAULT_TITLE = "穴狙いの二軸｜{shape}"
DEFAULT_COMMENT = (
    "本日の穴狙いをお届けします。\n\n"
    "当方の指数で頭ひとつ抜けた1車が、それでも4着以下に沈むと読んだレースだけを"
    "選んでいます。抜けた1番手が消えれば、配当は跳ねます。\n\n"
    "その1車と、同じラインの選手は買い目から外しました。"
    "本命が飛ぶときは番手も一緒に飛ぶ傾向があるためです。\n\n"
    "買い目は三連単と三連複の併せ買い。"
    "三連単で大きな配当を狙い、三連複で的中を拾う組み立てにしています。\n\n"
    "レース直前の最終オッズをご自身でご確認のうえ、ご活用ください。"
)


def _templates() -> tuple[str, str]:
    """netkeirin_settings に 7H1 の行があればそれを使う（無ければ既定値）。"""
    try:
        with get_connection() as c:
            row = c.execute(
                "SELECT title_template, comment_template FROM netkeirin_settings "
                "WHERE rank_key = ?", (RANK_KEY,)).fetchone()
    except Exception:
        row = None
    if not row:
        return DEFAULT_TITLE, DEFAULT_COMMENT
    return (row["title_template"] or DEFAULT_TITLE,
            row["comment_template"] or DEFAULT_COMMENT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--night-only", action="store_true", help="17時以降のみ")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = REPO / "data" / "picks" / f"wave_picks_wt_{args.date}_s7h1_candidates.json"
    if not src.exists():
        print(f"候補JSONがありません: {src}\n"
              f"  先に scripts/build_7h1_candidates.py --date {args.date} を実行してください",
              file=sys.stderr)
        sys.exit(1)
    cands = json.loads(src.read_text(encoding="utf-8"))

    title_tpl, comment_tpl = _templates()
    drafts = []
    for c in cands:
        try:
            t = datetime.fromtimestamp(int(c["start_time"]),
                                       tz=timezone.utc).astimezone(JST)
        except (TypeError, ValueError, KeyError):
            t = None
        if args.night_only and (t is None or t.hour < 17):
            continue
        # 🔴 軸2車も買い目も**自動入稿と同じ関数**で組む。ここで独自に導出すると、
        #    下書きで確認した原稿と実際の商品が食い違う。
        legs, _marks, axis1, axis2, _src = _normalize_multi_candidate(
            c, RANK_CONFIGS[RANK_KEY], c["race_key"].split("#")[0])
        shape, shape_note = _shape_texts(c["race_key"], RANK_KEY, axis1, axis2)
        stake_note = _stake_note_for(RANK_KEY, legs)
        tmpl_args = dict(
            venue_name=c.get("venue_name") or "",
            race_no=int(c.get("race_no") or 0), rank_key=RANK_KEY,
            target_date=args.date, axis1=axis1, axis2=axis2, shape=shape,
            shape_note=shape_note, stake_note=stake_note,
        )
        title = _apply_template(title_tpl, **tmpl_args)
        comment = _apply_template(comment_tpl, **tmpl_args)
        drafts.append({
            "race_key": c["race_key"], "venue": c.get("venue_name"),
            "race_no": c.get("race_no"),
            "start_time": t.strftime("%H:%M") if t else None,
            "race_type": c.get("race_type"),
            "title": title, "comment": comment,
            "excluded_fav": {"frame": c["fav"], "name": c.get("fav_name")},
            "gap12_pt": round(float(c.get("gap12") or 0) * 100, 1),
            "bust_prob_pct": round(float(c.get("bust_prob") or 0) * 100, 1),
            "trifecta": {"legs": c["legs_tf"], "stake": c["stake_tf"],
                         "n": len(c["legs_tf"])},
            "trio": {"legs": c["legs_trio"], "stake": c["stake_trio"],
                     "n": len(c["legs_trio"])},
            "bet_amount": c["bet_amount"],
        })

    out = Path(args.out) if args.out else (
        REPO / "data" / "picks" / f"netkeirin_draft_7h1_{args.date}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 7H1 netkeirin 投稿下書き  {args.date}"
          f"{'（夜のみ）' if args.night_only else ''}  {len(drafts)}件 ===")
    print(f"（保存先: {out}）\n")
    for d in drafts:
        print("─" * 66)
        print(f"■ {d['venue']}{d['race_no']}R  {d['start_time']}  {d['race_type']}")
        print(f"  タイトル: {d['title']}")
        print(f"  除外した本命: {d['excluded_fav']['frame']}番 "
              f"{d['excluded_fav']['name']}"
              f"（抜け度 {d['gap12_pt']}pt / バスト確率 {d['bust_prob_pct']}%）")
        print(f"  三連単 {d['trifecta']['n']}点 × {d['trifecta']['stake']:,}円")
        # 全目の列挙は読めないのでフォーメーション表記へ畳む（畳めなければ元の列挙）。
        # JSON 側（保存ファイル）は生の legs のままにする＝入稿の正本はあくまで全目。
        print("    " + (fold_trifecta_formation(d["trifecta"]["legs"])
                        or "  ".join(d["trifecta"]["legs"])))
        print(f"  三連複 {d['trio']['n']}点 × {d['trio']['stake']:,}円")
        print("    " + (fold_trio_box(d["trio"]["legs"])
                        or "  ".join(d["trio"]["legs"])))
        print(f"  合計 {d['bet_amount']:,}円")
    if drafts:
        print("─" * 66)
        # 本文は 2026-08-09 から**レースごとに変わる**（冒頭の見解 {shape_note} と
        # 配分の説明 {stake_note} がレース依存）。全件は長いので1件目を例示する。
        print(f"\n【本文（例: {drafts[0]['venue']}{drafts[0]['race_no']}R）】")
        print(drafts[0]["comment"])


if __name__ == "__main__":
    main()
