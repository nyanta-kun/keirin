#!/usr/bin/env python3
"""予想ランク（7SS/7S/7A/7B/9S/9A/7H1）をnetkeirin「ウマい車券」へ下書き自動入稿する。

2026-07-23に旧7SS/7S専用スクリプトとして新設、2026-07-28に全ランク対応へ全面再構成、
2026-08-01に旧7SS/旧9SS（gate_label='SS' 分岐・e994758で廃止済み）を削除して
現行4ランクへ整理（詳細は RANK_CONFIGS のコメント）。
朝バッチ(daily_picks_wt.sh)の候補生成直後に呼ばれる（2026-08-01の8:00一本化で
夕バッチはcronから撤去済み）。ランクごとの候補ファイル（候補生成時点で既にゲート
適用済み）から未入稿のレースのみ netkeirin へ下書き保存する。同一(race_key,
rank_key)への再送信は上書きされるだけなので、対象が重複しても無害。

各ランクのON/OFF・タイトル/コメントのテンプレートは kiseki 側の入稿設定画面
（/keirin/settings）で編集された keirin.netkeirin_settings を読む。OFFのランクは
スキップする。コメント末尾には、そのレースの出走選手ごとの1着率・3着内率テーブルを
自動付加する（数値はkiseki Web(/keirin)と同じロジット空間シフトによる正規化値）。

入稿完了後、新規に登録した件数が1件以上あれば1本のDiscordサマリーを送る。
公開は必ずユーザー本人が確認用URLから行う（本スクリプトは自動化しない）。

仕様の根拠: docs/netkeirin-input-api-spec.md

使い方:
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD morning
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD evening
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD morning --dry-run

--race-key を指定すると、そのレースのみをピンポイントで対象にする（kiseki Web
（/keirin）のレース行アイコンからの手動入稿用。ON/OFF・テンプレート・ゲート・
重複送信防止(already_submitted)は通常実行と完全に同一のルールを適用する）:
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD morning --race-key 20260728_04_07

--manual-rank-key/--axis1/--axis2 を指定すると、候補JSON検索を一切経由せず
指定した軸2車・ランクで直接入稿する（2026-07-31新設。推奨外レースをkiseki Web
のダイアログでランク選択して手動入稿するための経路）。--race-keyと併用必須。
対象ランクは7S/7A/9S/9Aのみ（S1・旧7SS・旧9SSはいずれも全廃済みのため対象外）:
    python3 scripts/netkeirin_submit_wt.py YYYY-MM-DD morning --race-key 20260728_04_07 \
        --manual-rank-key 7S --axis1 3 --axis2 5
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.netkeirin_client import (
    ACT_TYPE_DEFAULT,
    ACT_TYPE_LONGSHOT,
    BET_KIND_TRIFECTA_AXIS1,
    BET_KIND_TRIFECTA_FORMATION,
    BET_KIND_TRIO_AXIS2,
    BET_KIND_TRIO_BOX,
    BetLeg,
    NetkeirinClient,
    RACE_AUTH_URL,
    expand_bet,
)
from src.notify.discord import send
from src.strategy_wt import rank_7s_gate_label

SESSION_LABEL_JP = {"morning": "午前", "evening": "午後"}

_DEFAULT_TITLE_TEMPLATE = "{venue}{race_no}R 二軸探偵"
_DEFAULT_COMMENT_TEMPLATE = (
    "本日の二軸をお届けします。\n\n"
    "買い目は三連複・軸2車流し（5点均等）です。独自の検証では、この5点のうち"
    "最終オッズが低い（目安5〜10倍以下）組み合わせを購入対象から外すと、"
    "的中率は下がる一方で回収率は上昇する傾向を確認しています。"
    "二軸探偵の入稿は発走前の最終オッズを確認できないタイミングで行っているため、"
    "この絞り込みは行っておりません。\n\n"
    "レース直前の最終オッズをご自身でご確認いただき、低倍率の目を外すなど、"
    "回収率を意識したアレンジにもぜひご活用ください。"
)

# ランク定義。file_key は候補JSON（wave_picks_wt_{date}[_night]_{file_key}_candidates.json）の
# サフィックス。gate_filter は None なら候補全件対象、'S' なら rank_7s_gate_label() で絞り込む。
# S1は2026-07-31にdf31431でユーザー判断により全廃済み（picks_history のS1行も削除済み）。
#
# 【2026-08-01】旧7SS/旧9SS のエントリを削除した。これらは
# `{"file_key": "s7", "gate_filter": "SS"}` /`{"file_key": "s9", "gate_filter": "SS"}`＝
# 「S7(S9)の候補ファイルを読み rank_7s_gate_label()=='SS' で絞る」定義であり、
# 2026-07-31 の commit e994758 で gate_label が "S" のみを返すようになった時点から
# **どのレースにもマッチしない死んだ条件**になっていた。
# `_is_enabled()` は fail-open（keirin.netkeirin_settings に行が無いと常時ON扱い）
# のため、gate_filter の扱いを将来変えた際に誤入稿の入口になりうる点も踏まえ、
# 条件を直すのではなくエントリごと削除する（ユーザー判断）。
#
# 注意（命名衝突）: 2026-08-05 に新設された **現行7SS（内部rank `RANK_7SS`・
# entropy不合格 × 軸2車が同一ライン）は、ここで削除した旧7SS（波乱軸選出）とは
# 無関係の別ランク**で名前のみ継承している。現行7SSは 7S/7A と同じ候補プールから
# 選ばれ朝の候補JSON（file_key='s7ss'）を持つため、通常どおり本スクリプトで入稿できる。
# 🔴 **この dict の定義順がそのまま入稿の優先順位**（RANK_ORDER が list(RANK_CONFIGS)）。
#    netkeirin は1レース1商品なので、同じレースに複数ランクが該当したときは
#    先に来たランクが取り、後続はスキップする。
#    優先順位（2026-08-07 ユーザー指定）: **7H1 > 7SS > 7S > 7A > 7B > 7C**。
#    ⚠️ 2026-08-07 以前は 7H1 が dict の末尾にあり **最下位**だった。7H1 を先頭へ
#      移したのはこのとき（穴狙いを最優先で出す、というユーザー判断）。
#    9S/9A は9車立て専用なので7車ランクとは衝突しない（位置は成績に影響しない）。
RANK_CONFIGS: dict[str, dict[str, Any]] = {
    # 7H1（2026-08-06新設・穴推奨「本命バスト型」）。**唯一の2券種ランク**で、
    # 三連単フォーメーション（1着1車×2着2車×3着5車＝8点）と
    # 三連複BOX（プール上位5車＝最大10点）を **1商品にまとめて** 入稿する
    # （netkeirin の kaime は配列なので submit は1回。2回送ると上書きになる）。
    # 買い目は候補JSONの legs_tf / legs_trio を**正**として復元する（下記
    # _normalize_multi_candidate）。stake も候補JSON側（予算枠から算出済み）を使う
    # ため stake_per_line は持たない。
    "7H1": {"file_key": "s7h1", "n_cars": 7, "multi_bet": True, "gate_filter": None,
            "act_type": ACT_TYPE_LONGSHOT,   # 勝負アイコン「穴狙い」
            "default_comment": (
                "本日の穴狙いをお届けします。\n\n"
                "当方の指数で頭ひとつ抜けた1車が、それでも4着以下に沈むと読んだレースだけを"
                "選んでいます。抜けた1番手が消えれば、配当は跳ねます。\n\n"
                "その1車と、同じラインの選手は買い目から外しました。"
                "本命が飛ぶときは番手も一緒に飛ぶ傾向があるためです。\n\n"
                "買い目は三連単と三連複の併せ買い。"
                "三連単で大きな配当を狙い、三連複で的中を拾う組み立てにしています。\n\n"
                "レース直前の最終オッズをご自身でご確認のうえ、ご活用ください。"
            )},
    # 7SS（2026-08-05新設・entropy不合格 × 軸2車が同一ライン）。
    # ⚠️ 2026-08-02に全廃した旧RANK_7SS（波乱軸選出）とは無関係の別物で名前のみ継承。
    "7SS": {"file_key": "s7ss", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,    "stake_per_line": 2000, "gate_filter": None},
    "7S":  {"file_key": "s7",  "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 2000, "gate_filter": "S"},
    "7A":  {"file_key": "s7a", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 2000, "gate_filter": None},
    # 7B（2026-08-03新設）は総流しではなく相手を3点に絞る（partners_key）。
    # 1レース総額を他ランク（約10,000円）と揃えるため 3点×3,300円とする。
    # ⚠️ `_is_enabled()` は fail-open（netkeirin_settings に行が無いと常時ON）の
    #    ため、導入時に enabled=false の行を明示投入してある。ユーザーが
    #    /keirin/settings で明示的にONにするまで入稿されない。
    "7B":  {"file_key": "s7b", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 3300, "gate_filter": None,
            "partners_key": "legs_7b",
            # ⚠️ 2026-08-05 の PR#12 で 7B は「◎○一致 × **順序一致** × 準決勝」へ
            #    全面入替した。旧7Bは順序**不一致**が条件だったため、旧文面の
            #    「1番手評価が異なり」は現行条件と正反対になっていた（2026-08-06 是正）。
            #    また外部サイトの予想印を「公式予想」と呼ぶのは誤りなので言及しない。
            #    定義を変えるときは必ずこの文面と DB の comment_template も見ること。
            "default_comment": (
                "本日の二軸をお届けします。\n\n"
                "準決勝の中から、当方の指数で軸2車が明確に絞り込めたレースだけを"
                "お届けしています。相手も3点に絞りました。\n"
                "買い目は三連複・軸2車から相手3点の均等買いです。\n\n"
                "レース直前の最終オッズをご自身でご確認のうえ、ご活用ください。"
            )},
    "9S":  {"file_key": "s9",  "n_cars": 9, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 1400, "gate_filter": "S"},
    "9A":  {"file_key": "s9a", "n_cars": 9, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_per_line": 1400, "gate_filter": None},
    # 7C（2026-08-07新設・ベースモデル「終日の二軸」）。**必ず最下位に置くこと**。
    # 母集団が全7車レースで他ランクと排他ではないため、上位ランクが取った
    # レースは 7C が降りる。この衝突は**想定内**なので `overlap_expected` で
    # 失敗集計から外す（本物の失敗を埋もれさせないため）。
    # ⚠️ 軸は候補JSONの `axis1_7c`/`axis2_7c`（pred_top3 上位2車）で、
    #    `axis1`/`axis2`（3ヘッド軸）とは**別物**。取り違えると別の買い目になる。
    # ⚠️ **総流しではない**。相手は `legs_7c`（3着内率15%以上・4〜5点で可変）。
    # ⚠️ 賭け金も**可変**（1レース10,000円の予算枠 ÷ 点数）なので
    #    stake_per_line ではなく stake_budget を持つ。
    "7C":  {"file_key": "s7c", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "gate_filter": None,
            "axis_keys": ("axis1_7c", "axis2_7c"),
            "partners_key": "legs_7c",
            "stake_budget": 10000,
            "overlap_expected": True,
            # ⚠️ 文面は買い目の実態と必ず一致させること（7B で不一致を出した前例あり）。
            #    総流しではなく「3着に来そうにない車を外した4〜5点」である点、
            #    点数によって1点あたりの金額が変わる点を明記する。
            "default_comment": (
                "本日の二軸をお届けします。\n\n"
                "当方の指数で、複勝率の高い2車がはっきり抜けているレースだけを"
                "選んでいます。1日を通してお出しする、二軸探偵の土台となる予想です。\n\n"
                "買い目は三連複・軸2車からの流しですが、総流しではありません。"
                "3着以内に届きそうにない車は相手から外しています。\n"
                "また、相手が絞り込めすぎるレース（力の差が大きく、当たっても"
                "配当が期待できないレース）は、あらかじめ見送っています。\n\n"
                "レース直前の最終オッズをご自身でご確認のうえ、ご活用ください。"
            )},
}
# 入稿の処理順。**RANK_CONFIGS から導出する**（上位ランクから順に並べてあるため
# 定義順がそのまま優先順位になる）。
# ⚠️ ここを手書きのリストにしてはいけない。2026-08-05 に 7SS を新設した際、
#    RANK_CONFIGS には追加したが手書きの RANK_ORDER に入れ忘れたため、
#    **設定上は enabled=True なのに 7SS が一度も入稿されない**状態が
#    2026-08-06 朝まで続いた（メインループが RANK_ORDER を回すため）。
#    同じ「ランク一覧の二重管理」は kiseki 側でも繰り返し事故を起こしている。
RANK_ORDER = list(RANK_CONFIGS)


# ---------------------------------------------------------------------------
# 選手成績表（1着率・3着内率）
# frontend/src/app/keirin/page.tsx の sigmoid/logit/solveLogitShift と同一ロジックの
# Python移植。pred_win_pct/pred_top3_pct（選手ごと独立モデルの生確率）はレース内合計が
# 揃わないため、ロジット空間で一律シフトして単勝=100%・複勝=min(出走数,3)*100%に補正する。
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    eps = 1e-6
    c = min(max(p, eps), 1 - eps)
    return math.log(c / (1 - c))


def _solve_logit_shift(probs: list[float], target: float) -> float:
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        total = sum(_sigmoid(_logit(p) + mid) for p in probs)
        if total < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _build_entry_table(race_key: str, marks: dict[int, str]) -> str | None:
    """出走選手ごとの印・1着率・3着内率（正規化値）をHTMLテーブルとして返す。
    指数未算出（pred_win_pct が全件NULL）のレースは None を返し呼び出し側で省略する。
    netkeirinのコメント欄はscript/style/iframe以外のHTMLタグを許容するため、
    tableタグで見やすく整形する（車番昇順）。
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT frame_no, name, pred_win_pct, pred_top3_pct FROM wt_entries "
            "WHERE race_key = ? ORDER BY frame_no",
            (race_key,),
        ).fetchall()
    entries = [dict(r) for r in rows]
    if not entries or all(e["pred_win_pct"] is None for e in entries):
        return None

    win_probs = [float(e["pred_win_pct"] or 0) / 100 for e in entries]
    top3_probs = [float(e["pred_top3_pct"] or 0) / 100 for e in entries]
    win_shift = _solve_logit_shift(win_probs, 1) if any(p > 0 for p in win_probs) else None
    top3_shift = (
        _solve_logit_shift(top3_probs, min(len(entries), 3)) if any(p > 0 for p in top3_probs) else None
    )

    rows_html = []
    for e, wp, tp in zip(entries, win_probs, top3_probs):
        frame_no = int(e["frame_no"])
        mark = html.escape(marks.get(frame_no, ""))
        name = html.escape(e["name"] or "―")
        win_pct = (
            100 * _sigmoid(_logit(wp) + win_shift)
            if win_shift is not None and e["pred_win_pct"] is not None else None
        )
        top3_pct = (
            100 * _sigmoid(_logit(tp) + top3_shift)
            if top3_shift is not None and e["pred_top3_pct"] is not None else None
        )
        win_str = f"{win_pct:.1f}%" if win_pct is not None else "―"
        top3_str = f"{top3_pct:.1f}%" if top3_pct is not None else "―"
        rows_html.append(
            f"<tr><td align=\"center\">{frame_no}</td><td align=\"center\">{mark}</td>"
            f"<td align=\"center\">{name}</td><td align=\"center\">{win_str}</td>"
            f"<td align=\"center\">{top3_str}</td></tr>"
        )

    table = (
        "<table><thead><tr><th>車番</th><th>印</th><th>選手名</th><th>1着率</th>"
        "<th>3着内率</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )
    return f"【出走選手 1着率・3着内率】\n{table}"


# ---------------------------------------------------------------------------
# テンプレート
# ---------------------------------------------------------------------------

def _apply_template(
    template: str, *, venue_name: str, race_no: int, rank_key: str, target_date: str,
    axis1: int, axis2: int,
) -> str:
    """{venue}{race_no}{rank}{date}{axis1}{axis2} を置換する。
    str.format ではなく固定辞書の逐次 str.replace を使う（未定義の{...}をユーザーが
    書いても例外にせず素通しするため）。
    """
    repl = {
        "{venue}": venue_name,
        "{race_no}": str(race_no),
        "{rank}": rank_key,
        "{date}": target_date,
        "{axis1}": str(axis1),
        "{axis2}": str(axis2),
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


# ---------------------------------------------------------------------------
# 候補・設定・送信済み記録の読み書き
# ---------------------------------------------------------------------------

def _load_candidates(target_date: str, session: str, file_key: str) -> list[dict]:
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    suffix = f"_night_{file_key}_candidates.json" if session == "evening" else f"_{file_key}_candidates.json"
    path = picks_dir / f"wave_picks_wt_{target_date}{suffix}"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[netkeirin_submit] {path.name} 読み込み失敗: {e}", flush=True)
        return []


def _load_settings() -> dict[str, dict]:
    """keirin.netkeirin_settings を読む。テーブル未取得（migration未適用等）や
    行が無いランクは全ON・デフォルトテンプレート扱いにする（フェイルオープン）。
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT rank_key, enabled, title_template, comment_template FROM netkeirin_settings"
            ).fetchall()
        return {r["rank_key"]: dict(r) for r in rows}
    except Exception as e:
        print(f"[netkeirin_submit] netkeirin_settings読み込み失敗（全ON既定で継続）: {e}", flush=True)
        return {}


def _is_enabled(settings: dict[str, dict], rank_key: str) -> bool:
    row = settings.get(rank_key)
    return True if row is None else bool(row["enabled"])


def _already_submitted(race_keys: list[str]) -> set[tuple[str, str]]:
    if not race_keys:
        return set()
    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, rank_key FROM netkeirin_submissions WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    return {(r["race_key"], r["rank_key"]) for r in rows}


def _record_submission(
    race_key: str, rank_key: str, session: str, venue_name: str, race_no: int,
    gate_label: str | None, axis1: int, axis2: int, netkeirin_race_id: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO netkeirin_submissions "
            "(race_key,rank_key,session,venue_name,race_no,gate_label,axis1,axis2,netkeirin_race_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (race_key, rank_key, session, venue_name, race_no, gate_label, axis1, axis2, netkeirin_race_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# ランクごとの候補正規化（S1のキー構造は他ランクと異なるため吸収する）
# ---------------------------------------------------------------------------

def _stake_per_line(cfg: dict, n_lines: int) -> int:
    """1点あたりの賭け金を返す。

    通常ランクは cfg["stake_per_line"] の固定額。7C のように**点数が可変**な
    ランクは cfg["stake_budget"]（1レースの予算枠）を点数で割り 100円単位へ
    切り捨てる（strategy_wt.rank_7c_unit_stake と同じ式）。
    固定額のまま可変点数のランクを入稿すると、点数が少ない日ほど投資が減って
    ペーパー成績と実入稿が食い違う。
    """
    budget = cfg.get("stake_budget")
    if budget:
        if n_lines <= 0:
            raise ValueError("点数0では賭け金を決められません")
        return max(100, (int(budget) // n_lines) // 100 * 100)
    return int(cfg["stake_per_line"])


def _normalize_candidate(cand: dict, cfg: dict) -> tuple[int, int, list[int], dict[int, str]]:
    """候補dictから (axis1, axis2, partners, marks) を返す。
    axis2 は trifecta_axis1 では submit_pick に渡さないが、テンプレート変数用に
    p1（相手1）を充てて返す。
    """
    if cfg["bet_kind"] == BET_KIND_TRIFECTA_AXIS1:
        axis1, p1, p2 = int(cand["axis"]), int(cand["p1"]), int(cand["p2"])
        return axis1, p1, [p1, p2], {axis1: "◎", p1: "○", p2: "▲"}
    # 軸のキーはランクによって違う。7C は軸の選び方が他ランク（3ヘッド）と別で、
    # 候補JSONに `axis1_7c`/`axis2_7c` として入っている。`axis1`/`axis2` を
    # 読んでしまうと**別の買い目を入稿する**ので、cfg の宣言に従う。
    k1, k2 = cfg.get("axis_keys", ("axis1", "axis2"))
    if cand.get(k1) is None or cand.get(k2) is None:
        raise ValueError(f"軸キー {k1}/{k2} が候補JSONにありません")
    axis1, axis2 = int(cand[k1]), int(cand[k2])
    # 相手を絞るランク（7B: WT△を外した pred_prob 上位3車）は候補JSONが持つ
    # 絞り込み済みリストをそのまま使う。総流しランク（7S/7A/9S/9A）は従来通り
    # 軸以外の全車が相手。partners_key が無い＝総流し、が既定。
    partners_key = cfg.get("partners_key")
    if partners_key:
        partners = [int(x) for x in (cand.get(partners_key) or [])
                    if int(x) not in (axis1, axis2)]
        if not partners:   # 候補JSONが旧形式等で絞り込み結果を持たない場合は入稿しない
            raise ValueError(f"{partners_key} が空のため相手を決定できません")
    else:
        partners = [c for c in range(1, cfg["n_cars"] + 1) if c not in (axis1, axis2)]
    return axis1, axis2, partners, {axis1: "◎", axis2: "○"}


# ---------------------------------------------------------------------------
# 7H1（2券種ランク）— 候補JSONの買い目から入稿用の車番グループを復元する
#
# 🔴 **推測でフォーメーション/BOXを組み立てないこと。** 候補JSONが持つ
#    legs_tf / legs_trio（strategy_wt.rank_7h1_build_legs が生成した実際の買い目）を
#    唯一の正とし、復元したグループを expand_bet() で展開し直して
#    **元の目集合と完全一致すること**を毎回検証してから入稿する。
#    一致しなければ ValueError で落とし、そのレースは入稿しない
#    （誤った買い目を外部へ出さないため、握り潰さない）。
# ---------------------------------------------------------------------------

def _trifecta_formation_groups(legs_tf: list[str]) -> list[list[int]]:
    """['3-4-1', '3-4-2', …] から (1着列, 2着列, 3着列) を復元する。"""
    legs = set()
    for s in legs_tf:
        parts = [int(x) for x in str(s).split("-")]
        if len(parts) != 3:
            raise ValueError(f"三連単の目の形式が不正です: {s!r}")
        legs.add(tuple(parts))
    if not legs:
        raise ValueError("三連単の目が空です")
    groups = [sorted({leg[i] for leg in legs}) for i in range(3)]
    expanded = expand_bet(BET_KIND_TRIFECTA_FORMATION, groups)
    if expanded != legs:
        raise ValueError(
            f"フォーメーション復元が一致しません（元{len(legs)}点 / 復元{len(expanded)}点）: "
            f"{groups}")
    return groups


def _trio_box_group(legs_trio: list[str]) -> list[int]:
    """['1=2=4', '1=2=5', …] から BOX の車群を復元する。"""
    legs = set()
    for s in legs_trio:
        parts = [int(x) for x in str(s).split("=")]
        if len(parts) != 3:
            raise ValueError(f"三連複の目の形式が不正です: {s!r}")
        legs.add(frozenset(parts))
    if not legs:
        raise ValueError("三連複の目が空です")
    cars = sorted({c for leg in legs for c in leg})
    expanded = expand_bet(BET_KIND_TRIO_BOX, [cars])
    if expanded != legs:
        raise ValueError(
            f"BOX復元が一致しません（元{len(legs)}点 / 復元{len(expanded)}点）: {cars}")
    return cars


def _normalize_multi_candidate(
    cand: dict, cfg: dict,
) -> tuple[list[BetLeg], dict[int, str], int, int]:
    """7H1 候補から (買い目行, 印, ◎車番, ○車番) を返す。

    印（ユーザー確定・2026-08-06）:
      ◎ = 三連単の1着固定車 / ○ = 2着列の1番手 / ▲ = 2着列の2番手 /
      △ = 3着列の残り（3着だけで買っている車）/ 除外した本命は印なし(--)
    """
    tf_groups = _trifecta_formation_groups(cand.get("legs_tf") or [])
    trio_cars = _trio_box_group(cand.get("legs_trio") or [])

    stake_tf, stake_trio = int(cand["stake_tf"]), int(cand["stake_trio"])
    if stake_tf <= 0 or stake_trio <= 0:
        raise ValueError(f"賭け金が不正です（三連単{stake_tf}円 / 三連複{stake_trio}円）")

    first, second, third = tf_groups
    if len(first) != 1:
        raise ValueError(f"7H1 の1着は1車固定のはずです: {first}")
    # 2着列は候補生成側で「プール上位2車」の順序を持つが、bet_id は昇順に
    # 正規化される。印の ○/▲ は候補JSONの others（モデル3着内率の降順）の
    # 並びに従う＝表示上の序列を買い目の順序に依存させない。
    order = [int(x) for x in (cand.get("others") or [])]
    ranked_second = sorted(second, key=lambda c: order.index(c) if c in order else 99)

    marks: dict[int, str] = {first[0]: "◎"}
    if ranked_second:
        marks[ranked_second[0]] = "○"
    if len(ranked_second) > 1:
        marks[ranked_second[1]] = "▲"
    for c in third:
        marks.setdefault(c, "△")
    for c in trio_cars:          # BOXだけで買っている車も買い目に入っている
        marks.setdefault(c, "△")

    legs = [
        BetLeg(BET_KIND_TRIFECTA_FORMATION, tf_groups, stake_tf),
        BetLeg(BET_KIND_TRIO_BOX, [trio_cars], stake_trio),
    ]
    axis2 = ranked_second[0] if ranked_second else first[0]
    return legs, marks, first[0], axis2


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def _process_rank(
    rank_key: str, target_date: str, session: str, race_date, settings: dict[str, dict],
    already: set[tuple[str, str]], dry_run: bool, race_key_filter: str | None = None,
    claimed_races: set[str] | None = None,
) -> tuple[int, list[str]]:
    cfg = RANK_CONFIGS[rank_key]
    if not _is_enabled(settings, rank_key):
        return 0, []

    raw = _load_candidates(target_date, session, cfg["file_key"])
    if race_key_filter:
        raw = [c for c in raw if c.get("race_key") == race_key_filter]
    if not raw:
        return 0, []

    targets: list[tuple[dict, str | None]] = []
    for cand in raw:
        gate_label = None
        if cfg["gate_filter"] is not None:
            gate_label = rank_7s_gate_label(cand.get("wt_overlap_n"))
            if gate_label != cfg["gate_filter"]:
                continue
        targets.append((cand, gate_label))
    if not targets:
        return 0, []

    pending = [(c, g) for c, g in targets if (c["race_key"], rank_key) not in already]
    # 【2026-08-06】netkeirin は1レース1商品（同じ race_id へ action=add すると
    # 前の商品を上書きする）。ランク同士は設計上ほぼ排他だが（picks_history も
    # race_key を主キーにしており1レース1ランクを前提としている）、7H1 は
    # 「本命を買い目から外す」という**他ランクと真逆の**買い方をするため、
    # 万一同じレースが両方に該当すると先に入稿した予想が黙って消える。
    # 別ランクが既に押さえているレースはスキップし、失敗として可視化する。
    other_rank_races = {rk for rk, other in already if other != rank_key}
    if claimed_races:
        other_rank_races |= claimed_races
    conflicts = [c for c, _ in pending if c["race_key"] in other_rank_races]
    if conflicts:
        pending = [(c, g) for c, g in pending if c["race_key"] not in other_rank_races]
    if not pending and not conflicts:
        return 0, []

    setting = settings.get(rank_key)
    title_template = (setting or {}).get("title_template") or _DEFAULT_TITLE_TEMPLATE
    # ランク固有の既定コメント（cfg["default_comment"]）があればそれを既定にする。
    # 7B は買い目構造が「5点流し」ではなく「相手3点」で、共通既定文の説明が
    # 事実と食い違うため必須（設定画面で上書きされていればそちらが優先）。
    comment_template = ((setting or {}).get("comment_template")
                        or cfg.get("default_comment") or _DEFAULT_COMMENT_TEMPLATE)

    client = NetkeirinClient() if not dry_run else None
    n_submitted = 0
    failures: list[str] = []
    is_multi = bool(cfg.get("multi_bet"))

    # 衝突の扱いはランクによって意味が違う。**排他設計のランク**（7SS/7S/7A/7B/9S/
    # 9A/7H1）で衝突が起きたのは想定外なので失敗として可視化する。一方 7C のような
    # **重複前提のランク**（`overlap_expected`）では衝突は日常（実測 2.4件/日）で、
    # 失敗に混ぜるとサマリーが常時赤くなり本物の失敗が埋もれる。
    overlap_expected = bool(cfg.get("overlap_expected"))
    for cand in conflicts:
        msg = (f"{cand.get('venue_name', '?')}{cand.get('race_no', '?')}R({rank_key}): "
               f"別ランクが同じレースを入稿済みのためスキップ")
        if not overlap_expected:
            failures.append(msg)
        print(f"[netkeirin_submit] {msg}", flush=True)

    for cand, gate_label in pending:
        race_key = cand["race_key"]
        venue_name = cand.get("venue_name", "?")
        race_no = int(cand["race_no"])
        # 相手絞りランク（partners_key あり）は候補JSONが絞り込み結果を持たないと
        # 相手を決められず ValueError になる。7H1 は買い目の復元検証に失敗すると
        # 同じく ValueError になる。ここで捕まえないと RANK_ORDER の
        # ループごと落ち、**他ランクの入稿まで巻き添えで止まる**（本ループは
        # main() 側でも try されていない）。1レース分の失敗として記録し継続する。
        legs: list[BetLeg] = []
        partners: list[int] = []
        try:
            if is_multi:
                legs, marks, axis1, axis2_or_p1 = _normalize_multi_candidate(cand, cfg)
            else:
                axis1, axis2_or_p1, partners, marks = _normalize_candidate(cand, cfg)
        except (ValueError, KeyError, TypeError, IndexError) as e:
            failures.append(f"{race_key} ({rank_key}): 候補情報不正 - {e}")
            print(f"[netkeirin_submit] スキップ {venue_name}{race_no}R ({rank_key}): {e}",
                  flush=True)
            continue

        title = _apply_template(
            title_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
            target_date=target_date, axis1=axis1, axis2=axis2_or_p1,
        )
        comment = _apply_template(
            comment_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
            target_date=target_date, axis1=axis1, axis2=axis2_or_p1,
        )
        entry_table = _build_entry_table(race_key, marks)
        if entry_table:
            comment = f"{comment}\n\n{entry_table}"

        if dry_run:
            detail = (
                "\n".join(f"  {leg.bet_kind}: {leg.groups} × {leg.stake_per_line:,}円/点"
                          for leg in legs)
                if is_multi else
                f"  軸={axis1} 相手={partners} "
                f"賭け金={_stake_per_line(cfg, len(partners)):,}円/点"
            )
            print(
                f"[dry-run] {venue_name}{race_no}R ({rank_key}) 印={marks}\n"
                f"{detail}\n"
                f"  タイトル: {title}\n"
                f"  コメント:\n{comment}\n",
                flush=True,
            )
            n_submitted += 1
            continue

        try:
            assert client is not None
            if is_multi:
                ok, msg = client.submit_pick_multi(
                    race_date=race_date, venue_name=venue_name, race_no=race_no,
                    n_cars=cfg["n_cars"], legs=legs, marks=marks,
                    title=title, comment=comment,
                    act_type=cfg.get("act_type", ACT_TYPE_DEFAULT),
                )
            else:
                ok, msg = client.submit_pick(
                    race_date=race_date, venue_name=venue_name, race_no=race_no,
                    n_cars=cfg["n_cars"], bet_kind=cfg["bet_kind"],
                    axis1=axis1,
                    axis2=(axis2_or_p1 if cfg["bet_kind"] == BET_KIND_TRIO_AXIS2 else None),
                    partners=partners,
                    stake_per_line=_stake_per_line(cfg, len(partners)),
                    title=title, comment=comment,
                    # 「自信あり」は最上位の 7SS のみ（2026-08-05・ユーザー指示）。
                    # 上限に当たれば client 側が type=0 で自動リトライする。
                    confident=(rank_key == "7SS"),
                )
        except Exception as e:
            ok, msg = False, f"例外: {e}"

        if ok:
            _record_submission(
                race_key, rank_key, session, venue_name, race_no, gate_label, axis1, axis2_or_p1, msg,
            )
            if claimed_races is not None:
                claimed_races.add(race_key)
            n_submitted += 1
            print(f"[netkeirin_submit] 入稿成功 {venue_name}{race_no}R ({rank_key}) → {msg}", flush=True)
        else:
            failures.append(f"{venue_name}{race_no}R({rank_key}): {msg}")
            print(f"[netkeirin_submit] 入稿失敗 {venue_name}{race_no}R ({rank_key}): {msg}", flush=True)

    return n_submitted, failures


# ---------------------------------------------------------------------------
# 手動入稿（推奨外レース・kiseki Webのランク選択ダイアログ用）— 2026-07-31新設
# ---------------------------------------------------------------------------

# S1は全廃済み、旧7SS/旧9SSは2026-08-01に削除済み（RANK_CONFIGS のコメント参照）
# のためいずれも対象外。kiseki 側 _MANUAL_RANK_KEYS も ("7S","7A","9S","9A") で一致。
# 7H1 も対象外。手動入稿は「軸2車を選んで総流し」というUIで、7H1 の買い目
# （バスト予測モデルが決めるフォーメーション+BOX）は軸2車では表現できないため。
MANUAL_ALLOWED_RANKS = ("7S", "7A", "7B", "9S", "9A")


def _resolve_race_info(race_key: str) -> tuple[str, int, int] | None:
    """race_keyから (venue_name, race_no, n_entries) を候補JSON非依存で解決する。"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT venue_id, race_no, n_entries FROM wt_races WHERE race_key = ?",
            (race_key,),
        ).fetchone()
        if row is None:
            return None
        vrow = conn.execute(
            "SELECT name FROM venue_info WHERE venue_code = ?", (row["venue_id"],),
        ).fetchone()
        venue_name = vrow["name"] if vrow else str(row["venue_id"])
    return venue_name, int(row["race_no"]), int(row["n_entries"])


def _process_manual(
    race_key: str, rank_key: str, axis1: int, axis2: int, target_date: str, session: str,
    race_date, settings: dict[str, dict], dry_run: bool,
) -> tuple[int, list[str]]:
    """手動指定（推奨外レースへのランク選択入稿）。候補JSON検索を一切経由しない。

    ON/OFF（_global含む）・重複送信防止(already_submitted)は通常経路と同じルールを
    適用する。gate_filterはSS/S自動判定用のため参照しない（rank_key自体がユーザーの
    明示選択のため）。
    """
    if rank_key not in MANUAL_ALLOWED_RANKS:
        return 0, [f"{race_key}: 未対応ランク {rank_key}"]
    if not _is_enabled(settings, rank_key):
        return 0, []
    if (race_key, rank_key) in _already_submitted([race_key]):
        return 0, []

    cfg = RANK_CONFIGS[rank_key]
    info = _resolve_race_info(race_key)
    if info is None:
        return 0, [f"{race_key}: レース情報が見つかりません"]
    venue_name, race_no, n_entries = info
    if n_entries != cfg["n_cars"]:
        return 0, [f"{race_key}: 車数不一致（{n_entries}車 / {rank_key}は{cfg['n_cars']}車想定）"]
    if axis1 == axis2 or not (1 <= axis1 <= n_entries) or not (1 <= axis2 <= n_entries):
        return 0, [f"{race_key}: 不正な軸指定 axis1={axis1} axis2={axis2}"]

    partners = [c for c in range(1, n_entries + 1) if c not in (axis1, axis2)]
    gate_label = cfg["gate_filter"]

    setting = settings.get(rank_key)
    title_template = (setting or {}).get("title_template") or _DEFAULT_TITLE_TEMPLATE
    # ランク固有の既定コメント（cfg["default_comment"]）があればそれを既定にする。
    # 7B は買い目構造が「5点流し」ではなく「相手3点」で、共通既定文の説明が
    # 事実と食い違うため必須（設定画面で上書きされていればそちらが優先）。
    comment_template = ((setting or {}).get("comment_template")
                        or cfg.get("default_comment") or _DEFAULT_COMMENT_TEMPLATE)

    title = _apply_template(
        title_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
        target_date=target_date, axis1=axis1, axis2=axis2,
    )
    comment = _apply_template(
        comment_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
        target_date=target_date, axis1=axis1, axis2=axis2,
    )
    entry_table = _build_entry_table(race_key, {axis1: "◎", axis2: "○"})
    if entry_table:
        comment = f"{comment}\n\n{entry_table}"

    if dry_run:
        print(
            f"[dry-run][manual] {venue_name}{race_no}R ({rank_key}) "
            f"軸={axis1}-{axis2} 相手={partners} "
            f"賭け金={_stake_per_line(cfg, len(partners)):,}円/点\n"
            f"  タイトル: {title}\n"
            f"  コメント:\n{comment}\n",
            flush=True,
        )
        return 1, []

    try:
        ok, msg = NetkeirinClient().submit_pick(
            race_date=race_date, venue_name=venue_name, race_no=race_no,
            n_cars=cfg["n_cars"], bet_kind=cfg["bet_kind"],
            axis1=axis1, axis2=axis2, partners=partners,
            stake_per_line=_stake_per_line(cfg, len(partners)),
            title=title, comment=comment,
        )
    except Exception as e:
        ok, msg = False, f"例外: {e}"

    if ok:
        _record_submission(race_key, rank_key, session, venue_name, race_no, gate_label, axis1, axis2, msg)
        print(f"[netkeirin_submit][manual] 入稿成功 {venue_name}{race_no}R ({rank_key}) → {msg}", flush=True)
        return 1, []
    print(f"[netkeirin_submit][manual] 入稿失敗 {venue_name}{race_no}R ({rank_key}): {msg}", flush=True)
    return 0, [f"{venue_name}{race_no}R({rank_key}): {msg}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_date")
    parser.add_argument("session", choices=("morning", "evening"))
    parser.add_argument("--dry-run", action="store_true", help="送信せず生成内容を標準出力に出す")
    parser.add_argument(
        "--race-key", default=None,
        help="指定時はこのレース(race_key)のみをピンポイントで対象にする（それ以外は通常と同一ルール）",
    )
    parser.add_argument(
        "--manual-rank-key", default=None, choices=MANUAL_ALLOWED_RANKS,
        help="指定時は候補JSON検索を経由せず--axis1/--axis2で手動入稿する（--race-key必須）",
    )
    parser.add_argument("--axis1", type=int, default=None, help="--manual-rank-key指定時の軸1車番")
    parser.add_argument("--axis2", type=int, default=None, help="--manual-rank-key指定時の軸2車番")
    args = parser.parse_args()

    target_date, session = args.target_date, args.session
    race_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    settings = _load_settings()
    if not _is_enabled(settings, "_global"):
        print(f"[netkeirin_submit] {target_date} {session}: 全体OFF（スキップ）", flush=True)
        return

    if args.manual_rank_key:
        if not args.race_key or args.axis1 is None or args.axis2 is None:
            print("[netkeirin_submit] --manual-rank-key には --race-key/--axis1/--axis2 が必須です", flush=True)
            raise SystemExit(1)
        n, failures = _process_manual(
            args.race_key, args.manual_rank_key, args.axis1, args.axis2,
            target_date, session, race_date, settings, args.dry_run,
        )
        if args.dry_run:
            print(f"[dry-run][manual] {target_date} {session}: 完了（生成{n}件）", flush=True)
            return
        if n > 0:
            try:
                send(
                    f"📮 **[netkeirin手動入稿] {target_date}（{SESSION_LABEL_JP[session]}）: "
                    f"{args.manual_rank_key} 1件**\n確認: {RACE_AUTH_URL}\n内容を確認の上、公開してください。",
                    channel="netkeirin",
                )
            except Exception as e:
                print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)
        elif failures:
            try:
                send(f"⚠️ **[netkeirin手動入稿] {target_date}（{SESSION_LABEL_JP[session]}）: 失敗**\n"
                     + " / ".join(failures), channel="netkeirin")
            except Exception as e:
                print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)
        print(f"[netkeirin_submit][manual] {target_date} {session}: 完了（成功{n}件・失敗{len(failures)}件）",
              flush=True)
        return

    all_race_keys: set[str] = set()
    per_rank_raw: dict[str, list[dict]] = {}
    for rank_key in RANK_ORDER:
        if not _is_enabled(settings, rank_key):
            continue
        cfg = RANK_CONFIGS[rank_key]
        raw = _load_candidates(target_date, session, cfg["file_key"])
        if args.race_key:
            raw = [c for c in raw if c.get("race_key") == args.race_key]
        per_rank_raw[rank_key] = raw
        all_race_keys.update(c["race_key"] for c in raw)

    already = _already_submitted(sorted(all_race_keys))

    submitted_counts: dict[str, int] = {r: 0 for r in RANK_ORDER}
    all_failures: list[str] = []
    # 同一実行内で入稿済みのレース。netkeirin は1レース1商品なので、後続ランクが
    # 同じレースへ入稿すると先の商品を上書きしてしまう（_process_rank 参照）。
    claimed_races: set[str] = set()
    for rank_key in RANK_ORDER:
        if rank_key not in per_rank_raw:
            continue
        n, failures = _process_rank(
            rank_key, target_date, session, race_date, settings, already, args.dry_run,
            race_key_filter=args.race_key, claimed_races=claimed_races,
        )
        submitted_counts[rank_key] = n
        all_failures.extend(failures)

    total = sum(submitted_counts.values())
    if args.dry_run:
        print(f"[dry-run] {target_date} {session}: 完了（生成{total}件）", flush=True)
        return

    session_jp = SESSION_LABEL_JP[session]
    if total > 0:
        breakdown = "・".join(f"{k}{v}件" for k, v in submitted_counts.items() if v > 0)
        msg = (
            f"📮 **[netkeirin入稿完了] {target_date}（{session_jp}）: "
            f"{breakdown}（計{total}件）**\n"
            f"確認: {RACE_AUTH_URL}\n"
            f"内容を確認の上、公開してください。"
        )
        if all_failures:
            msg += f"\n⚠️ 入稿失敗 {len(all_failures)}件: " + " / ".join(all_failures)
        try:
            send(msg, channel="netkeirin")
        except Exception as e:
            print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)
    elif all_failures:
        try:
            send(
                f"⚠️ **[netkeirin入稿] {target_date}（{session_jp}）: 全{len(all_failures)}件が入稿失敗**\n"
                + " / ".join(all_failures),
                channel="netkeirin",
            )
        except Exception as e:
            print(f"[netkeirin_submit] Discord通知失敗: {e}", flush=True)
    else:
        print(f"[netkeirin_submit] {target_date} {session}: 対象なし（スキップ）", flush=True)

    print(
        f"[netkeirin_submit] {target_date} {session}: 完了（成功{total}件・失敗{len(all_failures)}件）",
        flush=True,
    )


if __name__ == "__main__":
    main()
