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
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.netkeirin_client import (
    ACT_TYPE_CONFIDENT,
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
from src.meeting_wave import (
    WAVE_MORNING,
    WAVE_NIGHT,
    WAVE_NOON,
    WAVE_LABEL_JP,
    wave_of_first_hour,
    waves_due_by,
)
from src.dutch_allocation import dutch_allocate
from src.race_shape import (
    classify_shape,
    logit,
    shape_note_text,
    shape_title_text,
    sigmoid,
    solve_logit_shift,
    stake_note_text,
)
from src.stake_allocation import group_by_stake, tilted_stakes
from src.strategy_wt import (
    RACE_BUDGET,
    RANK_7H1_TF_UNIT,
    rank_9h1_stakes,
    rank_7h1_trio_stakes,
    rank_7s_gate_label,
    unit_stake,
)

# 三連複の組み合わせ表記の区切り。**表で違う**（wt_odds は '1=2=3' /
# wt_odds_snapshot は '1-2-3'）ので両方受ける。
_SEP_RE = re.compile(r"[-=]")

SESSION_LABEL_JP = {"morning": "午前", "noon": "昼", "evening": "午後"}

# session → その回で入稿する開催の波（`src/meeting_wave.py`）。
# 🔴 **1つの開催は必ず1つの波でしか入稿されない**。netkeirin は公開後に
#    差し替えられないので、二重に出すと先の商品が消える。
SESSION_WAVE = {
    "morning": WAVE_MORNING,   # モーニング・デイ（第1R < 12時）
    "noon": WAVE_NOON,         # ナイター（第1R 12〜17時台）
    "evening": WAVE_NIGHT,     # ミッドナイト（第1R 18時〜）
}

_DEFAULT_TITLE_TEMPLATE = "{venue}{race_no}R 二軸探偵"
_DEFAULT_COMMENT_TEMPLATE = (
    "本日の二軸をお届けします。\n\n"
    # ⚠️ 7S/7A/7SS(5点) と 7C(4〜5点・可変) が共有するので**点数を書かない**。
    #    2026-08-07 以前は「（5点均等）」「この5点のうち」と書いており、
    #    7C が同じ文面を使うと買い目を偽ることになるため一般化した。
    # ⚠️ 2026-08-07 の傾斜配分導入で「均等買い」も嘘になったため方式の説明を差し替えた。
    #    **配分方式を変えるときは必ずこの文面と DB の comment_template も見ること。**
    "買い目は三連複・軸2車流しです。金額は均等ではなく、当方が想定する発走時オッズに"
    "応じて配分しています。配当が低くなりやすい買い目に厚く、高くなりやすい買い目に"
    "薄く置き、どの目で決まっても払戻が投資を上回ることを狙う組み立てです。\n\n"
    "【ご購入にあたって】\n"
    "この配分はあくまで想定オッズに基づくものです。"
    "レース直前の実際のオッズをご自身でご確認いただき、配分を調整いただくと"
    "精度が上がります。"
)

# --- 看板レース（決勝・特選クラス）専用の文面（2026-08-09 新設・`--marquee`）---
# 🔴 **通常ランクの文面をそのまま使ってはいけない。** 7A/9A の既定文面は
#    「本命が割れ、相手次第で配当が伸びるレースだけを絞ってお届けしています。
#      毎日は出ません。」と書いてある。看板レースは
#    (1) **必ず出す**方針なので「毎日は出ません」が嘘になり、
#    (2) 断然人気がいる決勝（例 2026-08-09 和歌山12R は軸1の3着内率 95.6%）でも
#        出すため「本命が割れ」が事実と逆になる。
#    7B で旧文面が現行条件と正反対だった事故（2026-08-06 是正）と同じ型なので、
#    **看板レース用は独立した文面を持たせる**。
#
# ⚠️ この文面では**レースの拮抗度について何も断定しない**。看板レースには
#    Q1_loose（拮抗）も Q4_chalk（断然）も混ざるため、どちらかに寄せた瞬間に
#    半分のレースで嘘になる。レース個別の見立ては `{shape_note}` が担う。
# ⚠️ 2026-08-09 に【この買い目について】【このレースについて】を削除した結果、
#    **本文は通常ランクの既定文と同一**になった。
#    定数を残しているのは、設定画面のランク別テンプレート編集に引きずられず
#    看板レースの文面を固定できるようにするため（PR#60 の設計判断）。
#
# タイトルは **固定文字列「本日の二軸」**（2026-08-09 ユーザー指示）。
# ⚠️ 通常ランクのような `{shape}`（レース形の一言）や `{race_type}`（決勝/特選）は
#    **入れない**。当初は「{race_type}の二軸｜{shape}」で種別を出していたが、
#    看板レースは商品名を揃える方針に変更した。
#    `{race_type}` の置換自体は `_apply_template` に残してある（設定画面の
#    独自テンプレートで使えるようにするため）。
_MARQUEE_TITLE_TEMPLATE = "本日の二軸"
_MARQUEE_COMMENT_TEMPLATE = (
    "{shape_note}\n\n"
    "【二軸】\n"
    "本レースで照らし出した二軸は、◎{axis1}番・○{axis2}番です。\n\n"
    "【ご購入にあたって】\n"
    "レース直前の実際のオッズをご自身でご確認いただき、必要に応じて配分を"
    "調整いただくと精度が上がります。\n\n"
    "【参考データ】\n"
    "出走選手全員の1着率・3着内率です。三連単で購入される際の着順・買い目の"
    "参考にご活用ください。"
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
#    優先順位（2026-08-07 ユーザー指定）: **7H1 > 7SS > 7S > 7A > 7C > 7B**。
#    ⚠️ 同日中に 7B を 7C の**後ろ**へ動かした。7C との重複では 7C が実質的中率で
#      上回る（39.0% vs 31.6%）一方、7B は 7C が拾わないレースを 3.14件/日 持つ。
#      「重複は 7C・独自は 7B」を優先順位だけで実現している。
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
    # 9H1（2026-08-08新設・穴推奨「9車・高配当狙い」）。三連単フォーメーション
    # （1着1車 × 2着2車 × 3着4車 ＝ 6点）の**単一券種**なので、7H1 の multi_bet
    # ではなく `formation_bet` を使う。
    # 🔴 **7H1 の直後に置いてある**＝同じ9車レースで 9S/9A と重なったとき 9H1 が取る。
    #    9H1 は約1件/日と薄いので 9S/9A(3.96件/日) が失う分は小さいが、これは
    #    「穴推奨を優先する」という商品判断なので、入れ替えたければ定義順を変える。
    "9H1": {"file_key": "s9h1", "n_cars": 9, "formation_bet": True, "gate_filter": None,
            "act_type": ACT_TYPE_LONGSHOT,   # 勝負アイコン「穴狙い」
            "default_comment": (
                "9車立ての高配当狙いをお届けします。\n\n"
                "9車立ては7車立てに比べて決着が大きく荒れやすく、"
                "三連単で500倍を超える決着の出やすさは約2.6倍あります。"
                "その中から、出走表の構成だけを見て特に荒れやすいと判断したレースを絞りました。\n\n"
                "買い目は三連単のフォーメーション6点。"
                "1着はあえて上位評価ではない1車に固定し、配当が伸びる形に寄せています。\n\n"
                "外れる日が続く買い方です。当たったときの大きさを狙う券種としてご活用ください。"
                "レース直前の最終オッズをご自身でご確認ください。"
            )},
    # 7SS（2026-08-05新設・entropy不合格 × 軸2車が同一ライン）。
    # ⚠️ 2026-08-02に全廃した旧RANK_7SS（波乱軸選出）とは無関係の別物で名前のみ継承。
    # `tilt_stakes` — 均等割りをやめ、想定着地オッズに応じて配分する（2026-08-07）。
    # netkeirin の的中率は**ガミを不的中として数える**ため、5点均等では
    # 5.0倍未満の的中が全部「不的中」表示になる（実測: 的中の51.8%がガミ）。
    # 詳細と実測値は src/stake_allocation.py のモジュール docstring。
    # ⚠️ 当初は 7B を対象外にしていた（3点買いで境界が3.0倍＝ガミが少ない）。
    #    ユーザー指摘「7Bも的中してガミだと的中扱いにならない」を受けて測り直し、
    #    **実質的中 +1.05pt [+0.66, +1.43] P=100%**（ROI −1.34pt は有意でない）
    #    だったので全ランク一律で対象にした。7B の伸びしろは**的中率30.5%が上限**
    #    なので構造的に +2.3pt しかなく、そのうち +1.05pt を回収する形になる。
    "7SS": {"file_key": "s7ss", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,    "stake_budget": RACE_BUDGET, "gate_filter": None, "tilt_stakes": True},
    "7S":  {"file_key": "s7",  "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_budget": RACE_BUDGET, "gate_filter": "S", "tilt_stakes": True},
    "7A":  {"file_key": "s7a", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_budget": RACE_BUDGET, "gate_filter": None, "tilt_stakes": True},
    # 7B（2026-08-03新設）は総流しではなく相手を3点に絞る（partners_key）。
    # 1レース総額を他ランク（約10,000円）と揃えるため 3点×3,300円とする。
    # ⚠️ `_is_enabled()` は fail-open（netkeirin_settings に行が無いと常時ON）の
    #    ため、導入時に enabled=false の行を明示投入してある。ユーザーが
    #    /keirin/settings で明示的にONにするまで入稿されない。
    # 9車は相手7点＝ガミ境界が7.0倍で7車より条件が悪いので傾斜配分の対象。
    # ⚠️ 実測の主対象は7車（1,061R）で、9車は件数が薄く単独では検証していない。
    #    仕組みは券種・点数に依らず同じなので同じ扱いにしてある。
    "9S":  {"file_key": "s9",  "n_cars": 9, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_budget": RACE_BUDGET, "gate_filter": "S", "tilt_stakes": True},
    "9A":  {"file_key": "s9a", "n_cars": 9, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_budget": RACE_BUDGET, "gate_filter": None, "tilt_stakes": True},
    # 7C（2026-08-07新設・ベースモデル「終日の二軸」）。**必ず最下位に置くこと**。
    # 母集団が全7車レースで他ランクと排他ではないため、上位ランクが取った
    # レースは 7C が降りる。この衝突は**想定内**なので `overlap_expected` で
    # 失敗集計から外す（本物の失敗を埋もれさせないため）。
    # ⚠️ 軸は候補JSONの `axis1_7c`/`axis2_7c`（pred_top3 上位2車）で、
    #    `axis1`/`axis2`（3ヘッド軸）とは**別物**。取り違えると別の買い目になる。
    # ⚠️ **総流しではない**。相手は `legs_7c`（3着内率15%以上・4〜5点で可変）。
    # ⚠️ 賭け金も**可変**（1レース10,000円の予算枠 ÷ 点数）なので
    #    stake_per_line ではなく stake_budget を持つ。
    "7C":  {"file_key": "s7c", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_budget": RACE_BUDGET, "gate_filter": None,
            "axis_keys": ("axis1_7c", "axis2_7c"),
            # 🔴 **買う相手**は `legs_7c_buy`（三連単=相手全部 / 三連複=上位2点）。
            #    `legs_7c` は選別用の全リスト（4〜5点）で、そのまま買うと
            #    2026-08-09 の絞り込みが効かない。取り違えると別の買い目になる。
            "partners_key": "legs_7c_buy",
            "overlap_expected": True,
            "tilt_stakes": True,
            # 単勝率で三連単へ切り替える（2026-08-09・`RANK_7C_TRIFECTA_PW_MIN`）。
            # 候補JSONの真偽値だけを読む。**点数は三連複と同じ**（1着=軸1 /
            # 2着=軸2 / 3着=相手流し）で、増やすと効果が消える。
            "trifecta_switch_key": "trifecta_7c",
            # 🔴 **専用文面は持たない。** 当初は「既定文が『買い目は三連複・軸2車
            #    流しです』と書いてあり、切替レース（実測 16.9%）で買っていない
            #    券種を説明することになる」ため専用文面を用意していたが、
            #    2026-08-09 に【この買い目について】を全ランクから削除した結果、
            #    既定文は**券種に一切言及しなくなった**ので不要になった。
            #    ⚠️ したがって本ランクは **DBテンプレートが更新済みであることが前提**。
            #       `scripts/update_netkeirin_templates.py --apply` を流す前に
            #       切替を有効化すると、三連単なのに「三連複・軸2車流し」と
            #       説明する商品が出る。
            # タイトル・文面は **7A と同じ既定テンプレート**を使う（ユーザー指示
            # 2026-08-07）。したがって default_comment は持たない。
            },
    "7B":  {"file_key": "s7b", "n_cars": 7, "bet_kind": BET_KIND_TRIO_AXIS2,     "stake_budget": RACE_BUDGET, "gate_filter": None,
            "partners_key": "legs_7b", "tilt_stakes": True,
            # 🔴 **7C より後ろに置いた**（2026-08-07 ユーザー判断）。重複するレースは
            #    7C が取り、7B は独自レース（3.14件/日）だけを出す。7C に譲るのは
            #    設計どおりなので `overlap_expected` で失敗集計から外す。
            "overlap_expected": True,
            # ⚠️ 2026-08-05 の PR#12 で 7B は「◎○一致 × **順序一致** × 準決勝」へ
            #    全面入替した。旧7Bは順序**不一致**が条件だったため、旧文面の
            #    「1番手評価が異なり」は現行条件と正反対になっていた（2026-08-06 是正）。
            #    また外部サイトの予想印を「公式予想」と呼ぶのは誤りなので言及しない。
            #    定義を変えるときは必ずこの文面と DB の comment_template も見ること。
            "default_comment": (
                "本日の二軸をお届けします。\n\n"
                "準決勝の中から、当方の指数で軸2車が明確に絞り込めたレースだけを"
                "お届けしています。相手も3点に絞りました。\n"
                "買い目は三連複・軸2車から相手3点。金額は均等ではなく、当方が想定する"
                "発走時オッズに応じて配分しています。\n\n"
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

# 勝負アイコン「自信あり」を付けるランク（2026-08-05・ユーザー指示で 7SS のみ）。
# 🔴 **単一正本にする**。傾斜配分の導入で submit_pick 経路と submit_pick_multi 経路の
#    2箇所で判定するようになったため、`rank_key == "7SS"` を手書きで2つ持つと
#    片方だけ直して静かに食い違う（本ファイルの RANK_ORDER で実際に起きた型）。
CONFIDENT_RANKS = {"7SS"}


# ---------------------------------------------------------------------------
# 選手成績表（1着率・3着内率）
# frontend/src/app/keirin/page.tsx の sigmoid/logit/solveLogitShift と同一ロジックの
# Python移植。pred_win_pct/pred_top3_pct（選手ごと独立モデルの生確率）はレース内合計が
# 揃わないため、ロジット空間で一律シフトして単勝=100%・複勝=min(出走数,3)*100%に補正する。
# ---------------------------------------------------------------------------

# 🔴 正規化の実装は `src/race_shape.py` が単一正本。タイトルの構造ラベルも
#    同じ正規化値で判定するため、ここに再実装すると表と見立てが食い違いうる。
_sigmoid = sigmoid
_logit = logit
_solve_logit_shift = solve_logit_shift


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

def _load_shape_entries(race_key: str) -> list[dict]:
    """構造ラベル判定に要る列だけを読む（`src/race_shape.py` の入力）。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT frame_no, pred_win_pct, pred_top3_pct, style, line_group "
            "FROM wt_entries WHERE race_key = ? ORDER BY frame_no",
            (race_key,),
        ).fetchall()
    return [dict(r) for r in rows]


def _shape_texts(race_key: str, rank_key: str, axis1: int, axis2: int) -> tuple[str, str]:
    """(タイトル後半, 見解本文の冒頭) を返す。

    ⚠️ 軸2車は**そのランク自身のもの**を渡すこと。7C だけ軸が `axis1_7c`/`axis2_7c`
       で他ランクと別物なので、取り違えると全レースで line/split が入れ替わる。
    """
    entries = _load_shape_entries(race_key)
    shape = classify_shape(rank_key, entries, axis1, axis2)
    title_text, warning = shape_title_text(rank_key, shape)
    note_text, _ = shape_note_text(rank_key, shape)
    if warning:
        print(f"[netkeirin_submit] WARN {race_key}: {warning}", flush=True)
    return title_text, note_text


def _shape_text(race_key: str, rank_key: str, axis1: int, axis2: int) -> str:
    """タイトル後半だけが要る呼び出し向け（`draft_7h1_submission.py`）。"""
    return _shape_texts(race_key, rank_key, axis1, axis2)[0]


def _stake_note_for(rank_key: str, legs: list[BetLeg]) -> str:
    """実際に入稿する買い目から配分の説明文を決める。

    🔴 **legs から導く**のが肝。ダッチ配分も傾斜配分も朝オッズが揃わなければ均等へ
       フォールバックする（欠損は約半数）ため、テンプレートに「オッズに応じて配分」と
       固定で書くと**半分のレースで嘘になる**（仕様書 §4-6 実態一致の原則）。
       券種ごとに単価がばらついているかで判定する（7H1 は三連単と三連複で単価が
       違うのが正常なので、券種をまたいで比べてはいけない）。
    """
    by_kind: dict[str, set[int]] = {}
    for leg in legs:
        by_kind.setdefault(leg.bet_kind, set()).add(leg.stake_per_line)
    tilted = any(len(v) > 1 for v in by_kind.values())
    return stake_note_text(rank_key, tilted)


def _apply_template(
    template: str, *, venue_name: str, race_no: int, rank_key: str, target_date: str,
    axis1: int, axis2: int, shape: str = "", shape_note: str = "",
    stake_note: str = "", race_type: str = "",
) -> str:
    """{venue}{race_no}{rank}{date}{axis1}{axis2}{shape}{shape_note}{stake_note}{race_type}
    を置換する。
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
        "{shape}": shape,
        "{shape_note}": shape_note,
        "{stake_note}": stake_note,
        # 看板レース用（`--marquee`）。決勝/特選/ガールズ決勝 等をそのまま入れる。
        # 通常経路では空文字なので、既存テンプレートに影響しない。
        "{race_type}": race_type,
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


# ---------------------------------------------------------------------------
# 候補・設定・送信済み記録の読み書き
# ---------------------------------------------------------------------------

def _load_meeting_waves(target_date: str) -> dict[str, str]:
    """race_key → 入稿の波（開催＝会場×日 の第1R発走時刻で決まる）。

    netkeirin は**公開後の差し替えができない**ので、板が育つのを待ってから
    入稿するしかない。どの開催をいつ出すかは `src/meeting_wave.py` が正本。
    """
    waves: dict[str, str] = {}
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key, venue_id, start_at FROM wt_races WHERE race_date = ?",
            (target_date,),
        ).fetchall()
    first: dict[str, float] = {}
    parsed: list[tuple[str, str, float | None]] = []
    for r in rows:
        try:
            hour = (int(r["start_at"]) + 9 * 3600) % 86400 / 3600 if r["start_at"] else None
        except (TypeError, ValueError):
            hour = None
        parsed.append((r["race_key"], str(r["venue_id"]), hour))
        if hour is not None:
            v = str(r["venue_id"])
            first[v] = min(first.get(v, 1e9), hour)
    for race_key, venue, _ in parsed:
        waves[race_key] = wave_of_first_hour(first.get(venue))
    return waves


def _load_started_races(target_date: str) -> set[str]:
    """**すでに発走した**レースの race_key。

    🔴 入稿は「まだ売れるレース」にしか意味がない。従来は入稿が朝の1回だけで
       第1レースより前に必ず終わっていたため誰も見ていなかったが、2026-08-07 に
       開催単位の波（昼13:00・夕18:00）と手動再実行が入ったことで、
       **終わったレースへ商品を出しうる**ようになった。
       実際 2026-08-07 17時の再入稿で、朝の波に 岐阜4R(09:32)・6R(10:14) が
       未入稿のまま残っており、ガードが無ければそのまま出していた。

    発走時刻が取れないレースは**発走していない扱い**にする（安全側＝出す）。
    情報が無いことを理由に商品を落とすと、黙って商品が消える。
    """
    now = datetime.now().timestamp()
    started: set[str] = set()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT race_key, start_at FROM wt_races WHERE race_date = ?",
            (target_date,),
        ).fetchall()
    for r in rows:
        try:
            if r["start_at"] and int(r["start_at"]) <= now:
                started.add(r["race_key"])
        except (TypeError, ValueError):
            continue
    return started


def _load_candidates(target_date: str, session: str, file_key: str) -> list[dict]:
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    # 波ごとの再生成ファイルがあればそれを使い、無ければ朝の生成物へ落とす。
    # 🔴 **フォールバックは必須**。夜の再生成（evening_picks_wt.sh）が動かなかった日に
    #    「ファイルが無いから入稿しない」だと、朝の入稿からも波で除外されている
    #    ミッドナイトが**その日まるごと商品ゼロ**になる。予想自体は朝に全開催ぶん
    #    出来ているので、それを使って出すほうが必ず良い。
    prefixes = {"evening": "_night", "noon": "_noon"}
    candidates = []
    if session in prefixes:
        candidates.append(picks_dir /
                          f"wave_picks_wt_{target_date}{prefixes[session]}_{file_key}_candidates.json")
    candidates.append(picks_dir / f"wave_picks_wt_{target_date}_{file_key}_candidates.json")
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
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


_BET_TYPE_JP = {
    BET_KIND_TRIO_AXIS2: "3連複",
    BET_KIND_TRIO_BOX: "3連複",
    BET_KIND_TRIFECTA_AXIS1: "3連単",
    BET_KIND_TRIFECTA_FORMATION: "3連単",
}


def _load_trifecta_board(race_key: str) -> dict[tuple[int, ...], float]:
    """三連単オッズ（7H1 の買い目にオッズを添えるため）。`_load_trio_board` と同方針。"""
    board: dict[tuple[int, ...], float] = {}
    with get_connection() as conn:
        for sql in (
            "SELECT combination, odds_value FROM wt_odds "
            "WHERE race_key = ? AND bet_type = 'trifecta'",
            "SELECT combination, odds_value FROM wt_odds_snapshot "
            "WHERE race_key = ? AND bet_type = 'trifecta' AND snapshot_type = 'morning'",
        ):
            for comb, od in conn.execute(sql, (race_key,)).fetchall():
                if od is None or not (0 < float(od) < 9000):
                    continue
                try:
                    key = tuple(int(x) for x in _SEP_RE.split(str(comb)))
                except ValueError:
                    continue
                if len(key) == 3:
                    board[key] = float(od)
            if board:
                return board
    return board


def build_bet_detail(legs: list[BetLeg], source: str | None = None,
                     odds: dict | None = None) -> str:
    """入稿した買い目と1点ごとの金額を JSON 文字列にする（Web 表示用）。

    🔴 **展開まで済ませて保存する。** 傾斜配分では点ごとに金額が違い、しかも
       その金額は入稿時点の想定オッズから決まるので、**あとから再現できない**。
       グループ表記のまま持つと表示側が `expand_bet` 相当を再実装することになり、
       買い目の解釈が2箇所に分かれる（この種の二重管理はこのリポジトリで
       繰り返し事故を起こしている）。

    形式:
        {"total": 10000, "source": "blend",
         "lines": [{"bet_type": "3連複", "combo": "1=2=5", "stake": 4100,
                    "odds": 8.3}, ...]}

    `source` は金額配分の出どころ（blend / odds / model / equal・
    `src.stake_allocation` 参照）。均等配分のランクは None。

    `odds` は**入稿時点の**オッズ（三連複は frozenset・三連単は tuple がキー）。
    🔴 **配分の根拠そのものなので一緒に保存する。** あとから引くと発走時の値に
       なってしまい、「なぜこの金額なのか」が読めなくなる。取れなければ None。
    """
    odds = odds or {}
    lines: list[dict[str, Any]] = []
    for leg in legs:
        for target in sorted(expand_bet(leg.bet_kind, leg.groups),
                             key=lambda t: tuple(sorted(t)) if isinstance(t, frozenset) else t):
            cars = sorted(target) if isinstance(target, frozenset) else list(target)
            sep = "=" if isinstance(target, frozenset) else "-"
            o = odds.get(target)
            lines.append({
                "bet_type": _BET_TYPE_JP.get(leg.bet_kind, leg.bet_kind),
                "combo": sep.join(str(c) for c in cars),
                "stake": int(leg.stake_per_line),
                "odds": round(float(o), 1) if o else None,
            })
    payload = {"total": sum(x["stake"] for x in lines), "source": source, "lines": lines}
    # ダッチ配分のときは保証倍率も一緒に残す（仕様書 §6 の前向き計測）。
    # 🔴 picks_history に列を足さずここへ入れているのは、**スキーマ変更を伴わずに
    #    記録したいから**。列が必要になったらここから移送できる。
    if source and source.startswith("dutch:") and lines:
        total = payload["total"]
        rets = [x["stake"] * x["odds"] / total for x in lines if x.get("odds") and total]
        if rets:
            payload["dutch_min_return"] = round(min(rets), 4)
    return json.dumps(payload, ensure_ascii=False)


def _legs_for_record(cfg: dict, axis1: int, axis2_or_p1: int, partners: list[int],
                     stake: int) -> list[BetLeg]:
    """`submit_pick`（均等配分）で送る買い目を BetLeg 表現へ揃える。

    記録・表示は `build_bet_detail()` に一本化したいので、傾斜配分経路と
    同じ形へ寄せる。ここで組む groups は `submit_pick` → `build_bet_id()` が
    組むものと同一（同 bet_kind の分岐をそのまま写している）。
    """
    if cfg["bet_kind"] == BET_KIND_TRIFECTA_AXIS1:
        groups = [[axis1], list(partners)]
    else:
        groups = [[axis1], [axis2_or_p1], list(partners)]
    return [BetLeg(cfg["bet_kind"], groups, stake)]


def _bet_detail_odds(race_key: str, cfg: dict, use_trifecta: bool = False) -> dict:
    """買い目に添えるオッズ。

    ⚠️ **cfg だけで決めてはいけない。** 7C は同じ cfg のまま単勝率で三連複と
       三連単を切り替えるため、cfg 由来の判定だと切替時に三連単オッズが欠け、
       `bet_detail` が買った目のオッズを持たない行になる（Web が空表示になる）。
       実際に何を買ったかを `use_trifecta` で受け取る。
    """
    base = str(race_key).split("#")[0]
    odds: dict = dict(_load_trio_board(base))
    if cfg.get("multi_bet") or cfg.get("formation_bet") or use_trifecta:
        odds.update(_load_trifecta_board(base))
    return odds


def _record_submission(
    race_key: str, rank_key: str, session: str, venue_name: str, race_no: int,
    gate_label: str | None, axis1: int, axis2: int, netkeirin_race_id: str,
    bet_detail: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO netkeirin_submissions "
            "(race_key,rank_key,session,venue_name,race_no,gate_label,axis1,axis2,"
            "netkeirin_race_id,bet_detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (race_key, rank_key, session, venue_name, race_no, gate_label, axis1, axis2,
             netkeirin_race_id, bet_detail),
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
        return unit_stake(n_lines, int(budget))
    return int(cfg["stake_per_line"])


def _load_trio_board(race_key: str) -> dict[frozenset[int], float]:
    """レースの三連複オッズ盤面（朝の値を優先）。

    ⚠️ **`wt_odds` は '1=2=3' / `wt_odds_snapshot` は '1-2-3'** と区切り文字が違う。
       片方だけを想定すると盤面が丸ごと空になり、**無言で均等割りへ落ちる**
       （検証スクリプトで実際にサンプルが 1/20 に消えた）。両方を受ける。

    🔴 **常に `wt_odds`（＝その時点の最新）を優先する。** 朝スナップショットは
    フォールバックにすぎない。理由は板の厚さが**時計時刻ではなく発走までの近さ**で
    決まるため（実測・2026-08-07）:

        朝8:12時点の三連複 未確定率 —
          〜10時台発走 0.8% / 11-13時 9.4% / 14-16時 17.2% /
          17-19時 34.6% / **20時以降 58.3%**

    夜のレースは朝には板がほぼ無い。あとから入稿するほど良くなるので、
    その時刻の最新値を使えなければ意味がない（`wt_odds` は日中 15分ごとの
    `intraday_results_wt.sh` で更新される）。朝の定時バッチではこの2つは
    同じ値になるので、優先順位を変えても朝の挙動は一切変わらない。

    ⚠️ **9999.9 は winticket の「オッズ未確定」センチネル**（朝の三連複の23%）。
       実際の確定値は 1.1倍〜と幅がありオッズとして使えないので、
       0<odds<9000 の範囲外は**採らない**（買う点が1つでも欠ければモデル配分に落ちる）。
    """
    board: dict[frozenset[int], float] = {}
    with get_connection() as conn:
        for sql, params in (
            ("SELECT combination, odds_value FROM wt_odds "
             "WHERE race_key = ? AND bet_type = 'trio'", (race_key,)),
            ("SELECT combination, odds_value FROM wt_odds_snapshot "
             "WHERE race_key = ? AND bet_type = 'trio' AND snapshot_type = 'morning'",
             (race_key,)),
        ):
            for comb, od in conn.execute(sql, params).fetchall():
                if od is None or not (0 < float(od) < 9000):
                    continue
                try:
                    key = frozenset(int(x) for x in _SEP_RE.split(str(comb)))
                except ValueError:
                    continue
                if len(key) == 3:
                    board[key] = float(od)
            if board:
                return board
    return board


def _load_top3_probs(race_key: str) -> dict[int, float]:
    """{車番: モデルの3着内率 0-1}。`wt_entries.pred_top3_pct` は日次バッチが
    候補生成の直後（入稿より前）に書くので、入稿時点で必ず読める。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT frame_no, pred_top3_pct FROM wt_entries WHERE race_key = ?",
            (race_key,),
        ).fetchall()
    return {int(r["frame_no"]): float(r["pred_top3_pct"]) / 100
            for r in rows if r["pred_top3_pct"] is not None}


def _build_tilted_legs(
    race_key: str, cfg: dict, axis1: int, axis2: int, partners: list[int],
) -> tuple[list[BetLeg], str, dict[int, int]]:
    """想定着地オッズに応じた傾斜配分の買い目行を組み立てる。

    returns (買い目行, 重みの出どころ, {相手車番: 賭け金})。
    同額の相手は1行にまとめる（netkeirin の1行は bet_money を1つしか持てない）。
    """
    board = _load_trio_board(race_key)
    morning = {t: board.get(frozenset({axis1, axis2, t})) for t in partners}
    morning = {t: o for t, o in morning.items() if o}
    stakes, source = tilted_stakes(
        partners, morning, _load_top3_probs(race_key),
        budget=int(cfg.get("stake_budget") or RACE_BUDGET),
    )
    legs = [BetLeg(BET_KIND_TRIO_AXIS2, [[axis1], [axis2], cars], stake)
            for stake, cars in group_by_stake(stakes)]
    return legs, source, stakes


def _build_trifecta_head_legs(
    cfg: dict, axis1: int, axis2: int, partners: list[int],
) -> tuple[list[BetLeg], dict[int, str]]:
    """三連単「1着=軸1 / 2着=軸2 / 3着=相手流し」を1行のフォーメーションで組む。

    `BET_KIND_TRIFECTA_FORMATION` の groups は [1着列, 2着列, 3着列] で、
    `expand_bet` は {(a,b,c)} を返す。[[axis1],[axis2],partners] なら
    **三連複と同じ len(partners) 点**になる（これが要点）。

    🔴 **点数を増やしてはいけない。** この予算方式では「ガミ ⟺ 的中オッズ < 点数」
       なので、点数を増やすとガミ境界も一緒に上がり、三連単化の意味が消える。
       検証では 2k点/4k点の案がいずれも実質的中を落とした
       （`RANK_7C_TRIFECTA_PW_MIN` の定義部を参照）。

    ⚠️ **傾斜配分はしない**（均等割り）。採用根拠となった検証が均等割り前提。
       三連単の板は三連複より薄く、朝の時点では配分の推定が当てにならない。
    """
    unit = _stake_per_line(cfg, len(partners))
    legs = [BetLeg(BET_KIND_TRIFECTA_FORMATION,
                   [[axis1], [axis2], list(partners)], unit)]
    marks = {**{c: "△" for c in partners}, axis1: "◎", axis2: "○"}
    return legs, marks


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
    # 手動入稿の `_process_manual` は axis1 == axis2 を弾くのに、自動経路には
    # そのチェックが無かった（2026-08-08 是正）。同じ車が2つ来ると
    # `expand_bet(BET_KIND_TRIO_AXIS2, ...)` が要素2つの frozenset を返し、
    # **三連複として不正な買い目**をそのまま入稿してしまう。
    # 現状は候補生成側が保証しているが、経路ごとに防御が非対称なのは危うい。
    if axis1 == axis2:
        raise ValueError(f"軸1と軸2が同じ車です（{k1}={axis1} / {k2}={axis2}）")
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


def _normalize_formation_candidate(
    cand: dict, cfg: dict, race_key: str | None = None,
) -> tuple[list[BetLeg], dict[int, str], int, int]:
    """9H1 候補から (買い目行, 印, ◎車番, ○車番) を返す。

    9H1 は**三連単フォーメーションの単一券種**なので、7H1 の 2券種併買
    （`_normalize_multi_candidate`）とは別経路にする。賭け金は
    `strategy_wt.rank_9h1_stakes()`（1レース1万円の予算枠 ÷ 点数）で決める。

    印: ◎ = 1着固定車 / ○▲ = 2着列（モデル3着内率の降順）/ △ = 3着だけの車。
    """
    legs_raw = list(cand.get("legs") or [])
    first, second, third = _trifecta_formation_groups(legs_raw)
    if len(first) != 1:
        raise ValueError(f"9H1 の1着は1車固定のはずです: {first}")
    unit, _total = rank_9h1_stakes(len(legs_raw))

    # 2着列の ○/▲ は候補JSONの order（モデル3着内率の降順）に従う。
    # bet_id は車番昇順に正規化されるため、買い目の並びに序列を委ねない。
    order = [int(x) for x in (cand.get("order") or [])]
    ranked_second = sorted(second, key=lambda c: order.index(c) if c in order else 99)
    marks: dict[int, str] = {first[0]: "◎"}
    if ranked_second:
        marks[ranked_second[0]] = "○"
    if len(ranked_second) > 1:
        marks[ranked_second[1]] = "▲"
    for c in third:
        marks.setdefault(c, "△")

    # ダッチ配分（2026-08-09・仕様書 §2B）。朝オッズが揃わない／条件不成立なら
    # 従来の均等（rank_9h1_stakes）へフォールバックする。
    tf_board = _load_trifecta_board(race_key) if race_key else {}
    tf_points = sorted(expand_bet(BET_KIND_TRIFECTA_FORMATION, [first, second, third]))
    dutch_legs, _dutch = _dutch_point_legs(tf_points, [], tf_board, {})
    legs = dutch_legs or [BetLeg(BET_KIND_TRIFECTA_FORMATION, [first, second, third], unit)]
    axis2 = ranked_second[0] if ranked_second else first[0]
    return legs, marks, first[0], axis2


def _dutch_point_legs(
    tf_points: list[tuple[int, ...]],
    trio_points: list[frozenset],
    tf_odds: dict,
    trio_odds: dict,
) -> tuple[list[BetLeg], object]:
    """高配当ランク(7H1/9H1)の買い目をダッチ配分し、**1点=1行**の BetLeg にする。

    仕様書 §2B。低オッズ目を切って「当たれば必ず予算を上回る」形へ寄せる。
    的中率ランク(7C/7A/7S)には**絶対に使わない**（人気の目が的中の源泉のため）。

    ⚠️ EV(Σp·o)≥1.3 の判定は**まだ効いていない**。候補JSON(`legs_tf`/`legs_trio`)に
       点ごとの的中確率が無く、submit 時点で honest に計算できないため
       `probs=None` を渡している（`reason='ok_no_ev_check'`）。EV を効かせるには
       候補生成側で点ごとの確率を JSON へ出す必要がある。

    🔴 **買う点すべてにオッズが揃っているときだけダッチにする。** 一部でも欠けると
       「安い目だから切った」のか「オッズが取れなかったから消えた」のか区別できず、
       券種がまるごと落ちる（三連単オッズだけ無い → 7H1 が三連複単券種になる）。
       欠けたら行を返さず、呼び出し側の従来配分へフォールバックさせる。

    returns (買い目行, DutchResult)。買わない判断のときは行が空リストになる。
    """
    wanted = [("tf", p) for p in tf_points] + [("trio", t) for t in trio_points]
    odds: dict = {}
    for key in wanted:
        o = (tf_odds if key[0] == "tf" else trio_odds).get(key[1])
        if not o:
            return [], dutch_allocate({}, probs=None, budget=RACE_BUDGET)
        odds[key] = float(o)

    result = dutch_allocate(odds, probs=None, budget=RACE_BUDGET)
    if not result.buy:
        return [], result

    legs: list[BetLeg] = []
    for key, stake in sorted(result.stakes.items(), key=lambda kv: str(kv[0])):
        kind, target = key
        if kind == "tf":
            # 1点だけの三連単は「各着に1車ずつのフォーメーション」で表せる
            legs.append(BetLeg(BET_KIND_TRIFECTA_FORMATION,
                               [[target[0]], [target[1]], [target[2]]], stake))
        else:
            legs.append(BetLeg(BET_KIND_TRIO_BOX, [sorted(target)], stake))
    return legs, result


def _normalize_multi_candidate(
    cand: dict, cfg: dict, race_key: str | None = None,
) -> tuple[list[BetLeg], dict[int, str], int, int, str | None]:
    """7H1 候補から (買い目行, 印, ◎車番, ○車番, 配分の出どころ) を返す。

    印（ユーザー確定・2026-08-06）:
      ◎ = 三連単の1着固定車 / ○ = 2着列の1番手 / ▲ = 2着列の2番手 /
      △ = 3着列の残り（3着だけで買っている車）/ 除外した本命は印なし(--)
    """
    tf_groups = _trifecta_formation_groups(cand.get("legs_tf") or [])
    trio_cars = _trio_box_group(cand.get("legs_trio") or [])

    # 🔴 賭け金は**候補JSONの値を使わず入稿時点で決め直す**（2026-08-07）。
    #    三連単は 1点 RANK_7H1_TF_UNIT 円の均等、残りを三連複へ回し、
    #    **入稿時点のオッズで払戻が等しくなるよう**配分する。
    #    候補JSONの stake_* は朝の候補生成時点の値で、板が育ってから入稿する
    #    設計（開催単位の3波）とは時点が食い違うため使わない。
    trio_legs = [frozenset(int(x) for x in _SEP_RE.split(str(c)))
                 for c in (cand.get("legs_trio") or [])]
    trio_legs = [t for t in trio_legs if len(t) == 3]
    board = _load_trio_board(race_key) if race_key else {}
    odds = {t: board.get(t) for t in trio_legs}
    source = "odds" if (odds and all(odds.values())) else "equal"
    stake_tf = RANK_7H1_TF_UNIT
    trio_stakes = rank_7h1_trio_stakes(
        trio_legs, odds if source == "odds" else None, len(cand.get("legs_tf") or []))

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

    axis2 = ranked_second[0] if ranked_second else first[0]

    # ── ダッチ配分（2026-08-09・仕様書 §2B）──────────────────────────────
    # 朝オッズが**全点**に揃えば低オッズ目を切って「当たれば予算超え」の形へ寄せる。
    # 揃わない／条件不成立なら**従来の配分へフォールバック**する（見送りにはしない。
    # 朝オッズ欠損は約半数あり、そこを全部落とすと入稿が消える＝仕様書 §7）。
    tf_board = _load_trifecta_board(race_key) if race_key else {}
    tf_points = sorted(expand_bet(BET_KIND_TRIFECTA_FORMATION, tf_groups))
    dutch_legs, dutch = _dutch_point_legs(tf_points, trio_legs, tf_board, board)
    if dutch_legs:
        return dutch_legs, marks, first[0], axis2, f"dutch:{dutch.reason}"

    # 三連複は目ごとに金額が違うので **1目 = 1行**（3車のBOX＝1点）で出す。
    # 同額でも束ねられない（BOXは車群でしか表現できず、任意の部分集合を作れない）。
    legs = [BetLeg(BET_KIND_TRIFECTA_FORMATION, tf_groups, stake_tf)]
    for t in sorted(trio_legs, key=lambda x: sorted(x)):
        legs.append(BetLeg(BET_KIND_TRIO_BOX, [sorted(t)], trio_stakes[t]))
    return legs, marks, first[0], axis2, source


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def _process_rank(
    rank_key: str, target_date: str, session: str, race_date, settings: dict[str, dict],
    already: set[tuple[str, str]], dry_run: bool, race_key_filter: str | None = None,
    claimed_races: set[str] | None = None, waves: dict[str, str] | None = None,
    started: set[str] | None = None,
) -> tuple[int, list[str]]:
    cfg = RANK_CONFIGS[rank_key]
    if not _is_enabled(settings, rank_key):
        return 0, []

    raw = _load_candidates(target_date, session, cfg["file_key"])
    if race_key_filter:
        raw = [c for c in raw if c.get("race_key") == race_key_filter]
    # 🔴 この回で担当する開催だけに絞る。朝の候補JSONは当日全開催ぶん入っている
    #    （予想・Discord・Web は朝に全部出す）ので、ここで落とさないと
    #    夜の開催まで朝に入稿してしまい、板が育つ前の配分で確定してしまう。
    #    ⚠️ 自分の波と**完全一致**で絞ると、発走時刻が前倒しに訂正された開催が
    #    通過済みの波へ移り、その日どの回からも入稿されない（2026-08-08 是正）。
    #    `waves_due_by()` で「自分の波 + 前の波」を対象にする。二重入稿は
    #    `_already_submitted()` が、発走済みは直下の `started` が止める。
    if waves is not None:
        due_waves = set(waves_due_by(SESSION_WAVE.get(session, WAVE_MORNING)))
        raw = [c for c in raw
               if waves.get(str(c.get("race_key", "")).split("#")[0], WAVE_MORNING) in due_waves]
    # 発走済みのレースへは出さない（売れないので商品にならない）。
    if started is not None:
        n_before = len(raw)
        raw = [c for c in raw if str(c.get("race_key", "")).split("#")[0] not in started]
        if len(raw) < n_before:
            print(f"[netkeirin_submit] {rank_key}: 発走済み {n_before - len(raw)}件を除外",
                  flush=True)
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
    is_formation = bool(cfg.get("formation_bet"))

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
        tilt_source: str | None = None
        tilt_stakes_map: dict[int, int] = {}
        use_trifecta = False
        try:
            if is_multi:
                legs, marks, axis1, axis2_or_p1, tilt_source = _normalize_multi_candidate(
                    cand, cfg, race_key.split("#")[0])
            elif is_formation:
                legs, marks, axis1, axis2_or_p1 = _normalize_formation_candidate(
                    cand, cfg, race_key.split("#")[0])
            else:
                axis1, axis2_or_p1, partners, marks = _normalize_candidate(cand, cfg)
                # 三連単への切替（`trifecta_switch_key` を持つランクのみ）。
                # 判定済みの真偽値を候補JSONから読むだけにして、ここで
                # win_probs から再判定しない（朝の予想と入稿の根拠がずれるため）。
                switch_key = cfg.get("trifecta_switch_key")
                if switch_key and cand.get(switch_key):
                    legs, marks = _build_trifecta_head_legs(
                        cfg, axis1, axis2_or_p1, partners)
                    use_trifecta = True
                elif cfg.get("tilt_stakes"):
                    legs, tilt_source, tilt_stakes_map = _build_tilted_legs(
                        race_key, cfg, axis1, axis2_or_p1, partners)
                    # 🔴 印を submit_pick が内部で作っていたものと**同じ**にする。
                    #    submit_pick は軸=◎○・**買った相手=△**・買っていない車=印なし
                    #    を自前で組むが、submit_pick_multi は渡された marks をそのまま
                    #    使う。`_normalize_candidate` の marks は軸2車しか持たないので、
                    #    ここで補わないと相手の△が全部消える（2026-08-03 に 7B で
                    #    直したのと同型の「表示と入稿の食い違い」）。
                    marks = {**{c: "△" for c in partners},
                             axis1: "◎", axis2_or_p1: "○"}
        except (ValueError, KeyError, TypeError, IndexError) as e:
            failures.append(f"{race_key} ({rank_key}): 候補情報不正 - {e}")
            print(f"[netkeirin_submit] スキップ {venue_name}{race_no}R ({rank_key}): {e}",
                  flush=True)
            continue

        shape, shape_note = _shape_texts(race_key, rank_key, axis1, axis2_or_p1)
        stake_note = _stake_note_for(rank_key, legs)
        title = _apply_template(
            title_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
            target_date=target_date, axis1=axis1, axis2=axis2_or_p1, shape=shape,
            shape_note=shape_note, stake_note=stake_note,
        )
        comment = _apply_template(
            comment_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
            target_date=target_date, axis1=axis1, axis2=axis2_or_p1, shape=shape,
            shape_note=shape_note, stake_note=stake_note,
        )
        entry_table = _build_entry_table(race_key, marks)
        if entry_table:
            comment = f"{comment}\n\n{entry_table}"

        if dry_run:
            # 🔴 `is_formation`(9H1) を落とすと **preview だけ** _stake_per_line で
            #    落ちる。本番経路は通るので気づきにくく、「本番で何が出るか確かめる
            #    道具」が肝心のときに使えない。legs を組み終えた経路は全部ここで出す。
            if legs and not tilt_source:
                detail = "\n".join(
                    f"  {leg.bet_kind}: {leg.groups} × {leg.stake_per_line:,}円/点"
                    + f"（{len(expand_bet(leg.bet_kind, leg.groups))}点）"
                    for leg in legs)
            elif tilt_source:
                detail = (
                    f"  軸={axis1},{axis2_or_p1} 傾斜配分(出どころ={tilt_source}・"
                    f"合計{sum(tilt_stakes_map.values()):,}円)\n"
                    + "\n".join(f"    相手{car}: {stake:,}円"
                                for car, stake in sorted(tilt_stakes_map.items()))
                )
            else:
                detail = (f"  軸={axis1} 相手={partners} "
                          f"賭け金={_stake_per_line(cfg, len(partners)):,}円/点")
            print(
                f"[dry-run] {venue_name}{race_no}R ({rank_key}) 印={marks}\n"
                f"{detail}\n"
                f"  タイトル: {title}\n"
                f"  コメント:\n{comment}\n",
                flush=True,
            )
            # 🔴 dry-run でも**レースを確保する**。netkeirin は1レース1商品なので
            #    本番では先に入稿したランクが取り、後続ランクは衝突としてスキップされる。
            #    ここで確保しないと dry-run だけ同じレースが複数ランクで出力され、
            #    「本番で何が出るか」を確かめる道具として嘘をつく
            #    （2026-08-07 実測: 伊東9R が 7B と 7C の両方に出ていた）。
            if claimed_races is not None:
                claimed_races.add(race_key)
            n_submitted += 1
            continue

        try:
            assert client is not None
            # 🔴 判定は **legs を組み終えたか** だけで行う（ランク名や個別フラグを
            #    列挙しない）。`submit_pick` は `cfg["bet_kind"]` で券種を決めるので、
            #    7C の三連単切替のように **cfg と実際の買い目が食い違う**ケースを
            #    ここへ流すと、三連単を組んだのに**三連複が入稿される**。
            #    フラグ列挙は 9H1 で実際に事故（2026-08-09 朝）を起こした型。
            if legs:
                # 傾斜配分は点ごとに bet_money が違うので、同額どうしをまとめた
                # 複数行として送る（`kaime` は配列なので submit は1回のまま）。
                # 🔴 `is_formation`(9H1) もここを通す。単一券種だが軸+相手では
                #    表現できないフォーメーションなので `submit_pick` は使えない
                #    （`cfg["bet_kind"]` を持たないため KeyError になる）。
                ok, msg = client.submit_pick_multi(
                    race_date=race_date, venue_name=venue_name, race_no=race_no,
                    n_cars=cfg["n_cars"], legs=legs, marks=marks,
                    title=title, comment=comment,
                    # 「自信あり」は最上位の 7SS のみ。submit_pick は confident=True
                    # から act_type を導くが submit_pick_multi は act_type 直指定なので、
                    # 傾斜配分経路でも同じ勝負アイコンになるようここで合わせる。
                    act_type=cfg.get(
                        "act_type",
                        ACT_TYPE_CONFIDENT if rank_key in CONFIDENT_RANKS
                        else ACT_TYPE_DEFAULT),
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
                    confident=(rank_key in CONFIDENT_RANKS),
                )
        except Exception as e:
            ok, msg = False, f"例外: {e}"

        if ok:
            # 🔴 `is_formation`(9H1) を外すと **入稿は成功した後に記録で落ちる**。
            #    submit_pick_multi → 成功 → ここで KeyError('stake_per_line') →
            #    _record_submission に到達せず、**netkeirin には出ているのに
            #    netkeirin_submissions に無い**行が生まれ、さらに例外が _process_rank を
            #    抜けて **その波の後続ランク(7SS/7S/7A/7C/7B)が丸ごと入稿されない**。
            #    9H1 導入(2026-08-08)から 2026-08-09 朝まで実際に発生した。
            #    formation/multi は候補正規化側で legs を組み終えているので
            #    _legs_for_record（軸+相手の均等割り前提）に渡してはいけない。
            #    ここも上の送信分岐と同じく **legs の有無**だけで判定する。
            record_legs = legs if legs else _legs_for_record(
                cfg, axis1, axis2_or_p1, partners, _stake_per_line(cfg, len(partners)))
            _record_submission(
                race_key, rank_key, session, venue_name, race_no, gate_label, axis1, axis2_or_p1, msg,
                bet_detail=build_bet_detail(
                    record_legs, tilt_source,
                    _bet_detail_odds(race_key, cfg, use_trifecta)),
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


def _resolve_race_info(race_key: str) -> tuple[str, int, int, str] | None:
    """race_keyから (venue_name, race_no, n_entries, race_type) を候補JSON非依存で解決する。

    race_type は看板レース用テンプレートの `{race_type}` に使う（「決勝」「特選」
    「ガールズ決勝」等）。NULL のときは空文字を返す。
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT venue_id, race_no, n_entries, race_type FROM wt_races WHERE race_key = ?",
            (race_key,),
        ).fetchone()
        if row is None:
            return None
        vrow = conn.execute(
            "SELECT name FROM venue_info WHERE venue_code = ?", (row["venue_id"],),
        ).fetchone()
        venue_name = vrow["name"] if vrow else str(row["venue_id"])
    return venue_name, int(row["race_no"]), int(row["n_entries"]), (row["race_type"] or "")


def _process_manual(
    race_key: str, rank_key: str, axis1: int, axis2: int, target_date: str, session: str,
    race_date, settings: dict[str, dict], dry_run: bool, marquee: bool = False,
) -> tuple[int, list[str]]:
    """手動指定（推奨外レースへのランク選択入稿）。候補JSON検索を一切経由しない。

    ON/OFF（_global含む）・重複送信防止(already_submitted)は通常経路と同じルールを
    適用する。gate_filterはSS/S自動判定用のため参照しない（rank_key自体がユーザーの
    明示選択のため）。

    marquee=True のときは看板レース（決勝・特選クラス）専用のタイトル・文面を使う。
    通常ランクの文面は「毎日は出ません」「本命が割れ」と書いてあり、
    **必ず出す**看板レースでは事実と食い違うため（_MARQUEE_COMMENT_TEMPLATE 参照）。
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
    venue_name, race_no, n_entries, race_type = info
    if n_entries != cfg["n_cars"]:
        return 0, [f"{race_key}: 車数不一致（{n_entries}車 / {rank_key}は{cfg['n_cars']}車想定）"]
    if axis1 == axis2 or not (1 <= axis1 <= n_entries) or not (1 <= axis2 <= n_entries):
        return 0, [f"{race_key}: 不正な軸指定 axis1={axis1} axis2={axis2}"]

    partners = [c for c in range(1, n_entries + 1) if c not in (axis1, axis2)]
    gate_label = cfg["gate_filter"]

    setting = settings.get(rank_key)
    if marquee:
        # 🔴 看板レースは設定画面のランク別テンプレートを**使わない**。
        #    設定側は「絞って出す」前提の文面なので、必ず出す看板レースに流用すると
        #    ランク設定を編集するたびに看板の文面まで嘘に戻る。
        title_template = _MARQUEE_TITLE_TEMPLATE
        comment_template = _MARQUEE_COMMENT_TEMPLATE
    else:
        title_template = (setting or {}).get("title_template") or _DEFAULT_TITLE_TEMPLATE
        # ランク固有の既定コメント（cfg["default_comment"]）があればそれを既定にする。
        # 7B は買い目構造が「5点流し」ではなく「相手3点」で、共通既定文の説明が
        # 事実と食い違うため必須（設定画面で上書きされていればそちらが優先）。
        comment_template = ((setting or {}).get("comment_template")
                            or cfg.get("default_comment") or _DEFAULT_COMMENT_TEMPLATE)

    # 手動入稿も自動入稿と**同じ商品**なので配分方式を揃える。
    # 片方だけ均等のままだと、共通の文面「想定オッズに応じて配分しています」が
    # 手動入稿分だけ嘘になる。なおこちらは日中に呼ばれるため朝スナップショットが
    # 無ければ現在の wt_odds を使う＝自動入稿より新しいオッズで配分できる。
    # 🔴 **文面より先に組む**。`{stake_note}` は実際に入稿する買い目から導くため、
    #    legs が確定する前にテンプレートを適用すると常に「均等」になる。
    tilt_source = None
    tilt_stakes_map: dict[int, int] = {}
    legs: list[BetLeg] = []
    if cfg.get("tilt_stakes"):
        legs, tilt_source, tilt_stakes_map = _build_tilted_legs(
            race_key, cfg, axis1, axis2, partners)

    shape, shape_note = _shape_texts(race_key, rank_key, axis1, axis2)
    stake_note = _stake_note_for(rank_key, legs)
    title = _apply_template(
        title_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
        target_date=target_date, axis1=axis1, axis2=axis2, shape=shape,
        shape_note=shape_note, stake_note=stake_note, race_type=race_type,
    )
    comment = _apply_template(
        comment_template, venue_name=venue_name, race_no=race_no, rank_key=rank_key,
        target_date=target_date, axis1=axis1, axis2=axis2, shape=shape,
        shape_note=shape_note, stake_note=stake_note, race_type=race_type,
    )
    entry_table = _build_entry_table(race_key, {axis1: "◎", axis2: "○"})
    if entry_table:
        comment = f"{comment}\n\n{entry_table}"

    if dry_run:
        detail = (
            f"傾斜配分(出どころ={tilt_source}) "
            + " / ".join(f"{c}:{s:,}円" for c, s in sorted(tilt_stakes_map.items()))
            if tilt_source else
            f"賭け金={_stake_per_line(cfg, len(partners)):,}円/点"
        )
        print(
            f"[dry-run][manual] {venue_name}{race_no}R ({rank_key}) "
            f"軸={axis1}-{axis2} 相手={partners} {detail}\n"
            f"  タイトル: {title}\n"
            f"  コメント:\n{comment}\n",
            flush=True,
        )
        return 1, []

    try:
        if tilt_source:
            ok, msg = NetkeirinClient().submit_pick_multi(
                race_date=race_date, venue_name=venue_name, race_no=race_no,
                n_cars=cfg["n_cars"], legs=legs,
                marks={**{c: "△" for c in partners}, axis1: "◎", axis2: "○"},
                title=title, comment=comment,
                act_type=(ACT_TYPE_CONFIDENT if rank_key in CONFIDENT_RANKS
                          else ACT_TYPE_DEFAULT),
            )
        else:
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
        record_legs = legs if tilt_source else _legs_for_record(
            cfg, axis1, axis2, partners, _stake_per_line(cfg, len(partners)))
        _record_submission(race_key, rank_key, session, venue_name, race_no, gate_label,
                           axis1, axis2, msg,
                           bet_detail=build_bet_detail(
                               record_legs, tilt_source, _bet_detail_odds(race_key, cfg)))
        print(f"[netkeirin_submit][manual] 入稿成功 {venue_name}{race_no}R ({rank_key}) → {msg}", flush=True)
        return 1, []
    print(f"[netkeirin_submit][manual] 入稿失敗 {venue_name}{race_no}R ({rank_key}): {msg}", flush=True)
    return 0, [f"{venue_name}{race_no}R({rank_key}): {msg}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_date")
    # session は「どの波の開催を入稿するか」を決める（src/meeting_wave.py 参照）。
    #   morning = モーニング・デイ / noon = ナイター / evening = ミッドナイト
    parser.add_argument("session", choices=("morning", "noon", "evening"))
    parser.add_argument("--dry-run", action="store_true", help="送信せず生成内容を標準出力に出す")
    # 🔴 **未公開の下書きを差し替えるための逃げ道**。文面や配分を直したあと、
    #    既に入稿済みのレースへ出し直したいときだけ使う。netkeirin は同じ
    #    race_id への再POSTで前の商品を**上書き**する（docs 2.5節）ので、
    #    公開前なら差し替えになる。公開後に何が起きるかは未検証なので、
    #    公開済みのレースに対しては使わないこと。
    parser.add_argument("--force", action="store_true",
                        help="入稿済みのレースにも再送する（未公開の下書きの差し替え用）")
    parser.add_argument(
        "--race-key", default=None,
        help="指定時はこのレース(race_key)のみをピンポイントで対象にする（それ以外は通常と同一ルール）",
    )
    parser.add_argument(
        "--manual-rank-key", default=None, choices=MANUAL_ALLOWED_RANKS,
        help="指定時は候補JSON検索を経由せず--axis1/--axis2で手動入稿する（--race-key必須）",
    )
    parser.add_argument(
        "--marquee", action="store_true",
        help="看板レース（決勝・特選クラス）専用のタイトル・文面で入稿する（--manual-rank-key と併用）",
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

    # `--marquee` は手動入稿専用。自動経路で渡されても黙って無視されると
    # 「看板用の文面で出したつもりが通常文面で出ていた」に気づけないので落とす。
    if args.marquee and not args.manual_rank_key:
        print("[netkeirin_submit] --marquee は --manual-rank-key と併用してください", flush=True)
        raise SystemExit(1)

    if args.manual_rank_key:
        if not args.race_key or args.axis1 is None or args.axis2 is None:
            print("[netkeirin_submit] --manual-rank-key には --race-key/--axis1/--axis2 が必須です", flush=True)
            raise SystemExit(1)
        n, failures = _process_manual(
            args.race_key, args.manual_rank_key, args.axis1, args.axis2,
            target_date, session, race_date, settings, args.dry_run,
            marquee=args.marquee,
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

    waves = _load_meeting_waves(target_date)
    started = _load_started_races(target_date)
    want_wave = SESSION_WAVE[session]
    # 自分の波 + 取りこぼした過去の波（発走時刻が前倒しに訂正された開催の救済）。
    # 二重入稿は _already_submitted() が、終わったレースへの入稿は
    # _load_started_races() が止めるので拾い直しても副作用は無い。
    due_waves = set(waves_due_by(want_wave))
    n_wave = sum(1 for w in waves.values() if w == want_wave)
    n_due = sum(1 for w in waves.values() if w in due_waves)
    carry = n_due - n_wave
    print(f"[netkeirin_submit] {target_date} {session}: "
          f"担当は {WAVE_LABEL_JP[want_wave]} — 当日{len(waves)}レース中{n_wave}レース"
          + (f"（+ 前の波の未入稿 {carry}レースも対象）" if carry else ""),
          flush=True)

    all_race_keys: set[str] = set()
    per_rank_raw: dict[str, list[dict]] = {}
    for rank_key in RANK_ORDER:
        if not _is_enabled(settings, rank_key):
            continue
        cfg = RANK_CONFIGS[rank_key]
        raw = _load_candidates(target_date, session, cfg["file_key"])
        if args.race_key:
            raw = [c for c in raw if c.get("race_key") == args.race_key]
        raw = [c for c in raw
               if waves.get(str(c.get("race_key", "")).split("#")[0], WAVE_MORNING) in due_waves
               and str(c.get("race_key", "")).split("#")[0] not in started]
        per_rank_raw[rank_key] = raw
        all_race_keys.update(c["race_key"] for c in raw)

    already = set() if args.force else _already_submitted(sorted(all_race_keys))
    if args.force:
        print('[netkeirin_submit] --force: 入稿済みのレースにも再送します'
              '（未公開の下書きのみ差し替わります）', flush=True)

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
            race_key_filter=args.race_key, claimed_races=claimed_races, waves=waves,
            started=started,
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
