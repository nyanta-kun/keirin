"""レース構造ラベル（netkeirin タイトル差し込み用）の単一正本。

netkeirin の商品タイトルは「狙い（ランクの性格）｜レース見解（構造）」の2ブロックで
構成する。前半の狙いは `keirin.netkeirin_settings.title_template`（DB・設定画面で編集可）、
後半の構造テキストは本モジュールの `SHAPE_TITLES` が持つ。

なぜ構造テキストだけコード側なのか
----------------------------------
構造テキストは「いま実際に何を買っているか」に直結する（例: 7A は本命が割れた
レースだけを通すゲートがあるので "1車抜け" とは書けない）。DB の自由文にすると
選出ロジックを変えたときに文言だけ取り残されて商品説明が事実と食い違う。
`RANK_CONFIGS` と同じく**コード側の単一正本**に置き、ランク一覧の手書き二重管理を作らない。

ラベルの定義
------------
入稿時点で確定している値（`wt_entries` の予測確率・登録脚質・ライン）だけで決まる。
`pred_win_pct` / `pred_top3_pct` はレース内合計が揃っていない生確率なので、
`_build_entry_table()` と同じロジット空間シフトで正規化してから比較する。

| ラベル | 判定 | 実測比率 |
|---|---|---|
| `solo`  | 1着率 1位−2位 ≥ 26pt | 30.4% |
| `duo`   | solo以外 ∧ 3着内率 2位−3位 ≥ 18pt | 19.4% |
| `mixed` | 上記以外 ∧ 1着率差 ≤ 10pt ∧ 3着内率差 ≤ 7pt | 9.9% |
| `line`  | 上記以外 ∧ 軸2車が同一ライン | 18.3% |
| `clash` | 上記以外 ∧ 軸2車が別ライン ∧ 逃げ ≥ 3人 | 8.6% |
| `split` | 上記以外（軸2車が別ライン） | 13.4% |

比率は 2026-07-01 以降・7車以上の 2,913 レースでの実測。閾値は「はっきり言い切れる
レースにだけラベルを付ける」意図で上位・下位30%点に置いている（1着率差の中央値は
17.3pt、3着内率差は12.0pt）。閾値を動かすとラベルの出現比率が変わるだけで、
どのレースを買うかには一切影響しない（表示専用）。

⚠️ **同ライン判定はそのランク自身の軸2車で行う**。7C だけ軸が `axis1_7c`/`axis2_7c`
   で他ランク（`axis1`/`axis2`）と別物なので、呼び出し側が渡す軸を取り違えると
   全レースで line/split が入れ替わる。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# --- ラベル ---------------------------------------------------------------
SHAPE_SOLO = "solo"      # 1着率で1車抜け
SHAPE_DUO = "duo"        # 3着内率で2車抜け
SHAPE_MIXED = "mixed"    # 混戦
SHAPE_LINE = "line"      # 軸2車が同一ライン
SHAPE_SPLIT = "split"    # 軸2車が別ライン
SHAPE_CLASH = "clash"    # 軸2車が別ライン × 先行争い

# --- 閾値（実測の30%点。モジュール docstring の表を参照） -------------------
WIN_GAP_SOLO = 26.0      # 1着率 1位−2位（ポイント）
TOP3_GAP_DUO = 18.0      # 3着内率 2位−3位（ポイント）
WIN_GAP_MIXED = 10.0     # 混戦とみなす 1着率差の上限
TOP3_GAP_MIXED = 7.0     # 混戦とみなす 3着内率差の上限
NIGE_CLASH = 3           # 「先行争い」とみなす逃げ（登録脚質）の人数

# 9車ランクは7車ランクと同じ商品性格なので同じ文言を使う。
# 購入者にとって車立ての区別は不要（2026-08-09 ユーザー判断）。
# 🔴 **同じ文言を2箇所に書かない**ためのエイリアス。ここを消して SHAPE_TITLES に
#    9系を直書きすると、片方だけ直して静かに食い違う（本リポジトリで繰り返し起きた型）。
RANK_ALIASES = {"9S": "7S", "9A": "7A", "9H1": "7H1"}

# 7A は top2_sum q20 ゲートで「本命が割れた」レースだけを通す。したがって
# solo（1車抜け）と同時に立つのは定義上おかしい。ラベルを付けずライン系へ倒す。
SKIP_SOLO_RANKS = {"7A"}

# 指数未算出などでラベルを決められないときの、ランクごとの既定テキスト。
SHAPE_FALLBACK = {
    "7S": "本命堅め",
    "7A": "本命が割れた一戦",
    "7B": "少点数で勝負",
    "7C": "手堅く的中狙い",
    "7SS": "自信の一戦",
    "7H1": "本命バスト警報",
}

# ランク × 構造 のタイトル後半。DB 側の title_template は
# 「自信の二軸｜{shape}」のように前半（狙い）と差し込み位置だけを持つ。
SHAPE_TITLES: dict[str, dict[str, str]] = {
    # 7S/9S — 軸2車が明確に抜けた本命堅め・総流し
    "7S": {
        SHAPE_SOLO: "本命一枚が抜けた",
        SHAPE_DUO: "二枚看板が堅い",
        SHAPE_LINE: "ライン決着を素直に",
        SHAPE_SPLIT: "別線対決を制す",
        SHAPE_CLASH: "先行争いでも軸は不動",
        SHAPE_MIXED: "混戦でも軸は譲らない",
    },
    # 7A/9A — q20ゲートON＝本命が割れた高配当レースだけを厳選（solo は使わない）
    "7A": {
        SHAPE_DUO: "上位2車堅く3着が妙味",
        SHAPE_LINE: "ライン決着に妙味",
        SHAPE_SPLIT: "別線対決に妙味",
        SHAPE_CLASH: "先行争いが妙味を生む",
        SHAPE_MIXED: "混戦の妙味を照らす",
    },
    # 7B — 相手まで絞った少点数構成
    "7B": {
        SHAPE_SOLO: "1番手が明確",
        SHAPE_DUO: "二枚看板を軸に",
        SHAPE_LINE: "ライン決着に絞る",
        SHAPE_SPLIT: "別線対決を絞る",
        SHAPE_CLASH: "先行争いを見切る",
        SHAPE_MIXED: "混戦を絞り込む",
    },
    # 7C — 主力・的中体験枠
    "7C": {
        SHAPE_SOLO: "軸1車が抜けた",
        SHAPE_DUO: "二枚看板で手堅く",
        SHAPE_LINE: "ライン通りに手堅く",
        SHAPE_SPLIT: "別線対決も本線で",
        SHAPE_CLASH: "先行争いでも本線で",
        SHAPE_MIXED: "混戦でも本線を通す",
    },
    # 7SS — entropy不合格 × 軸2車が同一ライン。
    # ⚠️ 選出条件が「軸2車が同一ライン」なので split/clash は**定義上発生しない**。
    #    出たら選出ロジックかライン判定のどちらかが壊れている合図。あえて欠かして
    #    おき、`shape_title_text()` が警告を出せるようにしている。
    "7SS": {
        SHAPE_SOLO: "1番手が抜けた",
        SHAPE_DUO: "二枚看板が堅い",
        SHAPE_LINE: "ライン決着濃厚",
        SHAPE_MIXED: "混戦こそ本線",
    },
    # 7H1/9H1 — 本命バスト型・高配当
    "7H1": {
        SHAPE_SOLO: "抜けた本命を疑う",
        SHAPE_DUO: "二枚看板の一角が崩れる",
        SHAPE_LINE: "ライン決着の盲点",
        SHAPE_SPLIT: "別線対決が荒れを呼ぶ",
        SHAPE_CLASH: "先行争いで波乱を待つ",
        SHAPE_MIXED: "混戦から高配当を",
    },
}


# 見解本文の冒頭に置く「レース見解」。タイトル後半（`SHAPE_TITLES`）と同じ
# ランク×構造で、こちらは1〜2文の文章。
# 🔴 **車番・点数を書かない**。netkeirin は本文の冒頭をプレビュー表示しうるので、
#    ここに買い目が出ると無料で買い目を配ることになる（仕様書 §4-3）。
SHAPE_NOTES: dict[str, dict[str, str]] = {
    "7S": {
        SHAPE_SOLO: "1着率で頭ひとつ抜けた1車がいます。ここを信頼して、まっすぐ当てにいく一戦です。",
        SHAPE_DUO: "3着内率で上位2車がはっきり抜けました。この2車を信頼した組み立てです。",
        SHAPE_LINE: "軸2車が同じライン。並びどおりに決まれば、そのまま的中に届く形です。",
        SHAPE_SPLIT: "軸2車は別ライン。どちらが主導権を握っても拾えるよう構えました。",
        SHAPE_CLASH: "先行を主張できる選手が複数います。展開が乱れても、軸2車の力は落ちないと見ました。",
        SHAPE_MIXED: "全体に力が拮抗した一戦。それでも当方の指数は、この2車を上位に置いています。",
    },
    "7A": {
        SHAPE_DUO: "3着内率で上位2車は堅い一方、3着争いは割れています。ここに配当の伸びしろがあります。",
        SHAPE_LINE: "軸2車が同じライン。並びどおりでも、人気の盲点になりやすい組み合わせと見ました。",
        SHAPE_SPLIT: "軸2車は別ライン。決まり手が読みにくいぶん、配当が跳ねやすい一戦です。",
        SHAPE_CLASH: "先行争いが激しく、展開次第で上位が入れ替わります。荒れれば妙味が出ます。",
        SHAPE_MIXED: "本命が割れた混戦。人気に偏りがないぶん、当たれば配当がついてきます。",
    },
    "7B": {
        SHAPE_SOLO: "1着率で1車が明確に抜けました。ここを起点に、相手を絞り込んでいます。",
        SHAPE_DUO: "3着内率で上位2車が抜けています。この2車を軸に、相手をさらに削りました。",
        SHAPE_LINE: "軸2車が同じライン。並びどおりの決着を厚く見て、相手を絞りました。",
        SHAPE_SPLIT: "軸2車は別ライン。両者の力を認めたうえで、3着候補だけを削り込みました。",
        SHAPE_CLASH: "先行争いが見込まれる一戦。総流しにせず、残り目を絞って構えています。",
        SHAPE_MIXED: "拮抗した一戦ですが、3着候補まで踏み込んで絞り込みました。",
    },
    "7C": {
        SHAPE_SOLO: "1着率で頭ひとつ抜けた1車がいます。素直にここから、まず当てにいきます。",
        SHAPE_DUO: "3着内率で上位2車が抜けています。無理をせず、この2車から的中を取りにいきます。",
        SHAPE_LINE: "軸2車が同じライン。並びどおりの決着を本線に据えました。",
        SHAPE_SPLIT: "軸2車は別ライン。どちらが前に出ても拾える形で本線を組みました。",
        SHAPE_CLASH: "先行争いが見込まれますが、荒れ待ちはしません。本線どおりに当てにいきます。",
        SHAPE_MIXED: "拮抗した一戦。大穴は追わず、指数上位から的中を積みにいきます。",
    },
    "7SS": {
        SHAPE_SOLO: "1着率で1車が抜け、しかも軸2車は同じライン。本線どおりの決着を厚く見ています。",
        SHAPE_DUO: "3着内率の上位2車が同じライン。並びがそのまま結果になりやすい形です。",
        SHAPE_LINE: "軸2車は同じライン。ライン決着を本線に据えた、当方が信頼を置く一戦です。",
        SHAPE_MIXED: "力は拮抗していますが、軸2車が同じライン。並びの利を買います。",
    },
    "7H1": {
        SHAPE_SOLO: "頭ひとつ抜けた1車がいます。当方はこれが4着以下に沈むと読みました。",
        SHAPE_DUO: "上位2車が抜けて見えるレースですが、その一角が崩れると読みました。",
        SHAPE_LINE: "人気の中心が並びで固まっています。そこが崩れたときの配当を狙います。",
        SHAPE_SPLIT: "有力どころが別ラインに分かれ、つぶし合いになりやすい一戦と見ました。",
        SHAPE_CLASH: "先行争いが激しく、共倒れの目があります。波乱を待つ組み立てです。",
        SHAPE_MIXED: "力が拮抗し、決着の形が定まりません。高配当が出やすい条件と見ました。",
    },
}

# 判定不能時の既定（ランクの性格だけを述べる。構造には触れない）。
SHAPE_NOTE_FALLBACK = {
    "7S": "当方の指数で軸2車がはっきり抜けた、当てにいく一戦です。",
    "7A": "本命が割れ、相手次第で配当が伸びる一戦を選びました。",
    "7B": "軸2車が明確に絞り込めたので、相手も絞って構えました。",
    "7C": "大穴は狙わず、まず当てることを優先した一戦です。",
    "7SS": "軸2車が同じライン。並びの利を買う一戦です。",
    "7H1": "抜けた本命が沈むと読んだ、高配当ねらいの一戦です。",
}

# 賭け金の配分の説明。**実際に入稿する買い目から導く**（`_stake_note_for`）。
# 🔴 実態一致（仕様書 §4-6）。ダッチ配分は朝オッズが全点そろわないと均等へ
#    フォールバックする（欠損は約半数）ため、「オッズに応じて配分しています」を
#    無条件に書くと**半分は嘘になる**。
STAKE_NOTE_EQUAL = "金額は各買い目に均等に置いています。"
STAKE_NOTE_TILTED = (
    "金額は均等ではなく、当方が想定する発走時オッズに応じて配分しています。"
    "配当が低くなりやすい買い目に厚く、高くなりやすい買い目に薄く置いています。"
)
# ガミ抑制を売り文句にできるのは 7H1/9H1 だけ（仕様書 §1・§4-6）。
STAKE_NOTE_TILTED_HIGHPAY = (
    STAKE_NOTE_TILTED
    + "どの目で決まっても払戻が投資を上回ることを狙い、当たったときの"
      "取りこぼし（ガミ）を抑える組み立てです。"
)
GAMI_CLAIM_RANKS = {"7H1"}   # エイリアス解決後。9H1 は 7H1 に寄る


def stake_note_text(rank_key: str, tilted: bool) -> str:
    """賭け金配分の説明文を返す。"""
    if not tilted:
        return STAKE_NOTE_EQUAL
    rank = RANK_ALIASES.get(rank_key, rank_key)
    return STAKE_NOTE_TILTED_HIGHPAY if rank in GAMI_CLAIM_RANKS else STAKE_NOTE_TILTED


# ---------------------------------------------------------------------------
# 確率の正規化（レース内合計を揃える）
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """ロジスティック関数。"""
    return 1 / (1 + math.exp(-x))


def logit(p: float) -> float:
    """ロジット。0/1 は eps でクリップする。"""
    eps = 1e-6
    c = min(max(p, eps), 1 - eps)
    return math.log(c / (1 - c))


def solve_logit_shift(probs: list[float], target: float) -> float:
    """ロジット空間の一律シフト量を二分探索する（合計を target に合わせる）。"""
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        total = sum(sigmoid(logit(p) + mid) for p in probs)
        if total < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _normalized(probs: list[float], target: float) -> list[float] | None:
    """生確率をレース内合計 target に正規化する。全件0/欠損なら None。"""
    if not probs or not any(p > 0 for p in probs):
        return None
    shift = solve_logit_shift(probs, target)
    return [sigmoid(logit(p) + shift) for p in probs]


# ---------------------------------------------------------------------------
# 構造ラベルの判定
# ---------------------------------------------------------------------------

def _strength_label(entries: Sequence[Mapping]) -> str | None:
    """予測確率の抜け具合から solo / duo / mixed を返す。該当なしは None。"""
    n = len(entries)
    win = _normalized([float(e["pred_win_pct"] or 0) / 100 for e in entries], 1.0)
    top3 = _normalized([float(e["pred_top3_pct"] or 0) / 100 for e in entries],
                       min(n, 3) * 1.0)
    if win is None or top3 is None:
        return None
    w = sorted(win, reverse=True)
    t = sorted(top3, reverse=True)
    if len(w) < 2 or len(t) < 3:
        return None
    win_gap = (w[0] - w[1]) * 100
    top3_gap = (t[1] - t[2]) * 100
    if win_gap >= WIN_GAP_SOLO:
        return SHAPE_SOLO
    if top3_gap >= TOP3_GAP_DUO:
        return SHAPE_DUO
    if win_gap <= WIN_GAP_MIXED and top3_gap <= TOP3_GAP_MIXED:
        return SHAPE_MIXED
    return None


def _line_label(entries: Sequence[Mapping], axis1: int, axis2: int) -> str:
    """軸2車のライン関係と先行人数から line / split / clash を返す。

    `line_group` が NULL の車は単騎。単騎どうしは「同じライン」ではないので、
    両方 NULL でも別ライン扱いにする（None == None を同ライン判定にしない）。
    """
    by_frame = {int(e["frame_no"]): e for e in entries if e.get("frame_no") is not None}
    a1 = by_frame.get(int(axis1))
    a2 = by_frame.get(int(axis2))
    g1 = a1.get("line_group") if a1 else None
    g2 = a2.get("line_group") if a2 else None
    if g1 is not None and g1 == g2:
        return SHAPE_LINE
    nige = sum(1 for e in entries if e.get("style") == "逃")
    return SHAPE_CLASH if nige >= NIGE_CLASH else SHAPE_SPLIT


def classify_shape(rank_key: str, entries: Sequence[Mapping],
                   axis1: int, axis2: int) -> str | None:
    """レース構造ラベルを返す。判定不能（指数未算出等）は None。

    entries は `frame_no` / `pred_win_pct` / `pred_top3_pct` / `style` /
    `line_group` を持つ dict の列（車番順である必要はない）。
    """
    if not entries:
        return None
    rank = RANK_ALIASES.get(rank_key, rank_key)
    label = _strength_label(entries)
    if label == SHAPE_SOLO and rank in SKIP_SOLO_RANKS:
        label = None
    if label is not None:
        return label
    # solo/duo/mixed のいずれでもない「中位」レースをライン構成で3分割する。
    # ここは予測確率を使わないので、指数未算出でも判定できる。
    return _line_label(entries, axis1, axis2)


def _lookup(rank_key: str, shape: str | None, table_by_rank: dict,
            fallback_by_rank: dict, what: str) -> tuple[str, str | None]:
    """ランク×構造の文言を引く。

    戻り値は `(テキスト, 警告メッセージ or None)`。**無言でフォールバックしない**：
    定義上あり得ない組み合わせ（7SS × split/clash 等）が来たら呼び出し側が
    ログへ出せるように警告文を一緒に返す。
    """
    rank = RANK_ALIASES.get(rank_key, rank_key)
    table = table_by_rank.get(rank)
    if table is None:
        return fallback_by_rank.get(rank, ""), f"{rank_key}: {what}が未定義のランク"
    if shape is None:
        return fallback_by_rank.get(rank, ""), None
    text = table.get(shape)
    if text is None:
        return (fallback_by_rank.get(rank, ""),
                f"{rank_key}: 構造ラベル {shape} は定義上発生しないはず（選出条件を確認）")
    return text, None


def shape_title_text(rank_key: str, shape: str | None) -> tuple[str, str | None]:
    """タイトル後半のテキストを返す。"""
    return _lookup(rank_key, shape, SHAPE_TITLES, SHAPE_FALLBACK, "構造タイトル")


def shape_note_text(rank_key: str, shape: str | None) -> tuple[str, str | None]:
    """見解本文の冒頭に置くレース見解（1〜2文）を返す。"""
    return _lookup(rank_key, shape, SHAPE_NOTES, SHAPE_NOTE_FALLBACK, "構造見解")
