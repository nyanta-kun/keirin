"""netkeirin の商品タイトル・見解本文をレース構造に応じた文面へ差し替える（2026-08-09）。

## なぜスクリプトが要るのか

タイトル／本文は `keirin.netkeirin_settings`（DB）が正で、**行があればコードの既定文
（`_DEFAULT_TITLE_TEMPLATE` / `_DEFAULT_COMMENT_TEMPLATE` / `cfg["default_comment"]`）は
絶対に使われない**。コードだけ直しても商品は「本日の二軸」等のまま残る。

## 🔴 実行タイミング

**コードをデプロイした直後に実行する。**
先に実行すると、`{shape}` 等を解決できない旧コードが本番で走り、商品タイトル・本文に
`{shape}` という文字列がそのまま出る（`_apply_template` は未定義の `{...}` を
例外にせず素通しする仕様のため、**エラーにならず静かに壊れる**）。

    # 1. keirin の PR を master へマージ（= VPS へ自動デプロイ）
    # 2. その直後に
    PYTHONPATH=. .venv/bin/python scripts/update_netkeirin_templates.py --apply

既定は dry-run（差分表示のみ・書き込みなし）。

## 変更の中身（タイトル）

- 会場・R番号・日付を**タイトルから外す**。netkeirin の一覧では別欄に出ており重複するため
  （2026-08-09 ユーザー判断）。
- 後半に `{shape}`（レース構造の見立て）を差し込む。文言は `src/race_shape.py` が正本。
- 7車/9車の区別を**書かない**（購入者には不要）。9S/9A/9H1 は 7S/7A/7H1 と同一文言で、
  `race_shape.RANK_ALIASES` が実際の文言を1箇所に寄せている。
- **9H1 の行を新規作成する**。これまで行が無く、`_is_enabled()` の fail-open で
  入稿はされていたがタイトルだけ既定値（`{venue}{race_no}R 二軸探偵`）に落ちていた。

## 変更の中身（見解本文）

- 冒頭を `{shape_note}`（レース構造の見解1〜2文・**車番を含まない**）にする。
  netkeirin は本文の先頭をプレビュー表示しうるので、従来の
  「本レースで照らし出した二軸は、◎1番・○2番です。」を先頭に置くと
  **無料で買い目を配る**ことになる（仕様書 §4-3）。◎○ は【二軸】節まで下げた。
- 配分の説明を `{stake_note}` にする。ダッチ／傾斜配分は朝オッズが揃わないと均等へ
  フォールバックする（欠損は約半数）ため、「オッズに応じて配分しています」を固定文で
  書くと**半分のレースで嘘になる**。実際に入稿する買い目から導く（仕様書 §4-6）。
- ランクごとに狙いを書き分ける。従来は 7S/7A/7C/7SS/9S/9A が
  「軸2車に自信が持てるレースだけを厳選」という**同一文**で、7C（主力・的中体験枠）や
  7SS（ライン本線）の実態と合っていなかった。
- 7B から「準決勝」を落とす（購入者に伝わりにくい・2026-08-09 ユーザー判断）。
- 集客導線（プロフィール・「ウマい！」お気に入り）を全ランクへ追加（仕様書 §4-5）。

## 対象外

`S1` / `9SS` は `enabled=false`（全廃済み）なので触らない。`_global` はランクではない。
旧 `update_netkeirin_comment_templates.py` は本スクリプトに置き換わった（実行不要）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.database import get_connection  # noqa: E402
from src.race_shape import RANK_ALIASES, SHAPE_NOTES, SHAPE_TITLES  # noqa: E402

# タイトル前半（狙い＝ランクの性格）。後半は `{shape}` で差し込む。
# 🔴 **前半はランク内で一定にすること**。構造ごとに前半を変えたくなったら、それは
#    前半ではなく `race_shape.SHAPE_TITLES` 側に持たせるべき違い。
TITLE_TEMPLATES: dict[str, str] = {
    "7S": "自信の二軸｜{shape}",
    "9S": "自信の二軸｜{shape}",
    "7A": "厳選の二軸｜{shape}",
    "9A": "厳選の二軸｜{shape}",
    "7B": "相手を絞った二軸｜{shape}",
    "7C": "本線の二軸｜{shape}",
    "7SS": "ライン本線の二軸｜{shape}",
    "7H1": "穴狙いの二軸｜{shape}",
    "9H1": "穴狙いの二軸｜{shape}",
}


# --- 見解本文 -------------------------------------------------------------
# 全ランク共通の後半。入稿時に出走選手の1着率・3着内率テーブルが本文の**末尾へ
# 自動追記**される（`_build_entry_table`）ので、【参考データ】は必ず最後に置く。
#
# ⚠️ **【予想者より】（実績の宣伝・お気に入り登録の依頼）は 2026-08-09 に
#    ユーザー指示で全ランクから削除した。復活させないこと。**
#    以前は「集客導線は【参考データ】より前に置く」としてここに挟んでいた。
_TAIL = (
    "\n\n【ご購入にあたって】\n"
    "レース直前の実際のオッズをご自身でご確認いただき、必要に応じて配分を"
    "調整いただくと精度が上がります。\n\n"
    "【参考データ】\n"
    "出走選手全員の1着率・3着内率です。三連単で購入される際の着順・買い目の"
    "参考にご活用ください。"
)


def _body() -> str:
    """見解本文を組み立てる。

    🔴 **冒頭に車番を書かない**。netkeirin は本文の先頭をプレビュー表示しうるので、
       ここに軸2車を出すと無料で買い目を配ることになる（仕様書 §4-3）。
       ◎○ の明示は【二軸】節まで下げる。

    ⚠️ **【この買い目について】（ランク別の狙いの説明＋`{stake_note}`）は
       2026-08-09 にユーザー指示で全ランクから削除した。復活させないこと。**
       この結果、**全ランクの見解本文は同一**になる（狙いの差はタイトルの
       `{shape}` と、商品に表示される買い目そのもので伝える）。
       `{stake_note}` も併せて消えたため、配分方式は本文では説明しない。
    """
    return (
        "{shape_note}\n\n"
        "【二軸】\n"
        "本レースで照らし出した二軸は、◎{axis1}番・○{axis2}番です。"
    ) + _TAIL


# ⚠️ 全ランクが**同一の本文**になる（2026-08-09・上記 `_body` 参照）。
#    ランク別の「狙いの説明」は【この買い目について】ごと削除したため、
#    ここでランクごとに文面を分ける余地は無い。dict の形を保っているのは
#    `netkeirin_settings` が rank_key ごとに行を持つため。
#    ランクを増減したら TITLE_TEMPLATES と揃っているか `_check_consistency` が見る。
COMMENT_TEMPLATES: dict[str, str] = {
    rank: _body() for rank in ("7S", "9S", "7A", "9A", "7B", "7C", "7SS", "7H1", "9H1")
}


def _check_consistency() -> list[str]:
    """テンプレと `race_shape` の食い違いを検出する（実行前の自己検査）。

    ランク一覧の二重管理は本リポジトリで繰り返し事故を起こしているので、
    書き込む前に必ず突き合わせる。
    """
    problems = []
    for rank, tpl in TITLE_TEMPLATES.items():
        if "{shape}" not in tpl:
            problems.append(f"{rank}: タイトルに {{shape}} が無い")
        base = RANK_ALIASES.get(rank, rank)
        if base not in SHAPE_TITLES:
            problems.append(f"{rank}: race_shape.SHAPE_TITLES に {base} が無い")
        if base not in SHAPE_NOTES:
            problems.append(f"{rank}: race_shape.SHAPE_NOTES に {base} が無い")
    if set(TITLE_TEMPLATES) != set(COMMENT_TEMPLATES):
        problems.append("TITLE_TEMPLATES と COMMENT_TEMPLATES のランクが揃っていない")
    for rank, tpl in COMMENT_TEMPLATES.items():
        # `{stake_note}` は 2026-08-09 に【この買い目について】ごと削除したので
        # ここでは要求しない。復活防止は tests/test_race_shape.py が担う。
        if "{shape_note}" not in tpl:
            problems.append(f"{rank}: 見解本文に {{shape_note}} が無い")
    for rank in SHAPE_TITLES:
        if rank not in TITLE_TEMPLATES:
            problems.append(f"{rank}: SHAPE_TITLES にあるが TITLE_TEMPLATES に無い")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="実際に書き込む（既定はdry-run）")
    args = ap.parse_args()

    problems = _check_consistency()
    if problems:
        for p in problems:
            print(f"[NG] {p}", file=sys.stderr)
        return 1

    changed = 0
    with get_connection() as conn:
        rows = {r["rank_key"]: dict(r) for r in conn.execute(
            "SELECT rank_key, enabled, title_template, comment_template "
            "FROM netkeirin_settings").fetchall()}
        for rank, new_title in TITLE_TEMPLATES.items():
            new_comment = COMMENT_TEMPLATES[rank]
            cur = rows.get(rank)
            old_title = cur["title_template"] if cur else None
            old_comment = cur["comment_template"] if cur else None
            if old_title == new_title and old_comment == new_comment:
                print(f"[skip] {rank}: 変更なし")
                continue
            changed += 1
            if cur is None:
                print(f"[新規] {rank}: 行を作成")
            else:
                if old_title != new_title:
                    print(f"[更新] {rank} タイトル: {old_title!r} -> {new_title!r}")
                if old_comment != new_comment:
                    print(f"[更新] {rank} 見解本文: 差し替え")
            if not args.apply:
                continue
            if cur is None:
                # 行が無いランクは fail-open で入稿対象なので enabled を真で作る
                # （既存の挙動を変えない）。
                # 🔴 **bool を渡すこと**。`src/database.py` の SQLite 用 DDL は
                #    `enabled INTEGER` だが、本番 PostgreSQL の列は boolean で、
                #    1 を渡すと `DatatypeMismatch: column "enabled" is of type
                #    boolean but expression is of type integer` で落ちる
                #    （2026-08-09 に実際に踏んだ。UPDATE は通るので、行を新規作成する
                #    ときだけ出る＝dry-run でも気づけない）。
                conn.execute(
                    "INSERT INTO netkeirin_settings "
                    "(rank_key, enabled, title_template, comment_template) "
                    "VALUES (?, ?, ?, ?)", (rank, True, new_title, new_comment))
            else:
                conn.execute(
                    "UPDATE netkeirin_settings SET title_template = ?, "
                    "comment_template = ?, updated_at = datetime('now') "
                    "WHERE rank_key = ?", (new_title, new_comment, rank))

    print(f"\n{'書き込み' if args.apply else 'dry-run'}: 対象 {changed} 件")
    if not args.apply and changed:
        print("実際に反映するには --apply を付けて再実行してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
