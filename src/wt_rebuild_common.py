"""rebuild_{7s,7a,9s,9a}_walkforward_pg.py 共通のガード処理（2026-08-01・F-4対応）。

## 背景

2026-08-01、日付が月初(8/1)に変わった直後 `scripts/rebuild_7s_walkforward_pg.py`
が `FileNotFoundError: data/models/lgbm_wt_eval_m2608.pkl` で失敗した。
`src/wt_vintage_config.py::monthly_windows()` は `date.today()` 基準で当月の窓を
生成するが、その月の凍結vintageモデル（`lgbm_wt_eval_mYYMM`/`lgbm_wt_win_mYYMM`）
は自動生成されないため、月が替わった瞬間に「まだ存在しないモデル」が要求され
落ちる。

`rebuild_*_walkforward_pg.py` は全期間・全窓を計算してから最後に一括で
wipe(DELETE)→insertする設計のため、最後の窓（最新月）で例外が発生すると、
それ以前の全期間の計算（実測約40分規模）が丸ごと失われていた
（計算結果がどこにもpersistされないまま例外でプロセスが落ちるため）。

加えて、wipeとinsertが別々の接続・別トランザクションで実行されていたため、
wipe成功後にinsertが失敗すると picks_history が空のまま残るリスクもあった
（2026-08-01実害・バックアップから復旧）。

## 本モジュールの役割

4本のrebuild_*_walkforward_pg.pyから共通importする（`src/wt_vintage_config.py`
と同じ「単一の正本」方針・将来どれか1つだけ改修されて食い違うリスクを避ける）。

1. `split_by_model_availability()`: 重い計算（build_rows）を始める前に、
   全窓のvintageモデルpklが存在するかを検証する。不足があれば
   呼び出し側は計算を一切開始せず即座に終了できる。
2. `notify_discord_warning()`: モデル不足・0件wipeスキップ等の異常を
   Discord `system` チャンネルへ通知する（`src/notify/discord.py::send` は
   channel引数必須のため明示指定）。
3. `rebuild_pg_atomic()`: wipe(DELETE)→insertを単一トランザクションに
   まとめて実行する。`src/database.py::get_connection()` の
   コンテキストマネージャは正常終了時に一括commit・例外発生時に自動rollback
   する設計のため、この関数内で個別に `conn.commit()` を呼ばない限り
   wipeとinsertはアトミックになる。挿入対象行が0件の窓はwipe自体をスキップし、
   置き換えデータが無いのに削除だけ行って picks_history を空にする事故を防ぐ。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.database import get_connection
from src.models.trainer import MODEL_DIR
from src.notify.discord import send as _discord_send
from src.strategy_wt import THREE_HEAD_AXIS_SINCE
from src.wt_vintage_config import bad_model_name, favbust_model_name

# (date_from, date_to, eval_model_name, win_model_name)
Window = tuple[str, str, str, str]
# (window, 不足モデル名のリスト)
MissingWindow = tuple[Window, list[str]]


def split_by_model_availability(
    windows: list[Window], require_bad: bool = False,
    require_favbust: bool = False,
) -> tuple[list[Window], list[MissingWindow]]:
    """各窓のeval/winモデルpklが存在するか事前チェックする。

    build_rows()（学習済みモデルの読み込み + 全レース分の推論・選出ロジック
    実行）は窓によっては数十分かかりうるため、これを始める前に軽量な
    ファイル存在チェックだけで不足を検出できるようにする。

    require_bad: 3ヘッド軸選定（軸2 = argmax z(3着内率) − 0.3×z(大敗率)）で
      再構築する場合に True。大敗モデルの vintage（lgbm_wt_bad_mYYMM）も
      存在チェックの対象に加える。**既定は False**（他ランクの rebuild は
      まだ旧2ヘッド軸のため、要求すると全窓が不足扱いになってしまう）。

    Returns:
        (available, missing)
        available: 必要なモデルpklが揃っている窓のリスト（元の順序を維持）。
        missing:   [(window, [不足モデル名, ...]), ...]（元の順序を維持）。
    """
    available: list[Window] = []
    missing: list[MissingWindow] = []
    for window in windows:
        _, _, eval_model, win_model = window
        required = [eval_model, win_model]
        if require_bad:
            required.append(bad_model_name(eval_model))
        if require_favbust:
            required.append(favbust_model_name(eval_model))
        missing_names = [
            name for name in required
            if not (MODEL_DIR / f"{name}.pkl").exists()
        ]
        if missing_names:
            missing.append((window, missing_names))
        else:
            available.append(window)
    return available, missing


def format_missing_report(rank_label: str, missing: list[MissingWindow]) -> str:
    """不足モデルのレポート文字列を組み立てる（ログ出力・Discord通知の両方で使う）。"""
    lines = [f"[{rank_label}] vintageモデル不足: {len(missing)}窓"]
    for (date_from, date_to, eval_model, win_model), names in missing:
        lines.append(f"  {date_from}〜{date_to} (eval={eval_model} win={win_model}): 不足={names}")
    return "\n".join(lines)


def notify_discord_warning(content: str) -> None:
    """Discord `system` チャンネルへ警告を送信する。失敗してもrebuild自体は止めない
    （通知経路の不調でバックフィル処理そのものを巻き込まないため）。
    """
    try:
        ok = _discord_send(content, channel="system")
        if not ok:
            print(f"[wt_rebuild_common] Discord通知に失敗しました。内容: {content}")
    except Exception as exc:  # noqa: BLE001 - 通知失敗はrebuild続行を妨げない
        print(f"[wt_rebuild_common] Discord通知で例外が発生しました: {exc}\n内容: {content}")


_ZERO_ROW_STATE = Path(__file__).resolve().parent.parent / "data" / "logs" / "zero_row_notified.json"
"""「0件で wipe を見送った」通知の既報状態。`data/logs/` は .gitignore 済み。"""


def _zero_row_should_notify(rank_label: str, per_window_rows: list) -> bool:
    """0件見送りを Discord へ通知すべきか。**同じランク・同じ月は1回だけ**。

    🔴 ここで DB を見てはいけない。「0件のときは picks_history に一切触れない」は
       `rebuild_pg_atomic` の明示的な安全性質で、既存行を数えるだけでも破れる
       （test_rebuild_pg_atomic_zero_total_rows_never_touches_db）。

    ⚠️ 窓の終端は毎日進む（tail は前日まで）ので、窓の**開始**を鍵にする。
       終端まで鍵に含めると毎日「別の状況」と見なされ、抑制の意味が無くなる。

    ⚠️ 状態ファイルが読めない/書けないときは **通知する側に倒す**（fail-open）。
       黙らせる方に倒すと、本当の異常まで気づけなくなる。
    """
    anchor = per_window_rows[0][0] if per_window_rows else "-"
    key = f"{rank_label}:{anchor}"
    try:
        state = {}
        if _ZERO_ROW_STATE.exists():
            state = json.loads(_ZERO_ROW_STATE.read_text(encoding="utf-8"))
        if state.get(key):
            return False      # 同じランク・同じ月は既報
        state[key] = True
        _ZERO_ROW_STATE.parent.mkdir(parents=True, exist_ok=True)
        _ZERO_ROW_STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[wt_rebuild_common] 0件通知の既報判定に失敗（通知します）: {exc}")
        return True


# 2026-08-06: 7H1（唯一の2券種ランク）のため trifecta_payout を追加した。
# 三連複のみのランク（7SS/7S/7A/7B/9S/9A）は rows に持たないので、挿入時に
# 0 を補う（下記 _row_defaults）。gate_label も 7H1 は持たないため同様に補う。
_INSERT_SQL = (
    "INSERT OR REPLACE INTO picks_history "
    "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,"
    " trio_payout,trifecta_payout,bet_amount,route,miwokuri,gate_label) "
    "VALUES (:race_date,:race_key,:rank,:pred_combo,:n_combos,:hit,"
    " :payout,:trio_payout,:trifecta_payout,:bet_amount,'wt',:miwokuri,:gate_label)"
)

#: ランクごとに持たない列を補う既定値（**行側の値が常に優先される**）。
_ROW_DEFAULTS = {"trifecta_payout": 0, "gate_label": None}


# 3ヘッド軸選定で選ばれた live 記録を、旧軸の再構築で上書きしてはいけないランク。
# 9車(RANK_9S/9A)は3ヘッドを適用していないため対象外。
# 3ヘッド軸（2026-08-04〜）で live が動いている＝旧軸で塗り潰してはいけないランク。
# ⚠️ **7車のランクを新設したら必ずここへ足すこと。** 9車（RANK_9S/RANK_9A）は
#    掃引で窓別に符号が反転したため3ヘッドを採用しておらず、対象外で正しい。
#    2026-08-05 に 7SS を新設した際ここへの追加が漏れており、旧軸での tail 再構築が
#    無警告で通る状態だった（2026-08-06 是正。同種の「ランク一覧の手書き二重管理」は
#    netkeirin_submit_wt.py の RANK_ORDER でも同日に事故を起こしている）。
_THREE_HEAD_RANKS = frozenset({"RANK_7SS", "RANK_7S", "RANK_7A", "RANK_7B", "RANK_7H1"})


def rebuild_pg_atomic(
    rank_label: str,
    delete_cond_sql: str,
    per_window_rows: list[tuple[str, str, list[dict]]],
    dry_run: bool,
    allow_legacy_axis: bool = False,
    axis_is_three_head: bool = False,
) -> None:
    """wipe(DELETE)→insertを単一トランザクションにまとめて実行する。

    Args:
        rank_label: ログ/Discord通知のプレフィックス（例: "RANK_7S"）。
        delete_cond_sql: DELETE/SELECT COUNT の WHERE 句。
            `race_date BETWEEN ? AND ?` を含み、rank/race_key条件は
            呼び出し側で埋め込み済みの完全な文字列を渡す
            （例: "rank='RANK_7S' AND race_key LIKE '%#7S' AND
            race_date BETWEEN ? AND ?"）。
        per_window_rows: [(date_from, date_to, rows), ...]。
            rows が空の窓は個別にwipeをスキップする（警告ログのみ・
            置き換えデータが無いのに既存行を消してしまう事故を防ぐ）。
        dry_run: True の場合、実際のDELETE/INSERTは行わず件数のみ表示する。

    全窓が0件（＝挿入対象が1件も無い）場合は `get_connection()` 自体を
    呼ばずに即座にreturnする（DBへ一切触れない・安全側）。この状態は
    通常発生しないはずなので Discord へも警告する。

    途中で例外が発生した場合、`get_connection()` のコンテキストマネージャが
    自動的に `rollback()` するため、既に実行済みのDELETE/INSERTを含め
    このトランザクション内の全変更が取り消される
    （wipeだけが確定してinsertが失われる事故の防止）。
    """
    # --- 3ヘッド期間を旧軸で塗り潰す事故の防止（2026-08-04） ---
    # backfill_7*_rank_wt.py の build_rows() は rank_7s_select_axis(win, top3) を
    # bad_probs 無しで呼ぶ＝旧軸。THREE_HEAD_AXIS_SINCE 以降を DELETE→INSERT すると
    # live の3ヘッド記録が静かに消える（S1が第4経路で自動再生成されていた事故と同型）。
    #
    # 【2026-08-05】呼び出し側が **月次vintageの大敗モデルを使って3ヘッド軸で**
    # 再構築する場合は、塗り潰しではなく正しい更新なのでガードを外す
    # （`axis_is_three_head=True`）。現時点で該当するのは RANK_7B のみ。
    # ⚠️ `allow_legacy_axis`（旧軸のまま強行）とは意味が違う。前者は「3ヘッドで
    # 正しく作り直す」、後者は「旧軸で塗り潰すと承知の上で強行する」。
    if rank_label in _THREE_HEAD_RANKS and not allow_legacy_axis and not axis_is_three_head:
        overlap = sorted(dt for _, dt, rows in per_window_rows
                         if rows and dt >= THREE_HEAD_AXIS_SINCE)
        if overlap:
            msg = (
                f"[{rank_label}] 再構築の対象に 3ヘッド軸選定の適用期間"
                f"（{THREE_HEAD_AXIS_SINCE} 以降・最終窓 {overlap[-1]}）が含まれています。"
                f"このスクリプトの軸選定は bad_probs を渡さない**旧軸**のため、"
                f"実行すると live の3ヘッド記録を上書きして消します。中止しました。"
                f"（意図して旧軸で塗り直すなら --allow-legacy-axis を明示してください）"
            )
            print(msg)
            notify_discord_warning(f"🚨 **{msg}**")
            raise SystemExit(1)

    total_rows = sum(len(rows) for _, _, rows in per_window_rows)
    if total_rows == 0:
        msg = (
            f"[{rank_label}] 挿入対象の行が0件のため、wipe(DELETE)を一切実行せず"
            f"終了します（安全側・picks_historyが空のまま残る事故を防止）。"
        )
        print(msg)
        # 🔴 **挙動（wipeしない）は変えない。通知だけを絞る。**
        #    9S のように候補がほぼ出ないランクは毎朝ここへ来るため、
        #    毎日同じ内容を Discord へ流すと警告全体が無視されるようになる。
        #    同じランク・同じ月は1回だけ通知する（ログには毎回残る）。
        if _zero_row_should_notify(rank_label, per_window_rows):
            notify_discord_warning(f"⚠️ **{msg}**")
        return

    if dry_run:
        with get_connection() as conn:
            for date_from, date_to, rows in per_window_rows:
                if not rows:
                    print(f"[{rank_label}] {date_from}〜{date_to}: 0件のためwipe対象外（dry-run）")
                    continue
                n = conn.execute(
                    f"SELECT COUNT(*) FROM picks_history WHERE {delete_cond_sql}",
                    (date_from, date_to),
                ).fetchone()[0]
                print(f"[{rank_label}] dry-run: {date_from}〜{date_to} 既存{n}件 → "
                      f"削除予定・挿入予定{len(rows)}件")
        print(f"[{rank_label}] DRY RUN（書き込みなし・合計挿入予定{total_rows}件）")
        return

    with get_connection() as conn:
        n_windows_written = 0
        for date_from, date_to, rows in per_window_rows:
            if not rows:
                print(f"[{rank_label}] {date_from}〜{date_to}: 0件のためwipeスキップ（警告）")
                continue
            n_existing = conn.execute(
                f"SELECT COUNT(*) FROM picks_history WHERE {delete_cond_sql}",
                (date_from, date_to),
            ).fetchone()[0]
            print(f"[{rank_label}] {date_from}〜{date_to}: 既存{n_existing}件 → 削除")
            conn.execute(
                f"DELETE FROM picks_history WHERE {delete_cond_sql}",
                (date_from, date_to),
            )
            rows_ins = [{**_ROW_DEFAULTS, **r, "miwokuri": False} for r in rows]
            conn.executemany(_INSERT_SQL, rows_ins)
            print(f"[{rank_label}] {date_from}〜{date_to}: {len(rows)}件 挿入（未コミット）")
            n_windows_written += 1
        # ここでは commit() を呼ばない。get_connection() の
        # コンテキストマネージャが `with` ブロック正常終了時に一括commitし、
        # 例外発生時は自動rollbackする（src/database.py参照）。
    print(f"[{rank_label}] 合計{total_rows}件（{n_windows_written}窓）書き込み完了"
          f"（VPS PG・単一トランザクション）")
