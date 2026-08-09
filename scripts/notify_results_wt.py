"""winticket 成績通知＋picks_history保存（7+車 SS=三連複）

wave_picks_wt_{date}.txt の公開買い目と prerace_decisions を、winticket の確定結果
(wt_entries.finish_order) と wt_odds(三連複/三連単) で採点し、Discord通知＋picks_history に保存する。
欠車(finish_order=0/NULL)は着外として除外。公開した買い目のみ採点（再導出しない）。

ランク体系（2026-07-23〜: 現行は S1/S7 の2ペーパーランク。S2/S3は2026-07-21全廃・
  2026-07-23コード削除済み。行は picks_history_u_archive / picks_history_m_archive へ退避済み）:
  S1(#7S1) = win軸1着固定×3着内モデル相手2車の三連単2点流し（内部rank SEVEN_S1・
    2026-07-19導入）
    ※ ペーパートレード検証中（実際の賭けなし）。正本は prerace_decisions の {rk}#S1。
      他ランクとの重複排除はない（独立戦略）。
  S7(#7S) = 単勝×複勝指数トップ3重なり軸×波乱度選出（内部rank RANK_7S・
    2026-07-21導入）三連複2軸総流し5点（オッズ下限なし）
    ※ ペーパートレード検証中（実際の賭けなし）。ヘッダー合計には算入する
      （kiseki Webサマリーのトップライン=SEVEN_S1+RANK_7Sと揃える）。
      正本は prerace_decisions の {rk}#S7。他ランクとの重複排除はない（独立戦略）。
      軸選定・当日上位15レースの選出は朝の候補生成（wave-picks-wt）時点で確定済み。
  旧A(#7A) = ◎一致×波乱×別ライン先頭軸の二連単
    ※ 正規プロトコル不合格のため 2026-07-17 全廃（行は picks_history_a_archive へ退避）
  旧S1(#6S1) = 6車三連単 m1→m2→{m3,m4}
    ※ 正規プロトコル不合格のため 2026-07-17 全廃（行は picks_history_r_archive へ退避）
  旧S1(#7R) = 三連複 レース単位 min(全目)≥7 全目購入（内部rank 7PLUS_R・旧称SS）
    ※ 2026-07-16 全廃（行は picks_history_r_archive へ退避・過去日再採点互換のみ残置）
  S/S+(#7ST) = 三連単 1着固定F（7PLUS_ST/STP）
    ※ 優位性なしのため 2026-07-15 に全廃（過去分も無効。採点・集計・DBから除外）
  旧SS(#7SS)/旧S(#7S) = 買い目カット方式（廃止済み・採点対象外）

また candidates.json にあり購入されなかった候補レースを miwokuri=True で保存する。
"""
import json
import os
import sys
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.discord import send
from src.evaluation.backtest_wt import _load_payouts_wt
from src.database import get_connection
from src.submitted_stakes import resolve_payout
from src.strategy_wt import (
    CURRENT_PAPER_RANKS, ABOLISHED_PAPER_RANKS, rank_7c_unit_stake,
    RANK_7S_STAKE, RANK_7A_STAKE, RANK_7B_STAKE, RANK_9S_STAKE, RANK_9A_STAKE,
)

# 集計対象ランクの単一正本は src/strategy_wt.py の CURRENT_PAPER_RANKS /
# ABOLISHED_PAPER_RANKS を参照する（2026-07-31 再発防止・是正タスク B-6/C-1）。
# 現行5ランクのサフィックス + "#"サフィックス方式を使っていた廃止済み2ランク
# （SEVEN_S1/SIX_S1）のサフィックスを合わせたもの。DELETE対象から保護する目的の
# ため、現行/廃止どちらのサフィックスも含める（廃止済みランクの残存行を誤って
# 巻き込み削除しないための安全網）。
_PAPER_SUFFIXES = tuple(spec.suffix for spec in CURRENT_PAPER_RANKS) + tuple(
    spec.suffix for spec in ABOLISHED_PAPER_RANKS if spec.suffix
)


def _parse_picks_full(target_date: str) -> dict:
    """公開買い目ファイルから {(venue, race_no, slot): (rank, time, combo_str)}

    2段階生成のため 昼〜夕 = wave_picks_wt_{date}.txt と
    夜 = wave_picks_wt_{date}_night.txt の両方を読み、採点対象を統合する
    （夜レースは start≥19時で昼と発走時刻が重ならず race_no 衝突なし）。
    slot は "wide"(ワイド1点)/"main"(SS/S/A)。同一レースで両プロダクトが並立するため分離。
    """
    base = Path(__file__).parent.parent / "data" / "picks"
    picks = {}
    for fname in (f"wave_picks_wt_{target_date}.txt", f"wave_picks_wt_{target_date}_night.txt"):
        p = base / fname
        if not p.exists():
            continue
        rank = None
        for line in p.read_text(encoding="utf-8").splitlines():
            if "【7+車 SSランク】" in line:
                # 旧S1(7PLUS_R)は 2026-07-16 全廃。全廃日以降の txt に残る SS セクション
                # （移行日の旧コード生成分）は採点しない（アーカイブ済み行の再作成防止）。
                # 2026-07-10〜07-15 は 7PLUS_R、それ以前は旧SS（過去日再採点の互換）。
                if target_date >= "2026-07-16":
                    rank = None
                else:
                    rank = "7PLUS_R" if target_date >= "2026-07-10" else "7PLUS_SS"
            elif "【7+車 Rランク】" in line: rank = "7PLUS_R"   # 移行期の旧表記互換
            elif "【7+車 Sランク】" in line:
                rank = None   # S/S+（三連単F）は 2026-07-15 全廃・過去分も採点対象外
            elif "【7+車 Aランク】" in line: rank = None   # 廃止済み
            elif "【7+車】" in line: rank = "7PLUS_S"  # 旧フォーマット後方互換
            elif "【SSランク】" in line: rank = None   # 旧SS/S/A/B/WIDEは採点対象外
            elif "【Sランク】" in line: rank = None
            elif "【Aランク】" in line: rank = None
            elif "【Bランク】" in line: rank = None
            elif "【ワイド1点】" in line: rank = None
            elif rank:
                m = re.match(r"\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\d+)R\s+\[\d+車\]\s+(.+?)\s+\(\d+点", line)
                if m:
                    slot = {"7PLUS_SS": "7plus_ss", "7PLUS_R": "7plus_r"}.get(rank, "7plus_s")
                    picks[(m.group(2), int(m.group(3)), slot)] = (rank, m.group(1), m.group(4))
    return picks


def _parse_combo(combo_str: str):
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-").replace("⇄", "-")   # ⇄=SS 1-2着BOX(両順)
    parts = body.split("-")
    thirds = [int(x) for x in parts[2].split(",")] if len(parts) >= 3 else []  # ワイド=2車で空
    return int(parts[0]), int(parts[1]), thirds


def _void_by_dns(p1, p2, thirds, board, is_wide=False):
    """欠車(購入不可=返還)の無効化ルール（実精算方式・2026-07-15）。

    board = 最終オッズ盤面に掲載されていた車（=実際に購入できた車）の集合。
    欠車はオッズ盤面から除外されるため board に含まれず、返還（集計除外）となる。
    落車・失格・棄権（発走前に不可知）は board に残るため買い目は購入扱いのまま
    外れ計上する（実際の精算と同じ。旧・完走者基準の返還扱いは廃止）。
      軸(p1/p2)が欠車      → レース無効（返還）。 returns (True, [])
      相手(thirds)が欠車   → その目のみ除外。     returns (False, 有効thirds)
      相手が全員欠車       → 買える目なし→無効。  returns (True, [])
    ワイドは2車とも軸扱い（どちらか欠車で無効）。
    """
    if p1 not in board or p2 not in board:
        return True, []
    if is_wide:
        return False, []
    valid = [t for t in thirds if t in board]
    return (not valid), valid


def _board_frames(conn, race_key: str) -> set[int]:
    """最終オッズ盤面(trio)に掲載されている車番集合を返す（欠車は掲載されない）。"""
    board: set[int] = set()
    for (comb,) in conn.execute(
        "SELECT combination FROM wt_odds WHERE race_key=? AND bet_type='trio'",
        (race_key,),
    ).fetchall():
        for part in re.split(r"[-=]", str(comb)):
            try:
                board.add(int(part))
            except ValueError:
                pass
    return board


def _write_miwokuri(target_date: str, purchased_base_keys: set[str], conn, pm: dict | None = None) -> int:
    """candidates.json にあり購入されなかったレースを miwokuri=True で書き込む。

    pm が渡された場合は三連複採点を行い hit/trio_payout を記録する。
    payout は 0 固定（見送りなので賭け金なし）。
    purchased_base_keys: 購入済み race_key の "#" 前の base 部分の集合。
    """
    # 旧S1(7PLUS_R)全廃日以降は candidates.json 由来の見送り行を書かない
    # （2026-07-16 の移行日分が旧コード生成の candidates を残しているため）
    if target_date >= "2026-07-16":
        return 0
    if pm is None:
        pm = {}
    picks_dir = Path(__file__).parent.parent / "data" / "picks"
    candidates: list[dict] = []
    for fname in (
        f"wave_picks_wt_{target_date}_candidates.json",
        f"wave_picks_wt_{target_date}_night_candidates.json",
    ):
        p = picks_dir / fname
        if p.exists():
            try:
                candidates += json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

    if not candidates:
        return 0

    count = 0
    for cand in candidates:
        rk = cand.get("race_key")
        if not rk or rk in purchased_base_keys:
            continue
        # 未確定レース（finish_order 未記録）はスキップ。
        # 30分cron（results_check_wt.sh）から呼ばれる場合、まだ発走していない
        # 候補を miwokuri=TRUE にしないための安全弁。翌朝には全レース確定済み。
        has_result = conn.execute(
            "SELECT 1 FROM wt_entries WHERE race_key=? AND finish_order > 0 LIMIT 1", (rk,)
        ).fetchone()
        if not has_result:
            continue
        # gap12 < 0.10（Aランク廃止帯）も見送り確定の対象に含める。
        # write_candidates_wt.py が SS 追跡用に gap12>=0.07 を #CAND 登録するため、
        # ここでスキップすると未購入のまま miwokuri=FALSE が残り、kiseki 一覧で
        # 推奨のように表示される（2026-07-08 大垣5R/取手6R で発生）。
        rank = "7PLUS_CAND"
        p1 = cand.get("pivot1")
        p2 = cand.get("pivot2")
        thirds = cand.get("thirds", [])
        pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
        n_combos = len(thirds)
        store_key = f"{rk}#CAND"

        # 三連複採点（finish_order が揃っていれば採点）
        hit_val, trio_pay_val = 0, 0
        mw_actual = None  # 実着順 (1着,2着,3着) — trifecta_payout 記録用
        if p1 is not None and p2 is not None and thirds:
            rows = conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                "ORDER BY finish_order", (rk,)
            ).fetchall()
            order_list = [int(r[0]) for r in rows]
            if len(order_list) >= 3:
                mw_actual = tuple(order_list[:3])
                top3_cand = frozenset(order_list[:3])
                for t in thirds:
                    if frozenset((p1, p2, t)) == top3_cand:
                        trio_pay_val = pm.get(rk, {}).get(("trio", frozenset((p1, p2, t))), 0)
                        hit_val = 1
                        break
                if not hit_val:
                    trio_pay_val = pm.get(rk, {}).get(("trio", top3_cand), 0)

        try:
            _tri_pay_val = pm.get(rk, {}).get(("trifecta", mw_actual), 0) if mw_actual else 0
            _g12 = cand.get("gap12")
            _g34 = _g23 = None
            _riders_mw = sorted(cand.get("riders", []), key=lambda r: r.get("ai_rank", 99))
            if len(_riders_mw) >= 3:
                try:
                    _g23 = _riders_mw[1]["pred_prob_pct"] - _riders_mw[2]["pred_prob_pct"]  # pt
                except (KeyError, TypeError):
                    _g23 = None
            if len(_riders_mw) >= 4:
                try:
                    _g34 = (_riders_mw[2]["pred_prob_pct"] - _riders_mw[3]["pred_prob_pct"]) / 100.0
                except (KeyError, TypeError):
                    _g34 = None
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,trifecta_payout,bet_amount,route,miwokuri,gap12,gap34,gap23) "
                "VALUES (?,?,?,?,?,?,0,?,?,0,'wt',TRUE,?,?,?)",
                (target_date, store_key, rank, pred, n_combos, hit_val, trio_pay_val, _tri_pay_val,
                 round(_g12, 4) if _g12 is not None else None,
                 round(_g34, 4) if _g34 is not None else None,
                 round(_g23, 2) if _g23 is not None else None),
            )
            count += 1
        except Exception as e:
            print(f"[notify_results_wt] 見送り書き込み失敗 {store_key}: {e}", flush=True)
    return count


def _backfill_miwokuri_trio_payout(conn) -> int:
    """trio_payout=0 の見送り記録を遡及採点する。

    notify_results_wt.py の実行タイミングによっては着順/オッズが未確定で
    trio_payout=0 のまま保存されることがある。
    wt_entries と wt_odds に今データがあれば更新する。
    """
    rows = conn.execute(
        "SELECT race_key FROM picks_history "
        "WHERE miwokuri=TRUE AND trio_payout=0 AND route='wt'"
    ).fetchall()
    if not rows:
        return 0

    base_keys = list({rk.split("#")[0] for (rk,) in rows})
    pm = _load_payouts_wt(base_keys)

    updated = 0
    for (store_key,) in rows:
        base_key = store_key.split("#")[0]
        top3_rows = conn.execute(
            "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
            "ORDER BY finish_order", (base_key,)
        ).fetchall()
        order_list = [int(r[0]) for r in top3_rows]
        if len(order_list) < 3:
            continue
        top3 = frozenset(order_list[:3])
        trio_pay = pm.get(base_key, {}).get(("trio", top3), 0)
        if trio_pay == 0:
            continue

        # candidates.json に記録された pred_combo から hit を再判定
        pred_row = conn.execute(
            "SELECT pred_combo FROM picks_history WHERE race_key=?", (store_key,)
        ).fetchone()
        hit_val = 0
        if pred_row and pred_row[0]:
            body = pred_row[0].split(":", 1)[1].strip() if ":" in pred_row[0] else pred_row[0]
            parts = body.replace("→", "-").replace("⇄", "-").split("-")
            if len(parts) >= 3:
                try:
                    p1, p2 = int(parts[0]), int(parts[1])
                    thirds = [int(x) for x in parts[2].split(",")]
                    for t in thirds:
                        if frozenset((p1, p2, t)) == top3:
                            hit_val = 1
                            break
                except (ValueError, IndexError):
                    pass

        conn.execute(
            "UPDATE picks_history SET trio_payout=?, hit=? WHERE race_key=?",
            (trio_pay, hit_val, store_key),
        )
        updated += 1
    return updated


def _stats_line(label, s):
    if not s or s["bets"] == 0:
        return f"{label}: データなし"
    roi = s["returns"] / s["bets"] * 100
    return (f"{label}: {s['races']}R 的中{s['hits']}回 "
            f"{s['hits']/s['races']*100:.1f}%  投資{s['bets']:,}→回収{s['returns']:,}  ROI{roi:.1f}%")


# 現行ランクのIN句（単一正本 src/strategy_wt.CURRENT_PAPER_RANKS から導出。
# 2026-07-31・B-6/C-1）。リテラル値を直接SQL文字列へ埋め込む（rank名は自コード内の
# 固定定数でユーザー入力を含まないため injection リスクなし。テストで
# inspect.getsource() 等により実際に発行されるSQLへ5ランク全てが含まれることを検証する）。
_QUERY_STATS_RANKS_SQL = ", ".join(f"'{spec.rank}'" for spec in CURRENT_PAPER_RANKS)


def _query_stats(like):
    """月間/年間サマリー用の集計（メッセージ末尾の📅/🗓行）。

    2026-07-16に全廃されたrank='7PLUS_R'をハードコードしたまま放置されており、
    それ以降ずっと0件（"データなし"）を返し続けていたバグを2026-07-28に修正。
    ヘッダー（p7b/p7r/p7h・S1+S7のみ）とは異なり、こちらは現行の全ペーパーランク
    （単一正本 CURRENT_PAPER_RANKS の4ランク: 7S・7A・9S・9A）を合算する
    （ユーザー要望・2026-07-28。RANK_7SS は 2026-08-02 に全廃し正本から除外済み）。

    【2026-07-31修正】IN句が独自ハードコードのままだったため、7SS新設(dc89f14)
    でpicks_historyに16,273行（2022-12-01〜・ROI73.5%）が入ったにもかかわらず
    このIN句に追加されず月次/年次サマリーに一切反映されない状態になっていた
    （全廃済みのSEVEN_S1が残存する一方でRANK_7SSが漏れる、という食い違いが
    3回目の再発だったため、単一正本 CURRENT_PAPER_RANKS からIN句を動的生成する
    構造に変更した。以後はこの関数を直さなくても strategy_wt.py 側の更新だけで
    追随する）。
    """
    with get_connection() as conn:
        r = conn.execute(
            "SELECT COUNT(*) AS races, SUM(hit) AS hits, SUM(payout) AS returns_, SUM(bet_amount) AS bets "
            "FROM picks_history WHERE route='wt' "
            f"AND rank IN ({_QUERY_STATS_RANKS_SQL}) "
            "AND NOT COALESCE(miwokuri, FALSE) AND race_date LIKE ?", (like,)).fetchone()
    return {"races": r["races"] or 0, "hits": r["hits"] or 0, "returns": r["returns_"] or 0, "bets": r["bets"] or 0}


def _query_stats_rank(like, rank):
    """ランク別の統計を取得。"""
    with get_connection() as conn:
        r = conn.execute(
            "SELECT COUNT(*) AS races, SUM(hit) AS hits, SUM(payout) AS returns_, SUM(bet_amount) AS bets "
            "FROM picks_history WHERE route='wt' AND rank=? "
            "AND NOT COALESCE(miwokuri, FALSE) AND race_date LIKE ?", (rank, like)).fetchone()
    return {"races": r["races"] or 0, "hits": r["hits"] or 0, "returns": r["returns_"] or 0, "bets": r["bets"] or 0}


def main():
    from datetime import date
    _main_inner(date)


# 成績サマリーの送信スイッチ（2026-08-07 ユーザー要望で廃止）。
# **採点と picks_history への保存は続ける**。
RESULTS_SUMMARY_NOTIFY_ENABLED = False


def _main_inner(date):
    # 位置引数=日付 / --silent=Discord抑止(picks_history修復のみ・バックフィル用)
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")
    silent = "--silent" in sys.argv
    # 🔴 **成績サマリー（競輪AI[wt]成績）の Discord 通知は 2026-08-07 に廃止**
    #    （ユーザー要望「レース個別の通知のみとする」）。レース単位の通知は
    #    `notify_race_result_wt.py` が担当する。
    #    ⚠️ 本スクリプトは通知だけの存在ではなく**採点して picks_history へ
    #       書き戻す**のが本体なので、呼び出し自体は止めてはいけない。
    #    再開は `RESULTS_SUMMARY_NOTIFY_ENABLED = True` の1行。
    emit = ((lambda m: None) if (silent or not RESULTS_SUMMARY_NOTIFY_ENABLED)
            else (lambda m: send(m, channel="results")))
    dc = target_date.replace("-", "")

    # 発走前判定（prerace_decisions_*.json）を読み込む。
    # 存在するレースは 15分前判定（推奨/見送り・ランク・購入買い目）を最優先で採点し、
    # 事後のオッズや txt のランクで上書きしない。
    decisions: dict[str, dict] = {}
    _dec_path = Path(__file__).parent.parent / "data" / f"prerace_decisions_{target_date}.json"
    # 判定永続化の運用日かどうか（.bak しか残っていない場合も運用日とみなす）
    decisions_mode = _dec_path.exists() or _dec_path.with_name(_dec_path.name + ".bak").exists()
    for _cand_path in (_dec_path, _dec_path.with_name(_dec_path.name + ".bak")):
        if not _cand_path.exists():
            continue
        try:
            decisions = json.loads(_cand_path.read_text(encoding="utf-8"))
            break
        except Exception as _e:
            print(f"[notify_results_wt] prerace_decisions 読み込み失敗 {_cand_path.name}: {_e}", flush=True)
    has_buy_decisions = any(d.get("decision") == "buy" for d in decisions.values())

    picks = _parse_picks_full(target_date)
    if not picks and not has_buy_decisions:
        # ファイル不在(真のエラー) と 7+車推奨0件(静かな日・正常) を区別する
        picks_file = Path(__file__).parent.parent / "data" / "picks" / f"wave_picks_wt_{target_date}.txt"
        if not picks_file.exists():
            emit(f"⚠️ 競輪AI[wt] [{target_date}] 予想ファイルが見つかりません")
        else:
            emit(f"📊 競輪AI[wt] [{target_date}] 7+車推奨なし＝採点対象なし"
                 f"（全目min≥7.0倍+gap12≥0.10 の該当レースなし）")
        return

    with get_connection() as conn:
        # picks_history に route 列が無ければ追加（後方互換）
        cols = [r[1] for r in conn.execute("PRAGMA table_info(picks_history)").fetchall()]
        if "route" not in cols:
            conn.execute("ALTER TABLE picks_history ADD COLUMN route TEXT DEFAULT 'ks'")
        if "trio_payout" not in cols:
            conn.execute("ALTER TABLE picks_history ADD COLUMN trio_payout INTEGER NOT NULL DEFAULT 0")
        if "trifecta_payout" not in cols:
            conn.execute("ALTER TABLE picks_history ADD COLUMN trifecta_payout INTEGER NOT NULL DEFAULT 0")
        if "gap12" not in cols:
            conn.execute("ALTER TABLE picks_history ADD COLUMN gap12 REAL")
        if "gap34" not in cols:
            conn.execute("ALTER TABLE picks_history ADD COLUMN gap34 REAL")
        name2code = {n: c for c, n in conn.execute("SELECT venue_code, name FROM venue_info").fetchall()}
        start_map = dict(conn.execute(
            "SELECT race_key, start_at FROM wt_races WHERE race_date=?", (target_date,)).fetchall())

    # 発走前判定で購入となったが txt に載っていないレースを picks に注入する。
    # （gap12∈[0.07,0.10) 候補の SS 昇格などは朝の txt に含まれず、従来は採点漏れしていた）
    # ガードは「同一スロットが未登録か」で判定する。ベースキー単位だと、txt に別スロット
    # （例: 旧txtのS section）で載っているレースの SS 買いが注入されず採点漏れする
    # （2026-07-10 移行日の伊東5R で発生）。decisions が正本のため同一スロットは上書きする。
    code2name = {c: n for n, c in name2code.items()}
    for _rk, _dec in decisions.items():
        if "#" in _rk:
            continue  # {rk}#ST（S/S+・全廃済み）等のサフィックス付きキーは対象外
        if _dec.get("decision") != "buy" or not _dec.get("thirds"):
            continue
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _rank = _dec.get("rank", "7PLUS_S")
        _slot = {"7PLUS_SS": "7plus_ss", "7PLUS_R": "7plus_r"}.get(_rank, "7plus_s")
        _combo = f"{_dec['pivot1']}-{_dec['pivot2']}-" + ",".join(map(str, _dec["thirds"]))
        picks[(_venue, int(_rno), _slot)] = (_rank, "", _combo)

    # S1=新設計（win軸1着固定・ペーパートレード検証・2026-07-19導入）:
    # decisions キー {rk}#S1（decision=buy）を picks に注入する（slot="seven_s1"）。
    # ※ 旧6車S1（SIX_S1）も同じ #S1 サフィックスを使っていたが 2026-07-17 全廃済みで
    #   その decisions フォーマット（axis/p1/p2/combos が無い）とはフィールドが異なるため、
    #   万一過去日の旧形式 decisions を誤って拾っても _slot=="seven_s1" 側の
    #   int(dec_s1.get("axis")) が TypeError→except で安全にスキップされる。
    for _key, _dec in decisions.items():
        if not _key.endswith("#S1") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-3]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "seven_s1")
        if _pk not in picks:
            picks[_pk] = ("SEVEN_S1", "", "")

    # S7=単勝×複勝指数重なり軸×波乱度選出（ペーパートレード検証・2026-07-21導入）:
    # decisions キー {rk}#S7（decision=buy）を picks に注入する（slot="seven_s7"）。
    for _key, _dec in decisions.items():
        if not _key.endswith("#S7") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-3]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "seven_s7")
        if _pk not in picks:
            picks[_pk] = ("RANK_7S", "", "")

    # 7SS（波乱軸選出・穴レース検知・RANK_7SS）は 2026-08-02 に全廃したため
    # decisions からの注入も停止した（live実績 n=16,298・ROI73.5%）。
    # 候補生成・ライブ判定も同時に止めているので {rk}#7SS の decisions 自体が
    # 新規発生しないが、S1全廃時の教訓に従い採点側でもコードレベルで停止する。

    # S9=S7の9車立て版（独立ランク・ペーパートレード検証・2026-07-26導入）:
    # decisions キー {rk}#S9（decision=buy）を picks に注入する（slot="nine_s9"）。
    for _key, _dec in decisions.items():
        if not _key.endswith("#S9") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-3]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "nine_s9")
        if _pk not in picks:
            picks[_pk] = ("RANK_9S", "", "")

    # 7A=S7の境界ランク（ペーパートレード検証・2026-07-27導入）:
    # decisions キー {rk}#7A（decision=buy）を picks に注入する（slot="seven_7a"）。
    for _key, _dec in decisions.items():
        if not _key.endswith("#7A") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-3]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "seven_7a")
        if _pk not in picks:
            picks[_pk] = ("RANK_7A", "", "")

    # 7B=◎◯一致だが順序・相手で不一致（ペーパートレード検証・2026-08-03導入）:
    # decisions キー {rk}#7B（decision=buy）を picks に注入する（slot="seven_7b"）。
    for _key, _dec in decisions.items():
        if not _key.endswith("#7B") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-3]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "seven_7b")
        if _pk not in picks:
            picks[_pk] = ("RANK_7B", "", "")

    # 7C=ベースモデル・終日の二軸（2026-08-07導入）:
    # decisions キー {rk}#7C（decision=buy）を picks に注入する（slot="seven_7c"）。
    # 買い目の形は 7A/7SS と同じ三連複2軸総流し5点。
    for _key, _dec in decisions.items():
        if not _key.endswith("#7C") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-3]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "seven_7c")
        if _pk not in picks:
            picks[_pk] = ("RANK_7C", "", "")

    # 7H1=穴推奨・本命バスト型（2026-08-06導入）:
    # decisions キー {rk}#7H1（decision=buy）を picks に注入する（slot="seven_7h1"）。
    # ⚠️ 他ランクと違い **三連単+三連複の2券種**なので `combos` ではなく
    #    `legs_trio` の有無で購入判定する。
    for _key, _dec in decisions.items():
        if not _key.endswith("#7H1") or _dec.get("decision") != "buy":
            continue
        if not _dec.get("legs_trio"):
            continue
        _rk = _key[:-4]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "seven_7h1")
        if _pk not in picks:
            picks[_pk] = ("RANK_7H1", "", "")

    # 9H1=穴推奨・9車高配当（2026-08-08導入）:
    # decisions キー {rk}#9H1（decision=buy）を picks に注入する（slot="nine_9h1"）。
    # ⚠️ **三連単フォーメーションの単一券種**。7H1 と違い `legs` 1本だけを見る。
    for _key, _dec in decisions.items():
        if not _key.endswith("#9H1") or _dec.get("decision") != "buy":
            continue
        if not _dec.get("legs"):
            continue
        _rk = _key[:-4]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "nine_9h1")
        if _pk not in picks:
            picks[_pk] = ("RANK_9H1", "", "")

    # 9A=S9の境界ランク（ペーパートレード検証・2026-07-27導入）:
    # decisions キー {rk}#9A（decision=buy）を picks に注入する（slot="nine_9a"）。
    for _key, _dec in decisions.items():
        if not _key.endswith("#9A") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-3]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "nine_9a")
        if _pk not in picks:
            picks[_pk] = ("RANK_9A", "", "")

    # 旧A（{rk}#A）・旧S1（6車三連単）の decisions 注入は 2026-07-17 全廃
    # （両ランク廃止。全廃日以前の decisions が残っていても採点・行再作成しない）

    # miwokuri採点用に candidates.json のレース分も先読みする
    # （gap12/gap34 もここから取得して picks_history に永続化する）
    _cand_keys_extra: set[str] = set()
    gap_map: dict[str, tuple[float | None, float | None, float | None]] = {}  # rk -> (gap12, gap34, gap23_pt)
    _picks_dir = Path(__file__).parent.parent / "data" / "picks"
    for _fname in (f"wave_picks_wt_{target_date}_candidates.json", f"wave_picks_wt_{target_date}_night_candidates.json"):
        _p = _picks_dir / _fname
        if _p.exists():
            try:
                for _cand in json.loads(_p.read_text(encoding="utf-8")):
                    _rk = _cand.get("race_key")
                    if _rk:
                        _cand_keys_extra.add(_rk)
                        _g12 = _cand.get("gap12")
                        _g34 = _g23 = None
                        _riders = sorted(_cand.get("riders", []), key=lambda r: r.get("ai_rank", 99))
                        if len(_riders) >= 3:
                            try:
                                _g23 = _riders[1]["pred_prob_pct"] - _riders[2]["pred_prob_pct"]  # pt
                            except (KeyError, TypeError):
                                _g23 = None
                        if len(_riders) >= 4:
                            try:
                                _g34 = (_riders[2]["pred_prob_pct"] - _riders[3]["pred_prob_pct"]) / 100.0
                            except (KeyError, TypeError):
                                _g34 = None
                        gap_map[_rk] = (round(_g12, 4) if _g12 is not None else None,
                                        round(_g34, 4) if _g34 is not None else None,
                                        round(_g23, 2) if _g23 is not None else None)
            except Exception:
                pass
    keys = list({f"{dc}_{name2code[v]}_{int(rn):02d}" for (v, rn, _s) in picks if v in name2code} | _cand_keys_extra)
    pm = _load_payouts_wt(keys)

    # prerace_gami を事前取得（DELETE前）。prerace_gami < 閾値 のピックは見送り扱いにする。
    # （下の 7.0 は判定永続化導入前=2026-07-08 以前の過去日再採点専用）
    # キーはサフィックス (#CAND/#7S 等) を除いた base_key で正規化することで、
    # 当日中は #CAND として保存されている prerace_gami を翌朝の #7S 等で参照できる。
    existing_gami: dict[str, float] = {}
    with get_connection() as _conn:
        for _rk, _pg in _conn.execute(
            "SELECT race_key, prerace_gami FROM picks_history "
            "WHERE route='wt' AND race_date=? AND prerace_gami IS NOT NULL",
            (target_date,),
        ).fetchall():
            existing_gami[_rk.split("#")[0]] = _pg

    results_7plus_ss, results_7plus_s, results_7plus_r, history = [], [], [], []
    results_7plus_s1 = []     # S1=win軸1着固定（ペーパー）行 — ヘッダー合計(p7b/p7r/p7h/n7)には含める
    results_7plus_s7 = []     # S7=単勝×複勝指数重なり軸×波乱度選出（ペーパー）行 — ヘッダー合計(p7b/p7r/p7h/n7)には含める
    results_s9 = []           # S9=S7の9車立て版（独立ランク・ペーパー）行 — ヘッダー合計には含めず[9車]別集計
    results_7a = []           # 7A=S7の境界ランク（ペーパー）行 — ヘッダー合計には含めず別集計（2026-07-27導入）
    results_7b = []           # 7B=◎◯一致×順序/相手不一致（ペーパー）行 — ヘッダー合計には含めず別集計（2026-08-03導入）
    results_7h1 = []          # 7H1=穴推奨・本命バスト型（2券種）行 — ヘッダー合計には含めず別集計（2026-08-06導入）
    results_9h1 = []          # 9H1=穴推奨・9車高配当（三連単単一券種）行 — 同上（2026-08-08導入）
    results_7c = []           # 7C=ベースモデル・終日の二軸（ペーパー）行 — ヘッダー合計には含めず別集計（2026-08-07導入）
    results_9a = []           # 9A=S9の境界ランク（ペーパー）行 — ヘッダー合計には含めず別集計（2026-07-27導入）
    p7ssb = p7ssr = p7ssh = 0  # 7+車 旧SSランク 合計
    p7sb = p7sr = p7sh = 0    # 7+車 旧Sランク 合計
    p7rb = p7rr = p7rh = 0    # 旧S1（7PLUS_R・2026-07-16全廃・過去日再採点互換）合計
    p7s1b = p7s1r = p7s1h = 0  # 7+車 S1=win軸（ペーパー・名目値。ヘッダー合計に含む）
    p7s7b = p7s7r = p7s7h = 0  # 7+車 S7=波乱度選出（ペーパー・名目値。ヘッダー合計に含む）
    # 2026-07-22: S7は表示ランクSS/Sに分離済み（gate_label）なので結果通知も内訳を分ける
    # 【2026-07-31】rank_7s_gate_label()がSS/SS+を返さなくなった（Sへ統合・廃止）ため、
    # 以下3行のSS+/SS用カウンタは今後永続的に0のまま残る過去互換専用（削除しない）。
    # 表示（_rank_line）はn_races=0だと空文字を返すため出力には現れない。
    p7s7sspn = p7s7sspb = p7s7sspr = p7s7ssph = 0  # S7のうちgate_label="SS+"（重ならない・軸に格上クラスなし）
    p7s7ssn = p7s7ssb = p7s7ssr = p7s7ssh = 0  # S7のうちgate_label="SS"（軸2車がWT◎◯と全く重ならない）
    p7s7sn = p7s7sb = p7s7sr = p7s7sh = 0      # S7のうちgate_label="S"（片方だけ重なる）
    # S9（9車立て・独立ランク）はS7とは別集計（[9車]セクション。ヘッダー[7+車]合計には含めない）
    p9s9sspn = p9s9sspb = p9s9sspr = p9s9ssph = 0
    p9s9ssn = p9s9ssb = p9s9ssr = p9s9ssh = 0
    p9s9sn = p9s9sb = p9s9sr = p9s9sh = 0
    # 7A/9A（S7/S9の境界ランク・単一サブランク・2026-07-27導入。ヘッダー合計には含めない）
    p7ab = p7ar = p7ah = 0
    p9ab = p9ar = p9ah = 0
    # 7B（◎◯一致×順序/相手不一致・相手絞り3点・2026-08-03導入。ヘッダー合計には含めない）
    p7bb = p7br = p7bh = 0
    p7h1b = p7h1r = p7h1h = 0   # 7H1（穴推奨）の 購入額 / 払戻 / 的中数
    p9h1b = p9h1r = p9h1h = 0   # 9H1（穴推奨・9車）の 購入額 / 払戻 / 的中数
    # 7C（ベースモデル・終日の二軸・2026-08-07導入。ヘッダー合計には含めない）
    # ⚠️ **賭け金が可変**（予算枠÷点数）なので購入額は decision の stake から積む。
    p7cb = p7cr = p7ch = 0
    skipped_dns = 0           # 軸欠車/全相手欠車でレース無効（返還）→不計上
    with get_connection() as conn:
        for (venue, race_no, _slot), (rank, ptime, combo_str) in sorted(picks.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            code = name2code.get(venue)
            if code is None:
                continue
            rk = f"{dc}_{code}_{int(race_no):02d}"

            if _slot == "seven_s1":
                # ── S1（新設計・win軸1着固定・ペーパートレード検証）採点 ──
                # 正本は decisions の {rk}#S1。返還処理なし（実精算方式:
                # 買い目確定後の落車・失格・欠車も外れ計上）。ペーパーだが
                # ヘッダー合計（p7b/p7r/p7h・n7）には含める（S1/S7のみが現行ランクのため）。
                # 三連単のため的中判定は「実着順が買い目2点のいずれかと完全一致」。
                dec_s1 = decisions.get(rk + "#S1")
                if not (dec_s1 and dec_s1.get("decision") == "buy" and dec_s1.get("combos")):
                    print(f"[notify_results_wt] S1判定記録なし {rk}: 不計上", flush=True)
                    continue
                s1_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                s1_order = [int(r[0]) for r in s1_rows]
                if len(s1_order) < 3:
                    continue
                s1_stake = int(dec_s1.get("stake") or 100)
                try:
                    s1_combos = [tuple(int(x) for x in str(c).split("-"))
                                 for c in dec_s1["combos"]]
                except (TypeError, ValueError):
                    continue
                s1_order3 = tuple(s1_order[:3])
                s1_hit = s1_order3 in s1_combos
                s1_trifecta_pay = pm.get(rk, {}).get(("trifecta", s1_order3), 0)
                s1_pay = s1_trifecta_pay * s1_stake // 100 if s1_hit else 0
                s1_bet = len(s1_combos) * s1_stake
                s1_pred = ",".join("-".join(map(str, c)) for c in s1_combos)
                s1_tstr = ptime
                _s1_stt = start_map.get(rk)
                if _s1_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        s1_tstr = _dt.fromtimestamp(int(_s1_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                s1_mark = f"◎ ¥{s1_pay:,}" if s1_hit else "×"
                results_7plus_s1.append(
                    f"[S1] {venue} {race_no}R {s1_tstr}  予:{s1_pred}"
                    f"  実:{'-'.join(map(str, s1_order3))}  {s1_mark}")
                p7s1b += s1_bet
                if s1_hit:
                    p7s1r += s1_pay
                    p7s1h += 1
                history.append((target_date, f"{rk}#7S1", "SEVEN_S1", s1_pred, len(s1_combos),
                                int(s1_hit), s1_pay, 0, s1_trifecta_pay, s1_bet, False, None,
                                *gap_map.get(rk, (None, None, None)), None))
                continue

            if _slot == "seven_s7":
                # ── S7（単勝×複勝指数重なり軸×波乱度選出・ペーパートレード検証）採点 ──
                # 正本は decisions の {rk}#S7。返還処理なし（実精算方式:
                # 買い目確定後の落車・失格・欠車も外れ計上）。ペーパーだが
                # ヘッダー合計（p7b/p7r/p7h・n7）には含める（S1/S7のみが現行ランクのため）。
                # 2026-07-21〜: オッズ見送り（decision=="skip"）も軸2車が実際に3着内へ
                # 入ったかだけ参考採点する（miwokuri=True・bet=0でサマリー集計対象外の
                # まま、見送りが「的中していたか」をWebで確認できるようにする）。
                dec_s7 = decisions.get(rk + "#S7")
                if not dec_s7 or dec_s7.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] S7判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_s7.get("decision") == "buy" and bool(dec_s7.get("combos"))
                rank_7s_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                rank_7s_order = [int(r[0]) for r in rank_7s_rows]
                if len(rank_7s_order) < 3:
                    continue
                # 旧 decisions は stake=100 で保存されている。予算枠へ統一した
                # 2026-08-07 以降と混ざらないよう、欠損時はランクの標準単価を使う。
                rank_7s_stake = int(dec_s7.get("stake") or RANK_7S_STAKE)
                try:
                    rank_7s_axis1 = int(dec_s7.get("axis1"))
                    rank_7s_axis2 = int(dec_s7.get("axis2"))
                except (TypeError, ValueError):
                    continue
                rank_7s_top3 = frozenset(rank_7s_order[:3])
                rank_7s_trio_pay = pm.get(rk, {}).get(("trio", rank_7s_top3), 0)
                rank_7s_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(rank_7s_order[:3])), 0)

                if is_buy:
                    try:
                        rank_7s_combos = [frozenset(int(x) for x in str(c).split("-"))
                                     for c in dec_s7["combos"]]
                    except (TypeError, ValueError):
                        continue
                    rank_7s_hit = any(cs == rank_7s_top3 for cs in rank_7s_combos)
                    rank_7s_pay, rank_7s_bet = resolve_payout(
                        conn, rk, "7S", hit=rank_7s_hit, winning_key=rank_7s_top3,
                        odds_payout=rank_7s_trio_pay, fallback_stake=rank_7s_stake, n_combos=len(rank_7s_combos))
                    rank_7s_n_combos = len(rank_7s_combos)
                    rank_7s_thirds = sorted(
                        next(iter(cs - {rank_7s_axis1, rank_7s_axis2}))
                        for cs in rank_7s_combos if len(cs - {rank_7s_axis1, rank_7s_axis2}) == 1)
                    rank_7s_pred = f"{rank_7s_axis1}={rank_7s_axis2}-" + ",".join(map(str, rank_7s_thirds))
                else:
                    # 見送り: 軸2車が両方とも実際の3着内に入っていれば「見送りだが的中」扱い
                    # （買っていれば5点全目のいずれかで的中していたはずのため）。実賭けなし。
                    rank_7s_hit = rank_7s_axis1 in rank_7s_top3 and rank_7s_axis2 in rank_7s_top3
                    rank_7s_pay = 0
                    rank_7s_bet = 0
                    rank_7s_n_combos = 0
                    rank_7s_pred = f"{rank_7s_axis1}={rank_7s_axis2}-見送り"
                rank_7s_tstr = ptime
                _rank_7s_stt = start_map.get(rk)
                if _rank_7s_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        rank_7s_tstr = _dt.fromtimestamp(int(_rank_7s_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    rank_7s_mark = f"◎ ¥{rank_7s_pay:,}" if rank_7s_hit else "×"
                    rank_7s_gate_label = dec_s7.get("gate_label") or "S7"
                    results_7plus_s7.append(
                        f"[{rank_7s_gate_label}] {venue} {race_no}R {rank_7s_tstr}  予:{rank_7s_pred}"
                        f"  実:{'-'.join(map(str, rank_7s_order[:3]))}  {rank_7s_mark}")
                    p7s7b += rank_7s_bet
                    if rank_7s_gate_label == "SS+":
                        p7s7sspn += 1
                        p7s7sspb += rank_7s_bet
                    elif rank_7s_gate_label == "SS":
                        p7s7ssn += 1
                        p7s7ssb += rank_7s_bet
                    elif rank_7s_gate_label == "S":
                        p7s7sn += 1
                        p7s7sb += rank_7s_bet
                    if rank_7s_hit:
                        p7s7r += rank_7s_pay
                        p7s7h += 1
                        if rank_7s_gate_label == "SS+":
                            p7s7sspr += rank_7s_pay
                            p7s7ssph += 1
                        elif rank_7s_gate_label == "SS":
                            p7s7ssr += rank_7s_pay
                            p7s7ssh += 1
                        elif rank_7s_gate_label == "S":
                            p7s7sr += rank_7s_pay
                            p7s7sh += 1
                history.append((target_date, f"{rk}#7S", "RANK_7S", rank_7s_pred, rank_7s_n_combos,
                                int(rank_7s_hit), rank_7s_pay, rank_7s_trio_pay, rank_7s_trifecta_pay, rank_7s_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), dec_s7.get("gate_label")))
                continue

            if _slot == "seven_ss":
                # 7SS（波乱軸選出・穴レース検知・RANK_7SS）は 2026-08-02 に全廃。
                # 候補生成・ライブ判定・decisions注入を全て止めているため
                # このスロットに到達することは無いが、万一過去日を再採点した際に
                # picks_history へ RANK_7SS 行が復活しないよう採点自体を停止する。
                continue

            if _slot == "nine_s9":
                # ── S9（S7の9車立て版・独立ランク・ペーパートレード検証）採点 ──
                # judge_rank_7s/_process_rank_7s_candidates と同じ設計の9車版（judge_rank_9s/
                # _process_rank_9s_candidates）。正本は decisions の {rk}#S9。返還処理なし。
                # ヘッダー[7+車]合計には含めず[9車]として別集計する（Option B・独立ランク）。
                dec_s9 = decisions.get(rk + "#S9")
                if not dec_s9 or dec_s9.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] S9判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_s9.get("decision") == "buy" and bool(dec_s9.get("combos"))
                rank_9s_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                rank_9s_order = [int(r[0]) for r in rank_9s_rows]
                if len(rank_9s_order) < 3:
                    continue
                rank_9s_stake = int(dec_s9.get("stake") or RANK_9S_STAKE)
                try:
                    rank_9s_axis1 = int(dec_s9.get("axis1"))
                    rank_9s_axis2 = int(dec_s9.get("axis2"))
                except (TypeError, ValueError):
                    continue
                rank_9s_top3 = frozenset(rank_9s_order[:3])
                rank_9s_trio_pay = pm.get(rk, {}).get(("trio", rank_9s_top3), 0)
                rank_9s_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(rank_9s_order[:3])), 0)

                if is_buy:
                    try:
                        rank_9s_combos = [frozenset(int(x) for x in str(c).split("-"))
                                     for c in dec_s9["combos"]]
                    except (TypeError, ValueError):
                        continue
                    rank_9s_hit = any(cs == rank_9s_top3 for cs in rank_9s_combos)
                    rank_9s_pay, rank_9s_bet = resolve_payout(
                        conn, rk, "9S", hit=rank_9s_hit, winning_key=rank_9s_top3,
                        odds_payout=rank_9s_trio_pay, fallback_stake=rank_9s_stake, n_combos=len(rank_9s_combos))
                    rank_9s_n_combos = len(rank_9s_combos)
                    rank_9s_thirds = sorted(
                        next(iter(cs - {rank_9s_axis1, rank_9s_axis2}))
                        for cs in rank_9s_combos if len(cs - {rank_9s_axis1, rank_9s_axis2}) == 1)
                    rank_9s_pred = f"{rank_9s_axis1}={rank_9s_axis2}-" + ",".join(map(str, rank_9s_thirds))
                else:
                    rank_9s_hit = rank_9s_axis1 in rank_9s_top3 and rank_9s_axis2 in rank_9s_top3
                    rank_9s_pay = 0
                    rank_9s_bet = 0
                    rank_9s_n_combos = 0
                    rank_9s_pred = f"{rank_9s_axis1}={rank_9s_axis2}-見送り"
                rank_9s_tstr = ptime
                _rank_9s_stt = start_map.get(rk)
                if _rank_9s_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        rank_9s_tstr = _dt.fromtimestamp(int(_rank_9s_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    rank_9s_mark = f"◎ ¥{rank_9s_pay:,}" if rank_9s_hit else "×"
                    rank_9s_gate_label = dec_s9.get("gate_label") or "S9"
                    results_s9.append(
                        f"[S9-{rank_9s_gate_label}] {venue} {race_no}R {rank_9s_tstr}  予:{rank_9s_pred}"
                        f"  実:{'-'.join(map(str, rank_9s_order[:3]))}  {rank_9s_mark}")
                    if rank_9s_gate_label == "SS+":
                        p9s9sspn += 1
                        p9s9sspb += rank_9s_bet
                    elif rank_9s_gate_label == "SS":
                        p9s9ssn += 1
                        p9s9ssb += rank_9s_bet
                    elif rank_9s_gate_label == "S":
                        p9s9sn += 1
                        p9s9sb += rank_9s_bet
                    if rank_9s_hit:
                        if rank_9s_gate_label == "SS+":
                            p9s9sspr += rank_9s_pay
                            p9s9ssph += 1
                        elif rank_9s_gate_label == "SS":
                            p9s9ssr += rank_9s_pay
                            p9s9ssh += 1
                        elif rank_9s_gate_label == "S":
                            p9s9sr += rank_9s_pay
                            p9s9sh += 1
                history.append((target_date, f"{rk}#9S", "RANK_9S", rank_9s_pred, rank_9s_n_combos,
                                int(rank_9s_hit), rank_9s_pay, rank_9s_trio_pay, rank_9s_trifecta_pay, rank_9s_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), dec_s9.get("gate_label")))
                continue

            if _slot == "seven_7a":
                # ── 7A（S7の境界ランク・ペーパートレード検証・2026-07-27導入）採点 ──
                # judge_rank_7s/_process_rank_7s_candidates と同一ロジック（_process_rank_7a_candidates）。
                # 正本は decisions の {rk}#7A。返還処理なし。単一サブランク（gate_labelなし）。
                dec_7a = decisions.get(rk + "#7A")
                if not dec_7a or dec_7a.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] 7A判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_7a.get("decision") == "buy" and bool(dec_7a.get("combos"))
                a7_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                a7_order = [int(r[0]) for r in a7_rows]
                if len(a7_order) < 3:
                    continue
                a7_stake = int(dec_7a.get("stake") or RANK_7A_STAKE)
                try:
                    a7_axis1 = int(dec_7a.get("axis1"))
                    a7_axis2 = int(dec_7a.get("axis2"))
                except (TypeError, ValueError):
                    continue
                a7_top3 = frozenset(a7_order[:3])
                a7_trio_pay = pm.get(rk, {}).get(("trio", a7_top3), 0)
                a7_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(a7_order[:3])), 0)

                if is_buy:
                    try:
                        a7_combos = [frozenset(int(x) for x in str(c).split("-"))
                                     for c in dec_7a["combos"]]
                    except (TypeError, ValueError):
                        continue
                    a7_hit = any(cs == a7_top3 for cs in a7_combos)
                    a7_pay, a7_bet = resolve_payout(
                        conn, rk, "7A", hit=a7_hit, winning_key=a7_top3,
                        odds_payout=a7_trio_pay, fallback_stake=a7_stake, n_combos=len(a7_combos))
                    a7_n_combos = len(a7_combos)
                    a7_thirds = sorted(
                        next(iter(cs - {a7_axis1, a7_axis2}))
                        for cs in a7_combos if len(cs - {a7_axis1, a7_axis2}) == 1)
                    a7_pred = f"{a7_axis1}={a7_axis2}-" + ",".join(map(str, a7_thirds))
                else:
                    a7_hit = a7_axis1 in a7_top3 and a7_axis2 in a7_top3
                    a7_pay = 0
                    a7_bet = 0
                    a7_n_combos = 0
                    a7_pred = f"{a7_axis1}={a7_axis2}-見送り"
                a7_tstr = ptime
                _a7_stt = start_map.get(rk)
                if _a7_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        a7_tstr = _dt.fromtimestamp(int(_a7_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    a7_mark = f"◎ ¥{a7_pay:,}" if a7_hit else "×"
                    results_7a.append(
                        f"[7A] {venue} {race_no}R {a7_tstr}  予:{a7_pred}"
                        f"  実:{'-'.join(map(str, a7_order[:3]))}  {a7_mark}")
                    p7ab += a7_bet
                    if a7_hit:
                        p7ar += a7_pay
                        p7ah += 1
                history.append((target_date, f"{rk}#7A", "RANK_7A", a7_pred, a7_n_combos,
                                int(a7_hit), a7_pay, a7_trio_pay, a7_trifecta_pay, a7_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), None))
                continue

            if _slot == "seven_7b":
                # ── 7B（◎◯一致だが順序・相手で不一致・2026-08-03導入）採点 ──
                # judge_rank_7b/_process_rank_7b_candidates と対。正本は decisions の
                # {rk}#7B。返還処理なし。単一サブランク（gate_labelなし）。
                # 7A との違いは買い目点数だけ（総流し5点ではなく相手絞り3点）で、
                # 採点式は完全に同一（三連複・軸2車+相手1車が3着内に揃えば的中）。
                dec_7b = decisions.get(rk + "#7B")
                if not dec_7b or dec_7b.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] 7B判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_7b.get("decision") == "buy" and bool(dec_7b.get("combos"))
                b7_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                b7_order = [int(r[0]) for r in b7_rows]
                if len(b7_order) < 3:
                    continue
                b7_stake = int(dec_7b.get("stake") or RANK_7B_STAKE)
                try:
                    b7_axis1 = int(dec_7b.get("axis1"))
                    b7_axis2 = int(dec_7b.get("axis2"))
                except (TypeError, ValueError):
                    continue
                b7_top3 = frozenset(b7_order[:3])
                b7_trio_pay = pm.get(rk, {}).get(("trio", b7_top3), 0)
                b7_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(b7_order[:3])), 0)

                if is_buy:
                    try:
                        b7_combos = [frozenset(int(x) for x in str(c).split("-"))
                                     for c in dec_7b["combos"]]
                    except (TypeError, ValueError):
                        continue
                    b7_hit = any(cs == b7_top3 for cs in b7_combos)
                    b7_pay, b7_bet = resolve_payout(
                        conn, rk, "7B", hit=b7_hit, winning_key=b7_top3,
                        odds_payout=b7_trio_pay, fallback_stake=b7_stake, n_combos=len(b7_combos))
                    b7_n_combos = len(b7_combos)
                    b7_thirds = sorted(
                        next(iter(cs - {b7_axis1, b7_axis2}))
                        for cs in b7_combos if len(cs - {b7_axis1, b7_axis2}) == 1)
                    b7_pred = f"{b7_axis1}={b7_axis2}-" + ",".join(map(str, b7_thirds))
                else:
                    b7_hit = b7_axis1 in b7_top3 and b7_axis2 in b7_top3
                    b7_pay = 0
                    b7_bet = 0
                    b7_n_combos = 0
                    b7_pred = f"{b7_axis1}={b7_axis2}-見送り"
                b7_tstr = ptime
                _b7_stt = start_map.get(rk)
                if _b7_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        b7_tstr = _dt.fromtimestamp(int(_b7_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    b7_mark = f"◎ ¥{b7_pay:,}" if b7_hit else "×"
                    results_7b.append(
                        f"[7B] {venue} {race_no}R {b7_tstr}  予:{b7_pred}"
                        f"  実:{'-'.join(map(str, b7_order[:3]))}  {b7_mark}")
                    p7bb += b7_bet
                    if b7_hit:
                        p7br += b7_pay
                        p7bh += 1
                history.append((target_date, f"{rk}#7B", "RANK_7B", b7_pred, b7_n_combos,
                                int(b7_hit), b7_pay, b7_trio_pay, b7_trifecta_pay, b7_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), None))
                continue

            if _slot == "seven_7c":
                # ── 7C（ベースモデル・終日の二軸・2026-08-07導入）採点 ──
                # judge_rank_7c/_process_rank_7c_candidates と対。正本は
                # decisions の {rk}#7C。返還処理なし。
                #
                # ⚠️ **賭け金が可変**（1レース10,000円の予算枠 ÷ 点数）。他ランクの
                #    ように固定 STAKE を掛けると投資額が実態とずれて ROI が壊れる。
                #    decision に保存した stake を必ず使う（無ければ点数から再計算）。
                # ⚠️ 相手は総流しではないので、見送り時の pred_combo も
                #    「軸=軸-見送り」表記にとどめ、残り全車を並べない。
                dec_7c = decisions.get(rk + "#7C")
                if not dec_7c or dec_7c.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] 7C判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_7c.get("decision") == "buy" and bool(dec_7c.get("combos"))
                c7_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                c7_order = [int(r[0]) for r in c7_rows]
                if len(c7_order) < 3:
                    continue
                try:
                    c7_axis1 = int(dec_7c.get("axis1"))
                    c7_axis2 = int(dec_7c.get("axis2"))
                except (TypeError, ValueError):
                    continue
                c7_top3 = frozenset(c7_order[:3])
                c7_trio_pay = pm.get(rk, {}).get(("trio", c7_top3), 0)
                c7_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(c7_order[:3])), 0)

                # 🔴 券種は**発走前の decision に記録された `bet_kind`** で決める。
                #    7C は単勝率で三連複と三連単を出し分けるので（2026-08-09・
                #    `RANK_7C_TRIFECTA_PW_MIN`）、三連複固定で採点すると
                #    **着順違いを的中に数え、買っていない三連複の配当を payout に
                #    書き込む**。picks_history は全ROI・的中率分析の台帳なので
                #    ここが狂うと分析全体が静かに壊れる。
                #    ⚠️ 既定は "trio"。古い decision（bet_kind 無し）は三連複として
                #      採点する＝切替導入前の記録がそのまま正しく読める。
                c7_is_tf = dec_7c.get("bet_kind") == "trifecta"
                c7_order3 = tuple(c7_order[:3])
                if is_buy:
                    try:
                        if c7_is_tf:
                            c7_combos = [tuple(int(x) for x in str(c).split("-"))
                                         for c in dec_7c["combos"]]
                        else:
                            c7_combos = [frozenset(int(x) for x in str(c).split("-"))
                                         for c in dec_7c["combos"]]
                    except (TypeError, ValueError):
                        continue
                    c7_n_combos = len(c7_combos)
                    c7_stake = int(dec_7c.get("stake")
                                   or rank_7c_unit_stake(c7_n_combos))
                    c7_win_key = c7_order3 if c7_is_tf else c7_top3
                    c7_hit = any(cs == c7_win_key for cs in c7_combos)
                    c7_pay, c7_bet = resolve_payout(
                        conn, rk, "7C", hit=c7_hit, winning_key=c7_win_key,
                        odds_payout=(c7_trifecta_pay if c7_is_tf else c7_trio_pay),
                        fallback_stake=c7_stake, n_combos=c7_n_combos)
                    if c7_is_tf:
                        # 買い目は必ず「軸1-軸2-相手」なので3列目だけ集めれば復元できる。
                        c7_thirds = sorted(cs[2] for cs in c7_combos if len(cs) == 3)
                        c7_pred = ("三単:" + f"{c7_axis1}-{c7_axis2}-"
                                   + ",".join(map(str, c7_thirds)))
                    else:
                        c7_thirds = sorted(
                            next(iter(cs - {c7_axis1, c7_axis2}))
                            for cs in c7_combos if len(cs - {c7_axis1, c7_axis2}) == 1)
                        c7_pred = f"{c7_axis1}={c7_axis2}-" + ",".join(map(str, c7_thirds))
                else:
                    c7_hit = c7_axis1 in c7_top3 and c7_axis2 in c7_top3
                    c7_pay = 0
                    c7_bet = 0
                    c7_n_combos = 0
                    c7_pred = f"{c7_axis1}={c7_axis2}-見送り"
                c7_tstr = ptime
                _c7_stt = start_map.get(rk)
                if _c7_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        c7_tstr = _dt.fromtimestamp(int(_c7_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    c7_mark = f"◎ ¥{c7_pay:,}" if c7_hit else "×"
                    results_7c.append(
                        f"[7C] {venue} {race_no}R {c7_tstr}  予:{c7_pred}"
                        f"  実:{'-'.join(map(str, c7_order[:3]))}  {c7_mark}")
                    p7cb += c7_bet
                    if c7_hit:
                        p7cr += c7_pay
                        p7ch += 1
                history.append((target_date, f"{rk}#7C", "RANK_7C", c7_pred, c7_n_combos,
                                int(c7_hit), c7_pay, c7_trio_pay, c7_trifecta_pay, c7_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), None))
                continue

            if _slot == "seven_7h1":
                # ── 7H1（穴推奨・本命バスト型・2026-08-06導入）採点 ──
                # judge_rank_7h1/_process_rank_7h1_candidates と対。正本は
                # decisions の {rk}#7H1。返還処理なし。
                #
                # ⚠️ **既存6ランクと違い2券種**なので採点式が違う:
                #   三連複と三連単をそれぞれ判定し、**払戻は合算**して payout に入れる
                #   （picks_history は1レース1行・ユーザー承認 2026-08-06）。
                #   券種別の払戻は trio_payout / trifecta_payout に残す。
                #   三連単が的中したときは三連複も必ず的中する（同じ3車）とは限らない
                #   ——三連単の3着は総流しで本命ラインを含むが、三連複はプールのみ
                #   なので **三連単だけ的中する組み合わせが存在する**。両方を独立に見る。
                dec_h1 = decisions.get(rk + "#7H1")
                if not dec_h1 or dec_h1.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] 7H1判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_h1.get("decision") == "buy" and bool(dec_h1.get("legs_trio"))
                h1_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? "
                    "AND finish_order BETWEEN 1 AND 3 ORDER BY finish_order", (rk,)
                ).fetchall()
                h1_order = [int(r[0]) for r in h1_rows]
                if len(h1_order) < 3:
                    continue
                h1_top3 = frozenset(h1_order[:3])
                h1_trio_odds = pm.get(rk, {}).get(("trio", h1_top3), 0)
                h1_tf_odds = pm.get(rk, {}).get(("trifecta", tuple(h1_order[:3])), 0)

                if is_buy:
                    legs_trio = [frozenset(int(x) for x in str(t).split("="))
                                 for t in dec_h1.get("legs_trio") or []]
                    legs_tf = [str(t) for t in dec_h1.get("legs_tf") or []]
                    u_trio = int(dec_h1.get("stake_trio") or 0)
                    u_tf = int(dec_h1.get("stake_tf") or 0)
                    hit_trio = h1_top3 in legs_trio
                    hit_tf = "-".join(map(str, h1_order[:3])) in legs_tf
                    # pm のオッズは「100円あたりの払戻」なので賭け金で按分する
                    h1_pay_trio = h1_trio_odds * u_trio // 100 if hit_trio else 0
                    h1_pay_tf = h1_tf_odds * u_tf // 100 if hit_tf else 0
                    h1_pay = h1_pay_trio + h1_pay_tf
                    h1_hit = hit_trio or hit_tf
                    h1_bet = int(dec_h1.get("bet_amount")
                                 or (u_trio * len(legs_trio) + u_tf * len(legs_tf)))
                    h1_n = len(legs_trio) + len(legs_tf)
                    h1_pred = ("三複:" + ",".join(dec_h1.get("legs_trio") or [])
                               + " / 三単:" + ",".join(legs_tf))
                else:
                    h1_hit = False
                    h1_pay = h1_pay_trio = h1_pay_tf = 0
                    h1_bet = 0
                    h1_n = 0
                    h1_pred = "見送り"
                h1_tstr = ptime
                _h1_stt = start_map.get(rk)
                if _h1_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        h1_tstr = _dt.fromtimestamp(
                            int(_h1_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    if h1_hit:
                        _kind = ("三複+三単" if h1_pay_trio and h1_pay_tf
                                 else ("三単" if h1_pay_tf else "三複"))
                        h1_mark = f"◎ {_kind} ¥{h1_pay:,}"
                    else:
                        h1_mark = "×"
                    results_7h1.append(
                        f"[7H1] {venue} {race_no}R {h1_tstr}"
                        f"  実:{'-'.join(map(str, h1_order[:3]))}  {h1_mark}")
                    p7h1b += h1_bet
                    if h1_hit:
                        p7h1r += h1_pay
                        p7h1h += 1
                # ⚠️ trio_payout / trifecta_payout は **全ランク共通で「100円あたりの
                #    確定配当」**（賭け金に依存しない生の値・migrate_picks_history_stake.py
                #    の docstring が正本）。7H1 はここに券種別の実払戻額を入れていたため、
                #    同じ列が他ランクと違う意味になり、Web が「✓¥19,780 複¥19,780」と
                #    同じ額を2度出していた（2026-08-08 是正）。実額は payout に入っている。
                history.append((target_date, f"{rk}#7H1", "RANK_7H1", h1_pred, h1_n,
                                int(h1_hit), h1_pay, h1_trio_odds, h1_tf_odds, h1_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), None))
                continue

            if _slot == "nine_9h1":
                # ── 9H1（穴推奨・9車の高配当狙い・2026-08-08導入）採点 ──
                # judge_rank_9h1/_process_rank_9h1_candidates と対。正本は
                # decisions の {rk}#9H1。返還処理なし。
                #
                # ⚠️ **三連単フォーメーションの単一券種**（6点）。7H1 と違い
                #    券種は1つなので払戻は trifecta_payout だけに入り、
                #    trio_payout は常に0（kiseki 側 Web もそれを前提にしている）。
                dec_h9 = decisions.get(rk + "#9H1")
                if not dec_h9 or dec_h9.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] 9H1判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_h9.get("decision") == "buy" and bool(dec_h9.get("legs"))
                h9_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? "
                    "AND finish_order BETWEEN 1 AND 3 ORDER BY finish_order", (rk,)
                ).fetchall()
                h9_order = [int(r[0]) for r in h9_rows]
                if len(h9_order) < 3:
                    continue
                h9_key = tuple(h9_order[:3])
                h9_tf_odds = pm.get(rk, {}).get(("trifecta", h9_key), 0)

                if is_buy:
                    h9_legs = [str(t) for t in dec_h9.get("legs") or []]
                    h9_n = len(h9_legs)
                    h9_unit = int(dec_h9.get("stake") or 0)
                    h9_hit = "-".join(map(str, h9_key)) in h9_legs
                    h9_pay, h9_bet = resolve_payout(
                        conn, rk, "9H1", hit=h9_hit, winning_key=h9_key,
                        odds_payout=h9_tf_odds, fallback_stake=h9_unit, n_combos=h9_n)
                    if not h9_bet:
                        h9_bet = int(dec_h9.get("bet_amount") or 0)
                    h9_pred = "三単:" + ",".join(h9_legs)
                else:
                    h9_hit = False
                    h9_pay = 0
                    h9_bet = 0
                    h9_n = 0
                    h9_pred = "見送り"
                h9_tstr = ptime
                _h9_stt = start_map.get(rk)
                if _h9_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        h9_tstr = _dt.fromtimestamp(
                            int(_h9_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    h9_mark = f"◎ 三単 ¥{h9_pay:,}" if h9_hit else "×"
                    results_9h1.append(
                        f"[9H1] {venue} {race_no}R {h9_tstr}"
                        f"  実:{'-'.join(map(str, h9_order[:3]))}  {h9_mark}")
                    p9h1b += h9_bet
                    if h9_hit:
                        p9h1r += h9_pay
                        p9h1h += 1
                # trio_payout は常に0（単一券種）。trifecta_payout は他ランクと同じく
                # 100円あたりの配当（実額は payout）。
                history.append((target_date, f"{rk}#9H1", "RANK_9H1", h9_pred, h9_n,
                                int(h9_hit), h9_pay, 0, h9_tf_odds, h9_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), None))
                continue

            if _slot == "nine_9a":
                # ── 9A（S9の境界ランク・ペーパートレード検証・2026-07-27導入）採点 ──
                # judge_rank_9s/_process_rank_9s_candidates と同一ロジック（_process_rank_9a_candidates）。
                # 正本は decisions の {rk}#9A。返還処理なし。単一サブランク（gate_labelなし）。
                dec_9a = decisions.get(rk + "#9A")
                if not dec_9a or dec_9a.get("decision") not in ("buy", "skip"):
                    print(f"[notify_results_wt] 9A判定記録なし {rk}: 不計上", flush=True)
                    continue
                is_buy = dec_9a.get("decision") == "buy" and bool(dec_9a.get("combos"))
                a9_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                a9_order = [int(r[0]) for r in a9_rows]
                if len(a9_order) < 3:
                    continue
                a9_stake = int(dec_9a.get("stake") or RANK_9A_STAKE)
                try:
                    a9_axis1 = int(dec_9a.get("axis1"))
                    a9_axis2 = int(dec_9a.get("axis2"))
                except (TypeError, ValueError):
                    continue
                a9_top3 = frozenset(a9_order[:3])
                a9_trio_pay = pm.get(rk, {}).get(("trio", a9_top3), 0)
                a9_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(a9_order[:3])), 0)

                if is_buy:
                    try:
                        a9_combos = [frozenset(int(x) for x in str(c).split("-"))
                                     for c in dec_9a["combos"]]
                    except (TypeError, ValueError):
                        continue
                    a9_hit = any(cs == a9_top3 for cs in a9_combos)
                    a9_pay, a9_bet = resolve_payout(
                        conn, rk, "9A", hit=a9_hit, winning_key=a9_top3,
                        odds_payout=a9_trio_pay, fallback_stake=a9_stake, n_combos=len(a9_combos))
                    a9_n_combos = len(a9_combos)
                    a9_thirds = sorted(
                        next(iter(cs - {a9_axis1, a9_axis2}))
                        for cs in a9_combos if len(cs - {a9_axis1, a9_axis2}) == 1)
                    a9_pred = f"{a9_axis1}={a9_axis2}-" + ",".join(map(str, a9_thirds))
                else:
                    a9_hit = a9_axis1 in a9_top3 and a9_axis2 in a9_top3
                    a9_pay = 0
                    a9_bet = 0
                    a9_n_combos = 0
                    a9_pred = f"{a9_axis1}={a9_axis2}-見送り"
                a9_tstr = ptime
                _a9_stt = start_map.get(rk)
                if _a9_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        a9_tstr = _dt.fromtimestamp(int(_a9_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                if is_buy:
                    a9_mark = f"◎ ¥{a9_pay:,}" if a9_hit else "×"
                    results_9a.append(
                        f"[9A] {venue} {race_no}R {a9_tstr}  予:{a9_pred}"
                        f"  実:{'-'.join(map(str, a9_order[:3]))}  {a9_mark}")
                    p9ab += a9_bet
                    if a9_hit:
                        p9ar += a9_pay
                        p9ah += 1
                history.append((target_date, f"{rk}#9A", "RANK_9A", a9_pred, a9_n_combos,
                                int(a9_hit), a9_pay, a9_trio_pay, a9_trifecta_pay, a9_bet,
                                not is_buy, None,
                                *gap_map.get(rk, (None, None, None)), None))
                continue

            # 7plus_a / six_s1 スロットの採点は 2026-07-17 全廃（A・旧S1廃止）

            # 発走前判定があるレースは判定時のランク・購入買い目（ガミ目カット済み）で採点する
            dec = decisions.get(rk)
            r_stake = 100  # doc53: ライン格差増額時は decisions.stake=200
            if dec and dec.get("decision") == "buy" and dec.get("thirds"):
                rank = dec.get("rank", rank)
                r_stake = int(dec.get("stake") or 100)
                combo_str = (f"{dec['pivot1']}-{dec['pivot2']}-"
                             + ",".join(map(str, dec["thirds"])))
            rows = conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                "ORDER BY finish_order", (rk,)).fetchall()
            order = [int(r[0]) for r in rows]
            if len(order) < 3:
                continue
            top3 = frozenset(order[:3])
            # 最終オッズ盤面掲載車（=購入できた車）。盤面に無い車=欠車のみ返還扱い。
            # 落車・失格・棄権は盤面に残る→買い目は購入のまま外れ計上（実精算・2026-07-15）。
            # 盤面データが無い場合のみ旧・完走者基準にフォールバック（誤没収防止）。
            board = _board_frames(conn, rk)
            if not board:
                board = {int(r[0]) for r in conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order >= 1",
                    (rk,)).fetchall()}
            p1, p2, thirds = _parse_combo(combo_str)
            # ── 欠車の無効化（返還＝損益に計上しない）──
            skip_race, thirds = _void_by_dns(p1, p2, thirds, board, is_wide=(rank == "WIDE"))
            if skip_race:
                skipped_dns += 1
                continue
            hit, pay = False, 0
            # 7+車は常に三連複（全相手流し）
            n_combos = len(thirds)
            pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
            for t in thirds:
                if frozenset((p1, p2, t)) == top3:
                    pay = pm.get(rk, {}).get(("trio", frozenset((p1, p2, t))), 0) * r_stake // 100
                    hit = True
                    break
            # 不的中に関わらずレース確定三連複/三連単払戻を記録
            trio_pay = pm.get(rk, {}).get(("trio", top3), 0)
            trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(order[:3])), 0)
            bet = n_combos * r_stake
            actual = "-".join(map(str, order[:3]))
            stt = start_map.get(rk)
            from datetime import datetime, timezone, timedelta
            tstr = ptime
            if stt:
                try:
                    tstr = datetime.fromtimestamp(int(stt), tz=timezone(timedelta(hours=9))).strftime("%H:%M")
                except (ValueError, TypeError):
                    pass
            # store_key を先定義（prerace_gami 参照のため stats より前）
            if rank == "7PLUS_SS":
                store_key = f"{rk}#7SS"
            elif rank == "7PLUS_R":
                store_key = f"{rk}#7R"
            else:
                store_key = f"{rk}#7S"
            # existing_gami は base_key で正規化済み（#CAND → #7S 等をまたいで参照可能）
            pg = existing_gami.get(rk)
            if dec is not None:
                # 発走前判定を最優先（15分前判定を事後変更しない）
                is_gami_skip = dec.get("decision") == "skip"
                if not is_gami_skip:
                    # 購入目（ガミ目カット後）の発走前最安オッズを prerace_gami に採用。
                    # 全thirds最安値のままだとカット済み低オッズ目で <7.0 になり
                    # kiseki 側でガミ見送り表示される。
                    _leg_odds = dec.get("leg_odds") or {}
                    _buy_ov = [float(_leg_odds[str(_t)]) for _t in dec.get("thirds", [])
                               if _leg_odds.get(str(_t))]
                    if _buy_ov:
                        pg = round(min(_buy_ov), 2)
            elif decisions_mode:
                # 判定永続化の運用日なのに記録がないレースは購入扱いにしない＝見送り側に倒す。
                # 記録消失時に旧フォールバック（SS無条件購入）が働くと、15分前判定で
                # 見送ったレースの的中が「幻の購入」としてサマリー計上される
                # （2026-07-08 広島4R で発生）。
                is_gami_skip = True
                print(f"[notify_results_wt] 判定記録なし {rk}: 見送り扱い（幻の購入防止）", flush=True)
            else:
                # 判定永続化の導入前の過去日: 従来のprerace_gamiフォールバック
                # SSはガミ目カット済み（定義上ガミ目なし）、Rは判定永続化後の新設 → Sのみ対象。
                # 当時の運用閾値 7.0 のまま維持する（過去日の再採点結果を変えないため）
                is_gami_skip = (rank not in ("7PLUS_SS", "7PLUS_R")) and (pg is not None and pg < 7.0)
            mark = f"◎ ¥{pay:,}" if hit else "×"
            if is_gami_skip:
                mark += "（見送り）"
            rank_label = {"7PLUS_SS": "7SS", "7PLUS_R": "7S1"}.get(rank, "7S")
            row_str = f"[{rank_label}] {venue} {race_no}R {tstr}  予:{pred}  実:{actual}  {mark}"
            if rank == "7PLUS_SS":
                if not is_gami_skip:
                    p7ssb += bet
                    if hit:
                        p7ssr += pay; p7ssh += 1
                results_7plus_ss.append(row_str)
            elif rank == "7PLUS_R":
                if not is_gami_skip:
                    p7rb += bet
                    if hit:
                        p7rr += pay; p7rh += 1
                results_7plus_r.append(row_str)
            else:  # 7PLUS_S
                if not is_gami_skip:
                    p7sb += bet
                    if hit:
                        p7sr += pay; p7sh += 1
                results_7plus_s.append(row_str)
            # prerace ガミ条件落ち → 見送り（bet/pay=0, miwokuri=True）として記録
            if is_gami_skip:
                history.append((target_date, store_key, rank, pred, n_combos, int(hit), 0, trio_pay, trifecta_pay, 0, True, pg, *gap_map.get(rk, (None, None, None)), None))
            else:
                history.append((target_date, store_key, rank, pred, n_combos, int(hit), pay, trio_pay, trifecta_pay, bet, False, pg, *gap_map.get(rk, (None, None, None)), None))

        if history:
            # 採点済みレースのベースキー単位で選択削除する。
            # 全日付削除にすると .txt が欠落した日（夜 .txt のみ読み込み）に
            # 日中スコア済みエントリが消えてしまうため。
            # S1（#7S1）/ S7（#7S）/ 旧A（#7A・現7A境界ランクと共用）/ 9A（#9A）のペーパー行は
            # 自キーのみ削除する。bk#% で消すと同一レースの他ランク記録（#CAND 見送り等）を
            # 巻き込むため。_PAPER_SUFFIXES は単一正本（src/strategy_wt.py）から
            # ファイル冒頭で導出済み（2026-07-31・B-6/C-1）。
            base_keys = {h[1].split("#")[0] for h in history
                         if not h[1].endswith(_PAPER_SUFFIXES)}
            _not_like_paper_sql = " ".join(
                f"AND race_key NOT LIKE '%{suffix}'" for suffix in _PAPER_SUFFIXES)
            for bk in base_keys:
                conn.execute(
                    "DELETE FROM picks_history WHERE race_key LIKE ? AND route='wt' "
                    + _not_like_paper_sql,
                    (bk + "#%",),
                )
            for h in history:
                if h[1].endswith(_PAPER_SUFFIXES):
                    conn.execute(
                        "DELETE FROM picks_history WHERE race_key = ? AND route='wt'",
                        (h[1],),
                    )
            conn.executemany(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,trifecta_payout,bet_amount,route,miwokuri,prerace_gami,gap12,gap34,gap23,gate_label) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,'wt',?,?,?,?,?,?)", history)

        # S1/S7/S9/7A/9A/7SS（ペーパー）は候補見送り集計に影響させない
        # （_PAPER_SUFFIXES のサフィックスを購入扱いにしない）
        purchased_base_keys = {h[1].split("#")[0] for h in history
                               if not h[1].endswith(_PAPER_SUFFIXES)}
        n_miwokuri = _write_miwokuri(target_date, purchased_base_keys, conn, pm)
        if n_miwokuri:
            print(f"[notify_results_wt] {target_date} 見送り {n_miwokuri} 件書き込み", flush=True)

        # trio_payout=0 の見送り記録を遡及採点（タイミング問題で 0 のまま残った分を修正）
        n_backfill = _backfill_miwokuri_trio_payout(conn)
        if n_backfill:
            print(f"[notify_results_wt] 見送り trio_payout バックフィル {n_backfill} 件", flush=True)

        # ペーパー候補（S1/S7/S9/7A/9A/7SS）で15分前判定に到達しなかった行（bet_amount=0・
        # miwokuri=False のまま残存）を見送りに倒す（オッズ取得失敗・候補生成後の中止等）。
        # intraday_results_wt.sh が当日分を毎時実行するため、start_at 未到来（＝発走15分前の
        # 判定窓にまだ入っていない）の候補まで日付一致だけで誤って見送り化しないよう、
        # 発走時刻を過ぎたレースのみを対象にする（2026-07-18 発見・判定自体は notify_prerace_wt
        # が INSERT OR REPLACE で miwokuri=False ごと上書きするため実害はなかったが表示が誤っていた）。
        # 【2026-07-31】OR条件は _PAPER_SUFFIXES（単一正本由来）から動的生成する。
        # 旧来のハードコード版は "#7SS" が漏れており、7SS新設(dc89f14)後もこの
        # 見送り確定処理の対象外になっていた（B-6と同根の潜在バグ・ここで解消）。
        _paper_cand_like_sql = " OR ".join(
            f"race_key LIKE '%{suffix}'" for suffix in _PAPER_SUFFIXES)
        _paper_cands = conn.execute(
            "SELECT race_key FROM picks_history "
            "WHERE race_date = ? AND route='wt' AND bet_amount = 0 AND NOT miwokuri "
            f"AND ({_paper_cand_like_sql})",
            (target_date,)).fetchall()
        _now_unix_paper = int(time.time())
        _paper_to_skip: list[str] = []
        if _paper_cands:
            _base_keys_paper = {r[0].rsplit("#", 1)[0] for r in _paper_cands}
            _start_map_paper = dict(conn.execute(
                f"SELECT race_key, start_at FROM wt_races WHERE race_key IN "
                f"({','.join('?' * len(_base_keys_paper))})",
                list(_base_keys_paper)).fetchall())
            for (store_key,) in _paper_cands:
                base = store_key.rsplit("#", 1)[0]
                sa = _start_map_paper.get(base)
                if sa is not None and int(sa) < _now_unix_paper:
                    _paper_to_skip.append(store_key)
        n_paper_skip = 0
        for store_key in _paper_to_skip:
            cur_p = conn.execute(
                "UPDATE picks_history SET miwokuri = True WHERE race_key = ?",
                (store_key,))
            n_paper_skip += cur_p.rowcount or 0
        if n_paper_skip:
            print(f"[notify_results_wt] ペーパー候補 未判定→見送り {n_paper_skip} 件", flush=True)

    total_7plus = results_7plus_ss + results_7plus_s + results_7plus_r
    if (not total_7plus and not results_7plus_s1 and not results_7plus_s7
            and not results_s9 and not results_7a and not results_9a):
        emit(f"📊 **競輪AI[wt]成績 {target_date}**\n確定レースなし")
        return

    # ヘッダー合計（p7b/p7r/p7h・n7）は現行ランク S1(SEVEN_S1)/S7(RANK_7S) を含めて集計する
    # （kiseki Web サマリーのトップライン = rank IN ('SEVEN_S1','RANK_7S') と揃える。
    #  旧SS/旧S/旧S1(7PLUS_R)は全廃済みで現在は常に0件だが、過去日再採点の互換のため残置）
    p7b = p7ssb + p7sb + p7rb + p7s1b + p7s7b
    p7r = p7ssr + p7sr + p7rr + p7s1r + p7s7r
    p7h = p7ssh + p7sh + p7rh + p7s1h + p7s7h
    p7roi = p7r / p7b * 100 if p7b else 0
    n7 = len(total_7plus) + len(results_7plus_s1) + len(results_7plus_s7)
    p7hit_pct = p7h / n7 * 100 if n7 else 0.0
    header = (
        f"📊 **競輪AI[wt]成績 {target_date}**  [7+車]\n"
        f"確定 {n7}R　的中 {p7h}回 ({p7hit_pct:.1f}%)\n"
        f"投資 {p7b:,}円 → 回収 {p7r:,}円　ROI {p7roi:.1f}%　損益 {p7r-p7b:+,}円"
    )

    # ランク別サマリー
    def _rank_line(label, n_races, bet_total, ret_total, hit_count):
        if not n_races:
            return ""
        roi = ret_total / bet_total * 100 if bet_total else 0
        return (f"[7+車 {label}] {n_races}R 的中{hit_count} "
                f"投資{bet_total:,}→回収{ret_total:,} ROI{roi:.1f}%")

    rank_lines = []
    r_line   = _rank_line("旧S1*", len(results_7plus_r), p7rb, p7rr, p7rh)  # 旧S1（7PLUS_R・過去日再採点時のみ）
    ss_line = _rank_line("SS*", len(results_7plus_ss), p7ssb, p7ssr, p7ssh)  # 廃止済み旧方式（過去日再採点時のみ）
    s_line  = _rank_line("S*",  len(results_7plus_s),  p7sb,  p7sr,  p7sh)
    # S1（ペーパー）はランク別内訳として独立行でも表示する（ヘッダー合計にも含む）
    s1_line = _rank_line("S1(win軸固定)", len(results_7plus_s1), p7s1b, p7s1r, p7s1h)
    # 【2026-07-31】旧「7SS+/7SS」（S7のgate_label="SS+"/"SS"）は廃止済み・
    # 常に0件のため出力には現れない（過去日再採点の互換のため関数呼び出しのみ残置）。
    # （波乱軸選出の RANK_7SS も 2026-08-02 に全廃したため、現在「7SS」の名を冠する
    #   行は Discord 出力に一切存在しない。）
    old7ssp_line = _rank_line("旧7SS+*", p7s7sspn, p7s7sspb, p7s7sspr, p7s7ssph)
    old7ss_line = _rank_line("旧7SS*", p7s7ssn, p7s7ssb, p7s7ssr, p7s7ssh)
    s7s_line = _rank_line("7S(波乱度選出)", p7s7sn, p7s7sb, p7s7sr, p7s7sh)
    # 7A（境界ランク・2026-07-27導入）はヘッダー合計(p7b/p7h)には含めず別行で表示する
    a7_line = _rank_line("7A(境界ランク)", len(results_7a), p7ab, p7ar, p7ah)
    # 7B（◎◯一致×順序/相手不一致・2026-08-03導入）もヘッダー合計には含めず別行で表示する
    b7_line = _rank_line("7B(相手絞り3点)", len(results_7b), p7bb, p7br, p7bh)
    # 7H1（穴推奨・本命バスト型・2026-08-06導入）。三連単+三連複の2券種を合算した行。
    # 既存6ランク（予想ベース）とは目的が違うのでヘッダー合計には含めない。
    h1_line = _rank_line("7H1(穴推奨/2券種)", len(results_7h1), p7h1b, p7h1r, p7h1h)
    h9_line = _rank_line("9H1(穴推奨/9車)", len(results_9h1), p9h1b, p9h1r, p9h1h)
    # 7C（ベースモデル・終日の二軸・2026-08-07導入）。件数が最多になるランク。
    c7_line = _rank_line("7C(ベース/相手可変)", len(results_7c), p7cb, p7cr, p7ch)
    # 7SS（波乱軸選出・RANK_7SS）は 2026-08-02 に全廃したため Discord 行も削除した。
    for _l in (s1_line, old7ssp_line, old7ss_line, s7s_line, a7_line, b7_line,
               c7_line, h1_line, h9_line, r_line, ss_line, s_line):
        if _l:
            rank_lines.append(_l)

    msg = header
    if rank_lines:
        msg += "\n" + "\n".join(rank_lines)
    msg += "\n```\n" + "\n".join(
        total_7plus + results_7plus_s1 + results_7plus_s7 + results_7a + results_7b
        + results_7c + results_7h1 + results_9h1) + "\n```"

    if skipped_dns:
        msg += f"\n※欠車返還によりレース無効: {skipped_dns}件（軸欠車/全相手欠車・損益不計上）"

    month = _query_stats(target_date[:7] + "%")
    year = _query_stats(target_date[:4] + "%")
    msg += f"\n{'─'*28}\n📅 {target_date[:7]}: {_stats_line('月', month)}\n🗓 {target_date[:4]}年: {_stats_line('年', year)}"

    emit(msg[:1900])

    # ── S9（9車立て・独立ランク）は[7+車]ヘッダーに含めず別メッセージで送信 ──
    # （既存メッセージへの追記だとmsg[:1900]切り詰めで低頻度のS9が消える恐れがあるため）
    if results_s9 or results_9a:
        p9b = p9s9sspb + p9s9ssb + p9s9sb
        p9r = p9s9sspr + p9s9ssr + p9s9sr
        p9h = p9s9ssph + p9s9ssh + p9s9sh
        p9n = p9s9sspn + p9s9ssn + p9s9sn
        p9roi = p9r / p9b * 100 if p9b else 0
        p9hit_pct = p9h / p9n * 100 if p9n else 0.0
        rank_9s_msg = (
            f"📊 **競輪AI[wt]成績 {target_date}**  [9車 S9]\n"
            f"確定 {p9n}R　的中 {p9h}回 ({p9hit_pct:.1f}%)\n"
            f"投資 {p9b:,}円 → 回収 {p9r:,}円　ROI {p9roi:.1f}%　損益 {p9r-p9b:+,}円"
        )
        rank_9s_rank_lines = []
        for _l in (
            _rank_line("9SS+(9車波乱度選出・軸格上なし)", p9s9sspn, p9s9sspb, p9s9sspr, p9s9ssph),
            _rank_line("9SS(9車波乱度選出)", p9s9ssn, p9s9ssb, p9s9ssr, p9s9ssh),
            _rank_line("9S(9車波乱度選出)", p9s9sn, p9s9sb, p9s9sr, p9s9sh),
            # 9A（境界ランク・2026-07-27導入）はヘッダー合計(p9b/p9h)には含めず別行で表示する
            _rank_line("9A(境界ランク)", len(results_9a), p9ab, p9ar, p9ah),
        ):
            if _l:
                rank_9s_rank_lines.append(_l)
        if rank_9s_rank_lines:
            rank_9s_msg += "\n" + "\n".join(rank_9s_rank_lines)
        rank_9s_msg += "\n```\n" + "\n".join(results_s9 + results_9a) + "\n```"
        emit(rank_9s_msg[:1900])

    print(f"[notify_results_wt] {target_date} "
          f"S1(ペーパー) {len(results_7plus_s1)}R 的中{p7s1h} / "
          f"旧7SS+(過去分互換) {p7s7sspn}R 的中{p7s7ssph} / "
          f"旧7SS(過去分互換) {p7s7ssn}R 的中{p7s7ssh} / "
          f"7S(ペーパー) {p7s7sn}R 的中{p7s7sh} / "
          f"7A(ペーパー) {len(results_7a)}R 的中{p7ah} / "
          f"7B(ペーパー) {len(results_7b)}R 的中{p7bh} / "
          f"7H1(穴推奨) {len(results_7h1)}R 的中{p7h1h} / "
          f"9H1(穴推奨/9車) {len(results_9h1)}R 的中{p9h1h} / "
          f"S9(ペーパー) {len(results_s9)}R / "
          f"9A(ペーパー) {len(results_9a)}R 的中{p9ah} / "
          f"旧SS {len(results_7plus_ss)}R / 旧S {len(results_7plus_s)}R / 欠車無効{skipped_dns}件")


if __name__ == "__main__":
    main()
