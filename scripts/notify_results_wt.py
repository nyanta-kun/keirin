"""winticket 成績通知＋picks_history保存（7+車 SS=三連複）

wave_picks_wt_{date}.txt の公開買い目と prerace_decisions を、winticket の確定結果
(wt_entries.finish_order) と wt_odds(三連複/三連単) で採点し、Discord通知＋picks_history に保存する。
欠車(finish_order=0/NULL)は着外として除外。公開した買い目のみ採点（再導出しない）。

ランク体系（2026-07-21〜: 現行は S1/S2/S3/S4 の4ペーパーランク）:
  S1(#7S1) = win軸1着固定×3着内モデル相手2車の三連単2点流し（内部rank SEVEN_S1・
    2026-07-19導入）
    ※ ペーパートレード検証中（実際の賭けなし）。正本は prerace_decisions の {rk}#S1。
      U/Mとの重複排除はない（独立戦略）。
  S2(#7U)  = 波乱ライン連れ込み（内部rank 7PLUS_U・2026-07-16〜・旧称U）
    ※ ペーパートレード検証中（実際の賭けなし）。採点はするがヘッダー合計・
      月/年集計（_query_stats）には含めない。正本は prerace_decisions の {rk}#U。
  S3(#7M)  = ◎不一致×システム◎×(gap12≥0.10 OR win_rank≥3 OR ratio≤0.30)
    （内部rank 7PLUS_M・2026-07-19 3way OR拡張。旧定義の波乱ゲート entropy/mto は廃止）
    ※ S2 と同じくペーパートレード検証中（実際の賭けなし・ヘッダー合計不算入）。
      正本は prerace_decisions の {rk}#M。S2 buy と同一ペアのレースは
      発走前判定（judge_m）で S2 優先の重複排除済み（S3 は skip 記録）。
  S4(#7S4) = 単勝×複勝指数トップ3重なり軸×波乱度選出（内部rank SEVEN_S4・
    2026-07-21導入）三連複2軸総流し5点（オッズ下限なし）
    ※ ペーパートレード検証中（実際の賭けなし・ヘッダー合計不算入）。
      正本は prerace_decisions の {rk}#S4。他ランクとの重複排除はない（独立戦略）。
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
import subprocess
import sys
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.notify.discord import send
from src.evaluation.backtest_wt import _load_payouts_wt
from src.database import get_connection


def _cleanup_vps_stale_cand(db_url: str, target_date: str) -> None:
    """VPS の picks_history から不要な #CAND エントリを削除する。

    2 パターンを処理する:
    1. 同一 base_key に購入済みエントリ(#7S/7A/7SS)が存在する #CAND
       （ローカルで購入に置き換えられた後も VPS に残留するもの）
    2. ローカル SQLite に存在しない孤立 #CAND
       （write_candidates_wt.py が書いた後に notify_results_wt.py の処理対象外と
       なりローカルから消えたのに VPS にだけ残ったもの）

    migrate_sqlite_to_pg.py は upsert のみで DELETE しないため、この関数で整合させる。
    """
    import sqlite3
    try:
        import psycopg2
        from src.database import DB_PATH
        # ローカルSQLiteのtarget_date全race_keyを取得
        local_conn = sqlite3.connect(str(DB_PATH))
        local_keys = {
            row[0] for row in local_conn.execute(
                "SELECT race_key FROM picks_history WHERE race_date=? AND route='wt'",
                (target_date,)
            ).fetchall()
        }
        local_conn.close()

        pg_conn = psycopg2.connect(db_url)
        cur = pg_conn.cursor()

        # パターン1: 同一base_keyに購入済みエントリが存在する#CAND
        # ※ ペーパー行（#7S1/#7U/#7M/#7A）は「購入済み」とみなさない（同一レースの
        #    S1系 #CAND 追跡を巻き込み削除しないため・2026-07-16）
        cur.execute("""
            DELETE FROM keirin.picks_history
            WHERE race_date = %s
              AND race_key LIKE %s
              AND SPLIT_PART(race_key, chr(35), 1) IN (
                  SELECT SPLIT_PART(race_key, chr(35), 1)
                  FROM keirin.picks_history
                  WHERE race_date = %s
                    AND race_key NOT LIKE %s
                    AND race_key NOT LIKE %s
                    AND race_key NOT LIKE %s
                    AND race_key NOT LIKE %s
                    AND race_key NOT LIKE %s
                    AND race_key NOT LIKE %s
                    AND race_key NOT LIKE %s
                    AND route = %s
              )
              AND route = %s
        """, (target_date, '%#CAND', target_date, '%#CAND',
              '%#7S1', '%#7U', '%#7M', '%#7A', '%#6S1', '%#7S4', 'wt', 'wt'))
        deleted1 = cur.rowcount

        # パターン2: ローカルSQLiteに存在しない孤立#CAND
        cur.execute("""
            SELECT race_key FROM keirin.picks_history
            WHERE race_date = %s
              AND race_key LIKE %s
              AND route = %s
        """, (target_date, '%#CAND', 'wt'))
        vps_cands = [row[0] for row in cur.fetchall()]
        orphans = [k for k in vps_cands if k not in local_keys]
        deleted2 = 0
        if orphans:
            cur.execute(
                "DELETE FROM keirin.picks_history WHERE race_key = ANY(%s)",
                (orphans,)
            )
            deleted2 = cur.rowcount

        pg_conn.commit()
        pg_conn.close()
        if deleted1:
            print(f"[notify_results_wt] VPS 旧CAND削除(購入重複): {deleted1} 件", flush=True)
        if deleted2:
            print(f"[notify_results_wt] VPS 孤立CAND削除: {deleted2} 件 {orphans}", flush=True)
    except Exception as e:
        print(f"[notify_results_wt] VPS 旧CAND削除失敗（継続）: {e}", flush=True)


def _sync_vps(db_url: str, target_date: str = "") -> None:
    """picks_history.payout 書き込み後に VPS PostgreSQL へ即時同期する。
    db_url 未設定時はスキップ（エラー非致命）。
    wt_odds_snapshot は大容量のためスキップ。
    同期後、target_date の旧 #CAND エントリ（購入済みと重複するもの）を削除する。
    """
    if not db_url:
        return
    script = Path(__file__).parent / "migrate_sqlite_to_pg.py"
    try:
        subprocess.run(
            [sys.executable, str(script), "--skip", "wt_odds_snapshot"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "KEIRIN_DB_URL": db_url},
        )
        print("[notify_results_wt] VPS 同期完了", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[notify_results_wt] VPS 同期失敗（継続）: {e.stderr[:200]}", flush=True)
        return
    if target_date:
        _cleanup_vps_stale_cand(db_url, target_date)


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


def _query_stats(like):
    with get_connection() as conn:
        r = conn.execute(
            "SELECT COUNT(*) AS races, SUM(hit) AS hits, SUM(payout) AS returns_, SUM(bet_amount) AS bets "
            "FROM picks_history WHERE route='wt' AND rank = '7PLUS_R' "
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
    import sqlite3 as _sqlite3
    from datetime import date
    from src.database import DB_PATH
    _db_url = os.environ.get("KEIRIN_DB_URL", "")

    def _sqlite_has_schema() -> bool:
        """SQLiteに最新スキーマ(miwokuri列あり)のpicks_historyが存在するか確認。
        miwokuri列がなければ放棄済みSQLiteとみなしFalseを返す（VPSネイティブモードへ）。
        過去7日以内のwt_entriesデータがなければVPSがPGに直接書いているとみなしFalseを返す。
        """
        try:
            with _sqlite3.connect(str(DB_PATH)) as c:
                if not c.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='picks_history'"
                ).fetchone():
                    return False
                cols = {r[1] for r in c.execute("PRAGMA table_info(picks_history)").fetchall()}
                if "miwokuri" not in cols:
                    return False
                # 過去7日以内のwt_entriesがなければVPSがPGに直接書いているとみなす
                from datetime import date as _date, timedelta as _td
                cutoff = (_date.today() - _td(days=7)).strftime("%Y%m%d")
                has_wt = c.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='wt_entries'"
                ).fetchone()
                if not has_wt:
                    return False
                return bool(c.execute(
                    "SELECT 1 FROM wt_entries WHERE race_key >= ? LIMIT 1",
                    (cutoff,),
                ).fetchone())
        except Exception:
            return False

    # VPS直接書き込みモード: KEIRIN_DB_URL が設定されていてSQLiteにスキーマがない場合
    # get_connection() が PostgreSQL を直接使う。_sync_vps は不要（既にPGへ書いている）。
    # Mac 通常モード: KEIRIN_DB_URL を退避して SQLite へ書き込み、後で _sync_vps で同期。
    _vps_native = bool(_db_url) and not _sqlite_has_schema()
    if not _vps_native:
        os.environ.pop("KEIRIN_DB_URL", "")
    try:
        _main_inner(date, "" if _vps_native else _db_url)
    finally:
        if _db_url and not _vps_native:
            os.environ["KEIRIN_DB_URL"] = _db_url


def _main_inner(date, _db_url):
    # 位置引数=日付 / --silent=Discord抑止(picks_history修復のみ・バックフィル用)
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")
    silent = "--silent" in sys.argv
    emit = (lambda m: None) if silent else send
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

    # U（波乱ライン連れ込み・ペーパートレード検証）: decisions キー {rk}#U（decision=buy）
    # を picks に注入する（slot="7plus_u"）。txt には載らないため decisions が唯一の正本。
    for _key, _dec in decisions.items():
        if not _key.endswith("#U") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-2]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "7plus_u")
        if _pk not in picks:
            picks[_pk] = ("7PLUS_U", "", "")

    # M=S3（◎不一致×システム◎・ペーパートレード検証）: decisions キー {rk}#M（decision=buy）
    # を picks に注入する（slot="7plus_m"）。txt には載らないため decisions が唯一の正本。
    for _key, _dec in decisions.items():
        if not _key.endswith("#M") or _dec.get("decision") != "buy" or not _dec.get("combos"):
            continue
        _rk = _key[:-2]
        if not _rk.startswith(dc):
            continue
        try:
            _, _code, _rno = _rk.split("_")
        except ValueError:
            continue
        _venue = code2name.get(_code)
        if _venue is None:
            continue
        _pk = (_venue, int(_rno), "7plus_m")
        if _pk not in picks:
            picks[_pk] = ("7PLUS_M", "", "")

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

    # S4=単勝×複勝指数重なり軸×波乱度選出（ペーパートレード検証・2026-07-21導入）:
    # decisions キー {rk}#S4（decision=buy）を picks に注入する（slot="seven_s4"）。
    for _key, _dec in decisions.items():
        if not _key.endswith("#S4") or _dec.get("decision") != "buy" or not _dec.get("combos"):
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
        _pk = (_venue, int(_rno), "seven_s4")
        if _pk not in picks:
            picks[_pk] = ("SEVEN_S4", "", "")

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
    results_7plus_u = []      # S2=U（波乱ライン連れ込み・ペーパー）行 — 合計には含めない
    results_7plus_m = []      # S3=M（◎不一致×システム◎・ペーパー）行 — 合計には含めない
    results_7plus_s1 = []     # S1=win軸1着固定（ペーパー）行 — 合計には含めない
    results_7plus_s4 = []     # S4=単勝×複勝指数重なり軸×波乱度選出（ペーパー）行 — 合計には含めない
    p7ssb = p7ssr = p7ssh = 0  # 7+車 旧SSランク 合計
    p7sb = p7sr = p7sh = 0    # 7+車 旧Sランク 合計
    p7rb = p7rr = p7rh = 0    # 旧S1（7PLUS_R・2026-07-16全廃・過去日再採点互換）合計
    p7ub = p7ur = p7uh = 0    # 7+車 S2=U（ペーパー・名目値。ヘッダー合計には不算入）
    p7mb = p7mr = p7mh = 0    # 7+車 S3=M（ペーパー・名目値。ヘッダー合計には不算入）
    p7s1b = p7s1r = p7s1h = 0  # 7+車 S1=win軸（ペーパー・名目値。ヘッダー合計には不算入）
    p7s4b = p7s4r = p7s4h = 0  # 7+車 S4=波乱度選出（ペーパー・名目値。ヘッダー合計には不算入）
    skipped_dns = 0           # 軸欠車/全相手欠車でレース無効（返還）→不計上
    with get_connection() as conn:
        for (venue, race_no, _slot), (rank, ptime, combo_str) in sorted(picks.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            code = name2code.get(venue)
            if code is None:
                continue
            rk = f"{dc}_{code}_{int(race_no):02d}"

            if _slot == "7plus_u":
                # ── U（波乱ライン連れ込み・ペーパートレード検証）採点 ──
                # 正本は decisions の {rk}#U。返還処理なし（実精算方式:
                # 買い目確定後の落車・失格・欠車も外れ計上）。ペーパーのため
                # ヘッダー合計（p7b/p7r/p7h・total_7plus）には算入しない。
                dec_u = decisions.get(rk + "#U")
                if not (dec_u and dec_u.get("decision") == "buy" and dec_u.get("combos")):
                    print(f"[notify_results_wt] U判定記録なし {rk}: 不計上", flush=True)
                    continue
                u_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                u_order = [int(r[0]) for r in u_rows]
                if len(u_order) < 3:
                    continue
                u_stake = int(dec_u.get("stake") or 100)
                try:
                    u_dark = int(dec_u.get("dark"))
                    u_mate = int(dec_u.get("mate"))
                    u_combos = [frozenset(int(x) for x in str(c).split("-"))
                                for c in dec_u["combos"]]
                except (TypeError, ValueError):
                    continue
                u_top3 = frozenset(u_order[:3])
                u_hit = any(cs == u_top3 for cs in u_combos)
                u_trio_pay = pm.get(rk, {}).get(("trio", u_top3), 0)
                u_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(u_order[:3])), 0)
                u_pay = u_trio_pay * u_stake // 100 if u_hit else 0
                u_bet = len(u_combos) * u_stake
                u_thirds = sorted(
                    next(iter(cs - {u_dark, u_mate}))
                    for cs in u_combos if len(cs - {u_dark, u_mate}) == 1)
                u_pred = f"{u_dark}-{u_mate}-" + ",".join(map(str, u_thirds))
                u_tstr = ptime
                _u_stt = start_map.get(rk)
                if _u_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        u_tstr = _dt.fromtimestamp(int(_u_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                u_mark = f"◎ ¥{u_pay:,}" if u_hit else "×"
                results_7plus_u.append(
                    f"[S2] {venue} {race_no}R {u_tstr}  予:{u_pred}"
                    f"  実:{'-'.join(map(str, u_order[:3]))}  {u_mark}（ペーパー）")
                p7ub += u_bet
                if u_hit:
                    p7ur += u_pay
                    p7uh += 1
                history.append((target_date, f"{rk}#7U", "7PLUS_U", u_pred, len(u_combos),
                                int(u_hit), u_pay, u_trio_pay, u_trifecta_pay, u_bet, False, None,
                                *gap_map.get(rk, (None, None, None))))
                continue

            if _slot == "7plus_m":
                # ── M（◎不一致×システム◎・ペーパートレード検証）採点 ──
                # 正本は decisions の {rk}#M。返還処理なし（実精算方式:
                # 買い目確定後の落車・失格・欠車も外れ計上）。ペーパーのため
                # ヘッダー合計（p7b/p7r/p7h・total_7plus）には算入しない。
                dec_m = decisions.get(rk + "#M")
                if not (dec_m and dec_m.get("decision") == "buy" and dec_m.get("combos")):
                    print(f"[notify_results_wt] M判定記録なし {rk}: 不計上", flush=True)
                    continue
                m_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                m_order = [int(r[0]) for r in m_rows]
                if len(m_order) < 3:
                    continue
                m_stake = int(dec_m.get("stake") or 100)
                try:
                    m_m1 = int(dec_m.get("m1"))
                    m_mate = int(dec_m.get("mate"))
                    m_combos = [frozenset(int(x) for x in str(c).split("-"))
                                for c in dec_m["combos"]]
                except (TypeError, ValueError):
                    continue
                m_top3 = frozenset(m_order[:3])
                m_hit = any(cs == m_top3 for cs in m_combos)
                m_trio_pay = pm.get(rk, {}).get(("trio", m_top3), 0)
                m_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(m_order[:3])), 0)
                m_pay = m_trio_pay * m_stake // 100 if m_hit else 0
                m_bet = len(m_combos) * m_stake
                m_thirds = sorted(
                    next(iter(cs - {m_m1, m_mate}))
                    for cs in m_combos if len(cs - {m_m1, m_mate}) == 1)
                m_pred = f"{m_m1}-{m_mate}-" + ",".join(map(str, m_thirds))
                m_tstr = ptime
                _m_stt = start_map.get(rk)
                if _m_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        m_tstr = _dt.fromtimestamp(int(_m_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                m_mark = f"◎ ¥{m_pay:,}" if m_hit else "×"
                results_7plus_m.append(
                    f"[S3] {venue} {race_no}R {m_tstr}  予:{m_pred}"
                    f"  実:{'-'.join(map(str, m_order[:3]))}  {m_mark}（ペーパー）")
                p7mb += m_bet
                if m_hit:
                    p7mr += m_pay
                    p7mh += 1
                history.append((target_date, f"{rk}#7M", "7PLUS_M", m_pred, len(m_combos),
                                int(m_hit), m_pay, m_trio_pay, m_trifecta_pay, m_bet, False, None,
                                *gap_map.get(rk, (None, None, None))))
                continue

            if _slot == "seven_s1":
                # ── S1（新設計・win軸1着固定・ペーパートレード検証）採点 ──
                # 正本は decisions の {rk}#S1。返還処理なし（実精算方式:
                # 買い目確定後の落車・失格・欠車も外れ計上）。ペーパーのため
                # ヘッダー合計（p7b/p7r/p7h・total_7plus）には算入しない。
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
                    f"  実:{'-'.join(map(str, s1_order3))}  {s1_mark}（ペーパー）")
                p7s1b += s1_bet
                if s1_hit:
                    p7s1r += s1_pay
                    p7s1h += 1
                history.append((target_date, f"{rk}#7S1", "SEVEN_S1", s1_pred, len(s1_combos),
                                int(s1_hit), s1_pay, 0, s1_trifecta_pay, s1_bet, False, None,
                                *gap_map.get(rk, (None, None, None))))
                continue

            if _slot == "seven_s4":
                # ── S4（単勝×複勝指数重なり軸×波乱度選出・ペーパートレード検証）採点 ──
                # 正本は decisions の {rk}#S4。返還処理なし（実精算方式:
                # 買い目確定後の落車・失格・欠車も外れ計上）。ペーパーのため
                # ヘッダー合計（p7b/p7r/p7h・total_7plus）には算入しない。
                dec_s4 = decisions.get(rk + "#S4")
                if not (dec_s4 and dec_s4.get("decision") == "buy" and dec_s4.get("combos")):
                    print(f"[notify_results_wt] S4判定記録なし {rk}: 不計上", flush=True)
                    continue
                s4_rows = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                s4_order = [int(r[0]) for r in s4_rows]
                if len(s4_order) < 3:
                    continue
                s4_stake = int(dec_s4.get("stake") or 100)
                try:
                    s4_axis1 = int(dec_s4.get("axis1"))
                    s4_axis2 = int(dec_s4.get("axis2"))
                    s4_combos = [frozenset(int(x) for x in str(c).split("-"))
                                 for c in dec_s4["combos"]]
                except (TypeError, ValueError):
                    continue
                s4_top3 = frozenset(s4_order[:3])
                s4_hit = any(cs == s4_top3 for cs in s4_combos)
                s4_trio_pay = pm.get(rk, {}).get(("trio", s4_top3), 0)
                s4_trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(s4_order[:3])), 0)
                s4_pay = s4_trio_pay * s4_stake // 100 if s4_hit else 0
                s4_bet = len(s4_combos) * s4_stake
                s4_thirds = sorted(
                    next(iter(cs - {s4_axis1, s4_axis2}))
                    for cs in s4_combos if len(cs - {s4_axis1, s4_axis2}) == 1)
                s4_pred = f"{s4_axis1}={s4_axis2}-" + ",".join(map(str, s4_thirds))
                s4_tstr = ptime
                _s4_stt = start_map.get(rk)
                if _s4_stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        s4_tstr = _dt.fromtimestamp(int(_s4_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                s4_mark = f"◎ ¥{s4_pay:,}" if s4_hit else "×"
                results_7plus_s4.append(
                    f"[S4] {venue} {race_no}R {s4_tstr}  予:{s4_pred}"
                    f"  実:{'-'.join(map(str, s4_order[:3]))}  {s4_mark}（ペーパー）")
                p7s4b += s4_bet
                if s4_hit:
                    p7s4r += s4_pay
                    p7s4h += 1
                history.append((target_date, f"{rk}#7S4", "SEVEN_S4", s4_pred, len(s4_combos),
                                int(s4_hit), s4_pay, s4_trio_pay, s4_trifecta_pay, s4_bet, False, None,
                                *gap_map.get(rk, (None, None, None))))
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
                history.append((target_date, store_key, rank, pred, n_combos, int(hit), 0, trio_pay, trifecta_pay, 0, True, pg, *gap_map.get(rk, (None, None, None))))
            else:
                history.append((target_date, store_key, rank, pred, n_combos, int(hit), pay, trio_pay, trifecta_pay, bet, False, pg, *gap_map.get(rk, (None, None, None))))

        if history:
            # 採点済みレースのベースキー単位で選択削除する。
            # 全日付削除にすると .txt が欠落した日（夜 .txt のみ読み込み）に
            # 日中スコア済みエントリが消えてしまうため。
            # S1（#7S1）/ S2（#7U）/ S3（#7M）/ S4（#7S4）/ 旧A（#7A）のペーパー行は自キーのみ削除する。
            # bk#% で消すと同一レースの他ランク記録（#CAND 見送り等）を巻き込むため。
            _PAPER_SUFFIXES = ("#7S1", "#7U", "#7M", "#7A", "#6S1", "#7S4")
            base_keys = {h[1].split("#")[0] for h in history
                         if not h[1].endswith(_PAPER_SUFFIXES)}
            for bk in base_keys:
                conn.execute(
                    "DELETE FROM picks_history WHERE race_key LIKE ? AND route='wt' "
                    "AND race_key NOT LIKE '%#7S1' AND race_key NOT LIKE '%#7U' "
                    "AND race_key NOT LIKE '%#7M' AND race_key NOT LIKE '%#7A' "
                    "AND race_key NOT LIKE '%#6S1' AND race_key NOT LIKE '%#7S4'",
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
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,trifecta_payout,bet_amount,route,miwokuri,prerace_gami,gap12,gap34,gap23) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,'wt',?,?,?,?,?)", history)

        # S1/S2/S3/S4（ペーパー）は候補見送り集計に影響させない（#7S1/#7U/#7M/#7S4/#7A を購入扱いにしない）
        purchased_base_keys = {h[1].split("#")[0] for h in history
                               if not h[1].endswith(("#7S1", "#7U", "#7M", "#7A", "#6S1", "#7S4"))}
        n_miwokuri = _write_miwokuri(target_date, purchased_base_keys, conn, pm)
        if n_miwokuri:
            print(f"[notify_results_wt] {target_date} 見送り {n_miwokuri} 件書き込み", flush=True)

        # trio_payout=0 の見送り記録を遡及採点（タイミング問題で 0 のまま残った分を修正）
        n_backfill = _backfill_miwokuri_trio_payout(conn)
        if n_backfill:
            print(f"[notify_results_wt] 見送り trio_payout バックフィル {n_backfill} 件", flush=True)

        # ペーパー候補（S1/S2/S3）で15分前判定に到達しなかった行（bet_amount=0・
        # miwokuri=False のまま残存）を見送りに倒す（オッズ取得失敗・候補生成後の中止等）。
        # intraday_results_wt.sh が当日分を毎時実行するため、start_at 未到来（＝発走15分前の
        # 判定窓にまだ入っていない）の候補まで日付一致だけで誤って見送り化しないよう、
        # 発走時刻を過ぎたレースのみを対象にする（2026-07-18 発見・判定自体は notify_prerace_wt
        # が INSERT OR REPLACE で miwokuri=False ごと上書きするため実害はなかったが表示が誤っていた）。
        _paper_cands = conn.execute(
            "SELECT race_key FROM picks_history "
            "WHERE race_date = ? AND route='wt' AND bet_amount = 0 AND NOT miwokuri "
            "AND (race_key LIKE '%#7S1' OR race_key LIKE '%#7U' OR race_key LIKE '%#7M' "
            "     OR race_key LIKE '%#7A' OR race_key LIKE '%#6S1' OR race_key LIKE '%#7S4')",
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
    if not total_7plus and not results_7plus_u and not results_7plus_m and not results_7plus_s1 \
            and not results_7plus_s4:
        emit(f"📊 **競輪AI[wt]成績 {target_date}**\n確定レースなし")
        _sync_vps(_db_url, target_date)
        return

    # ヘッダー合計（p7b/p7r/p7h・total_7plus）に S2/S3（ペーパー）は含めない
    p7b = p7ssb + p7sb + p7rb
    p7r = p7ssr + p7sr + p7rr
    p7h = p7ssh + p7sh + p7rh
    p7roi = p7r / p7b * 100 if p7b else 0
    n7 = len(total_7plus)
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
    # S1/S2/S3（ペーパー）は独立行で表示（ヘッダー合計には不算入）
    s1_line = _rank_line("S1(win軸固定・検証/ペーパー)", len(results_7plus_s1), p7s1b, p7s1r, p7s1h)
    u_line  = _rank_line("S2(波乱・検証/ペーパー)", len(results_7plus_u), p7ub, p7ur, p7uh)
    m_line  = _rank_line("S3(不一致×軸信頼・検証/ペーパー)", len(results_7plus_m), p7mb, p7mr, p7mh)
    s4_line = _rank_line("S4(波乱度選出・検証/ペーパー)", len(results_7plus_s4), p7s4b, p7s4r, p7s4h)
    for _l in (s1_line, u_line, m_line, s4_line, r_line, ss_line, s_line):
        if _l:
            rank_lines.append(_l)

    msg = header
    if rank_lines:
        msg += "\n" + "\n".join(rank_lines)
    msg += "\n```\n" + "\n".join(
        total_7plus + results_7plus_s1 + results_7plus_u + results_7plus_m + results_7plus_s4) + "\n```"

    if skipped_dns:
        msg += f"\n※欠車返還によりレース無効: {skipped_dns}件（軸欠車/全相手欠車・損益不計上）"

    month = _query_stats(target_date[:7] + "%")
    year = _query_stats(target_date[:4] + "%")
    msg += f"\n{'─'*28}\n📅 {target_date[:7]}: {_stats_line('月', month)}\n🗓 {target_date[:4]}年: {_stats_line('年', year)}"

    emit(msg[:1900])
    print(f"[notify_results_wt] {target_date} "
          f"S1(ペーパー) {len(results_7plus_s1)}R 的中{p7s1h} / "
          f"S2(ペーパー) {len(results_7plus_u)}R 的中{p7uh} / "
          f"S3(ペーパー) {len(results_7plus_m)}R 的中{p7mh} / "
          f"S4(ペーパー) {len(results_7plus_s4)}R 的中{p7s4h} / "
          f"旧SS {len(results_7plus_ss)}R / 旧S {len(results_7plus_s)}R / 欠車無効{skipped_dns}件")

    _sync_vps(_db_url, target_date)


if __name__ == "__main__":
    main()
