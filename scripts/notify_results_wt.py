"""winticket 成績通知＋picks_history保存（7+車 SS=三連複 / S・S+=三連単F）

wave_picks_wt_{date}.txt の公開買い目と prerace_decisions を、winticket の確定結果
(wt_entries.finish_order) と wt_odds(三連複/三連単) で採点し、Discord通知＋picks_history に保存する。
欠車(finish_order=0/NULL)は着外として除外。公開した買い目のみ採点（再導出しない）。

ランク体系（2026-07-10〜）:
  SS(#7R)  = 三連複 レース単位 min(全目)≥7 全目購入（内部rank 7PLUS_R）
  S/S+(#7ST) = 三連単 1着固定フォーメーション（7PLUS_ST/STP・S+は200円/点）
  旧SS(#7SS)/旧S(#7S) = 買い目カット方式（廃止済み・過去日再採点の互換のみ）

また candidates.json にあり購入されなかった候補レースを miwokuri=True で保存する。
"""
import json
import os
import subprocess
import sys
import re
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
        cur.execute("""
            DELETE FROM keirin.picks_history
            WHERE race_date = %s
              AND race_key LIKE %s
              AND SPLIT_PART(race_key, chr(35), 1) IN (
                  SELECT SPLIT_PART(race_key, chr(35), 1)
                  FROM keirin.picks_history
                  WHERE race_date = %s
                    AND race_key NOT LIKE %s
                    AND route = %s
              )
              AND route = %s
        """, (target_date, '%#CAND', target_date, '%#CAND', 'wt', 'wt'))
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
                # 2026-07-10〜 の「SSランク」は内部rank 7PLUS_R（レース単位・全目購入）。
                # それ以前の txt は旧SS（買い目カット）= 7PLUS_SS として過去日再採点の互換を保つ。
                rank = "7PLUS_R" if target_date >= "2026-07-10" else "7PLUS_SS"
            elif "【7+車 Rランク】" in line: rank = "7PLUS_R"   # 移行期の旧表記互換
            elif "【7+車 Sランク】" in line:
                # 2026-07-10〜 の「Sランク」は三連単フォーメーション 7PLUS_ST。以前は旧S。
                rank = "7PLUS_ST" if target_date >= "2026-07-10" else "7PLUS_S"
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
                    slot = {"7PLUS_SS": "7plus_ss", "7PLUS_R": "7plus_r",
                            "7PLUS_ST": "7plus_st"}.get(rank, "7plus_s")
                    picks[(m.group(2), int(m.group(3)), slot)] = (rank, m.group(1), m.group(4))
    return picks


def _parse_combo(combo_str: str):
    body = combo_str.split(":", 1)[1].strip() if ":" in combo_str else combo_str
    body = body.replace("→", "-").replace("⇄", "-")   # ⇄=SS 1-2着BOX(両順)
    parts = body.split("-")
    thirds = [int(x) for x in parts[2].split(",")] if len(parts) >= 3 else []  # ワイド=2車で空
    return int(parts[0]), int(parts[1]), thirds


def _void_by_dns(p1, p2, thirds, runners, is_wide=False):
    """欠車(購入不可=返還)の無効化ルール。

    runners = そのレースで出走した車(finish_order>=1)の集合。
      軸(p1/p2)が欠車      → レース無効（返還）。 returns (True, [])
      相手(thirds)が欠車   → その目のみ除外。     returns (False, 有効thirds)
      相手が全員欠車       → 買える目なし→無効。  returns (True, [])
    ワイドは2車とも軸扱い（どちらか欠車で無効）。
    """
    if p1 not in runners or p2 not in runners:
        return True, []
    if is_wide:
        return False, []
    valid = [t for t in thirds if t in runners]
    return (not valid), valid


def _write_miwokuri(target_date: str, purchased_base_keys: set[str], conn, pm: dict | None = None) -> int:
    """candidates.json にあり購入されなかったレースを miwokuri=True で書き込む。

    pm が渡された場合は三連複採点を行い hit/trio_payout を記録する。
    payout は 0 固定（見送りなので賭け金なし）。
    purchased_base_keys: 購入済み race_key の "#" 前の base 部分の集合。
    """
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
            conn.execute(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,trifecta_payout,bet_amount,route,miwokuri) "
                "VALUES (?,?,?,?,?,?,0,?,?,0,'wt',TRUE)",
                (target_date, store_key, rank, pred, n_combos, hit_val, trio_pay_val, _tri_pay_val),
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
            "FROM picks_history WHERE route='wt' AND rank IN ('7PLUS_SS','7PLUS_S','7PLUS_R','7PLUS_ST','7PLUS_STP') "
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
            continue  # {rk}#ST は下の三連単ブロックで処理
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

    # 三連単Sランク（decisions キー {rk}#ST・decision=buy）を picks に注入
    for _key, _dec in decisions.items():
        if not _key.endswith("#ST") or _dec.get("decision") != "buy" or not _dec.get("combos"):
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
        _pk = (_venue, int(_rno), "7plus_st")
        if _pk not in picks:
            picks[_pk] = (_dec.get("rank", "7PLUS_ST"), "", "")

    # miwokuri採点用に candidates.json のレース分も先読みする
    _cand_keys_extra: set[str] = set()
    _picks_dir = Path(__file__).parent.parent / "data" / "picks"
    for _fname in (f"wave_picks_wt_{target_date}_candidates.json", f"wave_picks_wt_{target_date}_night_candidates.json"):
        _p = _picks_dir / _fname
        if _p.exists():
            try:
                for _cand in json.loads(_p.read_text(encoding="utf-8")):
                    _rk = _cand.get("race_key")
                    if _rk:
                        _cand_keys_extra.add(_rk)
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
    results_7plus_st = []                       # 三連単S/S+（1着固定フォーメーション）行
    st_n, st_b, st_r, st_h = {}, {}, {}, {}     # 三連単S/S+ ランク別 件数/投資/回収/的中
    p7ssb = p7ssr = p7ssh = 0  # 7+車 旧SSランク 合計
    p7sb = p7sr = p7sh = 0    # 7+車 旧Sランク 合計
    p7rb = p7rr = p7rh = 0    # 7+車 SSランク（内部R・レース単位gami・全目購入）合計
    skipped_dns = 0           # 軸欠車/全相手欠車でレース無効（返還）→不計上
    with get_connection() as conn:
        for (venue, race_no, _slot), (rank, ptime, combo_str) in sorted(picks.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            code = name2code.get(venue)
            if code is None:
                continue
            rk = f"{dc}_{code}_{int(race_no):02d}"

            if _slot == "7plus_st":
                # ── 三連単Sランク（1着固定フォーメーション・doc52追記）採点 ──
                # 正本は decisions の {rk}#ST。記録がなければ幻の購入防止で不計上。
                dec_st = decisions.get(rk + "#ST")
                st_rows_q = conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                    "ORDER BY finish_order", (rk,)).fetchall()
                st_order = [int(r[0]) for r in st_rows_q]
                if len(st_order) < 3:
                    continue
                st_runners = {int(r[0]) for r in conn.execute(
                    "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order >= 1",
                    (rk,)).fetchall()}
                if not (dec_st and dec_st.get("decision") == "buy" and dec_st.get("combos")):
                    print(f"[notify_results_wt] ST判定記録なし {rk}: 不計上（幻の購入防止）", flush=True)
                    continue
                st_rank_v = dec_st.get("rank", "7PLUS_ST")
                st_stake = int(dec_st.get("stake") or 100)
                try:
                    st_p1 = int(dec_st.get("pivot1"))
                    st_combos = [tuple(int(x) for x in c.split("-")) for c in dec_st["combos"]]
                except (TypeError, ValueError, AttributeError):
                    continue
                if st_p1 not in st_runners:
                    skipped_dns += 1
                    continue
                # 欠車を含む目は返還（投資から除外）
                live_combos = [c for c in st_combos if set(c) <= st_runners]
                if not live_combos:
                    skipped_dns += 1
                    continue
                st_actual = (st_order[0], st_order[1], st_order[2])
                st_hit = st_actual in live_combos
                st_pay = 0
                if st_hit:
                    st_pay = pm.get(rk, {}).get(("trifecta", st_actual), 0) * st_stake // 100
                st_bet = len(live_combos) * st_stake
                st_trio_pay = pm.get(rk, {}).get(("trio", frozenset(st_actual)), 0)
                st_trifecta_pay = pm.get(rk, {}).get(("trifecta", st_actual), 0)
                _seconds = dec_st.get("seconds") or []
                st_pred = f"{st_p1}→{','.join(map(str, _seconds))}→全"
                st_label = "7S+" if st_rank_v == "7PLUS_STP" else "7S"
                st_tstr = ptime
                _stt = start_map.get(rk)
                if _stt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
                        st_tstr = _dt.fromtimestamp(int(_stt), tz=_tz(_td(hours=9))).strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                st_mark = f"◎ ¥{st_pay:,}" if st_hit else "×"
                results_7plus_st.append(
                    f"[{st_label}] {venue} {race_no}R {st_tstr}  予:{st_pred}"
                    f"  実:{'-'.join(map(str, st_actual))}  {st_mark}")
                st_n[st_rank_v] = st_n.get(st_rank_v, 0) + 1
                st_b[st_rank_v] = st_b.get(st_rank_v, 0) + st_bet
                if st_hit:
                    st_r[st_rank_v] = st_r.get(st_rank_v, 0) + st_pay
                    st_h[st_rank_v] = st_h.get(st_rank_v, 0) + 1
                history.append((target_date, f"{rk}#7ST", st_rank_v, st_pred, len(live_combos),
                                int(st_hit), st_pay, st_trio_pay, st_trifecta_pay, st_bet, False, None))
                continue

            # 発走前判定があるレースは判定時のランク・購入買い目（ガミ目カット済み）で採点する
            dec = decisions.get(rk)
            if dec and dec.get("decision") == "buy" and dec.get("thirds"):
                rank = dec.get("rank", rank)
                combo_str = (f"{dec['pivot1']}-{dec['pivot2']}-"
                             + ",".join(map(str, dec["thirds"])))
            rows = conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order BETWEEN 1 AND 3 "
                "ORDER BY finish_order", (rk,)).fetchall()
            order = [int(r[0]) for r in rows]
            if len(order) < 3:
                continue
            top3 = frozenset(order[:3])
            # 出走した車(=finish_order>=1)。これに無い車は欠車/失格=購入不可(返還)。
            runners = {int(r[0]) for r in conn.execute(
                "SELECT frame_no FROM wt_entries WHERE race_key=? AND finish_order >= 1", (rk,)).fetchall()}
            p1, p2, thirds = _parse_combo(combo_str)
            # ── 欠車の無効化（返還＝損益に計上しない）──
            skip_race, thirds = _void_by_dns(p1, p2, thirds, runners, is_wide=(rank == "WIDE"))
            if skip_race:
                skipped_dns += 1
                continue
            hit, pay = False, 0
            # 7+車は常に三連複（全相手流し）
            n_combos = len(thirds)
            pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
            for t in thirds:
                if frozenset((p1, p2, t)) == top3:
                    pay = pm.get(rk, {}).get(("trio", frozenset((p1, p2, t))), 0); hit = True; break
            # 不的中に関わらずレース確定三連複/三連単払戻を記録
            trio_pay = pm.get(rk, {}).get(("trio", top3), 0)
            trifecta_pay = pm.get(rk, {}).get(("trifecta", tuple(order[:3])), 0)
            bet = n_combos * 100
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
            rank_label = {"7PLUS_SS": "7SS", "7PLUS_R": "7SS"}.get(rank, "7S")
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
                history.append((target_date, store_key, rank, pred, n_combos, int(hit), 0, trio_pay, trifecta_pay, 0, True, pg))
            else:
                history.append((target_date, store_key, rank, pred, n_combos, int(hit), pay, trio_pay, trifecta_pay, bet, False, pg))

        if history:
            # 採点済みレースのベースキー単位で選択削除する。
            # 全日付削除にすると .txt が欠落した日（夜 .txt のみ読み込み）に
            # 日中スコア済みエントリが消えてしまうため。
            base_keys = {h[1].split("#")[0] for h in history}
            for bk in base_keys:
                conn.execute(
                    "DELETE FROM picks_history WHERE race_key LIKE ? AND route='wt'",
                    (bk + "#%",),
                )
            conn.executemany(
                "INSERT OR REPLACE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,trifecta_payout,bet_amount,route,miwokuri,prerace_gami) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,'wt',?,?)", history)

        purchased_base_keys = {h[1].split("#")[0] for h in history}
        n_miwokuri = _write_miwokuri(target_date, purchased_base_keys, conn, pm)
        if n_miwokuri:
            print(f"[notify_results_wt] {target_date} 見送り {n_miwokuri} 件書き込み", flush=True)

        # trio_payout=0 の見送り記録を遡及採点（タイミング問題で 0 のまま残った分を修正）
        n_backfill = _backfill_miwokuri_trio_payout(conn)
        if n_backfill:
            print(f"[notify_results_wt] 見送り trio_payout バックフィル {n_backfill} 件", flush=True)

    total_7plus = results_7plus_ss + results_7plus_s + results_7plus_r + results_7plus_st
    if not total_7plus:
        emit(f"📊 **競輪AI[wt]成績 {target_date}**\n確定レースなし")
        _sync_vps(_db_url, target_date)
        return

    p7b = p7ssb + p7sb + p7rb + sum(st_b.values())
    p7r = p7ssr + p7sr + p7rr + sum(st_r.values())
    p7h = p7ssh + p7sh + p7rh + sum(st_h.values())
    p7roi = p7r / p7b * 100 if p7b else 0
    n7 = len(total_7plus)
    header = (
        f"📊 **競輪AI[wt]成績 {target_date}**  [7+車]\n"
        f"確定 {n7}R　的中 {p7h}回 ({p7h/n7*100:.1f}%)\n"
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
    r_line   = _rank_line("SS", len(results_7plus_r), p7rb, p7rr, p7rh)  # 新SS（内部rank 7PLUS_R）
    stl_line = _rank_line("S(3連単)", st_n.get("7PLUS_ST", 0),
                          st_b.get("7PLUS_ST", 0), st_r.get("7PLUS_ST", 0), st_h.get("7PLUS_ST", 0))
    stp_line = _rank_line("S+(3連単増額)", st_n.get("7PLUS_STP", 0),
                          st_b.get("7PLUS_STP", 0), st_r.get("7PLUS_STP", 0), st_h.get("7PLUS_STP", 0))
    ss_line = _rank_line("SS*", len(results_7plus_ss), p7ssb, p7ssr, p7ssh)  # 廃止済み旧方式（過去日再採点時のみ）
    s_line  = _rank_line("S*",  len(results_7plus_s),  p7sb,  p7sr,  p7sh)
    for _l in (r_line, stl_line, stp_line, ss_line, s_line):
        if _l:
            rank_lines.append(_l)

    msg = header
    if rank_lines:
        msg += "\n" + "\n".join(rank_lines)
    msg += "\n```\n" + "\n".join(total_7plus) + "\n```"

    if skipped_dns:
        msg += f"\n※欠車返還によりレース無効: {skipped_dns}件（軸欠車/全相手欠車・損益不計上）"

    month = _query_stats(target_date[:7] + "%")
    year = _query_stats(target_date[:4] + "%")
    msg += f"\n{'─'*28}\n📅 {target_date[:7]}: {_stats_line('月', month)}\n🗓 {target_date[:4]}年: {_stats_line('年', year)}"

    emit(msg[:1900])
    print(f"[notify_results_wt] {target_date} 7+車SS {len(results_7plus_r)}R 的中{p7rh} / "
          f"7+車S(3連単) {sum(st_n.values())}R 的中{sum(st_h.values())} / "
          f"旧SS {len(results_7plus_ss)}R / 旧S {len(results_7plus_s)}R / 欠車無効{skipped_dns}件")

    _sync_vps(_db_url, target_date)


if __name__ == "__main__":
    main()
