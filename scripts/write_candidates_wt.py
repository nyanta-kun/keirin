#!/usr/bin/env python3
"""朝時点の候補レースを picks_history に即時書き込む。

gap12 条件を満たす全候補を picks_history に書き込むことで、
同日中から推奨ページに候補レース（miwokuri=False, bet_amount=0）を表示できる。
翌朝 notify_results_wt.py が購入済み/見送りを正確に上書きする。

実行:
    python3 scripts/write_candidates_wt.py [YYYY-MM-DD]

daily_picks_wt.sh および evening_picks_wt.sh から wave-picks-wt の直後に呼ばれる。

初回ガミ判定:
  書き込み後、winticket からその時点の三連複オッズを取得して prerace_gami を設定する。
  最安オッズ < 7.0 の候補は miwokuri=True として早期見送り扱いにする。
  発走 15 分前の notify_prerace_wt.py がリアルタイムオッズで上書き（最終確認）する。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection
from src.strategy_wt import rank_7s_gate_label

GAMI_THRESHOLD = 7.0  # レース単位ガミ閾値（min全目。main.py / notify_prerace_wt.py と揃える）


def _calc_gaps(cand: dict) -> tuple[float | None, float | None, float | None]:
    """candidates JSON から (gap12, gap34, gap23_pt) を計算する。

    notify_results_wt.py の採点時計算と同ロジック。朝の #CAND 書き込み時点で
    永続化することで、kiseki 側が当日中に SS/S/S+ の候補ランク判定
    （gap12≥0.10∧gap23≥1pt / gap12≥0.15 / gap12≥0.25∧gap34≥0.04）を表示できる。
    gap23 のみ pt スケール、gap12/gap34 は 0-1 スケール。
    """
    g12 = cand.get("gap12")
    g23 = g34 = None
    riders = sorted(cand.get("riders", []), key=lambda r: r.get("ai_rank", 99))
    if len(riders) >= 3:
        try:
            g23 = riders[1]["pred_prob_pct"] - riders[2]["pred_prob_pct"]  # pt
        except (KeyError, TypeError):
            g23 = None
    if len(riders) >= 4:
        try:
            g34 = (riders[2]["pred_prob_pct"] - riders[3]["pred_prob_pct"]) / 100.0
        except (KeyError, TypeError):
            g34 = None
    return (
        round(g12, 4) if g12 is not None else None,
        round(g34, 4) if g34 is not None else None,
        round(g23, 2) if g23 is not None else None,
    )


def _insert_candidates_sqlite(
    target_date: str, rows: list[tuple], existing: set[str]
) -> tuple[int, list[str]]:
    """SQLite に #CAND を INSERT し、新規挿入した race_key (base) リストを返す。"""
    inserted = 0
    newly_inserted_bases: list[str] = []
    with get_connection() as conn:
        for target_date_v, store_key, rank, pred, n_combos, g12, g34, g23 in rows:
            base = store_key.rsplit("#", 1)[0]
            if any(k.startswith(base + "#") and k != store_key for k in existing):
                continue
            if store_key in existing:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO picks_history "
                "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,bet_amount,route,miwokuri,gap12,gap34,gap23) "
                "VALUES (?,?,?,?,?,0,0,0,'wt',False,?,?,?)",
                (target_date_v, store_key, rank, pred, n_combos, g12, g34, g23),
            )
            existing.add(store_key)
            inserted += 1
            newly_inserted_bases.append(base)
    return inserted, newly_inserted_bases


def _insert_candidates_vps(target_date: str, rows: list[tuple], existing: set[str]) -> None:
    """VPS PostgreSQL に #CAND を直接 INSERT する（hourly sync を待たずに即時反映）。"""
    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    to_insert = []
    for row in rows:
        store_key = row[1]
        base = store_key.rsplit("#", 1)[0]
        if any(k.startswith(base + "#") and k != store_key for k in existing):
            continue
        to_insert.append(row)
    if not to_insert:
        return
    try:
        import psycopg2  # noqa: PLC0415
        with psycopg2.connect(db_url) as pg_conn:
            with pg_conn.cursor() as cur:
                for target_date_v, store_key, rank, pred, n_combos, g12, g34, g23 in to_insert:
                    cur.execute(
                        """INSERT INTO keirin.picks_history
                            (race_date, race_key, rank, pred_combo, n_combos,
                             hit, payout, trio_payout, bet_amount, route, miwokuri,
                             gap12, gap34, gap23)
                           VALUES (%s, %s, %s, %s, %s, 0, 0, 0, 0, 'wt', FALSE, %s, %s, %s)
                           ON CONFLICT DO NOTHING""",
                        (target_date_v, store_key, rank, pred, n_combos, g12, g34, g23),
                    )
    except Exception as e:
        print(f"[write_candidates_wt] VPS INSERT 失敗: {e}", flush=True)


def _save_initial_gami(race_key: str, race_date: str, min_odds: float, miwokuri: bool) -> None:
    """prerace_gami（初回）と miwokuri を SQLite + VPS に保存する。

    #CAND エントリのみを更新する。#7SS / #7S などの採点済みエントリは
    notify_results_wt.py が管理するため上書きしない。
    """
    rounded = round(min_odds, 2)
    cand_key = race_key + "#CAND"  # #CAND のみ対象（採点済みエントリを上書きしない）

    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE picks_history SET prerace_gami = ?, miwokuri = ? WHERE race_key = ?",
                (rounded, miwokuri, cand_key),
            )
            conn.commit()
    except Exception as e:
        print(f"[write_candidates_wt] SQLite prerace_gami 更新失敗 {race_key}: {e}", flush=True)

    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    try:
        import psycopg2  # noqa: PLC0415
        with psycopg2.connect(db_url) as pg_conn:
            with pg_conn.cursor() as cur:
                cur.execute(
                    "UPDATE keirin.picks_history SET prerace_gami = %s, miwokuri = %s"
                    " WHERE race_key = %s",
                    (rounded, miwokuri, cand_key),
                )
    except Exception as e:
        print(f"[write_candidates_wt] VPS prerace_gami 更新失敗 {race_key}: {e}", flush=True)


def _fetch_initial_gami(candidates: list[dict]) -> None:
    """候補レースのオッズを取得して初回ガミ判定（prerace_gami）を設定する。

    winticket から三連複オッズを取得し最安値を prerace_gami に保存する。
    最安値 < GAMI_THRESHOLD(7.0) なら miwokuri=True（早期見送り）。
    取得失敗時は prerace_gami=NULL のまま notify_prerace_wt.py に委ねる。
    """
    from src.scraper.winticket import WinticketScraper  # noqa: PLC0415

    race_keys = [c["race_key"] for c in candidates if c.get("race_key")]
    if not race_keys:
        return

    with get_connection() as conn:
        placeholders = ",".join("?" * len(race_keys))
        rows = conn.execute(
            f"SELECT race_key, venue_id, cup_id, day_index, race_date, race_no, start_at"
            f" FROM wt_races WHERE race_key IN ({placeholders})",
            race_keys,
        ).fetchall()
    race_info_map = {r["race_key"]: dict(r) for r in rows}

    scraper = WinticketScraper(request_interval=1.0)
    now_unix = int(time.time())

    for cand in candidates:
        rk = cand.get("race_key")
        if not rk:
            continue
        ri = race_info_map.get(rk)
        if not ri:
            print(f"[write_candidates_wt] {rk}: wt_races にレース情報なし（スキップ）", flush=True)
            continue

        # 発走15分前以降は notify_prerace_wt.py の15分前判定が正本。
        # 締切後・確定後のオッズ（跳ね上がる）で再判定すると、見送り済み(miwokuri=True)の
        # #CAND が「OK」に上書きされて一覧に有効推奨風に表示される（2026-07-08 広島1R/5R）。
        try:
            if ri.get("start_at") and now_unix >= int(ri["start_at"]) - 900:
                continue
        except (TypeError, ValueError):
            pass

        p1 = cand.get("pivot1")
        p2 = cand.get("pivot2")
        thirds = cand.get("thirds", [])
        if not thirds or p1 is None or p2 is None:
            continue

        try:
            odds_data = scraper.fetch_odds(
                venue_id=ri["venue_id"],
                race_date=ri["race_date"],
                race_no=ri["race_no"],
                cup_id=ri["cup_id"],
                day_index=ri["day_index"],
            )
        except Exception as e:
            print(f"[write_candidates_wt] {rk}: オッズ取得失敗: {e}", flush=True)
            time.sleep(0.5)
            continue

        if not odds_data:
            print(f"[write_candidates_wt] {rk}: オッズデータなし（スキップ）", flush=True)
            time.sleep(0.5)
            continue

        # 三連複ルックアップ
        trio_lookup: dict[frozenset, float] = {}
        for item in odds_data.get("trio", []):
            parts = re.split(r"[-=]", str(item.get("combination", "")))
            try:
                key = frozenset(int(p) for p in parts)
                if item.get("odds_value"):
                    trio_lookup[key] = float(item["odds_value"])
            except Exception:
                continue

        valid_odds = []
        for t in thirds:
            key = frozenset({int(p1), int(p2), int(t)})
            if key in trio_lookup:
                valid_odds.append(trio_lookup[key])

        if not valid_odds:
            print(f"[write_candidates_wt] {rk}: 対象組み合わせのオッズなし（スキップ）", flush=True)
            time.sleep(0.5)
            continue

        # レース単位判定（doc52・2026-07-10 SS/S置き換え）: min(全目) < 閾値 → 見送り初期値。
        # 最終判定は発走15分前の notify_prerace_wt.py が行う（ここは朝時点の目安）。
        min_odds = min(valid_odds)
        miwokuri = min_odds < GAMI_THRESHOLD
        _save_initial_gami(rk, ri["race_date"], min_odds, miwokuri)
        status = "⚠️見送り" if miwokuri else "✅OK"
        print(
            f"[write_candidates_wt] {rk}: 初回ガミ判定 最安{min_odds:.1f}倍 {status}",
            flush=True,
        )
        time.sleep(0.5)


def _third_list(axis1: int, axis2: int, n_cars: int) -> str:
    """軸2車(axis1/axis2)を除く残り車番号を昇順で返す（"3,4,5,6,7"形式）。

    S7/S9/7A/9A は「軸2車固定・残り全車への三連複流し」で、対象車数(7 or 9)は
    候補生成時点で確定済み（n_entries フィルタ済みの候補JSONのみを扱うため）。
    どの車がオッズ条件で最終的に買い目から漏れるかは発走直前まで未確定だが、
    候補となる残り車の顔ぶれ自体はオッズなしで一意に決まるため、候補時点から
    最終形と同じ表記（axis1=axis2-3,4,5,...）で表示できる。
    """
    return ",".join(str(x) for x in range(1, n_cars + 1) if x not in (axis1, axis2))


def _write_paper_candidates(target_date: str) -> None:
    """S7/S9/7A/9A（ペーパー検証ランク）の候補レースを picks_history に即時書き込む。

    2026-07-21〜: S7（{rk}#7S）を候補時点で書き込む。以前は発走15分前の
    買い判定が成立して初めて行が生成されるため、それ以前は他の推奨外レースと
    区別がつかず、また15分前判定がオッズ条件で見送りになった場合は行自体が
    存在せず _mark_paper_miwokuri() のUPDATEが対象0件で空振りしていた
    （候補だったのに「推奨外」と見分けがつかない・ユーザー指摘で発覚）。
    2026-07-28〜: S9（{rk}#9S）・7A（{rk}#7A）・9A（{rk}#9A）も同様に候補時点で
    書き込む。従来はS7のみ候補時点表示で、S9/7A/9Aは発走15分前判定
    （notify_prerace_wt）が成立するまでページに現れなかった（ユーザー指摘で発覚。
    軸2車・波乱度・ゲート判定はいずれもオッズ非依存でモデル計算のみから確定するため
    候補時点表示に技術的制約はない）。
    あわせてS7/S9の候補時点pred_comboを「axis1=axis2-候補」という汎用プレース
    ホルダーから「axis1=axis2-3,4,5,...」という実際の残り車番号リストに変更した
    （_third_list()）。軸2車を除く残り車の顔ぶれは対象車数(7/9)が候補生成時点で
    確定しているためオッズなしで一意に決まり、発走15分前判定後の最終表記と
    同じ形式で表示できる（7A/9Aはgate_labelなしのため元々この形式）。
    発走15分前判定（notify_prerace_wt）が buy なら本行を上書き、skip なら
    miwokuri=True（オッズ見送り・グレーアウト表示）に更新する。既存行（判定済み）は上書きしない。
    旧S1(7PLUS_R・6車三連複)・旧A(#6A)は 2026-07-17 全廃により書き込み対象外
    （現行の7A/9Aは2026-07-27導入の別ランクで無関係）。
    U（#7U）・M（#7M）は 2026-07-21 全廃。main.py は候補JSON自体を生成しなくなったが、
    全廃日当日は既存の古い候補JSON（コード修正前に生成済み）がまだ残っていたため
    intraday_results_wt.sh 等からの再実行のたびに廃止済みランクの行が復活する事故が
    発生した（write_candidates_wt.py は候補JSONの中身をそのまま信用してINSERT OR
    REPLACEするため、ファイルさえ存在すれば何度でも復活してしまう）。再発防止のため
    ファイルの有無に関わらずU/Mの読み込み自体をコードレベルで無効化する。
    S1（#7S1・SEVEN_S1）は 2026-07-31 に df31431 でユーザー判断により全廃されたが、
    本関数の読み込みブロックが対応漏れで残存していた（レビューで検出）。U/Mの前例に
    倣い、ファイルの有無に関わらずS1の読み込み自体をコードレベルで無効化する。
    """
    picks_dir = Path(__file__).parent.parent / "data" / "picks"

    def _load(suffixes: tuple[str, str]) -> list[dict]:
        out: list[dict] = []
        for fname in suffixes:
            p = picks_dir / fname
            if p.exists():
                try:
                    out += json.loads(p.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"[write_candidates_wt] {fname} 読み込み失敗: {e}", flush=True)
        return out

    rows: list[tuple] = []  # (race_key_store, rank, pred_placeholder, gate_label, n_combos)
    # U(#7U)/M(#7M) は 2026-07-21 全廃・S1(#7S1) は 2026-07-31 全廃のため
    # 読み込み自体を行わない（上記docstring参照）。
    for c in _load((f"wave_picks_wt_{target_date}_s7_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s7_candidates.json")):
        rk = c.get("race_key")
        axis1, axis2 = c.get("axis1"), c.get("axis2")
        if not rk or axis1 is None or axis2 is None:
            continue
        gate_label = rank_7s_gate_label(c.get("wt_overlap_n"), c.get("axis1_class"), c.get("axis2_class"))
        if gate_label is None:
            continue  # 重なり2・不明は候補として表示しない（rank_7s_daily_select と同じ除外対象）
        rows.append((f"{rk}#7S", "RANK_7S", f"{axis1}={axis2}-{_third_list(axis1, axis2, 7)}", gate_label, 0))

    for c in _load((f"wave_picks_wt_{target_date}_s9_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s9_candidates.json")):
        rk = c.get("race_key")
        axis1, axis2 = c.get("axis1"), c.get("axis2")
        if not rk or axis1 is None or axis2 is None:
            continue
        gate_label = rank_7s_gate_label(c.get("wt_overlap_n"), c.get("axis1_class"), c.get("axis2_class"))
        if gate_label is None:
            continue  # S7と同じ基準（重なり2・不明は候補として表示しない）
        rows.append((f"{rk}#9S", "RANK_9S", f"{axis1}={axis2}-{_third_list(axis1, axis2, 9)}", gate_label, 0))

    for c in _load((f"wave_picks_wt_{target_date}_s7a_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s7a_candidates.json")):
        rk = c.get("race_key")
        axis1, axis2 = c.get("axis1"), c.get("axis2")
        if not rk or axis1 is None or axis2 is None:
            continue
        # 7A候補JSON自体が既にrank_7a_daily_select()で境界ケースのみに絞り込み済み
        # （rank_7s_gate_label が None を返すケース）のため、ここでは再フィルタしない。
        rows.append((f"{rk}#7A", "RANK_7A", f"{axis1}={axis2}-{_third_list(axis1, axis2, 7)}", None, 0))

    # 7SS（2026-08-05新設・entropy不合格 × 軸2車が同一ライン）。
    # ⚠️ 2026-08-02に全廃した旧RANK_7SS（波乱軸選出）とは無関係の別物。
    # 買い目は7S/7Aと同じ総流し5点。候補JSONは rank_7ss_daily_select() で
    # 既に絞り込み済みのため、ここでは再フィルタしない。
    for c in _load((f"wave_picks_wt_{target_date}_s7ss_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s7ss_candidates.json")):
        rk = c.get("race_key")
        axis1, axis2 = c.get("axis1"), c.get("axis2")
        if not rk or axis1 is None or axis2 is None:
            continue
        rows.append((f"{rk}#7SS", "RANK_7SS",
                     f"{axis1}={axis2}-{_third_list(axis1, axis2, 7)}", None, 0))

    for c in _load((f"wave_picks_wt_{target_date}_s9a_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s9a_candidates.json")):
        rk = c.get("race_key")
        axis1, axis2 = c.get("axis1"), c.get("axis2")
        if not rk or axis1 is None or axis2 is None:
            continue
        rows.append((f"{rk}#9A", "RANK_9A", f"{axis1}={axis2}-{_third_list(axis1, axis2, 9)}", None, 0))

    # 7B（2026-08-03導入）。他ランクと違い相手を絞る（総流しではない）ため、
    # 候補時点の pred_combo も残り全車ではなく候補JSONの legs_7b（△除外・
    # 複勝指数上位3車）を並べる。発走15分前判定が盤面から相手を再計算するので
    # 欠車があれば最終的に上書きされる。
    for c in _load((f"wave_picks_wt_{target_date}_s7b_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s7b_candidates.json")):
        rk = c.get("race_key")
        axis1, axis2 = c.get("axis1"), c.get("axis2")
        legs = c.get("legs_7b") or []
        if not rk or axis1 is None or axis2 is None or not legs:
            continue
        rows.append((f"{rk}#7B", "RANK_7B",
                     f"{axis1}={axis2}-" + ",".join(str(x) for x in legs), None, 0))

    # 7C（ベースモデル・終日の二軸・2026-08-07新設）。
    # ⚠️ **軸は3ヘッドではなく pred_top3 上位2車**なので `axis1`/`axis2` ではなく
    #    `axis1_7c`/`axis2_7c` を読む（取り違えると別の買い目を表示してしまう）。
    # ⚠️ **総流しではない**。相手は3着内率15%以上に足切りした `legs_7c`（4〜5点・可変）。
    #    `_third_list` を使うと総流し表記になり買い目を偽るので使わない。
    # 候補JSONは rank_7c_daily_select() で合計・点数の両ゲート適用済み。
    # ⚠️ 7C は他ランクと**同一レースに併存する**（wt_overlap_n を見ないため
    #    論理的に排他ではない）。「1レース1ランク」ガードを掛けてはいけない。
    #    重複排除は netkeirin 入稿側だけで行う。
    for c in _load((f"wave_picks_wt_{target_date}_s7c_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s7c_candidates.json")):
        rk = c.get("race_key")
        axis1, axis2 = c.get("axis1_7c"), c.get("axis2_7c")
        # 買う相手は `legs_7c_buy`（三連単=全部 / 三連複=上位2点）。
        # 旧形式の候補JSONは持たないので `legs_7c` へフォールバックする。
        legs = c.get("legs_7c_buy") or c.get("legs_7c") or []
        if not rk or axis1 is None or axis2 is None or not legs:
            continue
        # 🔴 単勝率で三連単へ切り替えるレース（`trifecta_7c`・2026-08-09）は
        #    **表記も変える**。順不同の `軸1=軸2-相手` のままだと、実際には
        #    着順指定で買っている商品を順不同と偽ることになる。
        #    表記は発走前判定（notify_prerace_wt）が作る pred_combo と同一規約。
        if c.get("trifecta_7c"):
            pred = "三単:" + f"{axis1}-{axis2}-" + ",".join(str(x) for x in legs)
        else:
            pred = f"{axis1}={axis2}-" + ",".join(str(x) for x in legs)
        rows.append((f"{rk}#7C", "RANK_7C", pred, None, 0))

    # 7H1（穴推奨・本命バスト型／唯一の2券種ランク・2026-08-07 追加）。
    #
    # 【なぜ必要だったか】7H1 の picks_history 行は発走15分前の判定
    # （notify_prerace_wt._save_rank_7h1_pick）でしか作られず、**朝の推奨一覧に
    # 一切出ない**状態だった（7H1 新設時にこの登録が漏れていた）。
    # 他ランクと同じく朝に bet_amount=0 の暫定行を置き、発走前判定が
    # 買い目・金額を上書きする。
    #
    # ⚠️ **bet_amount は必ず 0 のまま置くこと。** 発走前に「盤面が取れない/欠車で
    #    無効」となった場合、notify_prerace_wt._mark_paper_miwokuri が
    #    `bet_amount = 0` の行だけを見送りへ更新する設計なので、ここで金額を
    #    入れると見送りに落とせず購入済みとして残る。
    # ⚠️ pred_combo の形式は _save_rank_7h1_pick と**完全に一致**させること
    #    （採点・Web 表示の両方が同じ文字列を前提にしている）。
    # ⚠️ 7H1 は他ランクと**同一レースに併存する**（2026-04以降の実データで
    #    7H1+7S 34件・7B+7H1 29件等）。「1レース1ランク」ガードを掛けてはいけない。
    for c in _load((f"wave_picks_wt_{target_date}_s7h1_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s7h1_candidates.json")):
        rk = c.get("race_key")
        trio = c.get("legs_trio") or []
        tf = c.get("legs_tf") or []
        if not rk or not trio or not tf:
            continue
        rows.append((f"{rk}#7H1", "RANK_7H1",
                     "三複:" + ",".join(trio) + " / 三単:" + ",".join(tf),
                     None, len(trio) + len(tf)))

    # 9H1（穴推奨・9車高配当／三連単フォーメーション単一券種・2026-08-08 追加）。
    # 7H1 と同じく朝に bet_amount=0 の暫定行を置き、発走前判定が上書きする。
    # ⚠️ pred_combo の形式は notify_prerace_wt._insert_rank_9h1_pick と
    #    **完全に一致**させること（採点・Web 表示が同じ文字列を前提にしている）。
    # ⚠️ 9H1 も他ランク（9S/9A）と同一レースに併存しうるので
    #    「1レース1ランク」ガードを掛けてはいけない。
    for c in _load((f"wave_picks_wt_{target_date}_s9h1_candidates.json",
                    f"wave_picks_wt_{target_date}_night_s9h1_candidates.json")):
        rk = c.get("race_key")
        legs = c.get("legs") or []
        if not rk or not legs:
            continue
        rows.append((f"{rk}#9H1", "RANK_9H1", "三単:" + ",".join(legs),
                     None, len(legs)))

    if not rows:
        return
    inserted = 0
    try:
        with get_connection() as conn:
            for store_key, rank, pred, gate_label, n_combos in rows:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO picks_history "
                    "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,gate_label) "
                    "VALUES (?,?,?,?,?,0,0,0,0,'wt',False,?)",
                    (target_date, store_key, rank, pred, n_combos, gate_label),
                )
                inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            conn.commit()
    except Exception as e:
        print(f"[write_candidates_wt] ペーパー候補書き込み失敗: {e}", flush=True)
        return
    print(f"[write_candidates_wt] ペーパー候補(7S/7A/7SS/7B/9S/9A/7H1/9H1) {inserted}/{len(rows)} 件書き込み", flush=True)

    # Mac（SQLiteモード）から実行された場合の VPS PG ミラー
    db_url = os.environ.get("KEIRIN_DB_URL")
    if not db_url:
        return
    try:
        import psycopg2  # noqa: PLC0415
        with psycopg2.connect(db_url) as pg_conn:
            with pg_conn.cursor() as cur:
                for store_key, rank, pred, gate_label, n_combos in rows:
                    cur.execute(
                        "INSERT INTO keirin.picks_history "
                        "(race_date,race_key,rank,pred_combo,n_combos,hit,payout,trio_payout,bet_amount,route,miwokuri,gate_label) "
                        "VALUES (%s,%s,%s,%s,%s,0,0,0,0,'wt',FALSE,%s) "
                        "ON CONFLICT (race_key) DO NOTHING",
                        (target_date, store_key, rank, pred, n_combos, gate_label),
                    )
    except Exception as e:
        print(f"[write_candidates_wt] ペーパー候補 VPS ミラー失敗: {e}", flush=True)


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target_date = pos[0] if pos else date.today().strftime("%Y-%m-%d")

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
            except Exception as e:
                print(f"[write_candidates_wt] {fname} 読み込み失敗: {e}", flush=True)

    if not candidates:
        # 旧S1(7PLUS_R)全廃（2026-07-16）以降 candidates.json は常に空リスト。
        # ここで return すると S2/S3 のペーパー候補行まで書かれなくなるため、
        # #CAND 処理だけスキップしてペーパー候補・初回ガミ判定は続行する。
        print(f"[write_candidates_wt] {target_date}: candidates なし（#CANDスキップ）", flush=True)
        try:
            _write_paper_candidates(target_date)
        except Exception as e:
            print(f"[write_candidates_wt] ペーパー候補処理エラー（継続）: {e}", flush=True)
        return

    # 7車レースのみを対象（9車・8車は ROI 構造的に不利のため除外）
    # n_entries を wt_races から一括取得
    all_rks = [c.get("race_key") for c in candidates if c.get("race_key")]
    n_entries_map: dict[str, int] = {}
    if all_rks:
        with get_connection() as conn:
            _phs = ",".join("?" * len(all_rks))
            n_entries_map = {
                r["race_key"]: r["n_entries"]
                for r in conn.execute(
                    f"SELECT race_key, n_entries FROM wt_races WHERE race_key IN ({_phs})",
                    all_rks,
                ).fetchall()
                if r["n_entries"] is not None
            }

    rows: list[tuple] = []
    for cand in candidates:
        rk = cand.get("race_key")
        if not rk:
            continue
        if n_entries_map.get(rk) != 7:
            continue  # 7車以外は推奨対象外
        gap12 = cand.get("gap12", 0.0)
        if gap12 < 0.07:
            continue  # SS候補（gap12 0.07〜0.10）も #CAND として追跡する
        rank = "7PLUS_CAND"
        p1 = cand.get("pivot1")
        p2 = cand.get("pivot2")
        thirds = cand.get("thirds", [])
        pred = f"{p1}-{p2}-" + ",".join(map(str, thirds))
        n_combos = len(thirds)
        store_key = f"{rk}#CAND"
        g12, g34, g23 = _calc_gaps(cand)
        rows.append((target_date, store_key, rank, pred, n_combos, g12, g34, g23))

    with get_connection() as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT race_key FROM picks_history WHERE race_date=? AND route='wt'",
                (target_date,),
            ).fetchall()
        }

    inserted, _ = _insert_candidates_sqlite(target_date, rows, existing)
    _insert_candidates_vps(target_date, rows, existing)

    print(
        f"[write_candidates_wt] {target_date}: {inserted}/{len(rows)} 件書き込み完了",
        flush=True,
    )

    # ペーパー検証ランク（S2/S3）の候補行も書き込む（失敗しても継続）
    try:
        _write_paper_candidates(target_date)
    except Exception as e:
        print(f"[write_candidates_wt] ペーパー候補処理エラー（継続）: {e}", flush=True)

    # 初回ガミ判定（INSERT 後に実行・失敗してもメイン処理には影響しない）
    try:
        _fetch_initial_gami(candidates)
    except Exception as e:
        print(f"[write_candidates_wt] 初回ガミ判定 予期しないエラー: {e}", flush=True)


if __name__ == "__main__":
    main()
