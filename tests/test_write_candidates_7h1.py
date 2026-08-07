"""朝の 7H1 暫定行の書き込み（write_candidates_wt.py）。

7H1 は 2026-08-06 の新設時に write_candidates_wt への登録が漏れており、
picks_history の行が発走15分前まで作られず **朝の推奨一覧に一切出なかった**。
本テストはその再発を防ぐ。

⚠️ 最重要の不変条件は **bet_amount = 0**。発走前判定が skip したとき
   `notify_prerace_wt._mark_paper_miwokuri` が `bet_amount = 0` の行だけを
   見送りへ更新するため、ここで金額を入れると見送りに落とせなくなる。
"""
from __future__ import annotations

import json
import re

import pytest

import scripts.write_candidates_wt as wc

DATE = "2026-08-07"
TRIO = ["1=3=7", "1=3=5", "1=3=4", "3=5=7"]
TF = ["7-3-1", "7-3-5", "7-1-3", "7-1-5"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """picks ディレクトリを差し替え、INSERT を捕捉する。"""
    (tmp_path / "scripts").mkdir()
    picks = tmp_path / "data" / "picks"
    picks.mkdir(parents=True)
    captured: list[tuple] = []

    class _Cur:
        rowcount = 1

    class _Conn:
        def execute(self, sql, params):
            captured.append((sql, params))
            return _Cur()

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    # picks_dir は Path(__file__).parent.parent / "data" / "picks" で決まる
    monkeypatch.setattr(wc, "__file__", str(tmp_path / "scripts" / "write_candidates_wt.py"))
    monkeypatch.setattr(wc, "get_connection", lambda: _Conn())
    monkeypatch.delenv("KEIRIN_DB_URL", raising=False)   # VPS ミラーは走らせない
    return picks, captured


def _write_cand(picks, legs_trio, legs_tf, race_key="20260807_13_02"):
    (picks / f"wave_picks_wt_{DATE}_s7h1_candidates.json").write_text(
        json.dumps([{
            "race_key": race_key, "legs_trio": legs_trio, "legs_tf": legs_tf,
            "bet_amount": 9600, "stake_trio": 600, "stake_tf": 900,
        }], ensure_ascii=False), encoding="utf-8")


def _h1_rows(captured):
    return [p for _, p in captured if len(p) > 2 and p[2] == "RANK_7H1"]


def test_7h1_の暫定行が朝に書かれる(env):
    picks, captured = env
    _write_cand(picks, TRIO, TF)
    wc._write_paper_candidates(DATE)

    rows = _h1_rows(captured)
    assert len(rows) == 1, "7H1 の行が書かれていない（新設時に漏れていた箇所）"
    target_date, store_key, rank, pred, n_combos, gate = rows[0]
    assert target_date == DATE
    assert store_key == "20260807_13_02#7H1"
    assert n_combos == len(TRIO) + len(TF)
    # pred_combo は notify_prerace_wt._save_rank_7h1_pick と同一形式でなければ
    # 採点・Web 表示が食い違う
    assert pred == "三複:" + ",".join(TRIO) + " / 三単:" + ",".join(TF)


def test_bet_amount_は0で書かれる(env):
    """発走前 skip 時に見送りへ落とせる唯一の条件。ここを壊してはいけない。"""
    picks, captured = env
    _write_cand(picks, TRIO, TF)
    wc._write_paper_candidates(DATE)

    sql = next(s for s, p in captured if len(p) > 2 and p[2] == "RANK_7H1")
    m = re.search(r"picks_history\s*\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)", sql, re.S)
    assert m, f"INSERT 文を解析できない: {sql}"
    cols = [c.strip() for c in m.group(1).split(",")]
    vals = [v.strip() for v in m.group(2).split(",")]
    assert len(cols) == len(vals)
    assert vals[cols.index("bet_amount")] == "0", "bet_amount が 0 でないと見送りに落とせない"
    assert vals[cols.index("miwokuri")] == "False"


def test_買い目が空の候補は書かない(env):
    picks, captured = env
    _write_cand(picks, [], [])
    wc._write_paper_candidates(DATE)
    assert not _h1_rows(captured)


def test_候補ファイルが無くても他ランクを壊さない(env):
    picks, captured = env
    (picks / f"wave_picks_wt_{DATE}_s7a_candidates.json").write_text(
        json.dumps([{"race_key": "20260807_13_09", "axis1": 1, "axis2": 3}]),
        encoding="utf-8")
    wc._write_paper_candidates(DATE)
    assert not _h1_rows(captured)
    assert [p for _, p in captured if len(p) > 2 and p[2] == "RANK_7A"], "他ランクまで壊れている"
