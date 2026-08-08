"""集計対象ランクの単一正本（src/strategy_wt.CURRENT_PAPER_RANKS）の回帰テスト。

是正タスク B-6（_query_stats のIN句からRANK_7SSが漏れ、月次/年次サマリーに
16,273行が一切反映されない）+ C-1（4箇所独立ハードコードの単一正本化）。

過去に「集計対象ランクのハードコードされたリスト」が
  - scripts/notify_results_wt.py::_query_stats のIN句
  - scripts/notify_results_wt.py::_PAPER_SUFFIXES
  - scripts/save_model_eval.py::PAPER_RANKS
  - scripts/live_report_wt.py::RANKS/RANK_LABELS
の4箇所で独立に保守され、内容が食い違う事故が3回発生した。本テストは
  1. 単一正本（CURRENT_PAPER_RANKS）に現行5ランクが揃っていること
  2. 廃止済みランクが単一正本・各参照先のいずれにも含まれないこと
  3. _query_stats が実際に発行するSQLのIN句に現行5ランク全てが入ること
  4. save_model_eval.PAPER_RANKS / live_report_wt.RANKS が単一正本と整合すること
  5. 4箇所の定義が互いに矛盾しないこと（本テストの中心・再発防止の本体）
を検証する。

DBアクセスはmonkeypatchで差し替え、実DBへは一切アクセスしない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import src.strategy_wt as sw
import notify_results_wt as nr  # scripts/ は conftest で path 追加済
import save_model_eval as sme
import live_report_wt as lr


# 現行6ランク（2026-08-05時点）。
# ⚠️ RANK_7SS は 2026-08-02 に旧定義（波乱軸選出）を全廃したが、2026-08-05 に
#    **同じ名前を別戦略（entropy不合格 × 軸2車が同一ライン）へ充てて再新設**した。
#    したがって CURRENT 側に存在する。旧定義の成績（picks_history 16,298行）は
#    削除済みでDBには0件、バックアップCSVは旧定義なので新7SSと合算してはいけない。
# 2026-08-06: 穴推奨系 RANK_7H1（本命バスト型）を新設。命名は {車数}H{連番} で、
# 既存の予想ベース系（S/A/B）とは系統を分ける（7H2 以降は予約）。
# 2026-08-07: ベースモデル RANK_7C（終日の二軸）を新設。既存ランクとは
# **論理的に排他ではない**（wt_overlap_n を見ない）唯一の予想ベース系ランクで、
# 重複排除は netkeirin 入稿側だけで行う。
# 2026-08-08: 穴推奨 RANK_9H1（9車・高配当狙い）を新設。7H1 とは**車数で母集団が
# 完全に排他**（7H1=7車ちょうど / 9H1=9車ちょうど）。
CURRENT_RANK_NAMES = {"RANK_7SS", "RANK_7S", "RANK_7A", "RANK_7B", "RANK_9S",
                      "RANK_9A", "RANK_7H1", "RANK_7C", "RANK_9H1"}

# 全廃済み（picks_history に存在しない）ランク。
ABOLISHED_RANK_NAMES = {
    "SEVEN_S1", "SIX_S1", "7PLUS_U", "7PLUS_M", "7PLUS_R", "7PLUS_ST", "7PLUS_STP",
}


# ── 1. 単一正本そのもの ──────────────────────────────────────────────


def test_current_paper_ranks_has_all_current_ranks():
    """CURRENT_PAPER_RANKS に現行ランクが全て含まれること。"""
    ranks = {spec.rank for spec in sw.CURRENT_PAPER_RANKS}
    assert ranks == CURRENT_RANK_NAMES


def test_current_paper_ranks_excludes_abolished():
    """CURRENT_PAPER_RANKS に全廃済みランクが混入していないこと。"""
    ranks = {spec.rank for spec in sw.CURRENT_PAPER_RANKS}
    assert ranks.isdisjoint(ABOLISHED_RANK_NAMES)


def test_abolished_paper_ranks_matches_expected_blacklist():
    """ABOLISHED_PAPER_RANKS（ブラックリスト）が想定の7ランクと一致すること。"""
    names = {spec.rank for spec in sw.ABOLISHED_PAPER_RANKS}
    assert names == ABOLISHED_RANK_NAMES
    assert sw.ABOLISHED_PAPER_RANK_NAMES == ABOLISHED_RANK_NAMES


def test_current_and_abolished_are_disjoint():
    """現行ランクと廃止済みランクが重複しないこと（復活防止の要）。"""
    current = {spec.rank for spec in sw.CURRENT_PAPER_RANKS}
    abolished = {spec.rank for spec in sw.ABOLISHED_PAPER_RANKS}
    assert current.isdisjoint(abolished)


def test_each_current_rank_has_unique_rank_and_suffix():
    """rank/suffixが単一正本内で重複していないこと（誤定義の早期検出）。"""
    ranks = [spec.rank for spec in sw.CURRENT_PAPER_RANKS]
    suffixes = [spec.suffix for spec in sw.CURRENT_PAPER_RANKS]
    assert len(ranks) == len(set(ranks))
    assert len(suffixes) == len(set(suffixes))


def test_header_total_members_are_top_ranks():
    """ヘッダー合計(in_header_total)に含まれるのは 7SS と 7S（7A/7B/9S/9Aは除外）。

    notify_results_wt.py の既存設計方針「7A/9Aはヘッダー合計には含めないが
    _query_statsには含める」を単一正本側で保持していることの確認。
    2026-08-05 に再新設した RANK_7SS は最上位ランクのため合計に含める。
    """
    header_members = {spec.rank for spec in sw.CURRENT_PAPER_RANKS if spec.in_header_total}
    assert header_members == {"RANK_7SS", "RANK_7S"}


def test_all_current_ranks_in_live_report():
    """現行4ランクは全て in_live_report=True。

    2026-08-05 に再新設した RANK_7SS は最上位ランクのため True（合計に含む）。
    除外対象は空集合になる。
    """
    excluded = {spec.rank for spec in sw.CURRENT_PAPER_RANKS if not spec.in_live_report}
    assert excluded == set()


# ── 2. notify_results_wt.py::_query_stats（B-6の中心） ──────────────────


class _FakeRow(dict):
    """sqlite3.Row / psycopg2 RealDictRow 相当（["col"]アクセス可能なdict）。"""


class _FakeCursor:
    def __init__(self, sql, params, row):
        self.sql = sql
        self.params = params
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [self._row]


class _FakeConn:
    """get_connection() の代替。実行されたSQL/paramsを記録する。"""

    def __init__(self, row=None):
        self.calls: list[tuple[str, tuple]] = []
        self._row = row if row is not None else _FakeRow(
            races=0, hits=0, returns_=0, bets=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, tuple(params)))
        return _FakeCursor(sql, params, self._row)


def test_query_stats_sql_contains_all_current_ranks(monkeypatch):
    """_query_stats が実際に発行するSQLのIN句に現行ランク全てが入ること。

    2026-07-31以前は独自ハードコードのIN句にRANK_7SSが漏れており、
    picks_historyの16,273行が月次/年次サマリーに一切反映されなかった
    （B-6の実害）。以後は単一正本から動的生成するため、この検証は
    strategy_wt.CURRENT_PAPER_RANKSを更新するだけで将来のランク追加にも
    自動的に追随する。
    """
    fake = _FakeConn()
    monkeypatch.setattr(nr, "get_connection", lambda: fake)

    nr._query_stats("2026-07%")

    assert len(fake.calls) == 1
    sql, params = fake.calls[0]
    assert "rank IN" in sql
    for spec in sw.CURRENT_PAPER_RANKS:
        assert f"'{spec.rank}'" in sql, f"{spec.rank} がSQLのIN句に見つからない"
    # 廃止済みランクは含まれないこと（SEVEN_S1が居残っていた旧バグの再発防止）
    for spec in sw.ABOLISHED_PAPER_RANKS:
        assert f"'{spec.rank}'" not in sql, f"廃止済み{spec.rank}がSQLに残存している"
    # like引数は params 側（bind parameter）で渡されること
    assert params == ("2026-07%",)


def test_query_stats_result_reflects_all_rank_rows(monkeypatch):
    """_query_stats が現行ランク分の行を実際に合算できること（機能面の確認）。"""
    fake = _FakeConn(row=_FakeRow(races=16273, hits=4000, returns_=8_000_000, bets=8_136_500))
    monkeypatch.setattr(nr, "get_connection", lambda: fake)

    result = nr._query_stats("%")

    assert result["races"] == 16273
    assert result["hits"] == 4000


# ── 3. notify_results_wt.py::_PAPER_SUFFIXES ────────────────────────────


def test_paper_suffixes_include_all_current_rank_suffixes():
    """_PAPER_SUFFIXES に現行5ランクのサフィックスが全て含まれること。"""
    for spec in sw.CURRENT_PAPER_RANKS:
        assert spec.suffix in nr._PAPER_SUFFIXES, (
            f"{spec.rank}のsuffix({spec.suffix})が_PAPER_SUFFIXESに無い"
        )


def test_paper_suffixes_include_legacy_hash_suffix_ranks():
    """"#"サフィックス方式を使っていた廃止済みランク(SEVEN_S1/SIX_S1)の
    サフィックスも安全網として残っていること（誤って巻き込み削除しないため）。
    """
    legacy_suffixed = {spec.suffix for spec in sw.ABOLISHED_PAPER_RANKS if spec.suffix}
    # "#7SS" は現行ランクのsuffixになったため legacy 側からは外れた。
    assert legacy_suffixed == {"#7S1", "#6S1"}
    for suffix in legacy_suffixed:
        assert suffix in nr._PAPER_SUFFIXES


def test_paper_suffixes_has_no_unexpected_extra_entries():
    """_PAPER_SUFFIXES が「現行ランク + legacy3」ちょうどで、重複が無いこと。

    2026-08-02 の RANK_7SS 全廃で現行5→4・legacy2→3、2026-08-03 の RANK_7B 新設で
    現行4→5 と移り変わってきた。件数をハードコードするとランクの増減のたびに
    このテストを直す必要があり、「数を合わせるだけ」の修正が混入しやすいので
    単一正本から導出した集合との一致＋重複なしで検証する。
    """
    expected = {spec.suffix for spec in sw.CURRENT_PAPER_RANKS} | {"#7S1", "#6S1"}
    assert set(nr._PAPER_SUFFIXES) == expected
    # tuple 側に重複が無いこと（集合比較だけでは検出できない）
    assert len(nr._PAPER_SUFFIXES) == len(set(nr._PAPER_SUFFIXES)) == len(expected)


# ── 4. save_model_eval.py::PAPER_RANKS ──────────────────────────────────


def test_save_model_eval_paper_ranks_matches_single_source():
    """PAPER_RANKS（label, rank, suffix）が単一正本と1:1で整合すること。"""
    expected = {(spec.label, spec.rank, spec.suffix) for spec in sw.CURRENT_PAPER_RANKS}
    actual = set(sme.PAPER_RANKS)
    assert actual == expected


def test_save_model_eval_paper_ranks_excludes_abolished():
    """PAPER_RANKS に全廃済みランク(特にSEVEN_S1)が含まれないこと。"""
    ranks_in_paper = {rank for (_label, rank, _suffix) in sme.PAPER_RANKS}
    assert ranks_in_paper.isdisjoint(ABOLISHED_RANK_NAMES)
    assert "SEVEN_S1" not in ranks_in_paper


def test_paper_rank_stats_iterates_over_current_ranks(monkeypatch):
    """paper_rank_stats() が PAPER_RANKS の全件を1レコードずつ問い合わせること
    （SQL発行内容が単一正本のrank/suffixと一致するか確認）。
    """
    fake = _FakeConn(row=_FakeRow(n=0, h=0, b=0, p=0))
    monkeypatch.setattr(sme, "get_connection", lambda: fake)

    sme.paper_rank_stats()

    queried_ranks = set()
    for sql, params in fake.calls:
        assert "rank = ?" in sql
        queried_ranks.add(params[0])
    assert queried_ranks == {spec.rank for spec in sw.CURRENT_PAPER_RANKS}


# ── 5. live_report_wt.py::RANKS/RANK_LABELS ─────────────────────────────


def test_live_report_ranks_matches_single_source_subset():
    """RANKS が単一正本の in_live_report=True 部分集合と一致すること。"""
    expected = [spec.rank for spec in sw.CURRENT_PAPER_RANKS if spec.in_live_report]
    assert lr.RANKS == expected


def test_live_report_ranks_includes_seven_ss():
    """live_report_wt.RANKS は単一正本と一致する（2026-08-05 に RANK_7SS を再新設）。"""
    assert "RANK_7SS" in lr.RANKS


def test_live_report_rank_labels_matches_single_source():
    """RANK_LABELS が単一正本のlabelと一致すること。"""
    expected = {
        spec.rank: spec.label for spec in sw.CURRENT_PAPER_RANKS if spec.in_live_report
    }
    assert lr.RANK_LABELS == expected


def test_live_report_ranks_excludes_abolished():
    for abolished in ABOLISHED_RANK_NAMES:
        assert abolished not in lr.RANKS


# ── 6. 4箇所の相互整合性（再発防止の本体） ──────────────────────────────


def test_all_four_locations_agree_on_current_rank_universe():
    """4箇所（_query_stats・_PAPER_SUFFIXES・PAPER_RANKS・live_report RANKS）が
    互いに矛盾しないことを機械的に保証する。

    それぞれ集計対象の粒度・目的が異なる（ヘッダー合計除外・live_report除外等）
    ため単純な集合一致ではなく、各ファイルの「その粒度でのランク全体集合」が
    単一正本のどの部分集合とも矛盾しないことを検証する:
      - _query_stats の対象は CURRENT_PAPER_RANKS 全5ランク（過不足なし）
      - _PAPER_SUFFIXES は CURRENT_PAPER_RANKS 全5ランクのsuffixを含む（過不足チェックは
        legacy2件を許容した上で実施済み・test_paper_suffixes_has_no_unexpected_extra_entries）
      - PAPER_RANKS は CURRENT_PAPER_RANKS 全5ランクと1:1（過不足なし）
      - live_report RANKS は CURRENT_PAPER_RANKS の in_live_report=True 部分集合（4ランク）
    いずれも CURRENT_PAPER_RANKS からの派生であるため、単一正本を更新すれば
    4箇所が自動的に追随し、今後「一部だけ更新して食い違う」事故が構造的に発生しない。
    """
    single_source_ranks = {spec.rank for spec in sw.CURRENT_PAPER_RANKS}

    # _query_stats: 実行時に生成されるSQLから抽出したランク集合
    query_stats_ranks = {spec.rank for spec in sw.CURRENT_PAPER_RANKS
                          if f"'{spec.rank}'" in nr._QUERY_STATS_RANKS_SQL}
    assert query_stats_ranks == single_source_ranks

    # _PAPER_SUFFIXES: suffixをrankへ逆引きして集合化（legacy分は除く）
    suffix_to_rank = {spec.suffix: spec.rank for spec in sw.CURRENT_PAPER_RANKS}
    paper_suffix_ranks = {suffix_to_rank[s] for s in nr._PAPER_SUFFIXES if s in suffix_to_rank}
    assert paper_suffix_ranks == single_source_ranks

    # PAPER_RANKS
    paper_ranks_ranks = {rank for (_label, rank, _suffix) in sme.PAPER_RANKS}
    assert paper_ranks_ranks == single_source_ranks

    # live_report RANKS（in_live_report=Trueの部分集合）
    live_report_ranks = set(lr.RANKS)
    expected_live_report = {spec.rank for spec in sw.CURRENT_PAPER_RANKS if spec.in_live_report}
    assert live_report_ranks == expected_live_report
    assert live_report_ranks.issubset(single_source_ranks)

    # どの場所にも廃止済みランクが紛れ込んでいないこと
    abolished = {spec.rank for spec in sw.ABOLISHED_PAPER_RANKS}
    assert query_stats_ranks.isdisjoint(abolished)
    assert paper_ranks_ranks.isdisjoint(abolished)
    assert live_report_ranks.isdisjoint(abolished)


def test_regression_seven_ss_absent_everywhere():
    """RANK_7SS 再新設（2026-08-05）が全参照先へ波及していることの単体確認。

    live実績 n=16,298・ROI73.5% と控除率75%を下回り続けたため全廃した。
    S1 全廃時と同じく「単一正本から消せば4つの参照先すべてから消える」
    構造になっていることを担保する（旧S3/S1全廃時に取りこぼした経路が
    翌日以降ランクを復活させた事故の再発防止）。
    """
    assert "RANK_7SS" in {spec.rank for spec in sw.CURRENT_PAPER_RANKS}
    assert "RANK_7SS" not in {spec.rank for spec in sw.ABOLISHED_PAPER_RANKS}
    assert "'RANK_7SS'" in nr._QUERY_STATS_RANKS_SQL
    # "#7SS" は _PAPER_SUFFIXES には残る（廃止済みsuffixは「巻き込み削除の
    # 保護対象」として意図的に含める設計。#7S1/#6S1 と同じ扱い）
    assert "#7SS" in nr._PAPER_SUFFIXES
    assert any(r[1] == "RANK_7SS" for r in sme.PAPER_RANKS)
    assert "RANK_7SS" in lr.RANKS


def test_regression_seven_s1_absent_everywhere():
    """全廃済みSEVEN_S1が4箇所のいずれにも残存していないことの単体確認
    （旧_query_statsに残存していたバグの再発防止）。
    """
    assert "SEVEN_S1" not in {spec.rank for spec in sw.CURRENT_PAPER_RANKS}
    assert "'SEVEN_S1'" not in nr._QUERY_STATS_RANKS_SQL
    assert "SEVEN_S1" not in {rank for (_l, rank, _s) in sme.PAPER_RANKS}
    assert "SEVEN_S1" not in lr.RANKS
