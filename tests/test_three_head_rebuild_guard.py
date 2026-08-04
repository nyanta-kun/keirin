"""旧軸の再構築が3ヘッド期間を塗り潰さないことを固定する（2026-08-04）。

`rebuild_7{s,a,b}_walkforward_pg.py` → `backfill_7*_rank_wt.py::build_rows()` は
`rank_7s_select_axis(win, top3)` を bad_probs 無しで呼ぶ＝**旧軸**。
THREE_HEAD_AXIS_SINCE 以降を DELETE→INSERT すると live の3ヘッド記録が静かに消える
（S1が第4経路 reconcile_walkforward_tail.sh で自動再生成されていた事故と同型）。
"""
import pytest

from src.strategy_wt import THREE_HEAD_AXIS_SINCE
from src.wt_rebuild_common import rebuild_pg_atomic

ROWS = [{"race_key": "x#7S"}]


def _windows(date_to: str):
    return [("2026-07-01", date_to, ROWS)]


@pytest.mark.parametrize("rank", ["RANK_7S", "RANK_7A", "RANK_7B"])
def test_blocks_seven_car_rebuild_over_three_head_period(rank, monkeypatch):
    """7車ランクは3ヘッド期間を含むと SystemExit で中止する。"""
    sent = []
    monkeypatch.setattr("src.wt_rebuild_common.notify_discord_warning", sent.append)
    with pytest.raises(SystemExit) as e:
        rebuild_pg_atomic(rank, "cond BETWEEN ? AND ?",
                          _windows(THREE_HEAD_AXIS_SINCE), dry_run=True)
    assert e.value.code == 1
    assert sent, "Discordへ警告が飛ぶこと"


@pytest.mark.parametrize("rank", ["RANK_9S", "RANK_9A"])
def test_nine_car_is_not_blocked(rank, monkeypatch):
    """9車は3ヘッドを適用していないため対象外（DBへ到達して落ちることを確認）。"""
    monkeypatch.setattr("src.wt_rebuild_common.notify_discord_warning", lambda *_: None)
    with pytest.raises(Exception) as e:
        rebuild_pg_atomic(rank, "cond BETWEEN ? AND ?",
                          _windows(THREE_HEAD_AXIS_SINCE), dry_run=True)
    assert not (isinstance(e.value, SystemExit) and e.value.code == 1)


def test_allows_when_explicitly_permitted(monkeypatch):
    """--allow-legacy-axis 相当を渡せばガードを通り抜ける（意図的な塗り直し）。"""
    monkeypatch.setattr("src.wt_rebuild_common.notify_discord_warning", lambda *_: None)
    with pytest.raises(Exception) as e:
        rebuild_pg_atomic("RANK_7S", "cond BETWEEN ? AND ?",
                          _windows(THREE_HEAD_AXIS_SINCE), dry_run=True,
                          allow_legacy_axis=True)
    assert not (isinstance(e.value, SystemExit) and e.value.code == 1)


def test_past_only_window_is_untouched(monkeypatch):
    """3ヘッド導入前だけを対象にする再構築は従来どおり通す。"""
    monkeypatch.setattr("src.wt_rebuild_common.notify_discord_warning", lambda *_: None)
    with pytest.raises(Exception) as e:
        rebuild_pg_atomic("RANK_7S", "cond BETWEEN ? AND ?",
                          _windows("2026-07-31"), dry_run=True)
    assert not (isinstance(e.value, SystemExit) and e.value.code == 1)


def test_empty_rows_window_does_not_trigger(monkeypatch):
    """行が空の窓は wipe 対象外なのでガードも発火しない（0件wipeスキップと整合）。

    ⚠️ このケースは「挿入対象0件」の警告経路を通るため、monkeypatch を忘れると
    本番の #システム障害 チャンネルへ実際に投稿される（2026-08-04 に5通投稿する
    事故を起こした）。conftest の `_block_discord` でも二重に塞いでいる。
    """
    monkeypatch.setattr("src.wt_rebuild_common.notify_discord_warning", lambda *_: None)
    rebuild_pg_atomic("RANK_7S", "cond BETWEEN ? AND ?",
                      [("2026-07-01", THREE_HEAD_AXIS_SINCE, [])], dry_run=True)
