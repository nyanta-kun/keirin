"""scripts/rebuild_{7s,7a,7ss,7b,9s,9a}_walkforward_pg.py の月初モデル不足ガードの回帰テスト
（2026-08-01・F-4対応）。

背景: 日付が月初に変わった直後、当月のvintageモデル(lgbm_wt_eval_mYYMM等)が
まだ存在しないと、従来は全期間・全窓を計算し終えた最後の窓でFileNotFoundErrorが
発生し、それまでの計算(約40分規模)が丸ごと失われていた。

本テストは6スクリプトすべてについて、以下を実DB・実モデルファイル・実
build_rows(重い計算)に一切触れずに検証する:
  1. モデル不足を検出した場合、`--skip-missing-models`未指定なら
     build_rows()を一切呼ばずに(＝計算を始めずに)SystemExit(1)で終了すること。
  2. `--skip-missing-models`指定時は、不足していない窓のみbuild_rows()を呼び、
     不足窓は除外して続行すること。
  3. 全窓でモデル不足の場合（--skip-missing-models指定時）は処理対象が0件
     になり、rebuild_pg_atomic()を呼ばずSystemExit(1)で終了すること。

conftest.py が `scripts/` を sys.path に追加済みのため、各スクリプトは
トップレベルモジュールとしてimportできる（例: `import rebuild_7s_walkforward_pg`）。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# 3ヘッド軸で再構築するランク（split_by_model_availability に require_bad=True、
# build_rows に bad_model_name を渡す）も必ず含めること。2026-08-05 に 7B を
# 3ヘッド化した際、このリストへ追加しなかったためフェイクが require_bad を
# 受け取れず、7B のガードが一度も検査されていなかった。
_MODULE_NAMES = [
    "rebuild_7s_walkforward_pg",
    "rebuild_7a_walkforward_pg",
    "rebuild_7ss_walkforward_pg",
    "rebuild_7b_walkforward_pg",
    "rebuild_9s_walkforward_pg",
    "rebuild_9a_walkforward_pg",
]

_WINDOW_OK = ("2026-07-01", "2026-07-31", "lgbm_wt_eval_m2607", "lgbm_wt_win_m2607")
_WINDOW_MISSING = ("2026-08-01", "2026-08-01", "lgbm_wt_eval_m2608", "lgbm_wt_win_m2608")


def _fake_split_by_model_availability(windows, require_bad: bool = False):
    """実MODEL_DIR/実ファイルシステムに一切触れないsplit_by_model_availability代替。
    テストが渡す `_WINDOW_OK` / `_WINDOW_MISSING` の等価性だけで仕分ける。

    require_bad: 3ヘッド軸のランク（7A/7SS/7B）が渡してくる。仕分け結果自体は
      窓の等価性だけで決まるので挙動には影響しないが、**受け取れないと
      TypeError でガードそのものが検査できなくなる**ため明示的に受ける。"""
    available = [w for w in windows if w == _WINDOW_OK]
    missing = [(w, ["dummy_missing_eval", "dummy_missing_win"])
               for w in windows if w == _WINDOW_MISSING]
    return available, missing


@pytest.fixture(params=_MODULE_NAMES)
def rebuild_module(request, monkeypatch):
    """各rebuildスクリプトをimportし、重い/危険な依存をすべてmonkeypatchで無害化する。
    実DB・実モデルファイル（data/models/実体）には一切触れない。"""
    mod = importlib.import_module(request.param)

    monkeypatch.setattr(mod, "split_by_model_availability", _fake_split_by_model_availability)

    build_rows_calls: list[tuple] = []

    # bad_model_name: 3ヘッド軸のランクが渡してくる（既定 None で2ヘッドのランクも通す）。
    # pool_history: 7A の低配当見送りゲートが窓をまたいで母集団を引き継ぐために
    # 駆動側が渡す（2026-08-09）。ダブル側で受けないと TypeError になる。
    def fake_build_rows(eval_model, date_from, date_to, win_model_name,
                        bad_model_name=None, pool_history=None):
        build_rows_calls.append((eval_model, date_from, date_to, win_model_name))
        return []  # 行の中身自体は本テストの関心事ではない

    monkeypatch.setattr(mod, "build_rows", fake_build_rows)
    mod._test_build_rows_calls = build_rows_calls  # テスト側から参照するためのフック

    rebuild_calls: list[tuple] = []
    monkeypatch.setattr(
        mod, "rebuild_pg_atomic",
        # **kwargs: 呼び出し側にキーワード引数が増えても（2026-08-04 の
        # allow_legacy_axis など）このスタブが TypeError で落ちないようにする。
        lambda rank_label, cond, per_window_rows, dry_run, **kwargs: rebuild_calls.append(
            (rank_label, cond, per_window_rows, dry_run)),
    )
    mod._test_rebuild_calls = rebuild_calls

    notified: list[str] = []
    monkeypatch.setattr(mod, "notify_discord_warning", lambda msg: notified.append(msg))
    mod._test_notified = notified

    return mod


def test_missing_model_without_skip_flag_aborts_before_any_computation(rebuild_module, monkeypatch):
    """モデル不足検出時、--skip-missing-models未指定ならbuild_rows()を一切呼ばず
    SystemExit(1)で終了すること（全期間計算後に失われる事故の再発防止の核心）。
    """
    mod = rebuild_module
    monkeypatch.setattr(mod, "monthly_windows", lambda upto=None: [_WINDOW_OK, _WINDOW_MISSING])
    monkeypatch.setattr(sys, "argv", [mod.__file__])

    with pytest.raises(SystemExit) as exc_info:
        mod.main()

    assert exc_info.value.code == 1
    assert mod._test_build_rows_calls == [], "モデル不足時は計算を一切開始してはいけない"
    assert mod._test_rebuild_calls == [], "計算を開始していないためDB書き込みも発生しないこと"
    assert mod._test_notified, "モデル不足はDiscordへ通知されること"


def test_skip_missing_models_flag_excludes_missing_window_and_continues(rebuild_module, monkeypatch):
    """--skip-missing-models指定時は、不足していない窓のみbuild_rows()を呼び、
    不足窓は除外して処理を続行すること。"""
    mod = rebuild_module
    monkeypatch.setattr(mod, "monthly_windows", lambda upto=None: [_WINDOW_OK, _WINDOW_MISSING])
    monkeypatch.setattr(sys, "argv", [mod.__file__, "--skip-missing-models"])

    mod.main()  # SystemExitを送出せず正常終了すること

    assert len(mod._test_build_rows_calls) == 1
    called_eval_model = mod._test_build_rows_calls[0][0]
    assert called_eval_model == _WINDOW_OK[2], "モデルが揃っている窓のみ計算されること"
    assert len(mod._test_rebuild_calls) == 1
    assert mod._test_notified, "スキップした旨もDiscordへ通知されること"


def test_all_windows_missing_with_skip_flag_aborts_without_db_write(rebuild_module, monkeypatch):
    """--skip-missing-models指定でも全窓が不足なら、処理対象0件でSystemExit(1)、
    rebuild_pg_atomic()は呼ばれないこと（空のDB書き込みで安全側に倒す）。"""
    mod = rebuild_module
    monkeypatch.setattr(mod, "monthly_windows", lambda upto=None: [_WINDOW_MISSING])
    monkeypatch.setattr(sys, "argv", [mod.__file__, "--skip-missing-models"])

    with pytest.raises(SystemExit) as exc_info:
        mod.main()

    assert exc_info.value.code == 1
    assert mod._test_build_rows_calls == []
    assert mod._test_rebuild_calls == []


def test_no_missing_models_runs_normally(rebuild_module, monkeypatch):
    """モデルが全窓揃っている通常ケースでは、警告なしに全窓が計算されること。"""
    mod = rebuild_module
    monkeypatch.setattr(mod, "monthly_windows", lambda upto=None: [_WINDOW_OK])
    monkeypatch.setattr(sys, "argv", [mod.__file__])

    mod.main()

    assert len(mod._test_build_rows_calls) == 1
    assert len(mod._test_rebuild_calls) == 1
    assert mod._test_notified == [], "モデル不足が無ければDiscord警告は発生しないこと"
