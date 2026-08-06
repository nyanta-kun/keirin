"""3ヘッド軸ガードの対象ランク一覧が実装とずれないことの回帰テスト（2026-08-06）。

## 背景（実際に起きた抜け）

`wt_rebuild_common._THREE_HEAD_RANKS` は「live が3ヘッド軸で動いているので、
旧2ヘッド軸で DELETE→INSERT して塗り潰してはいけない」ランクの集合。
`rebuild_pg_atomic()` はこの集合に載っているランクだけをガードする。

2026-08-05 に 7SS を新設した際、この集合への追加が漏れていた。結果、
**7SS だけは旧軸での tail 再構築が無警告で通る**状態になっていた
（同日に netkeirin_submit_wt.py の RANK_ORDER でも同型の抜けが発生している）。

## 何を守るか

`rebuild_*_walkforward_pg.py` のうち **`axis_is_three_head=True` を渡している
スクリプトのランクは、必ず `_THREE_HEAD_RANKS` に載っていること**。
この2つは「そのランクが3ヘッドで動いている」という同じ事実の別表現であり、
食い違うとガードが静かに無効化される。

逆向き（集合に居るのに誰も3ヘッドで再構築しない）は、まだ対応していない
ランクを先に登録しておく運用がありうるので許容する。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wt_rebuild_common import _THREE_HEAD_RANKS  # noqa: E402

_RANK_LABEL_RE = re.compile(r'^_RANK_LABEL\s*=\s*"([^"]+)"', re.MULTILINE)


def _rebuild_scripts_using_three_head() -> dict[str, Path]:
    """axis_is_three_head=True を渡している rebuild スクリプトの {ランク: パス}。"""
    found: dict[str, Path] = {}
    for path in sorted((ROOT / "scripts").glob("rebuild_*_walkforward_pg.py")):
        src = path.read_text(encoding="utf-8")
        if "axis_is_three_head=True" not in src:
            continue
        m = _RANK_LABEL_RE.search(src)
        assert m, f"{path.name} に _RANK_LABEL の定義が見つからない"
        found[m.group(1)] = path
    return found


def test_three_head_rebuilds_are_all_guarded():
    """3ヘッドで再構築するランクが漏れなくガード対象に入っていること。"""
    using = _rebuild_scripts_using_three_head()
    assert using, "axis_is_three_head=True を使う rebuild スクリプトが1つも無い（検出漏れ）"
    missing = {rank: p.name for rank, p in using.items() if rank not in _THREE_HEAD_RANKS}
    assert not missing, (
        "3ヘッドで再構築しているのに _THREE_HEAD_RANKS に無いランクがある。"
        f"旧軸での塗り潰しが無警告で通る: {missing}"
    )


def test_nine_car_ranks_are_not_three_head():
    """9車ランクは3ヘッドを採用していない（掃引で窓別に符号反転したため）。

    誤って登録すると、9車の正当な tail 再構築がガードで落ちるようになる。
    """
    wrongly_registered = {r for r in _THREE_HEAD_RANKS if r.startswith("RANK_9")}
    assert not wrongly_registered, (
        f"9車ランクが3ヘッド扱いになっている: {wrongly_registered}"
    )
