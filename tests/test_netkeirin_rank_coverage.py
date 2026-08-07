"""netkeirin 入稿の「ランク一覧の二重管理」に対する回帰テスト（2026-08-06）。

## 背景（実際に起きた事故）

`scripts/netkeirin_submit_wt.py` はランクを2箇所に持っていた:
  - `RANK_CONFIGS` … ランクごとの定義（候補JSONのキー・車数・賭け方など）
  - `RANK_ORDER`   … メインループが実際に回す**手書きのリスト**

2026-08-05 に 7SS を新設した際、`RANK_CONFIGS` には追加したが `RANK_ORDER` へ
入れ忘れた。メインループは `RANK_ORDER` を回すため、
**`netkeirin_settings.enabled=True` なのに 7SS が一度も入稿されない**状態になり、
2026-08-06 朝の入稿でユーザーが Web との不一致に気づくまで検知できなかった。

`_is_enabled()` が fail-open（設定行が無いと常時ON）である一方、この抜けは
fail-**closed**（黙って何もしない）だったため、ログにもエラーが出なかった。

## 何を守るか

1. `RANK_ORDER` が `RANK_CONFIGS` を全数カバーすること（順序も定義順と一致）。
2. 各ランクの `file_key` が一意であること（コピペで file_key を使い回すと、
   別ランクの候補を取り込んで誤入稿になる）。
3. `CURRENT_PAPER_RANKS`（strategy_wt の単一正本）に居る7車/9車のペーパーランクが
   入稿側にも存在すること。ここが食い違うと「Webには出るが入稿されない」
   （＝今回の事故）か、その逆が起きる。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.netkeirin_submit_wt import RANK_CONFIGS, RANK_ORDER  # noqa: E402
from src.strategy_wt import CURRENT_PAPER_RANKS  # noqa: E402


def test_rank_order_covers_all_configs():
    """RANK_ORDER が RANK_CONFIGS を全数・同順でカバーすること。

    これが崩れると、定義はあるのに一度も入稿されないランクが生まれる
    （2026-08-05〜08-06 の 7SS がまさにこれ）。
    """
    assert RANK_ORDER == list(RANK_CONFIGS), (
        "RANK_ORDER と RANK_CONFIGS が食い違っている。"
        f"未処理={set(RANK_CONFIGS) - set(RANK_ORDER)} / "
        f"定義なし={set(RANK_ORDER) - set(RANK_CONFIGS)}"
    )


def test_file_keys_are_unique():
    """file_key の重複は別ランクの候補を取り込む誤入稿につながる。"""
    keys = [cfg["file_key"] for cfg in RANK_CONFIGS.values()]
    assert len(keys) == len(set(keys)), f"file_key が重複している: {keys}"


def test_current_paper_ranks_are_submittable():
    """strategy_wt の現行ペーパーランクが入稿側にも存在すること。

    `CURRENT_PAPER_RANKS` は集計・表示の単一正本。ここに載っているランクが
    入稿側に無いと「Webには推奨として出るのに netkeirin には出ない」という
    利用者から見て説明のつかない不一致になる（2026-08-06 にユーザーが
    スクリーンショットで指摘した事象）。

    ⚠️ 入稿するかどうか自体は `netkeirin_settings.enabled` で運用中に切り替える
    （例: 7B は 2026-08-05 に無効化）。本テストが見るのは**定義の存在**であって
    有効/無効ではない。
    """
    # 2026-08-06: RANK_7H1（穴推奨・本命バスト型）の入稿に対応し、除外集合は空に
    # なった。**「なぜ除外されているか」をここに書かずに集合から落とすと、
    # 7SS のときと同じ「無警告で一度も入稿されない」事故になる。**
    NOT_YET_SUBMITTABLE: set[str] = set()
    missing = [spec.label for spec in CURRENT_PAPER_RANKS
               if spec.label not in RANK_CONFIGS
               and spec.label not in NOT_YET_SUBMITTABLE]
    assert not missing, (
        f"CURRENT_PAPER_RANKS にあるが netkeirin 入稿側に定義が無い: {missing}"
    )


def test_normalize_candidate_rejects_identical_axes():
    """自動経路でも axis1 == axis2 を弾くこと（2026-08-08 レビュー指摘）。

    手動入稿 `_process_manual` には同値チェックがあるのに自動経路には無く、
    同じ車が2つ来ると `expand_bet(BET_KIND_TRIO_AXIS2, ...)` が
    要素2つの frozenset を返して**三連複として不正な買い目**を入稿しうる。
    経路ごとに防御が非対称なのは危ういので揃える。
    """
    import pytest

    from scripts.netkeirin_submit_wt import RANK_CONFIGS, _normalize_candidate

    cfg = RANK_CONFIGS["7S"]
    k1, k2 = cfg.get("axis_keys", ("axis1", "axis2"))
    with pytest.raises(ValueError, match="軸1と軸2が同じ"):
        _normalize_candidate({k1: 3, k2: 3}, cfg)

    # 正常系は通ること
    axis1, axis2, partners, marks = _normalize_candidate({k1: 3, k2: 5}, cfg)
    assert (axis1, axis2) == (3, 5)
    assert 3 not in partners and 5 not in partners
