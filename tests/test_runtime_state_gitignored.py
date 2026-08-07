"""実行時に data/ へ書くファイルが .gitignore されていることを検査する。

背景（2026-07-08 の実害 → 2026-08-08 に再発が判明）:
  CI のデプロイは VPS で `git stash -u` してから `git pull` する。**pop しない**ので、
  ignore されていない未追跡ファイルは作業ツリーから**消えたきり戻らない**。

  2026-07-08 に `prerace_decisions_*.json` が消える事故が起きて .gitignore へ
  追加されたが、**そのとき判明した2ファイルにしか適用されず**、同じ性質の
  ファイルが漏れたままだった。VPS の stash 75件を掘ると実際に持ち去られていた:

    data/notify_race_result.lock      21回
    data/notified_race_results.json   13回
    data/netkeirin_session.json        5回   ← ログインセッション
    data/netkeirin_venue_codes.json    5回

  「気づいたものだけ足す」運用では次も漏れるので、**ソースを走査して
  data/ 直下へ書くファイルを見つけ、全件 ignore されているか**を機械的に見る。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# `DATA_DIR / "xxx"` / `"data" / "xxx"` の形でソースに現れる data/ 直下のファイル
_PATTERNS = (
    re.compile(r'DATA_DIR\s*/\s*"([^"/]+\.[a-z0-9]+)"'),
    re.compile(r'"data"\s*/\s*"([^"/]+\.[a-z0-9]+)"'),
)

#: 意図的に**追跡している** data/ 直下のファイル。ignore してはいけないもの。
#: 追加するときは「実行時に書き換わらない・リポジトリで配るべき」ことを確認すること。
_INTENTIONALLY_TRACKED: set[str] = set()


def _referenced_data_files() -> set[str]:
    found: set[str] = set()
    for d in ("src", "scripts"):
        for p in (_REPO / d).rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="replace")
            for pat in _PATTERNS:
                found |= set(pat.findall(text))
    return found


def _is_ignored(rel: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "-q", rel], cwd=_REPO, check=False).returncode == 0


def test_scan_finds_something() -> None:
    """走査が0件だと以降の検査が全部素通りする（この形の偽陽性を何度も踏んでいる）。"""
    files = _referenced_data_files()
    assert files, "ソースから data/ 直下のファイル参照を1件も抽出できなかった"


def test_runtime_state_files_are_gitignored() -> None:
    """data/ 直下へ書くファイルは全て ignore されていること。

    ignore が漏れると deploy の `git stash -u` に持ち去られ、pop されないので
    永久に戻らない（通知済み記録の消失＝再通知、ログインセッションの消失など）。
    """
    missing = sorted(
        f for f in _referenced_data_files()
        if f not in _INTENTIONALLY_TRACKED and not _is_ignored(f"data/{f}")
    )
    assert not missing, (
        f"data/ 直下に書くのに .gitignore されていない: {missing}。"
        " deploy の `git stash -u` に持ち去られて戻らない。"
        " ignore するか、追跡が正しいなら _INTENTIONALLY_TRACKED へ理由付きで足すこと")


@pytest.mark.parametrize("name", [
    "notified_race_results.json", "notify_race_result.lock",
    "netkeirin_session.json", "netkeirin_venue_codes.json",
])
def test_known_victims_stay_ignored(name: str) -> None:
    """実際に持ち去られた実績があるファイルは名指しで固定する。"""
    assert _is_ignored(f"data/{name}"), f"data/{name} の ignore が外れている"
