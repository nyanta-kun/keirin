"""WINTICKETの EX フィールド調査スクリプト (G42)

PRELOADED_STATE JSON に存在するが未抽出の EX フィールドを特定する。
11項目のうち既取得5項目 (exSpurt, exThrust, exLeftBehind, exSplitLine, exSnatch) 以外の
残り6項目のキーを調査する。

Usage:
    python scripts/inspect_winticket_ex_fields.py
"""
import sys
import json
import pprint
from pathlib import Path

# ワークツリー実行時はメインリポジトリを参照する
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
# worktree パスの場合はメインリポジトリに切り替える
if ".claude/worktrees" in str(_REPO_ROOT):
    _REPO_ROOT = Path("/Users/ysuzuki/GitHub/keirin")
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from src.database import get_connection
from src.scraper.winticket import WinticketScraper, _extract_state, _get_query, _BASE, VENUE_SLUGS

KNOWN_EX_FIELDS = {"exSpurt", "exThrust", "exLeftBehind", "exSplitLine", "exSnatch"}


def fetch_raw_records(venue_id: str, race_date: str, race_no: int = 1) -> list[dict]:
    """指定レースの records_raw をそのまま返す（parse前）。"""
    from src.scraper.winticket import _HEADERS
    import time
    import requests

    slug = VENUE_SLUGS.get(venue_id)
    if not slug:
        print(f"[WARN] Unknown venue_id: {venue_id}")
        return []

    scraper = WinticketScraper(request_interval=1.5)

    # find_cup_info で cup_id / day_index を取得
    info = scraper.find_cup_info(venue_id, race_date)
    if not info:
        print(f"[WARN] find_cup_info failed for venue={venue_id} date={race_date}")
        return []

    cup_id, day_index = info
    url = f"{_BASE}/keirin/{slug}/racecard/{cup_id}/{day_index}/{race_no}"
    print(f"[INFO] Fetching: {url}")

    resp = scraper._get(url)
    if resp is None or resp.status_code != 200:
        print(f"[WARN] HTTP {resp.status_code if resp else 'None'}")
        return []

    state = _extract_state(resp.text)
    data = _get_query(state, "FETCH_KEIRIN_RACE")
    if not data:
        print("[WARN] FETCH_KEIRIN_RACE query not found in state")
        return []

    records_raw = data.get("records", [])
    return records_raw


def inspect_ex_fields(records_raw: list[dict]) -> None:
    """records_raw の先頭選手の全キーと ex* キーの詳細を表示する。"""
    if not records_raw:
        print("[ERROR] records_raw is empty — fetch failed")
        return

    print("\n" + "=" * 70)
    print(f"records_raw: {len(records_raw)} 選手分")
    print("=" * 70)

    # 先頭選手の全キーを表示
    sample = records_raw[0]
    print(f"\n--- 選手1 の全キー ({len(sample)} 個) ---")
    pprint.pprint(list(sample.keys()))

    # ex* で始まる全キーを列挙
    ex_keys = [k for k in sample.keys() if k.startswith("ex")]
    print(f"\n--- ex* キー一覧 ({len(ex_keys)} 個) ---")
    for key in ex_keys:
        val = sample[key]
        if val is None:
            print(f"  {key}: None")
        elif isinstance(val, dict):
            sub_keys = list(val.keys())
            print(f"  {key}: dict({sub_keys})")
            # サブキーの値も表示
            for sk, sv in val.items():
                print(f"      .{sk} = {sv!r}")
        else:
            print(f"  {key}: {val!r}")

    print(f"\n--- 既取得5項目 ---")
    for k in sorted(KNOWN_EX_FIELDS):
        status = "取得済み" if k in ex_keys else "NOT FOUND in this record"
        print(f"  {k}: {status}")

    print(f"\n--- 未取得 ex* キー ---")
    new_ex_keys = [k for k in ex_keys if k not in KNOWN_EX_FIELDS]
    if new_ex_keys:
        print(f"  発見: {new_ex_keys}")
        for key in new_ex_keys:
            val = sample[key]
            print(f"\n  === {key} ===")
            pprint.pprint(val)
    else:
        print("  (なし — known 5項目のみ)")

    # 複数選手で出現するキーの統計
    print(f"\n--- 全{len(records_raw)}選手のex*キー出現状況 ---")
    all_ex_keys: set[str] = set()
    for rec in records_raw:
        all_ex_keys.update(k for k in rec.keys() if k.startswith("ex"))
    print(f"  全体で出現するex*キー: {sorted(all_ex_keys)}")

    for key in sorted(all_ex_keys):
        non_null = sum(1 for rec in records_raw if rec.get(key) is not None)
        print(f"    {key}: {non_null}/{len(records_raw)} 選手で非null")


def inspect_all_top_level_keys(records_raw: list[dict]) -> None:
    """全選手にわたる全キーを集約して表示（見落とし防止）。"""
    all_keys: set[str] = set()
    for rec in records_raw:
        all_keys.update(rec.keys())

    print(f"\n--- records_raw の全キー集合 ({len(all_keys)} 個) ---")
    pprint.pprint(sorted(all_keys))


def main():
    print("WINTICKET EX フィールド調査 (G42)")
    print()

    # DBから最新の (venue_id, race_date) を取得
    with get_connection() as conn:
        row = conn.execute(
            "SELECT venue_id, race_date FROM wt_races ORDER BY race_date DESC LIMIT 1"
        ).fetchone()

    if not row:
        print("[ERROR] wt_races テーブルにデータがありません")
        sys.exit(1)

    venue_id = row["venue_id"]
    race_date = row["race_date"]
    print(f"[INFO] 対象: venue_id={venue_id}, race_date={race_date}")

    # フェッチ
    records_raw = fetch_raw_records(venue_id, race_date, race_no=1)

    # EXフィールド調査
    inspect_ex_fields(records_raw)

    # 全キー確認
    inspect_all_top_level_keys(records_raw)

    print("\n調査完了")


if __name__ == "__main__":
    main()
