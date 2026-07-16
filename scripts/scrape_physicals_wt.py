"""身体測定データスクレイパー（keirin.jp → data/player_physicals.csv）

Usage:
  python3 scripts/scrape_physicals_wt.py           # 全選手をスクレイプ（再開可能）
  python3 scripts/scrape_physicals_wt.py --stats   # 取得済みの統計表示
  python3 scripts/scrape_physicals_wt.py --limit 5 # テスト用（5人だけ）

keirin.jp の選手プロフィールページ（静的HTML）から身体測定データを取得。
URL: https://keirin.jp/pc/racerprofile?snum={player_id:06d}

取得項目:
  height_cm        : 身長 (例: 168.5cm → 168.5)
  weight_kg        : 体重 (例: 73.0kg → 73.0)
  back_strength_kg : 背筋力 (例: 164.0kg → 164.0)
  lung_capacity_cc : 肺活量 (例: 4500cc → 4500.0; "-" → None)
  thigh_cm         : 太もも周径 (例: 64.0cm → 64.0)
  chest_cm         : 胸囲 (例: 103.0cm → 103.0)

HTML 構造（確認済み: 2026-06-15）:
  <p class="midasi2_fsz">■身長・体重・体力等</p>
  <table>
    <tr>
      <td class="tbl_header">星座</td> <td>九星</td> <td>血液型</td>
      <td class="tbl_header">身長</td> <td class="tbl_header">体重</td> ...
    </tr>
    <tr>
      <td>双子座</td> <td>一白</td> <td>O</td>
      <td>168.5cm</td> <td>73.0kg</td> ...
    </tr>
    <tr>
      <td class="tbl_header">胸囲</td> <td>太股</td> <td>背筋力</td> <td>肺活量</td>
    </tr>
    <tr>
      <td>103.0cm</td> <td>64.0cm</td> <td>164.0kg</td> <td>-</td>
    </tr>
  </table>

注意: 「太股」は「太もも」の略称（keirin.jp での表記）。
robots.txt は /pc/ を ALLOW。2 req/sec 遵守。
"""
import sys, re, time, csv, argparse
from pathlib import Path

# ワークツリー内でも本番DBを参照できるよう、リポジトリルートを特定する。
# 本スクリプトは scripts/ に置かれているため、親の親がリポジトリルート候補。
# keirin.db が data/ 配下に存在するディレクトリを優先的に使用する。
_script_dir = Path(__file__).resolve().parent
_candidates = [
    _script_dir.parent,                                      # 同一ツリーのルート
    Path("/Users/ysuzuki/GitHub/keirin"),                    # 本番リポジトリ（絶対パス）
]
for _repo_root in _candidates:
    _db = _repo_root / "data" / "keirin.db"
    if _db.exists() and _db.stat().st_size > 10_000:
        break
sys.path.insert(0, str(_repo_root))

import requests
from src.database import get_connection

OUTPUT = _repo_root / "data" / "player_physicals.csv"
BASE_URL = "https://keirin.jp/pc/racerprofile?snum={snum}"
SLEEP = 0.5  # 2 req/sec

FIELDS = ["height_cm", "weight_kg", "back_strength_kg",
          "lung_capacity_cc", "thigh_cm", "chest_cm"]

# HTML ラベル → CSV フィールドのマッピング
# keirin.jp は「太股」と表記（「太もも周径」の略）
LABEL_MAP = {
    "身長":   "height_cm",
    "体重":   "weight_kg",
    "背筋力": "back_strength_kg",
    "肺活量": "lung_capacity_cc",
    "太股":   "thigh_cm",
    "胸囲":   "chest_cm",
}


def _parse_value(raw: str) -> float | None:
    """'168.5cm' / '73.0kg' / '4500cc' / '-' → float or None."""
    raw = raw.strip()
    if raw in ("-", "", "―", "—"):
        return None
    # 数値部分を抽出（単位を除去）
    m = re.search(r"[\d.]+", raw)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _extract_table_values(html: str) -> dict[str, float | None]:
    """■身長・体重・体力等 セクションの表を解析して値を返す。"""
    result: dict[str, float | None] = {f: None for f in FIELDS}

    section_marker = "■身長・体重・体力等"
    idx = html.find(section_marker)
    if idx == -1:
        return result

    # セクション後の <table> を切り出す（次の </table> まで）
    tbl_start = html.find("<table", idx)
    tbl_end   = html.find("</table>", tbl_start) + len("</table>")
    if tbl_start == -1 or tbl_end < tbl_start:
        return result

    snippet = html[tbl_start:tbl_end]

    # <tr> ブロックを列挙
    rows_raw = re.split(r"<tr[^>]*>", snippet)[1:]  # 最初の空要素を除く

    headers: list[str] = []
    for row_html in rows_raw:
        # <td> の中身を取得
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]

        if not cells:
            continue

        # ヘッダ行かどうか判定（class="tbl_header" を含む）
        is_header = "tbl_header" in row_html

        if is_header:
            headers = cells
        else:
            # データ行: headers と zip して値をマッピング
            if not headers:
                continue
            for hdr, val in zip(headers, cells):
                field = LABEL_MAP.get(hdr)
                if field is not None:
                    result[field] = _parse_value(val)
            headers = []  # 次のヘッダブロックに備えてリセット

    return result


def get_player_ids() -> list[int]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT player_id FROM wt_entries ORDER BY player_id"
        ).fetchall()
    return [r[0] for r in rows]


def scrape_physicals(player_id: int, session: requests.Session) -> dict[str, float | None]:
    """身体測定データを取得する。HTTP エラーや解析失敗時は全 None を返す。"""
    snum = f"{player_id:06d}"
    empty = {f: None for f in FIELDS}
    try:
        r = session.get(BASE_URL.format(snum=snum), timeout=10)
        if r.status_code != 200:
            return empty
        return _extract_table_values(r.text)
    except Exception:
        return empty


def main():
    parser = argparse.ArgumentParser(description="keirin.jp 身体測定スクレイパー")
    parser.add_argument("--limit", type=int, default=None, help="テスト用: 最初の N 人だけ処理")
    parser.add_argument("--stats", action="store_true", help="取得済みデータの統計表示")
    args = parser.parse_args()

    if args.stats:
        if not OUTPUT.exists():
            print("No data yet. Run without --stats first.")
            return
        rows = list(csv.DictReader(open(OUTPUT, encoding="utf-8")))
        n = len(rows)
        print(f"取得済み選手数 : {n}")
        for field in FIELDS:
            has = sum(1 for r in rows if r.get(field, "").strip() not in ("", "None", "nan"))
            pct = 100 * has / n if n > 0 else 0.0
            print(f"  {field:<20}: {has:>5} / {n}  ({pct:.1f}%)")
        return

    all_ids = get_player_ids()
    if args.limit:
        all_ids = all_ids[: args.limit]

    # 既取得の player_id をスキップ（再開可能）
    done: set[int] = set()
    if OUTPUT.exists():
        for row in csv.DictReader(open(OUTPUT, encoding="utf-8")):
            try:
                done.add(int(row["player_id"]))
            except (ValueError, KeyError):
                pass

    todo = [p for p in all_ids if p not in done]
    print(f"Total: {len(all_ids)}  Done: {len(done)}  Remaining: {len(todo)}")

    if not todo:
        print("All done. Run with --stats for coverage summary.")
        return

    write_header = not OUTPUT.exists()
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )

    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["player_id"] + FIELDS)

        for i, pid in enumerate(todo):
            vals = scrape_physicals(pid, session)
            writer.writerow([pid] + [vals[fld] for fld in FIELDS])
            f.flush()
            if (i + 1) % 50 == 0 or (i + 1) == len(todo):
                print(f"  {i + 1}/{len(todo)}", flush=True)
            time.sleep(SLEEP)

    print("Done. Run with --stats for coverage summary.")


if __name__ == "__main__":
    main()
