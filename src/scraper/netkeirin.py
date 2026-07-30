"""netkeirin(keirin.netkeiba.com)の公開ページ取得・ローカルアーカイブモジュール。

netkeirin未活用データ調査（2026-07-28）の一環。方針:
  - 生HTMLは必ずローカル（Mac）へアーカイブしてから解析する（`ARCHIVE_DIR`）。
    今後別のフィールドが必要になった場合、再スクレイピングせずアーカイブ済み
    ページから再解析できるようにするため（VPSは空きディスク28GB・空きメモリ
    99MBと逼迫しており実データ本番と同居させたくない一方、Macは空き164GBと
    余裕があるため保存先はMac側に統一する。memory keirin_vps_mac_architecture 参照）。
  - 実際にDBへ格納するのは「使用が決まったフィールドのみ」。アーカイブ自体は
    使用未確定のフィールドも含め生HTMLをまるごと保持する（後から別フィールドを
    追加解析できるように）。
  - venue_code（keirin.venue_info.venue_code）と netkeirin の jyo_cd は完全一致する
    ことを確認済み（例: 立川=28で両者一致）。別途コード変換テーブルは不要。

認証: race/course・race/data 等の公開ページはログイン不要（requests.get() のみで200・
本文取得可能、2026-07-28確認）。tool.syakenv2.netkeiba.com/bettool（netkeirin入稿
ツール・別モジュール src/netkeirin_client.py）とは別サイトなので混同注意。

ARCHIVE_DIR 配下は data/raw/ 配下（.gitignore で除外済み・「大容量・再生成可能」区分）。
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://keirin.netkeiba.com"
ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "netkeirin"
DEFAULT_REQUEST_INTERVAL = 1.0
_UA = "Mozilla/5.0 (compatible; kiseki-keirin-research/1.0)"

# 見なし直線の表記ゆれ（実測4パターン確認済み・2026-07-28）:
#   「見なし直線は58.0メートル」「見なし直線も50.7メートルと標準的」（久留米）
#   「見なし直線は52mと比較的短い」（高知）「見なし直線は58.6mの典型的な」（松山）
_DEEMED_STRAIGHT_RE = re.compile(r"見なし直線(?:は|も)([\d.]+)(?:メートル|m)")


def _archive_path(url: str) -> Path:
    """URL(パス+クエリ)からアーカイブ先ファイルパスを決める。"""
    safe = quote(url.replace(BASE_URL, "").lstrip("/"), safe="")
    return ARCHIVE_DIR / f"{safe}.html"


def fetch_page(path_and_query: str, *, force: bool = False,
               request_interval: float = DEFAULT_REQUEST_INTERVAL) -> str:
    """netkeirinの1ページを取得する。アーカイブ済みならそれを返し、
    無ければ requests.get() で取得・保存してから返す。

    path_and_query: 例 "/race/course/?jyo_cd=28"（BASE_URL は含めない）
    force: True の場合アーカイブを無視して再取得する。
    """
    url = f"{BASE_URL}{path_and_query}"
    dest = _archive_path(path_and_query)
    if dest.exists() and not force:
        return dest.read_text(encoding="utf-8")

    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")
    logger.info("fetched+archived %s -> %s (%d bytes)", url, dest, len(resp.text))
    if request_interval > 0:
        time.sleep(request_interval)
    return resp.text


def parse_deemed_straight_m(html: str) -> float | None:
    """会場ページ(/race/course/?jyo_cd=)のHTMLから見なし直線(m)を抽出する。"""
    m = _DEEMED_STRAIGHT_RE.search(html)
    return float(m.group(1)) if m else None


def fetch_all_venue_course_pages(venue_codes: list[str], *, force: bool = False) -> dict[str, str]:
    """venue_code一覧に対し /race/course/?jyo_cd={venue_code} をアーカイブする。

    戻り値: {venue_code: html}
    """
    out: dict[str, str] = {}
    for code in venue_codes:
        out[code] = fetch_page(f"/race/course/?jyo_cd={code}", force=force)
    return out
