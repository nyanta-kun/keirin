"""winticket.jp スクレイパー

PRELOADED_STATE JSON から入出走表・ラインデータ・オッズ・結果を取得する。
APIは認証不要、SSRページのJSONに全データが埋め込まれている。

URL: https://www.winticket.jp/keirin/{slug}/racecard/{cupId}/{day_index}/{race_no}
     https://www.winticket.jp/keirin/{slug}/odds/{cupId}/{day_index}/{race_no}

cupId = YYYYMMDD(開催初日) + venue_id(2桁)
day_index = 1(初日), 2(2日目), 3(3日目) ...
"""
import json
import re
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

# JKA venue_code → winticket URL slug
VENUE_SLUGS: dict[str, str] = {
    "11": "hakodate",
    "12": "aomori",
    "13": "iwakidaira",
    "21": "yahiko",
    "22": "maebashi",
    "23": "toride",
    "24": "utsunomiya",
    "25": "omiya",
    "26": "seibuen",
    "27": "keiokaku",
    "28": "tachikawa",
    "31": "matsudo",
    "32": "chiba",
    "34": "kawasaki",
    "35": "hiratsuka",
    "36": "odawara",
    "37": "ito",
    "38": "shizuoka",
    "42": "nagoya",
    "43": "gifu",
    "44": "ogaki",
    "45": "toyohashi",
    "46": "toyama",
    "47": "matsusaka",
    "48": "yokkaichi",
    "51": "fukui",
    "53": "nara",
    "54": "mukomachi",
    "55": "wakayama",
    "56": "kishiwada",
    "61": "tamano",
    "62": "hiroshima",
    "63": "hofu",
    "71": "takamatsu",
    "73": "komatsushima",
    "74": "kochi",
    "75": "matsuyama",
    "81": "kokura",
    "83": "kurume",
    "84": "takeo",
    "85": "sasebo",
    "86": "beppu",
    "87": "kumamoto",
}

# (playerCurrentTermClass, playerCurrentTermGroup) → class string
_CLASS_MAP: dict[tuple[int, int], str] = {
    (0, 1): "SS",
    (1, 1): "S1",
    (1, 2): "S2",
    (2, 1): "A1",
    (2, 2): "A2",
    (2, 3): "A3",
    (3, 1): "B",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en;q=0.5",
}

_BASE = "https://www.winticket.jp"


def _extract_state(html: str) -> dict:
    """HTML から window.__PRELOADED_STATE__ の JSON を抽出する。"""
    marker = "window.__PRELOADED_STATE__ = "
    idx = html.find(marker)
    if idx < 0:
        return {}
    start = idx + len(marker)
    raw = html[start:]
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[: i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _get_query(state: dict, key_fragment: str) -> dict | None:
    """tanStackQuery.queries から指定のクエリデータを取得する。"""
    for q in state.get("tanStackQuery", {}).get("queries", []):
        if key_fragment in str(q.get("queryKey", [])):
            return q.get("state", {}).get("data")
    return None


def _parse_lineup(lineup_data: dict) -> dict[int, dict]:
    """
    linePrediction JSON → {frame_no: {line_group, line_size, line_pos}}

    line_group: 1-based index of the line group
    line_size : total riders in the group (including competing sub-entries)
    line_pos  : position within group (1=leader, 2=2nd, ...)
    """
    result: dict[int, dict] = {}
    lines = lineup_data.get("lines", [])
    for group_idx, line in enumerate(lines, 1):
        entries = line.get("entries", [])
        # entries: [{numbers: [n, ...]}, ...] — competing riders share same positional slot
        line_size = sum(len(e.get("numbers", [])) for e in entries)
        pos = 1
        for entry in entries:
            nums = entry.get("numbers", [])
            for fn in nums:
                result[fn] = {
                    "line_group": group_idx,
                    "line_size": line_size,
                    "line_pos": pos,
                    "is_line_leader": pos == 1,
                }
            pos += len(nums) if len(nums) == 1 else 1
    return result


class WinticketScraper:
    """winticket.jp からレースデータを取得するスクレイパー。"""

    def __init__(self, request_interval: float = 1.5):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._interval = request_interval
        self._last_req = 0.0

    def _get(self, url: str, timeout: int = 15) -> requests.Response | None:
        elapsed = time.time() - self._last_req
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        try:
            resp = self._session.get(url, timeout=timeout)
            self._last_req = time.time()
            return resp
        except requests.RequestException as e:
            logger.warning("GET failed %s: %s", url, e)
            self._last_req = time.time()
            return None

    def find_cup_info(self, venue_id: str, target_date: str) -> tuple[str, int] | None:
        """
        venue_id と target_date から (cup_id, day_index) を返す。
        イベント開始日を最大4日前まで試す。

        Returns None if venue not in winticket or no race found.
        """
        slug = VENUE_SLUGS.get(venue_id)
        if not slug:
            return None

        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        for days_back in range(4):
            start_dt = target_dt - timedelta(days=days_back)
            cup_id = start_dt.strftime("%Y%m%d") + venue_id
            day_index = days_back + 1
            url = f"{_BASE}/keirin/{slug}/racecard/{cup_id}/{day_index}/1"
            resp = self._get(url)
            if resp is None or resp.status_code != 200:
                continue

            state = _extract_state(resp.text)
            # Verify this schedule actually contains target_date
            cup_data = _get_query(state, "FETCH_KEIRIN_CUP_RACES")
            if not cup_data:
                continue
            target_ymd = target_date.replace("-", "")  # schedules use YYYYMMDD
            for sched in cup_data.get("schedules", []):
                if str(sched.get("date", "")).replace("-", "") == target_ymd:
                    actual_index = sched.get("index", day_index)
                    logger.debug("Found cup_id=%s day=%s for %s/%s", cup_id, actual_index, venue_id, target_date)
                    return cup_id, actual_index

        return None

    def fetch_race_data(
        self, venue_id: str, target_date: str, race_no: int,
        cup_id: str | None = None, day_index: int | None = None,
    ) -> dict | None:
        """
        1レース分のデータ（エントリー・選手情報・ライン・結果）を取得する。

        Returns dict with keys:
            race_key, venue_id, race_date, race_no,
            race_info, entries (list of per-rider dicts), lineup, results
        Or None on failure.
        """
        slug = VENUE_SLUGS.get(venue_id)
        if not slug:
            return None

        if cup_id is None or day_index is None:
            info = self.find_cup_info(venue_id, target_date)
            if not info:
                return None
            cup_id, day_index = info

        url = f"{_BASE}/keirin/{slug}/racecard/{cup_id}/{day_index}/{race_no}"
        resp = self._get(url)
        if resp is None or resp.status_code != 200:
            return None

        state = _extract_state(resp.text)
        data = _get_query(state, "FETCH_KEIRIN_RACE")
        if not data:
            return None

        race_info = data.get("race", {})
        entries_raw = data.get("entries", [])
        players_raw = {p["id"]: p for p in data.get("players", [])}
        records_raw = {r["playerId"]: r for r in data.get("records", [])}
        lineup_raw = _parse_lineup(data.get("linePrediction", {}))
        n_lines = len(data.get("linePrediction", {}).get("lines", []))
        results_raw = {r["playerId"]: r for r in data.get("results", [])}

        race_date_yyyymmdd = target_date.replace("-", "")
        race_key = f"{race_date_yyyymmdd}_{venue_id}_{race_no:02d}"

        entries = []
        for e in entries_raw:
            if e.get("absent"):
                continue
            player_id = e["playerId"]
            frame_no = e["number"]
            p = players_raw.get(player_id, {})
            rec = records_raw.get(player_id, {})
            lineup_info = lineup_raw.get(frame_no, {})
            result = results_raw.get(player_id, {})

            cls_int = e.get("playerCurrentTermClass", -1)
            grp_int = e.get("playerCurrentTermGroup", -1)
            player_class = _CLASS_MAP.get((cls_int, grp_int), f"cls{cls_int}")

            ex = rec
            entries.append({
                "frame_no":        frame_no,
                "player_id":       player_id,
                "name":            p.get("name", ""),
                "prefecture":      p.get("prefecture", ""),
                "player_class":    player_class,
                "term":            p.get("term", 0),
                "gear_ratio":      rec.get("gearRatio"),
                "style":           rec.get("style", ""),
                "race_point":      rec.get("racePoint"),
                "comment":         rec.get("comment", ""),
                "prediction_mark": rec.get("predictionMark", 0),
                "s_count":         rec.get("standing", 0),
                "h_count":         rec.get("home", 0),
                "b_count":         rec.get("back", 0),
                "front_runner":    rec.get("frontRunner", 0),
                "stalker":         rec.get("stalker", 0),
                "deep_closer":     rec.get("deepCloser", 0),
                "marker":          rec.get("marker", 0),
                "first_rate":      rec.get("firstRate"),
                "second_rate":     rec.get("secondRate"),
                "third_rate":      rec.get("thirdRate"),
                "ex_spurt_pct":    ex.get("exSpurt", {}).get("percentage"),
                "ex_thrust_pct":   ex.get("exThrust", {}).get("percentage"),
                "ex_left_behind_pct": ex.get("exLeftBehind", {}).get("percentage"),
                "ex_split_line_pct":  ex.get("exSplitLine", {}).get("percentage"),
                "ex_snatch_pct":   ex.get("exSnatch", {}).get("percentage"),
                "line_group":      lineup_info.get("line_group", 0),
                "line_size":       lineup_info.get("line_size", 1),
                "line_pos":        lineup_info.get("line_pos", 1),
                "is_line_leader":  int(lineup_info.get("is_line_leader", True)),
                "n_lines":         n_lines,
                "finish_order":    result.get("order"),
                "factor":          result.get("factor", ""),
            })

        return {
            "race_key":  race_key,
            "venue_id":  venue_id,
            "race_date": target_date,
            "race_no":   race_no,
            "cup_id":    cup_id,
            "day_index": day_index,
            "race_info": {
                "start_at":  race_info.get("startAt"),
                "grade":     race_info.get("class", ""),
                "race_type": race_info.get("raceType", ""),
                "distance":  race_info.get("distance"),
                "n_entries": race_info.get("entriesNumber"),
                "status":    race_info.get("status", 0),
                "cancel":    race_info.get("cancel", False),
            },
            "entries": entries,
        }

    def fetch_odds(
        self, venue_id: str, race_date: str, race_no: int,
        cup_id: str, day_index: int,
    ) -> dict[str, list[dict]] | None:
        """
        オッズページから trio(3連複)・trifecta(3連単)・quinella(2車複) を取得する。

        Returns {bet_type: [{combination, odds_value}, ...]}
        or None on failure.
        """
        slug = VENUE_SLUGS.get(venue_id)
        if not slug:
            return None

        url = f"{_BASE}/keirin/{slug}/odds/{cup_id}/{day_index}/{race_no}"
        resp = self._get(url)
        if resp is None or resp.status_code != 200:
            return None

        state = _extract_state(resp.text)
        data = _get_query(state, "FETCH_KEIRIN_RACE_ODDS")
        if not data:
            return None

        def _parse_odds(items: list[dict], bet_type: str) -> list[dict]:
            result = []
            for item in items:
                if item.get("absent"):
                    continue
                combo = item.get("key", "")
                odds_val = item.get("odds")
                # ワイド(quinellaPlace)等のレンジ市場は odds=0、minOdds を採用（保守的な下限）
                if not odds_val:
                    odds_val = item.get("minOdds")
                if combo and odds_val:
                    result.append({"combination": combo, "odds_value": float(odds_val), "bet_type": bet_type})
            return result

        return {
            "trifecta":     _parse_odds(data.get("trifecta", []),     "trifecta"),
            "trio":         _parse_odds(data.get("trio", []),          "trio"),
            "exacta":       _parse_odds(data.get("exacta", []),        "exacta"),
            "quinella":     _parse_odds(data.get("quinella", []),      "quinella"),
            "quinellaPlace":_parse_odds(data.get("quinellaPlace", []),"quinellaPlace"),
        }

    def get_race_count(self, venue_id: str, cup_id: str, day_index: int) -> int:
        """その日のレース数を取得する（スケジュールから）。"""
        slug = VENUE_SLUGS.get(venue_id)
        if not slug:
            return 0
        url = f"{_BASE}/keirin/{slug}/racecard/{cup_id}/{day_index}/1"
        resp = self._get(url)
        if resp is None or resp.status_code != 200:
            return 0
        state = _extract_state(resp.text)
        cup_data = _get_query(state, "FETCH_KEIRIN_CUP_RACES")
        if not cup_data:
            return 0
        races = cup_data.get("races", [])
        count = sum(1 for r in races if str(r.get("scheduleId", "")).endswith(f"{cup_id}"))
        # Fallback: count races in this day's schedule by cross-referencing scheduleId
        if count == 0:
            # Get scheduleId for this day
            for sched in cup_data.get("schedules", []):
                if sched.get("index") == day_index:
                    sched_id = sched.get("id")
                    count = sum(1 for r in races if r.get("scheduleId") == sched_id)
                    break
        return count or 12  # default 12 if unknown
