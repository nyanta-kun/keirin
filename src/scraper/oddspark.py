"""
OddsPark競輪スクレイパー
https://www.oddspark.com/keirin/
"""
import logging
import re
from datetime import datetime, date
from typing import Any

from .base import BaseScraper


logger = logging.getLogger(__name__)

# 開催場コード（OddsPark形式）
VENUE_CODES = {
    "11": "函館",
    "12": "青森",
    "13": "いわき平",
    "14": "弥彦",
    "15": "前橋",
    "16": "取手",
    "17": "宇都宮",
    "18": "大宮",
    "19": "西武園",
    "20": "京王閣",
    "21": "立川",
    "22": "松戸",
    "23": "千葉",
    "24": "川崎",
    "25": "平塚",
    "26": "小田原",
    "27": "静岡",
    "28": "名古屋",
    "29": "岐阜",
    "30": "大垣",
    "31": "豊橋",
    "32": "富山",
    "33": "松阪",
    "34": "四日市",
    "35": "福井",
    "36": "奈良",
    "37": "向日町",
    "38": "大阪",
    "39": "和歌山",
    "40": "岸和田",
    "41": "玉野",
    "42": "広島",
    "43": "防府",
    "44": "高松",
    "45": "小倉",
    "46": "佐世保",
    "47": "久留米",
    "48": "別府",
}


class OddsparkScraper(BaseScraper):
    """OddsPark競輪スクレイパー"""

    BASE_URL = "https://www.oddspark.com/keirin"

    def scrape_race_list(self, target_date: str) -> list[dict[str, Any]]:
        """
        指定日の開催情報一覧を取得
        target_date: "YYYY-MM-DD"形式
        """
        url = f"{self.BASE_URL}/OneDayRaceList.do"
        d = datetime.strptime(target_date, "%Y-%m-%d")
        params = {"hldYmd": d.strftime("%Y%m%d")}

        soup = self._get(url, params=params)
        if soup is None:
            return []

        races = []
        try:
            race_links = soup.select("a[href*='RaceList.do']")
            for link in race_links:
                href = link.get("href", "")
                venue_match = re.search(r"sponsorCd=(\d+)", href)
                if venue_match:
                    venue_code = venue_match.group(1)
                    races.append({
                        "date": target_date,
                        "venue_code": venue_code,
                        "venue_name": VENUE_CODES.get(venue_code, "不明"),
                        "url": href if href.startswith("http") else f"https://www.oddspark.com{href}",
                    })
        except Exception as e:
            logger.error(f"Failed to parse race list: {e}")

        logger.info(f"Found {len(races)} venues on {target_date}")
        return races

    def scrape_race_list_by_venue(self, target_date: str, venue_code: str) -> list[dict[str, Any]]:
        """
        指定日・指定開催場のレース一覧を取得
        """
        url = f"{self.BASE_URL}/RaceList.do"
        d = datetime.strptime(target_date, "%Y-%m-%d")
        params = {
            "sponsorCd": venue_code,
            "hldYmd": d.strftime("%Y%m%d"),
        }

        soup = self._get(url, params=params)
        if soup is None:
            return []

        races = []
        try:
            race_links = soup.select("a[href*='RaceTopView.do']")
            for link in race_links:
                href = link.get("href", "")
                race_no_match = re.search(r"raceCd=(\d+)", href)
                if race_no_match:
                    race_no = int(race_no_match.group(1))
                    race_key = f"{target_date.replace('-', '')}_{venue_code}_{race_no:02d}"
                    races.append({
                        "race_key": race_key,
                        "date": target_date,
                        "venue_code": venue_code,
                        "race_no": race_no,
                        "url": href if href.startswith("http") else f"https://www.oddspark.com{href}",
                    })
        except Exception as e:
            logger.error(f"Failed to parse race list for venue {venue_code}: {e}")

        return races

    def scrape_race_detail(self, race_key: str, url: str = None) -> dict[str, Any] | None:
        """
        出走表・選手情報を取得
        race_key: "YYYYMMDD_venue_raceNo"形式
        """
        if url is None:
            parts = race_key.split("_")
            if len(parts) != 3:
                logger.error(f"Invalid race_key format: {race_key}")
                return None
            date_str, venue_code, race_no = parts
            url_base = f"{self.BASE_URL}/RaceTopView.do"
            params = {
                "sponsorCd": venue_code,
                "hldYmd": date_str,
                "raceCd": race_no,
            }
            soup = self._get(url_base, params=params)
        else:
            soup = self._get(url)

        if soup is None:
            return None

        result = {
            "race_key": race_key,
            "entries": [],
            "line_info": [],
        }

        try:
            # 出走表テーブルを解析
            entry_table = soup.select_one("table.raceTable")
            if not entry_table:
                entry_table = soup.select_one("table.syuttohyo")

            if entry_table:
                rows = entry_table.select("tr")[1:]  # ヘッダー行をスキップ
                for row in rows:
                    cols = row.select("td")
                    if len(cols) < 3:
                        continue

                    entry = self._parse_entry_row(cols)
                    if entry:
                        result["entries"].append(entry)

        except Exception as e:
            logger.error(f"Failed to parse race detail {race_key}: {e}")

        return result

    def _parse_entry_row(self, cols: list) -> dict | None:
        """出走表の1行を解析"""
        try:
            frame_no_text = cols[0].get_text(strip=True)
            if not frame_no_text.isdigit():
                return None

            entry = {
                "frame_no": int(frame_no_text),
                "player_name": cols[1].get_text(strip=True) if len(cols) > 1 else "",
                "gear_ratio": None,
                "racing_score": None,
            }

            # 競走得点を探す
            for col in cols:
                text = col.get_text(strip=True)
                if re.match(r"^\d{2,3}\.\d{2}$", text):
                    entry["racing_score"] = float(text)
                    break

            # ギア比を探す（3.xx形式）
            for col in cols:
                text = col.get_text(strip=True)
                if re.match(r"^[34]\.\d{2}$", text):
                    entry["gear_ratio"] = float(text)
                    break

            return entry
        except Exception:
            return None

    def scrape_race_result(self, race_key: str, url: str = None) -> dict[str, Any] | None:
        """レース結果を取得"""
        parts = race_key.split("_")
        if len(parts) != 3:
            return None

        date_str, venue_code, race_no = parts
        url_base = f"{self.BASE_URL}/RaceResult.do"
        params = {
            "sponsorCd": venue_code,
            "hldYmd": date_str,
            "raceCd": race_no,
        }

        soup = self._get(url_base, params=params)
        if soup is None:
            return None

        result = {
            "race_key": race_key,
            "finish_order": [],
            "payouts": {},
        }

        try:
            result_table = soup.select_one("table.resultTable, table.kekka")
            if result_table:
                rows = result_table.select("tr")[1:]
                for row in rows:
                    cols = row.select("td")
                    if len(cols) >= 2:
                        pos_text = cols[0].get_text(strip=True)
                        frame_text = cols[1].get_text(strip=True)
                        if pos_text.isdigit() and frame_text.isdigit():
                            result["finish_order"].append({
                                "position": int(pos_text),
                                "frame_no": int(frame_text),
                            })

            # 払戻金テーブル
            payout_table = soup.select_one("table.payoutTable, table.haraimodoshi")
            if payout_table:
                result["payouts"] = self._parse_payouts(payout_table)

        except Exception as e:
            logger.error(f"Failed to parse race result {race_key}: {e}")

        return result

    def _parse_payouts(self, table) -> dict:
        """払戻金テーブルを解析"""
        payouts = {}
        bet_type_map = {
            "単勝": "win",
            "複勝": "place",
            "2車複": "quinella",
            "2車単": "exacta",
            "3連複": "trifecta_box",
            "3連単": "trifecta",
        }

        try:
            rows = table.select("tr")
            for row in rows:
                cols = row.select("td")
                if len(cols) >= 3:
                    bet_type_text = cols[0].get_text(strip=True)
                    combination = cols[1].get_text(strip=True)
                    payout_text = cols[2].get_text(strip=True).replace(",", "").replace("円", "")

                    for jp_name, en_name in bet_type_map.items():
                        if jp_name in bet_type_text:
                            if payout_text.isdigit():
                                payouts[f"{en_name}_{combination}"] = int(payout_text)
                            break
        except Exception:
            pass

        return payouts

    def scrape_odds(self, race_key: str, bet_type: str = "trifecta_box") -> dict[str, Any] | None:
        """オッズ情報を取得"""
        parts = race_key.split("_")
        if len(parts) != 3:
            return None

        date_str, venue_code, race_no = parts

        bet_type_params = {
            "win": "1",
            "place": "2",
            "quinella": "3",
            "exacta": "4",
            "trifecta_box": "6",
            "trifecta": "7",
        }

        url = f"{self.BASE_URL}/Odds.do"
        params = {
            "sponsorCd": venue_code,
            "hldYmd": date_str,
            "raceCd": race_no,
            "betType": bet_type_params.get(bet_type, "6"),
        }

        soup = self._get(url, params=params)
        if soup is None:
            return None

        odds_data = {"race_key": race_key, "bet_type": bet_type, "odds": {}}

        try:
            odds_table = soup.select_one("table.oddsTable")
            if odds_table:
                rows = odds_table.select("tr")[1:]
                for row in rows:
                    cols = row.select("td")
                    if len(cols) >= 2:
                        combo = cols[0].get_text(strip=True).replace("-", "-")
                        odds_val = cols[-1].get_text(strip=True)
                        if re.match(r"^\d+\.\d+$", odds_val):
                            odds_data["odds"][combo] = float(odds_val)
        except Exception as e:
            logger.error(f"Failed to parse odds {race_key}: {e}")

        return odds_data
