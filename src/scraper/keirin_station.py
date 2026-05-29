"""
競輪ステーション スクレイパー
https://keirin-station.com/

URL構造:
  開催情報: /keirindb/stadium/information/{venue_code}/{yyyymmdd}/
  出走表:   /keirindb/race/member/{venue_code}/{yyyymmdd}/{race_no}/
  オッズ:   /keirindb/race/odds/{venue_code}/{yyyymmdd}/{race_no}/{bet_type_no}/
  結果:     /keirindb/race/result/{venue_code}/{yyyymmdd}/{race_no}/

venue_codeはkeirin-station.com固有の数字（JKA公式コードとは異なる）
サイト実測値: code 11-96 (一部欠番あり)
"""
import logging
import re
from typing import Any

from .base import BaseScraper

# --- コンパイル済み正規表現 ---
_RE_GEAR     = re.compile(r"\b([34]\.\d{2})\b")
_RE_RATES    = re.compile(r"(\d+\.\d+)%.*?(\d+\.\d+)%.*?(\d+\.\d+)%")
_RE_PERIOD   = re.compile(r"(\d{2,3})期")
_RE_SCORE    = re.compile(r"\b(\d{2,3}\.\d{1,2})\b")
_RE_NAME     = re.compile(r"([ァ-ヴ\s]+)([一-龯\s]{2,8})")
# 都道府県: 全47都道府県を明示列挙（[一-龯]{2} フォールバックを廃止）
_RE_PREF     = re.compile(
    r"(北海道|青\s*森|岩\s*手|宮\s*城|秋\s*田|山\s*形|福\s*島"
    r"|東\s*京|神奈川|埼\s*玉|千\s*葉|茨\s*城|栃\s*木|群\s*馬"
    r"|新\s*潟|長\s*野|山\s*梨|静\s*岡|愛\s*知|岐\s*阜|三\s*重"
    r"|大\s*阪|兵\s*庫|京\s*都|奈\s*良|滋\s*賀|和歌山"
    r"|鳥\s*取|島\s*根|岡\s*山|広\s*島|山\s*口"
    r"|徳\s*島|香\s*川|愛\s*媛|高\s*知"
    r"|福\s*岡|佐\s*賀|長\s*崎|熊\s*本|大\s*分|宮\s*崎|鹿児島|沖\s*縄)"
)
# 登録クラス: S単体は存在しないので S1/S2/SS のみ
_RE_CLASS    = re.compile(r"\b(SS|S[12]|A[123]|B)\b")


logger = logging.getLogger(__name__)

BASE_URL = "https://keirin-station.com"

VENUE_CODES = {
    # 北海道・東北
    "11": "函館",  "12": "青森",   "13": "いわき平", "14": "会津",
    "15": "八戸",  "16": "六郷",   "17": "宮城",
    # 関東
    "21": "弥彦",  "22": "前橋",   "23": "取手",   "24": "宇都宮",
    "25": "大宮",  "26": "西武園", "27": "京王閣", "28": "立川",
    # 関東〜東海
    "31": "松戸",  "32": "千葉",   "34": "川崎",   "35": "平塚",
    "36": "小田原","37": "伊東",   "38": "静岡",
    # 東海〜北陸
    "41": "一宮",  "42": "名古屋", "43": "岐阜",   "44": "大垣",
    "45": "豊橋",  "46": "富山",   "47": "松阪",   "48": "四日市",
    "51": "福井",
    # 近畿
    "52": "大津",  "53": "奈良",   "54": "向日町", "55": "和歌山",
    "56": "岸和田","57": "大阪",
    # 中国
    "61": "玉野",  "62": "広島",   "63": "防府",   "64": "松江",
    # 四国
    "71": "高松",  "72": "観音寺", "73": "小松島", "74": "高知",
    "75": "松山",
    # 九州
    "81": "小倉",  "82": "門司",   "83": "久留米", "84": "武雄",
    "85": "佐世保","86": "別府",   "87": "熊本",   "88": "長崎",
    "89": "福岡",
}

# 賭式番号
BET_TYPE_NO = {
    "trifecta_box": "3",   # 3連複
    "trifecta": "4",       # 3連単
    "quinella": "1",       # 2車複
    "exacta": "2",         # 2車単
    "win": "5",            # 単勝
    "place": "6",          # 複勝
    "wide": "7",           # ワイド
}


class KeirinStationScraper(BaseScraper):
    """競輪ステーション スクレイパー"""

    def scrape_schedule(self, year: int, month: int) -> list[dict[str, Any]]:
        """
        指定年月の開催スケジュールを取得
        Returns: [{venue_code, venue_name, date, event_name}, ...]
        """
        url = f"{BASE_URL}/keirindb/search/race/"
        data = {
            "race_stage[grade_id]": "",
            "race_stage[area_id]": "",
            "race_stage[stadium_id]": "",
            "race_stage[held_year]": str(year),
            "race_stage[held_month]": str(month),
            "race_stage[held_day]": "",
            "race_stage[race_name]": "",
            "submit[btn][race_stage][get]": "この条件で検索する",
        }

        import requests
        self._wait()
        try:
            resp = self.session.post(url, data=data, timeout=30)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
        except Exception as e:
            logger.error(f"Failed to get schedule {year}/{month}: {e}")
            return []

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")

        schedules = []
        tables = soup.find_all("table")
        if len(tables) < 2:
            return []

        result_table = tables[1]
        rows = result_table.find_all("tr")[1:]  # ヘッダー行スキップ

        for row in rows:
            links = row.find_all("a")
            cols = row.find_all("td")
            if not links or len(cols) < 5:
                continue

            event_link = links[0]
            href = event_link.get("href", "")
            # URL例: /keirindb/stadium/information/21/20260422/
            m = re.search(r"/information/(\d+)/(\d{8})/", href)
            if not m:
                continue

            venue_code = m.group(1)
            date_str = m.group(2)
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

            schedules.append({
                "venue_code": venue_code,
                "venue_name": VENUE_CODES.get(venue_code, f"会場{venue_code}"),
                "date": date_formatted,
                "event_name": event_link.get_text(strip=True),
                "detail_url": f"{BASE_URL}{href}",
            })

        logger.info(f"Found {len(schedules)} events in {year}/{month}")
        return schedules

    def scrape_race_list(self, venue_code: str, date_str: str) -> list[dict[str, Any]]:
        """
        指定開催場・日付のレース一覧を取得
        date_str: "YYYY-MM-DD"形式
        Returns: [{race_key, race_no, race_class}, ...]
        """
        date_compact = date_str.replace("-", "")
        url = f"{BASE_URL}/keirindb/stadium/information/{venue_code}/{date_compact}/"

        soup = self._get(url)
        if soup is None:
            return []

        races = []
        # 出走表リンクから各レースのURLを取得
        member_links = soup.find_all("a", href=re.compile(r"/race/member/\d+/\d+/\d+/"))
        seen_races = set()

        for link in member_links:
            href = link.get("href", "")
            m = re.search(r"/race/member/(\d+)/(\d{8})/(\d+)/", href)
            if not m:
                continue

            vc = m.group(1)
            date_c = m.group(2)
            race_no = int(m.group(3))
            race_key = f"{date_c}_{vc}_{race_no:02d}"

            if race_key in seen_races:
                continue
            seen_races.add(race_key)

            # date_cはURL内の実際のレース日 (YYYYMMDD)。date_strはページを取得した日付で
            # 複数日イベントでは異なる場合がある。race_dateは常にURLから正確な日を取る。
            actual_date = f"{date_c[:4]}-{date_c[4:6]}-{date_c[6:]}"
            races.append({
                "race_key": race_key,
                "venue_code": vc,
                "date": actual_date,
                "race_no": race_no,
                "member_url": f"{BASE_URL}{href}",
                "result_url": f"{BASE_URL}/keirindb/race/result/{vc}/{date_c}/{race_no}/",
                "odds_url": f"{BASE_URL}/keirindb/race/odds/{vc}/{date_c}/{race_no}/3/",
            })

        races.sort(key=lambda x: x["race_no"])
        logger.info(f"Found {len(races)} races at venue {venue_code} on {date_str}")
        return races

    def scrape_race_detail(self, race_key: str = None, url: str = None) -> dict[str, Any] | None:
        """
        出走表を取得
        race_key: "YYYYMMDD_venue_raceNo"形式
        """
        if url is None:
            if race_key is None:
                return None
            parts = race_key.split("_")
            if len(parts) != 3:
                return None
            date_c, vc, rno = parts
            url = f"{BASE_URL}/keirindb/race/member/{vc}/{date_c}/{int(rno)}/"

        soup = self._get(url)
        if soup is None:
            return None

        result: dict[str, Any] = {
            "race_key": race_key,
            "race_info": {},
            "entries": [],
        }

        tables = soup.find_all("table")
        if not tables:
            return result

        # Table 0: レース基本情報（ヘッダー行）
        if tables:
            result["race_info"] = self._parse_race_header(tables[0])

        # Table 1以降: 各選手情報
        for table in tables[1:]:
            entry = self._parse_entry_table(table)
            if entry:
                result["entries"].append(entry)

        return result

    def _parse_race_header(self, table) -> dict:
        """レースヘッダーテーブルを解析"""
        text = table.get_text(strip=True)
        info = {}

        # 距離（例: "1625m4周"）
        m = re.search(r"(\d{3,4})m(\d+)周", text)
        if m:
            info["distance"] = int(m.group(1))
            info["laps"] = int(m.group(2))

        # グレード（例: "Ａ級予選"）
        grade_patterns = ["GP", "G1", "G2", "G3", "F1", "F2", "Ａ級", "Ｂ級"]
        for gp in grade_patterns:
            if gp in text:
                info["grade_text"] = gp
                break

        return info

    def _parse_entry_table(self, table) -> dict | None:
        """選手情報テーブルを1行解析"""
        rows = table.find_all("tr")
        all_text = " ".join([r.get_text(strip=True) for r in rows])

        if len(all_text) < 10:
            return None

        entry = {}

        # 選手名（カタカナ+漢字パターン）
        name_match = _RE_NAME.search(all_text)
        if name_match:
            entry["name_kana"] = name_match.group(1).strip()
            entry["name"] = name_match.group(2).strip()

        # ギア比（例: "3.93"）
        gear_match = _RE_GEAR.search(all_text)
        if gear_match:
            entry["gear_ratio"] = float(gear_match.group(1))

        # 勝率・2連対率・3着内率（例: "7.0% 45.0% 78.0%"）
        win_rate_match = _RE_RATES.search(all_text)
        if win_rate_match:
            entry["win_rate"] = float(win_rate_match.group(1)) / 100
            entry["quinella_rate"] = float(win_rate_match.group(2)) / 100
            entry["top3_rate"] = float(win_rate_match.group(3)) / 100

        # 府県（全47都道府県を明示列挙）
        pref_match = _RE_PREF.search(all_text)
        if pref_match:
            entry["prefecture"] = pref_match.group(1).replace(" ", "")

        # 期別（例: "91期"）
        period_match = _RE_PERIOD.search(all_text)
        if period_match:
            entry["period"] = int(period_match.group(1))

        # 登録クラス（SS/S1/S2/A1/A2/A3/B）
        class_match = _RE_CLASS.search(all_text)
        if class_match:
            entry["player_class"] = class_match.group(1)

        # 競走得点（50〜120の範囲の最大値）
        score_matches = _RE_SCORE.findall(all_text)
        if score_matches:
            scores = [float(s) for s in score_matches if 50 <= float(s) <= 120]
            if scores:
                entry["racing_score"] = max(scores)

        # 脚質（逃/捲/差/追）
        if "逃" in all_text:
            entry["riding_style"] = "先行"
        elif "捲" in all_text:
            entry["riding_style"] = "捲り"
        elif "差" in all_text:
            entry["riding_style"] = "差し"
        elif "追" in all_text:
            entry["riding_style"] = "追い込み"

        if not entry.get("name") and not entry.get("name_kana"):
            return None

        return entry

    def scrape_race_result(self, race_key: str = None, url: str = None) -> dict[str, Any] | None:
        """
        レース結果を取得
        """
        if url is None:
            if race_key is None:
                return None
            parts = race_key.split("_")
            if len(parts) != 3:
                return None
            date_c, vc, rno = parts
            url = f"{BASE_URL}/keirindb/race/result/{vc}/{date_c}/{int(rno)}/"

        soup = self._get(url)
        if soup is None:
            return None

        result: dict[str, Any] = {
            "race_key": race_key,
            "finish_order": [],
            "payouts": {},
        }

        tables = soup.find_all("table")

        # 着順テーブル（最後のテーブルが着順）
        if tables:
            finish_table = tables[-1]
            result["finish_order"] = self._parse_finish_order(finish_table)

        # 払戻金テーブル（最初の複数テーブル）
        for table in tables[:-1]:
            self._parse_payout_table(table, result["payouts"])

        return result

    def _parse_finish_order(self, table) -> list[dict]:
        """着順テーブルを解析"""
        finish_order = []
        rows = table.find_all("tr")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            pos_text = cols[0].get_text(strip=True)
            pos_match = re.match(r"^(\d+)着$", pos_text)
            if not pos_match:
                continue

            position = int(pos_match.group(1))

            # 車番: <img alt="1"> 形式
            img = cols[1].find("img")
            if img and img.get("alt", "").isdigit():
                frame_no = int(img["alt"])
            else:
                frame_no_text = cols[1].get_text(strip=True)
                if not frame_no_text.isdigit():
                    continue
                frame_no = int(frame_no_text)

            player_name_text = cols[2].get_text(strip=True)

            # 選手ページリンクからIDを取得
            player_link = cols[2].find("a", href=re.compile(r"/player/detail/(\d+)/"))
            player_id = None
            if player_link:
                m = re.search(r"/player/detail/(\d+)/", player_link["href"])
                if m:
                    player_id = m.group(1)

            finish_order.append({
                "position": position,
                "frame_no": frame_no,
                "player_name": player_name_text,
                "player_id": player_id,
            })

        return finish_order

    def _parse_payout_table(self, table, payouts: dict):
        """払戻金テーブルを解析"""
        bet_type_map = {
            "車番連複": "quinella",
            "車番単": "exacta",
            "3連勝複": "trifecta_box",
            "3連勝単": "trifecta",
            "単勝": "win",
            "複勝": "place",
            "ワイド": "wide",
        }

        table_text = table.get_text(strip=True)

        # 「未発売」テーブルはスキップ
        if "未発売" in table_text and "円" not in table_text:
            return

        # テーブル全体から賭式を特定
        current_bet_type = None
        for jp_name, en_name in bet_type_map.items():
            if jp_name in table_text:
                current_bet_type = en_name
                break

        if current_bet_type is None:
            return

        # 組み合わせ+払戻金パターンを全て抽出
        # =区切り → 複式（2連複/3連複）、-区切り → 単式（2連単/3連単）
        pattern = r"([0-9](?:[=\-][0-9])+)\s*([\d,]+)円"
        for m in re.finditer(pattern, table_text):
            combo = m.group(1)
            payout_str = m.group(2).replace(",", "")
            if not payout_str.isdigit():
                continue
            payout = int(payout_str)

            # セパレーターで賭式を判定（ワイドはテーブルヘッダー優先）
            parts_count = len(re.findall(r"[0-9]", combo))
            is_ordered = "-" in combo
            if current_bet_type == "wide":
                bet = "wide"
            elif parts_count == 2:
                bet = "exacta" if is_ordered else "quinella"
            elif parts_count == 3:
                bet = "trifecta" if is_ordered else "trifecta_box"
            else:
                bet = current_bet_type

            key = f"{bet}_{combo}"
            payouts[key] = payout

        # パターン2: 単勝・複勝（1桁の組み合わせ）
        if current_bet_type in ("win", "place"):
            pattern2 = r"\b([1-9])\s*([\d,]+)円"
            for m in re.finditer(pattern2, table_text):
                combo = m.group(1)
                payout_str = m.group(2).replace(",", "")
                if payout_str.isdigit():
                    key = f"{current_bet_type}_{combo}"
                    payouts[key] = int(payout_str)

        # ワイドは複数行ある
        if current_bet_type == "wide":
            for row in table.find_all("tr"):
                row_text = row.get_text(strip=True)
                m = re.search(r"([0-9]=[0-9])\s*([\d,]+)円", row_text)
                if m:
                    combo = m.group(1)
                    payout = int(m.group(2).replace(",", ""))
                    payouts[f"wide_{combo}"] = payout

    def scrape_odds(self, race_key: str, bet_type: str = "trifecta_box") -> dict[str, Any] | None:
        """オッズを取得"""
        parts = race_key.split("_")
        if len(parts) != 3:
            return None

        date_c, vc, rno = parts
        bet_no = BET_TYPE_NO.get(bet_type, "3")
        url = f"{BASE_URL}/keirindb/race/odds/{vc}/{date_c}/{int(rno)}/{bet_no}/"

        soup = self._get(url)
        if soup is None:
            return None

        odds_data: dict[str, Any] = {
            "race_key": race_key,
            "bet_type": bet_type,
            "odds": {},
        }

        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 2:
                    continue

                combo_text = cols[0].get_text(strip=True)
                odds_text = cols[-1].get_text(strip=True)

                if re.match(r"^[\d\-=]+$", combo_text) and re.match(r"^\d+\.\d+$", odds_text):
                    odds_data["odds"][combo_text] = float(odds_text)

        return odds_data
