import time
import random
import logging
from abc import ABC, abstractmethod
from typing import Any

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """スクレイパー基底クラス"""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, delay_min: float = 1.5, delay_max: float = 3.5):
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def _wait(self):
        duration = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Waiting {duration:.1f}s")
        time.sleep(duration)

    def _get(self, url: str, params: dict = None, max_retries: int = 3) -> BeautifulSoup | None:
        for attempt in range(max_retries):
            try:
                self._wait()
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, "lxml")
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 3.0
                    logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {url} - {e}. Retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Request failed: {url} - {e}")
                    return None
        return None

    @abstractmethod
    def scrape_race_list(self, venue_code: str, date_str: str) -> list[dict[str, Any]]:
        """指定会場・日のレース一覧を取得"""
        pass

    @abstractmethod
    def scrape_race_detail(self, race_key: str) -> dict[str, Any] | None:
        """レース詳細（出走表・選手情報）を取得"""
        pass

    @abstractmethod
    def scrape_race_result(self, race_key: str) -> dict[str, Any] | None:
        """レース結果を取得"""
        pass
