"""netkeirin「ウマい車券」入稿ツール（tool.syakenv2.netkeiba.com/bettool/）への
下書き自動入稿クライアント。

仕様の根拠は docs/netkeirin-input-api-spec.md（2026-07-23実機検証で確定。
ログインURL/フィールド名・type/pointパラメータの正体は同日の追加検証
（認証済みセッションで race.html / auth/login.html の実ソースを直接取得）で
確定済み — 詳細はdocs参照）。
「二軸探偵」方式（軸1=◎・軸2=○・残り全馬=△、三連複2軸ながし・各2,000円・5点）
専用のシンプルなクライアントであり、汎用の全券種対応は意図していない。
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://tool.syakenv2.netkeiba.com/bettool"
TOP_URL = f"{BASE_URL}/top/index.html"
LOGIN_URL = f"{BASE_URL}/auth/api_post_login.html"
LOGIN_ID_FIELD = "user_id"
PASSWORD_FIELD = "password"

RACE_LIST_URL = f"{BASE_URL}/bet/race_list.html"
POST_GOODS_URL = f"{BASE_URL}/bet/api_post_goods.html"
RACE_AUTH_URL = f"{BASE_URL}/bet/race_auth.html"

DATA_DIR = Path(__file__).parent.parent / "data"
SESSION_FILE = DATA_DIR / "netkeirin_session.json"
VENUE_CACHE_FILE = DATA_DIR / "netkeirin_venue_codes.json"

STAKE_PER_LINE = 2000  # 円/点（三連複2軸ながし・各2,000円・5点=10,000円=上限ぴったり）
SHIKIBETU_TRIO = "8"   # 3連複（bet_id内の b トークン）
HOUSHIKI_AXIS2_NAGASHI = "6"  # 軸2頭ながし（bet_id内の c トークン）

# race.html の実ソース確認済み（2026-07-23）: param.type = $('#act-type').val()
# （勝負アイコン: 0=指定しない/1=自信あり/2=穴狙い）、param.point = $('#act-point').val()
# （販売価格）。旧ドキュメントの「type=式別・point=ポイント数」という推測は誤りだった
# ため訂正済み。式別/方式は kaime[].bet_id 文字列にのみ含まれる。
ACT_TYPE_CONFIDENT = "1"
ACT_TYPE_DEFAULT = "0"
SALE_PRICE_DEFAULT = "300"
CONFIDENT_GATE_LABELS = {"SS+", "SS"}  # 勝負アイコン「自信あり」対象

# race.html の check_goods_data() 実装確認済み: comment/titleは必須（空文字だと
# クライアント側バリデーションで弾かれる）。
DEFAULT_COMMENT = "本日の二軸をお届けします。"


def _env(key: str) -> str:
    """.env（リポジトリルート）または環境変数から値を読む（src/notify/discord.py と同じ方式）。"""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return os.environ.get(key, "")


def build_bet_id(race_date: date, venue_code: str, race_no: int,
                  axis1: int, axis2: int, partners: list[int]) -> str:
    """3連複・軸2頭ながしのbet_idを組み立てる。

    実データ確認済み（2026-07-23・佐世保1R・2026-07-24=金曜）:
        "a5-85-1_b8_c6_1_2_3-4-5-6-7"
    曜日コードは isoweekday()%7（月=1…土=6・日=0）。月〜土はJSのgetDay()と
    一致するため、この1点の実測だけで月〜土は確定している。日曜のみ
    2つの規約(ISO=7 / JS=0)が分岐しうるため要目視確認（未検証）。
    レース番号はrace_id内ではゼロ埋めだが、bet_id内はゼロ埋めなし。
    """
    weekday = race_date.isoweekday() % 7
    partners_str = "-".join(str(p) for p in sorted(partners))
    return (
        f"a{weekday}-{venue_code}-{race_no}"
        f"_b{SHIKIBETU_TRIO}_c{HOUSHIKI_AXIS2_NAGASHI}_{axis1}_{axis2}_{partners_str}"
    )


class NetkeirinClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self._load_cookies()

    # ── セッション管理 ──────────────────────────────────────────────────

    def _load_cookies(self) -> None:
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    self.session.cookies.set(k, v, domain="tool.syakenv2.netkeiba.com")
            except Exception as e:
                print(f"[netkeirin] セッションCookie読み込み失敗: {e}")

    def _save_cookies(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(
            json.dumps(self.session.cookies.get_dict(), ensure_ascii=False),
            encoding="utf-8",
        )

    def _is_logged_in(self) -> bool:
        """認証状態を判定する。

        未ログイン時は top/index.html への GET が auth/login.html へリダイレクト
        される（2026-07-23確認）。単純に本文へ"ログアウト"の文字列有無で判定すると
        ログイン画面自体にも同文字列が含まれておりfalse positiveになるため、
        最終URLがログイン画面でないことも合わせて確認する。
        """
        try:
            r = self.session.get(TOP_URL, timeout=10)
            if r.status_code != 200:
                return False
            if "auth/login.html" in r.url:
                return False
            return "ログアウト" in r.text
        except requests.RequestException as e:
            print(f"[netkeirin] ログイン状態確認失敗: {e}")
            return False

    def login(self) -> bool:
        """既存セッションが有効ならそれを使う。無効ならログインを試みる。

        2026-07-23、認証済みセッションで auth/login.html の実ソースを取得し
        api_auth() の実装からログインPOSTの仕様を確定済み:
            POST https://tool.syakenv2.netkeiba.com/bettool/auth/api_post_login.html
            data: {output: 'json', action: 'login', user_id: <ID>, password: <PW>}
            成功時レスポンス: {"status":"OK","user_id":"<内部ID>"}
        """
        if self._is_logged_in():
            return True
        login_id = _env("NETKEIRIN_LOGIN_ID")
        password = _env("NETKEIRIN_PASSWORD")
        if not login_id or not password:
            print("[netkeirin] NETKEIRIN_LOGIN_ID / NETKEIRIN_PASSWORD が未設定です")
            return False
        try:
            r = self.session.post(
                LOGIN_URL,
                data={
                    "output": "json",
                    "action": "login",
                    LOGIN_ID_FIELD: login_id,
                    PASSWORD_FIELD: password,
                },
                timeout=10,
            )
            ok = r.status_code == 200 and r.json().get("status") == "OK"
        except (requests.RequestException, ValueError) as e:
            print(f"[netkeirin] ログインリクエスト失敗: {e}")
            return False
        if ok:
            self._save_cookies()
            return True
        print(f"[netkeirin] ログイン失敗: status={r.status_code} body={r.text[:200]}")
        return False

    # ── 場コード解決 ────────────────────────────────────────────────────

    def _load_venue_cache(self) -> dict[str, str]:
        if VENUE_CACHE_FILE.exists():
            try:
                return json.loads(VENUE_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_venue_cache(self, cache: dict[str, str]) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        VENUE_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def resolve_venue_code(self, race_date: date, venue_name: str) -> str | None:
        """netkeirin独自の場コード（2桁）を場名から解決する。

        race_list.html?kaisai_date=YYYYMMDD の会場ボタン href="#jyo_{date}_{code}"
        から場名→コードを都度取得しキャッシュする（場名は不変なので蓄積される）。
        """
        cache = self._load_venue_cache()
        if venue_name in cache:
            return cache[venue_name]

        date_str = race_date.strftime("%Y%m%d")
        try:
            r = self.session.get(RACE_LIST_URL, params={"kaisai_date": date_str}, timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[netkeirin] race_list取得失敗({date_str}): {e}")
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        found: dict[str, str] = {}
        pattern = re.compile(r"^#jyo_(\d+)_(\d+)$")
        for a in soup.find_all("a", href=pattern):
            m = pattern.match(a["href"])
            if not m:
                continue
            code = m.group(2)
            name = a.get_text(strip=True)
            if name:
                found[name] = code

        if found:
            cache.update(found)
            self._save_venue_cache(cache)
        return cache.get(venue_name)

    # ── 入稿本体 ────────────────────────────────────────────────────────

    def submit_pick(
        self, *, race_date: date, venue_name: str, race_no: int,
        axis1: int, axis2: int, n_entries: int, gate_label: str,
        title: str, comment: str = DEFAULT_COMMENT,
    ) -> tuple[bool, str]:
        """1レース分の下書き（action=add）を入稿する。

        戻り値: (成功したか, メッセージ)
        「二軸探偵」方式専用（7車ちょうど・3連複軸2頭ながし）のため n_entries!=7 は対象外。
        """
        if n_entries != 7:
            return False, f"対象外(n_entries={n_entries}、7車のみ対応)"
        if not comment:
            comment = DEFAULT_COMMENT

        if not self.login():
            return False, "ログイン失敗"

        venue_code = self.resolve_venue_code(race_date, venue_name)
        if venue_code is None:
            return False, f"場コード解決失敗: {venue_name}"

        partners = [c for c in range(1, 8) if c not in (axis1, axis2)]
        race_id = f"{race_date.strftime('%Y%m%d')}{venue_code}{race_no:02d}"
        bet_id = build_bet_id(race_date, venue_code, race_no, axis1, axis2, partners)

        # mark の値は race.html 実装上 DOM id (id="act-mark_{車番}_{code}") を
        # split した文字列がそのままセットされる（数値ではなく文字列）。
        mark = {str(axis1): "1", str(axis2): "2"}
        for p in partners:
            mark[str(p)] = "4"

        # 7車ちょうどの場合、車6・7は常に同一枠(枠番6)を共有する（keirin固有の固定ルール。
        # 2026-07-23、佐世保1R(7車)の実ページソースで waku_check = [6] を直接確認済み）。
        waku_check = [6]

        payload = {
            "output": "json",
            "action": "add",
            "race_id": race_id,
            "mark": json.dumps(mark, ensure_ascii=False),
            "title": title,
            "comment": comment,
            # race.html実ソース確認済み: type=勝負アイコン値・point=販売価格
            # （式別/方式はkaime[].bet_idにのみ含まれる。旧仮実装の誤りを訂正済み）。
            # 2026-07-24〜: 「自信あり」(type=1)の1日あたり投稿上限が不明なため
            # ACT_TYPE_CONFIDENT自動付与を一時停止し、常にACT_TYPE_DEFAULT（指定なし）
            # で送信する。SS/SS+の「自信あり」指定は上限が判明するまでユーザーが
            # netkeirin.jp上で手動設定する運用とする。上限判明後にCONFIDENT_GATE_LABELS
            # 判定へ戻すこと。
            "type": ACT_TYPE_DEFAULT,
            "point": SALE_PRICE_DEFAULT,
            "waku_check": json.dumps(waku_check),
            "kaime": json.dumps(
                [{"bet_id": bet_id, "bet_money": STAKE_PER_LINE}], ensure_ascii=False,
            ),
        }

        try:
            r = self.session.post(POST_GOODS_URL, data=payload, timeout=15)
            r.raise_for_status()
            resp = r.json()
        except (requests.RequestException, ValueError) as e:
            return False, f"入稿リクエスト失敗: {e}"

        if resp.get("status") != "OK":
            return False, f"入稿失敗: {resp}"
        return True, race_id
