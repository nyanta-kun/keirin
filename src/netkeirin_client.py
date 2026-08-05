"""netkeirin「ウマい車券」入稿ツール（tool.syakenv2.netkeiba.com/bettool/）への
下書き自動入稿クライアント。

仕様の根拠は docs/netkeirin-input-api-spec.md（2026-07-23実機検証で確定・
2026-07-28に3連単1着ながし(S1)・9車waku_checkを追加実測 — 詳細はdocs参照）。
「二軸探偵」方式（軸1=◎・軸2=○、三連複2軸ながし）と、S1方式（軸=◎・1着固定、
三連単1着ながし）の2種類の買い目構造に対応する。汎用の全券種対応は意図していない。
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

# 買い目構造（bet_kind）。式別(shikibetu)・方式(houshiki)はdocs 2.3節で実機確定済み。
BET_KIND_TRIO_AXIS2 = "trio_axis2"          # 3連複・軸2頭ながし（7SS/7S/7A/9SS/9S/9A）
BET_KIND_TRIFECTA_AXIS1 = "trifecta_axis1"  # 3連単・1着ながし（S1）

_SHIKIBETU = {BET_KIND_TRIO_AXIS2: "8", BET_KIND_TRIFECTA_AXIS1: "9"}
_HOUSHIKI = {BET_KIND_TRIO_AXIS2: "6", BET_KIND_TRIFECTA_AXIS1: "3"}

# 車数ごとの枠割当（keirin固有の固定ルール・車数のみに依存しレース非依存。
# 7車=[6]は2026-07-23佐世保1R、9車=[4,5,6]は2026-07-28豊橋4R/5Rで実測確定）。
_WAKU_CHECK = {7: [6], 9: [4, 5, 6]}

# race.html の実ソース確認済み（2026-07-23）: param.type = $('#act-type').val()
# （勝負アイコン: 0=指定しない/1=自信あり/2=穴狙い）、param.point = $('#act-point').val()
# （販売価格）。旧ドキュメントの「type=式別・point=ポイント数」という推測は誤りだった
# ため訂正済み。式別/方式は kaime[].bet_id 文字列にのみ含まれる。
ACT_TYPE_CONFIDENT = "1"
ACT_TYPE_DEFAULT = "0"
SALE_PRICE_DEFAULT = "300"
CONFIDENT_GATE_LABELS = {"SS"}  # 勝負アイコン「自信あり」対象（SS+は2026-07-27にSSへ統合・廃止）

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


def waku_check_for(n_cars: int) -> list[int]:
    """車数から waku_check（同一枠に2車以上入る枠のリスト）を返す。"""
    if n_cars not in _WAKU_CHECK:
        raise ValueError(f"未対応の車数: {n_cars}（7/9のみ対応）")
    return _WAKU_CHECK[n_cars]


def build_bet_id(
    race_date: date, venue_code: str, race_no: int, bet_kind: str,
    axis1: int, axis2: int | None, partners: list[int],
) -> str:
    """bet_idを組み立てる。

    trio_axis2（3連複・軸2頭ながし）実データ確認済み（2026-07-23・佐世保1R）:
        "a5-85-1_b8_c6_1_2_3-4-5-6-7"  （軸1=1・軸2=2・相手=3,4,5,6,7）
        → a{曜日}-{場}-{R}_b{式別}_c{方式}_{軸1}_{軸2}_{相手(ハイフン区切り)}

    trifecta_axis1（3連単・1着ながし）実データ確認済み（2026-07-28・取手1R）:
        "a2-23-1_b9_c3_1_2-3"  （1着軸=1・相手=2,3・マルチOFF）
        → a{曜日}-{場}-{R}_b{式別}_c{方式}_{1着軸}_{相手(ハイフン区切り)}
        （軸2頭ながしと異なり軸スロットは1つのみ。マルチはOFFにすること
        ＝ONだと1着固定を無視した全順序展開＝ボックス相当になる）

    曜日コードは isoweekday()%7（月=1…土=6・日=0）。日曜のみ要目視確認（未検証・docs 3節）。
    レース番号はrace_id内ではゼロ埋めだが、bet_id内はゼロ埋めなし。
    """
    weekday = race_date.isoweekday() % 7
    shikibetu = _SHIKIBETU[bet_kind]
    houshiki = _HOUSHIKI[bet_kind]
    partners_str = "-".join(str(p) for p in sorted(partners))
    prefix = f"a{weekday}-{venue_code}-{race_no}_b{shikibetu}_c{houshiki}"
    if bet_kind == BET_KIND_TRIO_AXIS2:
        return f"{prefix}_{axis1}_{axis2}_{partners_str}"
    if bet_kind == BET_KIND_TRIFECTA_AXIS1:
        return f"{prefix}_{axis1}_{partners_str}"
    raise ValueError(f"未対応のbet_kind: {bet_kind}")


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
        n_cars: int, bet_kind: str,
        axis1: int, partners: list[int], axis2: int | None = None,
        stake_per_line: int,
        title: str, comment: str = DEFAULT_COMMENT,
        confident: bool = False,
    ) -> tuple[bool, str]:
        """1レース分の下書き（action=add）を入稿する。

        bet_kind=BET_KIND_TRIO_AXIS2（三連複・軸2頭ながし）: axis1・axis2 が軸2車、
          partners は残り流し対象車（n_cars-2台）。
        bet_kind=BET_KIND_TRIFECTA_AXIS1（三連単・1着ながし＝S1）: axis1 が1着軸、
          axis2 は使わない（None固定）、partners は相手2車ちょうど。

        戻り値: (成功したか, メッセージ)
        """
        if n_cars not in _WAKU_CHECK:
            return False, f"対象外(n_cars={n_cars}、7/9車のみ対応)"
        if bet_kind == BET_KIND_TRIO_AXIS2 and axis2 is None:
            return False, "trio_axis2にはaxis2が必須です"
        if bet_kind == BET_KIND_TRIFECTA_AXIS1 and len(partners) != 2:
            return False, f"trifecta_axis1のpartnersは2車必須(実際={len(partners)})"
        if not comment:
            comment = DEFAULT_COMMENT

        if not self.login():
            return False, "ログイン失敗"

        venue_code = self.resolve_venue_code(race_date, venue_name)
        if venue_code is None:
            return False, f"場コード解決失敗: {venue_name}"

        race_id = f"{race_date.strftime('%Y%m%d')}{venue_code}{race_no:02d}"
        bet_id = build_bet_id(race_date, venue_code, race_no, bet_kind, axis1, axis2, partners)

        # mark の値は race.html 実装上 DOM id (id="act-mark_{車番}_{code}") を
        # split した文字列がそのままセットされる（数値ではなく文字列）。
        mark: dict[str, str] = {}
        if bet_kind == BET_KIND_TRIO_AXIS2:
            assert axis2 is not None
            mark[str(axis1)] = "1"
            mark[str(axis2)] = "2"
            marked = {axis1, axis2}
        else:
            p1, p2 = partners[0], partners[1]
            mark[str(axis1)] = "1"
            mark[str(p1)] = "2"
            mark[str(p2)] = "3"
            marked = {axis1, p1, p2}
        # 【2026-08-03改定】軸以外は「買い目に入っている相手だけ」を △(mark_code=4) にし、
        # 買い目から外した車は --(mark_code=0・印なし) にする。
        #
        # 旧実装は `for c in range(1, n_cars+1): if c not in marked: mark[c]="4"` と
        # partners を無視して**軸以外の全車**に △ を付けていた。総流しのランク
        # （7S/7A/9S/9A は partners = 軸以外の全車）では結果が同じなので問題に
        # ならなかったが、相手を絞る 7B では買っていない2車まで △ 表示になり、
        # 入稿内容と買い目が食い違っていた（ユーザー指摘・2026-08-03）。
        #
        # partners を正とすることで総流しランクの挙動は完全に不変のまま、
        # 絞り込みランクだけが正しく --(印なし) になる。
        partner_set = set(partners)
        for c in range(1, n_cars + 1):
            if c in marked:
                continue
            mark[str(c)] = "4" if c in partner_set else "0"

        waku_check = waku_check_for(n_cars)

        payload = {
            "output": "json",
            "action": "add",
            "race_id": race_id,
            "mark": json.dumps(mark, ensure_ascii=False),
            "title": title,
            "comment": comment,
            # race.html実ソース確認済み: type=勝負アイコン値・point=販売価格
            # （式別/方式はkaime[].bet_idにのみ含まれる。旧仮実装の誤りを訂正済み）。
            # 2026-07-24〜2026-08-05: 「自信あり」(type=1)の1日あたり投稿上限が
            # 不明なため自動付与を停止していた（2件目以降が yoso_tag_over で拒否
            # された実測あり。上限は1件/日の可能性が高い）。
            # 2026-08-05〜: **7SS（最上位ランク・実測1.9件/日）にのみ**付与を再開。
            # 上限に当たった場合は呼び出し側(submit_pick)が type=0 で自動リトライ
            # するため、拒否されても入稿自体は失われない。
            "type": ACT_TYPE_CONFIDENT if confident else ACT_TYPE_DEFAULT,
            "point": SALE_PRICE_DEFAULT,
            "waku_check": json.dumps(waku_check),
            "kaime": json.dumps(
                [{"bet_id": bet_id, "bet_money": stake_per_line}], ensure_ascii=False,
            ),
        }

        try:
            r = self.session.post(POST_GOODS_URL, data=payload, timeout=15)
            r.raise_for_status()
            resp = r.json()
            # 「自信あり」の1日上限（yoso_tag_over）に当たったら、タグ無しで
            # もう一度だけ送る。**入稿そのものを落とさないため**の措置。
            # 上限が1件/日と推定されるため 7SS が同日2件以上ある日は必ず起きる。
            if confident and not resp.get("result") \
                    and "yoso_tag_over" in str(resp):
                payload["type"] = ACT_TYPE_DEFAULT
                r = self.session.post(POST_GOODS_URL, data=payload, timeout=15)
                r.raise_for_status()
                resp = r.json()
        except (requests.RequestException, ValueError) as e:
            return False, f"入稿リクエスト失敗: {e}"

        if resp.get("status") != "OK":
            return False, f"入稿失敗: {resp}"
        return True, race_id
