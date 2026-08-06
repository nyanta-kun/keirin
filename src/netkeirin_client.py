"""netkeirin「ウマい車券」入稿ツール（tool.syakenv2.netkeiba.com/bettool/）への
下書き自動入稿クライアント。

仕様の根拠は docs/netkeirin-input-api-spec.md（2026-07-23実機検証で確定・
2026-07-28に3連単1着ながし(S1)・9車waku_checkを追加実測・2026-08-06に
3連単フォーメーション/3連複ボックス(7H1)を追加実測 — 詳細はdocs参照）。
対応する買い目構造は次の4つで、汎用の全券種対応は意図していない:
  - 三連複・軸2頭ながし  … 「二軸探偵」方式（7SS/7S/7A/7B/9S/9A）
  - 三連単・1着ながし    … S1方式（全廃済みだが形式は保持）
  - 三連単・フォーメーション / 三連複・ボックス … 7H1（2券種を1商品で入稿）
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from itertools import combinations as _combinations
from pathlib import Path
from typing import NamedTuple

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
BET_KIND_TRIFECTA_FORMATION = "trifecta_formation"  # 3連単・フォーメーション（7H1）
BET_KIND_TRIO_BOX = "trio_box"              # 3連複・ボックス（7H1）

_SHIKIBETU = {
    BET_KIND_TRIO_AXIS2: "8",
    BET_KIND_TRIFECTA_AXIS1: "9",
    BET_KIND_TRIFECTA_FORMATION: "9",
    BET_KIND_TRIO_BOX: "8",
}
_HOUSHIKI = {
    BET_KIND_TRIO_AXIS2: "6",
    BET_KIND_TRIFECTA_AXIS1: "3",
    BET_KIND_TRIFECTA_FORMATION: "1",
    BET_KIND_TRIO_BOX: "2",
}

# bet_kind ごとの車番グループ数（bet_id の `_` 区切りスロット数）。
_N_GROUPS = {
    BET_KIND_TRIO_AXIS2: 3,          # 軸1 / 軸2 / 相手
    BET_KIND_TRIFECTA_AXIS1: 2,      # 1着軸 / 相手
    BET_KIND_TRIFECTA_FORMATION: 3,  # 1着列 / 2着列 / 3着列
    BET_KIND_TRIO_BOX: 1,            # BOXの車群
}

# 印（表示記号）→ mark_code。docs/netkeirin-input-api-spec.md 2.2節。
# 印が付かない車は "0"（--）。表示用の印マップをそのまま入稿へ渡せるようにする
# （**表示と入稿で印を二重管理しないため**。7B の「買っていない車まで △」不具合と
# 同型の食い違いを構造的に防ぐ）。
MARK_CODE = {"◎": "1", "○": "2", "▲": "3", "△": "4", "☆": "5"}
MARK_CODE_NONE = "0"

# 車数ごとの枠割当（keirin固有の固定ルール・車数のみに依存しレース非依存。
# 7車=[6]は2026-07-23佐世保1R、9車=[4,5,6]は2026-07-28豊橋4R/5Rで実測確定）。
_WAKU_CHECK = {7: [6], 9: [4, 5, 6]}

# race.html の実ソース確認済み（2026-07-23）: param.type = $('#act-type').val()
# （勝負アイコン: 0=指定しない/1=自信あり/2=穴狙い）、param.point = $('#act-point').val()
# （販売価格）。旧ドキュメントの「type=式別・point=ポイント数」という推測は誤りだった
# ため訂正済み。式別/方式は kaime[].bet_id 文字列にのみ含まれる。
ACT_TYPE_CONFIDENT = "1"
ACT_TYPE_LONGSHOT = "2"   # 勝負アイコン「穴狙い」（7H1）
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


class BetLeg(NamedTuple):
    """1商品に含める買い目行（`kaime` の1要素に対応）。

    groups は bet_kind ごとの車番グループ（`build_bet_id_groups` 参照）。
    """
    bet_kind: str
    groups: list[list[int]]
    stake_per_line: int


def waku_check_for(n_cars: int) -> list[int]:
    """車数から waku_check（同一枠に2車以上入る枠のリスト）を返す。"""
    if n_cars not in _WAKU_CHECK:
        raise ValueError(f"未対応の車数: {n_cars}（7/9のみ対応）")
    return _WAKU_CHECK[n_cars]


def build_bet_id_groups(
    race_date: date, venue_code: str, race_no: int, bet_kind: str,
    groups: list[list[int]],
) -> str:
    """車番グループ列から bet_id を組み立てる（全 bet_kind 共通の低レベルAPI）。

    bet_id は例外なく次の形をとる（4形式すべてを実データで確認済み・docs 2.3節）:

        a{曜日}-{場}-{R}_b{式別}_c{方式}_{グループ1}_{グループ2}_…
        グループ内の車番はハイフン区切り・昇順

    | bet_kind            | b  | c  | グループ                          |
    |---------------------|----|----|-----------------------------------|
    | trio_axis2          | 8  | 6  | 軸1 / 軸2 / 相手                  |
    | trifecta_axis1      | 9  | 3  | 1着軸 / 相手（マルチOFF）         |
    | trifecta_formation  | 9  | 1  | 1着列 / 2着列 / 3着列             |
    | trio_box            | 8  | 2  | BOXの車群                         |

    曜日コードは isoweekday()%7（月=1…土=6・日=0）。日曜のみ要目視確認（未検証・docs 3節）。
    レース番号はrace_id内ではゼロ埋めだが、bet_id内はゼロ埋めなし。
    """
    if bet_kind not in _N_GROUPS:
        raise ValueError(f"未対応のbet_kind: {bet_kind}")
    if len(groups) != _N_GROUPS[bet_kind]:
        raise ValueError(
            f"{bet_kind} のグループ数は {_N_GROUPS[bet_kind]} 必須（実際={len(groups)}）")
    if any(not g for g in groups):
        raise ValueError(f"{bet_kind} に空のグループが含まれています: {groups}")
    weekday = race_date.isoweekday() % 7
    prefix = (f"a{weekday}-{venue_code}-{race_no}"
              f"_b{_SHIKIBETU[bet_kind]}_c{_HOUSHIKI[bet_kind]}")
    body = "_".join("-".join(str(c) for c in sorted(g)) for g in groups)
    return f"{prefix}_{body}"


def build_bet_id(
    race_date: date, venue_code: str, race_no: int, bet_kind: str,
    axis1: int, axis2: int | None, partners: list[int],
) -> str:
    """軸ながし系（trio_axis2 / trifecta_axis1）の bet_id を組み立てる。

    trio_axis2（3連複・軸2頭ながし）実データ確認済み（2026-07-23・佐世保1R）:
        "a5-85-1_b8_c6_1_2_3-4-5-6-7"  （軸1=1・軸2=2・相手=3,4,5,6,7）

    trifecta_axis1（3連単・1着ながし）実データ確認済み（2026-07-28・取手1R）:
        "a2-23-1_b9_c3_1_2-3"  （1着軸=1・相手=2,3・マルチOFF）
        （軸2頭ながしと異なり軸スロットは1つのみ。マルチはOFFにすること
        ＝ONだと1着固定を無視した全順序展開＝ボックス相当になる）

    フォーメーション/ボックスは軸という概念を持たないため
    `build_bet_id_groups()` を直接使う。
    """
    if bet_kind == BET_KIND_TRIO_AXIS2:
        if axis2 is None:
            raise ValueError("trio_axis2にはaxis2が必須です")
        groups = [[axis1], [axis2], list(partners)]
    elif bet_kind == BET_KIND_TRIFECTA_AXIS1:
        groups = [[axis1], list(partners)]
    else:
        raise ValueError(f"未対応のbet_kind: {bet_kind}")
    return build_bet_id_groups(race_date, venue_code, race_no, bet_kind, groups)


def expand_bet(bet_kind: str, groups: list[list[int]]) -> set:
    """bet_id が表す買い目を展開する（何点・どの目になるかの単一正本）。

    **推測で買い目を組み立てないための検算に使う。** 呼び出し側が持っている
    「買うつもりの目」と本関数の戻り値が一致しなければ、その bet_id は
    意図と違うものを外部へ入稿する。

    戻り値の要素は着順を持つ券種（3連単系）が `tuple`、持たない券種
    （3連複系）が `frozenset`。
    """
    if bet_kind == BET_KIND_TRIFECTA_FORMATION:
        c1, c2, c3 = groups
        return {(a, b, c) for a in c1 for b in c2 for c in c3
                if len({a, b, c}) == 3}
    if bet_kind == BET_KIND_TRIO_BOX:
        return {frozenset(t) for t in _combinations(sorted(groups[0]), 3)}
    if bet_kind == BET_KIND_TRIFECTA_AXIS1:
        axis, partners = groups[0][0], groups[1]
        return {(axis, a, b) for a in partners for b in partners if a != b}
    if bet_kind == BET_KIND_TRIO_AXIS2:
        a1, a2, partners = groups[0][0], groups[1][0], groups[2]
        return {frozenset((a1, a2, p)) for p in partners if p not in (a1, a2)}
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

        return self._post_goods(
            race_id=race_id, n_cars=n_cars, mark=mark, title=title, comment=comment,
            kaime=[{"bet_id": bet_id, "bet_money": stake_per_line}],
            act_type=(ACT_TYPE_CONFIDENT if confident else ACT_TYPE_DEFAULT),
        )

    def submit_pick_multi(
        self, *, race_date: date, venue_name: str, race_no: int, n_cars: int,
        legs: list[BetLeg], marks: dict[int, str],
        title: str, comment: str = DEFAULT_COMMENT,
        act_type: str = ACT_TYPE_DEFAULT,
    ) -> tuple[bool, str]:
        """複数券種を1件の商品として入稿する（7H1＝三連単F + 三連複BOX）。

        netkeirin の `kaime` は配列で、1レース1商品に**複数の買い目行**を持てる
        （入稿ツールUIの「投票内容」が複数行になるのと同じ。2026-08-06に
        3行＝三連単F+三連複BOX×2 を実機で確認済み）。したがって2券種でも
        submit は1回でよく、2回送ると同一 race_id の商品を上書きしてしまう。

        Args:
            legs:  買い目行。bet_kind ごとの車番グループと1点あたり金額。
            marks: 車番 → 表示印（"◎"/"○"/"▲"/"△"）。ここに無い車は印なし(--)。
                   **表示用の印マップをそのまま渡す**（表示と入稿の二重管理を避ける）。

        戻り値: (成功したか, メッセージ)
        """
        if n_cars not in _WAKU_CHECK:
            return False, f"対象外(n_cars={n_cars}、7/9車のみ対応)"
        if not legs:
            return False, "買い目が空です"
        if not comment:
            comment = DEFAULT_COMMENT

        if not self.login():
            return False, "ログイン失敗"

        venue_code = self.resolve_venue_code(race_date, venue_name)
        if venue_code is None:
            return False, f"場コード解決失敗: {venue_name}"

        race_id = f"{race_date.strftime('%Y%m%d')}{venue_code}{race_no:02d}"
        kaime = []
        for leg in legs:
            bet_id = build_bet_id_groups(
                race_date, venue_code, race_no, leg.bet_kind, leg.groups)
            kaime.append({"bet_id": bet_id, "bet_money": leg.stake_per_line})

        mark = {str(c): MARK_CODE.get(marks.get(c, ""), MARK_CODE_NONE)
                for c in range(1, n_cars + 1)}

        return self._post_goods(
            race_id=race_id, n_cars=n_cars, mark=mark, title=title, comment=comment,
            kaime=kaime, act_type=act_type,
        )

    def _post_goods(
        self, *, race_id: str, n_cars: int, mark: dict[str, str],
        title: str, comment: str, kaime: list[dict], act_type: str,
    ) -> tuple[bool, str]:
        """api_post_goods.html への POST（action=add）本体。"""
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
            # 2026-08-06〜: 7H1 に「穴狙い」(type=2)。上限の有無は未確認だが、
            # 下の yoso_tag_over フォールバックが type を問わず効くので入稿は落ちない。
            "type": act_type,
            "point": SALE_PRICE_DEFAULT,
            "waku_check": json.dumps(waku_check_for(n_cars)),
            "kaime": json.dumps(kaime, ensure_ascii=False),
        }

        try:
            r = self.session.post(POST_GOODS_URL, data=payload, timeout=15)
            r.raise_for_status()
            resp = r.json()
            # 勝負アイコンの1日上限（yoso_tag_over）に当たったら、タグ無しで
            # もう一度だけ送る。**入稿そのものを落とさないため**の措置。
            # 「自信あり」は上限が1件/日と推定され、同日2件以上ある日は必ず起きる。
            if act_type != ACT_TYPE_DEFAULT and not resp.get("result") \
                    and "yoso_tag_over" in str(resp):
                print(f"[netkeirin] 勝負アイコン(type={act_type})が上限のため"
                      f"タグ無しで再送します: {race_id}")
                payload["type"] = ACT_TYPE_DEFAULT
                r = self.session.post(POST_GOODS_URL, data=payload, timeout=15)
                r.raise_for_status()
                resp = r.json()
        except (requests.RequestException, ValueError) as e:
            return False, f"入稿リクエスト失敗: {e}"

        if resp.get("status") != "OK":
            return False, f"入稿失敗: {resp}"
        return True, race_id
