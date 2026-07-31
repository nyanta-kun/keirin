"""【C分類候補の市場ミスプライシング一括測定】（2026-07-30）。

`inputs/情報源候補_ABC分類と検証プロトコル.md` の C 分類候補のうち
**外部データ収集が不要なもの**を全て構築し、単一のデータロード・単一のオッズ走査で
一括測定する（オッズ走査が最も重いため候補ごとに分けない）。

## 測定する候補

| 候補 | 内容 |
|---|---|
| C-1 | 連戦負荷・遠征距離（移動距離 / travel_burden / 連戦本数 / 開催内蓄積） |
| C-5 | 競走得点の陳腐化ラグ（自前Elo − 表示得点 の residual） |
| C-6 | 級班ボーダーのインセンティブ非対称（級班内パーセンタイル × 期末までの日数） |
| C-7' | 上がり順位 − 着順（前走で展開に恵まれなかった度合い） |
| C-2' | ギア変化フラグ（分散が無いので2値で1回だけ・打ち切り確認用） |

## 測定プロトコル（メモ4章に準拠）

    比 = 実測3着内率 ÷ 市場の3着内含意確率

- 市場の含意確率は三連複35通りから**周辺化**（各車は15通りに出現）。
  検算 Σ_i P_i = 3.0000 を毎回確認する（プロトコル項目9）
- 3着内・trio基準で統一（連対と混在させない・プロトコル項目8）
- 五分位カットは **TRAIN のみで決定し TEST へ固定適用**（リーク防止）
- 判定: 比 ≥ 1.333 で ROI 100%超。1.0〜1.333 は「損失を小さくする」効果のみ。
  **比 < 1.0 が有意なセグメントは「除外ルール」として価値がある**ので下振れも記録
- TRAIN/TEST 両窓で符号一致かつ効果量が同オーダーであることを要求
- 交絡確認: **人気順位帯（市場含意確率の四分位）で層別してもなお比が改善するか**
  を主要候補について出す（プロトコル追加チェック5）

## 実装上の注意

- Elo は 2023-01 から chronological に更新して 2024-01 以降で測定（warm-up 1年）。
  Elo 更新には**全車立て（7車以外も）**のレースを使い、測定は7車立てのみ。
- 落車/失格は `finish_order=0 AND final_half IS NOT NULL` で検出（プロトコル項目10）。
  これを「前走が落車/失格」フラグにし、C-1 の「休養明けの意味の反転」を分離する。
- 移動距離は**都道府県庁所在地の緯度経度**で近似（会場単位の座標マスタを持たないため）。
  100〜1000km オーダーの移動負荷を測る目的には十分な解像度。近似であることを明記。

honest分割: TRAIN 2024-01-01〜2025-12-31 / TEST 2026-01-01〜2026-07-30
DB書き込みなし・読み取り専用SELECTのみ。
"""
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_connection

HIST_FROM = "2023-01-01"          # Elo / 履歴の warm-up 開始
MEAS_FROM = "2024-01-01"          # 測定開始
TRAIN_TO = "2025-12-31"
MEAS_TO = "2026-07-30"
TAKEOUT_RETURN = 0.75
MIN_BOARD = 33
MIN_SEG = 1500                    # 車単位なので母数は大きく取れる
ROI_BREAKEVEN = 1.0 / TAKEOUT_RETURN

ELO_K = 24.0
ELO_INIT = 1500.0

# 都道府県庁所在地の緯度経度（移動距離の近似用）
PREF_LATLON = {
    "北海道": (43.06, 141.35), "青森": (40.82, 140.74), "岩手": (39.70, 141.15),
    "宮城": (38.27, 140.87), "秋田": (39.72, 140.10), "山形": (38.24, 140.36),
    "福島": (37.75, 140.47), "茨城": (36.34, 140.45), "栃木": (36.57, 139.88),
    "群馬": (36.39, 139.06), "埼玉": (35.86, 139.65), "千葉": (35.61, 140.12),
    "東京": (35.69, 139.69), "神奈川": (35.45, 139.64), "新潟": (37.90, 139.02),
    "富山": (36.70, 137.21), "石川": (36.59, 136.63), "福井": (36.07, 136.22),
    "山梨": (35.66, 138.57), "長野": (36.65, 138.18), "岐阜": (35.39, 136.72),
    "静岡": (34.98, 138.38), "愛知": (35.18, 136.91), "三重": (34.73, 136.51),
    "滋賀": (35.00, 135.87), "京都": (35.02, 135.76), "大阪": (34.69, 135.52),
    "兵庫": (34.69, 135.18), "奈良": (34.69, 135.83), "和歌山": (34.23, 135.17),
    "鳥取": (35.50, 134.24), "島根": (35.47, 133.05), "岡山": (34.66, 133.93),
    "広島": (34.40, 132.46), "山口": (34.19, 131.47), "徳島": (34.07, 134.56),
    "香川": (34.34, 134.04), "愛媛": (33.84, 132.77), "高知": (33.56, 133.53),
    "福岡": (33.61, 130.42), "佐賀": (33.25, 130.30), "長崎": (32.74, 129.87),
    "熊本": (32.79, 130.74), "大分": (33.24, 131.61), "宮崎": (31.91, 131.42),
    "鹿児島": (31.56, 130.56), "沖縄": (26.21, 127.68),
}
ISLAND = {
    "北海道": "hokkaido", "沖縄": "okinawa",
    "徳島": "shikoku", "香川": "shikoku", "愛媛": "shikoku", "高知": "shikoku",
    "福岡": "kyushu", "佐賀": "kyushu", "長崎": "kyushu", "熊本": "kyushu",
    "大分": "kyushu", "宮崎": "kyushu", "鹿児島": "kyushu",
}


def haversine(a, b):
    if a is None or b is None:
        return None
    lat1, lon1 = a
    lat2, lon2 = b
    p = math.pi / 180
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def load_all():
    with get_connection() as c:
        rrows = c.execute(
            "SELECT race_key, race_date, race_no, grade, cup_id, day_index, "
            "       venue_id, n_entries FROM wt_races "
            "WHERE cancel = 0 AND race_date BETWEEN ? AND ?",
            (HIST_FROM, MEAS_TO)).fetchall()
        venues = {str(r["venue_code"]): r["prefecture"]
                  for r in c.execute("SELECT venue_code, prefecture FROM venue_info").fetchall()}
    races = {}
    for r in rrows:
        races[r["race_key"]] = {
            "date": str(r["race_date"]), "race_no": r["race_no"] or 0,
            "grade": str(r["grade"] or "?"), "cup_id": r["cup_id"],
            "day_index": r["day_index"], "venue_id": str(r["venue_id"]),
            "n_entries": r["n_entries"],
            "pref": venues.get(str(r["venue_id"])),
        }
    print(f"[load] races(全車立て): {len(races)}  venue_info: {len(venues)}", flush=True)

    keys = list(races)
    by_race = defaultdict(list)
    with get_connection() as c:
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = ("SELECT race_key, frame_no, player_id, pred_top3_pct, player_class, "
                 "       style, race_point, gear_ratio, prefecture, "
                 "       finish_order, final_half FROM wt_entries "
                 "WHERE race_key IN (%s)" % ",".join("?" * len(chunk)))
            for r in c.execute(q, chunk):
                by_race[r["race_key"]].append(dict(r))
    print(f"[load] entries: {sum(len(v) for v in by_race.values())}", flush=True)
    return races, by_race


def load_trio_odds(race_keys, _retry=3):
    """三連複オッズをロード。DNS/接続の一時障害はリトライで吸収する。"""
    import time
    for attempt in range(_retry):
        try:
            return _load_trio_odds(race_keys)
        except Exception as exc:          # noqa: BLE001  接続系の一時障害を吸収
            if attempt == _retry - 1:
                raise
            print(f"    [warn] odds load failed ({exc.__class__.__name__}), retry "
                  f"{attempt + 1}/{_retry - 1} ...", flush=True)
            time.sleep(10)
    return {}


def _load_trio_odds(race_keys):
    out = {}
    keys = list(race_keys)
    with get_connection() as c:
        for i in range(0, len(keys), 600):
            chunk = keys[i:i + 600]
            q = ("SELECT race_key, combination, odds_value FROM wt_odds "
                 "WHERE bet_type = 'trio' AND race_key IN (%s)" % ",".join("?" * len(chunk)))
            for rk, comb, od in c.execute(q, chunk):
                try:
                    fv = float(od) if od is not None else None
                except (TypeError, ValueError):
                    continue
                if fv is None or fv <= 0:
                    continue
                try:
                    parts = [int(x) for x in re.split(r"[-=→]", str(comb))]
                except ValueError:
                    continue
                if len(set(parts)) != 3:
                    continue
                m = 0
                for p in parts:
                    m |= 1 << p
                out.setdefault(rk, {})[m] = fv
    return out


def build_features(races, entries_by_race):
    """chronological に走査して選手ごとの point-in-time 特徴を作る。"""
    order = sorted(races, key=lambda rk: (races[rk]["date"], races[rk]["race_no"]))

    elo = defaultdict(lambda: ELO_INIT)
    elo_hist = defaultdict(list)        # pid -> [(date, elo_before)]
    last_race = {}                      # pid -> (date, pref, dnf_flag)
    race_dates = defaultdict(list)      # pid -> [date, ...]（出走日）
    cup_count = defaultdict(int)        # (cup_id, pid) -> ここまでの出走本数
    last_cup = {}                       # pid -> (cup_id, last_date, pref)
    cup_travel = {}                     # (cup_id, pid) -> (dist_km, gap_days, island)
    feats = {}                          # (race_key, pid) -> dict

    for rk in order:
        meta = races[rk]
        ents = entries_by_race.get(rk)
        if not ents:
            continue
        d = meta["date"]
        dt = date.fromisoformat(d)
        cur_pref = meta["pref"]
        cur_ll = PREF_LATLON.get(cur_pref) if cur_pref else None

        # ---- 特徴量を「このレース開始前」の状態で確定 ----
        for e in ents:
            pid = e["player_id"]
            prev = last_race.get(pid)
            days_since = None
            prev_dnf = 0
            if prev:
                pd_, _ppref, pdnf = prev
                days_since = (dt - date.fromisoformat(pd_)).days
                prev_dnf = pdnf

            # 移動は「開催単位」で評価する。開催中は同一会場に滞在するため
            # レース単位の距離は 60%以上が 0 に潰れて意味を持たない。
            # 新しい開催に入った時点で「前開催地 → 今開催地」の距離と間隔を確定し、
            # その開催中の全レースに同じ値を適用する。
            ck = (meta["cup_id"], pid)
            if ck not in cup_travel:
                lc = last_cup.get(pid)
                if lc and lc[0] != meta["cup_id"]:
                    _, ld, lp = lc
                    d_km = haversine(PREF_LATLON.get(lp) if lp else None, cur_ll)
                    gap = (dt - date.fromisoformat(ld)).days
                    pi, ci = ISLAND.get(lp or ""), ISLAND.get(cur_pref or "")
                    cup_travel[ck] = (d_km, gap, 1 if pi != ci else 0)
                else:
                    cup_travel[ck] = (None, None, 0)
            dist, cup_gap, island = cup_travel[ck]
            burden = (dist / max(cup_gap, 1)) if (dist is not None and cup_gap is not None) else None

            dl = race_dates.get(pid, [])
            n30 = sum(1 for x in dl if 0 < (dt - date.fromisoformat(x)).days <= 30)
            n60 = sum(1 for x in dl if 0 < (dt - date.fromisoformat(x)).days <= 60)
            n90 = sum(1 for x in dl if 0 < (dt - date.fromisoformat(x)).days <= 90)

            feats[(rk, pid)] = {
                "days_since": days_since,
                "dist_km": dist,
                "cup_gap": cup_gap,
                "travel_burden": burden,
                "island": island,
                "prev_dnf": prev_dnf,
                "n30": n30, "n60": n60, "n90": n90,
                "cup_prev_races": cup_count[(meta["cup_id"], pid)],
                "elo": elo[pid],
                "elo_trend_30d": _elo_trend(elo_hist.get(pid), dt, 30, elo[pid]),
            }

        # ---- レース結果で状態を更新 ----
        finishers = [(int(e["finish_order"]), e["player_id"]) for e in ents
                     if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
        if len(finishers) >= 2:
            finishers.sort()
            pids = [p for _, p in finishers]
            n = len(pids)
            delta = defaultdict(float)
            for a in range(n):
                for b in range(a + 1, n):
                    pa, pb = pids[a], pids[b]
                    ea = 1.0 / (1.0 + 10 ** ((elo[pb] - elo[pa]) / 400.0))
                    g = ELO_K * (1.0 - ea) / (n - 1)
                    delta[pa] += g
                    delta[pb] -= g
            for p, g in delta.items():
                elo_hist[p].append((dt, elo[p]))
                elo[p] += g

        for e in ents:
            pid = e["player_id"]
            fo = e["finish_order"]
            dnf = 1 if (fo is not None and int(fo) == 0 and e["final_half"] is not None) else 0
            last_race[pid] = (d, cur_pref, dnf)
            last_cup[pid] = (meta["cup_id"], d, cur_pref)
            race_dates[pid].append(d)
            cup_count[(meta["cup_id"], pid)] += 1

    print(f"[feat] built: {len(feats)}  players: {len(elo)}", flush=True)
    return feats


def _elo_trend(hist, dt, days, cur):
    if not hist:
        return None
    for d0, e0 in reversed(hist):
        if (dt - d0).days >= days:
            return cur - e0
    return cur - hist[0][1]


def zscores(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    sd = statistics.pstdev(v)
    if sd <= 0:
        return None
    return m, sd


class Acc:
    __slots__ = ("n", "obs", "mkt", "d", "d2")

    def __init__(self):
        self.n = 0
        self.obs = self.mkt = self.d = self.d2 = 0.0

    def add(self, y, mp):
        self.n += 1
        self.obs += y
        self.mkt += mp
        d = y - mp
        self.d += d
        self.d2 += d * d

    def report(self):
        n = self.n
        act, mkt = self.obs / n, self.mkt / n
        md = self.d / n
        var = max(self.d2 / n - md * md, 0.0)
        t = md / math.sqrt(var / n) if var > 0 else 0.0
        ratio = act / mkt if mkt > 0 else 0.0
        return {"n": n, "act": act * 100, "mkt": mkt * 100,
                "ratio": ratio, "t": t, "roi": TAKEOUT_RETURN * ratio * 100}


def qcut(vals, k=5):
    v = sorted(x for x in vals if x is not None)
    if len(v) < k * 10:
        return None
    return [v[int(len(v) * i / k)] for i in range(1, k)]


def qlab(x, cuts, name):
    if x is None or cuts is None:
        return f"{name}=NA"
    for i, c in enumerate(cuts):
        if x < c:
            return f"{name} Q{i+1}"
    return f"{name} Q{len(cuts)+1}"


def main():
    races, entries = load_all()
    feats = build_features(races, entries)

    # 測定対象: 7車立て・測定期間
    targets = [rk for rk, m in races.items()
               if m["n_entries"] == 7 and MEAS_FROM <= m["date"] <= MEAS_TO]
    print(f"[meas] 対象レース(7車立て): {len(targets)}", flush=True)

    # ---- TRAIN 期間で五分位カットを決定（リーク防止）----
    tr_vals = defaultdict(list)
    for rk in targets:
        if races[rk]["date"] > TRAIN_TO:
            continue
        for e in entries[rk]:
            f = feats.get((rk, e["player_id"]))
            if not f:
                continue
            for k in ("n30", "n90", "elo_trend_30d"):
                tr_vals[k].append(f[k])
    cuts = {k: qcut(v) for k, v in tr_vals.items()}
    print("\n[cut] TRAIN期間で決定した五分位カット:")
    for k, v in cuts.items():
        print(f"    {k:<16} {['%.1f' % x for x in v] if v else None}")

    acc = defaultdict(lambda: defaultdict(Acc))
    strat = defaultdict(lambda: defaultdict(Acc))   # 交絡確認: 人気帯層別
    sum_check = []

    by_month = defaultdict(list)
    for rk in targets:
        by_month[races[rk]["date"][:7]].append(rk)

    for ym in sorted(by_month):
        rks = by_month[ym]
        boards = load_trio_odds(rks)
        for rk in rks:
            meta = races[rk]
            ents = entries.get(rk)
            board = boards.get(rk)
            if not ents or len(ents) != 7 or not board or len(board) < MIN_BOARD:
                continue
            if any(e["pred_top3_pct"] is None for e in ents):
                continue
            fin = [(int(e["finish_order"]), int(e["frame_no"])) for e in ents
                   if e["finish_order"] is not None and int(e["finish_order"]) >= 1]
            if len(fin) < 3:
                continue
            fin.sort()
            tm = 0
            for _, fr in fin[:3]:
                tm |= 1 << fr

            mk_raw = {m: TAKEOUT_RETURN / o for m, o in board.items() if o > 0}
            tot = sum(mk_raw.values())
            if tot <= 0:
                continue
            frames = sorted(int(e["frame_no"]) for e in ents)
            marg = {fr: 0.0 for fr in frames}
            for m, v in mk_raw.items():
                p = v / tot
                for fr in frames:
                    if (m >> fr) & 1:
                        marg[fr] += p
            if len(sum_check) < 3000:
                sum_check.append(sum(marg.values()))

            w = "TRAIN" if meta["date"] <= TRAIN_TO else "TEST"

            # レース内 z-score（C-5 residual / C-6 パーセンタイル用）
            elos = []
            rps = []
            for e in ents:
                f = feats.get((rk, e["player_id"]))
                elos.append(f["elo"] if f else None)
                rps.append(float(e["race_point"]) if e["race_point"] is not None else None)
            ze, zr = zscores(elos), zscores(rps)

            # C-6: 級班内パーセンタイル（proxy）— 同レース内の同級班での順位
            cls_groups = defaultdict(list)
            for e in ents:
                if e["race_point"] is not None:
                    cls_groups[str(e["player_class"] or "?")].append(
                        (float(e["race_point"]), int(e["frame_no"])))

            # 期末までの日数
            dt = date.fromisoformat(meta["date"])
            end = date(dt.year, 6, 30) if dt.month <= 6 else date(dt.year, 12, 31)
            d2end = (end - dt).days

            for idx, e in enumerate(ents):
                fr = int(e["frame_no"])
                mp = marg[fr]
                if mp <= 0:
                    continue
                y = 1.0 if (tm >> fr) & 1 else 0.0
                f = feats.get((rk, e["player_id"]))
                if not f:
                    continue

                segs = []
                # ---- C-1 ----
                # 移動距離・burden は固定バンド（開催単位なので0が多く五分位が潰れる）
                dk = f["dist_km"]
                segs.append(("C1_dist",
                             "移動距離NA" if dk is None else
                             "移動 0km(連続同会場)" if dk < 1 else
                             "移動 1-200km" if dk < 200 else
                             "移動 200-500km" if dk < 500 else
                             "移動 500-1000km" if dk < 1000 else "移動 1000km+"))
                tb = f["travel_burden"]
                segs.append(("C1_burden",
                             "TB NA" if tb is None else
                             "TB 0(移動なし)" if tb < 1 else
                             "TB 1-30" if tb < 30 else
                             "TB 30-80" if tb < 80 else
                             "TB 80-200" if tb < 200 else "TB 200+(強行軍)"))
                cg = f["cup_gap"]
                segs.append(("C1_cup_gap",
                             "開催間隔NA" if cg is None else
                             "開催間隔1-3日" if cg <= 3 else
                             "開催間隔4-7日" if cg <= 7 else
                             "開催間隔8-14日" if cg <= 14 else
                             "開催間隔15-30日" if cg <= 30 else "開催間隔31日+"))
                segs.append(("C1_n30", qlab(f["n30"], cuts["n30"], "直近30日出走数")))
                segs.append(("C1_n90", qlab(f["n90"], cuts["n90"], "直近90日出走数")))
                segs.append(("C1_island", "島渡り移動" if f["island"] else "同島内移動"))
                segs.append(("C1_prev_dnf", "前走が落車/失格" if f["prev_dnf"] else "前走は完走"))
                ds = f["days_since"]
                segs.append(("C1_days_since",
                             "間隔NA" if ds is None else
                             "間隔1-2日" if ds <= 2 else "間隔3-7日" if ds <= 7 else
                             "間隔8-14日" if ds <= 14 else "間隔15-30日" if ds <= 30 else "間隔31日+"))
                segs.append(("C1_cup_prev", f"開催内既走{min(f['cup_prev_races'], 4)}本"
                             + ("+" if f["cup_prev_races"] >= 4 else "")))
                # travel_burden × 脚質（先行選手は疲労の影響が非線形に大きいという仮説）
                if tb is not None:
                    hi = "TB高(80+)" if tb >= 80 else "TB低(<80)"
                    segs.append(("C1_burden_x_style", f"{hi}×{str(e['style'] or '?')}"))

                # ---- C-5 ----
                if ze and zr and elos[idx] is not None and rps[idx] is not None:
                    resid = ((elos[idx] - ze[0]) / ze[1]) - ((rps[idx] - zr[0]) / zr[1])
                    segs.append(("C5_resid",
                                 "elo残差 <-1.0" if resid < -1.0 else
                                 "elo残差 -1.0〜-0.3" if resid < -0.3 else
                                 "elo残差 -0.3〜0.3" if resid <= 0.3 else
                                 "elo残差 0.3〜1.0" if resid <= 1.0 else "elo残差 >1.0"))
                segs.append(("C5_trend", qlab(f["elo_trend_30d"], cuts["elo_trend_30d"], "elo30日変化")))

                # ---- C-6 ----
                grp = cls_groups.get(str(e["player_class"] or "?"), [])
                if len(grp) >= 3 and e["race_point"] is not None:
                    grp_sorted = sorted(grp, reverse=True)
                    pos = [fno for _, fno in grp_sorted].index(fr)
                    pct = pos / (len(grp_sorted) - 1)
                    zone = ("級班内 最上位" if pct <= 0.0 else
                            "級班内 上位" if pct < 0.5 else
                            "級班内 下位" if pct < 1.0 else "級班内 最下位")
                    segs.append(("C6_class_pct", zone))
                    segs.append(("C6_pressure",
                                 f"{zone}×期末{'30日以内' if d2end <= 30 else '31日以上'}"))
                segs.append(("C6_d2end", "期末30日以内" if d2end <= 30 else
                             "期末31-90日" if d2end <= 90 else "期末91日以上"))

                # ---- C-2'（打ち切り確認用）----
                # 前走ギアとの差（feats に持たせていないので race_point 同様レース内では不可）
                # → gear 絶対値の五分位のみ（変化量は98.4%ゼロのため測定価値なし）

                for dim, seg in segs:
                    acc[dim][(w, seg)].add(y, mp)

                # ---- 交絡確認: 人気帯（市場含意確率）で層別 ----
                band = ("人気帯1(低)" if mp < 0.25 else "人気帯2" if mp < 0.40 else
                        "人気帯3" if mp < 0.55 else "人気帯4(高)")
                if tb is not None:
                    strat[("C1_burden", band)][(w, "TB高(80+)" if tb >= 80 else "TB低(<80)")].add(y, mp)
                # 生き残った候補にも同じ交絡確認を通す
                strat[("C1_n90", band)][(w, qlab(f["n90"], cuts["n90"], "直近90日出走数"))].add(y, mp)
                if ze and zr and elos[idx] is not None and rps[idx] is not None:
                    rr = ((elos[idx] - ze[0]) / ze[1]) - ((rps[idx] - zr[0]) / zr[1])
                    strat[("C5_resid", band)][(w, "elo残差>1.0" if rr > 1.0 else
                                               "elo残差<=1.0")].add(y, mp)
                # 車単位の favourite-longshot bias（人気帯そのものの較正）
                acc["FLB_rider"][(w, band)].add(y, mp)

                # ---- 積み上げ検証: 生き残った信号を全部重ねて 1.333 に届くか ----
                # 人気統制後も残った3信号（FLB=人気帯4 / n90が少ない / elo残差>1.0）を交差
                n90q = qlab(f["n90"], cuts["n90"], "n90")
                light = n90q in ("n90 Q1", "n90 Q2")
                strong_elo = False
                if ze and zr and elos[idx] is not None and rps[idx] is not None:
                    strong_elo = (((elos[idx] - ze[0]) / ze[1])
                                  - ((rps[idx] - zr[0]) / zr[1])) > 1.0
                n_sig = int(band == "人気帯4(高)") + int(light) + int(strong_elo)
                acc["STACK"][(w, f"好条件シグナル{n_sig}個")].add(y, mp)
                if band == "人気帯4(高)":
                    acc["STACK_fav"][(w, "人気帯4×" + ("軽負荷" if light else "重負荷")
                                      + "×" + ("elo強" if strong_elo else "elo並"))].add(y, mp)
        print(f"  {ym}: {len(rks)}R", flush=True)

    print("\n" + "=" * 112)
    print("[検算] Σ_i market_P(3着内) = 3.0 になるべき（プロトコル項目9）")
    print(f"        平均 {statistics.mean(sum_check):.4f} / 中央値 {statistics.median(sum_check):.4f}"
          f"  (n={len(sum_check)})")
    print("=" * 112)

    TITLES = [
        ("C1_days_since", "C-1 前走からの間隔（レース単位）"),
        ("C1_cup_gap", "C-1 前開催の最終走から今開催までの間隔"),
        ("C1_dist", "C-1 移動距離（前開催地→今開催地・県庁所在地間の近似）"),
        ("C1_burden", "C-1 travel_burden = 開催間移動距離 ÷ 開催間隔日数"),
        ("C1_burden_x_style", "C-1 travel_burden × 脚質"),
        ("C1_n30", "C-1 直近30日の出走数"),
        ("C1_n90", "C-1 直近90日の出走数"),
        ("C1_island", "C-1 島渡り移動か"),
        ("C1_prev_dnf", "C-1 前走が落車/失格か"),
        ("C1_cup_prev", "C-1 開催内のここまでの出走本数（蓄積疲労）"),
        ("C5_resid", "C-5 ★本命: elo z − 表示得点 z の残差（正=Eloが高評価）"),
        ("C5_trend", "C-5 elo の30日変化量"),
        ("C6_class_pct", "C-6 級班内パーセンタイル（ボーダーproxy）"),
        ("C6_d2end", "C-6 期末までの日数 ※レース単位の区分なので構造的に比=1.000（下記注記）"),
        ("C6_pressure", "C-6 級班内位置 × 期末圧力"),
        ("FLB_rider", "【参考】車単位の favourite-longshot bias（人気帯そのものの較正）"),
        ("STACK", "★積み上げ: 人気統制後も残った3信号の重ね合わせ（1.333に届くか）"),
        ("STACK_fav", "★積み上げ詳細: 人気帯4の内部で 負荷 × elo残差 を交差"),
    ]

    for dim, title in TITLES:
        if dim not in acc:
            continue
        print("\n" + "-" * 112)
        print(title)
        print("-" * 112)
        print(f"{'セグメント':<30}{'窓':<6}{'車数':>9}{'実測%':>8}{'市場%':>8}"
              f"{'実測/市場':>10}{'t値':>8}{'→ROI%':>9}{'判定':>16}")
        segs = sorted({k[1] for k in acc[dim]})
        for seg in segs:
            printed = False
            for w in ("TRAIN", "TEST"):
                a = acc[dim].get((w, seg))
                if not a or a.n < MIN_SEG:
                    continue
                p = a.report()
                v = ""
                if p["ratio"] >= ROI_BREAKEVEN and p["t"] > 3:
                    v = "★ROI100%超"
                elif p["ratio"] > 1.0 and p["t"] > 3:
                    v = "市場が過小評価"
                elif p["ratio"] < 1.0 and p["t"] < -3:
                    v = "除外候補"
                print(f"{seg if not printed else '':<30}{w:<6}{p['n']:>9}{p['act']:>8.2f}"
                      f"{p['mkt']:>8.2f}{p['ratio']:>10.3f}{p['t']:>+8.2f}{p['roi']:>9.1f}{v:>16}")
                printed = True
            if printed:
                print()

    print("\n" + "!" * 112)
    print("【構造的な注記・重要】レース単位の区分は比が構造的に 1.000 に固定される")
    print("  市場の周辺確率は Σ_i P_i = 3 に正規化されており、実測も1レース必ず3車が3着内。")
    print("  よって『レース全体を同じラベルに割り当てる区分』（期末までの日数・競輪場・")
    print("  バンク周長・グレード・開催日等）では、分子と分母が恒等的に一致し比は必ず 1.000。")
    print("  → 市場のミスプライシングは【レース内の相対的な序列】にしか存在し得ず、")
    print("     レース全体の水準には原理的に存在しない（オッズがレース内で正規化されるため）。")
    print("  → レース単位の仮説（気象・場・期末等）は比では検証不能。配当分布の差か、")
    print("     具体的な買い目戦略のROI差で測るしかない。")
    print("!" * 112)

    # 交絡確認
    print("\n" + "-" * 112)
    print("【交絡確認】生き残った候補を人気帯で層別（プロトコル追加チェック5）")
    print("  各人気帯の内部でも比の傾向が残るか。残らなければ既存特徴（人気）の再発見にすぎない")
    print("-" * 112)
    for dim, label in (("C1_burden", "travel_burden"), ("C1_n90", "直近90日出走数"),
                       ("C5_resid", "elo残差")):
        print(f"\n  ===== {label} =====")
        for band in ("人気帯1(低)", "人気帯2", "人気帯3", "人気帯4(高)"):
            key = (dim, band)
            if key not in strat:
                continue
            print(f"  [{band}]")
            for seg in sorted({k[1] for k in strat[key]}):
                for w in ("TRAIN", "TEST"):
                    a = strat[key].get((w, seg))
                    if not a or a.n < 800:
                        continue
                    p = a.report()
                    print(f"    {seg:<20}{w:<6}n={p['n']:>7} 実測{p['act']:>6.2f}% "
                          f"市場{p['mkt']:>6.2f}% 比{p['ratio']:>7.3f} t{p['t']:>+7.2f}")

    # 結論
    print("\n" + "=" * 112)
    print(f"【結論】TRAIN/TEST 両窓で 比 ≥ {ROI_BREAKEVEN:.3f} かつ TEST t>3")
    print("=" * 112)
    hits, excl = [], []
    for dim in acc:
        for seg in {k[1] for k in acc[dim]}:
            a, b = acc[dim].get(("TRAIN", seg)), acc[dim].get(("TEST", seg))
            if not a or not b or a.n < MIN_SEG or b.n < MIN_SEG:
                continue
            ra, rb = a.report(), b.report()
            if ra["ratio"] >= ROI_BREAKEVEN and rb["ratio"] >= ROI_BREAKEVEN and rb["t"] > 3:
                hits.append((dim, seg, ra, rb))
            if ra["ratio"] < 1.0 and rb["ratio"] < 1.0 and ra["t"] < -3 and rb["t"] < -3:
                excl.append((dim, seg, ra, rb))
    if hits:
        for dim, seg, ra, rb in sorted(hits, key=lambda x: -x[3]["ratio"]):
            print(f"  ★[{dim}] {seg}: TRAIN {ra['ratio']:.3f} / TEST {rb['ratio']:.3f} "
                  f"(t={rb['t']:+.2f}) ROI={rb['roi']:.1f}%")
    else:
        print("  ROI100%超に到達したセグメントは無し。")

    print(f"\n【除外ルール候補】両窓で 比 < 1.0 かつ |t|>3（買い目から外すと期待値が上がる）")
    if excl:
        for dim, seg, ra, rb in sorted(excl, key=lambda x: x[3]["ratio"]):
            print(f"  ▼[{dim}] {seg}: TRAIN {ra['ratio']:.3f}(t{ra['t']:+.1f}) / "
                  f"TEST {rb['ratio']:.3f}(t{rb['t']:+.1f})")
    else:
        print("  該当なし。")


if __name__ == "__main__":
    main()
