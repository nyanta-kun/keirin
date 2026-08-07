"""RANK_7H1（本命バスト型・穴推奨）の特徴量と役割判定の単一正本。

## この モジュールが存在する理由

7H1 の選別は「**モデル軸1 == WINTICKET公式印◎ の本命が4着以下に沈むか**」を予測する
**レース単位**の二値分類で、既存ランク（`lgbm_wt_eval` / `_win` / `_bad`）のような
**選手単位**モデルとは特徴量の粒度が違う。そのため専用の特徴量セットが要る。

**学習（`scripts/train_favbust_model.py`）・過去分構築（`scripts/backfill_7h1_rank_wt.py`）・
本番の候補生成（`src/cli/main.py`）がすべてこのモジュールを通る。**
検証で使ったものと本番で動くものが別実装になる事故（本リポジトリで繰り返し発生）を
構造的に防ぐのが目的なので、**ここ以外で特徴量を組み立ててはいけない**。

## 特徴量の構成（67列）

| 群 | 列数 | 内容 |
|---|---|---|
| レース単位 | 45 | WT公式予想の拡散度 / 競走得点の構成 / ライン構成 / 脚質構成 / 制度・会場 |
| 本命単位 | 22 | 本命自身の予測値・得点順位・ライン内の位置・脚質・対抗との関係 |

🔴 `win_*` / `top3_*` の出自（2026-08-08 是正・**以前ここに書いてあった説明は誤り**）

以前は「WINTICKET が表示する予想率で当方モデルの出力ではない」と書いてあったが
**逆**。`wt_entries.pred_win_pct` / `pred_top3_pct` は**当方モデルの出力**である。

- 書き込み元は `src/cli/main.py`（`wave-picks-wt`）の1箇所だけで、
  `lgbm_wt_win` / `lgbm_wt`（eval）の `predict_proba` をそのまま %化して
  `UPDATE wt_entries SET pred_win_pct = ?, pred_top3_pct = ?` している。
- スクレイパ（`src/scraper/pipeline_wt.py`）の `wt_entries` INSERT にこの2列は
  **含まれない**。WINTICKET から取っているのは `prediction_mark`（◎◯▲の記号）だけ。

なぜ重要か: 「外部サイトの値だから vintage 管理は要らない」と読めてしまい、
過去分を再構築するときの扱いを誤る。本モジュールの `fav_pp3`/`fav_ppw`/`fav_pbad`
は呼び出し側が月次凍結 vintage モデルを明示ロードして計算する一方、
`win_*`/`top3_*` は **DB に入っている値をそのまま読む**（`race_features()`）ので、
vintage の指定が効かない。

現状これは look-ahead ではない:
  - ライブ書き込みは `wave-picks-wt --date <当日>` が当日分だけを更新し、
    本番モデルは週次（日曜23:30）再学習なので「前日までで学習したモデルで
    当日レースを評価」＝ honest。過去日を遡って上書きすることはない。
  - 過去分は `scripts/backfill_index_pct_wt.py` が月次 vintage 体系で
    全期間 502,522件を再計算済み（2026-07-29・`docs/vintage_model_policy.md`）。

ただし **backfill 未実行の区間では「週次本番モデル由来の値」と
「月次凍結モデル」が混ざる**ため、再構築の再現性は保証されない。
特徴量の由来を変えたとき・新しい月を再構築するときは
`backfill_index_pct_wt.py` を流し直すこと。

当方モデルの選手単位出力は `fav_pp3` / `fav_ppw` / `fav_pbad`
（3着内率 / 1着率 / 大敗率）として本命の分だけ入る。こちらは呼び出し側が
vintage を明示するので指定が効く。両者を取り違えないこと。

## 役割判定（買い目構築で使う）

本命 fav から見た各車の構造的な役割を返す。実測（本命が飛んだ621R）:

| 役割 | 1着率 | 3着内率 |
|---|---|---|
| 別ライン先頭(最強) | 28.05% | 64.85% |
| 別ライン番手 | 17.52% | 58.72% |
| 単騎 | 15.70% | 43.93% |
| 本命ライン3番手以降 | 8.21% | 31.61% |
| **本命ライン番手** | **7.79%** | **33.27%** |

**本命が飛ぶときは番手も一緒に飛ぶ（ライン共倒れ）**ため、7H1 は本命ラインを
丸ごと買い目から落とす。詳細は memory `keirin_highpay_payout_ceiling_2026_08_06`。
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))
GRADE_MAP = {"L級": 0, "A級": 1, "S級": 2, "SA混合": 3}
STYLE_ENC = {"逃": 0, "両": 1, "追": 2}

# ── 役割ラベル（買い目構築が参照する）──────────────────────────────
ROLE_FAV_MATE = "本命ライン番手"
ROLE_FAV_THIRD = "本命ライン3番手以降"
ROLE_LEAD_TOP = "別ライン先頭(最強)"
ROLE_LEAD_OTHER = "別ライン先頭(その他)"
ROLE_OTHER_MATE = "別ライン番手"
ROLE_SOLO = "単騎"
#: 本命と同一ラインの役割。7H1 はここを買い目から丸ごと落とす。
FAV_LINE_ROLES = frozenset({ROLE_FAV_MATE, ROLE_FAV_THIRD})

RACE_FEATURE_COLS: list[str] = [
    "win_max", "win_gap12", "win_sum_top2", "win_sum_top3", "win_entropy",
    "top3_max", "top3_gap12", "top3_sum_top2", "top3_sum_top3", "top3_entropy",
    "mark_top3_sum", "mark_win_sum", "honmei_is_w1", "mark_same_line",
    "rp_max", "rp_min", "rp_mean", "rp_std", "rp_gap12", "rp_gap23",
    "rp_range", "rp_top2_edge",
    "n_lines", "max_line_size", "line_size_2nd", "n_solo",
    "line_top_share", "line_gap12", "n_samepref_pairs",
    "n_senko", "n_oikomi", "n_ryo", "n_senko_clash",
    "n_classes", "grade_enc", "day_index",
    "rt_final", "rt_semi", "rt_heat", "rt_senbatsu", "rt_tokusen",
    "bank_length", "is_indoor", "is_night", "distance",
]

FAV_FEATURE_COLS: list[str] = [
    "fav_pp3", "fav_ppw", "fav_pbad", "fav_pp3_gap12", "fav_ppw_gap12",
    "fav_rp_rank", "fav_rp", "fav_rp_gap_next", "fav_rp_gap_mean",
    "fav_frame", "fav_line_size", "fav_line_pos", "fav_is_leader", "fav_is_solo",
    "fav_style", "fav_class", "fav_line_rp_sum", "fav_line_rank",
    "taikou_same_line", "taikou_pp3", "taikou_rp_gap", "n_stronger_line",
]

#: 学習・推論で使う特徴量の順序（`load_model` の列名照合もこれを見る）
FAVBUST_FEATURE_COLS: list[str] = RACE_FEATURE_COLS + FAV_FEATURE_COLS


def _entropy(vals: list[float]) -> float:
    total = sum(vals)
    if total <= 0:
        return 0.0
    return -sum(max(v / total, 1e-9) * math.log(max(v / total, 1e-9)) for v in vals)


def _line_key(entry: dict) -> Any:
    """ライン識別子。単騎（line_group が None）は車番で一意にする。"""
    lg = entry.get("line_group")
    return lg if lg is not None else f"__solo_{int(entry['frame_no'])}"


def race_features(meta: dict, entries: list[dict]) -> dict | None:
    """レース単位の45特徴を返す。必須データが欠ける場合は None。

    Args:
        meta: race_date / grade / race_type / day_index / distance /
              bank_length / is_indoor / start_at を持つ dict
        entries: wt_entries 相当の dict のリスト
    """
    if len(entries) < 3:
        return None
    wv = sorted((float(e.get("pred_win_pct") or 0) for e in entries), reverse=True)
    tv = sorted((float(e.get("pred_top3_pct") or 0) for e in entries), reverse=True)
    by_frame = {int(e["frame_no"]): e for e in entries}
    w1 = max(by_frame, key=lambda f: float(by_frame[f].get("pred_win_pct") or 0))

    honmei = next((int(e["frame_no"]) for e in entries
                   if e.get("prediction_mark") == 1), None)
    taikou = next((int(e["frame_no"]) for e in entries
                   if e.get("prediction_mark") == 2), None)
    mark_t3 = mark_w = 0.0
    mark_same_line = -1
    if honmei is not None and taikou is not None:
        mark_t3 = (float(by_frame[honmei].get("pred_top3_pct") or 0)
                   + float(by_frame[taikou].get("pred_top3_pct") or 0))
        mark_w = (float(by_frame[honmei].get("pred_win_pct") or 0)
                  + float(by_frame[taikou].get("pred_win_pct") or 0))
        lh, lt = by_frame[honmei].get("line_group"), by_frame[taikou].get("line_group")
        mark_same_line = 1 if (lh is not None and lh == lt) else 0

    rps = sorted((float(e["race_point"]) for e in entries
                  if e.get("race_point") is not None), reverse=True)
    if len(rps) < 3:
        return None
    mean = sum(rps) / len(rps)
    var = sum((x - mean) ** 2 for x in rps) / len(rps)

    lines: dict = defaultdict(list)
    for e in entries:
        lines[_line_key(e)].append(e)
    sizes = sorted((len(v) for v in lines.values()), reverse=True)
    line_rp = sorted((sum(float(x.get("race_point") or 0) for x in v)
                      for v in lines.values()), reverse=True)
    rp_total = sum(rps) or 1.0

    pref: dict = defaultdict(int)
    for e in entries:
        if e.get("prefecture"):
            pref[e["prefecture"]] += 1

    styles: dict = defaultdict(int)
    for e in entries:
        styles[e.get("style") or "?"] += 1
    senko_lines: dict = defaultdict(int)
    for e in entries:
        if e.get("style") == "逃" and e.get("line_group") is not None:
            senko_lines[e["line_group"]] += 1

    is_night = 0
    try:
        dt = datetime.fromtimestamp(int(meta["start_at"]),
                                    tz=timezone.utc).astimezone(JST)
        is_night = 1 if dt.hour >= 17 else 0
    except (TypeError, ValueError, KeyError):
        pass

    rt = str(meta.get("race_type") or "")
    return {
        "win_max": wv[0], "win_gap12": wv[0] - wv[1],
        "win_sum_top2": wv[0] + wv[1], "win_sum_top3": sum(wv[:3]),
        "win_entropy": _entropy(wv),
        "top3_max": tv[0], "top3_gap12": tv[0] - tv[1],
        "top3_sum_top2": tv[0] + tv[1], "top3_sum_top3": sum(tv[:3]),
        "top3_entropy": _entropy(tv),
        "mark_top3_sum": mark_t3, "mark_win_sum": mark_w,
        "honmei_is_w1": 1 if honmei == w1 else 0,
        "mark_same_line": mark_same_line,
        "rp_max": rps[0], "rp_min": rps[-1], "rp_mean": mean,
        "rp_std": math.sqrt(var),
        "rp_gap12": rps[0] - rps[1], "rp_gap23": rps[1] - rps[2],
        "rp_range": rps[0] - rps[-1],
        "rp_top2_edge": (rps[0] + rps[1]) / 2 - mean,
        "n_lines": float(entries[0].get("n_lines") or len(lines)),
        "max_line_size": float(sizes[0]),
        "line_size_2nd": float(sizes[1]) if len(sizes) > 1 else 0.0,
        "n_solo": float(sum(1 for s in sizes if s == 1)),
        "line_top_share": line_rp[0] / rp_total,
        "line_gap12": (line_rp[0] - line_rp[1]) if len(line_rp) >= 2 else line_rp[0],
        "n_samepref_pairs": float(sum(v * (v - 1) // 2 for v in pref.values())),
        "n_senko": float(styles.get("逃", 0)),
        "n_oikomi": float(styles.get("追", 0)),
        "n_ryo": float(styles.get("両", 0)),
        "n_senko_clash": float(sum(1 for v in senko_lines.values() if v >= 2)),
        "n_classes": float(len({e.get("player_class") for e in entries
                                if e.get("player_class")})),
        "grade_enc": float(GRADE_MAP.get(meta.get("grade"), -1)),
        "day_index": float(meta.get("day_index") or 0),
        "rt_final": 1.0 if "決勝" in rt else 0.0,
        "rt_semi": 1.0 if "準決" in rt else 0.0,
        "rt_heat": 1.0 if "予選" in rt else 0.0,
        "rt_senbatsu": 1.0 if "選抜" in rt else 0.0,
        "rt_tokusen": 1.0 if "特選" in rt else 0.0,
        "bank_length": float(meta.get("bank_length") or 0),
        "is_indoor": float(meta.get("is_indoor") or 0),
        "is_night": float(is_night),
        "distance": float(meta.get("distance") or 0),
    }


def fav_features(entries: list[dict], preds: dict[int, tuple]) -> dict | None:
    """本命単位の22特徴を返す。**軸1（1着率最上位）が WT◎ と一致しない場合は None**。

    Args:
        preds: {frame_no: (pp3, ppw, pbad)} 当方モデルの出力
    Returns:
        22特徴 + `_fav`（本命の車番）。母集団外なら None。
    """
    if not preds or len(preds) < 3:
        return None
    by_frame = {int(e["frame_no"]): e for e in entries}
    fav = max(preds, key=lambda f: preds[f][1])          # ppw 最上位＝軸1
    honmei = next((int(e["frame_no"]) for e in entries
                   if e.get("prediction_mark") == 1), None)
    if honmei is None or fav != honmei or fav not in by_frame:
        return None                                       # ◎と不一致は母集団外
    taikou = next((int(e["frame_no"]) for e in entries
                   if e.get("prediction_mark") == 2), None)

    pp3 = sorted((preds[f][0] for f in preds), reverse=True)
    ppw = sorted((preds[f][1] for f in preds), reverse=True)
    rps = sorted(((int(e["frame_no"]), float(e.get("race_point") or 0))
                  for e in entries), key=lambda x: -x[1])
    rp_order = [f for f, _ in rps]
    rp_vals = [v for _, v in rps]
    if fav not in rp_order:
        return None
    fr = rp_order.index(fav)
    e = by_frame[fav]

    lines: dict = defaultdict(list)
    for x in entries:
        lines[_line_key(x)].append(x)
    line_sums = {k: sum(float(z.get("race_point") or 0) for z in v)
                 for k, v in lines.items()}
    fav_line_sum = line_sums[_line_key(e)]
    line_rank = sorted(line_sums.values(), reverse=True).index(fav_line_sum)
    mean_rp = sum(rp_vals) / len(rp_vals)

    return {
        "fav_pp3": preds[fav][0], "fav_ppw": preds[fav][1], "fav_pbad": preds[fav][2],
        "fav_pp3_gap12": pp3[0] - pp3[1], "fav_ppw_gap12": ppw[0] - ppw[1],
        "fav_rp_rank": float(fr), "fav_rp": rp_vals[fr],
        "fav_rp_gap_next": (rp_vals[fr] - rp_vals[fr + 1]
                            if fr + 1 < len(rp_vals) else 0.0),
        "fav_rp_gap_mean": rp_vals[fr] - mean_rp,
        "fav_frame": float(fav),
        "fav_line_size": float(e.get("line_size") or 1),
        "fav_line_pos": float(e.get("line_pos") or 0),
        "fav_is_leader": float(e.get("is_line_leader") or 0),
        "fav_is_solo": 1.0 if (e.get("line_size") or 1) == 1 else 0.0,
        "fav_style": float(STYLE_ENC.get(e.get("style"), -1)),
        "fav_class": float(1 if str(e.get("player_class") or "").startswith("S") else 0),
        "fav_line_rp_sum": fav_line_sum, "fav_line_rank": float(line_rank),
        "taikou_same_line": (
            1.0 if taikou is not None and taikou in by_frame
            and by_frame[taikou].get("line_group") is not None
            and by_frame[taikou].get("line_group") == e.get("line_group") else 0.0),
        "taikou_pp3": float(preds.get(taikou, (0.0, 0.0, 0.0))[0]) if taikou else 0.0,
        "taikou_rp_gap": (rp_vals[fr] - float(by_frame[taikou].get("race_point") or 0)
                          if taikou is not None and taikou in by_frame else 0.0),
        "n_stronger_line": float(line_rank),
        "_fav": fav,
    }


def build_favbust_row(meta: dict, entries: list[dict],
                      preds: dict[int, tuple]) -> dict | None:
    """レース1件分の67特徴 + `_fav` を返す。母集団外・データ不足なら None。"""
    rf = race_features(meta, entries)
    if rf is None:
        return None
    ff = fav_features(entries, preds)
    if ff is None:
        return None
    return {**rf, **ff}


def feature_vector(row: dict) -> list[float]:
    """`FAVBUST_FEATURE_COLS` の順に並べた特徴ベクトル。"""
    return [float(row[c]) for c in FAVBUST_FEATURE_COLS]


def roles_of(entries: list[dict], fav: int) -> dict[int, str]:
    """本命 fav から見た各車の役割を返す（fav 自身は含まない）。

    ⚠️ **fav が番手や3番手のこともある**（本命＝マーク屋）。その場合
    `本命ライン番手` は fav の**直後**の車を指す。fav がライン最後尾なら該当なし。
    """
    by_f = {int(e["frame_no"]): e for e in entries}
    if fav not in by_f:
        return {}
    fav_e = by_f[fav]
    fav_lg = fav_e.get("line_group")
    fav_pos = fav_e.get("line_pos") or 0

    line_rp: dict = defaultdict(float)
    for e in entries:
        if e.get("line_group") is not None:
            line_rp[e["line_group"]] += float(e.get("race_point") or 0)
    other_leads = [int(e["frame_no"]) for e in entries
                   if int(e["frame_no"]) != fav
                   and (e.get("is_line_leader") or 0) == 1
                   and e.get("line_group") is not None
                   and e.get("line_group") != fav_lg
                   and (e.get("line_size") or 1) > 1]
    strongest = (max(other_leads, key=lambda f: line_rp[by_f[f]["line_group"]])
                 if other_leads else None)

    out: dict[int, str] = {}
    for e in entries:
        f = int(e["frame_no"])
        if f == fav:
            continue
        if (e.get("line_size") or 1) == 1:
            out[f] = ROLE_SOLO
        elif fav_lg is not None and e.get("line_group") == fav_lg:
            out[f] = (ROLE_FAV_MATE if (e.get("line_pos") or 0) == fav_pos + 1
                      else ROLE_FAV_THIRD)
        elif (e.get("is_line_leader") or 0) == 1:
            out[f] = ROLE_LEAD_TOP if f == strongest else ROLE_LEAD_OTHER
        else:
            out[f] = ROLE_OTHER_MATE
    return out
