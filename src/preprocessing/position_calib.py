"""隊列位置バイアスの補正（2段目モデル）。

3着内モデル（`lgbm_wt`）は **隊列の後方に置かれる選手を系統的に過大評価する**。
検証（2026-08-03・honest split・test 2025-10-01〜2026-08-03）で測った
「実3着内率 − モデル予測」は次のとおりで、前方は較正済みかむしろ過小評価なのに
後方だけが全予測順位帯で 3.5〜10.9pt 過大評価されていた:

| モデル予測順位 | 前方(隊列前半) | 後方(隊列後半) |
|---|---|---|
| 1位 | −0.9% | −3.5% |
| 2位 | +1.6% | −5.4% |
| 3位 | +2.2% | −4.1% |

原因は物理的なもので、後方の選手は最終コーナーを外へ膨らんで回るため実走距離が
伸びる。上がりタイム（最終半周）から逆算した余剰距離は 400mバンク7車で約3.1m
（≒1車幅）、500mバンク9車では約8.0m（≒2.6車幅）で、**車数が増えるほど不利は増す**。

偏りは「モデルが高く買った後方選手」に集中する選択効果であるため、セル一律の
加算補正では順位がほとんど変わらず効かない（実測・軸2車的中 +0.43pt / Spearman −0.0014）。
`logit(pred_prob)` とレース内相対の位置・競走得点差を入力にした 2段目モデルで
組み替えるのが有効だった。

本番忠実構成（ベースモデルは車数を分けず1本・2段目は全車数合同）での効果:

| 指標 | 7車(n=19,422) | 9車(n=1,848) |
|---|---|---|
| 軸2車ともに3着内 | 52.1→52.8% (+0.68pt・有意) | 39.6→41.3% (+1.73pt・有意) |
| NDCG@3 | +0.02pt (ns) | 59.9→60.6% (+0.75pt・有意) |
| Brier | 0.19040→0.18858 | 0.18248→0.17978 |

**ROI は改善しない**（三連複5点で 73.9→74.3%）。的中率が下がるほど配当が上がって
相殺されるためで、狙いは軸2車の信頼度＝表示品質の向上にある。
**軸1（単独1位）の3着内率と1着的中率は改善しない**（いずれも有意差なし）ので、
1着を決め打つ買い目へ転用してはいけない。

詳細: kiseki メモリ `keirin_position_disadvantage_verification_2026_08_03`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 2段目モデルの入力。位置・予測順位は車数で正規化しているため 7車/9車で
# 閾値をハードコードせずに済む（9車は後方が4車以上になるなど車数で境界が動く）。
POSCAL_FEATURE_COLS: list[str] = [
    "pc_logit",        # logit(ベースモデルの3着内確率)
    "pc_pos_frac",     # 隊列推定位置（0=先頭 … 1=最後方）
    "pc_rp_adv",       # 競走得点 − 同レース他選手の平均
    "pc_rank_frac",    # ベースモデル予測順位（0=1位 … 1=最下位）
    "pc_line_size",
    "pc_line_pos",
    "pc_p_b",          # B（最終バック先頭）取得確率
    "pc_n_entries",
]

B_MODEL_NAME = "lgbm_wt_b"
POSCAL_MODEL_NAME = "lgbm_wt_poscal"

# 補正を適用する最小車数。**9車立てのみ**（7車には適用しない）。
#
# 【2026-08-03・7車への適用を取り下げた理由】
# 全車数に適用した初版では軸2車的中が 7車 +0.56pt / 9車 +1.41pt（いずれも有意）
# だったが、`wt_overlap_n`（軸2車とWT公式印◎◯の重なり）との整合を確認したところ、
# **7車の改善は「軸が◎◯の側へ寄った＝overlapが動いた」ことで全て説明され、
# 軸の質そのものは改善していない**ことが判明した:
#
# | 母集団を固定した比較 | 7車 | 9車 |
# |---|---|---|
# | overlapが変わらないレース | 54.2→54.2% (-0.01pt・ns) | 41.8→42.3% (+0.55pt・有意) |
# | 7S/7A(9S/9A)対象(overlap≤1のまま) | 39.0→39.2% (+0.25pt・ns) | 29.2→31.7% (+2.50pt・有意) |
#
# 7S/7A は overlap∈{0,1}（＝市場と不一致）だけを対象とするランクなので、
# overlap が 1→2 へ動いたレースは**改善ではなく7Bへの再分類**にすぎない。
# さらに 7S/7A の母集団が 4,346→4,285（純 -61件・-1.4%）と減る。
# 既に12件/日まで枯渇しているランクを見返りゼロで削ることになるため、7車では不適用とする。
#
# 9車は overlap を固定しても改善が残る＝市場追従では説明できない実質的な改善がある。
# 車数が増えるほど外を回る距離が伸びる（500mバンクで9車は7車の1.50倍）という
# 物理的根拠とも整合する。
POSCAL_MIN_ENTRIES = 8

_EPS = 1e-6


def _estimate_formation(rows: list[dict], p_b_col: str) -> None:
    """B取得確率から隊列（前→後）を推定し、各行に `pc_pos_frac` を書き込む。

    ライン単位で「先頭を取りそうな順」に並べる。ラインのスコアは構成員の
    B取得確率の最大値（B は基本的にライン先頭が取るため）。単騎は1車のライン
    として同じ土俵で順位づけする。ライン内は予想並び（`line_pos`）順。

    Note:
        winticket の `line_group` は**予想並びの配列インデックスであって隊列の
        前後を表さない**（第1ラインの実B取得率50.5%に対し第2/第3も約30%で
        差がつかない）。そのため `line_group` 順をそのまま隊列とみなしてはいけない。
    """
    n = len(rows)
    lines: dict[int, list[dict]] = {}
    solo: list[dict] = []
    for r in rows:
        lg = r.get("line_group")
        if r.get("line_size") in (1, None) or lg is None or pd.isna(lg):
            solo.append(r)
        else:
            lines.setdefault(int(lg), []).append(r)
    for members in lines.values():
        members.sort(key=lambda r: (r.get("line_pos") or 99))

    units: list[tuple[float, list[dict]]] = [
        (max(m[p_b_col] for m in members), members) for members in lines.values()
    ]
    units += [(s[p_b_col], [s]) for s in solo]
    units.sort(key=lambda u: -u[0])

    pos = 0
    for _, members in units:
        for r in members:
            r["pc_pos_frac"] = pos / (n - 1) if n > 1 else 0.0
            pos += 1


def add_position_features(
    df: pd.DataFrame,
    prob_col: str = "pred_prob",
    p_b_col: str = "pc_p_b",
) -> pd.DataFrame:
    """2段目モデルの入力特徴を付与する。

    Args:
        df: `race_key` / `frame_no` / `race_point` / `line_group` / `line_size` /
            `line_pos` と、`prob_col`・`p_b_col` を持つデータフレーム。
        prob_col: ベースモデルの3着内確率が入った列名。
        p_b_col: B取得確率が入った列名。

    Returns:
        `POSCAL_FEATURE_COLS` を追加した新しいデータフレーム。行順は入力を保つ。
    """
    out = df.copy()
    p = out[prob_col].astype(float).clip(_EPS, 1 - _EPS)
    out["pc_logit"] = np.log(p / (1 - p))
    out["pc_line_size"] = pd.to_numeric(out.get("line_size"), errors="coerce")
    out["pc_line_pos"] = pd.to_numeric(out.get("line_pos"), errors="coerce")

    out["_pc_order"] = np.arange(len(out))
    frames: list[pd.DataFrame] = []
    for _, grp in out.groupby("race_key", sort=False):
        n = len(grp)
        rows = grp.to_dict("records")
        _estimate_formation(rows, p_b_col)

        rp = pd.to_numeric(grp["race_point"], errors="coerce").fillna(0.0)
        total = float(rp.sum())
        # 自分以外の平均との差。1車立てはあり得ないが念のため0で守る。
        adv = rp - ((total - rp) / (n - 1) if n > 1 else 0.0)

        # ベースモデル予測順位（同値は安定ソートで frame_no 昇順）
        rank = grp[prob_col].rank(ascending=False, method="first") - 1

        g = pd.DataFrame(rows, index=grp.index)
        g["pc_rp_adv"] = adv.values
        g["pc_rank_frac"] = (rank / (n - 1)).values if n > 1 else 0.0
        g["pc_n_entries"] = n
        frames.append(g)

    res = pd.concat(frames).sort_values("_pc_order").drop(columns="_pc_order")
    return res.reset_index(drop=True) if df.index.equals(pd.RangeIndex(len(df))) else res


def apply_position_calibration(
    df: pd.DataFrame,
    prob_col: str = "pred_prob",
    feature_cols: list[str] | None = None,
    raw_suffix: str = "_raw",
) -> tuple[pd.DataFrame, bool]:
    """隊列位置バイアスを補正した確率で `prob_col` を置き換える。

    B取りモデル（`lgbm_wt_b`）と2段目モデル（`lgbm_wt_poscal`）の**両方が
    揃っているときだけ**適用し、どちらか欠けていれば入力をそのまま返す。
    補正前の値は `f"{prob_col}{raw_suffix}"` に必ず残す。

    Note:
        無言でフォールバックしない。適用したか否かを第2戻り値で返すので、
        呼び出し側は必ずログに出すこと。`add_sb_dyn_features_wt` の事故
        （特徴が全て0に潰れているのに誰も気づかない）と同型の失敗を避けるため。

    Returns:
        (補正後のデータフレーム, 補正を適用したか)
    """
    from src.models.trainer import load_model

    fcols = feature_cols if feature_cols is not None else POSCAL_FEATURE_COLS
    try:
        b_model = load_model(B_MODEL_NAME)
        calib = load_model(POSCAL_MODEL_NAME)
    except FileNotFoundError:
        out = df.copy()
        out[f"{prob_col}{raw_suffix}"] = out[prob_col]
        return out, False

    from src.preprocessing.feature_wt import prepare_X

    out = df.copy()
    out["pc_p_b"] = b_model.predict_proba(prepare_X(out))[:, 1]
    out = add_position_features(out, prob_col=prob_col)
    out[f"{prob_col}{raw_suffix}"] = out[prob_col]

    # 7車には適用しない（POSCAL_MIN_ENTRIES の解説を参照）。
    target = out["pc_n_entries"] >= POSCAL_MIN_ENTRIES
    if not target.any():
        return out, False
    out.loc[target, prob_col] = calib.predict_proba(out.loc[target, fcols])[:, 1]
    return out, True
