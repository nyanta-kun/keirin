"""
winticket 特徴量エンジニアリング

wt_entries + wt_races + venue_info から学習用データセットを構築する。
"""
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from ..database import get_connection


_STYLE_MAP: dict[str | None, int] = {
    # winticket の実際の脚質表記（逃/両/追）。前→後ろの順序エンコード。
    # （従来マップは「先行/捲り/差し…」前提で全件不一致→style_enc=-1 と死んでいたバグを2026-06-08修正）
    "逃": 0,   # 先行（逃げ）
    "両": 1,   # 両者（自在）
    "追": 2,   # 追込・差し
    # 後方互換（旧表記が来ても拾えるよう保持）
    "先行": 0, "捲り": 1, "差し": 2, "追い込み": 2, "追込": 2,
    None: -1,
    "": -1,
}

_CLASS_MAP = {
    "SS": 6, "S1": 5, "S2": 4, "A1": 3, "A2": 2, "A3": 1, "B": 0,
    # ガールズ L級（grade='L級'・girls-only レース）。winticket の
    # playerCurrentTermClass=4 がフォールバックで "cls4" として保存される。
    # 男子の 0-6 とは別カテゴリのため、別軸の識別子として 7 を付与
    # （girls レースは全車同クラス＝レース内では不変。男子と同一レースに混在しない）。
    "cls4": 7,
    # S級でグループ情報が欠損した稀な値（S級レースに S1/S2 と混在・約0.3%）→ S2 相当に寄せる
    "cls1": 4,
}


def load_raw_data_wt(min_date: str = "2025-01-01", max_date: str | None = None) -> pd.DataFrame:
    """wt_entries + wt_races + venue_info から生データを取得"""
    where = "WHERE r.race_date >= :min_date"
    params: dict = {"min_date": min_date}
    if max_date:
        where += " AND r.race_date <= :max_date"
        params["max_date"] = max_date

    query = f"""
        SELECT
            e.race_key,
            r.race_date,
            r.venue_id,
            r.grade,
            r.race_type,
            r.distance,
            r.start_at,
            e.frame_no,
            e.player_id,
            e.name,
            e.prefecture      AS player_prefecture,
            e.player_class,
            e.term,
            e.gear_ratio,
            e.style,
            e.race_point,
            e.prediction_mark,
            e.s_count,
            e.h_count,
            e.b_count,
            e.front_runner,
            e.stalker,
            e.deep_closer,
            e.marker,
            e.first_rate,
            e.second_rate,
            e.third_rate,
            e.ex_spurt_pct,
            e.ex_thrust_pct,
            e.ex_left_behind_pct,
            e.ex_split_line_pct,
            e.ex_snatch_pct,
            e.line_group,
            e.line_size,
            e.line_pos,
            e.is_line_leader,
            e.n_lines,
            e.finish_order,
            vi.bank_length,
            vi.is_indoor,
            vi.prefecture     AS venue_prefecture
        FROM wt_entries e
        JOIN wt_races r ON e.race_key = r.race_key
        LEFT JOIN venue_info vi ON r.venue_id = vi.venue_code
        {where}
        ORDER BY r.race_date, e.race_key, e.frame_no
    """

    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        from sqlalchemy import create_engine, text as sa_text
        engine = create_engine(db_url)
        pg_query = query.replace("wt_entries", "keirin.wt_entries") \
                        .replace("wt_races", "keirin.wt_races") \
                        .replace("venue_info", "keirin.venue_info")
        with engine.connect() as sa_conn:
            df = pd.read_sql_query(sa_text(pg_query), sa_conn, params=params)
        engine.dispose()
    else:
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

    return df


def build_features_wt(df: pd.DataFrame) -> pd.DataFrame:
    """winticket 生データから学習用特徴量を構築する"""
    df = df.copy()

    # ターゲット（finish_order=0 は欠車/失格＝着外。1〜3着のみを top3 とする）
    df["top3_flag"] = (df["finish_order"].notna()
                       & (df["finish_order"] >= 1)
                       & (df["finish_order"] <= 3)).astype(int)
    # 1着モデル用ターゲット（Phase B・2026-07-19〜。win_flag=1着のみ、DNF/2着以下は0）
    df["win_flag"] = (df["finish_order"].notna()
                      & (df["finish_order"] == 1)).astype(int)
    # 連帯モデル用ターゲット（2026-08-09 新設・ユーザー発案）。
    # top2_flag = 2着以内。**DNF/失格(finish_order=0) は含めない**（win_flag と同じ扱い）。
    #
    # 🔴 これが無いと「軸2を2着に置くのか3着に置くのか」をモデルが答えられない。
    #    3ヘッド(1着/3着内/着外)では並び順しか出せず、三連単の着順構成や
    #    二車複・二車単（定義上「2着以内」の券種）を3着内率で代用するしかなかった。
    #    win/top2/top3 が揃うと着順ごとの確率へ分解できる:
    #        P(1着) = win
    #        P(2着) = top2 − win
    #        P(3着) = top3 − top2
    #
    # ⚠️ **定義をここで明示すること。** bad6_flag は本番モデルが使う定義を作る
    #    コードがリポジトリに存在せず、2026-08-05 に実測で同定する羽目になった
    #    （下記コメント参照）。同じ轍を踏まないよう、定義とテストを最初から置く。
    df["top2_flag"] = (df["finish_order"].notna()
                       & (df["finish_order"] >= 1)
                       & (df["finish_order"] <= 2)).astype(int)
    # 大敗モデル用ターゲット（3ヘッド軸選定・2026-08-04〜／列の追加は 2026-08-05）。
    # 軸2 = argmax( z(3着内率) − 0.3×z(大敗率) ) の第2項に使う。
    #
    # ⚠️ **この列は本番モデル lgbm_wt_bad が既に使っている定義を後から明文化したもの**。
    # lgbm_wt_bad.meta.json は target="bad6_flag" と記録しているが、その列を作る
    # コードがリポジトリに存在せず（全git履歴を検索して0件）、本番モデルは
    # 再現不能な経路で学習されていた。2026-08-05 に実測で定義を同定した:
    #   本番モデルの平均予測 0.2795 に対し
    #     (finish_order >= 6)          → 実測率 0.2829（差 0.0034・AUC 0.768）✅
    #     (finish_order >= 6 or DNF=0) → 実測率 0.2941（差 0.0146・AUC 0.759）
    #   前者が一致。**DNF(finish_order=0) は「大敗」に含めない。**
    # exp系スクリプト（軸選定の検証に使用）の bad6 定義とも一致する。
    df["bad6_flag"] = (df["finish_order"].notna()
                       & (df["finish_order"] >= 6)).astype(int)

    # レート正規化（winticket は % 表記、0-1 スケールへ変換）
    df["first_rate_norm"]  = df["first_rate"].fillna(0.0) / 100.0
    df["second_rate_norm"] = df["second_rate"].fillna(0.0) / 100.0
    df["third_rate_norm"]  = df["third_rate"].fillna(0.0) / 100.0

    # 得点補完（2026-07-24修正: race_point=0.0はデビュー戦等の未点数選手を表す実値
    # であり欠損ではないが、そのまま使うとレース内平均・標準偏差(score_mean/score_std)
    # を引き下げ他選手のscore_zまで歪める。0.0もNaN同様に欠損扱いし中央値で補完する
    # （全車0.0＝ガールズ/新人戦は中央値算出の対象自体がNaNになりグローバル中央値で
    # 一律補完されるため、レース単位で見ても不自然にならない）。
    df["race_point"] = df["race_point"].replace(0.0, np.nan)
    med_rp = df["race_point"].median()
    df["race_point"] = df["race_point"].fillna(med_rp if not pd.isna(med_rp) else 50.0)

    df["gear_ratio"] = df["gear_ratio"].fillna(3.92)

    # 脚質エンコード
    df["style_enc"] = df["style"].map(_STYLE_MAP).fillna(-1).astype(int)

    # クラスエンコード
    df["player_class_enc"] = df["player_class"].map(_CLASS_MAP).fillna(-1).astype(int)

    # 期 正規化
    med_term = df["term"].median()
    df["period_norm"] = df["term"].fillna(med_term if not pd.isna(med_term) else 100) / 100.0

    # グレードエンコード（wt 実際の値: S級/A級/L級/SA混合）
    grade_map = {"S級": 3, "SA混合": 3, "A級": 2, "L級": 1}
    df["grade_enc"] = df["grade"].map(grade_map).fillna(2).astype(int)

    # 枠番特徴
    df["is_inner"] = (df["frame_no"] <= 3).astype(int)
    df["is_outer"] = (df["frame_no"] >= 7).astype(int)

    # ホーム判定
    if "player_prefecture" in df.columns and "venue_prefecture" in df.columns:
        df["is_home"] = (
            df["player_prefecture"].notna()
            & df["venue_prefecture"].notna()
            & (df["player_prefecture"] == df["venue_prefecture"])
        ).astype(int)
    else:
        df["is_home"] = 0

    # バンク長
    if "bank_length" in df.columns and df["bank_length"].notna().any():
        df["bank_length_enc"] = df["bank_length"].fillna(400) / 100.0
        df["is_indoor"] = df["is_indoor"].fillna(0).astype(int)
    else:
        df["bank_length_enc"] = 4.0
        df["is_indoor"] = 0

    # レース内相対特徴量
    grp_rp = df.groupby("race_key")["race_point"]
    df["score_rank"] = grp_rp.rank(ascending=False)
    df["score_mean"] = grp_rp.transform("mean")
    df["score_std"]  = grp_rp.transform("std").fillna(1.0).replace(0.0, 1.0)
    df["score_z"]    = ((df["race_point"] - df["score_mean"]) / df["score_std"]).clip(-5, 5)

    grp_wr = df.groupby("race_key")["first_rate_norm"]
    df["wr_rank"] = grp_wr.rank(ascending=False)

    grp_top3 = df.groupby("race_key")["third_rate_norm"]
    df["top3r_rank"] = grp_top3.rank(ascending=False)

    # AI予想マーク（0=なし, 1=本命, 2=対抗, 3=単穴, 4=連下）
    df["prediction_mark"] = df["prediction_mark"].fillna(0).astype(int)

    # セクター回数
    df["s_count"] = df["s_count"].fillna(0)
    df["h_count"] = df["h_count"].fillna(0)
    df["b_count"] = df["b_count"].fillna(0)

    # 上がり戦術率（%→0-1）。ex_spurt_pct/ex_thrust_pctは2026-07-31にFEATURE_COLS_WT
    # から除外済み（train/serve skew実測・FEATURE_COLS_WT直前のコメント参照）だが、
    # SELECT・正規化自体は分析用途・将来のpoint-in-time化のため残置する。
    df["ex_spurt_pct"]       = (df["ex_spurt_pct"].fillna(0.0)       / 100.0).clip(0, 1)
    df["ex_thrust_pct"]      = (df["ex_thrust_pct"].fillna(0.0)      / 100.0).clip(0, 1)
    df["ex_left_behind_pct"] = (df["ex_left_behind_pct"].fillna(0.0) / 100.0).clip(0, 1)

    # ライン特徴量（winticket 専有）
    df["line_size"]      = df["line_size"].fillna(1).astype(int)
    df["line_pos"]       = df["line_pos"].fillna(1).astype(int)
    df["is_line_leader"] = df["is_line_leader"].fillna(0).astype(int)
    df["n_lines"]        = df["n_lines"].fillna(0).astype(int)
    df["is_isolated"]    = (df["line_size"] == 1).astype(int)

    # レース内でのライン規模比率（大きいラインほど有利）
    n_in_race = df.groupby("race_key")["frame_no"].transform("count")
    df["line_frac"] = (df["line_size"] / n_in_race.replace(0, 1)).clip(0, 1)

    # 脚質構成（展開シグナル・レース内の逃げ人数）。n_lines と独立(相関-0.01)の新シグナル。
    # 先行0人=展開不分明で波乱・高配当（oddspark/競輪keirin 監査＋自前検証 2026-06-09）。
    df["n_senko"] = (df["style_enc"] == 0).astype(int).groupby(df["race_key"]).transform("sum")

    # ks流ローリング特徴（point-in-time。履歴 wt_entries から計算）
    df = add_rolling_features_wt(df)

    # 競走得点トレンド（point-in-time。履歴 wt_entries の得点時系列から計算）
    df = add_rp_trend_features_wt(df)

    # レース単位S/B・上がり由来のローリング特徴（point-in-time・2026-07-18追加）
    df = add_sb_dyn_features_wt(df)

    # 隊列推定位置（2026-08-03追加・b_rate_90 に依存するため sb_dyn の後に置く）
    df = add_formation_features_wt(df)

    # レース種別・ライン実力（2026-08-04追加）
    df = add_race_type_features_wt(df)
    df = add_line_strength_features_wt(df)

    # 頭対頭対戦成績(H2H)は2026-07-28に実装しFEATURE_COLS_WTへ追加したが、S1/S9の
    # honest全期間walk-forwardでROIが悪化(S1 443.0%→363.5%・S9 412.8%→286.8%)した
    # ため本番投入を撤回した（S7のみ改善401.1%→424.8%。詳細
    # [[keirin_netkeirin_h2h_feature_2026_07_28]]）。add_h2h_features_wt()自体は
    # 将来の再検証用に残すが、ここでは呼び出さない（FEATURE_COLS_WTにも含めない）。

    # M-1: 学習(train_lgbm dropna)・推論(prepare_X fillna)・バックテストで
    # 同一の特徴表現になるよう、ソースで FEATURE_COLS_WT の NaN を 0 に統一保証する
    # （現状 build 過程で各特徴は補完済＝実質no-op だが、将来の fill 漏れによる
    #  train/serve skew を構造的に防ぐ安全網）。
    present = [c for c in FEATURE_COLS_WT if c in df.columns]
    df[present] = df[present].fillna(0)

    return df


RACE_TYPE_COLS_WT = [
    "rt_is_final", "rt_is_semifinal", "rt_is_heat", "rt_is_senbatsu",
    "rt_is_tokusen", "rt_is_hatsu", "rt_is_ippan",
]
LINE_STRENGTH_COLS_WT = [
    "line_rp_sum", "line_rp_max", "line_rp_mean",
    "line_rank_by_rp", "line_rp_gap_top",
]


def add_race_type_features_wt(df: pd.DataFrame) -> pd.DataFrame:
    """レース種別（wt_races.race_type）をキーワードフラグとして付与する（2026-08-04追加）。

    3着内モデルは **勝ち上がりで実力者が揃うレースで軸1を系統的に過大評価する**。
    honest 実測（36,831レース・`scripts/exp_axis1_miss_analysis.py`）の較正誤差:

    | race_type | n | 予測p1 | 実測 | 乖離 |
    |---|---|---|---|---|
    | 初特選 | 1,536 | 75.1% | 68.4% | −6.7pt |
    | 選抜 | 1,228 | 76.0% | 69.7% | −6.3pt |
    | ガールズ決勝 | 549 | 91.7% | 85.4% | −6.2pt |
    | 決勝 | 1,554 | 76.0% | 70.4% | −5.6pt |
    | ガールズ予選 | 2,047 | 92.7% | 94.8% | +2.1〜+2.2pt |

    全体の較正は±1pt以内で正確なのに種別ごとに±7pt振れており、`grade_enc`
    （S級/A級/L級）だけでは表現できない。序数エンコードではなくキーワードの
    フラグにしてあるのは、race_type が100種類以上のロングテール（頻度上位20種で
    約95%）で、学習データに無い種別が出ても分解して表現できるようにするため。
    """
    out = df.copy()
    s = out["race_type"].fillna("") if "race_type" in out.columns else pd.Series(
        [""] * len(out), index=out.index)
    s = s.astype(str)
    out["rt_is_semifinal"] = s.str.contains("準決").astype(int)
    out["rt_is_final"] = (s.str.contains("決勝") & ~s.str.contains("準決")).astype(int)
    out["rt_is_heat"] = s.str.contains("予選").astype(int)
    out["rt_is_senbatsu"] = s.str.contains("選抜").astype(int)
    out["rt_is_tokusen"] = s.str.contains("特選").astype(int)
    out["rt_is_hatsu"] = s.str.startswith("初").astype(int)
    out["rt_is_ippan"] = s.str.contains("一般").astype(int)
    return out


def add_line_strength_features_wt(df: pd.DataFrame) -> pd.DataFrame:
    """ライン単位の実力集約を付与する（2026-08-04追加）。

    競輪はライン戦であるにもかかわらず、既存特徴はラインの**構造**
    （line_size / line_pos / is_line_leader / n_lines / line_frac）しか持たず、
    「そのラインが強いか」を一切持っていなかった。A/B（2窓×5seed・
    `scripts/exp_racetype_field_ab.py`）で `line_rp_gap_top` が全60特徴中
    **2位**の重要度に入り、ΔAUC +0.00206/+0.00218・Δ1位3着内 +0.22/+0.08pt。

    同時に検証した「レース単位の集約」（rp_mean/rp_std 等）は AUC を +0.0017〜
    0.0027 上げるのに 1位3着内率は窓1で −0.20pt と悪化したため**不採用**とした。
    レース内で全車同値の特徴はレース間の識別しか改善せず、レース内の順位付けに
    寄与しないため（AUCだけで採否を決めてはいけない実例）。

    ラインは `line_group` で識別する。値そのものは隊列の前後を表さないが所属の
    識別には使える。単騎（line_group 欠損含む）は1車ラインとして同じ土俵で扱う。
    `race_point` は開催中に更新されない安定値・`line_group` は出走表情報のため、
    `ex_*` 系のような train/serve skew リスクは持たない。
    """
    out = df.copy()
    rp = out["race_point"].astype(float).fillna(0.0)
    # line_group 欠損は「単騎扱い」。車番から負のグループIDを与えて一意にする。
    lg = out["line_group"]
    lg = lg.where(lg.notna(), -out["frame_no"].astype(float))
    key = out["race_key"].astype(str) + "#" + lg.astype(str)

    grp = rp.groupby(key)
    out["line_rp_sum"] = grp.transform("sum")
    out["line_rp_max"] = grp.transform("max")
    out["line_rp_mean"] = grp.transform("mean")

    # レース内でのラインの強さ順位（0=最強）と、最強ラインとの得点合計差
    per_line = out.groupby(["race_key", key.rename("_lk")])["line_rp_sum"].first()
    rank = (per_line.groupby(level=0).rank(ascending=False, method="min") - 1)
    top = per_line.groupby(level=0).transform("max")
    lk = list(zip(out["race_key"], key))
    out["line_rank_by_rp"] = [rank.get(k, 0.0) for k in lk]
    out["line_rp_gap_top"] = [top.get(k, 0.0) for k in lk] - out["line_rp_sum"]

    for c in LINE_STRENGTH_COLS_WT:
        out[c] = out[c].astype(float).fillna(0.0)
    return out


FORMATION_COLS_WT = ["formation_pos_frac", "formation_line_rank"]


def add_formation_features_wt(df: pd.DataFrame) -> pd.DataFrame:
    """隊列（前→後）の推定位置をレース内相対で付与する（2026-08-03追加）。

    3着内モデルは **隊列後方の選手を系統的に過大評価する**。後方の選手は最終コーナーを
    外へ膨らんで回るため実走距離が伸びるという物理的な理由による。上がりタイム
    （最終半周）の「捲−差」から逆算した余剰距離は次のとおりで、**車数が増えるほど
    不利が増す**（同一500mバンクで9車は7車の1.50倍）:

    | バンク | 車数 | 捲−差 | 余剰距離 | 内側からの幅 |
    |---|---|---|---|---|
    | 400m | 7車 | −0.190秒 | 3.13m | 約1.0m（≒1車幅）|
    | 500m | 7車 | −0.262秒 | 5.34m | 約1.7m |
    | 500m | 9車 | −0.371秒 | 8.01m | 約2.6m |

    位置そのものは既存特徴では表現できない。`line_pos` はライン内の順番、
    `line_group` は winticket の**予想並びの配列インデックスで隊列の前後を表さない**
    （第1ラインの先頭が実際にBを取る率50.5%に対し第2/第3も約30%で差がつかない）。
    木モデルは行をまたぐ順位を自力で構成できないため、ここで明示的に作る。

    - formation_pos_frac  : 隊列推定位置（0=先頭 … 1=最後方）
    - formation_line_rank : 所属ラインの推定順（0=先行ライン … 1=最後方ライン）

    どちらも**車数で正規化**してあるため、9車で「後方」が4車以上になるといった
    境界の違いを閾値のハードコードなしに吸収する。

    ラインの並び順は構成員の `b_rate_90`（既存のpoint-in-time B取得率）の最大値で決める
    （Bは基本的にライン先頭が取るため）。単騎は1車のラインとして同じ土俵で順位づける。
    **モデルを使わない純粋な特徴変換**なので追加の学習物・リーク経路を持たない。

    Note:
        `b_rate_90` に依存するため `add_sb_dyn_features_wt()` の後に呼ぶこと。
        全選手の `b_rate_90` が同値（履歴が無い期間など）だと順位が付かず
        `line_group` 順へ縮退する。2024-01 以前の学習データがこれに当たる。
    """
    if "b_rate_90" not in df.columns:
        out = df.copy()
        for c in FORMATION_COLS_WT:
            out[c] = 0.0
        return out

    out = df.copy()
    out["fm_row_id"] = np.arange(len(out))
    pos_frac = np.zeros(len(out))
    line_rank = np.zeros(len(out))
    idx_of = {v: i for i, v in enumerate(out["fm_row_id"].values)}

    for _, grp in out.groupby("race_key", sort=False):
        n = len(grp)
        lines: dict[int, list[tuple]] = {}
        solo: list[tuple] = []
        for row in grp.itertuples(index=False):
            lg = getattr(row, "line_group", None)
            key = (getattr(row, "b_rate_90", 0.0) or 0.0,
                   getattr(row, "line_pos", 99) or 99,
                   row.fm_row_id)
            if getattr(row, "line_size", 1) in (1, None) or lg is None or pd.isna(lg):
                solo.append(key)
            else:
                lines.setdefault(int(lg), []).append(key)
        for mem in lines.values():
            mem.sort(key=lambda t: t[1])          # ライン内は予想並び順
        units = [(max(t[0] for t in mem), mem) for mem in lines.values()]
        units += [(t[0], [t]) for t in solo]
        units.sort(key=lambda u: -u[0])

        i = 0
        n_units = max(len(units) - 1, 1)
        for li, (_, mem) in enumerate(units):
            for t in mem:
                j = idx_of[t[2]]
                pos_frac[j] = i / (n - 1) if n > 1 else 0.0
                line_rank[j] = li / n_units
                i += 1

    out["formation_pos_frac"] = pos_frac
    out["formation_line_rank"] = line_rank
    return out.drop(columns="fm_row_id")


ROLLING_COLS_WT = [
    "win_3m", "top3_3m", "quin_3m", "win_6m", "top3_6m", "quin_6m",
    "venue_wr", "days_since", "wr_trend",
]


def add_rolling_features_wt(df: pd.DataFrame) -> pd.DataFrame:
    """選手の過去成績から point-in-time ローリング特徴を付与する。

    df は race_key / player_id / race_date / venue_id 列を持つ前提。
    finish_order=0(欠車/失格) は実績から除外。現レース・未確定レースも
    「履歴に無い行」として as-of で正しく計算する（学習/予測 両対応）。
    """
    df = df.copy()
    if "player_id" not in df.columns or "race_date" not in df.columns:
        # 必要列が無ければ既定値で埋める（後方互換）
        for c in ROLLING_COLS_WT:
            df[c] = 0.0
        return df

    df["_dt"] = pd.to_datetime(df["race_date"])

    rolling_sql = (
        "SELECT e.race_key, e.player_id, e.finish_order, r.race_date, r.venue_id "
        "FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key "
        "WHERE e.finish_order >= 1"
    )
    db_url = os.environ.get("KEIRIN_DB_URL")
    if db_url:
        from sqlalchemy import create_engine, text as sa_text
        engine = create_engine(db_url)
        pg_sql = rolling_sql.replace("wt_entries", "keirin.wt_entries") \
                             .replace("wt_races", "keirin.wt_races")
        with engine.connect() as sa_conn:
            H = pd.read_sql_query(sa_text(pg_sql), sa_conn)
        engine.dispose()
    else:
        with get_connection() as conn:
            H = pd.read_sql_query(rolling_sql, conn)
    H["_dt"] = pd.to_datetime(H["race_date"])
    H["win"]  = (H["finish_order"] == 1).astype(float)
    H["top3"] = H["finish_order"].between(1, 3).astype(float)
    H["quin"] = H["finish_order"].between(1, 2).astype(float)
    H = H.sort_values(["player_id", "_dt"]).reset_index(drop=True)

    def _rm(col, w):
        return (H.set_index("_dt").groupby("player_id")[col]
                .rolling(w, closed="left").mean()
                .reset_index(level=0, drop=True).values)

    for c in ["win", "top3", "quin"]:
        H[f"{c}_3m"] = _rm(c, "90D")
        H[f"{c}_6m"] = _rm(c, "180D")
    H["venue_wr"] = (H.sort_values(["player_id", "venue_id", "_dt"])
                     .groupby(["player_id", "venue_id"])["win"]
                     .apply(lambda s: s.expanding().mean().shift(1))
                     .reset_index(level=[0, 1], drop=True))
    H["days_since"] = H.groupby("player_id")["_dt"].diff().dt.days
    H["wr_trend"] = H["win_3m"] - H["win_6m"]

    Hroll = H[["race_key", "player_id"] + ROLLING_COLS_WT]
    out = df.merge(Hroll, on=["race_key", "player_id"], how="left")

    # 履歴に存在しない行（当日・未確定レース）は as-of で個別計算
    hist_keys = set(map(tuple, Hroll[["race_key", "player_id"]].to_numpy()))
    for idx in out.index:
        rk, pid = out.at[idx, "race_key"], out.at[idx, "player_id"]
        if (rk, pid) in hist_keys:
            continue
        dt = out.at[idx, "_dt"]
        ven = out.at[idx, "venue_id"] if "venue_id" in out.columns else None
        hp = H[(H["player_id"] == pid) & (H["_dt"] < dt)]
        if hp.empty:
            continue
        w3 = hp[hp["_dt"] >= dt - pd.Timedelta("90D")]
        w6 = hp[hp["_dt"] >= dt - pd.Timedelta("180D")]
        out.at[idx, "win_3m"]  = w3["win"].mean()  if len(w3) else np.nan
        out.at[idx, "top3_3m"] = w3["top3"].mean() if len(w3) else np.nan
        out.at[idx, "quin_3m"] = w3["quin"].mean() if len(w3) else np.nan
        out.at[idx, "win_6m"]  = w6["win"].mean()  if len(w6) else np.nan
        out.at[idx, "top3_6m"] = w6["top3"].mean() if len(w6) else np.nan
        out.at[idx, "quin_6m"] = w6["quin"].mean() if len(w6) else np.nan
        hv = hp[hp["venue_id"] == ven] if ven is not None else hp.iloc[0:0]
        out.at[idx, "venue_wr"]   = hv["win"].mean() if len(hv) else np.nan
        out.at[idx, "days_since"] = (dt - hp["_dt"].max()).days
        out.at[idx, "wr_trend"]   = out.at[idx, "win_3m"] - out.at[idx, "win_6m"]

    # 履歴不足は固定既定値（学習/予測で同一）。rate=0, days_since=30, trend=0
    fill = {c: (30.0 if c == "days_since" else 0.0) for c in ROLLING_COLS_WT}
    out = out.fillna(value=fill)
    out = out.drop(columns=["_dt"], errors="ignore")
    return out


RP_TREND_COLS_WT = [
    "rp_prev_delta", "rp_delta_90", "rp_delta_180", "rp_trend",
]


def add_rp_trend_features_wt(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """選手単位の競走得点トレンド特徴を付与する（point-in-time）。

    df は player_id / race_date / race_point 列を持つ前提
    （race_point は build_features_wt で補完済みの当日発表値）。

    - rp_prev_delta : 今回得点 − 前回出走時（前回の異なる race_date）の得点
    - rp_delta_90   : 今回得点 − 過去90日の平均得点（当日を含まない）
    - rp_delta_180  : 同180日
    - rp_trend      : 過去90日平均 − 過去180日平均（中期トレンド）

    履歴の rolling は closed="left" で当日を除外（リークなし）。同一選手・
    同一日の複数走は median で1点に集約（得点は節内で不変）。当日・未確定
    レースの行も wt_entries に存在するため merge で解決できる。履歴不足
    （新人等）は 0.0 で補完する。

    汚染対策: finish_order IS NULL の過去行は wave-picks の AIスコア上書き
    （pred_prob_pct=0〜100）が恒久残存し得るため、race_point 値を NaN 化して
    集計（rolling 平均・median・rp_prev）から除外する。行自体は当日・未確定
    レースの merge キーとして残す（closed="left" のため当日の自値は元々窓に
    入らない）。SQL の race_point > 20 はゼロ・欠損系の除外として維持。

    Args:
        df: 特徴量付与対象の DataFrame。
        history: テスト用に注入する履歴
            （player_id/race_point/race_date/finish_order 列）。
            None の場合は DB（wt_entries × wt_races）から読む。
    """
    df = df.copy()
    if "player_id" not in df.columns or "race_date" not in df.columns:
        # 必要列が無ければ既定値で埋める（後方互換）
        for c in RP_TREND_COLS_WT:
            df[c] = 0.0
        return df

    if history is None:
        rp_sql = (
            "SELECT e.player_id, e.race_point, r.race_date, e.finish_order "
            "FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key "
            "WHERE e.race_point IS NOT NULL AND e.race_point > 20"
        )
        db_url = os.environ.get("KEIRIN_DB_URL")
        if db_url:
            from sqlalchemy import create_engine, text as sa_text
            engine = create_engine(db_url)
            pg_sql = rp_sql.replace("wt_entries", "keirin.wt_entries") \
                            .replace("wt_races", "keirin.wt_races")
            with engine.connect() as sa_conn:
                H = pd.read_sql_query(sa_text(pg_sql), sa_conn)
            engine.dispose()
        else:
            with get_connection() as conn:
                H = pd.read_sql_query(rp_sql, conn)
    else:
        H = history.copy()

    H["_dt"] = pd.to_datetime(H["race_date"])
    # finish_order 未確定（NULL）の過去行は AIスコア上書きが恒久残存し得るため
    # 値のみ NaN 化（行は merge キーとして残す。median/rolling/ffill は NaN を除外）
    H.loc[H["finish_order"].isna(), "race_point"] = np.nan
    # 同一選手・同一日の重複（同節複数走）は1点に集約（得点は節内で不変）
    H = (H.groupby(["player_id", "race_date"], as_index=False)
           .agg(race_point=("race_point", "median"), _dt=("_dt", "first")))
    H = H.sort_values(["player_id", "_dt"]).reset_index(drop=True)

    def _rm(w: str) -> np.ndarray:
        return (H.set_index("_dt").groupby("player_id")["race_point"]
                .rolling(w, closed="left").mean()
                .reset_index(level=0, drop=True).values)

    H["rp_ma90"] = _rm("90D")
    H["rp_ma180"] = _rm("180D")
    # 前回値は「直前の非NaN値」（NaN行の直後でも最後の実値を引く・選手境界は跨がない）
    H["rp_prev"] = H.groupby("player_id")["race_point"].transform(
        lambda s: s.ffill().shift(1))

    key = H[["player_id", "race_date", "rp_ma90", "rp_ma180", "rp_prev"]]
    out = df.merge(key, on=["player_id", "race_date"], how="left")
    out["rp_prev_delta"] = out["race_point"] - out["rp_prev"]
    out["rp_delta_90"] = out["race_point"] - out["rp_ma90"]
    out["rp_delta_180"] = out["race_point"] - out["rp_ma180"]
    out["rp_trend"] = out["rp_ma90"] - out["rp_ma180"]
    # 履歴不足（新人等）は 0.0（学習/予測で同一）
    for c in RP_TREND_COLS_WT:
        out[c] = out[c].fillna(0.0)
    return out.drop(columns=["rp_ma90", "rp_ma180", "rp_prev"], errors="ignore")


SB_DYN_COLS_WT = [
    "b_rate_90", "s_rate_90", "fh_rel_90", "fh_best_rate_90",
]


def add_sb_dyn_features_wt(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """レース単位の S/B 取得・上がりタイム由来のローリング特徴を付与する（point-in-time）。

    データ源はバックフィル済みの wt_entries.res_standing / res_back / final_half
    （2024-01〜。[[keirin_sb_dynamics_pipeline]] 参照）。全て過去レースのみ・
    closed="left" 90日窓・レース内相対化済み:

    - b_rate_90       : 直近90日の B（バック先頭）取得率
    - s_rate_90       : 直近90日の S（スタンディング先頭）取得率
    - fh_rel_90       : 直近90日の上がり相対値平均（自上がり − レース中央値・負=速い）
    - fh_best_rate_90 : 直近90日の「レース内上がり最速」率

    A/B検証（exp_sb_dyn_ab.py・2独立窓×5seed）: ΔAUC +0.013/+0.011・
    指数1位3着内率 +0.93pt/+1.10pt・重要度2〜9位/48（2026-07-18採用）。

    実装上の要点:
    - 履歴 H は wt_entries **全行**（未確定・ラベル欠損行を含む）。ラベル欠損は
      NaN のままにし rolling 平均から自動除外される一方、行は (race_key, player_id)
      merge キーとして残るため、当日・未確定レースの予測時も同一経路で as-of 値が
      付く（train/serve skew なし・rp_trend と同じ設計）。
    - 2024-01 以前はラベルが存在せず窓が空 → 0.0 補完（学習/予測で同一の既定値）。

    **【2026-07-18導入〜2026-07-28発見・修正の重大バグ】** 上記の設計方針にも
    関わらず、実装は `H[H["finish_order"] >= 1]` で未確定行（finish_order が
    NaN の当日・未来レース）を丸ごと drop していた（DNS/DNF除外フィックス時に
    誤って未確定行まで一緒に落としてしまっていた）。この結果、対象レース自身が
    merge キーとして存在しなくなり、**発走前ライブ予測（このメソッドが最も
    必要とされる対象）では全選手のsb_dyn 4特徴が常に0.0補完**になっていた。
    学習データは常に確定済みレースのみを使うため「全選手sb_dyn=0」という
    入力パターンをモデルは学習時に一度も見ておらず、この分布外入力に対して
    predict_proba() が全選手ほぼ0%という壊滅的に縮退した出力を返す事故が
    発生していた（"軸2車が実際には全く自信のない状態" になるため、
    S7/S9/7A/9Aのaxis_sum（軸2車のpred_prob合計）が異常に低くなり、
    `S7_AXIS_SUM_MAX<=1.3`等のゲートをほぼ無条件で通過してしまい、
    2026-07-27〜の候補選出数が過去のhonest walk-forward実績（rebuild系
    スクリプトは常に確定済みレースのみを対象とするため本バグの影響を受けず、
    正常なsb_dyn値で計算されていた）を大幅に上回る形で顕在化した）。
    修正: DNS/DNF・未確定いずれも「行は残し、集計対象の値だけNaN化」という
    rp_trend と同じ方式に統一し、対象レース自身がmerge キーから消えないように
    した。詳細はkeirin CLAUDE.md/docs/prediction-factors.md更新履歴参照。

    Args:
        df: 特徴量付与対象（race_key / player_id / race_date 列を持つ前提）。
        history: テスト用に注入する履歴（race_key/player_id/res_standing/
            res_back/final_half/race_date 列）。None の場合は DB から読む。
    """
    df = df.copy()
    if "player_id" not in df.columns or "race_date" not in df.columns:
        for c in SB_DYN_COLS_WT:
            df[c] = 0.0
        return df

    if history is None:
        sb_sql = (
            "SELECT e.race_key, e.player_id, e.res_standing, e.res_back, "
            "e.final_half, e.finish_order, r.race_date "
            "FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key"
        )
        db_url = os.environ.get("KEIRIN_DB_URL")
        if db_url:
            from sqlalchemy import create_engine, text as sa_text
            engine = create_engine(db_url)
            pg_sql = sb_sql.replace("wt_entries", "keirin.wt_entries") \
                            .replace("wt_races", "keirin.wt_races")
            with engine.connect() as sa_conn:
                H = pd.read_sql_query(sa_text(pg_sql), sa_conn)
            engine.dispose()
        else:
            with get_connection() as conn:
                H = pd.read_sql_query(sb_sql, conn)
    else:
        H = history.copy()

    H["_dt"] = pd.to_datetime(H["race_date"])

    # DNS/DNF（finish_order<1・欠車や途中棄権）・未確定（finish_order NaN・
    # 当日/未来の未終了レース）は res_back/standing/final_half の値が完走者と
    # 同じ意味を持たない、または存在しないため、集計値（_b/_s/_fh系）だけを
    # NaN化してローリング計算・レース内中央値/最速判定から除外する。
    # ただし行自体は (race_key, player_id) merge キーとして残す（rp_trend と
    # 同じ設計）。ここで行ごと drop すると、発走前ライブ予測（このメソッドが
    # 最も必要とされる対象）では対象レース自身が merge キーに存在せず必ず
    # 0.0補完になり、モデルが学習時に一度も見ない「全選手sb_dyn=0」という
    # 分布外入力を受け取って予測が崩壊する事故になる（2026-07-18〜07-28に
    # 実際に発生・詳細は本関数docstring参照）。
    if "finish_order" in H.columns:
        _confirmed = pd.to_numeric(H["finish_order"], errors="coerce") >= 1
    else:
        _confirmed = pd.Series(True, index=H.index)

    # レース内相対化: fh_rel = 自上がり − レース中央値（負=速い）・fh_best = レース内最速。
    # final_half<=0 や欠損、DNS/DNF/未確定は NaN（rolling・中央値/最速判定から除外）。
    fh = pd.to_numeric(H["final_half"], errors="coerce")
    H["_fh"] = fh.where(fh > 0)
    H.loc[~_confirmed, "_fh"] = np.nan
    med = H.groupby("race_key")["_fh"].transform("median")
    mn = H.groupby("race_key")["_fh"].transform("min")
    H["_fh_rel"] = H["_fh"] - med
    H["_fh_best"] = (H["_fh"] == mn).astype(float).where(H["_fh"].notna())
    H["_b"] = pd.to_numeric(H["res_back"], errors="coerce")
    H["_s"] = pd.to_numeric(H["res_standing"], errors="coerce")
    H.loc[~_confirmed, ["_b", "_s"]] = np.nan

    H = H.sort_values(["player_id", "_dt"]).reset_index(drop=True)

    def _rm(col: str) -> np.ndarray:
        return (H.set_index("_dt").groupby("player_id")[col]
                .rolling("90D", closed="left").mean()
                .reset_index(level=0, drop=True).values)

    H["b_rate_90"] = _rm("_b")
    H["s_rate_90"] = _rm("_s")
    H["fh_rel_90"] = _rm("_fh_rel")
    H["fh_best_rate_90"] = _rm("_fh_best")

    key = H[["race_key", "player_id"] + SB_DYN_COLS_WT]
    out = df.merge(key, on=["race_key", "player_id"], how="left")
    # 履歴不足（2024-01以前・新人等）は 0.0（学習/予測で同一の既定値）
    for c in SB_DYN_COLS_WT:
        out[c] = out[c].fillna(0.0)
    return out


H2H_COLS_WT = ["h2h_win_rate", "h2h_n_total", "h2h_net_norm"]


def add_h2h_features_wt(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """選手ペア間の頭対頭対戦成績(H2H)を point-in-time で付与する（netkeirin「対戦表」相当）。

    netkeirin未活用データ調査（[[keirin_netkeirin_h2h_feature_2026_07_28]]）の本命成果。
    5フォールドwalk-forwardでS4戦略ROIが91.0%(n=120)→130.3%(n=134)に改善（5期中4期で改善・
    1期は誤差内横ばい・悪化なし）。history 省略時は wt_entries 全履歴（未確定の当日行を含む）
    をDBから読み、race_date→start_at 順に1パス走査して各レース「前」時点の対戦成績を算出する。
    当日・未確定レースの行も history 自身に含まれるため（他の add_*_features_wt と異なり）
    別途のas-ofフォールバック処理は不要。

    - h2h_win_rate : 当該レース出走者のうち対戦履歴がある相手への勝率
                     （先着数/対戦数、対戦履歴が無ければ0.5補完）
    - h2h_n_total  : 対戦履歴がある相手との対戦回数の合計（カバレッジ）
    - h2h_net_norm : (先着数-後着数)の合計 / レース出走頭数（対戦履歴が無ければ0）

    Args:
        df: 特徴量付与対象（race_key / player_id / race_date 列を持つ前提）。
        history: テスト用に注入する履歴（race_key/player_id/finish_order/race_date/
            start_at 列）。None の場合は DB から読む。
    """
    df = df.copy()
    if "player_id" not in df.columns or "race_date" not in df.columns:
        for c in H2H_COLS_WT:
            df[c] = 0.5 if c == "h2h_win_rate" else 0.0
        return df

    if history is None:
        h2h_sql = (
            "SELECT e.race_key, e.player_id, e.finish_order, r.race_date, r.start_at "
            "FROM wt_entries e JOIN wt_races r ON e.race_key=r.race_key"
        )
        db_url = os.environ.get("KEIRIN_DB_URL")
        if db_url:
            from sqlalchemy import create_engine, text as sa_text
            engine = create_engine(db_url)
            pg_sql = h2h_sql.replace("wt_entries", "keirin.wt_entries") \
                             .replace("wt_races", "keirin.wt_races")
            with engine.connect() as sa_conn:
                H = pd.read_sql_query(sa_text(pg_sql), sa_conn)
            engine.dispose()
        else:
            with get_connection() as conn:
                H = pd.read_sql_query(h2h_sql, conn)
    else:
        H = history.copy()

    H["fin"] = pd.to_numeric(H["finish_order"], errors="coerce")
    race_order = (H.groupby("race_key")
                  .agg(race_date=("race_date", "first"), start_at=("start_at", "first"))
                  .sort_values(["race_date", "start_at"])
                  .index.tolist())

    h2h_win: dict = defaultdict(int)   # (pid_a, pid_b) a<b → a が先着した回数
    h2h_n: dict = defaultdict(int)     # (pid_a, pid_b) a<b → 対戦回数（両者完走）
    groups = {rk: g for rk, g in H.groupby("race_key", sort=False)}
    out: dict = {}

    for rk in race_order:
        g = groups[rk]
        pids = g["player_id"].tolist()
        fins = g["fin"].tolist()

        # --- 特徴（レース前の対戦履歴で） ---
        for p in pids:
            wins, matches, net = 0, 0, 0
            for q in pids:
                if q == p:
                    continue
                key = (p, q) if p < q else (q, p)
                n = h2h_n[key]
                if n == 0:
                    continue
                w_ab = h2h_win[key]  # min(p,q) が先着した回数
                w_p = w_ab if p < q else (n - w_ab)
                matches += n
                wins += w_p
                net += w_p - (n - w_p)
            out[(rk, p)] = (
                (wins / matches) if matches > 0 else np.nan,
                float(matches),
                float(net),
            )

        # --- 更新（レース後・完走者ペアのみ） ---
        finished = [(p, f) for p, f in zip(pids, fins) if f is not None and 1 <= f <= 99]
        for i in range(len(finished)):
            for j in range(i + 1, len(finished)):
                pa, fa = finished[i]
                pb, fb = finished[j]
                if fa == fb:
                    continue
                key = (pa, pb) if pa < pb else (pb, pa)
                h2h_n[key] += 1
                a_first = fa < fb
                a_is_min = pa < pb
                if a_first == a_is_min:
                    h2h_win[key] += 1

    key_series = list(zip(df["race_key"], df["player_id"]))
    vals = [out.get(k, (np.nan, 0.0, 0.0)) for k in key_series]
    df["h2h_win_rate"] = [v[0] for v in vals]
    df["h2h_n_total"] = [v[1] for v in vals]
    df["h2h_net_norm"] = [v[2] for v in vals]
    ne = df.groupby("race_key")["player_id"].transform("count").replace(0, np.nan)
    df["h2h_net_norm"] = (df["h2h_net_norm"] / ne).fillna(0.0)
    df["h2h_win_rate"] = df["h2h_win_rate"].fillna(0.5)
    df["h2h_n_total"] = df["h2h_n_total"].fillna(0.0)
    return df


# 【2026-07-31・ex_spurt_pct/ex_thrust_pct をFEATURE_COLS_WTから除外（48→46特徴）】
# 同一開催・同一選手のDay1時点と最終日時点の値をn=221,551ペアで比較した結果、
# `ex_spurt_pct`（捲り実行率）は5.36%のペアで値が変化（変化時平均+8.2pt）、
# `ex_thrust_pct`（差し実行率）は2.39%のペアで変化（変化時平均+22.9pt）しており、
# いずれも**開催期間中に値が更新される**ことが確定した。`_get_collected_keys`
# （`src/scraper/pipeline_wt.py:168-183`）は`finish_order>=1`（結果確定済み）の
# 行のみをスキップ対象とするため、未確定レースは結果が付くまで再収集され続け、
# 学習データには「そのレース自身の発走後の結果を反映した値」が混入しうる
# （sb_dyn バグ・commit c3dd62e と同型のtrain/serve skew）。
#
# `scripts/exp_ab_leaky_ex_features.py` で12ヶ月・約194,000サンプルのA/B測定を
# 実施した結果、eval(3着内) AUC は 0.7732(48特徴) vs 0.7731(46特徴)、
# win(1着) AUC は 0.8233 vs 0.8233 と**差は事実上ゼロ**（honest ROIも有意差なし）。
# 除外の理由は「性能が上がるから」ではなく、**予測に何も貢献していないのに
# train/serve skewというリスクだけを抱えているため**。
#
# 元のSELECT（load_raw_data_wt）・正規化処理（fillna(0.0)/100.0, build_features_wt
# 197-199行付近）は残置。DBからの読み出し自体は分析用途・将来のpoint-in-time化
# （朝スナップショット等）で使う可能性があるため保持するが、FEATURE_COLS_WTから
# 外れているため学習・推論には一切使われない。
#
# 参考: `ex_left_behind_pct`（21.4%）/ `ex_split_line_pct`（24.9%）/
# `ex_snatch_pct`（1.5%）も同様に開催中に更新される実測があるが、これらは
# 元々FEATURE_COLS_WTに含まれていない（load_raw_data_wtでSELECTのみ）ため
# 変更不要。ただし将来アドホック実験で誤って採用しないよう明記しておく。
FEATURE_COLS_WT = [
    # コア得点
    "race_point",
    "gear_ratio",
    "first_rate_norm",
    "third_rate_norm",
    # エンコード
    "style_enc",
    "player_class_enc",
    "frame_no",
    # レース内相対
    "score_rank",
    "score_z",
    "wr_rank",
    "top3r_rank",
    # 枠
    "is_inner",
    "is_outer",
    # 場・グレード
    "bank_length_enc",
    "is_indoor",
    "grade_enc",
    # 選手属性
    "period_norm",
    "is_home",
    # ライン（winticket 固有）
    "line_size",
    "line_pos",
    "is_line_leader",
    "n_lines",
    "is_isolated",
    "line_frac",
    "n_senko",          # 展開: レース内の逃げ(先行)人数（n_linesと独立の波乱シグナル）
    # セクター回数
    "s_count",
    "h_count",
    "b_count",
    # 上がり戦術率: ex_spurt_pct（捲り実行率）/ ex_thrust_pct（差し実行率）は
    # 2026-07-31にFEATURE_COLS_WTから除外（train/serve skew実測・下記参照）。
    # winticket AI 印（市場人気の代理変数）
    "prediction_mark",
    # ks流ローリング特徴（point-in-time・add_rolling_features_wt で付与）
    "win_3m", "top3_3m", "quin_3m", "win_6m", "top3_6m", "quin_6m",
    "venue_wr", "days_since", "wr_trend",
    # 競走得点トレンド（2026-07-16追加・選手の成長/好不調）
    *RP_TREND_COLS_WT,
    # レース単位S/B・上がりローリング（2026-07-18追加・展開/脚力の直接測定）
    *SB_DYN_COLS_WT,
    # 隊列推定位置（2026-08-03追加・後方は外を回る分だけ実走距離が伸びる）
    *FORMATION_COLS_WT,
    # レース種別（2026-08-04追加・勝ち上がりで実力者が揃うレースの過大評価を是正）
    *RACE_TYPE_COLS_WT,
    # ライン実力（2026-08-04追加・競輪はライン戦なのに構造しか持っていなかった）
    *LINE_STRENGTH_COLS_WT,
    # 頭対頭対戦成績(H2H)は2026-07-28に検証→S1/S9悪化のため撤回・非採用（add_h2h_features_wt参照）
    # レース単位集約(rp_mean/rp_std/rp_gap_top2/rp_gap_top_self)は2026-08-04に検証→
    # AUCは上がるが1位3着内率が窓で符号反転（−0.20pt/+0.07pt）のため非採用
    # （add_line_strength_features_wt の docstring 参照）
]

TARGET_COL_WT = "top3_flag"
WIN_TARGET_COL_WT = "win_flag"
BAD_TARGET_COL_WT = "bad6_flag"
TOP2_TARGET_COL_WT = "top2_flag"


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """推論用の特徴行列を統一生成する（M-1: train/serve/eval/backtest で同一表現）。

    FEATURE_COLS_WT の列順を固定し、NaN は 0 で補完する。
    build_features_wt 末尾で既に保証 fill 済みのため通常は no-op だが、
    全推論経路がこの関数を通ることで「dropna vs fillna」の不整合を構造的に排除する。
    """
    return df.reindex(columns=FEATURE_COLS_WT).fillna(0)


# ---------------------------------------------------------------------------
# 特徴量キャッシュ（2026-08-05 新設）
#
# 【なぜ必要か】`build_features_wt(load_raw_data_wt(...))` の実測内訳（6ヶ月分）:
#     load_raw_data_wt (SQL+ネットワーク)  54.5秒  19%
#     build_features_wt (特徴量計算)      227.6秒  81%
# **ボトルネックはネットワークではなく Mac 上の特徴量計算**。したがってローカルDB
# ミラーを作っても 19% しか縮まない（当初その案を検討したが測定して棄却した）。
# 効くのは特徴量そのもののキャッシュ。exp/検証スクリプト群は全て同じ期間を読むため、
# 1日に何本も回すと読み込みだけで累計1時間規模を消費していた。
#
# 【⚠️ なぜ期間ごとに持つのか — 切り出しは安全でない】
# 広い期間で1回作って日付で切り出す方式は**使えない**ことを実測で確認した
# （2022-12-01〜2024-06-30 を直接構築 vs 〜2024-12-31 から切り出しで比較）:
#     60列中7列が不一致 → race_point, score_rank, score_z,
#                          line_rp_sum, line_rp_max, line_rp_mean, line_rp_gap_top
# 原因は得点補完 `med_rp = df["race_point"].median()`（本モジュール上部）が
# **読み込み範囲全体の中央値**を使っていること。副作用として、これは軽微な
# look-ahead でもある（2024年のレースの欠損得点が2026年を含む中央値で埋まる）。
#
# ただし実害は無い: 補完対象は全行の **0.34%**（714,441行中2,452行）で、
# 中央値の振れも 85.66→85.55（**0.11点・約0.13%**）。修正すれば切り出しが安全になり
# キャッシュ効率は上がるが、0.34%の行の値が変わる＝既存の eval/win vintage 64本と
# 整合しなくなり96本の再学習が要る。**割に合わないので直さない**（2026-08-05判断）。
#
# 【鮮度検証は必須・スキップ経路を作らない】
# キャッシュキーに VPS 側の (MAX(race_date), COUNT(*)) を含める。この問い合わせは
# ミリ秒で終わる。黙って古い特徴量で学習する経路は作らない——廃止されたローカル
# SQLite が起こした事故（picks_history 消失・2026-07-20）と同じ形になるため。
# ---------------------------------------------------------------------------

FEATURE_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "feature_cache"
_FEATURE_CACHE_ENV = "KEIRIN_FEATURE_CACHE"


def _wt_data_fingerprint(min_date: str, max_date: str | None) -> str:
    """対象期間のデータ指紋 (MAX(race_date), COUNT(*)) を VPS から取る（ミリ秒）。"""
    from src.database import get_connection

    where = "WHERE r.race_date >= ?"
    params: list = [min_date]
    if max_date:
        where += " AND r.race_date <= ?"
        params.append(max_date)
    with get_connection() as conn:
        row = list(conn.execute(
            f"SELECT COUNT(*), MAX(r.race_date) FROM wt_entries e "
            f"JOIN wt_races r ON e.race_key = r.race_key {where}", params))[0]
    n, mx = row[0], row[1]
    return f"{n}_{str(mx).replace('-', '')}"


def load_features_wt(min_date: str, max_date: str | None = None, *,
                     use_cache: bool | None = None) -> pd.DataFrame:
    """`build_features_wt(load_raw_data_wt(...))` のキャッシュ付きラッパー。

    use_cache: None のとき環境変数 KEIRIN_FEATURE_CACHE=1 で有効。
      True/False で明示指定もできる。

    ⚠️ **キャッシュは期間ごと**。切り出して使い回してはいけない（上のコメント参照）。
    ⚠️ 毎回 VPS へ指紋クエリを投げ、データが増えていれば自動で作り直す。
    """
    if use_cache is None:
        use_cache = os.environ.get(_FEATURE_CACHE_ENV) == "1"
    if not use_cache:
        return build_features_wt(load_raw_data_wt(min_date=min_date, max_date=max_date))

    fp = _wt_data_fingerprint(min_date, max_date)
    tag = f"{min_date}_{max_date or 'latest'}".replace("-", "")
    path = FEATURE_CACHE_DIR / f"wtfeat_{tag}_f{len(FEATURE_COLS_WT)}_{fp}.pkl"
    if path.exists():
        print(f"  [feature-cache] {path.name} を利用（再計算なし）", flush=True)
        return pd.read_pickle(path)

    df = build_features_wt(load_raw_data_wt(min_date=min_date, max_date=max_date))
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # ⚠️ 保存失敗を握り潰してはいけない。キャッシュが書けないまま有効化されていると
    # 「毎回フル計算 + 指紋クエリ」で**素の実行より遅くなる**（2026-08-05 に実際に踏んだ:
    # pyarrow 未導入で parquet が書けず、握り潰した結果 54% 遅くなっていた）。
    # 黙って効かないキャッシュは、遅いだけでなく「効いているつもり」を作るので有害。
    # pickle にしているのは data/exp_cache/ の既存キャッシュと同方式で依存を増やさないため。
    df.to_pickle(path)
    print(f"  [feature-cache] {path.name} を保存", flush=True)
    # 同一期間の古い指紋のキャッシュはデータが増えた時点で不要になるため削除する
    for old in FEATURE_CACHE_DIR.glob(f"wtfeat_{tag}_f{len(FEATURE_COLS_WT)}_*.pkl"):
        if old != path:
            old.unlink(missing_ok=True)
    return df
