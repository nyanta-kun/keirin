"""
winticket 特徴量エンジニアリング

wt_entries + wt_races + venue_info から学習用データセットを構築する。
"""
import os
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

    # 上がり戦術率（%→0-1）
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

    # M-1: 学習(train_lgbm dropna)・推論(prepare_X fillna)・バックテストで
    # 同一の特徴表現になるよう、ソースで FEATURE_COLS_WT の NaN を 0 に統一保証する
    # （現状 build 過程で各特徴は補完済＝実質no-op だが、将来の fill 漏れによる
    #  train/serve skew を構造的に防ぐ安全網）。
    present = [c for c in FEATURE_COLS_WT if c in df.columns]
    df[present] = df[present].fillna(0)

    return df


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

    # DNS/DNF（finish_order<1・欠車や途中棄権）は res_back/standing/final_half が
    # 完走者と同じ意味を持たない（完走できず途中終了しただけの record）ため、
    # 履歴からもレース内中央値/最速判定からも除外する（元検証 exp_sb_dyn_ab.py の
    # SQL WHERE finish_order>=1 と同等。この除外漏れが2026-07-18に本番投入直後の
    # A/B効果ΔAUC+0.013をΔAUC+0.0006へ縮小させていたことが判明・修正）。
    if "finish_order" in H.columns:
        H = H[H["finish_order"] >= 1].copy()

    H["_dt"] = pd.to_datetime(H["race_date"])
    # レース内相対化: fh_rel = 自上がり − レース中央値（負=速い）・fh_best = レース内最速。
    # final_half<=0 や欠損は NaN（rolling から除外）。
    fh = pd.to_numeric(H["final_half"], errors="coerce")
    H["_fh"] = fh.where(fh > 0)
    med = H.groupby("race_key")["_fh"].transform("median")
    mn = H.groupby("race_key")["_fh"].transform("min")
    H["_fh_rel"] = H["_fh"] - med
    H["_fh_best"] = (H["_fh"] == mn).astype(float).where(H["_fh"].notna())
    H["_b"] = pd.to_numeric(H["res_back"], errors="coerce")
    H["_s"] = pd.to_numeric(H["res_standing"], errors="coerce")

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
    # 上がり戦術率
    "ex_spurt_pct",
    "ex_thrust_pct",
    # winticket AI 印（市場人気の代理変数）
    "prediction_mark",
    # ks流ローリング特徴（point-in-time・add_rolling_features_wt で付与）
    "win_3m", "top3_3m", "quin_3m", "win_6m", "top3_6m", "quin_6m",
    "venue_wr", "days_since", "wr_trend",
    # 競走得点トレンド（2026-07-16追加・選手の成長/好不調）
    *RP_TREND_COLS_WT,
    # レース単位S/B・上がりローリング（2026-07-18追加・展開/脚力の直接測定）
    *SB_DYN_COLS_WT,
]

TARGET_COL_WT = "top3_flag"
WIN_TARGET_COL_WT = "win_flag"


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """推論用の特徴行列を統一生成する（M-1: train/serve/eval/backtest で同一表現）。

    FEATURE_COLS_WT の列順を固定し、NaN は 0 で補完する。
    build_features_wt 末尾で既に保証 fill 済みのため通常は no-op だが、
    全推論経路がこの関数を通ることで「dropna vs fillna」の不整合を構造的に排除する。
    """
    return df.reindex(columns=FEATURE_COLS_WT).fillna(0)
