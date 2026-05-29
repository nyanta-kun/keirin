"""
特徴量エンジニアリング

DBから学習用データセットを構築する。
1レース1選手を1行として、特徴量とターゲット（top3フラグ）を生成する。
"""
import pandas as pd
import numpy as np
from ..database import get_connection


LINE_POSITION_MAP = {
    "先行": 0,
    "捲り": 1,
    "差し": 2,
    "追い込み": 3,
    None: -1,
}


def load_raw_data(min_date: str = "2025-01-01", max_date: str = None) -> pd.DataFrame:
    """DBからレース×選手の生データを取得"""
    where = "WHERE r.race_date >= :min_date"
    params: dict = {"min_date": min_date}
    if max_date:
        where += " AND r.race_date <= :max_date"
        params["max_date"] = max_date

    query = f"""
        SELECT
            e.race_key,
            r.race_date,
            r.venue_code,
            r.grade,
            r.distance,
            e.frame_no,
            e.player_id,
            e.gear_ratio,
            e.racing_score,
            e.recent_win_rate_3m,
            e.recent_top3_rate_3m,
            e.recent_win_rate_6m,
            e.recent_top3_rate_6m,
            e.venue_win_rate,
            e.days_since_last_race,
            e.line_position,
            e.quinella_rate,
            e.period,
            e.prefecture      AS player_prefecture,
            e.player_class,
            vi.bank_length,
            vi.is_indoor,
            vi.prefecture     AS venue_prefecture,
            res.finish_position
        FROM race_entries e
        JOIN races r ON e.race_key = r.race_key
        LEFT JOIN race_results res
            ON e.race_key = res.race_key AND e.frame_no = res.frame_no
        LEFT JOIN venue_info vi ON r.venue_code = vi.venue_code
        {where}
        ORDER BY r.race_date, e.race_key, e.frame_no
    """

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を構築して学習用DataFrameを返す"""
    df = df.copy()

    # ターゲット: 3着以内（finish_position が NaN の行は 0 として扱う）
    df["top3_flag"] = (df["finish_position"].notna() & (df["finish_position"] <= 3)).astype(int)

    # 欠損補完
    df["racing_score"] = df["racing_score"].fillna(df["racing_score"].median())
    df["gear_ratio"] = df["gear_ratio"].fillna(3.92)
    df["recent_win_rate_3m"] = df["recent_win_rate_3m"].fillna(0.0)
    df["recent_top3_rate_3m"] = df["recent_top3_rate_3m"].fillna(0.0)

    # 6ヶ月統計（compute-stats 実行済み かつ列が存在する場合のみ有効）
    if "recent_win_rate_6m" in df.columns and df["recent_win_rate_6m"].notna().mean() > 0.1:
        df["recent_win_rate_6m"] = df["recent_win_rate_6m"].fillna(df["recent_win_rate_3m"])
        df["recent_top3_rate_6m"] = df["recent_top3_rate_6m"].fillna(df["recent_top3_rate_3m"])
        df["wr_trend"] = df["recent_win_rate_3m"] - df["recent_win_rate_6m"]
    else:
        df["recent_win_rate_6m"] = df["recent_win_rate_3m"]
        df["recent_top3_rate_6m"] = df["recent_top3_rate_3m"]
        df["wr_trend"] = 0.0

    # 場別勝率（NULLは全体平均で補完）
    if "venue_win_rate" in df.columns and df["venue_win_rate"].notna().any():
        df["venue_win_rate"] = df["venue_win_rate"].fillna(df["recent_win_rate_3m"])
    else:
        df["venue_win_rate"] = df["recent_win_rate_3m"]

    # 前走からの経過日数（NULLは中央値で補完）
    if "days_since_last_race" in df.columns and df["days_since_last_race"].notna().any():
        median_days = df["days_since_last_race"].median()
        df["days_since_last_race"] = df["days_since_last_race"].fillna(median_days)
    else:
        df["days_since_last_race"] = 14.0  # デフォルト2週間

    # 脚質エンコード
    df["line_pos_enc"] = df["line_position"].map(LINE_POSITION_MAP).fillna(-1).astype(int)

    # レース内相対特徴量（スコアの相対順位・偏差）
    race_grp = df.groupby("race_key")["racing_score"]
    df["score_rank"] = race_grp.rank(ascending=False)          # 1=最高得点
    df["score_mean"] = race_grp.transform("mean")
    df["score_std"]  = race_grp.transform("std").fillna(1.0).replace(0.0, 1.0)
    df["score_z"]    = ((df["racing_score"] - df["score_mean"]) / df["score_std"]).clip(-5, 5)

    race_grp_wr = df.groupby("race_key")["recent_win_rate_3m"]
    df["wr_rank"] = race_grp_wr.rank(ascending=False)
    df["wr_mean"] = race_grp_wr.transform("mean")

    race_grp_top3 = df.groupby("race_key")["recent_top3_rate_3m"]
    df["top3r_rank"] = race_grp_top3.rank(ascending=False)

    # 枠番特徴量（内枠/外枠）
    df["is_inner"] = (df["frame_no"] <= 3).astype(int)
    df["is_outer"] = (df["frame_no"] >= 7).astype(int)

    # グレードエンコード（GP>G1>G2>G3>F1>F2>A級）
    grade_map = {"GP": 7, "G1": 6, "G2": 5, "G3": 4, "F1": 3, "F2": 2}
    df["grade_enc"] = df["grade"].map(grade_map).fillna(1).astype(int)

    # --- 新規特徴量（venue_info JOIN 済みの場合のみ有効） ---

    # ホーム判定（選手登録府県 == 開催場府県）
    if "player_prefecture" in df.columns and "venue_prefecture" in df.columns:
        df["is_home"] = (
            df["player_prefecture"].notna()
            & df["venue_prefecture"].notna()
            & (df["player_prefecture"] == df["venue_prefecture"])
        ).astype(int)
    else:
        df["is_home"] = 0

    # バンク長（正規化: 250→2.5, 333→3.33, 400→4.0, 500→5.0）
    if "bank_length" in df.columns:
        df["bank_length_enc"] = df["bank_length"].fillna(400) / 100.0
        df["is_indoor"] = df["is_indoor"].fillna(0).astype(int)
    else:
        df["bank_length_enc"] = 4.0
        df["is_indoor"] = 0

    # 期別正規化（数値が小さい＝古い期=ベテラン）
    if "period" in df.columns:
        df["period_norm"] = df["period"].fillna(df["period"].median()) / 100.0
    else:
        df["period_norm"] = 0.0

    # 2連対率（NULLは直近3ヶ月勝率×2で代替）
    if "quinella_rate" in df.columns:
        proxy = df["recent_win_rate_3m"] * 2
        df["quinella_rate"] = df["quinella_rate"].fillna(proxy)
    else:
        df["quinella_rate"] = df["recent_win_rate_3m"] * 2

    # 登録クラスエンコード
    _class_map = {"SS": 6, "S1": 5, "S2": 4, "A1": 3, "A2": 2, "A3": 1, "B": 0}
    if "player_class" in df.columns:
        df["player_class_enc"] = df["player_class"].map(_class_map).fillna(-1).astype(int)
    else:
        df["player_class_enc"] = -1

    # 同ライン内の先行選手の最高得点（ライングループなし選手は自身の得点）
    if "line_group" in df.columns and df["line_group"].notna().any():
        leader_scores = (
            df[df["line_position"] == "先行"]
            .groupby(["race_key", "line_group"])["racing_score"]
            .max()
            .rename("_leader_score")
        )
        df = df.join(leader_scores, on=["race_key", "line_group"])
        df["line_leader_score"] = df["_leader_score"].fillna(df["racing_score"])
        df.drop(columns=["_leader_score"], inplace=True)
    else:
        df["line_leader_score"] = df["racing_score"]

    return df


# v1: 既存モデル互換の13特徴量
FEATURE_COLS_V1 = [
    "racing_score",
    "gear_ratio",
    "recent_win_rate_3m",
    "recent_top3_rate_3m",
    "line_pos_enc",
    "frame_no",
    "score_rank",
    "score_z",
    "wr_rank",
    "top3r_rank",
    "is_inner",
    "is_outer",
    "grade_enc",
]

# v1.5: rolling stats + venue_info を追加した20特徴量
# compute-stats 実行済みかつ venue_info 登録済みで使用可能
FEATURE_COLS_V15 = FEATURE_COLS_V1 + [
    "recent_win_rate_6m",
    "recent_top3_rate_6m",
    "wr_trend",
    "venue_win_rate",
    "days_since_last_race",
    "bank_length_enc",
    "is_indoor",
]

# v2: スクレイピング取得フィールドを追加した25特徴量
# line_leader_score は line_group データ取得時に有効化
FEATURE_COLS_V2 = FEATURE_COLS_V15 + [
    "quinella_rate",
    "period_norm",
    "player_class_enc",
    "is_home",
    "line_leader_score",
]

# 現在有効な特徴量セット（v2実用版: line_group未取得のため line_leader_score を除外した24特徴量）
FEATURE_COLS = FEATURE_COLS_V15 + [
    "quinella_rate",
    "period_norm",
    "player_class_enc",
    "is_home",
]

TARGET_COL = "top3_flag"
