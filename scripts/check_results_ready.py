#!/usr/bin/env python3
"""
Aランク対象レースの全結果が確定していれば exit 0、未確定があれば exit 1。
daily_picks.sh の結果待ちループから呼び出す。
"""
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
from src.models.trainer import load_model


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m-%d")

    df_raw = load_raw_data(min_date=target_date, max_date=target_date)
    if df_raw.empty:
        print(f"[check] {target_date} DBにデータなし", file=sys.stderr)
        sys.exit(1)

    model = load_model("lgbm")
    df = build_features(df_raw)
    df = df.dropna(subset=FEATURE_COLS)
    df["pred_prob"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

    # Aランク対象レースを抽出
    a_keys = []
    for race_key, grp in df.groupby("race_key"):
        grp = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n = len(grp)
        top1 = grp["pred_prob"].iloc[0]
        top2 = grp["pred_prob"].iloc[1] if n > 1 else 0.0
        gap12 = top1 - top2
        if 0.60 <= top1 < 0.70 and gap12 > 0.12:
            a_keys.append(race_key)

    if not a_keys:
        print(f"[check] {target_date} Aランク対象なし、確定とみなす")
        sys.exit(0)

    # 全レースの結果が揃っているか確認
    pending = []
    for race_key in a_keys:
        grp = df[df["race_key"] == race_key]
        top3_set = frozenset(grp[grp["finish_position"] <= 3]["frame_no"].tolist())
        if len(top3_set) < 3:
            pending.append(race_key)

    if pending:
        print(f"[check] 未確定 {len(pending)}件: {', '.join(pending)}", file=sys.stderr)
        sys.exit(1)

    print(f"[check] 全{len(a_keys)}件 結果確定")
    sys.exit(0)


if __name__ == "__main__":
    main()
