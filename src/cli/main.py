"""
競輪AI予想システム CLI
"""
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database import init_db
from src.scraper.pipeline import CollectionPipeline, setup_logging


@click.group()
@click.option("--debug", is_flag=True, help="デバッグログを表示")
def cli(debug: bool):
    """競輪AI予想システム"""
    setup_logging("DEBUG" if debug else "INFO")


@cli.command()
@click.option("--date", "target_date", default=None, help="収集日 (YYYY-MM-DD), 省略時は昨日")
@click.option("--dry-run", is_flag=True, help="DBに保存しない（動作確認用）")
def collect(target_date: str | None, dry_run: bool):
    """指定日のレースデータを収集してDBに保存"""
    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    click.echo(f"Collecting data for {target_date} {'(dry-run)' if dry_run else ''}")

    init_db()
    pipeline = CollectionPipeline()
    stats = pipeline.collect_date(target_date, dry_run=dry_run)

    click.echo(f"Complete: venues={stats['venues']}, races={stats['races']}, "
               f"results={stats['results']}, errors={stats['errors']}")


@cli.command()
@click.option("--year", required=True, type=int, help="収集年 (例: 2025)")
@click.option("--month", required=True, type=int, help="収集月 (例: 11)")
@click.option("--dry-run", is_flag=True, help="DBに保存しない（動作確認用）")
def collect_month(year: int, month: int, dry_run: bool):
    """指定年月のレースデータを一括収集"""
    click.echo(f"Collecting data for {year}/{month:02d} {'(dry-run)' if dry_run else ''}")

    init_db()
    pipeline = CollectionPipeline()
    stats = pipeline.collect_month(year, month, dry_run=dry_run)

    click.echo(f"Complete: venues={stats['venues']}, races={stats['races']}, "
               f"results={stats['results']}, errors={stats['errors']}")


@cli.command()
@click.option("--from", "from_ym", required=True, help="開始年月 (YYYY-MM)")
@click.option("--to", "to_ym", default=None, help="終了年月 (YYYY-MM), 省略時は今月")
@click.option("--dry-run", is_flag=True, help="DBに保存しない（動作確認用）")
def collect_range(from_ym: str, to_ym: str | None, dry_run: bool):
    """指定期間（年月範囲）のレースデータを一括収集

    例: python src/cli/main.py collect-range --from 2025-02
        python src/cli/main.py collect-range --from 2025-02 --to 2025-12
    """
    from calendar import monthrange

    try:
        start_year, start_month = map(int, from_ym.split("-"))
    except ValueError:
        click.echo("Error: --from は YYYY-MM 形式で指定してください（例: 2025-02）", err=True)
        raise SystemExit(1)

    if to_ym is None:
        today = date.today()
        end_year, end_month = today.year, today.month
    else:
        try:
            end_year, end_month = map(int, to_ym.split("-"))
        except ValueError:
            click.echo("Error: --to は YYYY-MM 形式で指定してください（例: 2025-12）", err=True)
            raise SystemExit(1)

    # 月リストを生成
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    click.echo(f"Collecting {len(months)} months: {from_ym} ~ {end_year}-{end_month:02d} "
               f"{'(dry-run)' if dry_run else ''}")

    init_db()
    pipeline = CollectionPipeline()
    total = {"venues": 0, "races": 0, "results": 0, "errors": 0}

    for i, (year, month) in enumerate(months, 1):
        click.echo(f"\n[{i}/{len(months)}] {year}/{month:02d}")
        stats = pipeline.collect_month(year, month, dry_run=dry_run)
        for k in total:
            total[k] += stats.get(k, 0)
        click.echo(f"  -> venues={stats['venues']}, races={stats['races']}, "
                   f"results={stats['results']}, errors={stats['errors']}")

    click.echo(f"\nAll done: venues={total['venues']}, races={total['races']}, "
               f"results={total['results']}, errors={total['errors']}")


@cli.command("collect-reverse")
@click.option("--from", "from_ym", required=True, help="開始年月 (YYYY-MM) ※古い方")
@click.option("--to", "to_ym", default=None, help="終了年月 (YYYY-MM) ※新しい方、省略時は今月")
@click.option("--dry-run", is_flag=True)
def collect_reverse(from_ym: str, to_ym: str | None, dry_run: bool):
    """最新から過去に遡る順でデータ収集（最新データを優先的に取得）

    例: python -m src.cli.main collect-reverse --from 2024-01
    """
    from calendar import monthrange

    try:
        start_year, start_month = map(int, from_ym.split("-"))
    except ValueError:
        click.echo("Error: --from は YYYY-MM 形式で指定してください（例: 2024-01）", err=True)
        raise SystemExit(1)

    if to_ym is None:
        today = date.today()
        end_year, end_month = today.year, today.month
    else:
        try:
            end_year, end_month = map(int, to_ym.split("-"))
        except ValueError:
            click.echo("Error: --to は YYYY-MM 形式で指定してください（例: 2025-12）", err=True)
            raise SystemExit(1)

    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    months = list(reversed(months))

    click.echo(f"Collecting {len(months)} months (newest first): "
               f"{end_year}-{end_month:02d} ~ {from_ym} {'(dry-run)' if dry_run else ''}")

    init_db()
    pipeline = CollectionPipeline()
    total = {"venues": 0, "races": 0, "results": 0, "errors": 0}

    for i, (year, month) in enumerate(months, 1):
        click.echo(f"\n[{i}/{len(months)}] {year}/{month:02d}")
        stats = pipeline.collect_month(year, month, dry_run=dry_run)
        for k in total:
            total[k] += stats.get(k, 0)
        click.echo(f"  -> venues={stats['venues']}, races={stats['races']}, "
                   f"results={stats['results']}, errors={stats['errors']}")

    click.echo(f"\nAll done: venues={total['venues']}, races={total['races']}, "
               f"results={total['results']}, errors={total['errors']}")


@cli.command()
def init():
    """データベースを初期化"""
    init_db()
    click.echo("Database initialized.")


@cli.command()
def status():
    """DBの収集状況を確認"""
    from src.database import get_connection
    with get_connection() as conn:
        races = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        entries = conn.execute("SELECT COUNT(*) FROM race_entries").fetchone()[0]
        results = conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]
        odds = conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(race_date) FROM races"
        ).fetchone()[0]
        earliest = conn.execute(
            "SELECT MIN(race_date) FROM races"
        ).fetchone()[0]

    click.echo(f"Races:   {races:,}")
    click.echo(f"Entries: {entries:,}")
    click.echo(f"Results: {results:,}")
    click.echo(f"Odds:    {odds:,}")
    click.echo(f"Date range: {earliest or 'N/A'} ~ {latest or 'N/A'}")


@cli.command()
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="モデル種別")
@click.option("--from", "from_date", default="2025-01-01", help="学習データ開始日")
@click.option("--to", "to_date", default=None, help="学習データ終了日（省略=全て）")
@click.option("--test-from", "test_from", default=None,
              help="テスト開始日（指定時はこの日以降をテストに使用。未指定は後ろ20%%）")
@click.option("--save-as", "save_as", default=None,
              help="保存名（例: lgbm_v15）。省略時はモデル種別名で保存")
def train(model_type: str, from_date: str, to_date: str | None,
          test_from: str | None, save_as: str | None):
    """モデルを学習してdata/models/に保存"""
    from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
    from src.models.trainer import train_baseline, train_lgbm, save_model

    # --test-from 指定時は to_date を無視して全データ（学習+テスト分）を読み込む
    load_max = None if test_from else to_date
    click.echo(f"Loading data from {from_date} ~ {load_max or 'latest'} ...")
    click.echo(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
    df_raw = load_raw_data(min_date=from_date, max_date=load_max)
    df = build_features(df_raw)

    # 結果のあるデータのみ学習に使用
    df_train = df[df["finish_position"].notna()].copy()
    click.echo(f"Training samples: {len(df_train):,} entries / "
               f"{df_train['race_key'].nunique():,} races")

    if test_from:
        df_tr = df_train[df_train["race_date"] < test_from]
        df_te = df_train[df_train["race_date"] >= test_from]
        click.echo(f"Train: {df_tr['race_key'].nunique():,} races  "
                   f"Test: {df_te['race_key'].nunique():,} races  "
                   f"(split: {test_from})")
    else:
        dates = sorted(df_train["race_date"].unique())
        split_idx = int(len(dates) * 0.8)
        split_date = dates[split_idx]
        df_tr = df_train[df_train["race_date"] < split_date]
        df_te = df_train[df_train["race_date"] >= split_date]
        click.echo(f"Train: {df_tr['race_key'].nunique():,} races  "
                   f"Test: {df_te['race_key'].nunique():,} races  "
                   f"(split: {split_date})")

    if model_type == "baseline":
        click.echo("Training Logistic Regression baseline ...")
        model = train_baseline(df_tr)
        model_name = save_as or "baseline"
    else:
        click.echo("Training LightGBM ...")
        model = train_lgbm(df_tr)
        model_name = save_as or "lgbm"

    save_model(model, model_name)
    # lgbm.pkl も常に最新モデルで上書き（predict/weekly コマンドが参照）
    if model_name != "lgbm" and model_type == "lgbm":
        save_model(model, "lgbm")

    click.echo("\n=== 買い目戦略別バックテスト ===")
    from src.evaluation.backtest import run_backtest, print_backtest
    df_result = run_backtest(model, df_te)
    print_backtest(df_result, total_races=df_te["race_key"].nunique())


@cli.command()
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="使用するモデル")
@click.option("--from", "from_date", default="2025-01-01", help="評価開始日")
@click.option("--to", "to_date", default=None, help="評価終了日")
@click.option("--max-riders", "max_riders", default=None, type=int,
              help="出走頭数の上限（例: 6で6車立て以下のみ。実運用と同じ母集団）")
def backtest(model_type: str, from_date: str, to_date: str | None, max_riders: int | None):
    """買い目戦略ごとの的中率・回収率を比較"""
    from src.preprocessing.feature_engineer import load_raw_data, build_features
    from src.models.trainer import load_model
    from src.evaluation.backtest import run_backtest, print_backtest

    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo("モデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    click.echo(f"Loading data {from_date} ~ {to_date or 'latest'} ...")
    df_raw = load_raw_data(min_date=from_date, max_date=to_date)
    df = build_features(df_raw)
    df_eval = df[df["finish_position"].notna()].copy()
    riders_label = f"（{max_riders}車立て以下）" if max_riders else ""
    click.echo(f"Evaluating {df_eval['race_key'].nunique():,} races {riders_label}...")

    df_result = run_backtest(model, df_eval, max_riders=max_riders)
    n_races = df_eval["race_key"].nunique() if max_riders is None else (
        df_eval.groupby("race_key")["frame_no"].count()
        .pipe(lambda s: s[s <= max_riders]).count()
    )
    print_backtest(df_result, total_races=n_races)


@cli.command()
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="使用するモデル")
@click.option("--from", "from_date", default="2024-06-01", help="評価開始日")
@click.option("--to", "to_date", default=None, help="評価終了日")
@click.option("--thresholds", default="0.65,0.70,0.75,0.80,0.85,0.90", show_default=True,
              help="top1確率フィルター閾値（カンマ区切り）。この値を超えるレースを除外。全レースは常に含む")
def analyze(model_type: str, from_date: str, to_date: str | None, thresholds: str):
    """人気フィルター × 穴狙い戦略の回収率分析

    モデルが最も高い確率を割り当てた選手のtop1_probを閾値でフィルタリングし、
    人気偏重レースを除外したときの回収率変化を分析する。
    穴狙い戦略（#2・#3・#4を1着に想定した組み合わせ）も同時に評価する。

    例:
        python src/cli/main.py analyze
        python src/cli/main.py analyze --from 2025-06-01 --thresholds 0.35,0.28,0.22
    """
    from src.preprocessing.feature_engineer import load_raw_data, build_features
    from src.models.trainer import load_model
    from src.evaluation.backtest import run_threshold_analysis, print_threshold_analysis

    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo("モデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    click.echo(f"Loading data {from_date} ~ {to_date or 'latest'} ...")
    df_raw = load_raw_data(min_date=from_date, max_date=to_date)
    df = build_features(df_raw)
    df_eval = df[df["finish_position"].notna()].copy()
    click.echo(f"Evaluating {df_eval['race_key'].nunique():,} races ...")

    threshold_list: list[float | None] = [None]
    for t in thresholds.split(","):
        t = t.strip()
        if t:
            threshold_list.append(float(t))

    analysis = run_threshold_analysis(model, df_eval, thresholds=threshold_list)
    print_threshold_analysis(analysis)


@cli.command()
@click.option("--days", default=7, show_default=True, type=int, help="直近何日分")
@click.option("--from", "from_date", default=None, help="開始日 (YYYY-MM-DD)。省略時は--days前")
@click.option("--to", "to_date", default=None, help="終了日 (YYYY-MM-DD)。省略時は昨日")
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="使用するモデル")
@click.option("--max-top1", default=0.70, show_default=True, type=float,
              help="top1_prob上限フィルター")
@click.option("--venue-filter/--no-venue-filter", default=False, show_default=True,
              help="場×戦略フィルターを適用する（現在は空フィルター）")
def weekly(days: int, from_date: str | None, to_date: str | None,
           model_type: str, max_top1: float, venue_filter: bool):
    """日別・場別の的中・回収集計（直近N日）

    例: python src/cli/main.py weekly
        python src/cli/main.py weekly --from 2026-05-17 --to 2026-05-23
        python src/cli/main.py weekly --days 14
    """
    from datetime import date, timedelta
    from src.preprocessing.feature_engineer import load_raw_data, build_features
    from src.models.trainer import load_model
    from src.evaluation.backtest import (
        run_daily_venue_summary, print_daily_venue_summary, VENUE_STRATEGY_FILTER,
    )

    today = date.today()
    if to_date is None:
        end = today - timedelta(days=1)
        to_date = end.strftime("%Y-%m-%d")
    if from_date is None:
        start = date.fromisoformat(to_date) - timedelta(days=days - 1)
        from_date = start.strftime("%Y-%m-%d")

    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo("モデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    click.echo(f"Loading {from_date} ~ {to_date} ...")
    df_raw = load_raw_data(min_date=from_date, max_date=to_date)
    df = build_features(df_raw)
    df_eval = df[df["finish_position"].notna()].copy()

    if df_eval.empty:
        click.echo("結果データがありません。", err=True)
        raise SystemExit(1)

    vf = VENUE_STRATEGY_FILTER if venue_filter else None
    if venue_filter:
        click.echo(f"場フィルター適用中: {len(VENUE_STRATEGY_FILTER)}場")
    click.echo(f"Races with results: {df_eval['race_key'].nunique():,}")
    df_summary = run_daily_venue_summary(model, df_eval, max_top1_prob=max_top1,
                                         venue_filter=vf)
    print_daily_venue_summary(df_summary)


@cli.command("day-sim")
@click.option("--date", "target_date", required=True, help="対象日 (YYYY-MM-DD)")
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="使用するモデル")
@click.option("--max-top1", default=0.80, show_default=True, type=float,
              help="top1_prob上限。超えたレースはSKIP（穴<65%/通常<70%/安定<80%を自動ラベル）")
def day_sim(target_date: str, model_type: str, max_top1: float):
    """指定日の推奨戦略シミュレーション（購入判定・的中・回収を表示）

    例: python src/cli/main.py day-sim --date 2026-04-28
    """
    from src.preprocessing.feature_engineer import load_raw_data, build_features
    from src.models.trainer import load_model
    from src.evaluation.backtest import run_day_simulation, print_day_simulation

    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo("モデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    df_raw = load_raw_data(min_date=target_date, max_date=target_date)
    df = build_features(df_raw)
    df_eval = df[df["finish_position"].notna()].copy()

    if df_eval.empty:
        click.echo(f"{target_date} の結果データがありません。", err=True)
        raise SystemExit(1)

    df_races, df_summary = run_day_simulation(model, df_eval, max_top1_prob=max_top1)
    print_day_simulation(df_races, df_summary, target_date, max_top1)


@cli.command()
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="使用するモデル")
@click.option("--from", "from_date", default="2025-01-01", help="評価開始日")
@click.option("--to", "to_date", default=None, help="評価終了日")
@click.option("--max-top1", default=0.70, show_default=True, type=float,
              help="top1_prob上限フィルター")
@click.option("--min-races", default=50, show_default=True, type=int,
              help="表示する会場の最低レース数")
def venue(model_type: str, from_date: str, to_date: str | None,
          max_top1: float, min_races: int):
    """会場別の的中率・回収率を比較

    例: python src/cli/main.py venue
        python src/cli/main.py venue --min-races 30
    """
    from src.preprocessing.feature_engineer import load_raw_data, build_features
    from src.models.trainer import load_model
    from src.evaluation.backtest import run_venue_analysis, print_venue_analysis

    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo("モデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    click.echo(f"Loading data {from_date} ~ {to_date or 'latest'} ...")
    df_raw = load_raw_data(min_date=from_date, max_date=to_date)
    df = build_features(df_raw)
    df_eval = df[df["finish_position"].notna()].copy()
    click.echo(f"Analyzing {df_eval['race_key'].nunique():,} races across venues ...")

    df_venue = run_venue_analysis(model, df_eval, max_top1_prob=max_top1,
                                  min_races=min_races)
    print_venue_analysis(df_venue, max_top1_prob=max_top1)


@cli.command("upset-train")
@click.option("--from", "from_date", default="2024-06-01", show_default=True,
              help="学習開始日")
@click.option("--to", "to_date", default=None, help="学習終了日 (省略=全期間)")
@click.option("--threshold", default=2000, show_default=True, type=int,
              help="波乱閾値: 3連複払戻がこの値以上を波乱と定義(円)")
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="エントリーモデル")
@click.option("--save-as", "save_as", default="lgbm_upset", show_default=True,
              help="保存ファイル名（.pkl 拡張子なし）")
def upset_train(from_date: str, to_date: str | None, threshold: int,
                model_type: str, save_as: str):
    """波乱レース予測モデルを学習・保存

    エントリーモデルの予測確率分布とレース構造特徴量を組み合わせ、
    高配当（波乱）が見込めるレースを識別する二値分類器を学習する。

    例:
        python -m src.cli.main upset-train
        python -m src.cli.main upset-train --threshold 3000 --save-as lgbm_upset_3k
    """
    from src.preprocessing.feature_engineer import load_raw_data, build_features
    from src.models.trainer import load_model
    from src.evaluation.backtest import _apply_pred_prob
    from src.evaluation.upset_model import (
        build_race_features, add_upset_target,
        train_upset_model, save_upset_model,
        print_upset_feature_importance,
    )

    try:
        entry_model = load_model(model_type)
    except FileNotFoundError:
        click.echo("エントリーモデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    click.echo(f"Loading data {from_date} ~ {to_date or 'latest'} ...")
    df_raw = load_raw_data(min_date=from_date, max_date=to_date)
    df = build_features(df_raw)

    click.echo("Applying entry model predictions ...")
    df_prob = _apply_pred_prob(entry_model, df)

    click.echo("Building race-level features ...")
    df_race = build_race_features(df_prob)
    df_race = add_upset_target(df_race, upset_threshold=threshold)

    n_with_result = df_race["is_upset"].notna().sum()
    click.echo(f"払戻データあり: {n_with_result:,} レース (波乱閾値: {threshold:,}円)")

    click.echo("Training upset model ...")
    upset_model = train_upset_model(df_race)

    print_upset_feature_importance(upset_model)
    save_upset_model(upset_model, name=save_as)


@cli.command("upset-backtest")
@click.option("--from", "from_date", default="2026-03-01", show_default=True,
              help="バックテスト開始日")
@click.option("--to", "to_date", default=None, help="バックテスト終了日")
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="エントリーモデル")
@click.option("--upset-model", "upset_model_name", default="lgbm_upset", show_default=True,
              help="波乱モデルファイル名（.pkl なし）")
@click.option("--strategies", "strategy_names", default="quinella_23,exacta_21,wide_23,box_top3",
              show_default=True, help="カンマ区切りの戦略名")
def upset_backtest(from_date: str, to_date: str | None, model_type: str,
                   upset_model_name: str, strategy_names: str):
    """波乱フィルター×戦略バックテスト

    波乱モデルの予測確率閾値を変えながら、各戦略の的中率・回収率を比較する。

    例:
        python -m src.cli.main upset-backtest
        python -m src.cli.main upset-backtest --from 2026-01-01 --strategies quinella_23,exacta_21
    """
    from src.preprocessing.feature_engineer import load_raw_data, build_features
    from src.models.trainer import load_model
    from src.evaluation.upset_model import (
        load_upset_model, run_upset_threshold_analysis, print_upset_analysis,
    )

    try:
        entry_model = load_model(model_type)
    except FileNotFoundError:
        click.echo("エントリーモデルが見つかりません。", err=True)
        raise SystemExit(1)

    try:
        upset_model = load_upset_model(upset_model_name)
    except FileNotFoundError:
        click.echo(f"波乱モデル '{upset_model_name}' が見つかりません。"
                   " upset-train コマンドを先に実行してください。", err=True)
        raise SystemExit(1)

    click.echo(f"Loading data {from_date} ~ {to_date or 'latest'} ...")
    df_raw = load_raw_data(min_date=from_date, max_date=to_date)
    df = build_features(df_raw)
    df_eval = df[df["finish_position"].notna()].copy()
    click.echo(f"Backtesting {df_eval['race_key'].nunique():,} races ...")

    snames = [s.strip() for s in strategy_names.split(",") if s.strip()]
    results = run_upset_threshold_analysis(entry_model, upset_model, df_eval,
                                           strategy_names=snames)
    print_upset_analysis(results, strategy_names=snames)


@cli.command()
@click.option("--race-key", required=True, help="レースキー (例: 20250401_21_01)")
@click.option("--model", "model_type", default="lgbm",
              type=click.Choice(["baseline", "lgbm"]), help="使用するモデル")
@click.option("--top", default=10, help="上位N点を表示")
def predict(race_key: str, model_type: str, top: int):
    """指定レースの3連複・3連単予想を表示"""
    from src.models.trainer import load_model
    from src.prediction.predictor import predict_race, format_prediction

    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo(f"モデルが見つかりません。先に `train` コマンドを実行してください。", err=True)
        raise SystemExit(1)

    pred = predict_race(model, race_key, top_n=top)
    if pred is None:
        click.echo(f"レース {race_key} のデータがDBに存在しません。", err=True)
        raise SystemExit(1)

    click.echo(format_prediction(pred))


@cli.command("compute-stats")
@click.option("--force", is_flag=True, help="既存値を上書きして全エントリを再計算")
@click.option("--dry-run", is_flag=True, help="DBを更新しない（件数確認のみ）")
def compute_stats(force: bool, dry_run: bool):
    """race_results から rolling 統計（6ヶ月勝率・前走日数・場別勝率）を計算してDBに書き込む

    データ収集完了後や新規収集後に実行する。
    例:
        python -m src.cli.main compute-stats
        python -m src.cli.main compute-stats --force   # 全エントリ再計算
    """
    from src.preprocessing.rolling_stats import compute_rolling_stats, recompute_rolling_stats

    if force:
        click.echo("Re-computing rolling stats for ALL entries ...")
        result = recompute_rolling_stats(dry_run=dry_run)
    else:
        click.echo("Computing rolling stats for entries without data ...")
        result = compute_rolling_stats(dry_run=dry_run)

    click.echo(f"Done: updated={result['updated']:,}, with_data={result['with_data']:,}"
               + (" [dry-run]" if dry_run else ""))


@cli.command("wave-picks")
@click.option("--date", "target_date", default=None, help="対象日 YYYY-MM-DD（省略時: 今日）")
@click.option("--output", "output_path", default=None,
              help="出力先ファイルパス（省略時: data/picks/wave_picks_{date}.txt）")
@click.option("--model", "model_type", default="lgbm", type=click.Choice(["lgbm"]))
def wave_picks(target_date, output_path, model_type):
    """6車立て以下レースを3段階ランクで予想出力

    ランク定義（ホールドアウト 2025-06〜2026-02、lgbm_v6）:
      SS : gap12≥0.15 & ratio<1.3          →  3連単 1→2→{3,4,5}着 3点  ROI 3315%
      S  : gap12≥0.15 & ratio [1.3, 1.6)   →  3連複 2軸×3頭流し   3点  ROI 177%
      A  : gap12 [0.06, 0.15)              →  3連複 2軸×3頭流し   3点  ROI 215%
      skip: gap12 < 0.06 or (S条件 & ratio≥1.6)  →  対象外

    ratio = top1_prob / (3/n_riders)  ← AIの1位確率を期待値で正規化
    SS条件: 接戦(ratio<1.3)かつAIが1-2着を明確に区別(gap12≥0.15) → 市場の盲点を突く高配当
    S上限(ratio<1.6): 3連複の市場人気が過集中するレースを除外 → 配当品質向上
    """
    from datetime import datetime
    import json
    import pandas as pd
    from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
    from src.models.trainer import load_model
    from src.database import get_connection
    from pathlib import Path

    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    try:
        with get_connection() as conn:
            vi = pd.read_sql("SELECT venue_code, name FROM venue_info", conn)
            st = pd.read_sql(
                "SELECT race_key, start_time FROM races WHERE race_date = ?",
                conn, params=[target_date]
            )
        venue_map = dict(zip(vi["venue_code"], vi["name"]))
        start_time_map = dict(zip(st["race_key"], st["start_time"]))
    except Exception:
        venue_map = {}
        start_time_map = {}

    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo("モデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    model_dir = Path(__file__).parent.parent.parent / "data" / "models"
    model_label = model_type
    for candidate in sorted(model_dir.glob(f"{model_type}_v*.pkl"), reverse=True):
        model_label = candidate.stem
        break

    click.echo(f"Loading data for {target_date} ...")
    df_raw = load_raw_data(min_date=target_date, max_date=target_date)
    if df_raw.empty:
        click.echo(f"{target_date} のデータがDBに存在しません。", err=True)
        raise SystemExit(1)

    df = build_features(df_raw)
    X = df[FEATURE_COLS].fillna(0)
    df["pred_prob"] = model.predict_proba(X)[:, 1]

    def parse_race_no(rk):
        parts = rk.split("_")
        return int(parts[2]) if len(parts) >= 3 else 0

    df["race_no"] = df["race_key"].apply(parse_race_no)

    ss_races, s_races, a_races = [], [], []

    for race_key, grp in df.groupby("race_key"):
        grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n_riders = len(grp_sorted)
        if n_riders > 6:
            continue

        p = grp_sorted["pred_prob"].tolist()
        top1 = p[0]
        top2_prob = p[1] if n_riders >= 2 else 0.0
        gap12 = top1 - top2_prob
        ratio = top1 / (3 / n_riders)

        if gap12 < 0.06:
            continue

        venue_code = grp_sorted["venue_code"].iloc[0]
        venue_name = venue_map.get(venue_code, str(venue_code))
        race_no = grp_sorted["race_no"].iloc[0]
        start_time = start_time_map.get(race_key) or "--:--"

        frames = grp_sorted["frame_no"].astype(int).tolist()
        pivot1, pivot2 = frames[0], frames[1]
        thirds = frames[2:5]
        thirds_str = ",".join(str(t) for t in thirds)

        riders_detail = []
        for rank_idx, row in enumerate(grp_sorted.itertuples(index=False)):
            fn = int(row.frame_no)
            if rank_idx == 0:
                role = "軸1"
            elif rank_idx == 1:
                role = "軸2"
            elif rank_idx <= 4:
                role = "流し"
            else:
                role = "-"
            pc = row.player_class if isinstance(row.player_class, str) else ""
            lp = row.line_position if isinstance(row.line_position, str) else ""
            pv = getattr(row, "period", None)
            period_val = int(pv) if pv is not None and pv == pv else 0
            rs = row.racing_score
            rs_val = round(float(rs), 1) if rs == rs else 0.0
            wr = row.recent_win_rate_3m
            wr_val = round(float(wr) * 100, 1) if wr == wr else 0.0
            riders_detail.append({
                "frame_no":      fn,
                "ai_rank":       rank_idx + 1,
                "player_class":  pc,
                "period":        period_val,
                "racing_score":  rs_val,
                "win_rate_3m":   wr_val,
                "line_position": lp,
                "pred_prob_pct": round(float(row.pred_prob) * 100, 1),
                "role":          role,
            })

        entry = {
            "race_key":   race_key,
            "venue_name": venue_name,
            "race_no":    int(race_no),
            "start_time": start_time,
            "n_riders":   int(n_riders),
            "gap12":      float(gap12),
            "ratio":      float(ratio),
            "pivot1":     int(pivot1),
            "pivot2":     int(pivot2),
            "thirds":     [int(t) for t in thirds],
            "riders":     riders_detail,
        }

        if gap12 >= 0.15 and ratio < 1.3:
            # SS: 3連単 1→2→{thirds}
            entry["combo_str"] = f"{pivot1}→{pivot2}→{thirds_str}"
            entry["bet_type"]  = "3連単"
            ss_races.append(entry)
        elif gap12 >= 0.15 and ratio < 1.6:
            # S: 3連複 2軸×3頭流し（ratio≥1.6 は低配当リスクのため除外）
            entry["combo_str"] = f"{pivot1}-{pivot2}-{thirds_str}"
            entry["bet_type"]  = "3連複"
            s_races.append(entry)
        elif gap12 >= 0.15:
            # S条件だが ratio≥1.6 のためスキップ
            pass
        else:
            # A: 3連複 2軸×3頭流し
            entry["combo_str"] = f"{pivot1}-{pivot2}-{thirds_str}"
            entry["bet_type"]  = "3連複"
            a_races.append(entry)

    if not ss_races and not s_races and not a_races:
        click.echo("本日は6車立て以下の対象レース（gap12≥0.06）がありません。", err=True)
        raise SystemExit(1)

    sort_key = lambda x: (x["start_time"] == "--:--", x["start_time"], x["venue_name"], x["race_no"])
    for lst in (ss_races, s_races, a_races):
        lst.sort(key=sort_key)

    def _fmt(entry):
        n_str = f"{entry['n_riders']}車"
        return (
            f"  {entry['start_time']}  {entry['venue_name']:<6} {entry['race_no']:>2}R  "
            f"[{n_str}]  {entry['bet_type']}: {entry['combo_str']}  (3点/300円)"
        )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("=" * 66)
    lines.append(f" 競輪AI予想PICK  {target_date}  (SS:3連単 / S+A:3連複 / 3点300円)")
    lines.append(f" モデル: {model_label}  生成: {now_str}")
    lines.append("=" * 66)
    lines.append(" 対象: 6車立て以下  gap12≥0.06 のみ")
    lines.append(" SS: gap12≥0.15&ratio<1.3(3連単)  S: gap12≥0.15&ratio[1.3,1.6)(3連複)  A: gap12[0.06,0.15)(3連複)")
    lines.append("=" * 66)
    lines.append("")

    _RANK_INFO = [
        ("SS", ss_races, "gap12≥0.15 & ratio<1.3          / 3連単1→2 / ホールドアウト ROI 3315%"),
        ("S",  s_races,  "gap12≥0.15 & ratio [1.3, 1.6)   / 3連複    / ホールドアウト ROI 177%"),
        ("A",  a_races,  "gap12 [0.06,0.15)               / 3連複    / ホールドアウト ROI 215%"),
    ]
    for rank, races, desc in _RANK_INFO:
        lines.append(f"【{rank}ランク】 {len(races)}件  ({desc})")
        lines.append("─" * 60)
        if not races:
            lines.append("  (該当なし)")
        else:
            for e in races:
                lines.append(_fmt(e))
        lines.append("")

    lines.append("=" * 66)
    ss_cost = len(ss_races) * 300
    s_cost  = len(s_races)  * 300
    a_cost  = len(a_races)  * 300
    total_cost = ss_cost + s_cost + a_cost
    lines.append(f"  SS: {len(ss_races)}件 × 300円 = {ss_cost:,}円  (3連単)")
    lines.append(f"  S : {len(s_races)}件 × 300円 = {s_cost:,}円  (3連複)")
    lines.append(f"  A : {len(a_races)}件 × 300円 = {a_cost:,}円  (3連複)")
    lines.append(f"  合計投資額: {total_cost:,}円")
    lines.append("=" * 66)

    output_text = "\n".join(lines)

    if output_path is None:
        picks_dir = Path(__file__).parent.parent.parent / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(picks_dir / f"wave_picks_{target_date}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    click.echo(output_text)
    click.echo(f"\n[保存先] {output_path}")

    # per-race per-rider detail JSON（PDF生成用）
    all_race_details = (
        [{"rank": "SS", **e} for e in ss_races] +
        [{"rank": "S",  **e} for e in s_races] +
        [{"rank": "A",  **e} for e in a_races]
    )
    detail_path = picks_dir / f"wave_picks_{target_date}_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_race_details, f, ensure_ascii=False, indent=2)
    click.echo(f"[保存先] {detail_path}")


@cli.command("collect-wt")
@click.option("--date", "target_date", default=None, help="収集日 (YYYY-MM-DD), 省略時は昨日")
@click.option("--dry-run", is_flag=True, help="DBに保存しない（動作確認用）")
def collect_wt(target_date: str | None, dry_run: bool):
    """winticket からレースデータ（+オッズ）を収集してDBに保存"""
    from src.scraper.pipeline_wt import WinticketPipeline

    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    click.echo(f"[wt] Collecting {target_date} {'(dry-run)' if dry_run else ''}")
    init_db()
    pipeline = WinticketPipeline()
    stats = pipeline.collect_date(target_date, dry_run=dry_run)
    click.echo(f"[wt] Complete: venues={stats['venues']}, races={stats['races']}, "
               f"results={stats['results']}, errors={stats['errors']}")


@cli.command("collect-wt-range")
@click.option("--from", "from_ym", required=True, help="開始年月 (YYYY-MM)")
@click.option("--to", "to_ym", default=None, help="終了年月 (YYYY-MM), 省略時は今月")
@click.option("--dry-run", is_flag=True)
def collect_wt_range(from_ym: str, to_ym: str | None, dry_run: bool):
    """winticket データを年月範囲で一括収集（最新から過去順）

    例: python -m src.cli.main collect-wt-range --from 2025-01
        python -m src.cli.main collect-wt-range --from 2025-01 --to 2025-06
    """
    from src.scraper.pipeline_wt import WinticketPipeline

    try:
        start_year, start_month = map(int, from_ym.split("-"))
    except ValueError:
        click.echo("Error: --from は YYYY-MM 形式で指定してください", err=True)
        raise SystemExit(1)

    if to_ym is None:
        today = date.today()
        end_year, end_month = today.year, today.month
    else:
        try:
            end_year, end_month = map(int, to_ym.split("-"))
        except ValueError:
            click.echo("Error: --to は YYYY-MM 形式で指定してください", err=True)
            raise SystemExit(1)

    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    months = list(reversed(months))  # 最新優先

    click.echo(f"[wt] Collecting {len(months)} months (newest first) {'(dry-run)' if dry_run else ''}")
    init_db()
    pipeline = WinticketPipeline()
    total = {"venues": 0, "races": 0, "results": 0, "errors": 0}

    for i, (year, month) in enumerate(months, 1):
        click.echo(f"\n[{i}/{len(months)}] {year}/{month:02d}")
        stats = pipeline.collect_month(year, month, dry_run=dry_run)
        for k in total:
            total[k] += stats.get(k, 0)
        click.echo(f"  -> venues={stats['venues']}, races={stats['races']}, "
                   f"results={stats['results']}, errors={stats['errors']}")

    click.echo(f"\n[wt] All done: venues={total['venues']}, races={total['races']}, "
               f"results={total['results']}, errors={total['errors']}")


@cli.command("status-wt")
def status_wt():
    """winticket DB の収集状況を確認"""
    from src.database import get_connection
    with get_connection() as conn:
        races   = conn.execute("SELECT COUNT(*) FROM wt_races").fetchone()[0]
        entries = conn.execute("SELECT COUNT(*) FROM wt_entries").fetchone()[0]
        with_result = conn.execute(
            "SELECT COUNT(*) FROM wt_entries WHERE finish_order IS NOT NULL"
        ).fetchone()[0]
        odds    = conn.execute("SELECT COUNT(*) FROM wt_odds").fetchone()[0]
        latest  = conn.execute("SELECT MAX(race_date) FROM wt_races").fetchone()[0]
        earliest = conn.execute("SELECT MIN(race_date) FROM wt_races").fetchone()[0]

    click.echo(f"wt_races:   {races:,}")
    click.echo(f"wt_entries: {entries:,}  (with result: {with_result:,})")
    click.echo(f"wt_odds:    {odds:,}")
    click.echo(f"Date range: {earliest or 'N/A'} ~ {latest or 'N/A'}")


@cli.command("train-wt")
@click.option("--from", "from_date", default="2025-01-01", help="学習開始日")
@click.option("--to", "to_date", default=None, help="学習終了日")
@click.option("--test-from", "test_from", default=None,
              help="テスト開始日（省略時は後ろ20%）")
@click.option("--save-as", "save_as", default=None,
              help="保存名（例: lgbm_wt_v1）。省略時は lgbm_wt")
@click.option("--full-refit/--no-full-refit", "full_refit", default=False,
              help="ホールドアウト評価後、全データ(df_train)で配信用モデルを再学習して保存"
                   "（H-1: holdout打切りモデルを本番配信しない）")
@click.option("--promote/--no-promote", "promote", default=True,
              help="save-as≠lgbm_wt のとき lgbm_wt にも反映するか。--no-promote で評価runが本番を汚さない")
def train_wt(from_date: str, to_date: str | None, test_from: str | None, save_as: str | None,
             full_refit: bool, promote: bool):
    """winticket データでモデルを学習して data/models/ に保存

    例: python -m src.cli.main train-wt --from 2025-01-01
        python -m src.cli.main train-wt --from 2025-01-01 --test-from 2026-01-01
    """
    from src.preprocessing.feature_wt import (
        load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
    )
    from src.models.trainer import train_lgbm, save_model

    load_max = None if test_from else to_date
    click.echo(f"[wt] Loading {from_date} ~ {load_max or 'latest'} ...")
    click.echo(f"Features ({len(FEATURE_COLS_WT)}): {', '.join(FEATURE_COLS_WT)}")

    df_raw = load_raw_data_wt(min_date=from_date, max_date=load_max)
    if df_raw.empty:
        click.echo("データがありません。先に collect-wt を実行してください。", err=True)
        raise SystemExit(1)

    df = build_features_wt(df_raw)
    df_train = df[df["finish_order"].notna()].copy()
    click.echo(f"Training samples: {len(df_train):,} entries / "
               f"{df_train['race_key'].nunique():,} races")

    if len(df_train) < 100:
        click.echo("学習データが不足しています（100行未満）。", err=True)
        raise SystemExit(1)

    if test_from:
        df_tr = df_train[df_train["race_date"] < test_from]
        df_te = df_train[df_train["race_date"] >= test_from]
        click.echo(f"Train: {df_tr['race_key'].nunique():,} races  "
                   f"Test: {df_te['race_key'].nunique():,} races  "
                   f"(split: {test_from})")
    else:
        dates = sorted(df_train["race_date"].unique())
        split_idx = int(len(dates) * 0.8)
        split_date = dates[split_idx]
        df_tr = df_train[df_train["race_date"] < split_date]
        df_te = df_train[df_train["race_date"] >= split_date]
        click.echo(f"Train: {df_tr['race_key'].nunique():,} races  "
                   f"Test: {df_te['race_key'].nunique():,} races  "
                   f"(split: {split_date})")

    click.echo("Training LightGBM (winticket) ...")
    model = train_lgbm(df_tr, feature_cols=FEATURE_COLS_WT, target_col=TARGET_COL_WT)

    # --- ホールドアウト評価（保存前に算出。配信モデルとは独立の監視指標）---
    test_auc = None
    if not df_te.empty:
        from sklearn.metrics import roc_auc_score
        X_te = df_te[FEATURE_COLS_WT].fillna(0)
        y_te = df_te[TARGET_COL_WT].values
        test_auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        click.echo(f"\nHoldout Test AUC: {test_auc:.4f}  (n={len(df_te):,} entries)")

    # --- H-1: 配信モデルは全データで再学習（holdout打切りモデルを本番にしない）---
    if full_refit:
        click.echo(f"[full-refit] 全データ {df_train['race_key'].nunique():,} races "
                   f"で配信用モデルを再学習 ...")
        model = train_lgbm(df_train, feature_cols=FEATURE_COLS_WT, target_col=TARGET_COL_WT)

    model_name = save_as or "lgbm_wt"
    save_model(model, model_name)
    # 昇格（lgbm_wt への反映）。--no-promote で抑止（評価専用runが本番を汚さない）
    if promote and model_name != "lgbm_wt":
        save_model(model, "lgbm_wt")

    # --- メタデータ sidecar（再現性・H-1/M-5）---
    import json
    import subprocess
    from datetime import datetime as _dt
    models_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models"
    meta = {
        "model_name": model_name,
        "full_refit": bool(full_refit),
        "from": from_date,
        "test_from": test_from,
        "n_train_races": int(df_train["race_key"].nunique()),
        "fit_rows": int(len(df_train) if full_refit else len(df_tr)),
        "test_auc_holdout": test_auc,
        "feature_count": len(FEATURE_COLS_WT),
        "trained_at": _dt.now().isoformat(timespec="seconds"),
    }
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=str(models_dir)
        ).strip()
    except Exception:
        meta["git_commit"] = None
    (models_dir / f"{model_name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if promote and model_name != "lgbm_wt":
        (models_dir / "lgbm_wt.meta.json").write_text(
            json.dumps({**meta, "model_name": "lgbm_wt"}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    click.echo(f"[meta] {model_name}.meta.json 保存（fit_rows={meta['fit_rows']:,}, "
               f"full_refit={full_refit}, holdout_auc={test_auc}）")


@cli.command("wave-picks-wt")
@click.option("--date", "target_date", default=None,
              help="対象日 YYYY-MM-DD（省略時: 今日）")
@click.option("--output", "output_path", default=None,
              help="出力先ファイルパス（省略時: data/picks/wave_picks_wt_{date}.txt）")
@click.option("--model", "model_name", default="lgbm_wt",
              help="使用するモデルファイル名（.pkl なし）")
@click.option("--min-trio-odds", "min_trio_odds", default=0.0, show_default=True,
              type=float,
              help="3連複の最低オッズ（この値未満の組み合わせのみの場合はスキップ）。0=フィルター無効")
@click.option("--upset-gate", "upset_gate", default=None,
              type=click.Choice(["Q1_loose", "Q2", "Q3"]),
              help="波乱/非本命ゲート: 指定帯まで(loose側)のみ出力し本命堅レースを見送る。"
                   "省略時は全件出力（各pickにupset_tierをタグ付けのみ＝本番挙動不変・前向き検証用）")
@click.option("--gami-skip-odds", "gami_skip_odds", default=0.0, show_default=True,
              type=float,
              help="ガミ回避(見送り): 3点(SS=3連単/S・A=3連複)のうち1点でも朝オッズ<この倍率なら"
                   "レースごと見送り（鉄板=低価値レースの除外）。0=無効。推奨3.0")
@click.option("--b-rank-odds", "b_rank_odds", default=0.0, show_default=True,
              type=float,
              help="Bランク閾値: 見送りはしないが、3点中1点でも朝オッズ<この倍率なら"
                   "Bランク（購入者判断にゆだねる）として別枠表示。0=無効。推奨5.0"
                   "（gami-skip-odds≤オッズ<b-rank-odds がBランク帯）")
def wave_picks_wt(target_date, output_path, model_name, min_trio_odds, upset_gate,
                  gami_skip_odds, b_rank_odds):
    """winticket モデルで wave-picks を生成（オッズ表示・フィルター付き）

    オッズは AI 予想後の購入判断に使用。市場が既に織り込んでいる
    （低オッズ）組み合わせを --min-trio-odds でフィルターできる。

    例:
        python -m src.cli.main wave-picks-wt
        python -m src.cli.main wave-picks-wt --min-trio-odds 3.0
    """
    import json
    import re
    import pandas as pd
    from datetime import datetime, timezone, timedelta
    from src.preprocessing.feature_wt import (
        load_raw_data_wt, build_features_wt, FEATURE_COLS_WT,
    )
    from src.models.trainer import load_model
    from src.database import get_connection
    from src.strategy_wt import race_signals, passes_upset_gate
    from pathlib import Path

    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    # 会場名マップ
    try:
        with get_connection() as conn:
            vi = pd.read_sql("SELECT venue_code, name FROM venue_info", conn)
        venue_map = dict(zip(vi["venue_code"], vi["name"]))
    except Exception:
        venue_map = {}

    # オッズデータをロード（DB にあれば）
    def _load_odds(race_key: str) -> dict[str, list[dict]]:
        """wt_odds から {bet_type: [{combination, odds_value}]} を返す"""
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT bet_type, combination, odds_value "
                    "FROM wt_odds WHERE race_key = ?",
                    (race_key,),
                ).fetchall()
            result: dict[str, list[dict]] = {}
            for row in rows:
                result.setdefault(row[0], []).append(
                    {"combination": row[1], "odds_value": row[2]}
                )
            return result
        except Exception:
            return {}

    def _find_trio_odds(odds: dict, frames: list[int]) -> float | None:
        """3連複オッズの中でフレーム番号リストを含む組み合わせの最小値を返す"""
        trio_list = odds.get("trio", [])
        if not trio_list:
            return None
        key_set = set(str(f) for f in frames[:3])  # 軸2+流し1の組み合わせ
        min_odds = None
        for item in trio_list:
            # combination は "-" 区切りを仮定（例: "1-3-5"）
            parts = set(re.split(r"[-=]", item["combination"]))
            if key_set == parts:
                v = item["odds_value"]
                if min_odds is None or v < min_odds:
                    min_odds = v
        return min_odds

    def _find_trifecta_odds(odds: dict, pivot1: int, pivot2: int, thirds: list[int]) -> float | None:
        """3連単オッズの中で pivot1→pivot2→{thirds} の最小値を返す"""
        trifecta_list = odds.get("trifecta", [])
        if not trifecta_list:
            return None
        targets = {f"{pivot1}-{pivot2}-{t}" for t in thirds}
        # 区切り文字が不明なので正規化して比較
        min_odds = None
        for item in trifecta_list:
            raw = item["combination"]
            parts = re.split(r"[-=→]", raw)
            if len(parts) == 3:
                normalized = f"{parts[0]}-{parts[1]}-{parts[2]}"
                if normalized in targets:
                    v = item["odds_value"]
                    if min_odds is None or v < min_odds:
                        min_odds = v
        return min_odds

    try:
        model = load_model(model_name)
    except FileNotFoundError:
        click.echo(f"モデル '{model_name}' が見つかりません。先に train-wt を実行してください。",
                   err=True)
        raise SystemExit(1)

    click.echo(f"[wt] Loading data for {target_date} ...")
    df_raw = load_raw_data_wt(min_date=target_date, max_date=target_date)
    if df_raw.empty:
        click.echo(f"{target_date} の winticket データがありません。"
                   "先に collect-wt を実行してください。", err=True)
        raise SystemExit(1)

    df = build_features_wt(df_raw)
    X = df[FEATURE_COLS_WT].fillna(0)
    df["pred_prob"] = model.predict_proba(X)[:, 1]

    df["race_no"] = df["race_key"].apply(
        lambda rk: int(rk.split("_")[2]) if len(rk.split("_")) >= 3 else 0
    )
    def _fmt_start(s):
        # winticket start_at は unix秒(JST)。HH:MM へ整形
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return "--:--"
        try:
            ts = int(s)
            return datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=9))).strftime("%H:%M")
        except (ValueError, TypeError):
            s = str(s)
            return s[11:16] if len(s) > 10 else s
    df["start_time"] = df["start_at"].apply(_fmt_start)

    ss_races, s_races, a_races, b_races = [], [], [], []
    skipped_odds = 0
    skipped_gami = 0

    for race_key, grp in df.groupby("race_key"):
        grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n_riders = len(grp_sorted)
        if n_riders > 6:
            continue

        p = grp_sorted["pred_prob"].tolist()
        top1 = p[0]
        top2_prob = p[1] if n_riders >= 2 else 0.0
        gap12 = top1 - top2_prob
        ratio = top1 / (3 / n_riders)

        if gap12 < 0.06:
            continue

        # 波乱/非本命シグナル（確定前・朝算出可）
        sig = race_signals(p, n_riders)
        upset_t = sig["upset_tier"]
        # opt-in ゲート: 指定帯まで(loose側)のみ。省略時は全件タグ付けのみ（本番挙動不変）。
        if upset_gate is not None and not passes_upset_gate(sig["top3_sum"], upset_gate):
            continue

        venue_id = grp_sorted["venue_id"].iloc[0]
        venue_name = venue_map.get(str(venue_id), str(venue_id))
        race_no = int(grp_sorted["race_no"].iloc[0])
        start_time = grp_sorted["start_time"].iloc[0]

        frames = grp_sorted["frame_no"].astype(int).tolist()
        pivot1, pivot2 = frames[0], frames[1]
        thirds = frames[2:5]
        thirds_str = ",".join(str(t) for t in thirds)

        # オッズ取得
        odds = _load_odds(race_key)

        # ガミ判定: 3点それぞれの朝オッズの最小値で3段階に振り分ける。
        #   min < gami_skip_odds      → 見送り（鉄板=低価値）
        #   gami_skip_odds ≤ min < b_rank_odds → Bランク（購入者判断にゆだねる）
        #   min ≥ b_rank_odds         → 通常（SS/S/A）
        is_ss = (gap12 >= 0.15 and ratio < 1.3)
        gami_zone = None  # None=通常 / "B"=Bランク
        if gami_skip_odds > 0 or b_rank_odds > 0:
            leg_odds = []
            for tdr in thirds:
                if is_ss:
                    leg_odds.append(_find_trifecta_odds(odds, pivot1, pivot2, [tdr]))
                else:
                    leg_odds.append(_find_trio_odds(odds, [pivot1, pivot2, tdr]))
            known = [o for o in leg_odds if o is not None]
            min_leg = min(known) if known else None
            if min_leg is not None:
                if gami_skip_odds > 0 and min_leg < gami_skip_odds:
                    skipped_gami += 1
                    continue
                if b_rank_odds > 0 and min_leg < b_rank_odds:
                    gami_zone = "B"

        # riders_detail（PDF生成との互換性を保つ）
        riders_detail = []
        for rank_idx, row in enumerate(grp_sorted.itertuples(index=False)):
            fn = int(row.frame_no)
            role = "軸1" if rank_idx == 0 else "軸2" if rank_idx == 1 else "流し" if rank_idx <= 4 else "-"
            pc = row.player_class if isinstance(row.player_class, str) else ""
            lp = row.style if isinstance(getattr(row, "style", None), str) else ""
            pv = getattr(row, "term", None)
            period_val = int(pv) if pv is not None and pv == pv else 0
            rp = row.race_point
            rp_val = round(float(rp), 1) if rp == rp else 0.0
            wr = row.first_rate
            wr_val = round(float(wr), 1) if wr == wr else 0.0
            riders_detail.append({
                "frame_no":      fn,
                "ai_rank":       rank_idx + 1,
                "player_class":  pc,
                "period":        period_val,
                "racing_score":  rp_val,
                "win_rate_3m":   wr_val,
                "line_position": lp,
                "pred_prob_pct": round(float(row.pred_prob) * 100, 1),
                "role":          role,
            })

        # オッズフィルター
        trio_odds_val = None
        if gap12 >= 0.15 and ratio < 1.3:
            # SS: 3連単チェック
            trio_odds_val = _find_trifecta_odds(odds, pivot1, pivot2, thirds)
        else:
            trio_odds_val = _find_trio_odds(odds, [pivot1, pivot2] + thirds)

        odds_label = f"{trio_odds_val:.1f}倍" if trio_odds_val is not None else "オッズ未取得"

        if min_trio_odds > 0 and trio_odds_val is not None and trio_odds_val < min_trio_odds:
            skipped_odds += 1
            continue

        entry = {
            "race_key":   race_key,
            "venue_name": venue_name,
            "race_no":    race_no,
            "start_time": start_time,
            "n_riders":   int(n_riders),
            "gap12":      float(gap12),
            "ratio":      float(ratio),
            "pivot1":     int(pivot1),
            "pivot2":     int(pivot2),
            "thirds":     [int(t) for t in thirds],
            "riders":     riders_detail,
            "odds_label": odds_label,
            "top3_sum":   round(float(sig["top3_sum"]), 4),
            "upset_tier": upset_t,
        }

        if is_ss:
            base_rank = "SS"
            entry["combo_str"] = f"{pivot1}→{pivot2}→{thirds_str}"
            entry["bet_type"]  = "3連単"
        elif gap12 >= 0.15 and ratio < 1.6:
            base_rank = "S"
            entry["combo_str"] = f"{pivot1}-{pivot2}-{thirds_str}"
            entry["bet_type"]  = "3連複"
        elif gap12 >= 0.15:
            continue  # ratio≥1.6 は低配当リスクでスキップ
        else:
            base_rank = "A"
            entry["combo_str"] = f"{pivot1}-{pivot2}-{thirds_str}"
            entry["bet_type"]  = "3連複"

        if gami_zone == "B":
            # 3〜5倍未満の安い目を含む＝鉄板寄り。購入者判断にゆだねるBランクへ。
            entry["base_rank"] = base_rank
            b_races.append(entry)
        else:
            {"SS": ss_races, "S": s_races, "A": a_races}[base_rank].append(entry)

    if not ss_races and not s_races and not a_races and not b_races:
        msg = "本日は6車立て以下の対象レース（gap12≥0.06）がありません。"
        if skipped_odds > 0:
            msg += f"（オッズフィルターで {skipped_odds} 件スキップ）"
        if skipped_gami > 0:
            msg += f"（ガミ回避<{gami_skip_odds:.0f}倍で {skipped_gami} 件スキップ）"
        click.echo(msg, err=True)
        raise SystemExit(1)

    sort_key = lambda x: (x["start_time"] == "--:--", x["start_time"], x["venue_name"], x["race_no"])
    for lst in (ss_races, s_races, a_races, b_races):
        lst.sort(key=sort_key)

    def _fmt(entry):
        n_str = f"{entry['n_riders']}車"
        odds_str = f"  [{entry['odds_label']}]" if entry.get("odds_label") else ""
        base = f"(元{entry['base_rank']}) " if entry.get("base_rank") else ""
        return (
            f"  {entry['start_time']}  {entry['venue_name']:<6} {entry['race_no']:>2}R  "
            f"[{n_str}]  {base}{entry['bet_type']}: {entry['combo_str']}  (3点/300円){odds_str}"
        )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("=" * 70)
    lines.append(f" 競輪AI予想PICK [wt]  {target_date}  (SS:3連単 / S+A:3連複 / 3点300円)")
    lines.append(f" モデル: {model_name}  生成: {now_str}")
    if min_trio_odds > 0:
        lines.append(f" オッズフィルター: ≥{min_trio_odds:.1f}倍  (スキップ: {skipped_odds}件)")
    if gami_skip_odds > 0:
        lines.append(f" ガミ回避(見送り): 3点中<{gami_skip_odds:.0f}倍を含むレースを除外  (スキップ: {skipped_gami}件)")
    if b_rank_odds > 0:
        lines.append(f" Bランク: 最安目が{gami_skip_odds:.0f}〜{b_rank_odds:.0f}倍未満（購入は各自判断）")
    lines.append("=" * 70)
    lines.append(" 対象: 6車立て以下  gap12≥0.06 のみ")
    lines.append(" SS: gap12≥0.15&ratio<1.3(3連単)  S: gap12≥0.15&ratio[1.3,1.6)(3連複)  A: gap12[0.06,0.15)(3連複)")
    lines.append("=" * 70)
    lines.append("")

    _RANK_INFO = [
        ("SS", ss_races),
        ("S",  s_races),
        ("A",  a_races),
    ]
    for rank, races in _RANK_INFO:
        lines.append(f"【{rank}ランク】 {len(races)}件")
        lines.append("─" * 60)
        lines.append("  (該当なし)" if not races else "")
        for e in races:
            lines.append(_fmt(e))
        lines.append("")

    if b_rank_odds > 0:
        lines.append(f"【Bランク】 {len(b_races)}件  ※最安目が{gami_skip_odds:.0f}〜{b_rank_odds:.0f}倍未満＝鉄板寄り・購入は各自判断")
        lines.append("─" * 60)
        lines.append("  (該当なし)" if not b_races else "")
        for e in b_races:
            lines.append(_fmt(e))
        lines.append("")

    lines.append("=" * 70)
    ss_cost = len(ss_races) * 300
    s_cost  = len(s_races)  * 300
    a_cost  = len(a_races)  * 300
    total_cost = ss_cost + s_cost + a_cost
    lines.append(f"  SS: {len(ss_races)}件 × 300円 = {ss_cost:,}円  (3連単)")
    lines.append(f"  S : {len(s_races)}件 × 300円 = {s_cost:,}円  (3連複)")
    lines.append(f"  A : {len(a_races)}件 × 300円 = {a_cost:,}円  (3連複)")
    lines.append(f"  推奨合計投資額: {total_cost:,}円  (SS/S/A)")
    if b_rank_odds > 0:
        lines.append(f"  B : {len(b_races)}件 × 300円 = {len(b_races)*300:,}円  (購入は各自判断・上記合計に含めず)")
    lines.append("=" * 70)

    output_text = "\n".join(lines)

    if output_path is None:
        picks_dir = Path(__file__).parent.parent.parent / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(picks_dir / f"wave_picks_wt_{target_date}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    click.echo(output_text)
    click.echo(f"\n[保存先] {output_path}")

    # per-race detail JSON（notify_picks.py の PDF 生成と互換）
    all_race_details = (
        [{"rank": "SS", **e} for e in ss_races] +
        [{"rank": "S",  **e} for e in s_races] +
        [{"rank": "A",  **e} for e in a_races] +
        [{"rank": "B",  **e} for e in b_races]
    )
    detail_path = Path(output_path).parent / f"wave_picks_wt_{target_date}_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_race_details, f, ensure_ascii=False, indent=2)
    click.echo(f"[保存先] {detail_path}")


@cli.command("backtest-wt")
@click.option("--from", "from_date", default="2025-01-01", help="評価開始日")
@click.option("--to", "to_date", default=None, help="評価終了日")
@click.option("--model", "model_name", default="lgbm_wt", help="モデルファイル名（.pklなし）")
@click.option("--max-riders", "max_riders", default=None, type=int,
              help="出走頭数フィルター（実運用は6）")
@click.option("--min-gap12", "min_gap12", default=None, type=float,
              help="top1-top2 pred_prob 差フィルター（wave-picks-wtは0.06）")
@click.option("--tiered", is_flag=True,
              help="wave-picks-wt の SS/S/A 層別本番戦略で評価（ks production と同条件）")
@click.option("--value", "value_mode", is_flag=True,
              help="EV(期待値)ベースのバリューベッティングで評価")
@click.option("--ev-min", "ev_min", default=1.0, type=float, show_default=True,
              help="バリューモード: 購入する最低EV（1.0=損益分岐, >1=モデル優位分のみ）")
@click.option("--max-per-race", "max_per_race", default=5, type=int, show_default=True,
              help="バリューモード: 1レース最大購入点数")
@click.option("--max-ratio", "max_ratio", default=None, type=float,
              help="バリューモード: top1_prob/(3/n)<この値の拮抗レースのみ（例1.3）")
def backtest_wt(from_date: str, to_date: str | None, model_name: str,
                max_riders: int | None, min_gap12: float | None, tiered: bool,
                value_mode: bool, ev_min: float, max_per_race: int,
                max_ratio: float | None):
    """winticket モデルで買い目バックテストを実行（wt_odds の実オッズ使用）

    例: python -m src.cli.main backtest-wt --from 2026-01-01
        python -m src.cli.main backtest-wt --from 2026-01-01 --max-riders 6 --min-gap12 0.06
        python -m src.cli.main backtest-wt --from 2026-01-01 --tiered
    """
    from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
    from src.models.trainer import load_model
    from src.evaluation.backtest_wt import (
        run_backtest_wt, print_backtest_wt,
        run_tiered_backtest_wt, print_tiered_backtest_wt,
        run_value_backtest_wt, print_value_backtest_wt,
    )

    try:
        model = load_model(model_name)
    except FileNotFoundError:
        click.echo(f"モデル '{model_name}' が見つかりません。先に train-wt を実行してください。",
                   err=True)
        raise SystemExit(1)

    click.echo(f"[wt] Loading {from_date} ~ {to_date or 'latest'} ...")
    df_raw = load_raw_data_wt(min_date=from_date, max_date=to_date)
    if df_raw.empty:
        click.echo("データがありません。先に collect-wt を実行してください。", err=True)
        raise SystemExit(1)

    df = build_features_wt(df_raw)
    df = df[df["finish_order"].notna()].copy()
    n_races = df["race_key"].nunique()
    click.echo(f"評価対象: {len(df):,} entries / {n_races:,} races")

    if value_mode:
        result = run_value_backtest_wt(
            model, df, ev_min=ev_min, max_per_race=max_per_race,
            max_riders=max_riders or 9, max_ratio=max_ratio,
        )
        params = f"(ev_min={ev_min}, max/R={max_per_race}, max_ratio={max_ratio})"
        print_value_backtest_wt(result, params)
        return

    if tiered:
        df_result = run_tiered_backtest_wt(model, df, max_riders=max_riders or 6)
        print_tiered_backtest_wt(df_result)
        return

    df_result = run_backtest_wt(
        model, df, max_riders=max_riders, min_gap12=min_gap12,
    )
    eval_races = int(df_result["対象レース数"].iloc[0]) if not df_result.empty else 0
    print_backtest_wt(df_result, total_races=eval_races)


if __name__ == "__main__":
    cli()
