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
def backtest(model_type: str, from_date: str, to_date: str | None):
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
    click.echo(f"Evaluating {df_eval['race_key'].nunique():,} races ...")

    df_result = run_backtest(model, df_eval)
    print_backtest(df_result, total_races=df_eval["race_key"].nunique())


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
    """SS/S/Aランクレースを抽出して予想テキストを出力

    ランク定義:
      SS: 出走5車以下 & top1<60%  →  3連複上位3車BOX 1点
      S : 出走6車     & top1<58%  →  3連複上位3車BOX 1点
      A : top1 60-70% & gap12>0.12 →  3連複2軸(#1-#2)×3頭流し 3点
    """
    from datetime import datetime
    import itertools
    import pandas as pd
    from src.preprocessing.feature_engineer import load_raw_data, build_features, FEATURE_COLS
    from src.models.trainer import load_model
    from src.database import get_connection
    from pathlib import Path

    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    # venue_name マッピングをDBから取得
    try:
        with get_connection() as conn:
            vi = pd.read_sql("SELECT venue_code, name FROM venue_info", conn)
        venue_map = dict(zip(vi["venue_code"], vi["name"]))
    except Exception:
        venue_map = {}

    # モデルロード
    try:
        model = load_model(model_type)
    except FileNotFoundError:
        click.echo("モデルが見つかりません。先に train コマンドを実行してください。", err=True)
        raise SystemExit(1)

    # モデル名の取得（ファイル名から）
    model_dir = Path(__file__).parent.parent.parent / "data" / "models"
    model_label = model_type
    for candidate in sorted(model_dir.glob(f"{model_type}_v*.pkl"), reverse=True):
        model_label = candidate.stem
        break

    # データ取得
    click.echo(f"Loading data for {target_date} ...")
    df_raw = load_raw_data(min_date=target_date, max_date=target_date)
    if df_raw.empty:
        click.echo(f"{target_date} のデータがDBに存在しません。", err=True)
        raise SystemExit(1)

    df = build_features(df_raw)

    # 予測確率付与
    df["pred_prob"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

    # race_no を race_key から抽出（例: 20260529_21_01 → 01→1）
    def parse_race_no(rk):
        parts = rk.split("_")
        return int(parts[2]) if len(parts) >= 3 else 0

    df["race_no"] = df["race_key"].apply(parse_race_no)

    # レースごとにランク判定
    ss_races = []
    s_races = []
    a_races = []

    for race_key, grp in df.groupby("race_key"):
        grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n_riders = len(grp_sorted)
        top1 = grp_sorted["pred_prob"].iloc[0]
        top2_prob = grp_sorted["pred_prob"].iloc[1] if n_riders >= 2 else 0.0
        gap12 = top1 - top2_prob

        venue_code = grp_sorted["venue_code"].iloc[0]
        venue_name = venue_map.get(venue_code, str(venue_code))
        race_no = grp_sorted["race_no"].iloc[0]

        if n_riders <= 5 and top1 < 0.60:
            rank = "SS"
        elif n_riders == 6 and top1 < 0.58:
            rank = "S"
        elif 0.60 <= top1 < 0.70 and gap12 > 0.12:
            rank = "A"
        else:
            continue

        entry = {
            "race_key": race_key,
            "venue_name": venue_name,
            "race_no": race_no,
            "rank": rank,
            "n_riders": n_riders,
            "top1": top1,
            "gap12": gap12,
            "top5": grp_sorted.head(5),
        }

        if rank == "SS":
            ss_races.append(entry)
        elif rank == "S":
            s_races.append(entry)
        else:
            a_races.append(entry)

    # 会場・レース番号順にソート
    for lst in (ss_races, s_races, a_races):
        lst.sort(key=lambda x: (x["venue_name"], x["race_no"]))

    def _format_ss_s_entry(entry):
        """SS/S共通フォーマット: 3連複BOX3"""
        top3 = entry["top5"].head(3)
        frames = list(top3["frame_no"].astype(int))
        box_str = "-".join(str(f) for f in sorted(frames))
        n_str = f"{entry['n_riders']}車立て"
        rows = []
        rows.append("")
        rows.append(
            f"  ◇ {entry['venue_name']}  {entry['race_no']}R  "
            f"[{entry['rank']}]  {n_str}  top1={entry['top1']:.3f}"
        )
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            rows.append(
                f"    #{i}  {int(row['frame_no'])}車  {row['pred_prob']:.3f}  "
                f"得点:{row['racing_score']:.1f}"
            )
        rows.append(f"    → 3連複BOX: {box_str}  (1点/100円)")
        return rows

    def _format_a_entry(entry):
        """Aランクフォーマット: 3連複2軸(#1-#2)×3頭流し"""
        top5 = entry["top5"]
        r = list(top5["frame_no"].astype(int))
        pivot1, pivot2 = r[0], r[1]
        # {p1,p2,r2}, {p1,p2,r3}, {p1,p2,r4}
        thirds = r[2:5]
        combos = [sorted([pivot1, pivot2, t]) for t in thirds]
        combo_strs = ["  ".join(str(v) for v in c) for c in combos]
        n_str = f"{entry['n_riders']}車立て"
        rows = []
        rows.append("")
        rows.append(
            f"  ◇ {entry['venue_name']}  {entry['race_no']}R  "
            f"[A]  {n_str}  top1={entry['top1']:.3f}  gap12={entry['gap12']:.3f}"
        )
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            marker = "▲" if i <= 2 else " "
            rows.append(
                f"    {marker}#{i}  {int(row['frame_no'])}車  {row['pred_prob']:.3f}  "
                f"得点:{row['racing_score']:.1f}"
            )
        rows.append(f"    軸: {pivot1}車-{pivot2}車  相手: {'-'.join(str(t) for t in thirds)}")
        for cs in combo_strs:
            rows.append(f"    → 3連複: {cs}  (1点/100円)")
        rows.append(f"    計{len(combos)}点/{len(combos)*100}円")
        return rows

    # 出力テキスト構築
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("=" * 60)
    lines.append(f" 競輪AI予想PICK  {target_date}")
    lines.append(f" モデル: {model_label}  生成: {now_str}")
    lines.append("=" * 60)
    lines.append(" SS: ≤5車立て & top1<60%    →  3連複BOX3 (1点/100円)")
    lines.append(" S : 6車立て  & top1<58%    →  3連複BOX3 (1点/100円)")
    lines.append(" A : top1 60-70% & gap12>0.12 →  2軸×3頭流し (3点/300円)")
    lines.append("=" * 60)
    lines.append("")

    # SS セクション
    lines.append("┌" + "─" * 57 + "┐")
    lines.append(f"│ 【SSランク】 {len(ss_races)}件" + " " * (44 - len(str(len(ss_races)))) + "│")
    lines.append("└" + "─" * 57 + "┘")
    if not ss_races:
        lines.append("  (該当なし)")
    else:
        for entry in ss_races:
            lines.extend(_format_ss_s_entry(entry))

    lines.append("")

    # S セクション
    lines.append("┌" + "─" * 57 + "┐")
    lines.append(f"│ 【Sランク】  {len(s_races)}件" + " " * (44 - len(str(len(s_races)))) + "│")
    lines.append("└" + "─" * 57 + "┘")
    if not s_races:
        lines.append("  (該当なし)")
    else:
        for entry in s_races:
            lines.extend(_format_ss_s_entry(entry))

    lines.append("")

    # A セクション
    lines.append("┌" + "─" * 57 + "┐")
    lines.append(f"│ 【Aランク】  {len(a_races)}件" + " " * (44 - len(str(len(a_races)))) + "│")
    lines.append("└" + "─" * 57 + "┘")
    if not a_races:
        lines.append("  (該当なし)")
    else:
        for entry in a_races:
            lines.extend(_format_a_entry(entry))

    lines.append("")
    lines.append("=" * 60)
    ss_cost = len(ss_races) * 100
    s_cost = len(s_races) * 100
    a_cost = len(a_races) * 300
    total_cost = ss_cost + s_cost + a_cost
    lines.append(f"  SS: {len(ss_races)}件 × 100円 = {ss_cost}円")
    lines.append(f"  S : {len(s_races)}件 × 100円 = {s_cost}円")
    lines.append(f"  A : {len(a_races)}件 × 300円 = {a_cost}円")
    lines.append(f"  合計投資額: {total_cost}円")
    lines.append("=" * 60)

    output_text = "\n".join(lines)

    # ファイル出力
    if output_path is None:
        picks_dir = Path(__file__).parent.parent.parent / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(picks_dir / f"wave_picks_{target_date}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    # コンソールにも出力
    click.echo(output_text)
    click.echo(f"\n[保存先] {output_path}")


if __name__ == "__main__":
    cli()
