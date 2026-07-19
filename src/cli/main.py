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

# 7+車 3連複のガミ閾値（レース単位: min(全目) < この値 → レース見送り。doc52）
# 2026-07-10 に買い目カット方式(SS/S)を廃止し doc48 のレース単位セマンティクスへ回帰。
# notify_prerace_wt.py / write_candidates_wt.py の GAMI_THRESHOLD と揃えること。
GAMI_THRESHOLD = 7.0

# ※ S/S+（三連単 1着固定フォーメーション・7PLUS_ST/STP）は 2026-07-15 に全廃。
#   優位性なし（keirin_survivor_bias_inflation 調査で ROI 70-90% = 控除率の壁）。

# JKA venue_code → 場名（venue_info DBが取得できない場合のフォールバック）
_VENUE_NAMES: dict[str, str] = {
    "11": "函館",   "12": "青森",   "13": "いわき平", "21": "弥彦",
    "22": "前橋",   "23": "取手",   "24": "宇都宮",  "25": "大宮",
    "26": "西武園",  "27": "京王閣",  "28": "立川",    "31": "松戸",
    "32": "千葉",   "34": "川崎",   "35": "平塚",    "36": "小田原",
    "37": "伊東",   "38": "静岡",   "42": "名古屋",  "43": "岐阜",
    "44": "大垣",   "45": "豊橋",   "46": "富山",    "47": "松阪",
    "48": "四日市",  "51": "福井",   "53": "奈良",    "54": "向日町",
    "55": "和歌山",  "56": "岸和田",  "61": "玉野",    "62": "広島",
    "63": "防府",   "71": "高松",   "73": "小松島",  "74": "高知",
    "75": "松山",   "81": "小倉",   "83": "久留米",  "84": "武雄",
    "85": "佐世保",  "86": "別府",   "87": "熊本",
}


def _venue_name(venue_map: dict, venue_id) -> str:
    """venue_map から場名を取得。なければ _VENUE_NAMES フォールバック、それもなければ番号。"""
    vid = str(venue_id)
    return venue_map.get(vid) or _VENUE_NAMES.get(vid, vid)


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
@click.option("--full-scan", is_flag=True,
              help="全VENUE_SLUGS会場を走査して開催を検出（初日開催の取りこぼし防止）。"
                   "既収集日でも全会場を再探索。当日予想収集など漏れが許されない場面で使用")
def collect_wt(target_date: str | None, dry_run: bool, full_scan: bool):
    """winticket からレースデータ（+オッズ）を収集してDBに保存"""
    from src.scraper.pipeline_wt import WinticketPipeline

    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    click.echo(f"[wt] Collecting {target_date} {'(dry-run)' if dry_run else ''}{' (full-scan)' if full_scan else ''}")
    init_db()
    pipeline = WinticketPipeline()
    stats = pipeline.collect_date(target_date, dry_run=dry_run, full_scan=full_scan)
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
@click.option("--test-to", "test_to", default=None,
              help="テスト終了日（--test-from とセットで使用。これより後のデータは学習にも評価にも使わない）")
@click.option("--save-as", "save_as", default=None,
              help="保存名（例: lgbm_wt_v1）。省略時は lgbm_wt")
@click.option("--full-refit/--no-full-refit", "full_refit", default=False,
              help="ホールドアウト評価後、全データ(df_train)で配信用モデルを再学習して保存"
                   "（H-1: holdout打切りモデルを本番配信しない）")
@click.option("--promote/--no-promote", "promote", default=True,
              help="save-as≠lgbm_wt のとき lgbm_wt にも反映するか。--no-promote で評価runが本番を汚さない")
@click.option("--target", "target_kind", default="top3", type=click.Choice(["top3", "win"]),
              help="学習ターゲット。top3=3着内（既定・配信モデル）、"
                   "win=1着のみ（Phase B・軸信頼度/相手選定シグナル用）")
def train_wt(from_date: str, to_date: str | None, test_from: str | None, test_to: str | None,
             save_as: str | None, full_refit: bool, promote: bool, target_kind: str):
    """winticket データでモデルを学習して data/models/ に保存

    例: python -m src.cli.main train-wt --from 2025-01-01
        python -m src.cli.main train-wt --from 2025-01-01 --test-from 2026-01-01
        python -m src.cli.main train-wt --from 2022-12-01 --test-from 2026-04-01 --test-to 2026-06-30
        python -m src.cli.main train-wt --from 2022-12-01 --target win --save-as lgbm_wt_win --no-promote
    """
    from src.preprocessing.feature_wt import (
        load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, TARGET_COL_WT,
        WIN_TARGET_COL_WT, prepare_X,
    )
    from src.models.trainer import train_lgbm, save_model

    target_col = WIN_TARGET_COL_WT if target_kind == "win" else TARGET_COL_WT
    if target_kind == "win" and promote:
        # 1着モデルが誤って配信用3着内モデル(lgbm_wt)を上書きしないための安全弁
        # （--no-promote 付け忘れ対策）。
        click.echo("[guard] --target win では --promote は無視します（lgbm_wt を汚染しないため）", err=True)
        promote = False

    load_max = test_to if test_from else to_date
    click.echo(f"[wt] Loading {from_date} ~ {load_max or 'latest'} ...")
    click.echo(f"Features ({len(FEATURE_COLS_WT)}): {', '.join(FEATURE_COLS_WT)}")
    click.echo(f"Target: {target_col}")

    df_raw = load_raw_data_wt(min_date=from_date, max_date=load_max)
    if df_raw.empty:
        click.echo("データがありません。先に collect-wt を実行してください。", err=True)
        raise SystemExit(1)

    df = build_features_wt(df_raw)
    # M-2: 学習母集団は finish_order が確定済みの全行（NaN=未確定のみ除外）。
    # 予測時・バックテスト(_apply_pred_prob_wt)は全エントリーで確率付与するため
    # 学習も同一母集団にしないと train/serve skew（欠車楽観バイアス）が生じる。
    # ローリング特徴の履歴計算は引き続き finish_order>=1 のみを参照（仕様変更なし）。
    df_train = df[df["finish_order"].notna()].copy()

    # 落車・失格を含むレースの学習除外（2026-07-16 検証済み・既定OFF）。
    # A/B検証の結果、除外すると「事故が起きなかった」という事後情報で学習母集団を
    # 選別することになり、実運用（事故込み）との較正がズレて S1 テストROIが
    # 122.8%→87.9% に劣化した。落車者のモデル事前順位分布はほぼ一様＝個人レベルの
    # 落車予測（リーク）は存在しないことも確認済み。よって既定は含めて学習する。
    # 再検証したい場合のみ環境変数 WT_EXCLUDE_DNF_RACES=1 で有効化。
    import os as _os_dnf
    _exclude_dnf = _os_dnf.environ.get("WT_EXCLUDE_DNF_RACES") == "1"
    from src.database import get_connection as _gc_dnf
    _dnf0 = (df_train[df_train["finish_order"] == 0][["race_key", "frame_no"]]
             if _exclude_dnf else df_train.iloc[0:0][["race_key", "frame_no"]])
    _dnf_races: set[str] = set()
    if len(_dnf0):
        import re as _re_dnf
        _rk0 = _dnf0["race_key"].unique().tolist()
        _boards: dict[str, set[int]] = {}
        with _gc_dnf() as _c_dnf:
            for _i in range(0, len(_rk0), 900):
                _chunk = _rk0[_i:_i + 900]
                _q = ("SELECT race_key, combination FROM wt_odds "
                      "WHERE bet_type='trio' AND race_key IN (%s)"
                      % ",".join("?" * len(_chunk)))
                for _rk_d, _comb in _c_dnf.execute(_q, _chunk):
                    try:
                        _parts = {int(x) for x in _re_dnf.split(r"[-=→]", str(_comb))}
                    except ValueError:
                        continue
                    _boards.setdefault(_rk_d, set()).update(_parts)
        for _row in _dnf0.itertuples(index=False):
            if int(_row.frame_no) in _boards.get(_row.race_key, set()):
                _dnf_races.add(_row.race_key)
        if _dnf_races:
            _before = df_train["race_key"].nunique()
            df_train = df_train[~df_train["race_key"].isin(_dnf_races)].copy()
            click.echo(f"落車・失格レース除外: {len(_dnf_races):,}レースを学習から除外 "
                       f"({_before:,} → {df_train['race_key'].nunique():,}レース)")

    n_dns = (df_train["finish_order"] == 0).sum()
    click.echo(f"Training samples: {len(df_train):,} entries / "
               f"{df_train['race_key'].nunique():,} races  "
               f"(finish_order!=NaN; DNS/DNF={n_dns:,}件を負例に含む{'・落車失格レース除外' if _exclude_dnf else ''})")

    if len(df_train) < 100:
        click.echo("学習データが不足しています（100行未満）。", err=True)
        raise SystemExit(1)

    if test_from:
        df_tr = df_train[df_train["race_date"] < test_from]
        df_te = df_train[df_train["race_date"] >= test_from]
        if test_to:
            df_te = df_te[df_te["race_date"] <= test_to]
        click.echo(f"Train: {df_tr['race_key'].nunique():,} races  "
                   f"Test: {df_te['race_key'].nunique():,} races  "
                   f"(split: {test_from}{' 〜 ' + test_to if test_to else ''})")
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
    model = train_lgbm(df_tr, feature_cols=FEATURE_COLS_WT, target_col=target_col)

    # --- ホールドアウト評価（保存前に算出。配信モデルとは独立の監視指標）---
    test_auc = None
    if not df_te.empty:
        from sklearn.metrics import roc_auc_score
        X_te = prepare_X(df_te)
        y_te = df_te[target_col].values
        test_auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        click.echo(f"\nHoldout Test AUC: {test_auc:.4f}  (n={len(df_te):,} entries)")

    # --- H-1: 配信モデルは全データで再学習（holdout打切りモデルを本番にしない）---
    if full_refit:
        click.echo(f"[full-refit] 全データ {df_train['race_key'].nunique():,} races "
                   f"で配信用モデルを再学習 ...")
        model = train_lgbm(df_train, feature_cols=FEATURE_COLS_WT, target_col=target_col)

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
        "target": target_col,
        "full_refit": bool(full_refit),
        "from": from_date,
        "to": to_date,
        "test_from": test_from,
        "test_to": test_to,
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
@click.option("--start-from-hour", "start_from_hour", default=None, type=int,
              help="JST発走時がこの時(h)以降のレースのみ推奨対象（夜の部の再生成用）。例: 19")
@click.option("--start-to-hour", "start_to_hour", default=None, type=int,
              help="JST発走時がこの時(h)未満のレースのみ推奨対象（朝の部=昼〜夕用）。例: 19")
@click.option("--min-gap12", "min_gap12", default=0.07, show_default=True, type=float,
              help="A層の最低 gap12（pred1-pred2）閾値。この値未満はスキップ。"
                   "0.07: doc46 で最安定フィルタ（VAL 9/12ヶ月黒字・HOLD 196%）")
@click.option("--include-7plus/--no-include-7plus", "include_7plus", default=True,
              help="7車以上レースを対象に追加（gami≥GAMI_THRESHOLD倍+gap12≥min_gap12）。"
                   "doc48 Phase2通過: VAL 129.9%★(3143R)/HOLD 138.3%★(1381R)/12.93R/日。"
                   "既定on＝7+車専用本番モード。")
@click.option("--7plus-s-gap12", "seven_plus_s_gap12", default=0.10, show_default=True, type=float,
              help="7+車 Sランク閾値: gap12がこの値以上をSランク、未満をAランク（default: 0.10=HOLD143%）")
def wave_picks_wt(target_date, output_path, model_name,
                  start_from_hour, start_to_hour, min_gap12, include_7plus,
                  seven_plus_s_gap12):
    """winticket モデルで wave-picks を生成（7+車 SS=三連複 専用）

    2026-07-10 doc52 以降は 7車専用モード（SS=7PLUS_R）のみ。
    旧≤6車 SS/S/A/B・ワイドロジックは出力に使われないデッドコードだったため削除済み。
    S/S+（三連単F 7PLUS_ST/STP）は優位性なしのため 2026-07-15 に全廃。

    例:
        python -m src.cli.main wave-picks-wt
        python -m src.cli.main wave-picks-wt --date 2026-07-12 --start-to-hour 19
    """
    import json
    import re
    import pandas as pd
    from datetime import datetime, timezone, timedelta
    from src.preprocessing.feature_wt import (
        load_raw_data_wt, build_features_wt, FEATURE_COLS_WT, prepare_X,
    )
    from src.models.trainer import load_model
    from src.database import get_connection
    from src.strategy_wt import (
        M_GAP12_MIN, M_RATIO_MAX, M_WIN_RANK_MIN, S1W_TOP3_GAP_MIN, U_ENTROPY_MIN,
        line_score_features, m_axis_gate, race_signals, s1w_gate, s1w_select,
        ss_policy, u_entropy,
    )
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

    def _market_fav_frame(odds: dict) -> int | None:
        """trio盤面から市場の本命(implied P(top3)最大の車)を返す。盤面不足はNone。

        q_i = Σ_{iを含むtrio組} 1/odds（placeholder≥9000は除外）。
        モデル1位と市場本命の不一致(fav_mismatch)はOOSでROI 1168/576%の頑健レバー
        （docs/analysis/13）。タグとして記録しlive前向き検証する（挙動は変えない）。
        """
        q: dict[int, float] = {}
        n_combo = 0
        for item in odds.get("trio", []):
            ov = item["odds_value"]
            if ov is None or ov <= 0 or ov >= 9000:
                continue
            parts = re.split(r"[-=]", str(item["combination"]))
            try:
                frs = [int(x) for x in parts]
            except ValueError:
                continue
            if len(frs) != 3:
                continue
            n_combo += 1
            for fno in frs:
                q[fno] = q.get(fno, 0.0) + 1.0 / ov
        if n_combo < 4 or not q:
            return None
        return max(q, key=lambda k: q[k])

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
    X = prepare_X(df)
    df["pred_prob"] = model.predict_proba(X)[:, 1]

    # 1着モデル（Phase B・2026-07-19〜）。M候補のwin_rankゲートに使う。
    # 存在しなければ None のままにし、M候補生成側で gap12 単独ゲートにフォールバックする。
    try:
        win_model = load_model("lgbm_wt_win")
        df["pred_win"] = win_model.predict_proba(X)[:, 1]
    except FileNotFoundError:
        df["pred_win"] = None
        click.echo("[wt] lgbm_wt_win が見つかりません。M候補は gap12 単独ゲートで生成します。",
                   err=True)

    # Web表示用の単勝/複勝指数を wt_entries に書き込む（2026-07-19）。
    # 候補選定と無関係に全出走馬分を更新するため、この位置（pred_prob/pred_win
    # 算出直後・候補フィルタ前）で行う。
    with get_connection() as _conn_idx:
        _idx_rows = [
            (
                round(float(row.pred_win) * 100, 1) if pd.notna(row.pred_win) else None,
                round(float(row.pred_prob) * 100, 1) if pd.notna(row.pred_prob) else None,
                row.race_key, int(row.frame_no),
            )
            for row in df.itertuples(index=False)
        ]
        _conn_idx.executemany(
            "UPDATE wt_entries SET pred_win_pct = ?, pred_top3_pct = ? "
            "WHERE race_key = ? AND frame_no = ?",
            _idx_rows,
        )
        _conn_idx.commit()

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

    def _hour_of(g):
        """レースのJST発走時(h)。不明は None。"""
        s = g["start_time"].iloc[0]
        try:
            return int(str(s).split(":")[0])
        except (ValueError, IndexError):
            return None

    def _hour_skip(hh):
        """2段階生成のJST時刻フィルタ。Trueなら対象外（hh不明は朝の部に含める=to側のみ判定）。"""
        if start_to_hour is not None and hh is not None and hh >= start_to_hour:
            return True
        if start_from_hour is not None and (hh is None or hh < start_from_hour):
            return True
        return False

    # 7+車 Rランク（doc52・2026-07-10 SS/S置き換え）
    # レース単位セマンティクス: min(全目)≥GAMI_THRESHOLD ∧ gap12≥seven_plus_s_gap12 ∧ gap23≥1pt
    # → 全目購入（カットなし・SOフィルタなし）。的中条件=軸2車が3着内。
    # 検証: 2025通年 的中率29.3%・ROI147.6%（真OOS 11月120%/12月140%/2026-06 299%）
    plus7_candidates = []   # gap12≥min_gap12のみ（gamiフィルタなし・prerace用）
    plus7_r_races = []      # SSランク（三連複・レース単位gami）
    skipped_7plus_gami = 0
    skipped_7plus_policy = 0  # 選抜見送り件数（4分戦カット・格差増額は2026-07-16廃止）
    if include_7plus:
        with get_connection() as conn7:
            n_entries_map = dict(conn7.execute(
                "SELECT race_key, n_entries FROM wt_races WHERE race_date=?",
                (target_date,)
            ).fetchall())
            race_type_map = dict(conn7.execute(
                "SELECT race_key, race_type FROM wt_races WHERE race_date=?",
                (target_date,)
            ).fetchall())

        for race_key, grp in df.groupby("race_key"):
            n_ent = n_entries_map.get(race_key, 0)
            # 7車ちょうど限定（8/9車はROI構造的に不利。write_candidates_wt/notify_prerace_wt と同一基準。
            # 上限なしだと朝通知にだけ8/9車が乗り、直前判定・採点からは除外される非対称が生じる）
            if n_ent != 7:
                continue
            grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
            if len(grp_sorted) < 3:
                continue
            if _hour_skip(_hour_of(grp_sorted)):
                continue

            p = grp_sorted["pred_prob"].tolist()
            gap12_7 = p[0] - p[1]
            if gap12_7 < min_gap12:
                continue

            frames = grp_sorted["frame_no"].astype(int).tolist()
            pivot1_7, pivot2_7 = frames[0], frames[1]
            thirds_7 = frames[2:]

            # per-combo odds map（SSランクとS/A共用）
            odds7 = _load_odds(race_key)
            target_sets_7 = {frozenset({pivot1_7, pivot2_7, t}) for t in thirds_7}
            combo_odds_map = {}
            for item in odds7.get("trio", []):
                ov = item["odds_value"]
                if ov is None or ov <= 0 or ov >= 9000:
                    continue
                parts7 = re.split(r"[-=]", str(item["combination"]))
                try:
                    cs = frozenset(int(x) for x in parts7)
                except ValueError:
                    continue
                if cs in target_sets_7:
                    combo_odds_map[cs] = float(ov)
            gami_7 = min(combo_odds_map.values()) if combo_odds_map else 0.0

            try:
                mkt_fav7 = _market_fav_frame(odds7)
            except Exception:
                mkt_fav7 = None

            venue_id7 = grp_sorted["venue_id"].iloc[0]
            venue_name7 = _venue_name(venue_map, venue_id7)
            race_no7 = int(grp_sorted["race_no"].iloc[0])
            start_time7 = grp_sorted["start_time"].iloc[0]

            riders_detail7 = []
            for rank_idx7, row7 in enumerate(grp_sorted.itertuples(index=False)):
                fn7 = int(row7.frame_no)
                role7 = "軸1" if rank_idx7 == 0 else "軸2" if rank_idx7 == 1 else "流し" if rank_idx7 <= 4 else "-"
                pc7 = row7.player_class if isinstance(row7.player_class, str) else ""
                lp7 = row7.style if isinstance(getattr(row7, "style", None), str) else ""
                pv7 = getattr(row7, "term", None)
                rp7 = row7.race_point
                wr7 = row7.first_rate
                riders_detail7.append({
                    "frame_no":      fn7,
                    "ai_rank":       rank_idx7 + 1,
                    "player_class":  pc7,
                    "period":        int(pv7) if pv7 is not None and pv7 == pv7 else 0,
                    "racing_score":  round(float(rp7), 1) if rp7 == rp7 else 0.0,
                    "win_rate_3m":   round(float(wr7), 1) if wr7 == wr7 else 0.0,
                    "line_position": lp7,
                    "pred_prob_pct": round(float(row7.pred_prob) * 100, 1),
                    "role":          role7,
                })

            sig7 = race_signals(p, int(n_ent))

            # ライン構造特徴 + レース種別（ポリシー=選抜カットのみ。ライン特徴は分析用に記録継続）
            race_type7 = race_type_map.get(race_key)
            _line_pairs7 = [
                (None if pd.isna(_r.line_group) else int(_r.line_group),
                 None if pd.isna(_r.race_point) else float(_r.race_point))
                for _r in grp_sorted.itertuples(index=False)
            ]
            line_avg_gap7, line_n_lines7, line_all_solo7 = line_score_features(_line_pairs7)

            # 候補（gamiフィルタなし・発走前再検証用）
            plus7_candidates.append({
                "rank":          "7PLUS_CAND",
                "race_key":      race_key,
                "venue_name":    venue_name7,
                "race_no":       race_no7,
                "start_time":    start_time7,
                "n_riders":      int(n_ent),
                "gap12":         float(gap12_7),
                "ratio":         float(p[0] / (3 / n_ent)) if n_ent else 0.0,
                "pivot1":        int(pivot1_7),
                "pivot2":        int(pivot2_7),
                "thirds":        [int(t) for t in thirds_7],
                "riders":        riders_detail7,
                "top3_sum":      round(float(sig7["top3_sum"]), 4),
                "upset_tier":    sig7["upset_tier"],
                "bet_type":      "3連複",
                "min_trio_odds": round(float(gami_7), 2) if gami_7 > 0 else None,
                "gami_rank":     None,  # loop後に plus7_r_races との照合で上書き
                # doc53 統合ポリシー用コンテキスト（notify_prerace_wt が参照）
                "race_type":     race_type7,
                "line_avg_gap":  line_avg_gap7,
                "line_n_lines":  line_n_lines7,
                "line_all_solo": line_all_solo7,
            })

            # SSランク: レース単位除外セマンティクス (doc52・2026-07-10)
            # min(全目)≥GAMI_THRESHOLD ∧ gap12≥seven_plus_s_gap12 ∧ gap23≥1pt → 全目購入
            gap23_pt7 = (p[1] - p[2]) * 100.0 if len(p) >= 3 else 0.0
            if gami_7 < GAMI_THRESHOLD:
                skipped_7plus_gami += 1
                continue
            if gap12_7 < seven_plus_s_gap12 or gap23_pt7 < 1.0:
                continue

            # ポリシー: 選抜レースのみ見送り（4分戦カット・格差増額は2026-07-16廃止）
            ss_skip7, ss_stake7 = ss_policy(race_type7)
            if ss_skip7:
                skipped_7plus_policy += 1
                continue

            thirds_str7 = ",".join(str(t) for t in thirds_7)
            n_pts7 = len(thirds_7)
            plus7_r_races.append({
                "race_key":    race_key,
                "venue_name":  venue_name7,
                "race_no":     race_no7,
                "start_time":  start_time7,
                "n_riders":    int(n_ent),
                "gap12":       float(gap12_7),
                "ratio":       float(p[0] / (3 / n_ent)) if n_ent else 0.0,
                "pivot1":      int(pivot1_7),
                "pivot2":      int(pivot2_7),
                "thirds":      [int(t) for t in thirds_7],
                "riders":      riders_detail7,
                "odds_label":  f"min{gami_7:.1f}倍",
                "top3_sum":    round(float(sig7["top3_sum"]), 4),
                "upset_tier":  sig7["upset_tier"],
                "market_fav":  int(mkt_fav7) if mkt_fav7 is not None else None,
                "fav_mismatch": bool(mkt_fav7 is not None and mkt_fav7 != pivot1_7),
                "stake":       int(n_pts7 * ss_stake7),
                "stake_per_pt": int(ss_stake7),
                "n_points":    int(n_pts7),
                "combo_str":   f"{pivot1_7}-{pivot2_7}-{thirds_str7}",
                "bet_type":    "3連複",
            })

    # ── 旧S1(7PLUS_R・7車三連複)は 2026-07-16 全廃 ─────────────────────────────
    # 候補生成コードは上に残置し、ここで出力を無効化する（新S1=6車三連単へ置換）。
    # candidates JSON は空リストを書き出す（ファイル存在契約を維持し、
    # notify_prerace_wt / write_candidates_wt の読み込みを壊さない）。
    plus7_r_races = []
    plus7_candidates = []
    click.echo("旧S1(7PLUS_R)は2026-07-16全廃（候補・推奨は出力しません）。", err=True)

    sort_key = lambda x: (x["start_time"] == "--:--", x["start_time"], x["venue_name"], x["race_no"])
    plus7_r_races.sort(key=sort_key)

    def _fmt(entry):
        n_str = f"{entry['n_riders']}車"
        odds_str = f"  [{entry['odds_label']}]" if entry.get("odds_label") else ""
        base = f"(元{entry['base_rank']}) " if entry.get("base_rank") else ""
        npts = int(entry.get("n_points", 3))
        stk = int(entry.get("stake", 300))
        return (
            f"  {entry['start_time']}  {entry['venue_name']:<6} {entry['race_no']:>2}R  "
            f"[{n_str}]  {base}{entry['bet_type']}: {entry['combo_str']}  ({npts}点/{stk:,}円){odds_str}"
        )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("=" * 70)
    lines.append(f" 競輪AI予想PICK [wt]  {target_date}")
    lines.append(f" モデル: {model_name}  生成: {now_str}")
    lines.append(" 現行ランク: S1(win軸1着固定)・S2(波乱ライン連れ込み)・S3(不一致×gap12≥0.10) の3ペーパー。")
    lines.append("   → 候補は s1_candidates.json / u_candidates.json / m_candidates.json、")
    lines.append("     発走前判定・通知は notify_prerace_wt.py 参照")
    lines.append("=" * 70)

    # 旧S1(7PLUS_R)の「【7+車 SSランク】」txtセクションは 2026-07-16 全廃により出力しない
    # （notify_results_wt._parse_picks_full は過去日 txt の後方互換のためパース処理を残置）。

    lines.append("=" * 70)
    lines.append("  実賭け推奨なし（S1/S2/S3 は全てペーパー検証・旧新S1/A は 2026-07-17 全廃）")
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
    all_race_details = [{"rank": "7PLUS_R", **e} for e in plus7_r_races]
    detail_path = Path(output_path).parent / f"wave_picks_wt_{target_date}_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_race_details, f, ensure_ascii=False, indent=2)
    click.echo(f"[保存先] {detail_path}")

    # candidates に gami_rank を付与（SS に入ったレースかを朝通知で表示するため）
    _r_keys = {r["race_key"] for r in plus7_r_races}
    for _cand in plus7_candidates:
        if _cand["race_key"] in _r_keys:
            _cand["gami_rank"] = "SS"

    # 候補JSON（gamiフィルタなし・gap12≥min_gap12のみ。notify_prerace_wt.py が発走前再検証に使用）
    # 夜run（output_path が _night.txt）は _night_candidates.json に書き、朝分を上書きしない。
    out_stem = Path(output_path).stem
    cands_suffix = "_night_candidates.json" if out_stem.endswith("_night") else "_candidates.json"
    cands_path = Path(output_path).parent / f"wave_picks_wt_{target_date}{cands_suffix}"
    with open(cands_path, "w", encoding="utf-8") as f:
        json.dump(plus7_candidates, f, ensure_ascii=False, indent=2)
    click.echo(f"[保存先] {cands_path}  (旧S1全廃につき空リスト固定・ファイル存在契約のみ維持)")

    # 全レース指数 JSON（全レース。推奨レースは rank/買い目を付与）。
    # notify_picks.py がこれを読み「全レース指数PDF」を朝のDiscordに添付する。
    rec_by_key = {}
    for rk_, ent in [("7PLUS_R", e) for e in plus7_r_races]:
        rec_by_key.setdefault(ent["race_key"], (rk_, ent))

    all_index = []
    for race_key, grp in df.groupby("race_key"):
        grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        n_riders = len(grp_sorted)
        if n_riders < 2:
            continue
        p = grp_sorted["pred_prob"].tolist()
        sig = race_signals(p, n_riders)
        riders_detail = []
        for rank_idx, row in enumerate(grp_sorted.itertuples(index=False)):
            rp = row.race_point
            wr = row.first_rate
            pv = getattr(row, "term", None)
            riders_detail.append({
                "frame_no":      int(row.frame_no),
                "ai_rank":       rank_idx + 1,
                "player_class":  row.player_class if isinstance(row.player_class, str) else "",
                "period":        int(pv) if pv is not None and pv == pv else 0,
                "racing_score":  round(float(rp), 1) if rp == rp else 0.0,
                "win_rate_3m":   round(float(wr), 1) if wr == wr else 0.0,
                "line_position": row.style if isinstance(getattr(row, "style", None), str) else "",
                "pred_prob_pct": round(float(row.pred_prob) * 100, 1),
                "role":          "軸1" if rank_idx == 0 else "軸2" if rank_idx == 1 else "流し" if rank_idx <= 4 else "-",
            })
        rec = rec_by_key.get(race_key)
        if rec:
            rank, ent = rec
            bet_type, combo_str = ent.get("bet_type", ""), ent.get("combo_str", "")
        else:
            rank, bet_type, combo_str = "-", "指数のみ", "(参考)"
        all_index.append({
            "race_key":   race_key,
            "rank":       rank,
            "venue_name": _venue_name(venue_map, grp_sorted["venue_id"].iloc[0]),
            "race_no":    int(grp_sorted["race_no"].iloc[0]),
            "start_time": grp_sorted["start_time"].iloc[0],
            "n_riders":   int(n_riders),
            "gap12":      float(sig["gap12"]),
            "ratio":      float(sig["ratio"]),
            "top3_sum":   round(float(sig["top3_sum"]), 4),
            "upset_tier": sig["upset_tier"],
            "bet_type":   bet_type,
            "combo_str":  combo_str,
            "riders":     riders_detail,
        })
    from src.database import get_connection
    _score_updates = [
        (rider["pred_prob_pct"], item["race_key"], rider["frame_no"])
        for item in all_index
        for rider in item["riders"]
    ]
    if _score_updates:
        with get_connection() as _conn:
            _conn.executemany(
                "UPDATE wt_entries SET race_point = ? WHERE race_key = ? AND frame_no = ?",
                _score_updates,
            )
            _conn.commit()

    all_index.sort(key=lambda x: (x["start_time"] == "--:--", x["start_time"], x["venue_name"], x["race_no"]))
    allindex_path = Path(output_path).parent / f"wave_picks_wt_{target_date}_allindex.json"
    with open(allindex_path, "w", encoding="utf-8") as f:
        json.dump(all_index, f, ensure_ascii=False, indent=2)
    click.echo(f"[保存先] {allindex_path}  (全{len(all_index)}レース指数)")

    # ── U候補（波乱ライン連れ込み・ペーパートレード検証 2026-07-16〜）──────────
    # 朝はモデル情報のみで候補を抽出する（オッズ判定は発走15分前に
    # notify_prerace_wt.judge_u が盤面7車・mto・市場順位・買い目オッズで確定）。
    #   対象: 7車ちょうど ∧ u_entropy(全7車 pred_prob) >= U_ENTROPY_MIN
    #   ペア候補: 穴=モデル3位内 ∧ (単騎 or ライン先頭/番手)、
    #             相方=同ライン ∧ 脚質「逃」∧ 穴と別人
    if include_7plus:
        u_candidates = []
        for race_key, grp in df.groupby("race_key"):
            if n_entries_map.get(race_key, 0) != 7:
                continue
            grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
            if len(grp_sorted) != 7:
                continue
            if _hour_skip(_hour_of(grp_sorted)):
                continue
            ent = u_entropy([float(x) for x in grp_sorted["pred_prob"].tolist()])
            if ent < U_ENTROPY_MIN:
                continue

            def _u_int(v):
                return None if v is None or pd.isna(v) else int(v)

            rows_u = list(grp_sorted.itertuples(index=False))
            pairs_u = []
            for rank_idx_u, r_u in enumerate(rows_u[:3], start=1):  # モデル順位3位以内
                lg_u = _u_int(getattr(r_u, "line_group", None))
                ls_u = _u_int(getattr(r_u, "line_size", None))
                lp_u = _u_int(getattr(r_u, "line_pos", None))
                if not (ls_u == 1 or lp_u in (1, 2)):
                    continue
                if lg_u is None:
                    continue  # ラインなし → 同ラインの相方なし
                dark_fno = int(r_u.frame_no)
                for m_u in rows_u:
                    m_fno = int(m_u.frame_no)
                    m_lg = _u_int(getattr(m_u, "line_group", None))
                    m_style = m_u.style if isinstance(getattr(m_u, "style", None), str) else ""
                    if m_fno == dark_fno or m_lg is None or m_lg != lg_u or m_style != "逃":
                        continue
                    pairs_u.append({
                        "dark": dark_fno,
                        "dark_model_rank": rank_idx_u,
                        "mate": m_fno,
                    })
            if not pairs_u:
                continue
            u_candidates.append({
                "race_key":   race_key,
                "venue_name": _venue_name(venue_map, grp_sorted["venue_id"].iloc[0]),
                "race_no":    int(grp_sorted["race_no"].iloc[0]),
                "start_time": grp_sorted["start_time"].iloc[0],
                "entropy":    round(float(ent), 4),
                "pairs":      pairs_u,
            })

        u_suffix = "_night_u_candidates.json" if out_stem.endswith("_night") else "_u_candidates.json"
        u_path = Path(output_path).parent / f"wave_picks_wt_{target_date}{u_suffix}"
        with open(u_path, "w", encoding="utf-8") as f:
            json.dump(u_candidates, f, ensure_ascii=False, indent=2)
        click.echo(f"[保存先] {u_path}  (U候補 {len(u_candidates)}件・波乱ライン連れ込み/ペーパー検証)")

    # ── M候補（S3・◎不一致×軸信頼ゲート(gap12 OR win_rank OR ratio)・2026-07-19 3way OR拡張）─
    # WT◎（prediction_mark==1）とシステム◎（モデル指数1位）が不一致で、以下いずれか:
    #   (a) gap12 >= M_GAP12_MIN（3着内モデルの軸信頼ゲート）
    #   (b) システム◎の1着モデル内レース順位 >= M_WIN_RANK_MIN（勝ちきれない評価ほど
    #       ROIが上がる逆説的シグナル・exp_win_axis_sweep_wt.py で発見・ほぼ独立）
    #   (c) システム◎の p_win/p_top3 比 <= M_RATIO_MAX（(b)の連続量版・
    #       exp_composite_prob_diff_wt.py で発見・母数を更に+22〜26%拡張）
    # を満たすレースで、システム◎と同ライン「逃」相方を2車軸にする。
    # オッズ判定（盤面7車・15倍以上の目・U重複排除）は発走15分前に
    # notify_prerace_wt.judge_m が確定する。市場順位条件はなし（Uとの相違点）。
    if include_7plus:
        m_candidates = []
        # prediction_mark が df に無い場合のフォールバック（wt_entries から取得）
        pm_fallback = None
        if "prediction_mark" not in df.columns:
            pm_fallback = {}
            with get_connection() as conn_m:
                for _rk_m, _fno_m, _pm_m in conn_m.execute(
                    "SELECT e.race_key, e.frame_no, e.prediction_mark "
                    "FROM wt_entries e JOIN wt_races r ON e.race_key = r.race_key "
                    "WHERE r.race_date = ?", (target_date,)
                ).fetchall():
                    if _pm_m is not None:
                        pm_fallback.setdefault(_rk_m, {})[int(_fno_m)] = int(_pm_m)

        for race_key, grp in df.groupby("race_key"):
            if n_entries_map.get(race_key, 0) != 7:
                continue
            grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
            if len(grp_sorted) != 7:
                continue
            if _hour_skip(_hour_of(grp_sorted)):
                continue

            def _m_int(v):
                return None if v is None or pd.isna(v) else int(v)

            rows_m = list(grp_sorted.itertuples(index=False))

            # WT◎（prediction_mark==1）の存在確認
            if pm_fallback is None:
                wt_marks = [int(r_m.frame_no) for r_m in rows_m
                            if _m_int(getattr(r_m, "prediction_mark", None)) == 1]
            else:
                wt_marks = [fno for fno, pm_v in pm_fallback.get(race_key, {}).items()
                            if pm_v == 1]
            if not wt_marks:
                continue
            wt_fno = min(wt_marks)

            # システム◎ = pred_prob 1位。WT◎と不一致のレースのみ対象
            r1_m = rows_m[0]
            m1_fno = int(r1_m.frame_no)
            if m1_fno == wt_fno:
                continue

            p_m = grp_sorted["pred_prob"].tolist()
            gap12_m = float(p_m[0] - p_m[1])

            # win_rank: システム◎(m1_fno) の1着モデル内レース順位（1着モデル未ロード時は None）
            # ratio: システム◎の p_win/p_top3 比（1着モデル未ロード時は None）
            win_rank_m = None
            ratio_m = None
            if "pred_win" in grp_sorted.columns and grp_sorted["pred_win"].notna().all():
                grp_by_win = grp_sorted.sort_values("pred_win", ascending=False).reset_index(drop=True)
                win_order = [int(f) for f in grp_by_win["frame_no"].tolist()]
                win_rank_m = win_order.index(m1_fno) + 1
                p_win_axis = float(r1_m.pred_win)
                p_top3_axis = float(r1_m.pred_prob)
                if p_top3_axis > 0:
                    ratio_m = p_win_axis / p_top3_axis

            gate_ok, gate_label = m_axis_gate(gap12_m, win_rank_m, ratio_m)  # 軸信頼ゲート（2026-07-19 3way OR拡張）
            if not gate_ok:
                continue

            # 相方 = システム◎と同 line_group ∧ 脚質「逃」∧ 別人。
            # 複数該当は line_pos 相補（◎が番手なら先頭/先頭なら番手）を優先、
            # 無ければ車番最小の同ライン逃。
            lg1_m = _m_int(getattr(r1_m, "line_group", None))
            if lg1_m is None:
                continue  # ラインなし → 同ラインの相方なし
            lp1_m = _m_int(getattr(r1_m, "line_pos", None))
            want_lp_m = 1 if lp1_m == 2 else 2
            mates_m = []
            for r_m in rows_m:
                fno_m = int(r_m.frame_no)
                lg_m = _m_int(getattr(r_m, "line_group", None))
                style_m = r_m.style if isinstance(getattr(r_m, "style", None), str) else ""
                if fno_m == m1_fno or lg_m is None or lg_m != lg1_m or style_m != "逃":
                    continue
                mates_m.append((fno_m, _m_int(getattr(r_m, "line_pos", None))))
            if not mates_m:
                continue
            mates_m.sort()  # 車番昇順（フォールバック時の決定性）
            mate_fno = next((f for f, lp in mates_m if lp == want_lp_m), mates_m[0][0])

            m_candidates.append({
                "race_key":   race_key,
                "venue_name": _venue_name(venue_map, grp_sorted["venue_id"].iloc[0]),
                "race_no":    int(grp_sorted["race_no"].iloc[0]),
                "start_time": grp_sorted["start_time"].iloc[0],
                "gap12":      round(gap12_m, 4),
                "win_rank":   win_rank_m,
                "ratio":      round(ratio_m, 4) if ratio_m is not None else None,
                "gate":       gate_label,
                "pair":       {"m1": m1_fno, "mate": mate_fno},
                "wt_mark":    wt_fno,
            })

        m_suffix = "_night_m_candidates.json" if out_stem.endswith("_night") else "_m_candidates.json"
        m_path = Path(output_path).parent / f"wave_picks_wt_{target_date}{m_suffix}"
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(m_candidates, f, ensure_ascii=False, indent=2)
        click.echo(f"[保存先] {m_path}  (M候補 {len(m_candidates)}件・◎不一致×システム◎/ペーパー検証)")

    # ── S1候補（新設計・win軸1着固定×3着内モデル相手2車・三連単2点流し・2026-07-19導入）──
    # WT◎/システム◎の一致・不一致は問わない。7車全レース対象。
    #   軸 = win model(lgbm_wt_win) レース内1位
    #   相手 = 3着内モデルで軸を除いた残り車の上位2頭(p1,p2)
    #   ゲート: top3_gap(p1-p2の3着内確率差) >= S1W_TOP3_GAP_MIN
    #   買い目: 三連単 軸→p1→p2, 軸→p2→p1 の2点流し（目オッズ下限なし）
    if include_7plus:
        s1_candidates = []
        if "pred_win" in df.columns:
            for race_key, grp in df.groupby("race_key"):
                if n_entries_map.get(race_key, 0) != 7:
                    continue
                grp_sorted = grp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
                if len(grp_sorted) != 7 or grp_sorted["pred_win"].isna().any():
                    continue
                if _hour_skip(_hour_of(grp_sorted)):
                    continue
                win_probs = {int(r.frame_no): float(r.pred_win)
                             for r in grp_sorted.itertuples(index=False)}
                top3_probs = {int(r.frame_no): float(r.pred_prob)
                              for r in grp_sorted.itertuples(index=False)}
                sel = s1w_select(win_probs, top3_probs)
                if sel is None:
                    continue
                axis, p1, p2, top3_gap = sel
                if not s1w_gate(top3_gap):
                    continue
                s1_candidates.append({
                    "race_key":   race_key,
                    "venue_name": _venue_name(venue_map, grp_sorted["venue_id"].iloc[0]),
                    "race_no":    int(grp_sorted["race_no"].iloc[0]),
                    "start_time": grp_sorted["start_time"].iloc[0],
                    "top3_gap":   round(top3_gap, 4),
                    "axis": axis, "p1": p1, "p2": p2,
                })
        else:
            click.echo("[wt] lgbm_wt_win が見つかりません。S1候補は生成しません。", err=True)

        s1_suffix = "_night_s1_candidates.json" if out_stem.endswith("_night") else "_s1_candidates.json"
        s1_path = Path(output_path).parent / f"wave_picks_wt_{target_date}{s1_suffix}"
        with open(s1_path, "w", encoding="utf-8") as f:
            json.dump(s1_candidates, f, ensure_ascii=False, indent=2)
        click.echo(f"[保存先] {s1_path}  (S1候補 {len(s1_candidates)}件・win軸1着固定/ペーパー検証)")

    # ── A候補（◎一致×波乱×別L先頭・二連単）・旧S1候補（6車三連単）は 2026-07-17 全廃 ──
    # 正規プロトコル（学習〜2025-03／検証2025-04〜2026-03の1年／テスト2026-04〜）の
    # 再検証で両者とも検証ROI100%超なし → 候補生成を停止（src/strategy_wt.py 参照）。
    # 現行のペーパーランクは S2(7PLUS_U)・S3(7PLUS_M)・S1(SEVEN_S1) の3つ。


@cli.command("backtest-wt")
@click.option("--from", "from_date", default="2025-01-01", help="評価開始日")
@click.option("--to", "to_date", default=None, help="評価終了日")
@click.option("--model", "model_name", default="lgbm_wt",
              help="モデルファイル名（.pklなし）。"
                   "注意: デフォルトの lgbm_wt は週次再学習済みで評価期間をin-sampleで学習している。"
                   "リーク無し検証には --eval-model オプションで期間限定学習モデルを指定すること"
                   "（docs/analysis/18-backtest-bias-rescore.md バイアス③参照）。")
@click.option("--eval-model", "eval_model_name", default=None,
              help="評価専用モデルのファイル名（.pklなし）。"
                   "指定すると --model の代わりにこのモデルで予測確率を計算する。"
                   "週次再学習 lgbm_wt のリークを避けるため、"
                   "TRAIN期間のみで学習したモデル（例: lgbm_wt_train_only）を指定すると"
                   "doc18セマンティクスのリーク無し評価ができる。")
@click.option("--max-riders", "max_riders", default=None, type=int,
              help="出走頭数フィルター（実運用は6）。出走表基準で適用する。")
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
                eval_model_name: str | None,
                max_riders: int | None, min_gap12: float | None, tiered: bool,
                value_mode: bool, ev_min: float, max_per_race: int,
                max_ratio: float | None):
    """winticket モデルで買い目バックテストを実行（wt_odds の実オッズ使用）

    [doc18 本番忠実セマンティクス適用済み]
    - 出走表基準の ≤6車フィルタ（完走者基準ではない）
    - 全エントリーでランキング（欠車を事前に知らない）
    - 欠車処理: 軸欠車=レース無効 / 相手欠車=その目のみ除外

    週次再学習済み lgbm_wt をデフォルトモデルとして使う場合は評価期間内にリークがある。
    リーク無し評価には --eval-model でTRAIN期間限定学習モデルを指定すること。

    例: python -m src.cli.main backtest-wt --from 2026-01-01
        python -m src.cli.main backtest-wt --from 2026-01-01 --max-riders 6 --min-gap12 0.06
        python -m src.cli.main backtest-wt --from 2026-01-01 --tiered
        python -m src.cli.main backtest-wt --from 2025-07-01 --tiered --eval-model lgbm_wt_train_only
    """
    from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
    from src.models.trainer import load_model
    from src.evaluation.backtest_wt import (
        run_backtest_wt, print_backtest_wt,
        run_tiered_backtest_wt, print_tiered_backtest_wt,
        run_value_backtest_wt, print_value_backtest_wt,
    )

    # --eval-model が指定されている場合はそちらを使う（リーク無し評価用）
    active_model_name = eval_model_name if eval_model_name else model_name
    try:
        model = load_model(active_model_name)
    except FileNotFoundError:
        click.echo(f"モデル '{active_model_name}' が見つかりません。先に train-wt を実行してください。",
                   err=True)
        raise SystemExit(1)

    if eval_model_name:
        click.echo(f"[wt] 評価モデル: {eval_model_name} (リーク無し専用モデル)")
    elif active_model_name == "lgbm_wt":
        click.echo(f"[wt] 警告: lgbm_wt は週次再学習済みで評価期間をin-sampleで学習しています。"
                   f" リーク上振れに注意（doc18 バイアス③）。リーク無し評価には --eval-model を使用してください。")

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
