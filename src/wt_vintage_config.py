"""wt系（競輪）honest walk-forward検証で使う凍結vintageモデルの期間定義 単一の
正本（2026-07-29・[[keirin_s7_foundational_rethink_2026_07_29]]）。

背景: 従来はrebuild_s1/s7/s9/s7a/s9a_walkforward_pg.py・backfill_index_pct_wt.py
の6ファイルにQUARTERSという同一内容の定数がコピーされており、将来どれか1つだけ
更新されて食い違うリスクを常に抱えていた（監査で実際に非本番exp_*.pyスクリプトで
食い違いが発生済みと判明）。加えて、静的な"TAIL_FROM"定数と週次で可動する
`lgbm_wt_eval`のtest_fromが乖離し続けるバグ（2週間分がリーク区間化）も発覚した。

新設計: 月単位の凍結vintageモデルのみを使い、「直近の未確定tail」という概念を
撤廃する。月Mのモデル(lgbm_wt_eval_mYYMM/lgbm_wt_win_mYYMM)は必ず
「2022-12-01 〜 M月の前月末」を学習データとし、M月のレース（進行中の当月を含む）
をスコアリングする用途に固定して使う。当月分がまだ終わっていなくても、
「その月のレースは全て前月末までのデータで学習したモデルでスコアする」という
契約は当月中ずっと不変であり、TAIL_FROMのような別概念のドリフトが構造的に
発生しない。

このモジュールをrebuild_*_walkforward_pg.py・backfill_index_pct_wt.py・
train_monthly_vintage_models.py全てから共通importする。
"""
import os
import re
from calendar import monthrange
from datetime import date, timedelta

BASE_FROM = "2022-12-01"
FIRST_MONTH = (2024, 1)

# vintage モデル名の接尾辞（例: lgbm_wt_eval_m2404）。
# これが付いていないものは「本番モデル」＝全期間を週次で再学習しているもの。
_VINTAGE_SUFFIX = re.compile(r"_m\d{4}$")

#: 過去日に本番モデルを使うことを意図的に許すときの環境変数。
#: 「モデルを差し替えずに素の挙動を見たい」等の調査用の逃げ道で、
#: 常用すると in-sample な数字を本物と取り違えるので普段は設定しないこと。
ALLOW_PRODUCTION_ENV = "KEIRIN_ALLOW_PRODUCTION_MODELS_FOR_PAST"


def _month_tag(y: int, m: int) -> str:
    return f"m{y % 100:02d}{m:02d}"


def bad_model_name(eval_model_name: str) -> str:
    """eval の vintage 名から、対応する大敗モデルの vintage 名を導く。

    例: "lgbm_wt_eval_m2401" → "lgbm_wt_bad_m2401"

    ⚠️ `monthly_windows()` は互換のため 4-tuple のまま（eval/win のみ）にしてある。
    大敗モデルは 3ヘッド軸選定（2026-08-04〜）でのみ必要で、全ての rebuild 系が
    要求するわけではないため、必要な呼び出し側だけが本関数で名前を導く設計にした
    （タプルを 5 要素に広げると既存の全 rebuild スクリプトが壊れる）。
    """
    return eval_model_name.replace("lgbm_wt_eval_", "lgbm_wt_bad_", 1)


def favbust_model_name(eval_model_name: str) -> str:
    """eval の vintage 名から、対応するバスト予測モデルの vintage 名を導く。

    例: "lgbm_wt_eval_m2404" → "lgbm_wt_favbust_m2404"

    7H1（穴推奨・本命バスト型）専用。`bad_model_name()` と同じ理由で
    `monthly_windows()` のタプルは広げず、必要な呼び出し側だけが本関数で導く。
    ⚠️ favbust の vintage は **2024-04 以降のみ**存在する（それ以前は学習に
    必要な履歴が足りない）。2024-01〜03 の窓は `--skip-missing-models` で除外される。
    """
    return eval_model_name.replace("lgbm_wt_eval_", "lgbm_wt_favbust_", 1)


def is_vintage_model(name: str) -> bool:
    """モデル名が月次凍結 vintage（`..._mYYMM`）かどうか。"""
    return bool(_VINTAGE_SUFFIX.search(name))


def assert_vintage_for_past(date_to: str, models: dict[str, str],
                            today: date | None = None) -> None:
    """過去日のスコアリングに本番モデルを使っていたら落とす（2026-08-08 追加）。

    書き込み側（`trainer.save_model`）には vintage 名の検査があるのに、
    **読み込み側には対称な防御が無かった**。`backfill_*_rank_wt.py` /
    `build_7h1_candidates.py` の `--eval-model` 等は既定値が本番モデル名
    （`lgbm_wt_eval` 等）なので、過去日を指定しても**エラーにも警告にもならず
    静かに in-sample な数字が出る**。docstring に「vintage を明示すること」と
    書いてあるだけで、機械的には何も止めていなかった。

    判定規則:
      - 対象範囲が **今日を含む/未来** → 本番モデルで正しい（ライブ予想）。素通し。
      - 対象範囲が **すべて過去** → vintage 必須。本番モデル名なら ValueError。

    調査目的でどうしても本番モデルを過去へ当てたいときは環境変数
    `KEIRIN_ALLOW_PRODUCTION_MODELS_FOR_PAST=1` で解除する（警告は出る）。

    Args:
        date_to: 対象範囲の終端（"YYYY-MM-DD"）
        models: {役割名: モデル名} 例 {"eval": "lgbm_wt_eval_m2404", ...}
                値が None のものは「使わない」意味なので検査しない
        today: テスト用の今日日付
    """
    today = today or date.today()
    if date_to >= today.isoformat():
        return  # ライブ予想。本番モデルで正しい
    offenders = {role: name for role, name in models.items()
                 if name and not is_vintage_model(name)}
    if not offenders:
        return
    detail = ", ".join(f"{role}={name}" for role, name in sorted(offenders.items()))
    if os.environ.get(ALLOW_PRODUCTION_ENV):
        import warnings
        warnings.warn(
            f"[vintage] 過去日({date_to})に本番モデルを使用: {detail}。"
            f" {ALLOW_PRODUCTION_ENV} で明示的に許可されているが、得られる数字は"
            " in-sample であり honest な評価には使えない。",
            RuntimeWarning, stacklevel=2)
        return
    raise ValueError(
        f"過去日({date_to})のスコアリングに本番モデルが指定されている: {detail}。\n"
        "本番モデルは全期間を週次で再学習しているため、過去へ当てると"
        "model-vintage look-ahead で in-sample な数字になる。\n"
        "月次凍結 vintage（例: lgbm_wt_eval_m2404）を明示するか、"
        f"調査目的なら {ALLOW_PRODUCTION_ENV}=1 を設定すること。\n"
        "honest な全期間再構築には rebuild_*_walkforward_pg.py を使う"
        "（monthly_windows() が窓ごとの vintage を渡す）。")


def month_bounds(y: int, m: int, upto: date | None = None) -> tuple[str, str]:
    """月(y,m)の(test_from, test_to)を返す。upto指定時は当月分をuptoで打ち切る。"""
    first = date(y, m, 1)
    _, n_days = monthrange(y, m)
    last = date(y, m, n_days)
    if upto is not None and last > upto:
        last = upto
    return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")


def monthly_windows(upto: date | None = None) -> list[tuple[str, str, str, str]]:
    """(date_from, date_to, eval_model_name, win_model_name) のリストを
    2024-01から`upto`（省略時は今日）が属する月まで生成する。

    date_from は全月共通で BASE_FROM（実際の学習開始日は各モデル自身が
    2022-12-01からになるようtrain-wt --from で別途指定するため、ここでの
    date_from/date_toはrebuild系スクリプトが「その月のレース候補をどの範囲で
    読み込むか」に使うものであり、モデル自体の学習範囲とは別概念）。
    """
    if upto is None:
        upto = date.today()
    windows = []
    y, m = FIRST_MONTH
    while (y, m) <= (upto.year, upto.month):
        test_from, test_to = month_bounds(y, m, upto)
        tag = _month_tag(y, m)
        windows.append((test_from, test_to, f"lgbm_wt_eval_{tag}", f"lgbm_wt_win_{tag}"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return windows


def tail_windows(today: date | None = None) -> list[tuple[str, str, str, str]]:
    """`--tail-only` 用の窓（直近1窓）を **当日を含めずに** 返す。

    【なぜ当日を除くか（2026-08-07）】
    rebuild 系は対象期間の picks_history を**一旦全削除してから**再計算した行を
    入れ直す。ところが再構築できるのは結果が確定したレースだけなので、
    まだ発走していない当日分は「削除されたまま戻ってこない」。
    実際 08:40 の `reconcile_walkforward_tail.sh` が当日の推奨行を消し、
    10:00 の `intraday_results_wt.sh`（write_candidates_wt.py）が復元するまで
    **約75分間 Web から推奨が消える**状態になっていた。

    当日は結果が無く再構築のしようがない＝削除は純粋な損失なので、
    tail の窓は前日で打ち切る。月初(1日)なら前月の窓がそのまま返る。

    ⚠️ `monthly_windows()` 側の既定は変えていない。全期間再構築は
       当日を含んだままでよい（結果のあるレースだけが入り直るため実害がなく、
       既定を変えると全 rebuild スクリプトの挙動が一斉に変わる）。
    """
    if today is None:
        today = date.today()
    ws = monthly_windows(upto=today - timedelta(days=1))
    return ws[-1:] if ws else []
