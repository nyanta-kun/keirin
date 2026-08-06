"""7車ランクの「候補」を月次vintageで作り、月単位でディスクへキャッシュする。

## なぜ必要か

7S / 7A / 7SS / 空白1・2 / overlap2 はすべて**同じ候補集合**を条件で切り分けた
ものにすぎない（被覆マップ: memory keirin_7car_coverage_gaps_2026_08_05）。
にもかかわらず解析スクリプトごとに `build_rows` を回すと、1ヶ月あたり約110秒の
特徴量構築を毎回やり直すことになる。12ヶ月の突合と13ヶ月の掃引を1回ずつ
回すだけで50分かかり、**途中で落ちると全部やり直し**になる（2026-08-06 に実際に
2本とも道半ばで kill された）。

そこで**候補の生成だけを共通化してキャッシュ**する。切り分け条件は各スクリプト側。

## 本番との同一性をどう担保しているか

候補は必ず**本番の `build_rows` を通して**取り出す（`rank_7ss_daily_select` を
素通しの spy に差し替えて横取りする）。軸選定・欠車判定・盤面判定・精算規則を
複製しないので、「検証したものと実装したものが違う」事故が起きない。

## キャッシュの約束（過去に踏んだ穴への対策）

- **保存失敗は握り潰さず例外を上げる**。2026-08-05 に parquet + try/except で
  「一度も保存できていないのに動いているつもり」になり、素の実行より54%遅く
  なった事故がある（memory keirin_feature_cache_and_local_db_rejected）。
- **キーに月・eval/win/bad の各モデル名・軸方式を含める**。モデルを差し替えたら
  別キーになるので、古い候補を黙って使う経路が存在しない。
- **期間ごとに持ち、切り出して使い回さない**。特徴量が読み込み範囲に依存する
  列があるため（同メモリ参照）。ここでは月単位で固定している。

DB書き込みなし（読み取りのみ）。
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.backfill_7ss_rank_wt as M  # noqa: E402
from src.wt_vintage_config import bad_model_name, monthly_windows  # noqa: E402

CACHE_DIR = REPO / "data" / "exp_cache" / "wt_candidates"

# キャッシュに残すキー。`trio` は frozenset を鍵にした dict なので dict() 化して持つ
# （pickle 可能）。`others` は list、`actual_top3` は frozenset。
_KEEP = ("race_key", "race_date", "axis1", "axis2", "axis_sum", "entropy",
         "wt_overlap_n", "wt_mark3_overlap_n", "same_line", "others", "actual_top3")


def _cache_path(date_from: str, eval_model: str, win_model: str,
                bad_model: str | None) -> Path:
    axis = bad_model or "twohead"
    return CACHE_DIR / f"cands_{date_from[:7]}_{eval_model}_{win_model}_{axis}.pkl"


def _capture(date_from: str, date_to: str, eval_model: str,
             win_model: str, bad_model: str | None) -> list[dict]:
    """本番 build_rows を通して候補を横取りする（判定ロジックを複製しない）。"""
    captured: list[dict] = []
    original = M.rank_7ss_daily_select

    def spy(cands):
        captured.extend(cands)
        return original(cands)

    M.rank_7ss_daily_select = spy
    try:
        M.build_rows(eval_model, date_from, date_to,
                     win_model_name=win_model, bad_model_name=bad_model)
    finally:
        M.rank_7ss_daily_select = original

    slim = []
    for c in captured:
        row = {k: c[k] for k in _KEEP}
        row["trio"] = dict(c["trio"])
        slim.append(row)
    return slim


def month_candidates(date_from: str, date_to: str, eval_model: str,
                     win_model: str, bad_model: str | None,
                     *, verbose: bool = True) -> list[dict]:
    """1ヶ月分の候補を返す。キャッシュがあれば再計算しない。"""
    path = _cache_path(date_from, eval_model, win_model, bad_model)
    if path.exists():
        with path.open("rb") as f:
            cands = pickle.load(f)
        if verbose:
            print(f"  {date_from[:7]}  [cache] {len(cands):5}件", flush=True)
        return cands

    cands = _capture(date_from, date_to, eval_model, win_model, bad_model)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    # 保存失敗は**例外を上げる**（握り潰すと「効いていないキャッシュ」になる）。
    # tmp へ書いてから rename＝途中で kill されても壊れたキャッシュを残さない。
    with tmp.open("wb") as f:
        pickle.dump(cands, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    if verbose:
        print(f"  {date_from[:7]}  [built] {len(cands):5}件 → {path.name}", flush=True)
    return cands


def iter_months(date_from: str, date_to: str, *, two_head: bool = False,
                verbose: bool = True):
    """指定期間の月次窓を順に回し (月, 候補リスト) を yield する。"""
    windows = [w for w in monthly_windows() if w[0] >= date_from and w[1] <= date_to]
    for win_from, win_to, eval_model, win_model in windows:
        bad_model = None if two_head else bad_model_name(eval_model)
        yield win_from[:7], month_candidates(win_from, win_to, eval_model,
                                             win_model, bad_model, verbose=verbose)


def main() -> None:
    """事前ウォームアップ用。指定期間のキャッシュを作るだけ。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--two-head", action="store_true")
    args = ap.parse_args()
    total = 0
    for _month, cands in iter_months(args.date_from, args.date_to,
                                     two_head=args.two_head):
        total += len(cands)
    print(f"完了: {total} 候補")


if __name__ == "__main__":
    main()
