"""winticket 波乱/非本命ゲート（確定前情報のみ・朝7:00算出可）

3タスク（特徴ablation・波乱予測・オッズ活用）が収束した結論:
「本命が堅いレースは低ROI、本命が割れた=波乱余地のあるレースが高ROI」。

指標 top3_sum = 上位3頭の pred_prob(=P(top3)) 合計。
  小さい = 上位3頭に確率が集中していない = レースが割れている = 波乱余地大。
  大きい = 鉄板 = 低配当。

検証（lgbm_wt・本番3点戦略・TRAIN 2023-07〜2026-02 → TEST 2026-03〜, OOS）:
  TRAIN四分位カット = [1.70, 1.90, 2.08]
  Q1_loose(top3_sum<1.70): TRAIN ROI 1224% / TEST ROI 1136%（最大払戻除外でも934%）
  Q4_chalk(>2.08):         TRAIN ROI  88% / TEST ROI  107%
  単調・train/test一致・volume十分(test 125R=25%)・万車券単発非依存。

注意: ROIは最終データbacktest=実運用上限値（実測は別途 picks_history で検証）。
ゲートは「本命堅レースを見送り、波乱余地レースに絞る」フィルターとして使う。
"""
from __future__ import annotations

import json
import math
from itertools import combinations as _combinations
from dataclasses import dataclass
from pathlib import Path

# TRAIN(2023-07-01〜2026-02-28) の top3_sum 四分位カット（既定値＝コミット済フォールバック）。
# 再学習でモデル確率分布が変わると四分位がズレるため、週次再学習後に
# scripts/recompute_upset_cuts_wt.py が data/models/upset_cuts_wt.json を更新し、
# 下記 _load_cuts() がそれを優先採用する（無ければこの既定値）。
UPSET_TOP3SUM_CUTS_DEFAULT = (1.70, 1.90, 2.08)
UPSET_TIERS = ("Q1_loose", "Q2", "Q3", "Q4_chalk")

_CUTS_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "upset_cuts_wt.json"


def _load_cuts() -> tuple[float, float, float]:
    """再計測済みカット(JSON)を読む。無効/不在なら既定値。"""
    try:
        d = json.loads(_CUTS_PATH.read_text(encoding="utf-8"))
        c = d.get("cuts")
        if isinstance(c, (list, tuple)) and len(c) == 3:
            cuts = tuple(float(x) for x in c)
            if cuts[0] < cuts[1] < cuts[2]:   # 単調性チェック
                return cuts  # type: ignore[return-value]
    except Exception:
        pass
    return UPSET_TOP3SUM_CUTS_DEFAULT


# 実効カット（プロセス起動時に確定。日次cronは毎回新プロセスなので最新を反映）
UPSET_TOP3SUM_CUTS = _load_cuts()


def upset_tier(top3_sum: float) -> str:
    """top3_sum を TRAIN 四分位カットで Q1_loose〜Q4_chalk に割り当てる。"""
    c1, c2, c3 = UPSET_TOP3SUM_CUTS
    if top3_sum < c1:
        return "Q1_loose"
    if top3_sum < c2:
        return "Q2"
    if top3_sum < c3:
        return "Q3"
    return "Q4_chalk"


def race_signals(probs_desc: list[float], n_riders: int) -> dict:
    """pred_prob 降順リストから確定前シグナルを計算する。

    probs_desc: そのレースの pred_prob を降順に並べたリスト
    n_riders:   出走車数
    """
    p1 = probs_desc[0] if probs_desc else 0.0
    p2 = probs_desc[1] if len(probs_desc) >= 2 else 0.0
    p3 = probs_desc[2] if len(probs_desc) >= 3 else 0.0
    top3_sum = p1 + p2 + p3
    return {
        "gap12": p1 - p2,
        "ratio": p1 / (3.0 / n_riders) if n_riders else 0.0,
        "top2_sum": p1 + p2,
        "top3_sum": top3_sum,
        "upset_tier": upset_tier(top3_sum),
    }


# ステーク傾斜の既定方針（方針A・scripts/exp_stake_tilt_wt.py で検証）。
# 波乱帯(Q1_loose)に厚く、本命堅(Q3/Q4)は見送り。100円単位の整数倍率。
# TEST(OOS) ROI: flat 351% → この傾斜 745%（最大払戻除去640%・上限値）。
STAKE_TILT_DEFAULT = {"Q1_loose": 2, "Q2": 1, "Q3": 0, "Q4_chalk": 0}


def stake_units(top3_sum: float, policy: dict | None = None) -> int:
    """波乱帯に応じた賭け金倍率（×100円単位）。0=見送り。"""
    pol = policy or STAKE_TILT_DEFAULT
    return int(pol.get(upset_tier(top3_sum), 1))


def passes_upset_gate(top3_sum: float, max_tier: str = "Q1_loose") -> bool:
    """ゲート通過判定。max_tier までの帯（loose側）のみ通す。

    max_tier='Q1_loose' なら最もlooseな四分位のみ、'Q2' なら Q1+Q2 を通す。
    """
    order = {t: i for i, t in enumerate(UPSET_TIERS)}
    return order[upset_tier(top3_sum)] <= order[max_tier]


# ═══════════════════════════════════════════════════════════════════════════
# SS 購入ポリシー（2026-07-16: 選抜カットのみ）
#
# ※ 旧S1（7車三連複・内部rank 7PLUS_R・旧称SS）は 2026-07-16 に全廃。
#   本セクションの SS_STAKE / ss_policy / is_senbatsu / line_score_features は
#   呼び出し側互換（過去日再採点・分析スクリプト）のため残置する。
#   新S1（6車三連単・ペーパー）は下の S1_* 定数を参照。
#
# doc53（2026-07-12）の 4分戦カット・ライン格差≥1.5増額は、実精算方式
# （盤面ランキング・落車失格=外れ計上）での再検証（exp_ss_policy_realistic_wt.py）で
# 窓間の方向不一致（4分戦: テスト有効/VAL逆効果、格差帯: テスト110%/VAL56%）と判明し削除。
# 選抜カットのみ全3窓一貫（選抜セグメント ROI 26%/39%/0%）で維持。
#
# ※ S/S+（三連単F 7PLUS_ST/STP）は優位性なしのため 2026-07-15 に全廃
#   （keirin_survivor_bias_inflation 調査: ROI 70-90% = 控除率の壁）。
# ═══════════════════════════════════════════════════════════════════════════

SS_STAKE = 100             # SS 賭け金（円/点）

# ═══════════════════════════════════════════════════════════════════════════
# 新S1（6車三連単・モデル1位→2位→{3位,4位} 2点）— 2026-07-17 全廃
#
# 3独立窓（2026-07-16 検証）では全窓100%超だったが、正規プロトコル
# （学習〜2025-03-31・検証2025-04-01〜2026-03-31の1年・テスト2026-04-01〜07-15）
# の再検証で検証最良70.3%・100%超なし→棄却（exp_ranks_valtest.py）。
# 6車全域スイープ（約500セル）・新S1候補（適応型2車軸トリオ/m1 1着固定三連単・
# exp_s1_adaptive.py）も検証ROI≥95%のセルなしで全滅。→ 2026-07-17 に候補生成・
# judge・採点を全停止し、picks_history の #6S1 行は picks_history_r_archive へ退避。
# 定数は過去スクリプト（backfill_s1_six_wt.py 等）の互換のため残置。
# ═══════════════════════════════════════════════════════════════════════════

S1_NE = 6                  # 対象車数（6車ちょうど）
S1_GAP12_MIN = 0.11        # gap12 下限（rawスケール・凍結値）
S1_STAKE = 100             # 円/点（ペーパー）

# ═══════════════════════════════════════════════════════════════════════════
# S1（新設計・win軸1着固定×3着内モデル相手2車・三連単2点流し）— 2026-07-19 導入
#
# 旧S1（7車三連複7PLUS_R）・新S1（6車三連単SIX_S1）はいずれも全廃されたが、
# 「1着専用モデル(win model)で軸を固定し、3着内モデルで相手2車を選ぶ」構造は
# 未検証だった。ユーザー指示で再検討し、7車で頑健な生存条件を発見
# （exp_s1_win_axis_trifecta.py・正規プロトコル）。
#
# 軸 = win model（lgbm_wt_win）のレース内1位。
# 相手 = 3着内モデル（配信モデル）で軸を除いた残り車の上位2頭(p1,p2)。
# ゲート: top3_gap（p1とp2の3着内確率差）>= S1W_TOP3_GAP_MIN。
# 買い目: 三連単 軸→p1→p2, 軸→p2→p1 の2点流し（目オッズ下限なし＝leg=0）。
#
# 正規プロトコル: 検証2025-04-01〜2026-03-31 ROI145.8%(n=9949) →
# テスト2026-04-01〜07-15 ROI135.3%(n=2851・約28R/日)。閾値0.08〜0.20で
# 検証・テストとも単調に改善（過去のS1候補群のような窓間の符号反転なし）。
# S2/S3との重複はわずか4.3%とほぼ独立。月次11/16・年次2025/2026年とも100%超
# （S2:9/16月・S3:9/16月より高い一貫性）。
# 払戻分布は一部の高額配当に偏る（的中476件中上位3件除外でROI99.2%まで低下）。
# レース単位ROIのmean±2SDでは不合格だが、同基準でS2/S3も不合格（三連系券種の
# 払戻分布が的中時に大きく偏る構造的性質であり、S1固有の弱点ではないと確認済み）。
# ユーザー判断によりペーパートレードで運用開始（2026-07-19）。
#
# 2026-07-19 同日中の追加チューニング: 母数を1日15R以下に絞り的中率を上げたい
# というユーザー要望を受け、top3_gap閾値を0.15→0.22へ引き上げ（exp_s1w_gap_tighten.py・
# 同一正規プロトコルの継続、多重比較ではなく既存の単調帯[0.05,0.20]の自然な延長）。
# 検証15.2R/日・的中率18.1%・ROI171.6%、テスト15.3R/日・的中率18.2%・ROI146.0%
# （0.15時点: 27.3/26.9R・16.7-16.8%的中・135.3-145.8%ROI から改善）。
# あわせて、gap12/win_rankモデルの本番リーク（[[keirin_composite_ratio_gate]]参照・
# lgbm_wt_winがfull_refit=Trueでホールドアウトなしのため過去picks_history再構築時に
# 未来データ込みでスコアリングしていた問題）と同型の問題がS1にも存在したため、
# 同時に四半期walk-forwardモデル（lgbm_wt_eval_q24xx/lgbm_wt_win_q24xx等）で
# 全期間再構築した。
#
# 2026-07-21 再チューニング: 高配当（万車券含む）を取りこぼさない方向へ再設計。
# top3_gap閾値を0.22→0.15へ戻したうえ、軸の単勝勝率(pred_win)が高すぎる
# （＝本命決着で低配当になりやすい）レースを除外する新ゲートを追加
# （exp_s1_20x_filter_design.py・honest全期間 th>=0.15 母集団 n=25,268 で検証）。
# 軸勝率<=50%フィルター単体の実績: n=13,510(53.5%)・的中率10.7%・ROI146.3%、
# 20倍以上再現率65.9%・30倍以上70.3%・50倍以上72.5%・万車券再現率84.0%
# （無フィルター時: 的中率16.2%・ROI120.3%・母数25,268）。
# 的中率は下がるが、S1の的中条件（軸が1着固定）と高配当（＝波乱決着）は
# 構造的にトレードオフのため、的中率を維持したまま高配当のみ拾うことは
# できないとユーザーに説明のうえ、高配当の取りこぼし防止を優先する方針で採用。
# ═══════════════════════════════════════════════════════════════════════════

S1W_NE = 7                  # 対象車数（7車ちょうど）
S1W_TOP3_GAP_MIN = 0.15     # 相手2車(p1,p2)の3着内モデル確率差 下限（2026-07-21再変更）
S1W_AXIS_WIN_PROB_MAX = 0.50  # 軸の単勝勝率 上限（本命決着＝低配当レースを除外・2026-07-21新設）
S1W_DENY_AXIS_CLASS = {"S1", "A1"}  # 軸級班denyフィルター（2026-07-22新設）
S1W_STAKE = 100              # 円/点（ペーパー）

# フィールド全体の指数エントロピー上限（2026-07-27導入）。S7/S9で有効だった
# entropyシグナル（exp_upset_trio30_v2_wt.py等）がS1でも独立に機能するか
# exp_s1w_entropy_wt.pyで検証: S1w_gate通過済み母集団(n=6,502)で2024Q1のみ
# entropy下位25%点(=1.7571)を決定→残り9四半期へブラインド適用（真のwalk-forward・
# 9四半期**全て**で方向一致）:
#   entropy<=1.7571: n=1,686 的中14.9% ROI454.7%（30倍+的中47/107=44%を独占）
#   entropy> 1.7571: n=4,203 的中8.5%  ROI71.5%（赤字帯）
# axis_win_prob<=0.50・軸級班denyとは独立な追加ゲート（S7のentropyがaxis_sumと
# ほぼ無相関だったのと同型）。
S1W_ENTROPY_MAX = 1.7571


def s1w_select(
    win_probs: dict[int, float], top3_probs: dict[int, float],
) -> tuple[int, int, int, float] | None:
    """S1(新設計)の軸・相手2車を選定する。

    win_probs / top3_probs: {frame_no: 確率} の辞書（レース内全車）。
    軸 = win_probsの1位。相手p1/p2 = 軸を除いたtop3_probsの上位2頭。

    returns (axis, p1, p2, top3_gap) or None（データ不足で選定不能）。
    """
    if not win_probs or not top3_probs:
        return None
    axis = max(win_probs, key=lambda f: win_probs[f])
    remainder = sorted(
        (f for f in top3_probs if f != axis), key=lambda f: -top3_probs[f])
    if len(remainder) < 2:
        return None
    p1, p2 = remainder[0], remainder[1]
    top3_gap = top3_probs[p1] - top3_probs[p2]
    return axis, p1, p2, top3_gap


def s1w_gate(
    top3_gap: float, axis_win_prob: float | None = None,
    axis_player_class: str | None = None, entropy: float | None = None,
) -> bool:
    """S1(新設計)のゲート判定。

    - top3_gap（相手2車の3着内モデル確信度）>= S1W_TOP3_GAP_MIN
    - axis_win_prob（軸の単勝勝率）が渡された場合は <= S1W_AXIS_WIN_PROB_MAX も要求
      （本命決着＝低配当レースを除外し、高配当の取りこぼしを防ぐ・2026-07-21新設）。
      axis_win_prob=None の場合はこの条件をスキップ（過去分析スクリプト互換）。
    - axis_player_class（軸選手の級班）が渡された場合は S1W_DENY_AXIS_CLASS
      （各グレード内の最上位クラス=S1/A1）を除外する（2026-07-22新設）。
      軸がそのグレードの「格上」認定選手だと配当が低くなりやすい傾向を確認した
      （honest全期間: 的中率は変化なし・ROI 138.5%→173.5%・5万円以上配当の
      再現率85.7%を維持しつつ母数を約半分に絞る）。
      axis_player_class=None の場合はこの条件をスキップ（過去分析スクリプト互換）。
    - entropy（フィールド全体の指数エントロピー）が渡された場合は
      <= S1W_ENTROPY_MAX も要求（2026-07-27新設。S7/S9で有効だったentropy
      シグナルがS1でも独立に機能することをexp_s1w_entropy_wt.pyで確認：
      entropy<=1.7571 ROI454.7% / entropy>1.7571 ROI71.5%）。
      entropy=None の場合はこの条件をスキップ（過去分析スクリプト互換）。
    """
    if top3_gap < S1W_TOP3_GAP_MIN:
        return False
    if axis_win_prob is not None and axis_win_prob > S1W_AXIS_WIN_PROB_MAX:
        return False
    if axis_player_class is not None and axis_player_class in S1W_DENY_AXIS_CLASS:
        return False
    if entropy is not None and entropy > S1W_ENTROPY_MAX:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# S7（単勝×複勝指数トップ3重なり軸×波乱度選出・三連複2軸総流し）— 2026-07-21 導入
#
# ユーザー仮説の検証（exp_upset_axis_trio.py 相当・正規プロトコル: 検証2025-04-01〜
# 2026-03-31／テスト2026-04-01〜07-10）で発見:
#
# 軸 = win_top3(pred_win_pct上位3) ∩ top3_top3(pred_top3_pct上位3) の重なり車。
#   重なり>=2: 重なりの中からpred_top3_pct上位2を軸に採用。
#   重なり==1: その1車 + 残りでpred_top3_pct最上位の1車。
#   重なり==0: 対象外（実データで58,616中1件のみ、事実上発生しない）。
# 波乱度指数 = 軸2車のpred_top3_pct合計（axis_sum）。低いほど「軸自体が本命でない」
#   ＝波乱度が高いレースと解釈する。レース全体のエントロピー（拮抗度）で絞ると
#   ROIが悪化する（絞り込みなし85.7%→73.5%）ことを確認済みで不採用。
# 選出 = 当日の該当レースをaxis_sum昇順に並べ、上位 RANK_7S_DAILY_TOP_N 件を採用
#   （1レース単位の閾値ゲートではなく日次クロスレースランキング）。
# 買い目 = 三連複 軸2車 + 残り5車のいずれか1車（5点・オッズ下限なし）。
#
# 正規プロトコル結果（N=15/日）: 検証ROI116.3%(n=5475)・テストROI116.3%(n=1515・
# ほぼ完全一致）。的中率は検証37.8%/テスト36.0%。的中時に三連複20倍以上となる
# 割合は絞り込みなし7.3%に対しN=15で16.0%(検証)/18.5%(テスト)と倍以上に向上。
# Nを5/10/15/20/30と変えた際のROIは両窓とも単調減衰（181.5→136.0→116.3→107.4→97.4%
# 検証・153.4→134.7→116.3→107.9→101.0%テスト）で自然な閾値の延長として信頼できる。
# 単勝指数側の信号（win_max・単勝トップ2合計）との複合も試したが改善なし
# （複勝指数トップ2合計との相関が強く追加情報量が乏しいため、単独採用のままとする）。
# ユーザー判断によりペーパートレードで運用開始（2026-07-21）。
#
# 2026-07-21（同日中の追加検証）: 軸2車がWINTICKET公式予想の◎◯
# （prediction_mark∈{1,2}）と重なる場合、期待値が下がるのではというユーザー仮説を
# 検証（exp_s4_wt_axis_overlap.py・honest全期間再構築 2024-01-01〜2026-07-20・
# 四半期walk-forwardモデル使用）。日次Top10選出内で重なり数別に分解した結果:
#   重なり0（◎◯と全く重ならない）  : n=438  的中35.4% ROI**408.1%**
#   重なり1（片方だけ重なる）      : n=4618 的中33.4% ROI148.7%
#   重なり2（◎◯と完全一致）      : n=4164 的中37.1% ROI 75.7%（赤字）
# 的中率はほぼ横ばいなのにROIが重なり数に応じて単調に悪化する構造を確認
# （完全一致時は市場に織り込まれ済みで払戻が縮む＝コンセンサスピックの低配当化）。
# ユーザー指示により、重なり0は無条件で全件採用・重なり1はaxis_sum昇順で固定
# RANK_7S_DAILY_TOP_N件・重なり2は完全除外という選出方式へ変更（1日の採用本数は
# 重なり0の発生数に応じて可変・honest全期間で平均10.77R/日）。
# honest全期間再構築（この方式）: 9,927R（922日・10.77R/日）・的中36.3%・
# **ROI131.3%**（旧方式の128.1%から改善）。内訳: 重なり0(943R)的中39.4%/ROI232.8%・
# 重なり1(8984R)的中36.0%/ROI120.6%。
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 賭け金の単一正本（2026-08-07・ユーザー指示で全ランク統一）
#
# 【why】従来ペーパー成績は **1点100円固定**で記録していた（7S/7A/7SS 5点=500円 /
#   7B 3点=300円 / 9S/9A 7点=700円）。一方 netkeirin 入稿は 1レース約10,000円
#   （2,000円×5点 等）で、7H1 は最初から予算枠10,000円。**同じランクでも
#   ペーパーと入稿で金額が20倍違い、ランク間でも投資額が揃っていなかった**ため、
#   Web で投資・回収を並べても比較にならなかった。
#
# 【what】**1レース RACE_BUDGET 円を点数で均等割りし STAKE_UNIT 単位へ切り捨てる**方式に
#   全ランクを統一する（7H1 が既に採っていた方式）。
#     5点 → 2,000円/点（計10,000円） 4点 → 2,500円（10,000） 3点 → 3,300円（9,900）
#     7点 → 1,400円（9,800）        2点 → 5,000円（10,000） 1点 → 10,000円
#   点数固定のランク（7S/7A/7SS=5点・7B=3点・9S/9A=7点）は netkeirin の
#   現行 stake_per_line と**完全に一致**する。
#
# 【ROI への影響】点数が一定なら投資も払戻も同じ倍率で増えるので **ROI は不変**。
#   変わるのは表示される金額と、**点数が可変なランク（7C・7H1）や欠車で点数が
#   減ったレースの重み**（1レース1万円で揃うので、レース単位の平均になる）。
#
# ⚠️ 過去分の picks_history は 100円/点 で記録済み。**新旧混在は禁止**なので
#   `scripts/migrate_picks_history_stake.py` で全期間を再計算すること。
# ═══════════════════════════════════════════════════════════════════════════

RACE_BUDGET = 10000   # 1レースあたりの投資額（円）
STAKE_UNIT = 100      # 賭け金の最小単位（円）


def unit_stake(n_legs: int, budget: int = RACE_BUDGET,
               unit: int = STAKE_UNIT) -> int:
    """点数 n_legs のときの1点あたり賭け金（unit 単位・切り捨て）。

    切り捨てにより実投資は予算枠をわずかに下回る（3点なら 9,900円）。
    n_legs<=0 は呼び出し側のバグだが、金額計算で落ちると通知全体が止まるため
    unit を返す（0点のレースはそもそも購入されない）。
    """
    if n_legs <= 0:
        return unit
    return max(unit, (budget // n_legs) // unit * unit)

# モデル3着内率の上位2車の合計の下限。**絶対閾値**（日次の相対順位で切らない）。
# 7H1 で相対順位にしたら件数が半減した前例があるため。この値は pred_top3 の
# 較正に依存するので、**モデルを再学習したら分布を確認して更新すること**。
RANK_7C_P3_SUM_MIN = 1.44

# 3列目（相手）の足切り。軸2車を除く5車のうち **モデル3着内率がこの値以上**の車だけ買う。
# ＝「3着内に入れなそうな車を外す」（ユーザー指示 2026-08-07）。
RANK_7C_LEG_P3_MIN = 0.15

# 🔴 **相手が RANK_7C_LEGS_MIN 点に満たないレースは買わない**（ユーザー発案）。
#    相手が少ししか残らない＝残り5車の実力差が大きい＝**当たっても配当が付かない**。
#    的中率は点数帯によらずほぼ一定（63〜68%）なのに、的中のうち2.0倍以下の割合だけが
#    1点100% / 2点67.0% / 3点45.6% / 4点24.9% / 5点12.9% と単調に効く。
#    したがって**的中率とROIを犠牲にせず低配当だけを落とせる唯一のゲート**。
RANK_7C_LEGS_MIN = 4

# 🔴 低配当パターンの除外（2026-08-07 追加・ユーザー発見）。
#   **「複勝率の3位と4位が大きく離れている（＝上位3車が抜けている）」かつ
#     「その上位3車が同一ライン」**のレースは見送る。
#   実例 豊橋5R(2026-08-07): 複勝率 82.5/62.6/59.0 → 27.2 と3-4位差31.8pt、
#   上位3車(7・4・2)が全員ライン1。結果 2-4-7 で三連複 2.0倍＝10,000円投資に対し
#   払戻4,000円。**ライン3車で決まると配当が付かない**。
#
#   評価窓での除外対象の実力（母集団の4.8%）:
#     的中 68.68%（全体58.08%より**高い**）／**的中の66.1%が2.0倍以下**／ROI 73.6%
#   ＝「よく当たるが儲からない」型を正確に切り出せている。
#
#   見送った場合（評価窓/確認窓）: 件数 23.48→22.36 / 22.47→21.56 件/日、
#   的中 58.08→57.55 / 59.42→59.07%、ROI 77.3→77.5 / 77.8→77.9%、
#   **低配当的中 2.10→1.59 / 2.08→1.66 件/日（−24%/−20%）**。両窓で符号一致。
#
#   ⚠️ 片方だけでは切らない。「3-4位差だけ」は7.6%・「同一ラインだけ」は29.4%が該当し、
#      後者は的中率を3.2pt も落とす。**両方揃ったときだけ**が最小の犠牲で効く。
#   ⚠️ ROI はほとんど動かない（+0.2pt）。これは収益改善ではなく
#      「当たったのに損をした」体験の比率を下げる施策（15.4%→12.4%）。
RANK_7C_LOWPAY_GAP34_MIN = 0.30

# 1レースの予算枠。点数で均等割りし 100円単位へ切り捨てる（7H1 と同じ方式）。
# 🔴 **固定額/点ではない。** 点数が可変なので、固定額にすると点数の多いレースほど
#    投資が増えて ROI の重み付けが歪む。予算枠なら全レースが等しく効く。
#    この方式では **ガミ ⟺ 的中オッズ < 点数**（戻り = (BUDGET/k) × odds）。
# ⚠️ 値をここに書き直さないこと（2026-08-08 是正）。以前は 10000/100 という
#    リテラルの再定義で、`RACE_BUDGET`/`STAKE_UNIT` と**同じ値を2箇所に手書き**
#    していた。一方 netkeirin 入稿側（`netkeirin_submit_wt.RANK_CONFIGS`）は
#    `RACE_BUDGET` を直接参照しているため、全ランク一律で予算を改定したときに
#    「入稿される金額」と「picks_history に記録される賭け金」が無言で食い違う。
#    同じ値の手書き二重管理はこのリポジトリが繰り返し踏んでいる事故の型。
RANK_7C_BUDGET = RACE_BUDGET
RANK_7C_UNIT = STAKE_UNIT


RANK_7S_NE = 7                  # 対象車数（7車ちょうど）
RANK_7S_STAKE = unit_stake(5)   # 2,000円/点（5点=10,000円/レース）

# 三連複が安くなりやすい（極端な人気決着になりやすい）レースの除外上限。
#
# 【2026-07-31改定】方針転換: 「まずは十分な的中率の上での安定したROI確保」
# （ユーザー方針）を目的に、月次凍結vintageモデルでのhonest全期間検証
# （2024-01-01〜2026-07-31・31ヶ月・配分はuniform固定・
#   scripts/exp_s7_cdf_regime_full_period.py）で再較正した。
#   現行(axis_sum<=1.3・mark3<=1併用): 576R・0.61件/日・的中34.0%・ROI78.5%・
#     月次ROI標準偏差43.7（月次0%回数1）
#   新設定(axis_sum<=1.5・mark3ゲート撤廃＝下記rank_7s_daily_select参照):
#     6,546R・6.94件/日・**的中41.0%**・**ROI79.3%**・
#     **月次ROI標準偏差17.3**（月次0%回数0）
# 的中率・ROIとも改善しつつ月次変動を約1/2.5に抑え、日次件数を約11倍
# （目標の5〜10件/日レンジ）に拡大できることを確認したため採用。
# mark3ゲート単体撤廃（axis_sum据え置き）はROI74.9%に悪化するため不採用
# （axis_sum<=1.5との組み合わせで初めて機能する）。
#
# 【旧経緯（2026-07-24導入時、汚染モデル時代の数値・参考情報として残置）】
# 買い目は5点流し（1点100円=500円）のため、三連複配当が500円(5倍)を下回ると
# 的中しても賭け金を割る、という着眼から導入。axis_sumとレース着地時の
# 三連複配当<500円の相関 AUC 0.64(train)/0.67(test)。当時の汚染モデルでの
# シミュレーションでROI131.3%→147.1%として1.3を採用したが、
# [[keirin_wt_foundational_audit_2026_07_29]]で当時のvintageモデルが
# 汚染されていたと判明したため、絶対値は参考にしないこと。
# 【2026-08-05】1.5 → 1.40 へ引き下げ（ユーザー承認済み）。7車専用の定数で、
# 9車(S9/9A)には元々導入していない（下記L689付近のコメント参照）ため影響しない。
#
# 目的は「7S の純度を上げる」こと。**全体の期待値を上げる変更ではない**。
#
# 掃引窓（2025-07〜2026-07・4窓）で 7S の ROI 86.0→92.1% と出たが、
# 掃引した窓の数字は多重比較で膨らむ。**掃引に一度も使っていない確認窓
# （2024-07〜2025-06・4窓、`scripts/exp_7s7a_threshold_confirm.py`）**で
# 閾値を固定したまま一度きり検証した結果:
#
# | 区分 | 件/日 | 的中 | ROI | 窓別ROI(c1 c2 c3 c4) |
# |---|---|---|---|---|
# | 現行 7S (<=1.50) | 6.69 | 44.2% | 82.3% | 75.6 79.5 88.7 85.4 |
# | 新 7S (<=1.40)   | 3.68 | 41.0% | **84.4%** | 76.7 78.4 89.1 93.3 |
# | 現行 7A          | 11.45 | 39.0% | 77.7% | 81.4 81.6 68.8 79.0 |
# | 新 7A            | 13.62 | 40.9% | 78.7% | 80.2 80.8 74.1 79.6 |
# | 現行 7S+7A       | 18.15 | 41.0% | 79.3% | 78.9 80.9 76.5 81.1 |
# | 新 7S+7A         | 17.30 | 41.0% | **79.8%** | 79.3 80.3 77.2 82.1 |
#
# 掃引窓の +6.1pt は確認窓で **+2.1pt** に縮んだ（多重比較ぶんが落ちた）が、
# **確認窓4つすべてで ROI>=75% かつ現行を上回った**ため採用。
#
# ⚠️ **7S+7A の合計では +0.5pt しか変わらない**。1.40〜1.50 の帯は消えるのではなく
#    大半が 7A へ移るため（n_fail が 0→1 になるだけ）。実際に除外されるのは
#    「axis_sum 1.40〜1.50 かつ entropy も不合格」の 0.85件/日のみ。
#    **7S だけを買う運用なら +2.1pt、7S+7A を買う運用ならほぼ無意味**。
# ⚠️ 変更後は 7S が 6.69→3.68件/日 とほぼ半減し、その分 7A が増える。
RANK_7S_AXIS_SUM_MAX = 1.40

# フィールド全体の指数エントロピー上限（2026-07-26・ユーザー要望「30倍以上の
# 高配当が見込めるレースに絞りたい」への対応。exp_upset_trio30_v2_wt.py /
# exp_s4_entropy_walkforward.py / exp_s4_entropy_uncapped_wt.py 参照）。
#
# 注意: 2026-07-21のS7設計時点では「レース全体のエントロピーで絞るとROIが
# 悪化する（絞り込みなし85.7%→73.5%）」という逆方向（entropy**高い**ほど波乱＝
# 採用、旧Uランクu_entropyと同じ発想でaxis_sumの代替ランキング基準として試行）
# の検証結果が残っているが、本フィルタはそれとは別物: **低い**entropy
# （軸2車に予測確率が集中＝残り5車が拮抗）を、axis_sum/wt_overlap等の既存ゲート
# を通過した候補への**追加ゲート**として使う。方向も用途も異なるため矛盾しない。
#
# 検証（2026-07-26・quarterly walk-forwardモデルの pred_prob のみ使用＝発走前
# 確定情報のみ・オッズ非依存）: 2024Q1(n=1125, 件数cap解除後の生プール)の
# entropy下位25%点(=1.8329)だけを閾値として固定し、2024Q2〜2026Q2-3の残り
# 7四半期へブラインド適用（真のwalk-forward・8四半期全てで方向一致）:
#   entropy<=1.8329: n=1,617 的中38.2% ROI266.1%（30倍+的中187/252件=74%を独占）
#   entropy> 1.8329: n=2,605 的中29.9% ROI 78.1%（構造的な赤字帯）
#   フィルタなし全体: n=4,222 的中33.1% ROI150.1%
# 同数条件での比較（axis_sum昇順で同じ件数を採用した場合）でも、entropy選定は
# 7四半期中6四半期で明確に上回り、残り1四半期も同水準（axis_sumの代替ではなく
# 独立した追加情報。spearman相関≈-0.08で axis_sum とはほぼ無相関）。
# 採用ペースは平均2.56件/日（RANK_7S_AXIS_SUM_MAX等の既存ゲートは全て維持のまま）。
RANK_7S_ENTROPY_MAX = 1.8329

# 日次合計の上限（entropy昇順で採用・2026-07-26再導入）。
# 件数capをentropyゲートに置換した初日（2026-07-26）、entropyフィールドを
# 持たない旧形式の生候補JSON（デプロイ前に生成された朝バッチ分）が
# rank_7s_daily_select() の `c.get("entropy", 0.0)` フォールバックにより
# entropy=0.0扱い＝常にゲート通過してしまい、1日26件という honest全期間
# walk-forward(2024-01-01〜2026-07-25・832日、最大9件/日)では一度も
# 発生しなかった規模の異常発生を招いた（原因判明後、フォールバックは
# 安全側のfloat("inf")＝常に除外に修正済み）。
# ユーザー要望「朝夕合わせて10レースちょっとに絞りたい・信頼度の高い方を残す」
# を受け、entropyゲート通過候補が多い日はentropy昇順（＝最も自信がある順）で
# 上位のみ採用する日次capを追加。honest全期間(832日)ではentropyゲート通過が
# 最大9件/日のため、この上限は通常運用ではほぼ発火しない安全網であり、
# capの値を8/10/12/15/無制限で振っても全期間ROI/件数は完全に同一
# （exp_s4_daily_cap_by_entropy.py参照）。異常発生時のみ効く設計。
RANK_7S_DAILY_CAP = 12

# ---- 3ヘッド軸選定の大敗ペナルティ重み（2026-08-04・7車立てのみ） ----------
# 軸1 = pred_win 最上位、軸2 = z(pred_prob) − w2·z(bad_prob) の最上位（軸1を除く）。
# w1（軸1側の大敗ペナルティ）は 0 で固定＝定数を置かない。軸1の外れは惜敗(4着)が
# 62% で大敗が少なく、ペナルティを掛けるほど1着率が落ちるため（w1=0.6 で −3.5pt）。
# 対して軸2の外れは6着以下の大敗が67%を占めるので、軸2にだけ掛ける。
#
# w2=0.3 の根拠（scripts/exp_three_head_axis.py・月次凍結せず窓ごとに再学習した
# honest walk-forward 4窓 2025-07〜2026-07・約4,300推奨）:
#   現行(win∩top3重なり) → w2=0.3 で 的中 38.9→41.4% / ROI 77.5→82.8% /
#   軸2の6着以下 17.2→14.1% / 軸2の3着内 57.4→60.1%
#   **4窓すべてで的中・ROIとも改善し符号反転なし**（85.9→88.9 / 80.4→81.4 /
#   69.3→79.1 / 74.5→81.8）。
# w2=0.6 も平均は近い(ROI 81.0%)が、w1=0.3 を併用する案は 2026-01〜04 窓で
# ROI が −0.9pt 反転する。4/4クリーンなのは w1=0.0 × w2=0.3 だけ。
#
# ⚠️ ROI は改善するが 100% には届かない（77.5→82.8%）。控除率75%の壁は破れておらず、
#    改善するのは的中率と軸品質であって黒字化ではない。
# ⚠️ **9車立てには適用しない**。同じ掃引を9車(4窓・約550推奨)で回すと、平均では
#    +2.0pt に見えるが窓別では 2026-01〜04 で 的中 −2.8pt・ROI −19.9pt と反転する。
#    窓あたり139推奨ではノイズに埋もれ判定不能。9B・S9/9A再較正と同じく、7車の
#    知見が9車へ移らなかった事例（scripts/exp_three_head_axis.py --n-entries 9）。
RANK_AXIS2_BAD_WEIGHT = 0.3

# 3ヘッド軸選定を live 本番へ入れた日（この日以降の7車 picks は3ヘッドで選ばれている）。
# `backfill_7*_rank_wt.py` / `rebuild_7*_walkforward_pg.py` は bad_probs を渡さないため
# **旧軸で再構築すると live の3ヘッド記録を上書きして消す**。この日以降を含む再構築は
# `src/wt_rebuild_common.py::rebuild_pg_atomic` が既定で拒否する。
# 過去を旧軸で塗り直すこと自体が目的なら --allow-legacy-axis を明示すること。
# 3ヘッドで honest に再構築できるようにするには月次 vintage の bad モデルが必要だが、
# `train-wt --target` は top3/win しか作れない（2026-08-04 時点の制約）。
THREE_HEAD_AXIS_SINCE = "2026-08-04"


def _race_zscore(probs: dict[int, float]) -> dict[int, float]:
    """レース内で z 化する。全車同値・1車のみなら 0（ゼロ除算しない）。"""
    if not probs:
        return {}
    vals = list(probs.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    if var <= 0:
        return {f: 0.0 for f in probs}
    sd = math.sqrt(var)
    return {f: (v - mean) / sd for f, v in probs.items()}


def rank_7s_field_entropy(top3_probs: dict[int, float]) -> float:
    """レース全体（出走7車）の指数エントロピー（占有率ベースの拮抗度）を返す。

    top3_probs: {frame_no: pred_prob}（rank_7s_select_axis と同じ入力）。
    値が低いほど予測確率が一部の車（主に軸2車）に集中している状態。
    オッズを一切使わないため、発走前・オッズ非公開の朝の時点でも計算可能。
    """
    vals = list(top3_probs.values())
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def rank_7s_select_axis(
    win_probs: dict[int, float], top3_probs: dict[int, float],
    bad_probs: dict[int, float] | None = None,
    bad_weight: float = RANK_AXIS2_BAD_WEIGHT,
) -> tuple[int, int, float] | None:
    """S7の軸2車とaxis_sum（波乱度指数の元）を選定する。

    win_probs / top3_probs: {frame_no: 確率(0-1 or pct、比較にのみ使うのでスケール不問)}
      レース内全車分。
    bad_probs: 大敗（6着以下）確率。**渡された場合のみ3ヘッド選定**に切り替わる。
      None なら従来の重なり方式を完全に維持する（9車立て・過去分バックフィル用）。

    軸選定（3ヘッド版・bad_probs あり）:
      軸1 = win_probs 最上位
      軸2 = z(top3_probs) − bad_weight × z(bad_probs) の最上位（軸1を除く）
      重なりを要求しないため、旧ロジックが None を返す「重なり0」でも軸が立つ。

    軸選定（従来版・bad_probs なし）:
      win_probs上位3 ∩ top3_probs上位3 の重なり車から、
      重なり>=2ならtop3_probs上位2、重なり==1ならその1車+残りのtop3_probs最上位。

    returns (axis1, axis2, axis_sum) or None（重なり0・データ不足で選定不能）。
    axis_sum は axis1/axis2 の top3_probs 合計（波乱度指数・低いほど波乱寄り）。
    **3ヘッド版でも axis_sum の定義は変えない**（RANK_7S_AXIS_SUM_MAX ゲートの
    意味を変えないため）。
    """
    if not win_probs or not top3_probs or len(win_probs) < 3 or len(top3_probs) < 3:
        return None

    if bad_probs:
        axis1 = max(win_probs, key=lambda f: win_probs[f])
        zp = _race_zscore(top3_probs)
        zb = _race_zscore(bad_probs)
        rest = [f for f in top3_probs if f != axis1 and f in zb]
        if not rest:
            return None
        axis2 = max(rest, key=lambda f: zp[f] - bad_weight * zb[f])
        return axis1, axis2, top3_probs[axis1] + top3_probs[axis2]

    win_top3 = {f for f, _ in sorted(win_probs.items(), key=lambda kv: -kv[1])[:3]}
    place_top3 = {f for f, _ in sorted(top3_probs.items(), key=lambda kv: -kv[1])[:3]}
    overlap = win_top3 & place_top3
    if not overlap:
        return None
    if len(overlap) >= 2:
        cands = sorted(overlap, key=lambda f: -top3_probs[f])
        axis1, axis2 = cands[0], cands[1]
    else:
        axis1 = next(iter(overlap))
        rest = sorted((f for f in top3_probs if f != axis1), key=lambda f: -top3_probs[f])
        if not rest:
            return None
        axis2 = rest[0]
    axis_sum = top3_probs[axis1] + top3_probs[axis2]
    return axis1, axis2, axis_sum


def rank_7s_wt_overlap_n(
    axis1: int, axis2: int, wt_honmei: int | None, wt_taikou: int | None,
) -> int | None:
    """S7の軸2車とWINTICKET公式予想の◎◯（honmei/taikou）との重なり数を返す。

    wt_honmei: prediction_mark==1（◎）の frame_no。
    wt_taikou: prediction_mark==2（◯）の frame_no。
    いずれか欠損時は None（重なり判定不能・rank_7s_daily_select では除外対象）。
    """
    if wt_honmei is None or wt_taikou is None:
        return None
    return len({axis1, axis2} & {wt_honmei, wt_taikou})


# 2026-07-27: 軸2車がWINTICKET公式印◎◯△（mark1/2/3）のうち2つと一致する場合、
# 市場人気と重なり払戻が下がりやすいという仮説を検証（exp_s4s9_3mark_overlap_wt.py・
# S7+S9現行ライブ採用条件と同一母集団・n=2,560）:
#   軸2車のうち2車が◎◯△のいずれかと一致: n=1,357 ROI182.9%
#   それ以外                          : n=1,203 ROI434.4%
# 払戻トップ5は全てこの「2車一致」に該当しない側（overlap3<=1）に集中しており、
# 「2車一致」側でも黒字(183%)ではあるが明確にROIが低い。ただし軸2車のうち
# **1車のみ**が◎◯△のいずれかと一致するのは既存のwt_overlap_n==1（S）の定義上
# 常に発生する（除外しない）。除外対象は軸2車**両方**が◎◯△のいずれかと一致する
# ケースのみ。既存のwt_overlap_n（◎◯=mark1/2のみで判定・完全一致=2を既に除外）
# とは独立な追加ゲート（mark3=△も加味）。
#
# 【2026-07-31改定】S7/7Aはこのゲートを撤廃した（下記rank_7s_daily_select/
# rank_7a_daily_select参照。当時の検証は汚染モデル時代のもので、クリーンな
# 月次vintageモデルでの再検証ではaxis_sum<=1.5との組み合わせにより
# mark3ゲート無しの方がROI・的中率とも上回った）。
# S9/9A（rank_9s_daily_select/rank_9a_daily_select）は9車立てでは軸選定の母集団が
# 異なりS7と同一の再検証を行っていないため、このゲートを引き続き使用する。
RANK_7S_MARK3_OVERLAP_MAX = 1


def rank_7s_wt_mark3_overlap_n(
    axis1: int, axis2: int,
    wt_honmei: int | None, wt_taikou: int | None, wt_ana: int | None,
) -> int | None:
    """S7/S9の軸2車とWINTICKET公式印◎◯△（mark1/2/3・honmei/taikou/ana）との
    重なり数を返す（rank_7s_wt_overlap_nの◎◯のみの判定に△を加えた拡張版）。

    wt_ana: prediction_mark==3（△）の frame_no。
    いずれか欠損時は None（判定不能・rank_7s_daily_select/rank_9s_daily_select では
    フェイルセーフとして除外対象扱いにする）。
    """
    if wt_honmei is None or wt_taikou is None or wt_ana is None:
        return None
    return len({axis1, axis2} & {wt_honmei, wt_taikou, wt_ana})


# S7のSS(重なり0)のうち、軸2車のいずれかが各グレード最上位クラス（S1/A1）だと
# 配当が下がりやすい傾向を確認（2026-07-23・honest全期間検証）。当初はSS内の
# 格上非該当サブセットを観察用サブランク"SS+"として分岐表示していたが、
# サンプル数が少なすぎる（全期間で数十件規模）という理由でユーザー判断により
# 2026-07-27に廃止・SSへ統合した（picks_history.gate_label='SS+'の既存行も
# 'SS'へ一括更新済み。買い目・投資額は元々SSと同一のため実害はない）。
#
# 2026-07-31: SS自体（7SS/9SS・重なり0）も廃止・Sへ統合した。導入初期
# （2024-01〜07）は月13〜27件出ていたが、モデルの軸選定がWT公式印に近づく
# 方向へ進化した結果、2024-09以降は月0〜4件（2年間で19件）まで激減し
# 実質的にほぼ発生しない条件になっていたため（ユーザー判断）。
# 既存picks_history.gate_label IN ('SS','SS+') 行も 'S' へ一括更新済み。
# 買い目・投資額は元々SS/SS+/Sで同一のため実害なし。
# axis1_class/axis2_classパラメータは廃止後もコール側の互換のため残置（未使用）。


def rank_7s_gate_label(
    wt_overlap_n: int | None,
    axis1_class: str | None = None, axis2_class: str | None = None,
) -> str | None:
    """S7の表示ランク(gate_label)を返す。

    - wt_overlap_n == 0 または 1: "S"（2026-07-31以前は重なり0を"SS"として
      分岐表示していたが、発生頻度が実質ゼロまで激減したため廃止・Sへ統合）。
    - それ以外（重なり2・None）: None（除外対象）
    """
    if wt_overlap_n in (0, 1):
        return "S"
    return None


def rank_7s_daily_select(candidates: list[dict]) -> list[dict]:
    """S7の選出（2026-07-31改定: mark3ゲートを撤廃・axis_sum<=1.5に緩和）。

    candidates: 候補レースのリスト。各要素は最低限
      {"axis_sum": float, "wt_overlap_n": int | None, "entropy": float} を持つ dict。

    選出ロジック（全て閾値ゲート。件数による打ち切りは行わない）:
      - axis_sum > RANK_7S_AXIS_SUM_MAX（三連複が5倍未満に安くなりやすい極端な人気決着
        想定レース）は除外（2026-07-24導入。2026-07-31に1.3→1.5へ緩和）
      - entropy > RANK_7S_ENTROPY_MAX（フィールド全体の予測確率が拡散＝軸2車に集中して
        いない）は除外（2026-07-26導入。低いentropy＝軸2車に予測確率が集中し
        残り5車が拮抗、という状態が三連複高配当の的中と強く相関することを
        2024-2026の8四半期walk-forwardで確認。詳細はRANK_7S_ENTROPY_MAX定義部参照）。
        entropyキー欠損時は float("inf")扱い＝必ず除外する（フェイルセーフ。
        2026-07-26に0.0デフォルトだった旧実装が「欠損=常に通過」というフェイル
        オープンな挙動になっており、デプロイ当日の旧形式生候補JSON経由で
        entropyゲートが実質無効化される事故を招いたため修正済み）
      - wt_overlap_n == 0（◎◯と全く重ならない）: 上記ゲート通過分は全件採用
      - wt_overlap_n == 1（片方だけ重なる）: 上記ゲート通過分を全件採用
        （2026-07-26以前は axis_sum昇順で日次/バッチ上限cap件のみ採用していたが、
        「capを解除した生プールでentropyゲートが機能するか」を検証した結果、
        cap無しでentropyゲート単体の方が同数条件のaxis_sum選定より優れることを
        確認したため、件数capそのものを廃止した）
      - wt_overlap_n == 2（◎◯と完全一致）・None（WTマーク欠損）: 除外
        （完全一致は honest全期間検証でROI75.7%の赤字区分と判明したため）

    【2026-07-31撤廃】wt_mark3_overlap_n によるゲートは廃止した。
    クリーンな月次vintageモデルでのhonest全期間再検証
    （scripts/exp_s7_cdf_regime_full_period.py・2024-01〜2026-07・31ヶ月）で、
    axis_sum<=1.5との組み合わせにより mark3ゲート無しの方が
    的中率41.0%(旧34.0%)・ROI79.3%(旧78.5%)・月次ROI標準偏差17.3(旧43.7)・
    1日平均6.94件(旧0.61件)と全指標で上回ることを確認したため。
    詳細は RANK_7S_AXIS_SUM_MAX / RANK_7S_MARK3_OVERLAP_MAX 定義部のコメント参照。
    ※この変更によりS7の母集団が広がったため、旧mark3ゲートに依存していた
    7Aの選出ロジック(rank_7a_daily_select)も2026-07-31に2ゲート化し、
    新S7との重複選出がないことを検算済み（重複0件）。

    日次件数の上限（RANK_7S_DAILY_CAP）は本関数では適用しない（朝夜どちらか一方の
    バッチだけでは日次合計が分からないため）。日次合計への適用は
    rank_7s_evening_reselect() を参照。

    returns 採用された候補のリスト（axis_sum昇順・表示用の並び順のみ）。
    """
    pool = [
        c for c in candidates
        if c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX
        and c.get("entropy", float("inf")) <= RANK_7S_ENTROPY_MAX
        and c.get("wt_overlap_n") in (0, 1)
    ]
    return sorted(pool, key=lambda c: c["axis_sum"])


def rank_7s_evening_reselect(
    day_raw: list[dict], night_raw: list[dict], locked_keys: set[str] = frozenset(),
) -> list[dict]:
    """S7の朝夜統合選出（2026-07-26改定: entropyゲート通過後、日次合計を
    RANK_7S_DAILY_CAP件まで entropy昇順（＝最も自信がある順）でトリムする）。

    day_raw/night_raw: 朝/夜それぞれの生候補（選出前の全件、rank_7s_select_axis+
      rank_7s_wt_overlap_n+entropy計算を通した dict のリスト。各要素に "race_key" が必要）。
    locked_keys: 既に買い判定済み（picks_history に bet_amount>0 で記録済み）の
      race_key の集合。ゲート・トリムいずれでも除外しない（実購入は取り消せない
      ため）。ロック済み候補は rank_7s_daily_select() のゲート判定より前に分離する
      （2026-07-26修正: ロック済みでもゲート内で先に弾かれれば結果的に未保護に
      なる抜け穴があったため、ゲート適用前に確定で救済する設計に変更）。

    RANK_7S_DAILY_CAP は honest全期間(832日)で実際にゲート通過が最大9件/日だった
    ことから、通常運用ではほぼ発火しない安全網として設計されている
    （exp_s4_daily_cap_by_entropy.py参照）。

    returns 採用された候補のリスト。
    """
    # 同一 race_key が朝夜の両方に居たら夜側（新しい情報で評価し直した方）を採る。
    # 2026-08-04 の第2パスは「朝に◎◯が未公開だったレース」を再評価するため、
    # 同じ race_key が day_raw（overlap=None・ゲートで落ちる）と night_raw の両方に
    # 現れる。現状は day 側が必ずゲートで落ちるので実害は無いが、素朴な連結のままだと
    # ゲート条件が変わった途端に同じレースを2行採用してしまうため明示的に潰す。
    merged: dict[str, dict] = {}
    for c in day_raw + night_raw:
        merged[c.get("race_key")] = c
    all_raw = list(merged.values())
    locked = [c for c in all_raw if c.get("race_key") in locked_keys]
    gated = rank_7s_daily_select([c for c in all_raw if c.get("race_key") not in locked_keys])
    unlocked = sorted(gated, key=lambda c: c["entropy"])
    remaining_budget = max(0, RANK_7S_DAILY_CAP - len(locked))
    return locked + unlocked[:remaining_budget]


# ═══════════════════════════════════════════════════════════════════════════
# S9（S7の9車立て版・独立ランク）— 2026-07-26 導入
#
# 背景: 2026-08開催予定「ドリームレース」（S級・毎年8月・過去3回2023-2025年
# 全て9車立て）をターゲットに含めるため、7車専用だったS7のロジックを9車立てへ
# 拡張した。9車立ては全レースの8.0%（約6.5件/日・7車の85.5%に次ぐ規模）。
# ユーザー判断（Option B）により、7車のS7とは独立した別ランクとして実装
# （表示・集計を分離。ボリューム・買い目コスト(7点流し=700円 vs 5点=500円)が
# 異なるため）。
#
# 軸選定(rank_7s_select_axis)・フィールドentropy計算(rank_7s_field_entropy)・
# WT◎◯重なり判定(rank_7s_wt_overlap_n)・表示ランク(rank_7s_gate_label)はいずれも
# 車数非依存の汎用実装のためそのまま再利用する。
#
# 買い目 = 三連複 軸2車 + 残り7車のいずれか1車（7点・オッズ下限なし）
#
# 検証（2026-07-26・quarterly walk-forwardモデルのpred_prob使用・オッズ非依存・
# 9車軸選定候補5,632件・2024-01-01〜2026-07-25）: 2024Q1(n=406, wt_overlap∈{0,1})
# のentropy下位25%点(=1.9938)だけを閾値として固定し、2024Q2〜2026Qbの残り
# 9四半期へブラインド適用（真のwalk-forward・9四半期**全て**で方向一致・
# 例外なし）:
#   entropy<=1.9938: n=495 的中41.8% ROI279.2%（9四半期全て126.9〜675.1%で黒字）
#   entropy> 1.9938: n=1,287 的中29.1% ROI 72.4%（9四半期全て50.3〜100.7%の赤字帯）
# wt_overlap_n==2（完全一致）は7車同様66-91%で不採用。wt_overlap_n==0（軸2車が
# WT公式◎◯と全く重ならない）は小n(3-53/四半期)だが多くの四半期でROI200〜4683%と
# 極めて高い（7車のSS+/SS帯と同型のパターン）。
#
# axis_sum閾値（RANK_7S_AXIS_SUM_MAX相当）は9車では未較正のため導入していない
# （entropy単体で真のwalk-forward検証済み・複数の未較正閾値を積み増すことに
# よる過学習リスクを避けた。将来的な追加検証の余地あり）。
RANK_9S_NE = 9                   # 対象車数（9車ちょうど）
RANK_9S_STAKE = unit_stake(7)     # 1,400円/点（7点=9,800円/レース）
RANK_9S_ENTROPY_MAX = 1.9938


def rank_9s_daily_select(candidates: list[dict]) -> list[dict]:
    """S9の選出。S7と同じ閾値ゲート方式（axis_sum閾値は9車では未導入）。

    candidates: 候補レースのリスト。各要素は最低限
      {"wt_overlap_n": int | None, "entropy": float} を持つ dict。

    選出ロジック:
      - entropy > RANK_9S_ENTROPY_MAX は除外（詳細は上部コメント参照）。
        entropyキー欠損時は float("inf")扱い＝必ず除外（フェイルセーフ。
        S7での同種事故を踏まえた設計）
      - wt_overlap_n == 0（◎◯と全く重ならない）・1（片方だけ重なる）:
        上記ゲート通過分を全件採用（件数capなし。S9は元々低ボリュームのため
        S7のようなRANK_9S_DAILY_CAP安全網は現時点で不要と判断）
      - wt_overlap_n == 2（◎◯と完全一致）・None（WTマーク欠損）: 除外
      - wt_mark3_overlap_n（軸2車とWT公式印◎◯△=mark1/2/3との重なり数）が2は
        除外（2026-07-27導入・S7と共通のゲート。詳細はRANK_7S_MARK3_OVERLAP_MAX参照）

    returns 採用された候補のリスト（axis_sum昇順・表示用の並び順のみ）。
    """
    pool = [
        c for c in candidates
        if c.get("entropy", float("inf")) <= RANK_9S_ENTROPY_MAX
        and c.get("wt_overlap_n") in (0, 1)
        and c.get("wt_mark3_overlap_n", 2) <= RANK_7S_MARK3_OVERLAP_MAX
    ]
    return sorted(pool, key=lambda c: c["axis_sum"])


# ═══════════════════════════════════════════════════════════════════════════
# 7A/9A（S7/S9の「惜しい」境界ランク・ボリューム拡大用）— 2026-07-27 導入
#
# ユーザー要望: S7/S9(SS+/SS/S)は日次ボリュームが小さい（7車合計約1.2件/日・
# 9車約0.25件/日）。ROIはS7/S9ほど高くなくてよいので、的中率のあるゾーンを
# フィルタして推奨レースを増やしたい（三連複2軸のまま・低〜中配当帯を狙う）。
#
# 検証（exp_7a9a_boundary_wt.py / exp_7a9a_deep_dive.py / exp_7a9a_combo_check.py・
# quarterly walk-forward・honest全期間2024-01-01〜2026-07-25）:
#   まず wt_overlap==2（◎◯完全一致）側を含めた単純なaxis_sum区切りを検討したが、
#   全10四半期でROI70〜96%と一度も100%を安定して超えず、市場効率の壁で不採用と判断。
#   代わりに「S7の3ゲート（axis_sum<=1.3・entropy<=1.8329・mark3<=1）のうち
#   "ちょうど1つだけ"不合格の"惜しいレース"」を三連複2軸のまま評価したところ、
#   直近7四半期連続でROI100%超（101.7〜172.9%）・直近4四半期合算ROI150.3%
#   （n=778・約2.3〜4.0件/日）という頑健な結果を確認。wt_overlap==2は依然として
#   対象外（この条件でも常にROI<100%）。
#
# 設計:
#   母集団 = rank_7s_select_axis()/s9相当で軸選定成功 ∧ wt_overlap_n∈{0,1}（◎◯完全一致
#     と印欠損は既存同様に除外）。
#   7A: axis_sum<=RANK_7S_AXIS_SUM_MAX・entropy<=RANK_7S_ENTROPY_MAX
#       の2条件のうち、不合格がちょうど1個（0個=S7・2個とも不合格=対象外）。
#       【2026-07-31改定】旧来はmark3も含む3条件だったが、S7自体がmark3ゲートを
#       撤廃した（rank_7s_daily_select参照）ため、7Aも2条件に揃えた（mark3を条件に
#       残すと「mark3のみ不合格」の候補が新S7にも旧7Aにも該当し重複選出になる
#       ため）。
#   9A: 9車はaxis_sum閾値が未導入・S9側のmark3ゲートは変更していないため、
#       entropy<=RANK_9S_ENTROPY_MAX・mark3<=RANK_7S_MARK3_OVERLAP_MAX の2条件のうち、
#       不合格がちょうど1個（変更なし）。
#   S7/S9とは論理的に排他（全条件合格=S7/S9、ちょうど1条件のみ不合格=7A/9A）。
#   新7A(2条件)とのhonest全期間再検証で重複選出0件を確認済み
#   （scripts/exp_7a_2gate_redefinition_validation.py・2024-01〜2026-07）。
#   買い目 = 三連複 軸2車+残り流し（7車5点・9車7点。S7/S9と同一構造）。
#
# 【2026-07-31・7A 2ゲート化の honest全期間再検証】
#   旧7A(3ゲート・mark3含む): 4,691R・4.97件/日・的中42.8%・ROI81.4%・
#     月次ROI標準偏差22.0
#   新7A(2ゲート・mark3撤廃): 8,306R・8.81件/日・的中44.8%・ROI77.6%・
#     月次ROI標準偏差13.4
#   ROIは控除率75%を上回る水準を維持しつつ、的中率向上・変動縮小・
#   件数増（約1.8倍）を確認したため採用。
#
# 直近実力（旧数値・参考。2026-07-31の2ゲート化で7Aの実績は上記に更新）:
#   7A 約2.3〜4.0件/日 + S7 約1.15件/日 ≈ 7車合計 約3.5〜5件/日（旧設定時）
#   9A 約0.5〜1.7件/日 + S9 約0.25件/日 ≈ 9車合計 約0.75〜2件/日（変更なし）
# ═══════════════════════════════════════════════════════════════════════════

RANK_7A_STAKE = unit_stake(5)  # 2,000円/点（7車5点=10,000円/レース）
RANK_9A_STAKE = unit_stake(7)  # 1,400円/点（9車7点=9,800円/レース）


def rank_7a_daily_select(candidates: list[dict]) -> list[dict]:
    """7Aの選出: S7の2ゲート(axis_sum/entropy)のうちちょうど1つだけ不合格の候補。

    candidates: 各要素は最低限
      {"axis_sum": float, "entropy": float, "wt_overlap_n": int | None} を持つ dict。

    【2026-07-31改定】旧来はmark3も含む3ゲートだったが、S7自体がmark3ゲートを
    撤廃した（rank_7s_daily_select参照）ため2ゲートに揃えた。新S7との重複選出が
    ないことをhonest全期間で検算済み（本セクション冒頭コメント参照）。

    - wt_overlap_n ∈ {0,1} 必須（◎◯完全一致=2・マーク欠損=None は対象外、S7と同様）
    - **axis_sum だけが不合格**（entropy は合格）の候補のみ採用

    【2026-08-05改定】旧来は「不合格ちょうど1つ」だったため、
      A群 axis_sum だけ不合格（軸2車が堅い）  的中52-53% / 中央値0.78倍
      E群 entropy  だけ不合格（レースが荒れ）  的中30-32% / 中央値1.57倍
    という**性質が正反対の2群の混合**になっていた（同じラベルなのに当たり方も
    配当も一貫しない）。E群は 7SS（同一ライン条件付き）へ分離したため、
    7A は A群のみを指す。詳細は RANK_7SS_STAKE 定義部のコメント参照。

    returns 採用された候補のリスト（axis_sum昇順）。
    """
    pool = []
    for c in candidates:
        if c.get("wt_overlap_n") not in (0, 1):
            continue
        axis_ok = c["axis_sum"] <= RANK_7S_AXIS_SUM_MAX
        ent_ok = c.get("entropy", float("inf")) <= RANK_7S_ENTROPY_MAX
        if (not axis_ok) and ent_ok:
            pool.append(c)
    return sorted(pool, key=lambda c: c["axis_sum"])


def rank_7ss_daily_select(candidates: list[dict]) -> list[dict]:
    """7SSの選出: entropy だけ不合格 ∧ 軸2車が同一ライン。

    candidates: 各要素は最低限
      {"axis_sum": float, "entropy": float, "wt_overlap_n": int | None,
       "same_line": bool} を持つ dict。

    - wt_overlap_n ∈ {0,1} 必須（7S/7Aと同様）
    - axis_sum は合格（<=RANK_7S_AXIS_SUM_MAX）かつ entropy が不合格
    - **軸1と軸2が同一ライン**（`same_line`）。判定は rank_7ss_same_line()。
      候補生成側で埋める。**未取得は False 側に倒す**（推奨を増やさない）

    returns 採用された候補のリスト（entropy昇順＝荒れが小さい順）。
    """
    pool = []
    for c in candidates:
        if c.get("wt_overlap_n") not in (0, 1):
            continue
        if c["axis_sum"] > RANK_7S_AXIS_SUM_MAX:
            continue
        if c.get("entropy", float("inf")) <= RANK_7S_ENTROPY_MAX:
            continue
        if not c.get("same_line"):
            continue
        pool.append(c)
    return sorted(pool, key=lambda c: c.get("entropy", 0.0))


def rank_9a_daily_select(candidates: list[dict]) -> list[dict]:
    """9Aの選出: S9の2ゲート(entropy/mark3)のうちちょうど1つだけ不合格の候補。

    rank_7a_daily_select() の9車版。9車はaxis_sum閾値が未導入（S9同様）のため、
    entropy<=RANK_9S_ENTROPY_MAX・mark3<=RANK_7S_MARK3_OVERLAP_MAX の2条件のうち
    不合格がちょうど1個の候補のみ採用する。
    """
    pool = []
    for c in candidates:
        if c.get("wt_overlap_n") not in (0, 1):
            continue
        mark3 = c.get("wt_mark3_overlap_n")
        if mark3 is None:
            continue
        ent_ok = c.get("entropy", float("inf")) <= RANK_9S_ENTROPY_MAX
        mark3_ok = mark3 <= RANK_7S_MARK3_OVERLAP_MAX
        n_fail = (not ent_ok) + (not mark3_ok)
        if n_fail == 1:
            pool.append(c)
    return sorted(pool, key=lambda c: c["axis_sum"])


# ═══════════════════════════════════════════════════════════════════════════
# 7B（◎◯一致だが順序・相手で市場と不一致・三連複3点）— 2026-08-03 導入
#
# 背景（ユーザー方針）:
#   「予想家としての売りは単純な市場との不一致を避けることに価値を見出される
#     ため、7S/7Aの方針は正しい。ただし ◎◯ が一致してもある程度の配当を
#     見込める・的中率が他よりも高い・相手を絞ることができる時は価値がある。」
#
# 7S/7A は wt_overlap_n==2（軸2車がWT公式印◎◯と完全一致）を全除外している。
# しかしモデルが公式印へ収束する方向に進化した結果、この完全一致が多数派に
# なり、対象レースが構造的に枯渇した（overlap∈{0,1}の比率:
# 2024-01 68.6% → 2025年以降 18〜23%。7S+7A の選出は 39件/日 → 12件/日）。
# 2026-08-02・08-03 は overlap∈{0,1} のレースを1本残らず採用しても6件・5件で、
# entropy/axis_sum をどれだけ緩めても1件も増えない状態だった。
#
# 【検証: honest 月次凍結vintage・2025-01〜2026-08・579日・36,791候補】
#   scripts/exp_7s7a_overlap2_conditional_value.py / _sweep.py / _disagreement.py
#
#   ① overlap2 は「的中率は高いが配当が消えている」:
#        現行7S+7A(K=5): 13.42件/日 的中36.9% ROI78.3% 的中中央値5.9倍 ガミ率41.2%
#        overlap2 (K=5): 50.01件/日 的中56.3% ROI72.4% 的中中央値4.0倍 ガミ率60.2%
#      的中率は+19.4pt だが、当たっても賭け金を割る（ガミ）率が6割。
#   ② entropy上限 × 相手集中度 × 買い目点数Kの120セルを掃引したが、
#      ROIは全セル 72〜76% で完全に平坦（控除率の壁）。「高ROIの隠れ帯」は無い。
#      的中率と配当は「予測確率の集中度」という同一軸の両端で、同時に立たない。
#   ③ ただし**相手側で市場と不一致にする**と配当が戻る:
#        overlap2・相手上位3点（△込み）: 的中47.0% ROI73.6% ガミ42.4% 中央値3.4倍 20倍超175本
#        overlap2・相手上位3点（△除外）: 的中27.3% ROI74.7% ガミ10.8% 中央値6.1倍 20倍超435本
#      さらに順序不一致（モデルのpred_win最上位 ≠ WT◎・overlap2の11.2%）を
#      重ねると ROI が現行7S/7Aと同水準まで戻る。これを 7B として採用する。
#
# 【7B の実績（honest・2025-01〜2026-08）】
#     5.58件/日・的中25.2%・ROI78.7%（現行7S+7A は 78.3%）
#     ガミ率8.1%（現行41.2%）・的中中央値6.8倍（現行5.9倍）
#     20倍超64本・最高116.6倍
#     年次 2025:78.6% / 2026:78.8%、四半期7期すべて68〜89%で崩れなし
#     月次ROI<60% は20ヶ月で0回・月次ROI標準偏差14.3（現行7Sは13.3）
#
# 【重要な位置づけ】
#   ROI は改善しない（どの案も控除率75%の壁の周辺から出ない）。
#   本ランクは「ROI同等のまま件数・見せ場・ガミ率を改善する」ための増枠であり、
#   収支改善策ではない。[[keirin_clean_baseline_market_efficiency_2026_07_30]]。
#
# 【設計】
#   母集団 = 7車立て ∧ rank_7s_select_axis 成功 ∧ wt_overlap_n == 2
#            （7S/7A とは wt_overlap_n で論理的に排他。重複選出は起こり得ない）
#   ゲート = 順序不一致のみ（entropy/axis_sum ゲートは掛けない）。
#            entropy<=1.8329 を追加すると 3.54件/日・ROI79.8% と一見良くなるが、
#            月次ROI標準偏差が 14.3→24.9 に悪化し ROI60%割れ月が0→3回に増える
#            ため不採用（件数も足りない）。
#   買い目 = 三連複 軸2車 + 相手（残り5車から WT△(ana) を除外し pred_prob 上位
#            RANK_7B_LEGS 車）＝ 3点。7S/7A の5点流しとは点数が異なる。
#   日次上限 = なし（honest 5.58件/日で暴走の懸念がないため。7S の
#              RANK_7S_DAILY_CAP のような安全網は設けていない）。
# ═══════════════════════════════════════════════════════════════════════════

RANK_7B_NE = 7      # 対象車数（7車ちょうど・7S/7Aと同じ）
RANK_7B_LEGS = 3     # 相手の購入点数（△除外後の pred_prob 上位K車）
RANK_7B_STAKE = unit_stake(RANK_7B_LEGS)  # 3,300円/点（3点=9,900円/レース）


def rank_7b_order_disagree(win_probs: dict[int, float], wt_honmei: int | None) -> bool | None:
    """7Bの順序不一致判定: モデルの単勝予測 pred_win 最上位が WT◎ でないか。

    win_probs:  {frame_no: pred_win}（rank_7s_select_axis と同じ入力）
    wt_honmei:  prediction_mark==1（◎）の frame_no

    軸2車が◎◯と完全一致していても、モデルが「◯の方が強い」と見ているなら
    完全なコンセンサスではない、という部分的不一致の検出。

    returns True=不一致（7B対象）/ False=一致 / None=判定不能（◎欠損・要除外）
    """
    if wt_honmei is None or not win_probs:
        return None
    return max(win_probs, key=lambda f: win_probs[f]) != wt_honmei


def rank_7b_select_legs(
    others: list[int], top3_probs: dict[int, float], wt_ana: int | None,
    k: int = RANK_7B_LEGS,
) -> list[int]:
    """7Bの相手選択: 残り5車から WT△(ana) を除外し pred_prob 上位k車を返す。

    others:     軸2車を除いた残り車（通常5車）
    top3_probs: {frame_no: pred_prob}
    wt_ana:     prediction_mark==3（△）の frame_no。None なら除外しない。

    △を切ることが配当を戻す本体（検証: 的中中央値3.4倍→6.1倍・ガミ率42.4%→10.8%・
    20倍超175本→435本）。的中率は下がる（47.0%→27.3%）が、これは
    「市場が推す相手を買わない」という7B設計の意図そのものである。
    """
    ranked = sorted(others, key=lambda x: -top3_probs.get(x, 0.0))
    if wt_ana is not None:
        ranked = [x for x in ranked if x != wt_ana]
    return ranked[:k]


# ═══════════════════════════════════════════════════════════════════════════
# 7B — 2026-08-05 に**中身を全面的に入れ替えた**（ユーザー判断）
#
# 【旧7B（2026-08-03〜2026-08-05）】overlap==2 ∧ **order不一致**。
#   測ったすべての窓で控除率75%の壁を越えず廃止した:
#   確認窓(2024-07〜2025-06) 71.1% / 掃引窓(2025-07〜2026-07) 75.3% / live全期間 75.9%。
#   絞り込みも全滅（朝の時点で△の強さを測る量はどれも確認窓を越えず、最良候補ですら
#   +8.2pt [−2.1, +19.1] と有意差なし）。memory: keirin_7b_filter_rejected_2026_08_05
#
# 【新7B（本定義）】overlap==2 ∧ **order一致** ∧ **race_type=="準決勝"**。
#   7車レースの被覆マップを作った結果、現行ランクがどこも触っていない空白が
#   構造上ちょうど3つあると判明し、そのうち唯一3窓を生き残ったのがこれ。
#
#   ⚠️ **旧7Bとは order_disagree の向きが真逆**（不一致→一致）。同じ「7B」の名前で
#      **別戦略**であり、picks_history の旧7B行と成績を合算してはいけない。
#
#   | 窓 | n | 件/日 | 的中 | ROI |
#   |---|---|---|---|---|
#   | 掃引窓 2025-07〜2026-07 | 2,258 | 5.94 | 29.8% | 82.8%（4窓すべて75%以上）|
#   | 確認窓 2024-07〜2025-06 | 1,727 | 4.73 | 30.1% | 83.0%（同上）|
#   | 未使用期間 2026-07-16〜08-04 | 115 | 5.75 | 32.2% | 81.7% |
#
#   採用根拠は「差」ではなく **水準が ±0.7pt に収まっていること**。同じ未使用20日で
#   7SS 48.4% / 7S 51.1% / 7A 115.4% と他ランクが激しく振れ、土台（空白3全件）自体も
#   74.3→68.4% に悪化した中で、この定義だけが水準を保った。確認窓のブートストラップで
#   土台との差 +8.7pt [+1.2, +16.3] 有意。裾依存も全ランク中最小（上位5本の配当が
#   回収に占める割合 5.2%・上位10本を除いても ROI 76.1%）。
#
# 【✂️ 含めないもの — いずれも実測で落とした】
#   - **決勝**: 掃引窓 87.1%（4窓合格）→ 確認窓 **66.7%** と完全反転。準決勝と束ねると
#     全体が有意でなくなる（+5.4pt [−0.6, +11.6]）
#   - **「準決勝」の部分一致**: `race_type` には `"チャレンジ準決勝"`(掃引窓 ROI 77.7%)・
#     `"ガールズ準決勝"` が別値で存在する。**検証は完全一致でしか行っていない**ので
#     `in` 判定にすると未検証の母集団が約30%混入し、掃引窓ですら 82.8→81.7% に薄まる。
#     **必ず完全一致で判定すること**
#   - **空白3全件**（48.5件/日）: 確認窓 74.3% [71.6, 76.7] で壁を越えるとは言えない
#
# memory: keirin_7car_coverage_gaps_2026_08_05
# ═══════════════════════════════════════════════════════════════════════════

# 対象レース種別（**完全一致**。部分一致にしてはいけない・上記コメント参照）。
RANK_7B_RACE_TYPES = ("準決勝",)


# 7B の停止スイッチ。2026-08-07 に一度 True（廃止）にしたが、同日中に
# **「7C の下に置き、重複は 7C・独自レースだけ 7B」**へユーザー判断が変わったため
# False（稼働）へ戻した。優先順位は `netkeirin_submit_wt.RANK_CONFIGS` の定義順が正本。
RANK_7B_STOPPED = False


def rank_7b_daily_select(candidates: list[dict]) -> list[dict]:
    """7Bの選出: ◎◯完全一致 ∧ 順序も一致（市場と完全合意）∧ 準決勝。

    candidates: rank_7s_* と同じ生候補 dict のリスト。最低限
      {"wt_overlap_n": int|None, "order_disagree": bool|None, "race_type": str|None}
      を持つこと。

    - `wt_overlap_n == 2` 必須（7SS/7S/7A＝overlap∈{0,1} とは論理的に排他＝純増）
    - `order_disagree is False` 必須（モデル1位 == WT◎ ＝市場と完全合意）。
      **`is not True` ではなく `is False` で書く**: overlap==2 は ◎◯ が両方存在しないと
      成立しないため `order_disagree is None`（◎欠損）とは構造的に両立しない
      （掃引窓 18,440 件すべてが False であることを実測確認済み）。将来 overlap の
      定義が変わって None が混じったときに黙って通さないためのフェイルセーフでもある。
    - `race_type` が RANK_7B_RACE_TYPES に**完全一致**すること

    returns 採用された候補のリスト（entropy昇順・表示用の並び順のみ）。

    🔴 **2026-08-07: 7C より下の優先順位で運用する（ユーザー判断）。**
       7C との比較で、両方に傾斜配分をかけた公平な条件では 7C が実質的中率で
       上回った（同一母集団 2,203R: 7B 31.6% / 7C 39.0%）。一方で 7B は 7C が
       拾わないレースを **3.14件/日** 持ち、ROI も 1.2pt 高い。
       → **重複するレースは 7C に譲り、7B 独自のレースだけを 7B として出す。**
       実現は `netkeirin_submit_wt.RANK_CONFIGS` の定義順（＝入稿の優先順位）で
       **7C より後ろに置く**ことによる。netkeirin は1レース1商品なので、
       先に来たランクがそのレースを取り、後続は自動的に降りる。

       ⚠️ **候補生成の側では重複を排除しない**（7C が他ランクと重ならないのと同じ）。
          picks_history には両方の行が残り、成績はランクごとに独立して集計される。
          入稿されるのは優先順位で勝った側だけ。
    """
    return [] if RANK_7B_STOPPED else rank_7b_select_pool(candidates)


def rank_7b_select_pool(candidates: list[dict]) -> list[dict]:
    """7B の選出ゲート本体（廃止中も**検査は生かしておく**）。

    `rank_7b_daily_select` から切り出したのは、廃止で到達不能になったコードを
    そのまま置くとテストが当てられず、再開したときに壊れているのに気づけないため。
    """
    pool = [
        c for c in candidates
        if c.get("wt_overlap_n") == 2
        and c.get("order_disagree") is False
        and c.get("race_type") in RANK_7B_RACE_TYPES
    ]
    return sorted(pool, key=lambda c: c.get("entropy", float("inf")))


# ═══════════════════════════════════════════════════════════════════════════
# A（◎一致×波乱×別ライン先頭・二連単）戦略 — 2026-07-17 全廃
#
# 正規プロトコル（学習〜2025-03-31・検証2025-04-01〜2026-03-31の1年）の再検証で
# 検証最良 88.5-94.2%・100%超なし→棄却（exp_ranks_valtest.py / exp_axis_redesign.py）。
# → 2026-07-17 に候補生成・judge・採点を全停止し、picks_history の #7A 行は
# picks_history_a_archive へ退避。定数は過去スクリプト（backfill_a_rank_wt.py 等）の
# 互換のため残置。
# ═══════════════════════════════════════════════════════════════════════════

A_EX_MIN_ODDS = 5.0        # 買い目の二連単オッズ下限（未満はカット）
A_EX_MAX_ODDS = 50.0       # 買い目の二連単オッズ上限（以上はカット）
A_STAKE = 100              # 円/点（ペーパー）


def is_senbatsu(race_type: str | None) -> bool:
    """「選抜」系レース種別か（選抜/チャレンジ選抜/ガールズ選抜等）。"""
    return bool(race_type) and "選抜" in str(race_type)


def line_score_features(
    line_points: list[tuple[int | None, float | None]],
) -> tuple[float | None, int | None, bool | None]:
    """出走全車の (line_group, race_point) からライン構造特徴を返す。

    returns (avg_gap, n_lines, all_solo)
      - avg_gap: ライン別 race_point 平均の 1位 − 2位（ライン2本未満は None）
      - n_lines: ライン本数（line_group の distinct 数）
      - all_solo: 全員単騎（=ライン本数が車数と一致）か
    line_group 欠損車が1台でもあれば (None, None, None)（判定はフォールバック側）。
    """
    if not line_points:
        return None, None, None
    groups: dict[int, list[float]] = {}
    for lg, rp in line_points:
        if lg is None or rp is None:
            return None, None, None
        groups.setdefault(int(lg), []).append(float(rp))
    n_lines = len(groups)
    all_solo = n_lines == len(line_points)
    if n_lines < 2:
        return None, n_lines, all_solo
    means = sorted((sum(v) / len(v) for v in groups.values()), reverse=True)
    return round(means[0] - means[1], 3), n_lines, all_solo


def ss_policy(
    race_type: str | None,
    avg_gap: float | None = None,
    n_lines: int | None = None,
    all_solo: bool | None = None,
) -> tuple[str | None, int]:
    """SS(7PLUS_R) の購入ポリシー判定（2026-07-16〜: 選抜カットのみ）。

    ※ 旧S1（7PLUS_R）は 2026-07-16 に全廃。本関数は過去日再採点・
      フォールバック経路の互換のため残置。

    returns (skip_reason, stake_per_pt)
      - skip_reason: "選抜" / None（None=購入可）
      - stake_per_pt: SS_STAKE（増額は廃止・常に100円/点）
    ライン特徴引数（avg_gap/n_lines/all_solo）は 4分戦カット・格差増額の削除に伴い
    未使用（呼び出し側互換のため残置）。
    """
    if is_senbatsu(race_type):
        return "選抜", 0
    return None, SS_STAKE


# ═══════════════════════════════════════════════════════════════════════════
# 7SS（entropy不合格 × 軸2車が同一ライン・2026-08-05 新設）
#
# ⚠️ **この 7SS は 2026-08-02 に全廃した旧 RANK_7SS（波乱軸選出・穴レース検知・
#    モデル非依存の別戦略）とは無関係の別物**。名前だけを引き継いだ。
#    旧実装は「順位が名前で伝わること」を優先するユーザー判断で破棄した
#    （commit `7048db5` 以前の git 履歴から復元可能）。picks_history に旧7SSの
#    行は0件（16,298行は2026-08-02に削除・CSVへ退避済み）なので成績は混ざらない。
#
# ## 定義
#
#     wt_overlap_n ∈ {0,1}
#     ∧ axis_sum <= RANK_7S_AXIS_SUM_MAX      （7S と同じ）
#     ∧ entropy  >  RANK_7S_ENTROPY_MAX       （＝レースが荒れている側）
#     ∧ **軸1と軸2が同一ライン**（wt_entries.line_group が一致）
#
# 買い目は 7S/7A と同一（三連複 軸2車＋残り5車流し・5点500円）。
#
# ## 経緯：7A が性質の違う2群の混合だった
#
# 2026-08-05 の分解で、7A（不合格ちょうど1つ）は正反対の2群の混合と判明した:
#   A群 axis_sum だけ不合格（＝軸2車が堅い）  的中52-53% / 低配当(中央値0.78倍)
#   E群 entropy  だけ不合格（＝レースが荒れ）  的中30-32% / 高配当(中央値1.57倍)
# 同じラベルで出しているため利用者から見て当たり方も配当も一貫しなかった。
#
# ## 同一ライン条件の根拠
#
# 7S/7A では**二軸に負の相関**（実測 −2.7〜−2.9pt。独立仮定より両方3着内率が低い）
# がある。7車で3枠を奪い合うためだが、**その正体の一部がライン関係**だった。
# 別ラインの2車は競合するが、同一ラインなら連動して残る（競輪はライン戦）。
#
# 確認窓（2024-07〜2025-06・掃引に一度も使っていない4窓）:
#
#   E群 基準       7.34件/日  的中30.2%  ROI 76.8%  (82.3 76.3 67.4 81.4)
#   E群 同一ライン  1.90件/日  的中41.2%  ROI 85.9%  (94.1 86.5 88.7 74.3)
#   E群 別ライン    5.44件/日  的中26.7%  ROI 75.0%  (79.6 73.6 61.7 85.1) ← 除外する層
#
# **的中率 +11.0pt・ROI +9.1pt**。除外される別ライン側は7Aの中で最も弱い。
#
# ⚠️ **c4窓が 74.3% で「4窓すべて75%以上」を 0.7pt 割る**（ユーザー承知の上で採用）。
#    構成全体（7SS+7S+7A）では 79.2/84.2/83.8/80.8 と4窓すべて合格する。
#
# ⚠️ **A群には適用しない**。確認窓で同一ライン80.4% vs 別ライン78.7%、基準80.8%を
#    下回り適用根拠がない（掃引窓では+10ptに見えたが確認窓で消えた）。
#
# ## 構成全体（確認窓・件数は目標10〜15件/日に対して 11.87）
#
#   7SS  1.90件/日  的中41.2%  ROI 85.9%   ← 最上位
#   7S   3.68件/日  的中41.0%  ROI 84.4%
#   7A   6.28件/日  的中53.2%  ROI 80.8%
#   合計 11.87件/日 的中47.5%  ROI 82.0%  (79.2 84.2 83.8 80.8) ✓
#
# ROI が 7SS > 7S > 7A と単調になり「S が特別に良く A から下がる」並びが成立する。
# ═══════════════════════════════════════════════════════════════════════════

RANK_7SS_STAKE = unit_stake(5)   # 2,000円/点（5点=10,000円/レース。7S/7Aと同一）


def rank_7ss_same_line(axis1: int, axis2: int,
                       line_groups: dict[int, object] | None) -> bool:
    """軸1と軸2が同一ラインかを判定する。

    line_groups: {frame_no: wt_entries.line_group}。単騎や未取得は None/空文字。
    **どちらかが不明なら False**（不明を同一ラインとみなして推奨を増やさない）。
    """
    if not line_groups:
        return False
    g1, g2 = line_groups.get(axis1), line_groups.get(axis2)
    if g1 is None or g2 is None:
        return False
    s1, s2 = str(g1).strip(), str(g2).strip()
    return bool(s1) and s1 == s2



# ═══════════════════════════════════════════════════════════════════════════
# RANK_7H1 — 穴推奨「本命バスト型」（2026-08-06 新設・7車立て専用）
#
# 【命名体系】穴推奨モデルは `{車数}H{連番}` を表示ラベルとし、内部rank は
#   "RANK_" + ラベル、suffix は "#" + ラベル とする（既存6ランクと同一規則）。
#   H = Hole（穴）。既存の S/A/B（的中率重視・予想ベース系）とは系統を分ける。
#     7H1 = 本命バスト型（本モジュール）
#     7H2 = 三着穴型（1・2着は堅く3着だけ荒れる。未実装・予約）
#
# 【選別】3条件すべてを満たすレースだけを対象にする。
#   (1) 7車ちょうど
#   (2) 軸1（モデル1着率 pred_win 最上位）== WINTICKET公式印 ◎    …「市場と合意した本命」
#   (3) 抜け度（モデル1着率の1位−2位差） >= RANK_7H1_GAP_MIN      …「確実に抜けている」
#   (4) バスト確率（本命が4着以下になる確率）が当日の上位 RANK_7H1_TOP_FRAC
#
#   実測（honest walk-forward 38,085R・memory keirin_highpay_payout_ceiling_2026_08_06）:
#     母集団(2)      38,085R / バスト基準率 19.50% / AUC 0.6848
#     +(3) >=20pt    21,669R / バスト率 13.69%
#     +(4) 上位10%    2,167R / **3.22件/日・実バスト率 28.66%（lift 2.09）**
#     月次一貫性: 23/23ヶ月で基準超え（平均 +12.14pt・最悪 +5.14pt）
#
# 【買い目】本命が4着以下という前提なので3着以内は必ず残り6車で埋まる。
#   さらに **本命が飛ぶときは番手も一緒に飛ぶ**（本命ライン番手の bust時 1着率7.79%・
#   3着内率33.27% で全役割中最低）ため、**本命ラインを丸ごと落とす**。
#
#     プール = 別ライン勢 + 単騎（本命ラインを除いた車）をモデル3着内率順に r1..
#     三連単 = 1着: 別ライン先頭(最強ライン)を固定
#              2着: プールの r1・r2
#              3着: 本命を除く全5車（総流し）          → 8点
#     三連複 = プール上位5車BOX（最大10点）
#
#   実測（選別2,086R・100円単位の実購入）:
#     的中 18.70%（三複17.11 / 三単5.90）/ ROI 99.1% / 平均払戻 49,778円 /
#     30万円+ 16件（月0.8回）/ 平均購入額 9,388円
#     窓別 ROI: 掃引窓 94.9%(n=1,344) / 確認窓 113.2%(n=681)
#     ⚠️ 三連単側は裾依存（上位3本が回収の15.9%）。**ROIは80〜110%のレンジで読む**。
#
# 【9車は不採用】9車ではバスト予測 AUC 0.5967（7車0.6848・n=3,018で信頼できる）、
#   ROI 36〜61% で壁を大きく割る。**抜け度条件は9車では逆効果**で7車と正反対だった。
#   「7車の知見が9車へ移らない」事例（3ヘッド軸・9B・S9/9A再較正と同型）。
#
# 【検証済みの否定結果（再提案しない）】
#   - 残り6車の力関係によるセグメント別の買い目出し分け（拮抗度/強い単騎/
#     別ライン先頭の有無/本命ライン残存数の4次元）。標本3倍・自由度削減でも
#     確認窓で「常に三連単」を超えない
#   - 選別を緩めて標本を増やす（上位10%→30%で三連単ROIが 95→79% へ両窓一致で劣化）
#   - 本命を買い目から完全に外す版（確認窓で反転。本命は飛ぶときも3着には残る）
#   - 番手のみ除外（3番手以降も弱いので本命ラインは丸ごと落とすのが正しい）
# ═══════════════════════════════════════════════════════════════════════════

RANK_7H1_NE = 7                    # 対象車数（7車ちょうど）
RANK_7H1_GAP_MIN = 0.20            # 抜け度（モデル1着率の1位−2位差）の下限
# バスト確率の採用閾値。**日ごとの相対順位ではなく全体の絶対閾値**を使う。
# 検証（honest walk-forward）が「抜け度>=20pt 母集団 21,673件の上位10%」＝
# 90%点 0.2666 で切っており、日次の相対順位で切り直すと切り捨てにより
# 系統的に少なくなる（実測 3.22件/日 → 1.6件/日）ため。
# 絶対閾値なら開催規模がそのまま件数に反映される
# （実測の日次件数: 中央3件・p25 2件・p75 4件・最大11件・0件の日が7%）。
# ⚠️ この値はモデルの較正に依存する。モデルを再学習したら
#    `scripts/check_7h1_threshold.py` 相当で分布を確認し、必要なら更新すること。
RANK_7H1_BUST_PROB_MIN = 0.2666
RANK_7H1_BUDGET_TF = 7500          # 三連単の予算枠（円/レース）
RANK_7H1_BUDGET_TRIO = 2500        # 三連複の予算枠（円/レース）
# ⚠️ 同じ値のリテラル再定義にしない（RANK_7C_BUDGET と同じ理由・2026-08-08）。
RANK_7H1_BUDGET_CAP = RACE_BUDGET  # 1レースの購入上限（円）
RANK_7H1_UNIT = STAKE_UNIT         # 最低賭け金単位（円）
RANK_7H1_TRIO_POOL_MAX = 5         # 三連複BOXに使うプール上限車数（→最大10点）
RANK_7H1_TF_SECOND_N = 2           # 三連単の2着に使うプール上位の車数


def rank_7h1_pool(others: list[int], roles: dict[int, str]) -> list[int]:
    """本命ラインを落とした購入プール（別ライン勢＋単騎）を返す。

    others はモデル3着内率の降順で渡すこと（順序をそのまま引き継ぐ）。
    """
    from .preprocessing.favbust_features import FAV_LINE_ROLES
    return [f for f in others if roles.get(f) not in FAV_LINE_ROLES]


def rank_7h1_build_legs(others: list[int],
                        roles: dict[int, str]) -> tuple[list[frozenset], list[str]]:
    """(三連複の目, 三連単の目) を返す。組めない場合は空リスト。

    Args:
        others: 本命を除く6車を**モデル3着内率の降順**で並べたもの
        roles:  `favbust_features.roles_of()` の戻り値
    """
    from .preprocessing.favbust_features import ROLE_LEAD_TOP

    pool = rank_7h1_pool(others, roles)
    if len(pool) < 3:
        return [], []
    trio = [frozenset(c)
            for c in _combinations(pool[:RANK_7H1_TRIO_POOL_MAX], 3)]
    lead = next((f for f in others if roles.get(f) == ROLE_LEAD_TOP), None)
    if lead is None:
        # 別ラインの先頭が存在しない（全員単騎など）レースは対象外にする。
        # 検証では成立率96.3%で、残り3.7%は買い目が定義できない。
        return trio, []
    rest = [f for f in pool if f != lead]
    tf = [f"{lead}-{a}-{c}" for a in rest[:RANK_7H1_TF_SECOND_N]
          for c in others if c not in (lead, a)]
    return trio, tf


def rank_7h1_unit(budget: int, n_legs: int) -> int:
    """枠内の100円単位・均等配分の1点あたり金額。100円未満なら0（買わない）。"""
    if n_legs <= 0:
        return 0
    u = (budget // n_legs) // RANK_7H1_UNIT * RANK_7H1_UNIT
    return u if u >= RANK_7H1_UNIT else 0


# 三連単の1点あたり（円）。2026-08-07 ユーザー指定で 900円前後の均等割りから固定 500円へ。
# 三連単の枠を薄くして残りを三連複へ回す設計で、実測（2,414R・最終オッズ）:
#   実質的中 9.94% → **18.31%**（的中18.35% にほぼ一致＝ガミがほぼ消える）
#   ROI 80.54% → **86.85%**
# ⚠️ 引き換えに**最高払戻が 1,148,340円 → 633,000円**へ下がる（三連単が44%薄くなる）。
#    「三連単で大きな配当を狙う」性格は half になる。ユーザー了承済み。
RANK_7H1_TF_UNIT = 500


def rank_7h1_trio_budget(n_tf: int, cap: int = RANK_7H1_BUDGET_CAP,
                         tf_unit: int = RANK_7H1_TF_UNIT) -> int:
    """三連単を 1点 `tf_unit` 円で買った残り＝三連複へ回す予算。

    現行の 7車立てなら 三連単8点×500 = 4,000円 → 三連複へ 6,000円。
    """
    return max(0, cap - tf_unit * max(n_tf, 0))


def rank_7h1_trio_stakes(trio_legs: list, trio_odds: dict | None, n_tf: int,
                         unit: int = RANK_7H1_UNIT) -> dict:
    """三連複の目ごとの賭け金。**入稿時点のオッズで払戻が等しくなるよう配分**する。

    trio_odds が無い（朝の板が買う目すべてに揃わない）場合は**均等**へ落とす
    （ユーザー指定の (a) フォールバック）。7H1 は穴狙いで人気薄を買うため
    朝の板が埋まるのは実測 53.3% しかない。

    ⚠️ 端数は `allocate_budget` の規則（想定払戻が最小の点）に従って配る。
       均等割りの切り捨てで捨てていた分（実測 平均613円/レース）も使い切る。
    """
    from .stake_allocation import allocate_budget

    if not trio_legs:
        return {}
    budget = rank_7h1_trio_budget(n_tf)
    if budget < unit * len(trio_legs):
        return {leg: unit for leg in trio_legs}
    usable = (trio_odds and all(trio_odds.get(leg) for leg in trio_legs))
    w = ({leg: 1.0 / trio_odds[leg] for leg in trio_legs} if usable
         else {leg: 1.0 for leg in trio_legs})
    return allocate_budget(w, budget=budget, unit=unit)


def rank_7h1_stakes(n_trio: int, n_tf: int) -> tuple[int, int, int]:
    """(三連複の1点あたり, 三連単の1点あたり, 合計購入額) を返す。

    **合計は必ず RANK_7H1_BUDGET_CAP 以下**になる（枠を分けて切り捨てるため）。
    """
    ut = rank_7h1_unit(RANK_7H1_BUDGET_TRIO, n_trio)
    uf = rank_7h1_unit(RANK_7H1_BUDGET_TF, n_tf)
    total = ut * n_trio + uf * n_tf
    if total > RANK_7H1_BUDGET_CAP:      # 到達しない想定だが不変条件として守る
        raise ValueError(f"7H1 の購入額が上限を超えました: {total}円")
    return ut, uf, total


def rank_7h1_daily_select(candidates: list[dict]) -> list[dict]:
    """当日の候補から 7H1 を選出する。

    candidates の各要素に必要なキー:
      `n_entries`(=7) / `gap12`（抜け度） / `bust_prob`（バスト確率） /
      `legs_trio` / `legs_tf`（買い目。空なら除外）

    **選別は `RANK_7H1_BUST_PROB_MIN` の絶対閾値で行う**（日ごとの相対順位ではない）。
    理由は同定数のコメント参照。0件の日があるのは正常（実測で7%の日が0件）。
    """
    elig = [c for c in candidates
            if c.get("n_entries") == RANK_7H1_NE
            and c.get("gap12") is not None
            and float(c["gap12"]) >= RANK_7H1_GAP_MIN
            and c.get("bust_prob") is not None
            and float(c["bust_prob"]) >= RANK_7H1_BUST_PROB_MIN
            and c.get("legs_trio") and c.get("legs_tf")]
    elig.sort(key=lambda c: -float(c["bust_prob"]))
    return elig


# ═══════════════════════════════════════════════════════════════════════════
# RANK_7C — ベースモデル「終日の二軸」（2026-08-07 新設・7車立て専用）
#
# 【why】既存6ランクは合計 13.1件/日 しか対象がなく、しかも二軸的中率は
#   7S 41.8% / 7SS 45.0% / 7A 52.6% / 7B 57.0%（加重 約52%）＝**的中体験が薄い**。
#   7S/7A は「軸2車の信頼が低い＝配当が高い」側を意図的に取るランクなので、
#   その**逆側（軸2車が堅い側）**を取るベースを1本置く。両者は構造的に補完関係。
#
# 【目的関数】**二軸的中率 = P(軸2車がともに3着内)**。
#   7車立ての三連複2軸総流し5点は3列目が総流しなので、その的中率と**数学的に同値**。
#   ROI は目標にしない（ユーザー判断: ガミは購入者の金額配分に委ねる）。
#
# 【選別】ただ1つの条件。**モデル3着内率の上位2車の合計 >= RANK_7C_P3_SUM_MIN**。
#   軸2車もその上位2車。オッズを一切使わないので朝8:00の入稿に間に合う。
#
# 【選別】2条件。どちらもオッズ不使用（朝8:00に確定する）。
#   (1) モデル3着内率の**上位2車の合計 >= RANK_7C_P3_SUM_MIN**
#   (2) 相手（下記の足切り後）が **RANK_7C_LEGS_MIN 点以上**
#
# 【買い目】三連複 軸2車（=上位2車）+ 相手（3着内率 >= RANK_7C_LEG_P3_MIN の車のみ）。
#   1レース RANK_7C_BUDGET 円を点数で均等割り（100円単位切捨て）。実測 4点63%/5点37%。
#
#   実測（honest walk-forward・p3 は月次凍結vintageの予測）:
#     | 窓 | 件/日 | 全レース比 | 二軸的中 | 的中/日 | ROI | 低配当的中(<=2倍) |
#     |---|---|---|---|---|---|---|
#     | 評価窓 2025-07〜2026-08 (400日) | 23.48 | 30.8% | **58.08%** | 13.64 | 77.3% | 2.10件/日 |
#     | 確認窓 2024-07〜2025-06 (365日) | 22.47 | 29.4% | **59.42%** | 13.35 | 77.8% | 2.08件/日 |
#   四半期別 的中 55.8〜62.0% / ROI 74.3〜79.0%。**0件の日は無い**
#   （日次件数 中央値24・最小9・最大39）。的中時平均配当 5.82倍。
#   設計は評価窓で行い、**確認窓でも件数・的中率・ROI・低配当率すべて再現した**。
#
# 【✂️ ガミ対策として検証して落としたもの — 再提案しない】
#   - **上位2車の合計に上限を付けて堅いレースを見送る**: ROI がほぼ動かず件数だけ失う
#     （上限なし 77.3/78.8% → 上限1.70 で 76.8/80.3% と**両窓で符号が逆**）。
#   - **上位3車の合計に上限**（連続量では最良の低配当予測子）: 件数を揃えると
#     低配当削減は18〜30%止まりで的中率も落ちる。**点数ゲート(2)に完敗**。
#   - **実質的中率（払戻>=投資）はどの合計帯でも 18〜24% で横ばい**。
#     合計が上がると的中率は 38.5→88.2% と伸びるのに実質的中は動かない＝
#     **増えた的中はほぼ全部ガミ**。これは控除率の壁そのもので買い目の形では超えない。
#     到達点は ROI 80〜82%（相手を1〜2点まで絞った場合・的中率は44〜52%へ低下）。
#   - **発走15分前の最終オッズで切れば低配当はほぼ全消し**にできるが、netkeirin は
#     8:00 入稿済みなので**商品からは外せない**。picks_history/Web の見送り記録に限る。
#
# 【✂️ 検証して落としたもの — 再提案しない】
#   - **オッズを選別/軸に使う**: 最終オッズなら三連複の市場確率上位5組で的中65.1%と
#     強いが、**朝オッズだと52.1%まで落ちて現行モデル(53.5%)を下回る**
#     （上位5組の一致 3.30/5点）。朝の板が薄く順位が確定していない。
#     運用要件（直前まで監視できない）とも合致するため恒久的に不採用。
#   - **「2位と3位の差」（＝2車が抜けている）**: 65.47%。差(gap)は「1・2位が3位から
#     離れている」だけで、その2車が絶対的に強いことを要求しないため 4.4pt 劣る。
#   - **「3位と4位の差」（＝3車が抜けている）**: 62.10%。3車拮抗は上位2車が突出して
#     いないということなので、二軸的中は**全体平均53.5%を下回る47.7%**。直感と逆。
#   - **上位2車の"積"**: 合計と実質同一（どの網羅率でも差0.1pt未満）。式が単純な合計を採る。
#   - **エントロピー / 上位3車合計 / 2位の高さ**: 68.6 / 67.6 / 68.2% でいずれも劣る。
#   - **条件分け**（同一ライン・分戦数・グレード・種別17区分・バンク長・競走得点の
#     ばらつき）: スコア5分位の**中**では差が残らない＝選別値が既に吸収している。
#     ユーザー方針「汎用条件で足りるなら条件分けしない」に合致。
#   - **ペアモデル**（21ペアをLightGBMで直接スコア）: 二軸的中 +1.3pt は出る
#     （53.48→54.84% 全レース時、95%CI [+0.93,+1.76]）が、月次vintage学習・配布・
#     backfill が必要。**第2弾として保留**（本ランクの閾値を差し替えるだけで移行可能）。
#
# 【重複の扱い】7C の母集団は「軸選定に成功した全7車レース」で、既存ランクとは
#   **論理的に排他ではない**（7SS/7S/7A/7B が wt_overlap_n で排他なのとは違う）。
#   `picks_history.race_key` は `{レースキー}#{suffix}` 形式なので**同一レースに
#   複数ランクの行を持てる**（実データでも 7H1 と 7B が共存している）。したがって
#   **候補生成・記録の段階では重複を排除しない**（ユーザー判断: 重なりは気にしない）。
#   重複排除は **netkeirin 入稿でのみ**行う（1レース1商品という外部仕様のため）。
#   優先順位は `netkeirin_submit_wt.RANK_ORDER`（7H1 > 7SS > 7S > 7A > 7C > 7B）。
#   実測の重なりは 2.4〜3.2件/日で、入稿に残る 7C は 16.7件/日。
#
# memory: keirin_base_model_two_axis_2026_08_07
# ═══════════════════════════════════════════════════════════════════════════

RANK_7C_NE = 7        # 対象車数（7車ちょうど）


def rank_7c_select_axis(top3_probs: dict[int, float]) -> tuple[int, int, float] | None:
    """7Cの軸2車を選ぶ: モデル3着内率の上位2車と、その合計を返す。

    top3_probs: {frame_no: pred_top3}（レース内全車分・スケール不問だが
      RANK_7C_P3_SUM_MIN は 0-1 スケール前提なので呼び出し側で揃えること）

    3ヘッド軸（rank_7s_select_axis）とは**別の選び方**である点に注意。
    全7車で測ると両者の二軸的中率はほぼ同じ（53.48% vs 53.53%）だが、
    7C は「上位2車の合計」で選別するランクなので、選別値と軸を同じ量から
    導くほうが定義として一貫する（軸だけ3ヘッドにしても的中率は変わらない）。

    returns (axis1, axis2, p3_sum) or None（車数不足）。
    axis1 が上位・axis2 が次点。同値のときは車番の小さい方を上位とする。
    """
    if not top3_probs or len(top3_probs) < 2:
        return None
    ranked = sorted(top3_probs, key=lambda f: (-top3_probs[f], f))
    a1, a2 = ranked[0], ranked[1]
    return a1, a2, top3_probs[a1] + top3_probs[a2]


def rank_7c_is_lowpay_pattern(
    top3_probs: dict[int, float], line_groups: dict[int, object] | None,
    gap34_min: float = RANK_7C_LOWPAY_GAP34_MIN,
) -> bool:
    """7Cの低配当パターン判定: 上位3車が抜けている ∧ その3車が同一ライン。

    top3_probs:  {frame_no: pred_top3}（0-1 スケール・レース内全車）
    line_groups: {frame_no: wt_entries.line_group}。**不明なら False**
      （情報が無いことを理由に推奨を減らさない＝安全側は「買う」）。

    True なら見送る。根拠は RANK_7C_LOWPAY_GAP34_MIN 定義部のコメント参照。
    ⚠️ 片方だけの条件で切ってはいけない（的中率の犠牲が大きい）。
    """
    if not line_groups or len(top3_probs) < 4:
        return False
    ranked = sorted(top3_probs, key=lambda f: (-top3_probs[f], f))
    if top3_probs[ranked[2]] - top3_probs[ranked[3]] < gap34_min:
        return False
    gs = [line_groups.get(f) for f in ranked[:3]]
    if any(g is None for g in gs):
        return False
    ss = [str(g).strip() for g in gs]
    return bool(ss[0]) and ss[0] == ss[1] == ss[2]


def rank_7c_daily_select(candidates: list[dict]) -> list[dict]:
    """7Cの選出: 上位2車の3着内率合計が閾値以上 ∧ 相手が RANK_7C_LEGS_MIN 点以上
    ∧ 低配当パターン（`lowpay_pattern`）でないこと。

    candidates: rank_7s_* と同じ生候補 dict のリスト。最低限
      {"p3_sum_top2": float, "legs_7c": list[int]} を持つこと
      （`rank_7c_select_axis` / `rank_7c_select_legs` の結果を候補生成時に載せておく）。
      `lowpay_pattern` は `rank_7c_is_lowpay_pattern()` の結果。**欠けていたら
      False 扱い**（旧形式の候補JSONを読んでも落ちないようにするため）。

    **他ランクとの重複は排除しない**（上記セクションコメント参照）。
    日次件数の上限も設けない（実測 最大39件/日で暴走の懸念がないため）。

    returns 採用された候補のリスト（p3_sum_top2 の降順＝自信のある順）。
    """
    return sorted(
        (c for c in candidates
         if c.get("p3_sum_top2") is not None
         and float(c["p3_sum_top2"]) >= RANK_7C_P3_SUM_MIN
         and len(c.get("legs_7c") or []) >= RANK_7C_LEGS_MIN
         and not c.get("lowpay_pattern")),
        key=lambda c: -float(c["p3_sum_top2"]),
    )


def rank_7c_select_legs(
    others: list[int], top3_probs: dict[int, float],
    p3_min: float = RANK_7C_LEG_P3_MIN,
) -> list[int]:
    """7Cの相手選択: 3着内率が p3_min 以上の車だけを買う（点数可変）。

    others:     軸2車を除いた残り車（通常5車。欠車時は盤面に残った車だけを渡す）
    top3_probs: {frame_no: pred_top3}（0-1 スケール）

    「3着内に入れなそうな車を外す」ための足切り。返す点数が
    RANK_7C_LEGS_MIN 未満なら**そのレースは買わない**（判定は呼び出し側）。
    採用構成では実測 4点63% / 5点37%（平均4.37点）。

    ⚠️ ここで**最低1車を無理に残してはいけない**。「相手が少ししか残らない」
       こと自体が「配当が付かない」の指標なので、埋め合わせると効果が消える。
    ⚠️ 相対比（最強相手の◯%以上）やギャップ切りの方が ROI は高い（78〜82%）が、
       的中率が 43〜62% まで落ちる。**的中体験を優先するユーザー判断で絶対15%を採用**。

    returns 買う相手のリスト（pred_top3 の降順）。
    """
    ranked = sorted(others, key=lambda x: (-top3_probs.get(x, 0.0), x))
    return [x for x in ranked if top3_probs.get(x, 0.0) >= p3_min]


def rank_7c_unit_stake(n_legs: int, budget: int = RANK_7C_BUDGET,
                       unit: int = RANK_7C_UNIT) -> int:
    """7Cの1点あたり賭け金。全ランク共通の `unit_stake` へ委譲する。

    2026-08-07 に全ランクが予算枠方式へ統一されたため、7C 専用の実装ではなくなった。
    名前は既存の呼び出し元との互換のために残している。
    """
    return unit_stake(n_legs, budget, unit)


# ═══════════════════════════════════════════════════════════════════════════
# 集計対象ランクの単一正本（2026-07-31 新設・是正タスク B-6 + C-1）
#
# 背景: 「集計対象ランクのハードコードされたリスト」が
#   - scripts/notify_results_wt.py::_query_stats のIN句（月次/年次サマリー）
#   - scripts/notify_results_wt.py::_PAPER_SUFFIXES（picks_history上書き保護）
#   - scripts/save_model_eval.py::PAPER_RANKS（model_evaluation保存）
#   - scripts/live_report_wt.py::RANKS/RANK_LABELS（live実測レポート）
# の計4箇所で独立に保守されており、内容が食い違う事故が3回発生した
# （7PLUS_R全廃後3週間 _query_stats が0件のまま放置・一部ランクのみ合算・
#  そして7SS新設(commit dc89f14)がpicks_historyに16,273行を投入したにも
#  かかわらず _query_stats のIN句に追加されず月次/年次サマリーに一切反映
#  されなかった事故＝本セクション導入の直接の契機）。
#
# 上記4箇所は全てここ（CURRENT_PAPER_RANKS / ABOLISHED_PAPER_RANKS）を参照し、
# 独自にランク一覧をハードコードしないこと。新ランクの追加/廃止時は、この
# セクションのみを更新すればよい（他4ファイルへの追随修正は不要）設計とする。
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PaperRankSpec:
    """1つの現行ペーパーランクの集計メタデータ（picks_history.rank に対応）。"""

    rank: str              # picks_history.rank の内部値（例: "RANK_7S"）
    suffix: str            # race_key の "#" サフィックス（例: "#7S"）
    label: str             # 表示ラベル（例: "7S"）
    in_header_total: bool  # notify_results_wt.py の[7+車]ヘッダー合計(p7b/p7r/p7h・n7)に含めるか
    in_live_report: bool   # live_report_wt.py の集計対象(RANKS)に含めるか


# 現行4ランク（2026-07-31 内部rank/suffixをRANK_+表示ラベル方式へ全面改名。
# 表示ラベル自体は変更なし。件数・期間・ROIは旧名時点の実績値）。
# 並び順は Discord 表示・各参照先ファイルでの反復順とも一致させる。
#
# 命名規則: 内部rank = "RANK_" + 表示ラベル、suffix = "#" + 表示ラベル
# （3表現が完全に1対1）。旧名は S7/S9 のみ suffix が "#7S7"/"#9S9" と表示ラベル
# 非対称だったため、今回の改名でその不揃いを是正した（7A/9A/7SSは元々規則に
# 合致していたため suffix・表示ラベルとも変更なし）。
#
#   新rank    新suffix  表示ラベル  旧rank(参考)  旧suffix(参考)  件数     期間                    ROI
#   RANK_7S   #7S       7S        SEVEN_S7      #7S7            6,572   2024-01-01〜2026-07-31  79.1%
#   RANK_7A   #7A       7A        SEVEN_7A      #7A(変更なし)  11,419   2024-01-01〜2026-07-31  77.4%
#   RANK_9S   #9S       9S        NINE_S9       #9S9              109   2024-01-11〜2026-06-12  79.2%
#   RANK_9A   #9A       9A        NINE_9A       #9A(変更なし)     967   2024-01-05〜2026-07-23  71.9%
#   RANK_7SS  #7SS      7SS       SEVEN_SS      #7SS(変更なし) 16,273   2022-12-01〜2026-07-30  73.5%
#
# 旧名→新名の機械的な参照が必要な場合（過去CSV/ログの読み解き等）は
# LEGACY_RANK_NAME_MAP / LEGACY_SUFFIX_MAP を参照すること。
#
# in_header_total: RANK_7SのみTrue（ヘッダー合計 p7b/p7r/p7h・n7 に算入する既存
#   方針。kiseki Webサマリーのトップラインと揃える）。RANK_7A/RANK_9S/RANK_9A は
#   境界ランク・独立ランクとして別集計する既存方針のためFalse。
# in_live_report: 現行4ランクは全てTrue（live_report_wt.py はモデル系ランクの
#   採否判断ツール）。唯一Falseだった RANK_7SS はモデル非依存の別戦略だったが
#   2026-08-02 に全廃したため CURRENT から除去した（ABOLISHED_PAPER_RANKS 参照）。
#   RANK_7B   #7B       7B        （新設・旧名なし）                 —      2026-08-03〜            —
CURRENT_PAPER_RANKS: tuple[PaperRankSpec, ...] = (
    # 2026-08-05 新設。旧 RANK_7SS(波乱軸選出・2026-08-02全廃)とは**無関係の別物**で
    # 名前だけを引き継いだ。picks_history に旧7SSの行は0件なので成績は混ざらない。
    PaperRankSpec("RANK_7SS", "#7SS", "7SS", in_header_total=True,  in_live_report=True),
    PaperRankSpec("RANK_7S",  "#7S",  "7S",  in_header_total=True,  in_live_report=True),
    PaperRankSpec("RANK_7A",  "#7A",  "7A",  in_header_total=False, in_live_report=True),
    PaperRankSpec("RANK_7B",  "#7B",  "7B",  in_header_total=False, in_live_report=True),
    PaperRankSpec("RANK_9S",  "#9S",  "9S",  in_header_total=False, in_live_report=True),
    PaperRankSpec("RANK_9A",  "#9A",  "9A",  in_header_total=False, in_live_report=True),
    # ベースモデル（2026-08-07〜）。終日を対象にする的中体験の土台で、
    # 既存の厳選ランクとは母集団の性格が違うため in_header_total=False。
    PaperRankSpec("RANK_7C",  "#7C",  "7C",  in_header_total=False, in_live_report=True),
    # 穴推奨系（2026-08-06〜）。既存6ランク（予想ベース・的中率重視）とは
    # 目的が違うため in_header_total=False（ヘッダー合計に混ぜない）。
    PaperRankSpec("RANK_7H1", "#7H1", "7H1", in_header_total=False, in_live_report=True),
)

# 旧名(2026-07-31改名前)→新名のマッピング（過去のCSVバックアップ・Discordログ・
# netkeirin_submissions等、DB移行対象外の履歴データを読み解く際にのみ使用する。
# 本番コードから通常参照する必要はない（新規ロジックは必ずCURRENT_PAPER_RANKSの
# 新名を直接使うこと）。表示ラベルは変更していないため LEGACY_LABEL_MAP は無い。
LEGACY_RANK_NAME_MAP: dict[str, str] = {
    "SEVEN_S7": "RANK_7S",
    "SEVEN_7A": "RANK_7A",
    "NINE_S9": "RANK_9S",
    "NINE_9A": "RANK_9A",
    "SEVEN_SS": "RANK_7SS",
}
LEGACY_SUFFIX_MAP: dict[str, str] = {
    "#7S7": "#7S",
    "#7A": "#7A",
    "#9S9": "#9S",
    "#9A": "#9A",
    "#7SS": "#7SS",
}


@dataclass(frozen=True)
class AbolishedRankSpec:
    """全廃済みランク（picks_history には存在しない）。誤って復活させないための
    ブラックリスト（CLAUDE.md が警告する「ランク全廃時は候補生成/ライブ判定/
    欠損自動補完の3箇所すべてを止める必要がある」教訓と同型の再発防止）。
    """

    rank: str             # picks_history.rank の内部値（廃止後は0件のはず）
    suffix: str | None    # "#"サフィックス方式のpicks_history上書き保護を使っていた場合のみ値あり
    note: str             # 廃止理由・廃止日


ABOLISHED_PAPER_RANKS: tuple[AbolishedRankSpec, ...] = (
    # ⚠️ 旧 RANK_7SS（波乱軸選出・穴レース検知／モデル非依存の別戦略）は
    #    2026-08-02 に全廃（live n=16,298・ROI73.5%で控除率割れ）。当初は
    #    「将来の再設定に備えて」判定ロジックを残置していたが、2026-08-05 に
    #    **同じ "7SS" の名前を別戦略（entropy不合格×同一ライン）へ充てた**ため、
    #    残置の意味が消え、むしろ新旧の取り違え事故の元になることから
    #    ユーザー判断で判定ロジックごと破棄した（commit `7048db5` 以前の
    #    git 履歴から復元可能）。
    #    **したがって RANK_7SS は ABOLISHED ではなく CURRENT に存在する。**
    #    picks_history の旧7SS行は 2026-08-02 に16,298行削除済み（0件）で、
    #    バックアップは data/backup/picks_history_rank_7ss_before_abolition_20260802.csv。
    #    このCSVだけは旧定義の成績なので、新7SSと合算してはいけない。
    AbolishedRankSpec("SEVEN_S1", "#7S1",
                       "win軸1着固定×3着内モデル相手2車・三連単2点流し（2026-07-31全廃）"),
    AbolishedRankSpec("SIX_S1", "#6S1",
                       "6車三連単 m1→m2→{m3,m4}（2026-07-17全廃）"),
    AbolishedRankSpec("7PLUS_R", None,
                       "三連複レース単位min(全目)>=7全目購入・旧称SS（2026-07-16全廃）"),
    AbolishedRankSpec("7PLUS_U", None, "波乱ライン連れ込み（2026-07-21全廃）"),
    AbolishedRankSpec("7PLUS_M", None, "◎不一致×軸信頼（2026-07-21全廃）"),
    AbolishedRankSpec("7PLUS_ST", None, "三連単1着固定F（2026-07-15全廃）"),
    AbolishedRankSpec("7PLUS_STP", None, "三連単1着固定F+（2026-07-15全廃）"),
)

# 廃止済みランクの内部rank名の集合（frozenset）。CURRENT_PAPER_RANKSに含まれて
# いないことをテストで機械的に検証する（ブラックリスト側）。
ABOLISHED_PAPER_RANK_NAMES: frozenset[str] = frozenset(s.rank for s in ABOLISHED_PAPER_RANKS)
