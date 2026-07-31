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
# 選出 = 当日の該当レースをaxis_sum昇順に並べ、上位 S7_DAILY_TOP_N 件を採用
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
# S7_DAILY_TOP_N件・重なり2は完全除外という選出方式へ変更（1日の採用本数は
# 重なり0の発生数に応じて可変・honest全期間で平均10.77R/日）。
# honest全期間再構築（この方式）: 9,927R（922日・10.77R/日）・的中36.3%・
# **ROI131.3%**（旧方式の128.1%から改善）。内訳: 重なり0(943R)的中39.4%/ROI232.8%・
# 重なり1(8984R)的中36.0%/ROI120.6%。
# ═══════════════════════════════════════════════════════════════════════════

S7_NE = 7                  # 対象車数（7車ちょうど）
S7_STAKE = 100             # 円/点（ペーパー・5点=500円/レース）

# 三連複が安くなりやすい（極端な人気決着になりやすい）レースの除外上限。
#
# 【2026-07-31改定】方針転換: 「まずは十分な的中率の上での安定したROI確保」
# （ユーザー方針）を目的に、月次凍結vintageモデルでのhonest全期間検証
# （2024-01-01〜2026-07-31・31ヶ月・配分はuniform固定・
#   scripts/exp_s7_cdf_regime_full_period.py）で再較正した。
#   現行(axis_sum<=1.3・mark3<=1併用): 576R・0.61件/日・的中34.0%・ROI78.5%・
#     月次ROI標準偏差43.7（月次0%回数1）
#   新設定(axis_sum<=1.5・mark3ゲート撤廃＝下記s7_daily_select参照):
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
S7_AXIS_SUM_MAX = 1.5

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
# 採用ペースは平均2.56件/日（S7_AXIS_SUM_MAX等の既存ゲートは全て維持のまま）。
S7_ENTROPY_MAX = 1.8329

# 日次合計の上限（entropy昇順で採用・2026-07-26再導入）。
# 件数capをentropyゲートに置換した初日（2026-07-26）、entropyフィールドを
# 持たない旧形式の生候補JSON（デプロイ前に生成された朝バッチ分）が
# s7_daily_select() の `c.get("entropy", 0.0)` フォールバックにより
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
S7_DAILY_CAP = 12


def s7_field_entropy(top3_probs: dict[int, float]) -> float:
    """レース全体（出走7車）の指数エントロピー（占有率ベースの拮抗度）を返す。

    top3_probs: {frame_no: pred_prob}（s7_select_axis と同じ入力）。
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


def s7_select_axis(
    win_probs: dict[int, float], top3_probs: dict[int, float],
) -> tuple[int, int, float] | None:
    """S7の軸2車とaxis_sum（波乱度指数の元）を選定する。

    win_probs / top3_probs: {frame_no: 確率(0-1 or pct、比較にのみ使うのでスケール不問)}
      レース内全車分。

    軸選定: win_probs上位3 ∩ top3_probs上位3 の重なり車から、
      重なり>=2ならtop3_probs上位2、重なり==1ならその1車+残りのtop3_probs最上位。

    returns (axis1, axis2, axis_sum) or None（重なり0・データ不足で選定不能）。
    axis_sum は axis1/axis2 の top3_probs 合計（波乱度指数・低いほど波乱寄り）。
    """
    if not win_probs or not top3_probs or len(win_probs) < 3 or len(top3_probs) < 3:
        return None
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


def s7_wt_overlap_n(
    axis1: int, axis2: int, wt_honmei: int | None, wt_taikou: int | None,
) -> int | None:
    """S7の軸2車とWINTICKET公式予想の◎◯（honmei/taikou）との重なり数を返す。

    wt_honmei: prediction_mark==1（◎）の frame_no。
    wt_taikou: prediction_mark==2（◯）の frame_no。
    いずれか欠損時は None（重なり判定不能・s7_daily_select では除外対象）。
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
# 【2026-07-31改定】S7/7Aはこのゲートを撤廃した（下記s7_daily_select/
# s7a_daily_select参照。当時の検証は汚染モデル時代のもので、クリーンな
# 月次vintageモデルでの再検証ではaxis_sum<=1.5との組み合わせにより
# mark3ゲート無しの方がROI・的中率とも上回った）。
# S9/9A（s9_daily_select/s9a_daily_select）は9車立てでは軸選定の母集団が
# 異なりS7と同一の再検証を行っていないため、このゲートを引き続き使用する。
S7_MARK3_OVERLAP_MAX = 1


def s7_wt_mark3_overlap_n(
    axis1: int, axis2: int,
    wt_honmei: int | None, wt_taikou: int | None, wt_ana: int | None,
) -> int | None:
    """S7/S9の軸2車とWINTICKET公式印◎◯△（mark1/2/3・honmei/taikou/ana）との
    重なり数を返す（s7_wt_overlap_nの◎◯のみの判定に△を加えた拡張版）。

    wt_ana: prediction_mark==3（△）の frame_no。
    いずれか欠損時は None（判定不能・s7_daily_select/s9_daily_select では
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


def s7_gate_label(
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


def s7_daily_select(candidates: list[dict]) -> list[dict]:
    """S7の選出（2026-07-31改定: mark3ゲートを撤廃・axis_sum<=1.5に緩和）。

    candidates: 候補レースのリスト。各要素は最低限
      {"axis_sum": float, "wt_overlap_n": int | None, "entropy": float} を持つ dict。

    選出ロジック（全て閾値ゲート。件数による打ち切りは行わない）:
      - axis_sum > S7_AXIS_SUM_MAX（三連複が5倍未満に安くなりやすい極端な人気決着
        想定レース）は除外（2026-07-24導入。2026-07-31に1.3→1.5へ緩和）
      - entropy > S7_ENTROPY_MAX（フィールド全体の予測確率が拡散＝軸2車に集中して
        いない）は除外（2026-07-26導入。低いentropy＝軸2車に予測確率が集中し
        残り5車が拮抗、という状態が三連複高配当の的中と強く相関することを
        2024-2026の8四半期walk-forwardで確認。詳細はS7_ENTROPY_MAX定義部参照）。
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
    詳細は S7_AXIS_SUM_MAX / S7_MARK3_OVERLAP_MAX 定義部のコメント参照。
    ※この変更によりS7の母集団が広がったため、旧mark3ゲートに依存していた
    7Aの選出ロジック(s7a_daily_select)も2026-07-31に2ゲート化し、
    新S7との重複選出がないことを検算済み（重複0件）。

    日次件数の上限（S7_DAILY_CAP）は本関数では適用しない（朝夜どちらか一方の
    バッチだけでは日次合計が分からないため）。日次合計への適用は
    s7_evening_reselect() を参照。

    returns 採用された候補のリスト（axis_sum昇順・表示用の並び順のみ）。
    """
    pool = [
        c for c in candidates
        if c["axis_sum"] <= S7_AXIS_SUM_MAX
        and c.get("entropy", float("inf")) <= S7_ENTROPY_MAX
        and c.get("wt_overlap_n") in (0, 1)
    ]
    return sorted(pool, key=lambda c: c["axis_sum"])


def s7_evening_reselect(
    day_raw: list[dict], night_raw: list[dict], locked_keys: set[str] = frozenset(),
) -> list[dict]:
    """S7の朝夜統合選出（2026-07-26改定: entropyゲート通過後、日次合計を
    S7_DAILY_CAP件まで entropy昇順（＝最も自信がある順）でトリムする）。

    day_raw/night_raw: 朝/夜それぞれの生候補（選出前の全件、s7_select_axis+
      s7_wt_overlap_n+entropy計算を通した dict のリスト。各要素に "race_key" が必要）。
    locked_keys: 既に買い判定済み（picks_history に bet_amount>0 で記録済み）の
      race_key の集合。ゲート・トリムいずれでも除外しない（実購入は取り消せない
      ため）。ロック済み候補は s7_daily_select() のゲート判定より前に分離する
      （2026-07-26修正: ロック済みでもゲート内で先に弾かれれば結果的に未保護に
      なる抜け穴があったため、ゲート適用前に確定で救済する設計に変更）。

    S7_DAILY_CAP は honest全期間(832日)で実際にゲート通過が最大9件/日だった
    ことから、通常運用ではほぼ発火しない安全網として設計されている
    （exp_s4_daily_cap_by_entropy.py参照）。

    returns 採用された候補のリスト。
    """
    all_raw = day_raw + night_raw
    locked = [c for c in all_raw if c.get("race_key") in locked_keys]
    gated = s7_daily_select([c for c in all_raw if c.get("race_key") not in locked_keys])
    unlocked = sorted(gated, key=lambda c: c["entropy"])
    remaining_budget = max(0, S7_DAILY_CAP - len(locked))
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
# 軸選定(s7_select_axis)・フィールドentropy計算(s7_field_entropy)・
# WT◎◯重なり判定(s7_wt_overlap_n)・表示ランク(s7_gate_label)はいずれも
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
# axis_sum閾値（S7_AXIS_SUM_MAX相当）は9車では未較正のため導入していない
# （entropy単体で真のwalk-forward検証済み・複数の未較正閾値を積み増すことに
# よる過学習リスクを避けた。将来的な追加検証の余地あり）。
S9_NE = 9                   # 対象車数（9車ちょうど）
S9_STAKE = 100               # 円/点（ペーパー・7点=700円/レース）
S9_ENTROPY_MAX = 1.9938


def s9_daily_select(candidates: list[dict]) -> list[dict]:
    """S9の選出。S7と同じ閾値ゲート方式（axis_sum閾値は9車では未導入）。

    candidates: 候補レースのリスト。各要素は最低限
      {"wt_overlap_n": int | None, "entropy": float} を持つ dict。

    選出ロジック:
      - entropy > S9_ENTROPY_MAX は除外（詳細は上部コメント参照）。
        entropyキー欠損時は float("inf")扱い＝必ず除外（フェイルセーフ。
        S7での同種事故を踏まえた設計）
      - wt_overlap_n == 0（◎◯と全く重ならない）・1（片方だけ重なる）:
        上記ゲート通過分を全件採用（件数capなし。S9は元々低ボリュームのため
        S7のようなS9_DAILY_CAP安全網は現時点で不要と判断）
      - wt_overlap_n == 2（◎◯と完全一致）・None（WTマーク欠損）: 除外
      - wt_mark3_overlap_n（軸2車とWT公式印◎◯△=mark1/2/3との重なり数）が2は
        除外（2026-07-27導入・S7と共通のゲート。詳細はS7_MARK3_OVERLAP_MAX参照）

    returns 採用された候補のリスト（axis_sum昇順・表示用の並び順のみ）。
    """
    pool = [
        c for c in candidates
        if c.get("entropy", float("inf")) <= S9_ENTROPY_MAX
        and c.get("wt_overlap_n") in (0, 1)
        and c.get("wt_mark3_overlap_n", 2) <= S7_MARK3_OVERLAP_MAX
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
#   母集団 = s7_select_axis()/s9相当で軸選定成功 ∧ wt_overlap_n∈{0,1}（◎◯完全一致
#     と印欠損は既存同様に除外）。
#   7A: axis_sum<=S7_AXIS_SUM_MAX・entropy<=S7_ENTROPY_MAX
#       の2条件のうち、不合格がちょうど1個（0個=S7・2個とも不合格=対象外）。
#       【2026-07-31改定】旧来はmark3も含む3条件だったが、S7自体がmark3ゲートを
#       撤廃した（s7_daily_select参照）ため、7Aも2条件に揃えた（mark3を条件に
#       残すと「mark3のみ不合格」の候補が新S7にも旧7Aにも該当し重複選出になる
#       ため）。
#   9A: 9車はaxis_sum閾値が未導入・S9側のmark3ゲートは変更していないため、
#       entropy<=S9_ENTROPY_MAX・mark3<=S7_MARK3_OVERLAP_MAX の2条件のうち、
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

S7A_STAKE = 100  # 円/点（ペーパー・7車5点=500円/レース）
S9A_STAKE = 100  # 円/点（ペーパー・9車7点=700円/レース）


def s7a_daily_select(candidates: list[dict]) -> list[dict]:
    """7Aの選出: S7の2ゲート(axis_sum/entropy)のうちちょうど1つだけ不合格の候補。

    candidates: 各要素は最低限
      {"axis_sum": float, "entropy": float, "wt_overlap_n": int | None} を持つ dict。

    【2026-07-31改定】旧来はmark3も含む3ゲートだったが、S7自体がmark3ゲートを
    撤廃した（s7_daily_select参照）ため2ゲートに揃えた。新S7との重複選出が
    ないことをhonest全期間で検算済み（本セクション冒頭コメント参照）。

    - wt_overlap_n ∈ {0,1} 必須（◎◯完全一致=2・マーク欠損=None は対象外、S7と同様）
    - axis_sum<=S7_AXIS_SUM_MAX・entropy<=S7_ENTROPY_MAX の2条件のうち、
      不合格の個数がちょうど1個の候補のみ採用
      （0個=S7本体の対象・2個とも不合格は市場効率の壁でROI不採用、
        詳細は本セクション冒頭参照）

    returns 採用された候補のリスト（axis_sum昇順）。
    """
    pool = []
    for c in candidates:
        if c.get("wt_overlap_n") not in (0, 1):
            continue
        axis_ok = c["axis_sum"] <= S7_AXIS_SUM_MAX
        ent_ok = c.get("entropy", float("inf")) <= S7_ENTROPY_MAX
        n_fail = (not axis_ok) + (not ent_ok)
        if n_fail == 1:
            pool.append(c)
    return sorted(pool, key=lambda c: c["axis_sum"])


def s9a_daily_select(candidates: list[dict]) -> list[dict]:
    """9Aの選出: S9の2ゲート(entropy/mark3)のうちちょうど1つだけ不合格の候補。

    s7a_daily_select() の9車版。9車はaxis_sum閾値が未導入（S9同様）のため、
    entropy<=S9_ENTROPY_MAX・mark3<=S7_MARK3_OVERLAP_MAX の2条件のうち
    不合格がちょうど1個の候補のみ採用する。
    """
    pool = []
    for c in candidates:
        if c.get("wt_overlap_n") not in (0, 1):
            continue
        mark3 = c.get("wt_mark3_overlap_n")
        if mark3 is None:
            continue
        ent_ok = c.get("entropy", float("inf")) <= S9_ENTROPY_MAX
        mark3_ok = mark3 <= S7_MARK3_OVERLAP_MAX
        n_fail = (not ent_ok) + (not mark3_ok)
        if n_fail == 1:
            pool.append(c)
    return sorted(pool, key=lambda c: c["axis_sum"])


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
# 7SS（波乱軸選出・穴レース検知・2026-07-31導入）
#
# S7/S9の「本命ピックアップ」とは逆に、高配当の的中（見せ場・購入者への
# アピール）を狙う独立戦略。モデル予測に依存せず、wt_entriesの公表値
# （競走得点・1着率・3着内率・WT公式印・ライン構成）のみで判定する
# （2022年以降のデータで即座にhonest backtestが可能な設計）。
#
# 検証（TRAIN=2022-01-01〜2023-12-31 / TEST=2024-01-01〜2026-07-30・
# scripts/exp_upset50_*_2026_07_31.py 系列）:
#   ①「拮抗度（際立った実力上位馬の不在）」を表す13特徴の標準化合算スコアが
#     高いレースほど、勝ち三連複が50倍以上になる確率が高い（再現性あり・
#     lift 1.1〜1.2倍程度と控えめ）。
#   ②軸選定は「race_point(競走得点)単独top1」が「1着率+3着内率」複合より
#     一貫して優れる。
#   ③軸2はWT公式印(◯△✕=prediction_mark 2,3,4)のうち軸1と重ならない
#     3着内率最大の1頭が最も安定（TRAIN/TESTの目減りが最小）。
#   ④この組み合わせでもROIは70〜72%（TEST）止まりで控除率75%の壁・
#     ROI100%には届かないが、的中時の配当は最高354.2倍（TEST全体）・
#     最高250.5倍（穴指数上位20%）に達し、購入者へのアピュールとして
#     ユーザー判断により採用。ROI改善ではなく「見せ場」が目的の商品。
#
# 買い目 = 三連複 軸2車+残り5車流し（5点・S7と同じ100円/点）。
# ═══════════════════════════════════════════════════════════════════════════

SEVENSS_STAKE = 100  # 円/点（ペーパー・5点=500円/レース）

SEVENSS_FEATURES = (
    "rp_max", "rp_std", "rp_gap12",
    "fr_max", "fr_std", "fr_gap12",
    "tr_max", "tr_std", "tr_gap12",
    "n_lines", "max_line_size", "n_solo", "line_entropy",
)

# TRAIN(2022-01-01〜2023-12-31・7車立て・n=22,953)のみで確定した凍結パラメータ。
# TESTでの再計算・再学習は一切行わない（honest固定閾値）。
SEVENSS_MU = {
    "rp_max": 88.290582, "rp_std": 3.683336, "rp_gap12": 2.264474,
    "fr_max": 35.481724, "fr_std": 11.360250, "fr_gap12": 13.195604,
    "tr_max": 66.454028, "tr_std": 15.218289, "tr_gap12": 10.974465,
    "n_lines": 3.454015, "max_line_size": 2.885941, "n_solo": 1.105607,
    "line_entropy": 1.146982,
}
SEVENSS_SD = {
    "rp_max": 14.343346, "rp_std": 2.951000, "rp_gap12": 2.233535,
    "fr_max": 17.586289, "fr_std": 5.714892, "fr_gap12": 13.676320,
    "tr_max": 14.768138, "tr_std": 5.438257, "tr_gap12": 9.756276,
    "n_lines": 1.160679, "max_line_size": 0.749428, "n_solo": 1.871859,
    "line_entropy": 0.283591,
}
SEVENSS_SIGN = {
    "rp_max": 1.0, "rp_std": -1.0, "rp_gap12": -1.0,
    "fr_max": -1.0, "fr_std": -1.0, "fr_gap12": -1.0,
    "tr_max": -1.0, "tr_std": -1.0, "tr_gap12": -1.0,
    "n_lines": -1.0, "max_line_size": -1.0, "n_solo": -1.0,
    "line_entropy": 1.0,
}
# TRAIN上位20%点（この値以上を「穴指数」高＝波乱予兆レースとして採用）
SEVENSS_SCORE_THRESHOLD = 4.796886


def _sevenss_entropy(vals: list[float]) -> float:
    total = sum(vals)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in vals:
        s = max(v / total, 1e-9)
        ent -= s * math.log(s)
    return ent


def _sevenss_pop_std(vals: list[float]) -> float:
    """母集団標準偏差（ddof=0・exp_upset50系スクリプトのnp.std(default)と同一定義）。"""
    n = len(vals)
    if n == 0:
        return 0.0
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / n)


def sevenss_field_features(entries: list[dict]) -> dict[str, float] | None:
    """entries（各要素: race_point/first_rate/third_rate/line_groupを持つdict）
    から13特徴を計算する。値欠損時は None。
    """
    if any(e.get("race_point") is None or e.get("first_rate") is None
           or e.get("third_rate") is None for e in entries):
        return None
    rps = sorted((float(e["race_point"]) for e in entries), reverse=True)
    frs = sorted((float(e["first_rate"]) for e in entries), reverse=True)
    trs = sorted((float(e["third_rate"]) for e in entries), reverse=True)
    if len(rps) < 2 or len(frs) < 2 or len(trs) < 2:
        return None

    line_sizes: dict[int, int] = {}
    for e in entries:
        lg = e.get("line_group")
        if lg is not None:
            line_sizes[lg] = line_sizes.get(lg, 0) + 1
    n_lines = float(entries[0].get("n_lines") or len(line_sizes) or 0)
    max_line_size = float(max(line_sizes.values())) if line_sizes else 0.0
    n_solo = float(sum(1 for v in line_sizes.values() if v == 1))
    line_entropy = _sevenss_entropy(list(line_sizes.values())) if line_sizes else 0.0

    return {
        "rp_max": rps[0], "rp_std": _sevenss_pop_std(rps), "rp_gap12": rps[0] - rps[1],
        "fr_max": frs[0], "fr_std": _sevenss_pop_std(frs), "fr_gap12": frs[0] - frs[1],
        "tr_max": trs[0], "tr_std": _sevenss_pop_std(trs), "tr_gap12": trs[0] - trs[1],
        "n_lines": n_lines, "max_line_size": max_line_size, "n_solo": n_solo,
        "line_entropy": line_entropy,
    }


def sevenss_score(feat: dict[str, float]) -> float:
    """穴指数（標準化合算スコア）。高いほど波乱（三連複50倍以上）寄り。"""
    return sum(
        SEVENSS_SIGN[f] * (feat[f] - SEVENSS_MU[f]) / SEVENSS_SD[f]
        for f in SEVENSS_FEATURES
    )


def sevenss_select_axis(entries: list[dict]) -> tuple[int, int] | None:
    """7SSの軸2車を選定する。

    軸1 = race_point(競走得点)単独top1。
    軸2 = WT公式印(prediction_mark in (2,3,4)=◯△✕)のうち軸1以外で
          third_rate(3着内率)最大の1頭。
    候補なし（◯△✕が軸1以外に存在しない・マーク欠損）は None（選定不能）。

    entries の各要素は frame_no/race_point/third_rate/prediction_mark を持つdict。
    """
    by_frame = {int(e["frame_no"]): e for e in entries}
    if any(by_frame[f].get("race_point") is None for f in by_frame):
        return None
    axis1 = max(by_frame, key=lambda f: float(by_frame[f]["race_point"]))

    mark_frames = [
        f for f in by_frame
        if by_frame[f].get("prediction_mark") in (2, 3, 4) and f != axis1
        and by_frame[f].get("third_rate") is not None
    ]
    if not mark_frames:
        return None
    axis2 = max(mark_frames, key=lambda f: float(by_frame[f]["third_rate"]))
    return axis1, axis2
