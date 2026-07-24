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
    axis_player_class: str | None = None,
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
    """
    if top3_gap < S1W_TOP3_GAP_MIN:
        return False
    if axis_win_prob is not None and axis_win_prob > S1W_AXIS_WIN_PROB_MAX:
        return False
    if axis_player_class is not None and axis_player_class in S1W_DENY_AXIS_CLASS:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# S4（単勝×複勝指数トップ3重なり軸×波乱度選出・三連複2軸総流し）— 2026-07-21 導入
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
# 選出 = 当日の該当レースをaxis_sum昇順に並べ、上位 S4_DAILY_TOP_N 件を採用
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
# S4_DAILY_TOP_N件・重なり2は完全除外という選出方式へ変更（1日の採用本数は
# 重なり0の発生数に応じて可変・honest全期間で平均10.77R/日）。
# honest全期間再構築（この方式）: 9,927R（922日・10.77R/日）・的中36.3%・
# **ROI131.3%**（旧方式の128.1%から改善）。内訳: 重なり0(943R)的中39.4%/ROI232.8%・
# 重なり1(8984R)的中36.0%/ROI120.6%。
# ═══════════════════════════════════════════════════════════════════════════

S4_NE = 7                  # 対象車数（7車ちょうど）
S4_DAILY_TOP_N = 10        # 重なり1（片方一致）候補の1日あたり最終固定採用件数（axis_sum昇順）
                           # 2026-07-21: 「N件」の意味が変更された（旧: 全候補中の上位N件 →
                           # 新: 重なり0は別枠で全件採用・本値は重なり1のみに適用する固定枠）
S4_HALF_CAP = 6            # 朝/夜それぞれの生候補プールからの一次選出上限（重なり1のみ・2026-07-22新設）
S4_STAKE = 100             # 円/点（ペーパー・5点=500円/レース）

# 三連複が安くなりやすい（極端な人気決着になりやすい）レースの除外上限
# （2026-07-24・ユーザー要望「三連複5倍未満は購入対象から除外したい」への対応）。
# 買い目は5点流し（1点100円=500円）のため、三連複配当が500円(5倍)を下回ると
# 的中しても賭け金を割る。honest全期間検証（2024-01-01〜2026-07-23・935日・
# quarterly walk-forwardモデルのpred_top3_pctのみ使用＝発走前確定情報のみ・
# train=〜2026-04-30でしきい値検討→test=2026-05-01〜で確認）の結果:
#   axis_sum とレース着地時の三連複配当<500円の相関 AUC 0.64(train)/0.67(test)。
#   他の発走前特徴量（field合計/軸単勝率等）を組み合わせてもAUC改善なし
#   （axis_sumと相関0.83で情報量が乏しい・公開情報の壁）。
# axis_sum<=1.3 全期間シミュレーション: 全体 10.75件/日→7.83件/日(-27%)・
#   的中36.3%→34.4%・ROI 131.3%→147.1%（SS+ 363→444%・SS 150→185%・S 121→132%）。
# ユーザー判断で 1.3 を採用（1.2はROI182%まで伸びるが件数-59%と減りすぎ、
# 1.4は件数維持だがROI改善が+5pt程度に留まる）。次点繰り上げなし（S4_HALF_CAP/
# S4_DAILY_TOP_N の cap 内で足切りするだけ＝S1のS1/A1級班denyフィルタと同じ設計。
# 重なり0(SS/SS+)はcap無しのため単純カット、重なり1(S)はaxis_sum昇順選出後の
# 末尾が削れるだけで繰り上がり由来のROI悪化は発生しない）。
S4_AXIS_SUM_MAX = 1.3


def s4_select_axis(
    win_probs: dict[int, float], top3_probs: dict[int, float],
) -> tuple[int, int, float] | None:
    """S4の軸2車とaxis_sum（波乱度指数の元）を選定する。

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


def s4_wt_overlap_n(
    axis1: int, axis2: int, wt_honmei: int | None, wt_taikou: int | None,
) -> int | None:
    """S4の軸2車とWINTICKET公式予想の◎◯（honmei/taikou）との重なり数を返す。

    wt_honmei: prediction_mark==1（◎）の frame_no。
    wt_taikou: prediction_mark==2（◯）の frame_no。
    いずれか欠損時は None（重なり判定不能・s4_daily_select では除外対象）。
    """
    if wt_honmei is None or wt_taikou is None:
        return None
    return len({axis1, axis2} & {wt_honmei, wt_taikou})


# S4のSS(重なり0)のうち、軸2車のいずれかが各グレード最上位クラス（S1/A1）だと
# 配当が下がりやすい傾向を確認（2026-07-23・honest全期間検証）。SSは無制限採用
# （日次cap無し）のため、S1と異なり「除外→繰り上がり」の副作用がなく単純に
# 効く: train+val ROI222.3%→351.6%・全期間237.1%→362.2%（的中率は不変〜微増）。
# 一方Sは日次axis_sum上位10件のcap付き選出のため、除外すると繰り上がり候補で
# ROIが悪化する（train+val 116.3%→111.5%・test 132.6%→119.2%）ことを確認済み。
# → SS内の格上非該当サブセットを新表示ランク"SS+"として観察する（実際の
# 購入対象・買い目は変更しない。あくまで表示分岐）。
S4_TOP_CLASS = {"S1", "A1"}


def s4_gate_label(
    wt_overlap_n: int | None,
    axis1_class: str | None = None, axis2_class: str | None = None,
) -> str | None:
    """S4の表示ランク(gate_label)を返す。

    - wt_overlap_n == 0: 軸2車の級班情報が両方揃っており、いずれもS4_TOP_CLASS
      でなければ "SS+"（観察用サブランク）、そうでなければ "SS"。
      級班情報が欠損している場合は従来通り "SS"（後方互換）。
    - wt_overlap_n == 1: "S"
    - それ以外（重なり2・None）: None（除外対象）
    """
    if wt_overlap_n == 0:
        if axis1_class is not None and axis2_class is not None:
            has_top = axis1_class in S4_TOP_CLASS or axis2_class in S4_TOP_CLASS
            return "SS" if has_top else "SS+"
        return "SS"
    if wt_overlap_n == 1:
        return "S"
    return None


def s4_daily_select(candidates: list[dict], cap: int = S4_HALF_CAP) -> list[dict]:
    """S4の一次選出（朝または夜、片方のバッチ内での選出・2026-07-22改定）。

    candidates: 同一バッチ（朝races または 夜races）の候補レースのリスト。
      各要素は最低限 {"axis_sum": float, "wt_overlap_n": int | None} を持つ dict。

    選出ロジック:
      - wt_overlap_n == 0（◎◯と全く重ならない）: 該当があれば無条件で全件採用
        （的中率は変わらずROIを押し上げる区分のため最優先・本数上限なし）
      - wt_overlap_n == 1（片方だけ重なる）: axis_sum昇順で上位 cap 件を採用
      - wt_overlap_n == 2（◎◯と完全一致）・None（WTマーク欠損）: 除外
        （完全一致は honest全期間検証でROI75.7%の赤字区分と判明したため）
      - axis_sum > S4_AXIS_SUM_MAX（三連複が5倍未満に安くなりやすい極端な人気決着
        想定レース）は上記いずれの区分でも除外（2026-07-24導入。次点繰り上げなし）

    2026-07-21〜07-22の変遷: 当初は日次上限をそのままバッチ単位に適用していたが、
    朝夕2回が独立にTOP_N件ずつ選ぶと1日で最大20件になるバグを発見（07-21）。
    「朝が先着で枠を使い切り、夜の優良候補を取りこぼす」というユーザー指摘を受け、
    朝夕それぞれの一次選出をS4_HALF_CAP(=6)件に縮小し、夕方バッチで
    s4_evening_reselect() により朝夜合算のaxis_sumランキングへ組み直す方式へ
    07-22に再設計した（honest全期間バックテストでROI120.8%・理論上限120.6%と
    ほぼ同等・選出一致率89.5%を確認）。

    cap: 重なり1の一次選出上限。朝夕バッチでは既定のS4_HALF_CAP(6)を使う。

    returns 採用された候補のリスト（重なり0が前・重なり1がaxis_sum昇順で続く）。
    """
    pool = [c for c in candidates if c["axis_sum"] <= S4_AXIS_SUM_MAX]
    tier0 = [c for c in pool if c.get("wt_overlap_n") == 0]
    tier1 = sorted(
        (c for c in pool if c.get("wt_overlap_n") == 1),
        key=lambda c: c["axis_sum"])
    return tier0 + tier1[:cap]


def s4_evening_reselect(
    day_raw: list[dict], night_raw: list[dict], locked_keys: set[str],
) -> list[dict]:
    """S4の夕方最終選出（朝夜統合→ロック考慮で日次S4_DAILY_TOP_N件へトリム・2026-07-22新設）。

    day_raw/night_raw: 朝/夜それぞれの生候補（選出前の全件、s4_select_axis+
      s4_wt_overlap_n を通した dict のリスト。各要素に "race_key" キーが必要）。
    locked_keys: 既に買い判定済み（picks_history に bet_amount>0 で記録済み）の
      race_key の集合。この夕方の組み直しでは変更しない（実購入は取り消せないため）。

    手順:
      1. 朝夜それぞれの生候補（重なり1のみ）から s4_daily_select() でS4_HALF_CAP件ずつ
         一次選出し、最大12件の統合プールを作る（重なり0は別枠で無条件採用のまま）。
      2. 統合プールのうちロック済み（既に買い判定済み）のものは無条件で残す。
      3. 残り（未判定）はaxis_sum昇順で、日次合計が S4_DAILY_TOP_N 件に収まる範囲だけ
         採用し、それ以外は候補から外す（次点繰り上げなし＝質で足切り）。

    returns 最終採用候補のリスト（重なり0全件 + 重なり1の最終選出）。
    """
    day_sel = s4_daily_select(day_raw, cap=S4_HALF_CAP)
    night_sel = s4_daily_select(night_raw, cap=S4_HALF_CAP)

    tier0 = [c for c in day_sel + night_sel if c.get("wt_overlap_n") == 0]
    tier1_union = [c for c in day_sel + night_sel if c.get("wt_overlap_n") == 1]

    locked = [c for c in tier1_union if c.get("race_key") in locked_keys]
    unlocked = sorted(
        (c for c in tier1_union if c.get("race_key") not in locked_keys),
        key=lambda c: c["axis_sum"])
    remaining_budget = max(0, S4_DAILY_TOP_N - len(locked))
    tier1_final = locked + unlocked[:remaining_budget]

    return tier0 + tier1_final


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
