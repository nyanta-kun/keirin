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
# ═══════════════════════════════════════════════════════════════════════════

S1W_NE = 7                  # 対象車数（7車ちょうど）
S1W_TOP3_GAP_MIN = 0.15     # 相手2車(p1,p2)の3着内モデル確率差 下限（凍結値・2026-07-19）
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


def s1w_gate(top3_gap: float) -> bool:
    """S1(新設計)のゲート判定（相手2車の3着内モデル確信度）。"""
    return top3_gap >= S1W_TOP3_GAP_MIN


# ═══════════════════════════════════════════════════════════════════════════
# U（波乱ライン連れ込み）戦略 — 2026-07-16 ペーパートレード検証中
#
# 波乱見込みレース（指数エントロピー高 ∧ 盤面min三連複オッズ高）で、
# 市場4-7位∧モデル3位内∧ライン先頭/番手の「穴」と、同ラインの脚質「逃」の相方を
# 2車軸にした三連複流し（オッズ15倍以上の目のみ）。
# 検証: exp_dark_pair_features_wt.py ほか。テスト110.9% / VAL 118.9%（プール約117%）。
# 多重比較上振れの懸念があるためライブは記録のみ（ペーパー）で8月末に採否判定。
# 閾値は 2026-01〜06 の本番モデル(lgbm_wt)分布のQ3で凍結（都度分位は使わない）。
# ═══════════════════════════════════════════════════════════════════════════

U_ENTROPY_MIN = 1.84       # 指数エントロピー下限（7車クリーン分布のQ3・凍結値）
U_MTO_MIN = 4.3            # 盤面min三連複オッズ下限（同Q3・凍結値）
U_LEG_MIN_ODDS = 15.0      # 買い目の三連複オッズ下限（15倍未満はカット）
U_STAKE = 100              # 円/点（ペーパー）


def u_entropy(pred_probs: list[float]) -> float:
    """レースの指数エントロピー（占有率ベースの混戦度）。"""
    import math
    total = sum(pred_probs)
    if total <= 0:
        return 0.0
    ent = 0.0
    for p in pred_probs:
        s = max(p / total, 1e-9)
        ent -= s * math.log(s)
    return ent


# ═══════════════════════════════════════════════════════════════════════════
# M=S3（◎不一致×システム◎×軸信頼ゲート）戦略 — 2026-07-19 3way OR拡張（ペーパー検証中）
#
# WT◎（prediction_mark==1）とシステム◎（モデル指数1位）が不一致のレースのうち、
# 以下いずれかの軸信頼シグナルを満たすレースで、システム◎と同ライン脚質「逃」の
# 相方を2車軸にした三連複流し（オッズ >= U_LEG_MIN_ODDS の目のみ）。市場順位条件なし。
#   (a) gap12（3着内モデル予測確率 1位−2位・rawスケール）>= M_GAP12_MIN
#   (b) システム◎の1着モデル（lgbm_wt_win）内でのレース内順位 >= M_WIN_RANK_MIN
#       （＝1着モデル視点では「勝ちきれない」と評価されている軸ほど、実際の
#       3着内的中率は下がるが市場オッズがそれ以上に甘くなりROIが上がる逆説的シグナル）
#   (c) ratio（システム◎の p_win / p_top3・乗法比）<= M_RATIO_MAX
#       （＝1着モデル確率が3着内モデル確率に対して相対的に低いほど「勝ちきれない」
#       度合いが強い、という(b)の連続量版。win_rank(離散順位)とは別のシグナル）
#
# 2026-07-17: 旧定義の波乱ゲート（entropy≥U_ENTROPY_MIN ∧ mto≥U_MTO_MIN）を廃止し
# 軸信頼ゲート gap12≥0.10 へ転換（不一致システム◎の3着内率 68.3%→73.1%）。
# 2026-07-19: Phase B（1着モデル導入）で win_rank ゲートを発見・OR統合。
# (a)(b) はほぼ独立（重複率5%・相関-0.265）で、統合すると母数が約2倍
# （0.6R/日→1.2R/日相当）に増えつつROIも維持（exp_win_axis_sweep_wt.py・
# 正規プロトコル: 検証158.2%(531R)→テスト149.5%(152R・目≥15限定)）。
# 2026-07-19: 複合シグナル追加探索（win_rankの連続量版）で、加法差
# diff=p_top3-p_winは無判別力（母数ほぼ不変）だったが、乗法比ratioは有効と判明。
# (a)(b)への第3項OR追加として評価（閾値スイープでなく事前仮説としての評価）:
# 母数が更に+22〜26%増えつつROI維持〜微増（exp_composite_prob_diff_wt.py・
# 正規プロトコル: 検証158.2%(531R)→158.6%(671R)、テスト149.5%(152R)→154.3%(186R)）。
# 買い目オッズ閾値は U と同一値（U_LEG_MIN_ODDS）を再利用する。
# 同一レースで U（buy）と同一ペア集合になった場合は U 優先で M は記録しない。
# ═══════════════════════════════════════════════════════════════════════════

M_GAP12_MIN = 0.10         # gap12 下限（rawスケール・軸信頼ゲート・凍結値）
M_WIN_RANK_MIN = 3         # システム◎の1着モデル内レース順位 下限（凍結値・2026-07-19）
M_RATIO_MAX = 0.30         # システム◎の p_win/p_top3 比 上限（凍結値・2026-07-19）
M_STAKE = 100              # 円/点（ペーパー）


def m_axis_gate(
    gap12: float, win_rank: int | None, ratio: float | None = None,
) -> tuple[bool, str | None]:
    """S3(M) の軸信頼ゲート判定（gap12 OR win_rank OR ratio・2026-07-19 3way OR拡張）。

    gap12:    システム◎(モデル1位)の gap12（1位-2位・rawスケール）。
    win_rank: システム◎の1着モデル（lgbm_wt_win）内レース順位（1-indexed）。
              1着モデル未ロード時は None（gap12単独ゲートにフォールバック）。
    ratio:    システム◎の p_win/p_top3 比（1着モデル確率 ÷ 3着内モデル確率）。
              いずれか未算出時は None。

    returns (passed, gate_label)
      passed:     いずれかのゲートを満たせば True
      gate_label: "gap12" / "win_rank" / "ratio" / None（すべて不成立）。
                  複数成立時は gap12 > win_rank > ratio の順で優先表示する。
    """
    if gap12 >= M_GAP12_MIN:
        return True, "gap12"
    if win_rank is not None and win_rank >= M_WIN_RANK_MIN:
        return True, "win_rank"
    if ratio is not None and ratio <= M_RATIO_MAX:
        return True, "ratio"
    return False, None


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
