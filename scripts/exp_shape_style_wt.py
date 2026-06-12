"""≤6車: 指数分布の「形」× 脚質 × ライン の構造検証（ユーザー仮説 2026-06-10・拡張版）。

仮説空間の洗い出し（CONTINUATION/会話 2026-06-10）から事前登録した第1陣7仮説＋規律チェック＋第2陣探索:
  H1: 1車抜け×脚質（抜け逃げは単騎逃げ/逃げ被りで信頼度が変わる）
  H2: 2車抜け×ライン（同ライン完結=理想型 vs 別ライン逃逃=共倒れ）
  H3: 拮抗×逃げ人数（消耗戦=波乱の質・逃げ1人で指数下位の単騎逃げ=穴）
  H4: 抜け追込×孤立（ついて行く先がない=脆い）
  H5: 抜け追込×前(ライン先頭)の質（前が弱いと位置を失う）
  H6: 3車抜け×ライン構成（1ライン完結/2+1/3別線=trio適性）
  H7: 1車抜け×残り構造（残り拮抗=1着固定で配当 / 残り階段=順当低配当）
  PG: 直交性(top3_sum)・ランク内部判別・較正残差・ガールズ除外再現
  第2陣: 2車抜け脚質ペア・拮抗×追被り・下位断絶（TRAINで方向→TESTで確認）

形の分類（pred_prob降順 p1..pn, G=[p1-p2,p2-p3,p3-p4], spread=ΣG）:
  拮抗: spread<TRAIN q30 / 階段: max(G)/spread<0.45 / 1・2・3車抜け: argmax(G)
判定規律: 較正残差(実績-モデル予測)が系統的→モデル未知=特徴化候補。
        残差≈0でROIだけ割れる→戦略レイヤー（買い目出し分け）の材料。
model=lgbm_wt_eval(OOS)・TRAIN 2023-07〜2026-02 / TEST 2026-03〜06-08・払戻=最終オッズ上限値。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt
from roi_robustness_wt import roi_summary

GIRLS = {"cls4", "L級"}


def collect(f, t):
    model = load_model("lgbm_wt_eval")
    df = build_features_wt(load_raw_data_wt(min_date=f, max_date=t))
    df = _apply_pred_prob_wt(model, df)
    sizes = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sizes[sizes <= 6].index)]
    pm = _load_payouts_wt(df["race_key"].unique().tolist())
    rows = []
    for rk, g in df.groupby("race_key"):
        g = g.sort_values("pred_prob", ascending=False)
        n = len(g)
        if n < 4:
            continue
        p = g["pred_prob"].tolist()
        fr = g["frame_no"].astype(int).tolist()
        styles = [str(s) if isinstance(s, str) and s else "?" for s in g["style"].tolist()]
        lg = g["line_group"].fillna(0).astype(int).tolist()
        lp = g["line_pos"].fillna(1).astype(int).tolist() if "line_pos" in g.columns else [1] * n
        cls = [str(c) for c in g["player_class"].tolist()] if "player_class" in g.columns else []
        girls = bool(cls) and all(c in GIRLS for c in cls)
        gaps = [p[0] - p[1], p[1] - p[2], p[2] - p[3]]
        spread = sum(gaps)
        sec_spread = p[1] - p[3]                     # H7: 2〜4位の広がり（残り構造）
        bottom_gap = (p[3] - p[4]) if n >= 5 else None   # P6-1: 4位と5位の断絶
        gap12 = gaps[0]
        ratio = p[0] / (3.0 / n)
        if gap12 < 0.06:
            tier = None
        elif gap12 >= 0.15 and ratio < 1.3:
            tier = "SS"
        elif gap12 >= 0.15 and ratio < 1.6:
            tier = "S"
        elif gap12 >= 0.15:
            tier = None
        else:
            tier = "A"
        # p1のライン文脈
        n_nige = sum(1 for s in styles if s == "逃")
        other_nige = n_nige - (1 if styles[0] == "逃" else 0)
        p1_iso = (lg[0] == 0)
        leader_prob = None          # p1が追/両のとき、同ライン先頭(line_pos最小)のpred_prob
        if not p1_iso and styles[0] != "逃":
            mates = [(lp[i], p[i]) for i in range(n) if lg[i] == lg[0] and i != 0]
            ahead = [pp for pos, pp in mates if pos < lp[0]]
            if ahead:
                leader_prob = max(ahead)
        # モデル上位3のライン構成（0/単騎は各々独立扱い）
        t3lg = []
        for i in range(3):
            t3lg.append(f"iso{i}" if lg[i] == 0 else f"L{lg[i]}")
        n_lines_top3 = len(set(t3lg))
        # 結果
        fin = g[g["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        order = tuple(int(x) for x in fin.sort_values("finish_order")["frame_no"])
        top3 = frozenset(order)
        rp = pm.get(rk, {})
        pay, bet = 0, 0
        if tier:
            p1f, p2f = fr[0], fr[1]
            thirds = fr[2:5]
            bet = len(thirds) * 100
            for x in thirds:
                if tier == "SS":
                    if order == (p1f, p2f, x):
                        pay = rp.get(("trifecta", (p1f, p2f, x)), 0); break
                else:
                    if frozenset((p1f, p2f, x)) == top3:
                        pay = rp.get(("trio", frozenset((p1f, p2f, x))), 0); break
        rows.append({
            "spread": spread, "gaps": gaps, "sec_spread": sec_spread, "bottom_gap": bottom_gap,
            "gap12": gap12, "tier": tier, "girls": girls,
            "p1": p[0], "top3_sum": p[0] + p[1] + p[2],
            "s1": styles[0], "s2": styles[1], "s12": "".join(sorted(styles[:2])),
            "top2_same_line": (lg[0] == lg[1] != 0),
            "top2_nige_clash": (not (lg[0] == lg[1] != 0)) and styles[0] == styles[1] == "逃",
            "top3_all_oikomi": all(s == "追" for s in styles[:3]),
            "p1_isolated": p1_iso, "other_nige": other_nige, "n_nige": n_nige,
            "leader_prob": leader_prob,
            "n_lines_top3": n_lines_top3,
            # 拮抗×逃げ1人: その逃げ車の指数順位と実績（P4-3 穴仮説）
            "lone_nige_rank": (styles.index("逃") + 1) if n_nige == 1 else None,
            "lone_nige_top3": (fr[styles.index("逃")] in top3) if n_nige == 1 else None,
            "lone_nige_prob": p[styles.index("逃")] if n_nige == 1 else None,
            "p1_top3": fr[0] in top3, "p1_win": order[0] == fr[0],
            "top2_both": frozenset(fr[:2]).issubset(top3),
            "model_top3_exact": frozenset(fr[:3]) == top3,
            "pay": pay, "bet": bet,
        })
    return rows


def classify(rows, q30, dom=0.45):
    for r in rows:
        G = r["gaps"]
        if r["spread"] < q30:
            r["shape"] = "拮抗"
        elif max(G) / max(r["spread"], 1e-9) < dom:
            r["shape"] = "階段"
        else:
            r["shape"] = ["1車抜け", "2車抜け", "3車抜け"][int(np.argmax(G))]
    return rows


def seg_roi(rows, cond):
    sub = [r for r in rows if r["tier"] and cond(r)]
    return roi_summary([r["pay"] for r in sub], [r["bet"] for r in sub]), len(sub)


def rate(rows, cond, key):
    sub = [r for r in rows if cond(r) and r[key] is not None]
    if not sub:
        return 0.0, 0, 0.0
    return (sum(r[key] for r in sub) / len(sub), len(sub),
            sum(r["p1"] for r in sub) / len(sub))


def calib_line(label, rows, cond, key="p1_top3", prob_key="p1"):
    sub = [r for r in rows if cond(r)]
    if not sub:
        return f"  {label:<26}    0R"
    act = sum(r[key] for r in sub) / len(sub)
    pred = sum(r[prob_key] for r in sub) / len(sub)
    return f"  {label:<26}{len(sub):>5}R  実績{act:>7.1%} 予測{pred:>7.1%} 残差{act-pred:>+7.1%}"


def roi_line(label, rows, cond):
    s, n = seg_roi(rows, cond)
    return (f"  {label:<26}{n:>5}R 的中{s['hit_rate']:>6.1%} ROI{s['roi']:>6.0%}"
            f" [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}] 最大除{s['roi_ex_max']:>5.0%} 中央{s['median_hit']:>6,.0f}円")


tr = collect("2023-07-01", "2026-02-28")
te = collect("2026-03-01", "2026-06-08")
q30 = float(np.quantile([r["spread"] for r in tr], 0.30))
sec_q30 = float(np.quantile([r["sec_spread"] for r in tr], 0.30))
bg = [r["bottom_gap"] for r in tr if r["bottom_gap"] is not None]
bg_q70 = float(np.quantile(bg, 0.70)) if bg else 0.1
tr = classify(tr, q30)
te = classify(te, q30)
print(f"\n{'='*96}")
print(f" ≤6車 分布形状×脚質×ライン 拡張検証  TRAIN {len(tr)}R / TEST {len(te)}R")
print(f" cuts: spread_q30={q30:.3f} sec_q30={sec_q30:.3f} bottom_gap_q70={bg_q70:.3f}（最終オッズ上限値）")
print(f"{'='*96}")

# ---- Part0/PG-1: 形の分布と既知レバーの重なり ----
print(f"\n【Part0/PG-1】形の出現率・top3_sum重なり・基準率（TRAIN）")
for sh in ["1車抜け", "2車抜け", "3車抜け", "階段", "拮抗"]:
    sub = [r for r in tr if r["shape"] == sh]
    if not sub:
        continue
    print(f"  {sh:<8} {len(sub):>5}R ({len(sub)/len(tr)*100:4.1f}%)  top3_sum={np.mean([r['top3_sum'] for r in sub]):.3f}"
          f"  p1_top3={np.mean([r['p1_top3'] for r in sub]):.1%}  girls率={np.mean([r['girls'] for r in sub]):.1%}")

# ---- H1: 1車抜け×脚質×逃げ被り ----
print(f"\n【H1】1車抜け: 抜けた選手の脚質×逃げ被り → 較正残差（残差≠0=モデル未知）")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    print(calib_line("抜け逃げ×単騎逃げ(他0)", rows, lambda r: r["shape"] == "1車抜け" and r["s1"] == "逃" and r["other_nige"] == 0))
    print(calib_line("抜け逃げ×逃げ被り(他1+)", rows, lambda r: r["shape"] == "1車抜け" and r["s1"] == "逃" and r["other_nige"] >= 1))
    print(calib_line("抜け両", rows, lambda r: r["shape"] == "1車抜け" and r["s1"] == "両"))
    print(calib_line("抜け追(全体)", rows, lambda r: r["shape"] == "1車抜け" and r["s1"] == "追"))

# ---- H4/H5: 抜け追込のライン文脈 ----
print(f"\n【H4/H5】1車抜けの追込: 孤立 / 前(ライン先頭)の質 → 較正残差")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    print(calib_line("追×孤立(ライン無)", rows, lambda r: r["shape"] == "1車抜け" and r["s1"] == "追" and r["p1_isolated"]))
    print(calib_line("追×前が強(prob≥.5)", rows, lambda r: r["shape"] == "1車抜け" and r["s1"] == "追" and (r["leader_prob"] or 0) >= 0.5))
    print(calib_line("追×前が弱(prob<.5)", rows, lambda r: r["shape"] == "1車抜け" and r["s1"] == "追" and r["leader_prob"] is not None and r["leader_prob"] < 0.5))

# ---- H7: 1車抜け×残り構造 ----
print(f"\n【H7】1車抜け: 残り(2-4位)が拮抗 vs 階段 → 1着率と本番ROI・配当")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    for lab, cond in [("残り拮抗(sec<q30)", lambda r: r["shape"] == "1車抜け" and r["sec_spread"] < sec_q30),
                      ("残り階段(sec≥q30)", lambda r: r["shape"] == "1車抜け" and r["sec_spread"] >= sec_q30)]:
        sub = [r for r in rows if cond(r)]
        if sub:
            w = np.mean([r["p1_win"] for r in sub])
            print(f"  {lab:<26}{len(sub):>5}R p1_1着{w:>6.1%}")
        print(roi_line("  └ 本番ROI", rows, cond))

# ---- H2: 2車抜け×ライン関係 ----
print(f"\n【H2】2車抜け: ライン関係 → top2両方残存率と本番ROI")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    for lab, cond in [("同ライン(完結)", lambda r: r["shape"] == "2車抜け" and r["top2_same_line"]),
                      ("別ライン", lambda r: r["shape"] == "2車抜け" and not r["top2_same_line"]),
                      ("別ライン逃逃(共倒れ?)", lambda r: r["shape"] == "2車抜け" and r["top2_nige_clash"])]:
        both, n, _ = rate(rows, cond, "top2_both")
        if n:
            print(f"  {lab:<26}{n:>5}R top2両方残存{both:>6.1%}")
        print(roi_line("  └ 本番ROI", rows, cond))

# ---- H6: 3車抜け×ライン構成 ----
print(f"\n【H6】3車抜け: 上位3車のライン構成 → trio適性(model top3一致率)と本番ROI")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    for lab, k in [("1ライン完結", 1), ("2+1", 2), ("3別線(3すくみ)", 3)]:
        cond = lambda r, kk=k: r["shape"] == "3車抜け" and r["n_lines_top3"] == kk
        ex, n, _ = rate(rows, cond, "model_top3_exact")
        if n:
            print(f"  {lab:<26}{n:>5}R model_top3一致{ex:>6.1%}")
        print(roi_line("  └ 本番ROI", rows, cond))

# ---- H3: 拮抗×逃げ人数（P4-1/2/3）----
print(f"\n【H3】拮抗: 逃げ人数 → 本番ROI（波乱の質）＋ P4-3 指数下位の単騎逃げ")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    for k, lab in [(0, "拮抗×逃げ0人"), (1, "拮抗×逃げ1人"), (2, "拮抗×逃げ2人+")]:
        cond = (lambda r, kk=k: r["shape"] == "拮抗" and (r["n_nige"] >= 2 if kk == 2 else r["n_nige"] == kk))
        print(roi_line(lab, rows, cond))
    # P4-3: 拮抗×逃げ1人でその逃げが指数下位(rank4+)→単騎逃げの穴
    sub = [r for r in rows if r["shape"] == "拮抗" and r["n_nige"] == 1 and (r["lone_nige_rank"] or 0) >= 4]
    if sub:
        act = np.mean([r["lone_nige_top3"] for r in sub])
        pred = np.mean([r["lone_nige_prob"] for r in sub])
        print(f"  P4-3 指数下位(4位-)の単騎逃げ {len(sub):>4}R  その車top3実績{act:>6.1%} 予測{pred:>6.1%} 残差{act-pred:>+6.1%}")

# ---- PG-2: ランク内部での形の判別力 ----
print(f"\n【PG-2】現行ランク内部での形別ROI（実装直結・TESTのみ表示/参考TRAIN ROIを括弧）")
for tier in ["A", "S", "SS"]:
    print(f"  ── {tier}ランク内 ──")
    for sh in ["1車抜け", "2車抜け", "3車抜け", "階段", "拮抗"]:
        s_te, n_te = seg_roi(te, lambda r, s=sh, t=tier: r["shape"] == s and r["tier"] == t)
        s_tr, n_tr = seg_roi(tr, lambda r, s=sh, t=tier: r["shape"] == s and r["tier"] == t)
        if n_te or n_tr:
            print(f"  {sh:<10} TE {n_te:>4}R 的中{s_te['hit_rate']:>6.1%} ROI{s_te['roi']:>6.0%}"
                  f" [{s_te['ci_lo']:>4.0%},{s_te['ci_hi']:>5.0%}]  (TR {n_tr}R {s_tr['roi']:.0%})")

# ---- PG-1b: top3_sumと直交させた形の追加判別力 ----
print(f"\n【PG-1b】top3_sum Q1_loose(下位25%)内/外での形別ROI（TEST・形は既知レバーを超えるか）")
t3cut = float(np.quantile([r["top3_sum"] for r in tr], 0.25))
for zone, zcond in [("loose内", lambda r: r["top3_sum"] < t3cut), ("loose外", lambda r: r["top3_sum"] >= t3cut)]:
    print(f"  ── {zone} ──")
    for sh in ["1車抜け", "2車抜け", "3車抜け", "階段", "拮抗"]:
        s, n = seg_roi(te, lambda r, s_=sh, z=zcond: r["shape"] == s_ and z(r))
        if n:
            print(f"  {sh:<10}{n:>5}R 的中{s['hit_rate']:>6.1%} ROI{s['roi']:>6.0%} [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]")

# ---- PG-4: ガールズ除外での主要セル再現 ----
print(f"\n【PG-4】ガールズ(全単騎L級)除外でのH1/H2再現（TEST）")
ng = [r for r in te if not r["girls"]]
print(calib_line("抜け逃げ×単騎逃げ", ng, lambda r: r["shape"] == "1車抜け" and r["s1"] == "逃" and r["other_nige"] == 0))
print(calib_line("抜け追(全体)", ng, lambda r: r["shape"] == "1車抜け" and r["s1"] == "追"))
b, n, _ = rate(ng, lambda r: r["shape"] == "2車抜け" and r["top2_same_line"], "top2_both")
print(f"  2車抜け×同ライン           {n:>5}R top2両方残存{b:>6.1%}")

# ---- 第2陣（探索・TRAIN方向 → TEST確認）----
print(f"\n【第2陣/探索】2車抜け脚質ペア・拮抗×追被り・下位断絶")
for name, rows in [("TRAIN", tr), ("TEST", te)]:
    print(f"  ── {name} ──")
    for pair in ["逃追", "逃逃", "追追", "両追", "両逃", "両両"]:
        cond = lambda r, pp=pair: r["shape"] == "2車抜け" and r["s12"] == "".join(sorted(pp))
        both, n, _ = rate(rows, cond, "top2_both")
        if n >= 10:
            print(f"  2車抜け×{pair:<6}{n:>5}R top2両方{both:>6.1%}")
    print(roi_line("拮抗×上位3全員追込", rows, lambda r: r["shape"] == "拮抗" and r["top3_all_oikomi"]))
    print(roi_line("下位断絶あり(bg≥q70)", rows, lambda r: r["bottom_gap"] is not None and r["bottom_gap"] >= bg_q70))
    print(roi_line("下位断絶なし", rows, lambda r: r["bottom_gap"] is not None and r["bottom_gap"] < bg_q70))
print()
