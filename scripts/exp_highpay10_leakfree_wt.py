"""万車券狙い三連単~10点 × 構成別構造 — リーク無し・本番忠実セマンティクスでの検証。

ユーザーの問い(2026-06-12): 三連単で万車券(≥10,000円)以上を狙い、的中率10-20%・
1R~10点・出走馬の構成により買い目構造をわけて、ROI>100% を達成できるか。

前提となる既存資産:
  - docs/analysis/14: 高配当検知×~10点は検証済みだが旧セマンティクス（欠車生存バイアス＋
    完走者基準≤6車＋週次再学習リーク）。TE826%等は要再解釈。
  - docs/analysis/18: リーク無し再採点で全レバー~70-90%。本実験は doc14 の問いを
    `exp_leakfree_rescore_wt.py` と同一の本番忠実セマンティクスで初めて再検証するもの。

成立条件の算数: 10点×100円=1,000円/R。的中15%でROI100%には平均獲得配当≥6,667円、
的中10%なら≥10,000円が必要＝万車券帯を常用的に取る必要がある。

事前登録（後出し追加はしない）:
  対象: 出走表基準 n∈{5,6}（7+はdocs/analysis/05,14でクローズ・対象外）
  モデル: リーク無し（TRAIN 2023-07〜2025-06 限定学習）。ランキングは全エントリー。
  欠車: 買い目にDNS選手を含む点は除外（返還相当・notify準拠）。
  構造4種（モデル順位 fr[0..]=p1,p2,..・市場本命mf）:
    F0 doc14のS3: perms{p1,p2,p3}(6点) + p1⇄p2→{r4,r5}(4点) = 10点【対照・再採点】
    F1 本命崩し: H={モデル上位2(mfを除く)}, K={残り上位3}。
       BOX(h1,h2)→K(6点) + h1→{K上位2}→h2 / h2→{K上位2}→h1(4点) = 10点
    F2 モデル最尤10点: top4順列24点をHarvilleスコア順に上位10点【本命型対照】
    F3 2着波乱: p1→2着{r3,r4,r5}→3着{p2,r3,r4,r5}\2着 = 9点
  HPオーバーレイ(+HP): 各構造のうち三連単最終オッズ≥100倍の点のみ購入（0点=見送り）。
    ※最終オッズでの点選別は本番では朝オッズ＋ドリフトで変動する（上限値の留保が最大の帯）。
  選別: ALL / UPSET(top3_sum≤TRAIN Q1・TRAINで固定) / MM(fav_mismatch)
  Part2 構成別ポリシー: セル= n(5/6)×gap12三分位(TRAIN固定)。各セルで
    TRAIN ROI最大の(構造,overlay)を選択（制約: TRAIN的中≥10%・該当無しは見送り）。
    合成ポリシーを VAL/HOLD で評価（=「構成によりわけて購入」の直接検証）。
  判定: リーク無しモデルで VAL と HOLDOUT の両方が ROI>100% かつ 的中10-20%。
  期間: TRAIN 2023-07〜2025-06 / VAL 2025-07〜2026-02 / HOLDOUT 2026-03〜06-12。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import itertools
import numpy as np
import lightgbm as lgb

from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, prepare_X
from exp_segment_first_wt import load_boards, market_fav, LGB_PARAMS, TRAIN, VAL, HOLD
from roi_robustness_wt import roi_summary

HIGH = 100.0  # 万車券 = 100倍以上（100円賭け→10,000円）


def harville(probs, idx3):
    """正規化pred_probのHarville近似で並び(a,b,c)の尤度。順序付けにのみ使用。"""
    a, b, c = idx3
    pa, pb, pc = probs[a], probs[b], probs[c]
    d1 = 1.0 - pa
    d2 = 1.0 - pa - pb
    if d1 <= 0 or d2 <= 0:
        return 0.0
    return pa * (pb / d1) * (pc / d2)


def build_structs(fr, probs, mf):
    """モデル順位fr・正規化勝率probs(dict frame->p)・市場本命mfから構造別の買い目リスト。"""
    p1, p2, p3 = fr[0], fr[1], fr[2]
    out = {}
    # F0: doc14 S3
    c = [t for t in itertools.permutations((p1, p2, p3))]
    c += [(a, b, x) for a, b in ((p1, p2), (p2, p1)) for x in fr[3:5]]
    out["F0"] = c
    # F1: 本命崩し（市場本命を1着から外す）
    if mf is None:
        out["F1"] = None
    else:
        heads = [f for f in fr if f != mf][:2]
        h1, h2 = heads
        K = [f for f in fr if f not in (h1, h2)][:3]
        c = [(a, b, x) for a, b in ((h1, h2), (h2, h1)) for x in K]
        c += [(h1, k, h2) for k in K[:2]] + [(h2, k, h1) for k in K[:2]]
        out["F1"] = c
    # F2: モデル最尤10点（本命型対照）
    top4 = fr[:4]
    cands = sorted(itertools.permutations(top4, 3),
                   key=lambda t: -harville(probs, t))
    out["F2"] = list(cands[:10])
    # F3: 2着波乱（p1 1着・2着に4番手以下、対抗p2は2着から外す）
    K3 = fr[2:5]
    c = []
    for k in K3:
        for x in [p2] + [y for y in K3 if y != k]:
            c.append((p1, k, x))
    out["F3"] = c
    return out


def collect():
    print("loading & building features (2023-07〜2026-06-12)...", flush=True)
    df = build_features_wt(load_raw_data_wt(min_date=TRAIN[0], max_date=HOLD[1]))

    fit = df[(df["race_date"] <= TRAIN[1]) & (df["finish_order"] >= 1)]
    print(f"  leakfree LGBM on TRAIN only ({len(fit):,} rows)...", flush=True)
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(prepare_X(fit), fit["top3_flag"])
    df["p"] = m.predict_proba(prepare_X(df))[:, 1]

    # 本番忠実: 出走表(エントリー)基準 n∈{5,6}・ランキングは全エントリー
    sz = df.groupby("race_key")["frame_no"].count()
    df = df[df["race_key"].isin(sz[sz.isin([5, 6])].index)].copy()
    done = df.groupby("race_key")["finish_order"].apply(lambda s: (s >= 1).sum() >= 3)
    df = df[df["race_key"].isin(done[done].index)]
    print(f"  universe: {df['race_key'].nunique():,} races (entries 5-6, 結果確定)", flush=True)
    trio_b, tf_b, _ = load_boards(df["race_key"].unique().tolist())

    races = []
    for rk, g0 in df.groupby("race_key"):
        bd = trio_b.get(rk, {})
        tfb = tf_b.get(rk, {})
        if not bd or not tfb:
            continue
        fin = g0[g0["finish_order"].between(1, 3)]
        if len(fin) < 3:
            continue
        order = tuple(fin.sort_values("finish_order")["frame_no"].astype(int).tolist())
        dns = set(g0[g0["finish_order"] == 0]["frame_no"].astype(int).tolist())
        g = g0.sort_values("p", ascending=False)
        fr = g["frame_no"].astype(int).tolist()
        p = g["p"].tolist()
        tot = sum(p)
        probs = {f: v / tot for f, v in zip(fr, p)}
        mf = market_fav(bd)
        structs = build_structs(fr, probs, mf)
        legs = {}
        for st, combos in structs.items():
            if combos is None:
                legs[st] = None
                continue
            pts = []
            for cmb in combos:
                if any(x in dns for x in cmb):     # 欠車点=返還相当で除外
                    continue
                o = tfb.get(cmb)
                if o is None:
                    continue
                pts.append((o, cmb == order))
            legs[st] = pts
        races.append({
            "date": g0["race_date"].iloc[0], "n": len(g0),
            "gap12": p[0] - p[1], "t3s": p[0] + p[1] + p[2],
            "mm": (mf is not None and mf != fr[0]),
            "legs": legs,
        })
    return races


def score(races, st, hp, sel):
    """構造st(+HPオーバーレイ)×選別selのROI集計。"""
    pays, bets, hi_hits = [], [], 0
    for r in races:
        if not sel(r):
            continue
        pts = r["legs"].get(st)
        if pts is None:
            continue
        if hp:
            pts = [(o, h) for o, h in pts if o >= HIGH]
        if not pts:
            continue
        pay = sum(o * 100 for o, h in pts if h)
        if pay >= HIGH * 100:
            hi_hits += 1
        pays.append(pay)
        bets.append(len(pts) * 100)
    s = roi_summary(pays, bets)
    s["hi_rate"] = hi_hits / max(s["hits"], 1)      # 的中のうち万車券だった率
    s["avg_pts"] = float(np.mean([b / 100 for b in bets])) if bets else 0.0
    return s


def fmt(s):
    if s["n"] == 0:
        return f"{0:>5}R  --"
    return (f"{s['n']:>5}R {s['avg_pts']:>4.1f}点 的{s['hit_rate']:>5.1%}"
            f" ROI{s['roi']:>5.0%} [{s['ci_lo']:>4.0%},{s['ci_hi']:>5.0%}]"
            f" 除{s['roi_ex_max']:>4.0%} 中央{s['median_hit']:>7,.0f}円 万率{s['hi_rate']:>4.0%}")


def main():
    races = collect()
    by = {"TRAIN": [r for r in races if r["date"] <= TRAIN[1]],
          "VAL":   [r for r in races if TRAIN[1] < r["date"] <= VAL[1]],
          "HOLD":  [r for r in races if r["date"] > VAL[1]]}
    print(f"  TRAIN {len(by['TRAIN'])} / VAL {len(by['VAL'])} / HOLDOUT {len(by['HOLD'])}")

    q1 = float(np.percentile([r["t3s"] for r in by["TRAIN"]], 25))
    g33, g67 = np.percentile([r["gap12"] for r in by["TRAIN"]], [33.4, 66.7])
    print(f"  TRAIN固定閾値: top3_sum Q1={q1:.3f} / gap12三分位=({g33:.3f},{g67:.3f})")

    SELS = [("ALL",   lambda r: True),
            ("UPSET", lambda r: r["t3s"] <= q1),
            ("MM",    lambda r: r["mm"])]
    STRUCTS = ["F0", "F1", "F2", "F3"]

    print(f"\n{'='*132}")
    print("【Part1】構造×HPオーバーレイ×選別（リーク無しモデル・出走表基準5-6車・欠車点除外・払戻=最終オッズ上限値）")
    print(f"  判定 = VAL&HOLD 両方 ROI>100% かつ 的中10-20%")
    print(f"{'='*132}")
    for slab, sel in SELS:
        print(f"\n  ════ 選別: {slab} ════")
        for st in STRUCTS:
            for hp in (False, True):
                lab = f"{st}{'+HP' if hp else '   '}"
                rows = {per: score(by[per], st, hp, sel) for per in ("TRAIN", "VAL", "HOLD")}
                ok = all(rows[p]["roi"] > 1 and 0.10 <= rows[p]["hit_rate"] <= 0.20
                         for p in ("VAL", "HOLD")) and rows["VAL"]["n"] > 0
                star = " ◎" if ok else ""
                print(f"  ── {lab}{star}")
                for per in ("TRAIN", "VAL", "HOLD"):
                    print(f"     {per:<6}{fmt(rows[per])}")

    # ---- Part2: 構成別ポリシー（n×gap12三分位の6セルでTRAIN最良構造を選択）----
    def cell_of(r):
        gtier = 0 if r["gap12"] < g33 else (1 if r["gap12"] < g67 else 2)
        return (r["n"], gtier)

    print(f"\n{'='*132}")
    print("【Part2】構成別ポリシー: セル=n(5/6)×gap12三分位。TRAINでROI最大(制約:的中≥10%)の構造を選択→VAL/HOLDで合成評価")
    print(f"{'='*132}")
    GLAB = {0: "混戦", 1: "中", 2: "固い"}
    for slab, sel in [("ALL", SELS[0][1]), ("UPSET", SELS[1][1])]:
        choice = {}
        print(f"\n  ════ 選別: {slab} ════")
        for cell in sorted({cell_of(r) for r in by["TRAIN"]}):
            insel = lambda r, c=cell: sel(r) and cell_of(r) == c
            best, best_roi = None, -1
            for st in STRUCTS:
                for hp in (False, True):
                    s = score(by["TRAIN"], st, hp, insel)
                    if s["n"] >= 30 and s["hit_rate"] >= 0.10 and s["roi"] > best_roi:
                        best, best_roi = (st, hp), s["roi"]
            choice[cell] = best
            blab = f"{best[0]}{'+HP' if best[1] else ''}" if best else "見送り"
            print(f"    セル n={cell[0]} gap12={GLAB[cell[1]]:<3} → {blab:<7} (TRAIN ROI {best_roi:.0%})"
                  if best else f"    セル n={cell[0]} gap12={GLAB[cell[1]]:<3} → 見送り")
        # 合成評価
        for per in ("TRAIN", "VAL", "HOLD"):
            pays, bets = [], []
            for r in by[per]:
                if not sel(r):
                    continue
                ch = choice.get(cell_of(r))
                if not ch:
                    continue
                st, hp = ch
                pts = r["legs"].get(st)
                if pts is None:
                    continue
                if hp:
                    pts = [(o, h) for o, h in pts if o >= HIGH]
                if not pts:
                    continue
                pays.append(sum(o * 100 for o, h in pts if h))
                bets.append(len(pts) * 100)
            s = roi_summary(pays, bets)
            s["hi_rate"] = 0.0
            s["avg_pts"] = float(np.mean([b / 100 for b in bets])) if bets else 0.0
            days = max(1, len({r["date"] for r in by[per]}))
            print(f"    合成 {per:<6}{fmt(s)}  ({s['n']/days:.2f}R/日)")

    print(f"\n{'='*132}")
    print("  読み方: リーク無しVAL/HOLDが実力値。払戻=最終オッズ上限値（万車券帯は朝→確定ドリフトの下振れ最大）。")
    print("  +HPの点選別も最終オッズ基準＝本番では朝オッズ選別となりさらに不利。")
    print(f"{'='*132}")


if __name__ == "__main__":
    main()
