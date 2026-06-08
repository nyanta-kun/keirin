"""keibaAI-v2 流: レース内標準化スコア × 閾値スイープ × 可変ベット
quinella/wide/trio BOX を「閾値超えの選手」に賭け、回収率とstd(リスク)を測定。
我々の固定pivot/固定EVとは別軸。自信ある時だけ賭ける（0点レースあり）。
"""
import itertools, numpy as np, pandas as pd
from scipy.special import comb
from src.preprocessing.feature_wt import load_raw_data_wt, build_features_wt, FEATURE_COLS_WT
from src.models.trainer import load_model
from src.evaluation.backtest_wt import _apply_pred_prob_wt, _load_payouts_wt

model = load_model('lgbm_wt_interim')
df = build_features_wt(load_raw_data_wt(min_date='2026-03-01'))
df = df[df['finish_order'].notna()].copy()
df = _apply_pred_prob_wt(model, df)
pm = _load_payouts_wt(df['race_key'].unique().tolist())

# レース内標準化スコア(z-score)
g = df.groupby('race_key')['pred_prob']
df['score_z'] = (df['pred_prob'] - g.transform('mean')) / g.transform('std').replace(0,1)

def simulate(score_col, threshold, market, max_riders=None):
    # returns list of per-race return rate (return_amount - bet)/... we track bet & ret
    races=bet=ret=hits=0; race_rets=[]
    for rk, grp in df.groupby('race_key'):
        n=len(grp)
        if max_riders and n>max_riders: continue
        sel = grp[grp[score_col] >= threshold]['frame_no'].astype(int).tolist()
        fin = grp[grp['finish_order']<=3].sort_values('finish_order')
        top3 = fin['frame_no'].astype(int).tolist()
        if len(top3)<3: continue
        races+=1
        rp = pm.get(rk,{})
        rb=0; rr=0
        if market=='quinella' and len(sel)>=2:
            top2=frozenset(top3[:2])
            for pair in itertools.combinations(sel,2):
                rb+=100
                if frozenset(pair)==top2: rr+=rp.get(('quinella',frozenset(pair)),0)
        elif market=='wide' and len(sel)>=2:
            t3=frozenset(top3)
            for pair in itertools.combinations(sel,2):
                rb+=100
                if frozenset(pair).issubset(t3): rr+=rp.get(('quinellaPlace',frozenset(pair)),0)
        elif market=='trio' and len(sel)>=3:
            t3=frozenset(top3)
            for tri in itertools.combinations(sel,3):
                rb+=100
                if frozenset(tri)==t3: rr+=rp.get(('trio',frozenset(tri)),0)
        elif market=='trifecta' and len(sel)>=3:
            order=tuple(top3)
            for perm in itertools.permutations(sel,3):
                rb+=100
                if tuple(perm)==order: rr+=rp.get(('trifecta',tuple(perm)),0)
        if rb>0:
            bet+=rb; ret+=rr; hits+= (rr>0); race_rets.append(rr-rb)
    if bet==0: return None
    roi=ret/bet
    std = (np.std(race_rets)*np.sqrt(len(race_rets))/bet) if bet else 0
    return dict(market=market, thr=threshold, n_bet_races=len(race_rets),
                bet=bet, roi=roi, hit_rate=hits/len(race_rets), std=std)

print(f"{'市場':>9} {'閾値z':>6} {'購入R':>6} {'的中率':>7} {'ROI':>8} {'std':>6}")
for market in ['trio','trifecta']:
    for thr in [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0]:
        r = simulate('score_z', thr, market)
        if r:
            print(f"{market:>9} {thr:>6.1f} {r['n_bet_races']:>6} {r['hit_rate']:>7.1%} {r['roi']:>8.1%} {r['std']:>6.2f}")
