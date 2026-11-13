"""Second-leg scanner: thrust -> shallow pullback -> RSI cools in 44-60 -> closes >60 over 9/21 EMA. 1-minute bars, entries by 12:00 ET."""
import json,sys,requests,numpy as np,pandas as pd
from common import bars,HDR
P=dict(move5_min=3.0,atr_min=4.5,leg_atr=0.4,ret_lo=0.23,ret_hi=0.786,band=(44,60),band_bars=6,buf=0.002,cut=12*60)
U=json.load(open('universe.json'));syms=list(U['symbols'])
now=pd.Timestamp.now(tz='UTC');end=(now-pd.Timedelta(minutes=16)).strftime('%Y-%m-%dT%H:%M:%SZ')
def extras():
    try:
        c=[]
        m=requests.get('https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=20',headers=HDR,timeout=30).json()
        c+=[x['symbol'] for x in m.get('gainers',[])+m.get('losers',[])]
        a=requests.get('https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=40',headers=HDR,timeout=30).json()
        c+=[x['symbol'] for x in a.get('most_actives',[])]
        return [s for s in dict.fromkeys(c) if s.isalpha() and len(s)<=5 and s not in syms]
    except Exception as e:print('movers skipped',e);return []
ex=extras();allsyms=syms+ex+['SPY']
dstart=(now-pd.Timedelta(days=330)).strftime('%Y-%m-%d')
D=bars(list(dict.fromkeys(allsyms)),'1Day',dstart,end=end)
spy1=bars(['SPY'],'1Min',(now-pd.Timedelta(days=6)).strftime('%Y-%m-%d'),end=end)['SPY']
sess=spy1.index.max().normalize();sd=sess.strftime('%Y-%m-%d')
def gate(s,d):
    d=d[d.index.normalize()<sess]
    if len(d)<60:return None
    pc=d.close.shift();tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    atr=tr.tail(14).mean();last=d.iloc[-1]
    if atr/last.close*100<P['atr_min'] or last.close<5:return None
    c=d.close;ret=(c.iloc[-1]/c.iloc[-21]-1)*100 if len(c)>21 else 0
    ret5=(c.iloc[-1]/c.iloc[-6]-1)*100
    if abs(ret5)<P['move5_min']:return None
    d40=d.tail(41);cc=d40.close.pct_change().abs()*100;gp=((d40.open/d40.close.shift()-1).abs()*100)
    rng=last.high-last.low
    return dict(ret5=ret5,atr=atr,atr_pct=atr/last.close*100,prev_high=last.high,prev_close=last.close,ret20=ret,
        quiet40=bool(cc.tail(40).max()<6 and gp.tail(40).max()<4),
        far200=bool(len(c)>=200 and c.iloc[-1]/c.tail(200).mean()-1<-0.088),
        closed_low=bool(rng>0 and (last.close-last.low)/rng<=0.1))
spyd=D['SPY'][D['SPY'].index.normalize()<sess];spy20=(spyd.close.iloc[-1]/spyd.close.iloc[-21]-1)*100
G={s:g for s,d in D.items() if s!='SPY' and (g:=gate(s,d))}
print(sd,'session;',len(G),'of',len(D)-1,'names pass daily ATR >=4.5%')
M=bars(list(G),'1Min',sess.strftime('%Y-%m-%d'),end=end) if G else {}
def rsi(c,n=14):
    d=c.diff();u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();w=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+u/w.replace(0,np.nan))
rows=[]
for s,g in G.items():
    d=M.get(s)
    if d is None or len(d)<30:continue
    d=d[d.index.normalize()==sess]
    d=d[(d.index.hour*60+d.index.minute)<960]
    d=d.assign(rsi=rsi(d.close),e9=d.close.ewm(span=9,adjust=False).mean(),e21=d.close.ewm(span=21,adjust=False).mean())
    mins=d.index.hour*60+d.index.minute;rt=np.where((mins>=570)&(mins<960))[0]
    if len(rt)<15:continue
    last=None;o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values;r=d.rsi.values;e9=d.e9.values;e21=d.e21.values
    r0=rt[0];hod=h[r0];hi=r0;base=l[r0:r0+1].min();ev=None
    for i in rt:
        if mins[i]>P['cut'] and ev is None:break
        last=None
        if h[i]>hod:hod=h[i];hi=i
        base=min(l[r0:hi+1]);leg=hod-base
        if leg<P['leg_atr']*g['atr'] or i<=hi:continue
        pl=l[hi+1:i+1].min();ret=(hod-pl)/leg
        if not(P['ret_lo']<=ret<=P['ret_hi']):continue
        k=0;j=i-1
        while j>hi and P['band'][0]<=r[j]<=P['band'][1]:k+=1;j-=1
        if ev is None:
            if r[i]>P['band'][1] and k>=P['band_bars'] and c[i]>e9[i] and c[i]>e21[i] and mins[i]<=P['cut']:
                stop=pl*(1-P['buf']);tgt=pl+leg
                if tgt>c[i]>stop:ev=dict(i=i,entry=c[i],stop=stop,tgt=tgt,pl=pl,leg=leg,ret=ret,k=k,hi=hi,hod=hod,base=base)
            else:
                kk=0;j=i
                while j>hi and P['band'][0]<=r[j]<=P['band'][1]:kk+=1;j-=1
                last=dict(i=i,k=kk,pl=pl,leg=leg,ret=ret,hi=hi,hod=hod,base=base)
        # keep trigger frozen
    if ev:
        st='Triggered';ex_=None
        for j in range(ev['i']+1,len(d)):
            if l[j]<=ev['stop']:st='Stopped';ex_=j;break
            if h[j]>=ev['tgt']:st='Target hit';ex_=j;break
        e=ev;t=d.index[e['i']]
    else:
        if mins[len(d)-1]>P['cut']:continue
        if last is None or last['i']!=len(d)-1 or last['k']<1:continue
        e=last
        st='Ready' if e['k']>=P['band_bars'] else 'Cooling';ex_=None;t=d.index[e['i']]
        e=dict(e,entry=None,stop=e['pl']*(1-P['buf']),tgt=e['pl']+e['leg'])
    fl=[]
    if g['quiet40']:fl.append('Quiet 40 days')
    if abs(e['hod']/g['prev_high']-1)<=0.003:fl.append('Stalled at prior high')
    if g['ret20']-spy20<-8.6:fl.append('Trails SPY >8.6 pts')
    if g['far200']:fl.append('Below 200d >8.8%')
    if g['closed_low']:fl.append('Prior close at low')
    risk=(e['entry']-e['stop']) if e['entry'] else None
    ep=e['entry'] if e['entry'] else float(c[-1]);rk=ep-e['stop'];rr_=(e['tgt']-ep)/rk if rk>0 and e['tgt']>ep else None
    tmin=int(mins[e['i']]-570);leg_len=int(e['hi']-r0);rpct=rk/ep*100 if rk>0 else 0
    chk=[('Early (within 45 min of open)',tmin<=45),('Fast first leg (30 bars or fewer)',leg_len<=30),('Stop 1.2% or wider',rpct>=1.2),('Pullback 32% or deeper',e['ret']*100>=32),('R:R 2 or less',rr_ is not None and rr_<=2),('Clean thrust (8 or fewer new highs in the first leg)',int((np.diff(np.maximum.accumulate(h[r0:e['hi']+1]))>0).sum())<=8)]
    if e['entry'] and rk>0:
        rr_val=-1.0 if st=='Stopped' else ((e['tgt']-e['entry'])/rk if st=='Target hit' else (float(c[-1])-e['entry'])/rk)
    else:rr_val=None
    done=bool(mins[len(d)-1]>=955)
    pts=sum(1 for _,ok in chk if ok);met=[n for n,ok in chk if ok];miss=[n for n,ok in chk if not ok]
    sp=[float(x) for x in c[rt[0]:][::2]]
    rows.append(dict(symbol=s,status=st,atr_pct=round(g['atr_pct'],1),ret5=round(g['ret5'],1),leg_pct=round(e['leg']/e['base']*100,2),retrace=round(e['ret']*100),band_bars=int(e['k']),rsi=round(float(r[-1]),1),
        entry=None if e['entry'] is None else round(float(e['entry']),2),stop=round(float(e['stop']),2),target=round(float(e['tgt']),2),
        rr=None if not risk else round(float((e['tgt']-e['entry'])/risk),2),last=round(float(c[-1]),2),
        time=str(t)[:16],flags=fl,pts=pts,met=met,miss=miss,r=None if rr_val is None else round(float(rr_val),2),done=done,skip=bool(fl),group='Mover' if s in ex else U.get('groups',{}).get(s,''),spark=[round(x,2) for x in sp],
        lv=[round(float(e['stop']),2),round(float(e['tgt']),2)],asof=str(pd.Timestamp(end).tz_convert('America/New_York'))[:16]))
order={'Triggered':0,'Ready':1,'Cooling':2,'Target hit':3,'Stopped':4}
rows.sort(key=lambda x:(order[x['status']],x['time']),reverse=False)
json.dump(dict(session=sd,asof=str(pd.Timestamp(end).tz_convert('America/New_York'))[:16],n_pass=len(G),rows=rows),open('sl.json','w'))
print(len(rows),'rows:',{k:sum(1 for r in rows if r['status']==k) for k in order})
