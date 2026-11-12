import json,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from common import bars
P=dict(atr_min=4.5,leg_atr=0.4,ret_lo=0.23,ret_hi=0.786,bb=6,buf=0.002,cut=720)
U=json.load(open('universe.json'));syms=list(U['symbols'])
now=pd.Timestamp.now(tz='UTC');end=(now-pd.Timedelta(minutes=16)).strftime('%Y-%m-%dT%H:%M:%SZ')
D=bars(syms+['SPY'],'1Day',(now-pd.Timedelta(days=330)).strftime('%Y-%m-%d'),end=end)
M=bars(syms,'1Min',(now-pd.Timedelta(days=45)).strftime('%Y-%m-%d'),end=end)
def rsi(c,n=14):
    d=c.diff();u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();w=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+u/w.replace(0,np.nan))
spy=D['SPY'];out=[]
for s,m in M.items():
    d=D.get(s)
    if d is None:continue
    pc=d.close.shift();tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1);atrs=tr.rolling(14).mean()
    for day,g in m.groupby(m.index.normalize()):
        prior=d[d.index.normalize()<day]
        if len(prior)<60 or day not in set(d.index.normalize()):continue
        atr=atrs.loc[prior.index[-1]];last=prior.iloc[-1]
        if atr/last.close*100<P['atr_min'] or last.close<5:continue
        sp=spy[spy.index.normalize()<day];c=prior.close
        rel=((c.iloc[-1]/c.iloc[-21]-1)-(sp.close.iloc[-1]/sp.close.iloc[-21]-1))*100
        far=(c.iloc[-1]/c.tail(200).mean()-1)*100 if len(c)>=200 else 0
        g=g.copy();g['rsi']=rsi(g.close);g['e9']=g.close.ewm(span=9,adjust=False).mean();g['e21']=g.close.ewm(span=21,adjust=False).mean();g['vma']=g.volume.rolling(20).mean()
        mins=(g.index.hour*60+g.index.minute).values;rt=np.where((mins>=570)&(mins<960))[0]
        if len(rt)<60:continue
        h=g.high.values;l=g.low.values;c1=g.close.values;r=g.rsi.values;e9=g.e9.values;e21=g.e21.values;v=g.volume.values;vm=g.vma.values
        r0=rt[0];hod=h[r0];hi=r0;ev=None
        for i in rt:
            if mins[i]>P['cut']:break
            if h[i]>hod:hod=h[i];hi=i
            if i<=hi:continue
            base=l[r0:hi+1].min();leg=hod-base
            if leg<P['leg_atr']*atr:continue
            pl=l[hi+1:i+1].min();ret=(hod-pl)/leg
            if not(P['ret_lo']<=ret<=P['ret_hi']):continue
            k=0;j=i-1
            while j>hi and 44<=r[j]<=60:k+=1;j-=1
            if r[i]>60 and k>=P['bb'] and c1[i]>e9[i] and c1[i]>e21[i]:
                stop=pl*(1-P['buf']);tgt=pl+leg
                if tgt>c1[i]>stop:ev=(i,c1[i],stop,tgt,pl,leg,ret,k,hi,hod,base);break
        if not ev:continue
        i,en,stop,tgt,pl,leg,ret,k,hi,hod,base=ev;res='open';xi=len(g)-1
        for j in range(i+1,len(g)):
            if mins[j]>=960:xi=j-1;break
            if l[j]<=stop:res='stop';xi=j;break
            if h[j]>=tgt:res='target';xi=j;break
        if res=='open':res='eod'
        risk=en-stop;R=(tgt-en)/risk if res=='target' else (-1 if res=='stop' else (c1[xi]-en)/risk)
        open0=g.open.values[r0]
        out.append(dict(sym=s,day=str(day.date()),res=res,R=R,atr_pct=atr/last.close*100,leg_pct=leg/base*100,leg_atr=leg/atr,ret=ret*100,band=k,rsi=r[i],
          tmin=mins[i]-570,since_hod=i-hi,pb_len=i-hi,leg_len=hi-r0,risk_pct=risk/en*100,rr=(tgt-en)/risk,volx=v[i]/vm[i] if vm[i] else np.nan,
          gap=(open0/last.close-1)*100,rel20=rel,far200=far,ext9=(en/e9[i]-1)*100,hod_vs_prev=(hod/last.high-1)*100,
          dayrng_ret=(en/open0-1)*100,mfe_before=0))
df=pd.DataFrame(out);df.to_csv('bt_sl.csv',index=False);print(len(df),'triggers',df.day.nunique(),'days');print(df.res.value_counts())
