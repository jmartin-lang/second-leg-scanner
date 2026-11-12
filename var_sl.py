import pickle,itertools,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
D,M=pickle.load(open('/tmp/sl_cache.pkl','rb'))
def rsi(c,n=14):
    d=c.diff();u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();w=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+u/w.replace(0,np.nan))
def sim(g,i0,en,stop,tgt):
    """walk from bar i0+1; stop-first on same bar. returns (res,R)"""
    risk=en-stop
    if risk<=0 or tgt<=en:return None
    for j in range(i0+1,len(g['m'])):
        if g['m'][j]>=960:return 'eod',(g['c'][j-1]-en)/risk
        if g['l'][j]<=stop:return 'stop',-1.0
        if g['h'][j]>=tgt:return 'target',(tgt-en)/risk
    return 'eod',(g['c'][-1]-en)/risk
spy=D['SPY'];rows=[]
for s,m in M.items():
    d=D.get(s)
    if d is None:continue
    pc=d.close.shift();tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1);atrs=tr.rolling(14).mean();dset=set(d.index.normalize())
    for day,g0 in m.groupby(m.index.normalize()):
        prior=d[d.index.normalize()<day]
        if len(prior)<60 or day not in dset:continue
        atr=atrs.loc[prior.index[-1]];last=prior.iloc[-1]
        if atr/last.close*100<4.5 or last.close<5:continue
        g0=g0.copy();g0['rsi']=rsi(g0.close);g0['e9']=g0.close.ewm(span=9,adjust=False).mean();g0['e21']=g0.close.ewm(span=21,adjust=False).mean();g0['vma']=g0.volume.rolling(20).mean()
        mins=(g0.index.hour*60+g0.index.minute).values;rt=np.where((mins>=570)&(mins<960))[0]
        if len(rt)<60:continue
        g=dict(h=g0.high.values,l=g0.low.values,c=g0.close.values,o=g0.open.values,m=mins);r=g0.rsi.values;e9=g0.e9.values;e21=g0.e21.values;v=g0.volume.values;vm=g0.vma.values
        r0=rt[0];hod=g['h'][r0];hi=r0;ev=None
        for i in rt:
            if mins[i]>720:break
            if g['h'][i]>hod:hod=g['h'][i];hi=i
            if i<=hi:continue
            base=g['l'][r0:hi+1].min();leg=hod-base
            if leg<0.4*atr:continue
            pl=g['l'][hi+1:i+1].min();ret=(hod-pl)/leg
            if not(0.23<=ret<=0.786):continue
            k=0;j=i-1
            while j>hi and 44<=r[j]<=60:k+=1;j-=1
            if r[i]>60 and k>=6 and g['c'][i]>e9[i] and g['c'][i]>e21[i] and g['c'][i]>pl*1.002>0:
                ev=(i,pl,leg,ret,k,hi,hod,base);break
        if not ev:continue
        i,pl,leg,ret,k,hi,hod,base=ev;tc=g['c'][i]
        # entries: (idx, price)
        E={'E1 close':(i,tc)}
        if i+1<len(g['o']):E['E2 next open']=(i+1,g['o'][i+1])
        for j in range(i+1,min(i+6,len(g['h']))):
            if g['h'][j]>g['h'][i]:E['E3 break trigger high (5 bars)']=(j,max(g['o'][j],g['h'][i]+0.01));break
        for j in range(i+1,min(i+11,len(g['l']))):
            if g['l'][j]<=e9[i]:E['E4 pullback to 9EMA (10 bars)']=(j,min(g['o'][j],e9[i]));break
        mid=(tc+pl)/2
        for j in range(i+1,min(i+11,len(g['l']))):
            if g['l'][j]<=mid:E['E5 limit halfway to pullback low']=(j,min(g['o'][j],mid));break
        dayret=None
        ctx=dict(sym=s,day=str(day.date()),tmin=mins[i]-570,leg_len=hi-r0,pb_len=i-hi,leg_atr=leg/atr,ret=ret*100,band=k,volx=v[i]/vm[i] if vm[i] else np.nan,
                 rr1=(pl+leg-tc)/(tc-pl*0.998),risk_pct=(tc-pl*0.998)/tc*100,atr_pct=atr/last.close*100,gap=(g['o'][r0]/last.close-1)*100)
        stops={'S1 below PB low 0.2%':pl*0.998,'S2 PB low -0.15 daily ATR':pl-0.15*atr,'S3 PB low -0.3 daily ATR':pl-0.3*atr}
        for en_n,(ij,ep) in E.items():
            for st_n,st in stops.items():
                for tg_n,tg in {'T1 1.0x leg':pl+leg,'T2 retest HOD':hod,'T3 1R':ep+(ep-st),'T4 2R':ep+2*(ep-st)}.items():
                    x=sim(g,ij,ep,st,tg)
                    if x:rows.append(dict(ctx,entry=en_n,stop=st_n,tgt=tg_n,res=x[0],R=x[1],risk_pct_v=(ep-st)/ep*100))
df=pd.DataFrame(rows);df.to_pickle('/tmp/var_sl.pkl');print(len(df),'rows',df[['sym','day']].drop_duplicates().shape[0],'setups')
