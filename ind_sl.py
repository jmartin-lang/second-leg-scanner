import pickle,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from common import bars
D,M=pickle.load(open('/tmp/sl_cache.pkl','rb'))
now=pd.Timestamp.now(tz='UTC');end=(now-pd.Timedelta(minutes=16)).strftime('%Y-%m-%dT%H:%M:%SZ')
SP=bars(['SPY'],'1Min',(now-pd.Timedelta(days=45)).strftime('%Y-%m-%d'),end=end)['SPY']
def rsi(c,n=14):
    d=c.diff();u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();w=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+u/w.replace(0,np.nan))
def sim(g,i0,en,stop,tgt):
    risk=en-stop
    for j in range(i0+1,len(g['m'])):
        if g['m'][j]>=960:return 'eod',(g['c'][j-1]-en)/risk
        if g['l'][j]<=stop:return 'stop',-1.0
        if g['h'][j]>=tgt:return 'target',(tgt-en)/risk
    return 'eod',(g['c'][-1]-en)/risk
rows=[]
for s,m in M.items():
    d=D.get(s)
    if d is None:continue
    pc=d.close.shift();tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1);atrs=tr.rolling(14).mean();dset=set(d.index.normalize())
    for day,g0 in m.groupby(m.index.normalize()):
        prior=d[d.index.normalize()<day]
        if len(prior)<60 or day not in dset:continue
        atr=atrs.loc[prior.index[-1]];last=prior.iloc[-1]
        if atr/last.close*100<4.5 or last.close<5:continue
        g0=g0.copy();c=g0.close;g0['rsi']=rsi(c);e9s=c.ewm(span=9,adjust=False).mean();e21s=c.ewm(span=21,adjust=False).mean()
        macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();sig=macd.ewm(span=9,adjust=False).mean();mh=macd-sig
        tp=(g0.high+g0.low+g0.close)/3;vwap=(tp*g0.volume).cumsum()/g0.volume.cumsum()
        ma20=c.rolling(20).mean();sd20=c.rolling(20).std();bbw=(4*sd20/ma20);pctb=(c-(ma20-2*sd20))/(4*sd20)
        ll=g0.low.rolling(14).min();hh=g0.high.rolling(14).max();stoch=(c-ll)/(hh-ll)*100
        up=g0.high.diff();dn=-g0.low.diff();pdm=np.where((up>dn)&(up>0),up,0);ndm=np.where((dn>up)&(dn>0),dn,0)
        tr1=pd.concat([g0.high-g0.low,(g0.high-c.shift()).abs(),(g0.low-c.shift()).abs()],axis=1).max(axis=1);a14=tr1.ewm(alpha=1/14,adjust=False).mean()
        pdi=100*pd.Series(pdm,index=g0.index).ewm(alpha=1/14,adjust=False).mean()/a14;ndi=100*pd.Series(ndm,index=g0.index).ewm(alpha=1/14,adjust=False).mean()/a14
        adx=(100*(pdi-ndi).abs()/(pdi+ndi)).ewm(alpha=1/14,adjust=False).mean()
        obv=(np.sign(c.diff()).fillna(0)*g0.volume).cumsum()
        vma=g0.volume.rolling(20).mean()
        mins=(g0.index.hour*60+g0.index.minute).values;rt=np.where((mins>=570)&(mins<960))[0]
        if len(rt)<60:continue
        g=dict(h=g0.high.values,l=g0.low.values,c=c.values,o=g0.open.values,m=mins,v=g0.volume.values);r=g0.rsi.values;e9=e9s.values;e21=e21s.values
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
            if r[i]>60 and k>=6 and g['c'][i]>e9[i] and g['c'][i]>e21[i] and g['c'][i]>pl*1.002:ev=(i,pl,leg,ret,k,hi,hod,base);break
        if not ev:continue
        i,pl,leg,ret,k,hi,hod,base=ev;tc=g['c'][i];stop=pl*0.998;tgt=pl+leg
        if not(tgt>tc>stop):continue
        res,R=sim(g,i,tc,stop,tgt)
        # leg bars = r0..hi (but leg start = argmin low); pullback bars hi+1..i
        pb=slice(hi+1,i+1);lg=slice(r0,hi+1);rb=g['h'][i]-g['l'][i]
        sp=SP[SP.index.normalize()==day];st=sp.index[min(i,len(sp)-1)] if len(sp) else None
        spret=(sp.close.iloc[min(i,len(sp)-1)]/sp.open.iloc[0]-1)*100 if len(sp) else np.nan
        spv=(sp.close.iloc[min(i,len(sp)-1)]>((sp.high+sp.low+sp.close)/3*sp.volume).cumsum().iloc[min(i,len(sp)-1)]/sp.volume.cumsum().iloc[min(i,len(sp)-1)]) if len(sp) else np.nan
        rows.append(dict(sym=s,day=str(day.date()),res=res,R=R,tmin=mins[i]-570,leg_len=hi-r0,ret=ret*100,risk_pct=(tc-stop)/tc*100,rr=(tgt-tc)/(tc-stop),
          # new indicators
          macd_h=mh.values[i]/tc*100,macd_h_slope=(mh.values[i]-mh.values[i-3])/tc*100,macd_pos=float(macd.values[i]>0),
          vwap_dist=(tc/vwap.values[i]-1)*100,above_vwap=float(tc>vwap.values[i]),vwap_dist_pb_low=(pl/vwap.values[i]-1)*100,
          pctb=pctb.values[i],bbw=bbw.values[i],bbw_chg=bbw.values[i]/bbw.values[hi]-1 if bbw.values[hi]>0 else np.nan,
          stoch=stoch.values[i],adx=adx.values[i],di_diff=(pdi-ndi).values[i],
          obv_slope=(obv.values[i]-obv.values[hi])/max(1,g['v'][r0:i+1].sum()),
          vol_dryup=g['v'][pb].mean()/max(1,g['v'][lg].mean()),volx=g['v'][i]/vma.values[i] if vma.values[i] else np.nan,
          rsi_peak=np.nanmax(r[lg]),rsi_min_pb=np.nanmin(r[pb]),rsi_slope=r[i]-r[i-3],
          ema_gap=(e9[i]/e21[i]-1)*100,ema9_slope=(e9[i]/e9[i-3]-1)*100,ext9=(tc/e9[i]-1)*100,
          close_pos=(tc-g['l'][i])/rb if rb>0 else np.nan,trig_range_atr=rb/atr*100,trig_body=(tc-g['o'][i])/rb if rb>0 else np.nan,
          pb_below_e21=float((g['c'][pb]<e21[pb]).mean()),pb_len=i-hi,
          spy_ret=spret,spy_above_vwap=float(spv) if spv==spv else np.nan,
          roc5=(tc/g['c'][i-5]-1)*100,
          rel20=((prior.close.iloc[-1]/prior.close.iloc[-21]-1)-(D['SPY'][D['SPY'].index.normalize()<day].close.pipe(lambda z:z.iloc[-1]/z.iloc[-21]-1)))*100,
          far200=(prior.close.iloc[-1]/prior.close.tail(200).mean()-1)*100,gap=(g['o'][r0]/last.close-1)*100,atr_pct=atr/last.close*100))
df=pd.DataFrame(rows);df.to_csv('ind_sl.csv',index=False);print(len(df),'rows')
