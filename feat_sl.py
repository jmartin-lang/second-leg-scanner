import pickle,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from common import bars
D,M=pickle.load(open('/tmp/sl_cache.pkl','rb'))
now=pd.Timestamp.now(tz='UTC');end=(now-pd.Timedelta(minutes=16)).strftime('%Y-%m-%dT%H:%M:%SZ')
ST=(now-pd.Timedelta(days=45)).strftime('%Y-%m-%d')
IDX=bars(['SPY','QQQ','IWM'],'1Min',ST,end=end)
def rsi(c,n=14):
    d=c.diff();u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();w=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+u/w.replace(0,np.nan))
def mm(ix):return ix.hour*60+ix.minute
RM=np.arange(570,960)
# per-day matrix of return since the 9:30 open, for the whole universe
cols={}
for s,m in M.items():
    mi=mm(m.index);k=(mi>=570)&(mi<960);x=m[k];dd=x.index.normalize()
    for day,gx in x.groupby(dd):
        o=gx.open.iloc[0];ser=pd.Series(gx.close.values/o-1,index=mm(gx.index)).reindex(RM).ffill()
        cols.setdefault(day,{})[s]=ser
RET={day:pd.DataFrame(c) for day,c in cols.items()}
def idxser(sym,day):
    x=IDX[sym];x=x[x.index.normalize()==day];mi=mm(x.index);k=(mi>=570)&(mi<960);x=x[k]
    if len(x)<30:return None,None
    o=x.open.iloc[0];return pd.Series(x.close.values/o-1,index=mm(x.index)).reindex(RM).ffill(),x
def sim(h,l,c,m,i0,en,stop,tgt):
    risk=en-stop
    for j in range(i0+1,len(m)):
        if m[j]>=960:return 'eod',(c[j-1]-en)/risk
        if l[j]<=stop:return 'stop',-1.0
        if h[j]>=tgt:return 'target',(tgt-en)/risk
    return 'eod',(c[-1]-en)/risk
def mtf(g0,i,tf):
    r=g0.resample(f'{tf}min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    ts=g0.index[i]+pd.Timedelta(minutes=1)-pd.Timedelta(minutes=tf);p=r.index.searchsorted(ts,side='right')-1
    if p<30:return {}
    r=r.iloc[:p+1];c=r.close;e9=c.ewm(span=9,adjust=False).mean();e21=c.ewm(span=21,adjust=False).mean();ms=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();mh=ms-ms.ewm(span=9,adjust=False).mean()
    rs=rsi(c);ma=c.rolling(20).mean();sd=c.rolling(20).std();last=r.iloc[-1]
    return {f'm{tf}_rsi':rs.iloc[-1],f'm{tf}_rsi_slope':rs.iloc[-1]-rs.iloc[-4],f'm{tf}_e9gt21':float(e9.iloc[-1]>e21.iloc[-1]),f'm{tf}_c_gt_e21':float(c.iloc[-1]>e21.iloc[-1]),
      f'm{tf}_emagap':(e9.iloc[-1]/e21.iloc[-1]-1)*100,f'm{tf}_macdh':mh.iloc[-1]/c.iloc[-1]*100,f'm{tf}_macdh_up':float(mh.iloc[-1]>mh.iloc[-3]),f'm{tf}_pctb':(c.iloc[-1]-(ma.iloc[-1]-2*sd.iloc[-1]))/(4*sd.iloc[-1]) if sd.iloc[-1]>0 else np.nan,
      f'm{tf}_green':float(last.close>last.open),f'm{tf}_hh3':float((c.diff().iloc[-3:]>0).sum()),f'm{tf}_ret3':(c.iloc[-1]/c.iloc[-4]-1)*100}
rows=[];bad=0
for s,m in M.items():
    d=D.get(s)
    if d is None:continue
    pc=d.close.shift();tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1);atrs=tr.rolling(14).mean();dset=set(d.index.normalize())
    for day,g0 in m.groupby(m.index.normalize()):
        prior=d[d.index.normalize()<day]
        if len(prior)<130 or day not in dset or day not in RET:continue
        atr=atrs.loc[prior.index[-1]];last=prior.iloc[-1]
        if atr/last.close*100<4.5 or last.close<5:continue
        try:
            g0=g0.copy();c=g0.close;mins=mm(g0.index).values;rt=np.where((mins>=570)&(mins<960))[0]
            if len(rt)<60:continue
            h=g0.high.values;l=g0.low.values;cv=c.values;o=g0.open.values;v=g0.volume.values
            rsv=rsi(c).values;e9=c.ewm(span=9,adjust=False).mean().values;e21=c.ewm(span=21,adjust=False).mean().values
            tp=(g0.high+g0.low+g0.close)/3;vw=((tp*g0.volume).cumsum()/g0.volume.cumsum()).values
            r0=rt[0];hod=h[r0];hi=r0;ev=None
            for i in rt:
                if mins[i]>720:break
                if h[i]>hod:hod=h[i];hi=i
                if i<=hi:continue
                base=l[r0:hi+1].min();leg=hod-base
                if leg<0.4*atr:continue
                pl=l[hi+1:i+1].min();ret=(hod-pl)/leg
                if not(0.23<=ret<=0.786):continue
                k=0;j=i-1
                while j>hi and 44<=rsv[j]<=60:k+=1;j-=1
                if rsv[i]>60 and k>=6 and cv[i]>e9[i] and cv[i]>e21[i] and cv[i]>pl*1.002:ev=(i,pl,leg,ret,k,hi,hod,base);break
            if not ev:continue
            i,pl,leg,ret,k,hi,hod,base=ev;tc=cv[i];stop=pl*0.998;tgt=pl+leg
            if not(tgt>tc>stop):continue
            res,R=sim(h,l,cv,mins,i,tc,stop,tgt)
            f=dict(sym=s,day=str(day.date()),res=res,R=R)
            tm=mins[i];pb=slice(hi+1,i+1);lg=slice(r0,hi+1);pre=np.where(mins<570)[0]
            # leg start index (argmin low in r0..hi)
            ls=r0+int(np.argmin(l[r0:hi+1]));
            f.update(tmin=tm-570,leg_len=hi-r0,pb_len=i-hi,leg_atr=leg/atr,leg_pct=leg/base*100,ret=ret*100,band=k,rsi_trig=rsv[i],risk_pct=(tc-stop)/tc*100,rr=(tgt-tc)/(tc-stop),
              gap=(o[r0]/last.close-1)*100,gap_atr=(o[r0]-last.close)/atr,ext9=(tc/e9[i]-1)*100,atr_pct=atr/last.close*100,hod_vs_prevhigh=(hod/last.high-1)*100,open_to_trig=(tc/o[r0]-1)*100)
            # ---- index context / relative strength
            sp,spx=idxser('SPY',day);qq,_=idxser('QQQ',day);iw,_=idxser('IWM',day)
            so=cv[i]/o[r0]-1
            def at(ser,mn):return ser.loc[mn] if ser is not None else np.nan
            if sp is not None:
                for nm,ser in (('spy',sp),('qqq',qq),('iwm',iw)):
                    if ser is None:continue
                    f[f'{nm}_ret_open']=at(ser,tm)*100;f[f'rs_{nm}_open']=(so-at(ser,tm))*100
                    f[f'{nm}_ret_30']=(at(ser,tm)-at(ser,max(570,tm-30)))*100
                    f[f'rs_{nm}_30']=((cv[i]/cv[max(r0,i-30)]-1)-(at(ser,tm)-at(ser,max(570,tm-30))))*100
                f['rs_spy_15']=((cv[i]/cv[max(r0,i-15)]-1)-(at(sp,tm)-at(sp,max(570,tm-15))))*100
                f['rs_spy_60']=((cv[i]/cv[max(r0,i-60)]-1)-(at(sp,tm)-at(sp,max(570,tm-60))))*100
                f['spy_ret_15']=(at(sp,tm)-at(sp,max(570,tm-15)))*100;f['spy_ret_60']=(at(sp,tm)-at(sp,max(570,tm-60)))*100
                f['rs_spy_leg']=(leg/base-(at(sp,min(959,mins[hi]))-at(sp,mins[ls])))*100
                f['rs_spy_pb']=((cv[i]/hod-1)-(at(sp,tm)-at(sp,mins[hi])))*100
                sx=spx;sc=sx.close;f['spy_rsi5']=rsi(sc).iloc[min(len(sc)-1,tm-570)]
                tpp=(sx.high+sx.low+sx.close)/3;svw=(tpp*sx.volume).cumsum()/sx.volume.cumsum();f['spy_above_vwap']=float(sc.iloc[min(len(sc)-1,tm-570)]>svw.iloc[min(len(sc)-1,tm-570)])
                hh=sx.high.cummax();ll=sx.low.cummin();kk=min(len(sc)-1,tm-570);f['spy_range_pos']=(sc.iloc[kk]-ll.iloc[kk])/max(1e-9,hh.iloc[kk]-ll.iloc[kk])
            rm=RET[day].loc[tm];f['rank_open']=(rm<so).mean()*100;f['univ_med_open']=rm.median()*100;f['breadth_up']=(rm>0).mean()*100
            r15=RET[day].loc[max(570,tm-15)];f['rank_15']=((rm-r15)<(cv[i]/cv[max(r0,i-15)]-1)).mean()*100;f['univ_med_15']=(rm-r15).median()*100
            # ---- daily RS & trend
            dc=prior.close.values;sc_=D['SPY'][D['SPY'].index.normalize()<day].close.values
            for n in (5,10,20,60,120):f[f'rs_d{n}']=((dc[-1]/dc[-1-n]-1)-(sc_[-1]/sc_[-1-n]-1))*100
            f['rs_d20_slope']=f['rs_d20']-(((dc[-6]/dc[-26]-1)-(sc_[-6]/sc_[-26]-1))*100)
            for n in (1,5,20):f[f'ret_d{n}']=(dc[-1]/dc[-1-n]-1)*100
            f['dist_52wh']=(dc[-1]/prior.high.tail(252).max()-1)*100;f['dist_20dh']=(dc[-1]/prior.high.tail(20).max()-1)*100;f['new20high']=float(dc[-1]>=prior.close.tail(21).iloc[:-1].max())
            f['up_days10']=float((np.diff(dc[-11:])>0).sum())
            s20,s50,s200=dc[-20:].mean(),dc[-50:].mean(),dc[-200:].mean() if len(dc)>=200 else dc.mean()
            f.update(above_s20=float(dc[-1]>s20),above_s50=float(dc[-1]>s50),above_s200=float(dc[-1]>s200),s20gt50=float(s20>s50),dist_s20_atr=(dc[-1]-s20)/atr,dist_s50_atr=(dc[-1]-s50)/atr,dist_s200_atr=(dc[-1]-s200)/atr,far200=(dc[-1]/s200-1)*100)
            f['d_rsi']=rsi(pd.Series(dc)).iloc[-1];dm=pd.Series(dc).ewm(span=12,adjust=False).mean()-pd.Series(dc).ewm(span=26,adjust=False).mean();f['d_macdh']=(dm-dm.ewm(span=9,adjust=False).mean()).iloc[-1]/dc[-1]*100
            rg=last.high-last.low;f['pd_closepos']=(last.close-last.low)/rg if rg>0 else np.nan;f['pd_range_atr']=rg/atr;f['pd_ret']=(last.close/prior.close.iloc[-2]-1)*100
            f['pd_vol_ratio']=last.volume/prior.volume.tail(20).mean();f['inside_day']=float(last.high<=prior.high.iloc[-2] and last.low>=prior.low.iloc[-2]);f['nr7']=float(rg<=(prior.high-prior.low).tail(7).min()+1e-9)
            f['atr_trend']=atr/atrs.loc[prior.index[-1]-pd.Timedelta(days=0)] if False else atr/tr.tail(50).mean();f['range5_atr']=(prior.high.tail(5).max()-prior.low.tail(5).min())/atr
            avgv=prior.volume.tail(20).mean();f['log_price']=np.log(tc);f['log_dollar_vol']=np.log(avgv*tc);f['log_avgvol']=np.log(avgv)
            cum=v[r0:i+1].sum();f['rvol_cum']=cum/(avgv*max(1,(tm-570))/390)
            # ---- premarket + opening range + levels
            if len(pre)>5:
                pmh=h[pre].max();pml=l[pre].min();pmv=v[pre].sum();f.update(pm_vol_ratio=pmv/avgv,pm_range_atr=(pmh-pml)/atr,pm_ret=(cv[pre][-1]/last.close-1)*100,trig_vs_pmh_atr=(tc-pmh)/atr,trig_above_pmh=float(tc>pmh),open_vs_pmh_atr=(o[r0]-pmh)/atr,hod_above_pmh=float(hod>pmh))
            else:pmh=pml=np.nan
            for n in (5,15,30):
                if r0+n<=i:
                    oh=h[r0:r0+n].max();ol=l[r0:r0+n].min();f[f'or{n}_range_atr']=(oh-ol)/atr;f[f'trig_vs_or{n}h_atr']=(tc-oh)/atr;f[f'above_or{n}h']=float(tc>oh);f[f'or{n}_ret']=(cv[r0+n-1]/o[r0]-1)*100
            f['hod_time']=mins[hi]-570;f['n_newhod_leg']=float((np.diff(np.maximum.accumulate(h[r0:hi+1]))>0).sum());f['hod_is_day_high']=float(hod>=(np.nanmax(h[pre]) if len(pre) else 0))
            lv={'pdh':last.high,'pdl':last.low,'pdc':last.close,'s20':s20,'s50':s50,'s200':s200,'vwap':vw[i]}
            if pmh==pmh:lv['pmh']=pmh
            for nm,x in lv.items():f[f'dist_{nm}_atr']=(tc-x)/atr
            above=[x for x in lv.values() if tc<x<=tgt*1.0005];f['levels_to_target']=float(len(above));f['room_to_pdh_atr']=(last.high-tc)/atr
            rnd=np.round(tc/ (5 if tc>100 else 1))*(5 if tc>100 else 1);f['dist_round_atr']=abs(tc-rnd)/atr
            # ---- structure
            lgc=cv[ls:hi+1];f['leg_eff']=abs(lgc[-1]-lgc[0])/max(1e-9,np.abs(np.diff(lgc)).sum()) if len(lgc)>2 else np.nan
            f['leg_up_frac']=float((np.diff(lgc)>0).mean()) if len(lgc)>2 else np.nan
            rb1=(h-l);a1=pd.Series(rb1).rolling(14).mean().values;f['leg_maxbar_atr1m']=rb1[lg].max()/a1[hi] if a1[hi]>0 else np.nan
            f['leg_vol_ratio']=v[lg].mean()/max(1,np.mean(v[max(r0,0):i+1]));f['vol_dryup']=v[pb].mean()/max(1,v[lg].mean());f['trig_vol_vs_pb']=v[i]/max(1,v[pb].mean());f['trig_vol_rank_pb']=(v[pb]<v[i]).mean()
            pbc=cv[hi+1:i+1];f['pb_slope']=((pbc[-1]/pbc[0]-1)*100)/len(pbc);f['pb_range_atr']=(hod-pl)/atr;f['pb_lower_highs']=float((np.diff(h[hi:i+1])<0).sum());f['pb_close_vs_mid']=(tc-(base+leg/2))/leg
            f['rsi_peak']=np.nanmax(rsv[lg]);f['rsi_min_pb']=np.nanmin(rsv[pb]);f['rsi_mean_pb']=np.nanmean(rsv[pb]);f['rsi_in_band']=float(((rsv[pb]>=44)&(rsv[pb]<=60)).mean());f['rsi_below44']=float((rsv[pb]<44).any());f['rsi_slope3']=rsv[i]-rsv[i-3];f['rsi_prev']=rsv[i-1]
            f['e9_cross']=float((np.diff((cv[pb]>e9[pb]).astype(int))!=0).sum());f['pb_below_e9']=float((cv[pb]<e9[pb]).mean());f['pb_below_e21']=float((cv[pb]<e21[pb]).mean());f['pb_below_vwap']=float((cv[pb]<vw[pb]).mean())
            f['pl_vs_vwap_atr']=(pl-vw[i])/atr;f['hod_vs_vwap_atr']=(hod-vw[i])/atr;f['trig_vs_vwap_atr']=(tc-vw[i])/atr;f['pl_above_vwap']=float(pl>vw[i]);f['ls_vs_vwap_atr']=(base-vw[ls])/atr
            f['pl_vs_e21_atr']=(pl-e21[i])/atr;f['ema_gap']=(e9[i]/e21[i]-1)*100;f['ema9_slope']=(e9[i]/e9[i-3]-1)*100
            br=h[i]-l[i];f['trig_body_atr']=(tc-o[i])/atr*100;f['trig_range_atr']=br/atr*100;f['trig_close_pos']=(tc-l[i])/br if br>0 else np.nan;f['trig_upper_wick']=(h[i]-max(tc,o[i]))/br if br>0 else np.nan
            f['trig_breaks_prev3']=float(tc>h[i-3:i].max());f['atr1m_pct']=a1[i]/tc*100;f['atr1m_vs_daily']=a1[i]/atr*100;f['spread_proxy']=np.mean(rb1[i-19:i+1]/cv[i-19:i+1])*100
            f['dow']=float(day.dayofweek);f['mins_to_close']=960-tm
            for tf in (5,15,30):f.update(mtf(g0,i,tf))
            f['mtf_align']=sum(f.get(f'm{t}_e9gt21',0) for t in (5,15,30));f['mtf_rsi50']=sum(float(f.get(f'm{t}_rsi',0)>50) for t in (5,15,30));f['mtf_above_e21']=sum(f.get(f'm{t}_c_gt_e21',0) for t in (5,15,30))
            rows.append(f)
        except Exception as e:
            bad+=1
            if bad<4:print('err',s,day.date(),repr(e))
df=pd.DataFrame(rows);df.to_csv('feat_sl.csv',index=False);print(len(df),'rows',df.shape[1]-4,'features; errors',bad)
