"""Append the finished session's triggers to track.json (one row per symbol per day). Run after the close."""
import json,os
S=json.load(open('sl.json'));T=json.load(open('track.json')) if os.path.exists('track.json') else []
have={(t['day'],t['sym']) for t in T};n=0
for r in S['rows']:
    if r['status'] not in('Triggered','Target hit','Stopped') or not r.get('done') or r.get('r') is None:continue
    if (S['session'],r['symbol']) in have:continue
    T.append(dict(day=S['session'],sym=r['symbol'],pts=r['pts'],res={'Target hit':'target','Stopped':'stop','Triggered':'eod'}[r['status']],r=r['r'],time=r['time'][11:16],entry=r['entry'],stop=r['stop'],target=r['target'],source='live'));n+=1
T.sort(key=lambda t:(t['day'],t['time']))
json.dump(T,open('track.json','w'),indent=1);print('added',n,'rows; total',len(T))
