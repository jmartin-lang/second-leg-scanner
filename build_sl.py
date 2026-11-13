import json,html as H,datetime as dt
J=json.load(open('sl.json'));R=J['rows']
def f2(x):return '' if x is None else f'{x:.2f}'
def ft(s):
    d=dt.datetime.strptime(s[:16],'%Y-%m-%d %H:%M');h=d.hour%12 or 12
    return f"{h}:{d.minute:02d} {'PM' if d.hour>=12 else 'AM'}"
def spark(r):
    s=r['spark'];n=len(s)
    if n<2:return ''
    lo=min(min(s),*r['lv']);hi=max(max(s),*r['lv']);rg=(hi-lo) or 1;w,h,p=120,34,2
    X=lambda i:p+i/(n-1)*(w-2*p);Y=lambda v:h-p-(v-lo)/rg*(h-2*p)
    pts=' '.join(f'{X(i):.1f},{Y(v):.1f}' for i,v in enumerate(s))
    return (f'<svg viewBox="0 0 120 34" width="120" height="34" role="img" aria-label="session price with stop and target"><line x1="0" x2="120" y1="{Y(r["lv"][0]):.1f}" y2="{Y(r["lv"][0]):.1f}" stroke="var(--bad)" stroke-dasharray="3 2" opacity=".7"/>'
            f'<line x1="0" x2="120" y1="{Y(r["lv"][1]):.1f}" y2="{Y(r["lv"][1]):.1f}" stroke="var(--ok)" stroke-dasharray="3 2" opacity=".7"/><polyline points="{pts}" fill="none" stroke="var(--fg)" stroke-width="1.2"/><circle cx="{X(n-1):.1f}" cy="{Y(s[-1]):.1f}" r="2" fill="var(--fg)"/></svg>')
def row(r):
    s=H.escape(r['symbol']);tv=f'<a class="tv" href="https://www.tradingview.com/chart/?symbol={s}&amp;interval=1" target="_blank" rel="noopener" title="Open {s} on TradingView, 1 minute chart">{s}</a>'
    fl='; '.join(r['flags']) or ''
    pt=r['pts'];tip=H.escape('Meets: '+('; '.join(r['met']) or 'none')+'. Misses: '+('; '.join(r['miss']) or 'none')+'.',quote=True)
    return (f'<tr data-st="{r["status"]}" data-skip="{int(r["skip"])}" data-sym="{s}" data-pts="{pt}" data-time="{r["time"]}"><td class="sym">{tv}</td><td class="st {r["status"].replace(" ","")}">{r["status"]}</td><td class="pts p{pt}" title="{tip}">{pt}/6</td><td class="ch">{spark(r)}</td>'
            f'<td class="n">{r["atr_pct"]:.1f}%</td><td class="n">{r["leg_pct"]:.1f}%</td><td class="n">{r["retrace"]}%</td><td class="n hn">{r["band_bars"]}</td><td class="n hn">{r["rsi"]:.0f}</td>'
            f'<td class="n">{f2(r["entry"]) or "–"}</td><td class="n">{f2(r["stop"])}</td><td class="n">{f2(r["target"])}</td><td class="n hn">{r["rr"] if r["rr"] else "–"}</td><td class="n hn">{f2(r["last"])}</td>'
            f'<td class="t" data-t="{r["time"]}">{ft(r["time"])}</td><td class="fl">{H.escape(fl) if fl else "clean"}</td></tr>')
body=''.join(row(r) for r in R)
t=open('sl_template.html').read()
t=t.replace('__ROWS__',body).replace('__ASOF__',J['asof']).replace('__SESSION__',J['session']).replace('__NPASS__',str(J['n_pass']))
open('public/index.html','w').write(t);open('public/data.json','w').write(json.dumps(J,separators=(',',':')))
print(len(R),'rows written')
