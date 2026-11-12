import json,pickle,pandas as pd,warnings;warnings.filterwarnings('ignore')
from common import bars
U=json.load(open('universe.json'));syms=list(U['symbols'])
now=pd.Timestamp.now(tz='UTC');end=(now-pd.Timedelta(minutes=16)).strftime('%Y-%m-%dT%H:%M:%SZ')
D=bars(syms+['SPY'],'1Day',(now-pd.Timedelta(days=330)).strftime('%Y-%m-%d'),end=end)
M=bars(syms,'1Min',(now-pd.Timedelta(days=45)).strftime('%Y-%m-%d'),end=end)
pickle.dump((D,M),open('/tmp/sl_cache.pkl','wb'));print('cached',len(D),len(M))
