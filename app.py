import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import streamlit as st
import yfinance as yf
import requests

st.set_page_config(page_title="Demand & Supply Zone Scanner", page_icon="📈", layout="wide")
st.title("📈 Demand & Supply Zone Scanner")
st.subheader("Fresh Zone Radar")
st.caption("Candidate zones only. Reachability rank is a heuristic, not a probability or trading advice.")

@st.cache_data(ttl=86400, show_spinner=False)
def nifty500_symbols():
    url = "https://en.wikipedia.org/wiki/NIFTY_500"
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    for t in pd.read_html(io.StringIO(r.text)):
        col = next((c for c in t.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
        if col is not None:
            vals = t[col].dropna().astype(str).str.strip().tolist()
            vals = [x for x in vals if x and x.upper() not in ("SYMBOL", "TICKER")]
            if len(vals) >= 400:
                return sorted(set(vals))
    raise ValueError("Could not load the Nifty 500 list. Try again later.")

def clean(raw):
    if raw is None or raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    raw = raw.loc[:, ~pd.Index(raw.columns).duplicated()]
    cols = ["open", "high", "low", "close"]
    if any(c not in raw.columns for c in cols): return pd.DataFrame()
    raw = raw[cols].dropna().sort_index()
    return raw[~raw.index.duplicated(keep="last")]

@st.cache_data(ttl=900, show_spinner=False)
def stock_data(symbol, tf):
    ticker = symbol + ".NS"
    if tf == "Daily":
        return ticker, clean(yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False, threads=False))
    raw = clean(yf.download(ticker, period="60d", interval="5m", progress=False, auto_adjust=False, threads=False))
    if raw.empty: return ticker, raw
    pieces = []
    for _, session in raw.groupby(raw.index.date):
        session = session.between_time("09:15", "15:30")
        if session.empty: continue
        bars = session.resample("125min", origin="start_day", offset="9h15min", label="left", closed="left").agg(
            {"open":"first","high":"max","low":"min","close":"last"}).dropna()
        pieces.append(bars)
    return ticker, pd.concat(pieces).sort_index() if pieces else pd.DataFrame()

def detect_zones(df):
    zones=[]
    if len(df)<3: return zones
    bodies=(df["close"]-df["open"]).abs()
    for i in range(1,len(df)-1):
        a,b,c=float(bodies.iloc[i-1]),float(bodies.iloc[i]),float(bodies.iloc[i+1])
        if min(a,b,c)<=0 or not (a<b<c): continue
        base, out=df.iloc[i],df.iloc[i+1]
        demand=float(out.close)>float(base.high) and float(out.close)>float(out.open)
        supply=float(out.close)<float(base.low) and float(out.close)<float(out.open)
        if demand or supply:
            zones.append({"Zone":"DEMAND" if demand else "SUPPLY","Formed At":df.index[i],
                          "Low":float(base.low),"High":float(base.high)})
    return zones

def radar_rows(df,zones,symbol,tf):
    if df.empty: return []
    current=float(df.close.iloc[-1])
    avg_range=(df.high-df.low).rolling(14).mean().iloc[-1]
    if pd.isna(avg_range) or avg_range<=0: avg_range=float((df.high-df.low).tail(20).mean())
    if pd.isna(avg_range) or avg_range<=0: return []
    rows=[]
    for z in zones:
        pos=df.index.get_loc(z["Formed At"])
        future=df.iloc[pos+2:]  # ignore base and leg-out candles
        if future.empty: continue
        if z["Zone"]=="DEMAND":
            touched=((future.low<=z["High"])&(future.high>=z["Low"])).any()
            broken=(future.close<z["Low"]).any()
            if touched or broken or current<=z["High"]: continue
            distance=current-z["High"]; direction="Price would need to fall"
        else:
            touched=((future.high>=z["Low"])&(future.low<=z["High"])).any()
            broken=(future.close>z["High"]).any()
            if touched or broken or current>=z["Low"]: continue
            distance=z["Low"]-current; direction="Price would need to rise"
        pct=distance/current*100 if current else 999
        if pct>(8 if tf=="Daily" else 4): continue
        units=distance/avg_range
        rank="HIGHER relative proximity" if units<=1 else ("MODERATE relative proximity" if units<=2 else "LOWER relative proximity")
        score=round(max(0,100-(units/2)*100),1)
        rows.append({"Symbol":symbol,"Timeframe":tf,"Zone":z["Zone"],"Formed At":str(z["Formed At"]),
                     "Zone Low":round(z["Low"],2),"Zone High":round(z["High"],2),"Current Price":round(current,2),
                     "Distance %":round(pct,2),"Distance / Avg Range":round(units,2),"Direction":direction,
                     "1–2 Day Reachability Rank":rank,"Rank Score (heuristic)":score})
    return rows

with st.sidebar:
    st.header("Radar settings")
    timeframes=st.multiselect("Timeframes",["Daily","125m"],default=["Daily","125m"])
    max_stocks=st.select_slider("Stocks to scan",options=[25,50,100,200,500],value=50,
        help="Start with 50. Scanning 500 symbols on both timeframes can be slow or rate-limited.")
    run=st.button("Scan Fresh Zones",type="primary",use_container_width=True)
    st.caption("Daily asks Yahoo Finance for maximum available history. Intraday history is limited, typically around 60 days for 5-minute data.")

if run:
    if not timeframes:
        st.error("Select at least one timeframe."); st.stop()
    try:
        with st.spinner("Loading Nifty 500 list..."): symbols=nifty500_symbols()
        symbols=symbols[:max_stocks]
        rows=[]; total=len(symbols)*len(timeframes); done=0
        progress=st.progress(0,text="Starting scan...")
        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs={pool.submit(stock_data,s,tf):(s,tf) for s in symbols for tf in timeframes}
            for fut in as_completed(jobs):
                s,tf=jobs[fut]; done+=1
                progress.progress(done/total,text=f"Checked {done} of {total} stock/timeframe requests...")
                try:
                    ticker,df=fut.result()
                    if df.empty: continue
                    rows.extend(radar_rows(df,detect_zones(df),s,tf))
                except Exception:
                    continue
        progress.empty()
        result=pd.DataFrame(rows)
        if not result.empty:
            result=result.sort_values(["Rank Score (heuristic)","Distance %"],ascending=[False,True]).reset_index(drop=True)
        st.session_state["radar_result"]=result
        st.session_state["radar_info"]={"count":len(symbols),"tfs":timeframes,"time":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        st.error(f"Could not start scan: {e}"); st.stop()

if "radar_result" not in st.session_state:
    st.info("Choose the timeframes and number of stocks in the sidebar, then tap Scan Fresh Zones.")
    st.markdown("Rule used: Leg In body < Base body < Leg Out body; base candle High–Low defines the zone.")
    st.stop()

res=st.session_state["radar_result"]; info=st.session_state["radar_info"]
st.caption(f"Last scan: {info['time']} · Symbols requested: {info['count']} · Timeframes: {', '.join(info['tfs'])}")
if res.empty:
    st.warning("No fresh zones matched the filters in this batch. Scan more stocks or try again later."); st.stop()

a,b,c=st.columns(3)
a.metric("Fresh candidates",len(res)); b.metric("Demand",int((res.Zone=="DEMAND").sum())); c.metric("Supply",int((res.Zone=="SUPPLY").sum()))
st.markdown("### Filter candidates")
f1,f2,f3=st.columns(3)
with f1: types=st.multiselect("Zone type",["DEMAND","SUPPLY"],default=["DEMAND","SUPPLY"])
with f2: tfs=st.multiselect("Timeframe",sorted(res.Timeframe.unique()),default=sorted(res.Timeframe.unique()))
with f3: ranks=st.multiselect("Proximity rank",["HIGHER relative proximity","MODERATE relative proximity","LOWER relative proximity"],default=["HIGHER relative proximity","MODERATE relative proximity","LOWER relative proximity"])
view=res[res.Zone.isin(types)&res.Timeframe.isin(tfs)&res["1–2 Day Reachability Rank"].isin(ranks)]
st.dataframe(view,use_container_width=True,hide_index=True)
st.download_button("Download candidates CSV",view.to_csv(index=False).encode("utf-8"),"fresh_zone_radar.csv","text/csv",use_container_width=True)
st.warning("Reachability rank is a rough distance ÷ average candle range heuristic, not a probability or guaranteed 1–2 day forecast. Verify every zone on a chart before trading.")
