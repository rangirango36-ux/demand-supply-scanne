import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Demand & Supply Zone Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Demand & Supply Zone Scanner")
st.caption("Rule-based candidate zones • Educational use only, not trading advice")

st.info(
    "Pattern rule: Leg In body < Base body < Leg Out body. "
    "Zones use the base candle's full High–Low range. "
    "This is a custom heuristic, not a verified GTF or trading system."
)

@st.cache_data(ttl=900, show_spinner=False)
def download_history(symbol: str, timeframe: str):
    ticker = symbol if "." in symbol else symbol + ".NS"
    settings = {
        "Monthly": ("1mo", "max"),
        "Weekly": ("1wk", "max"),
        "Daily": ("1d", "max"),
        "15m": ("15m", "60d"),
        "125m": ("5m", "60d"),
    }
    interval, period = settings[timeframe]
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False
    )
    if df.empty:
        raise ValueError(
            f"No data returned for {ticker}. Check the symbol or try another timeframe."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")
    df = df[required].dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if timeframe in ("15m", "125m"):
        if timeframe == "125m":
            pieces = []
            for _, session in df.groupby(df.index.date):
                session = session.between_time("09:15", "15:30")
                if session.empty:
                    continue
                candles = session.resample(
                    "125min", origin="start_day", offset="9h15min",
                    label="left", closed="left"
                ).agg({
                    "open": "first", "high": "max",
                    "low": "min", "close": "last"
                }).dropna()
                pieces.append(candles)
            if not pieces:
                raise ValueError("Could not create 125-minute candles from the downloaded data.")
            df = pd.concat(pieces).sort_index()
        else:
            df = df.between_time("09:15", "15:30")
    return ticker, df

def find_zones(df: pd.DataFrame):
    zones = []
    if len(df) < 3:
        return zones
    body = (df["close"] - df["open"]).abs()

    for i in range(1, len(df) - 1):
        leg_in = df.iloc[i - 1]
        base = df.iloc[i]
        leg_out = df.iloc[i + 1]

        li_body = float(body.iloc[i - 1])
        base_body = float(body.iloc[i])
        lo_body = float(body.iloc[i + 1])

        if min(li_body, base_body, lo_body) <= 0:
            continue
        if not (li_body < base_body < lo_body):
            continue

        demand = (
            float(leg_out["close"]) > float(base["high"])
            and float(leg_out["close"]) > float(leg_out["open"])
        )
        supply = (
            float(leg_out["close"]) < float(base["low"])
            and float(leg_out["close"]) < float(leg_out["open"])
        )
        if not (demand or supply):
            continue

        zones.append({
            "Type": "DEMAND" if demand else "SUPPLY",
            "Formed At": df.index[i],
            "Low": float(base["low"]),
            "High": float(base["high"]),
            "Leg In Body": li_body,
            "Base Body": base_body,
            "Leg Out Body": lo_body,
            "Touches": 0,
            "First Touch": pd.NaT,
            "Last Touch": pd.NaT,
            "Broken At": pd.NaT,
            "Status": "FRESH",
        })

    for zone in zones:
        future = df.loc[df.index > df.index[df.index.get_loc(zone["Formed At"]) + 1]] if False else df.loc[df.index > zone["Formed At"]]
        # Ignore the base candle and the departure candle when assessing later touches.
        formed_pos = df.index.get_loc(zone["Formed At"])
        future = df.iloc[formed_pos + 2:]
        touch_count = 0

        for timestamp, candle in future.iterrows():
            low, high, close = float(candle["low"]), float(candle["high"]), float(candle["close"])
            if zone["Type"] == "DEMAND":
                broken = close < zone["Low"]
            else:
                broken = close > zone["High"]

            if broken:
                zone["Status"] = "BROKEN"
                zone["Broken At"] = timestamp
                break

            overlaps = low <= zone["High"] and high >= zone["Low"]
            if overlaps:
                touch_count += 1
                if pd.isna(zone["First Touch"]):
                    zone["First Touch"] = timestamp
                zone["Last Touch"] = timestamp

        zone["Touches"] = touch_count
        if zone["Status"] != "BROKEN":
            if touch_count == 0:
                zone["Status"] = "FRESH"
            elif touch_count == 1:
                zone["Status"] = "TOUCHED"
            else:
                zone["Status"] = "USED"
    return zones

with st.sidebar:
    st.header("Scan settings")
    symbol = st.text_input("NSE symbol", value="RELIANCE", help="Examples: RELIANCE, TCS, STAR, SBIN").strip().upper()
    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly", "15m", "125m"], index=0)
    run_scan = st.button("Scan zones", type="primary", use_container_width=True)
    st.caption("Daily/Weekly/Monthly request maximum available history. Intraday history is limited by Yahoo Finance.")

if run_scan:
    if not symbol:
        st.error("Please enter a stock symbol.")
        st.stop()

    try:
        with st.spinner(f"Downloading data and scanning {symbol}..."):
            ticker, df = download_history(symbol, timeframe)
            zones = find_zones(df)
        st.session_state["scan_result"] = {
            "ticker": ticker,
            "timeframe": timeframe,
            "df": df,
            "zones": zones,
        }
    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        st.stop()

if "scan_result" not in st.session_state:
    st.write("Enter a stock symbol in the sidebar and tap **Scan zones**.")
    st.markdown("### What you will get")
    st.markdown("- All detected candidate Demand and Supply zones\n- Fresh, Touched, Used and Broken counts\n- Filterable zone table\n- CSV export")
    st.stop()

data = st.session_state["scan_result"]
df = data["df"]
zones = data["zones"]
result = pd.DataFrame(zones)

st.subheader(f"{data['ticker']} · {data['timeframe']}")
st.caption(f"Data range: {df.index.min()} to {df.index.max()} · Candles analyzed: {len(df)}")

if result.empty:
    st.warning("No zones matched the selected rules in the downloaded data.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("All candidate zones", len(result))
c2.metric("Demand zones", int((result["Type"] == "DEMAND").sum()))
c3.metric("Supply zones", int((result["Type"] == "SUPPLY").sum()))

st.markdown("### Zone status summary")
summary_rows = []
for ztype in ["DEMAND", "SUPPLY"]:
    subset = result[result["Type"] == ztype]
    row = {"Type": ztype, "Total": len(subset)}
    for status in ["FRESH", "TOUCHED", "USED", "BROKEN"]:
        row[status.title()] = int((subset["Status"] == status).sum())
    summary_rows.append(row)
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.markdown("### All detected zones")
filters = st.columns(3)
with filters[0]:
    type_filter = st.multiselect("Zone type", ["DEMAND", "SUPPLY"], default=["DEMAND", "SUPPLY"])
with filters[1]:
    status_filter = st.multiselect("Status", ["FRESH", "TOUCHED", "USED", "BROKEN"], default=["FRESH", "TOUCHED", "USED", "BROKEN"])
with filters[2]:
    search_text = st.text_input("Filter date (optional)", placeholder="e.g. 2024-")

view = result[result["Type"].isin(type_filter) & result["Status"].isin(status_filter)].copy()
view["Formed At"] = pd.to_datetime(view["Formed At"]).astype(str)
for col in ["First Touch", "Last Touch", "Broken At"]:
    view[col] = pd.to_datetime(view[col]).astype(str).replace("NaT", "")
if search_text:
    view = view[view["Formed At"].str.contains(search_text, case=False, na=False)]

st.dataframe(view, use_container_width=True, hide_index=True)
csv_bytes = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download all zones as CSV",
    data=csv_bytes,
    file_name=f"{data['ticker'].replace('.', '_')}_{data['timeframe']}_all_zones.csv",
    mime="text/csv",
    use_container_width=True,
)

st.warning(
    "This is a rule-based candidate scanner, not a verified trading system. "
    "Status labels follow the code's rules and should be checked against charts before any trading decision."
)
