import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np

# API CONFIGURATION
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}
NIFTY_LOT_SIZE = 75

# Fetch Nifty Spot Price using YFinance
@st.cache_data(ttl=60)
def fetch_nifty_price():
    try:
        nifty = yf.Ticker("^NSEI")
        spot_price = nifty.history(period="1d")['Close'][0]
        return spot_price
    except Exception as e:
        st.error(f"Error fetching Nifty spot price: {e}")
        return None

# Fetch Options Data from Upstox API
@st.cache_data(ttl=300)
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="24-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch option chain data: {response.status_code}")
        return None

# Process the Options Data
def process_options_data(raw_data, spot_price):
    if not raw_data or 'data' not in raw_data:
        return None
    strike_map = raw_data['data']['strategyChainData']['strikeMap']
    processed = []
    for strike, data in strike_map.items():
        strike = float(strike)
        for option_type in ['callOptionData', 'putOptionData']:
            o_data = data.get(option_type, {})
            market = o_data.get('marketData', {})
            analytics = o_data.get('analytics', {})
            if market:
                processed.append({
                    'type': 'CE' if option_type == 'callOptionData' else 'PE',
                    'strike': strike,
                    'oi': market.get('oi', 0),
                    'volume': market.get('volume', 0),
                    'ltp': market.get('ltp', 0),
                    'iv': analytics.get('iv', 0),
                    'delta': analytics.get('delta', 0),
                    'gamma': analytics.get('gamma', 0),
                    'change_oi': market.get('changeOi', 0),
                    'moneyness': 'ATM' if abs(strike - spot_price) < 1 else 'ITM' if (
                        (option_type == 'callOptionData' and strike < spot_price) or
                        (option_type == 'putOptionData' and strike > spot_price)) else 'OTM'
                })
    return pd.DataFrame(processed)

# Analysis based on OI, Volume, IV, PCR, etc.
def generate_analysis(df):
    analysis = {}

    # OI Analysis
    max_oi = df['oi'].max()
    max_oi_strike = df.loc[df['oi'] == max_oi, 'strike'].values[0]
    analysis['OI Analysis'] = f"Maximum OI is at strike {max_oi_strike}, which indicates strong support or resistance at this level."

    # Volume Analysis
    max_volume = df['volume'].max()
    max_volume_strike = df.loc[df['volume'] == max_volume, 'strike'].values[0]
    analysis['Volume Analysis'] = f"Maximum Volume is at strike {max_volume_strike}, indicating strong market participation."

    # IV Analysis
    max_iv = df['iv'].max()
    max_iv_strike = df.loc[df['iv'] == max_iv, 'strike'].values[0]
    analysis['IV Analysis'] = f"Maximum IV is at strike {max_iv_strike}, which suggests higher market expectations of volatility at this strike."

    # Put-Call Ratio (PCR)
    pcr = df.groupby('strike').apply(lambda x: (x[x['type'] == 'PE']['oi'].sum()) / (x[x['type'] == 'CE']['oi'].sum()))
    pcr_max = pcr.max()
    pcr_max_strike = pcr.idxmax()
    if pcr_max > 1:
        analysis['PCR Analysis'] = f"Put-Call Ratio (PCR) is highest at strike {pcr_max_strike} with a value of {pcr_max}. A PCR > 1 suggests a bearish sentiment."
    else:
        analysis['PCR Analysis'] = f"Put-Call Ratio (PCR) is lowest at strike {pcr_max_strike} with a value of {pcr_max}. A PCR < 1 suggests a bullish sentiment."

    # Delta-Gamma Exposure
    delta_gamma_analysis = df[['strike', 'delta', 'gamma']].groupby('strike').agg({'delta': 'sum', 'gamma': 'sum'}).reset_index()
    analysis['Delta-Gamma Exposure'] = "Delta and Gamma values show potential hedging pressures. High Delta indicates a more directional move."

    # Change in OI (ΔOI)
    change_oi_analysis = df.groupby('strike')['change_oi'].sum().sort_values(ascending=False).head(1)
    change_oi_strike = change_oi_analysis.index[0]
    change_oi_value = change_oi_analysis.values[0]
    if change_oi_value > 0:
        analysis['ΔOI Analysis'] = f"Change in OI is highest at strike {change_oi_strike} with a value of {change_oi_value}. Positive ΔOI in CE indicates a bullish sentiment."
    else:
        analysis['ΔOI Analysis'] = f"Change in OI is highest at strike {change_oi_strike} with a value of {change_oi_value}. Positive ΔOI in PE indicates a bearish sentiment."

    return analysis

# Display Analysis
def display_analysis(analysis):
    for key, value in analysis.items():
        st.subheader(f"🔹 {key}")
        st.write(value)

# Build Sankey Diagram
def build_sankey(df, metric='volume', top_n=10):
    df = df.sort_values(by=metric, ascending=False).head(top_n)

    labels = []
    sources, targets, values = [], [], []

    for _, row in df.iterrows():
        opt_type = row['type']
        strike_label = f"{row['strike']:.0f}"
        label = f"{strike_label}-{opt_type}"

        for val in [strike_label, opt_type, label]:
            if val not in labels:
                labels.append(val)

        source = labels.index(strike_label)
        mid = labels.index(label)
        target = labels.index(opt_type)

        sources += [source, mid]
        targets += [mid, target]
        values += [row[metric], row[metric]]

    return go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=15, thickness=20, label=labels, line=dict(color="black", width=0.5)),
        link=dict(source=sources, target=targets, value=values)
    ))

# Streamlit UI
def main():
    st.set_page_config(page_title="Nifty Option Flow", layout="wide")
    st.title("📊 Real-Time Alluvial Flow of Most Active Nifty Options")

    expiry = st.text_input("Expiry Date (DD-MM-YYYY):", "24-04-2025")
    metric = st.selectbox("Select Metric", ["volume", "oi", "ltp"])
    top_n = st.slider("Top N Active Options", 5, 20, 10)

    spot_price = fetch_nifty_price()

    if spot_price is None:
        st.error("Unable to retrieve the Nifty spot price. Please try again later.")
        return

    st.markdown(f"**Spot Price:** {spot_price}")

    raw_data = fetch_options_data(expiry=expiry)
    df = process_options_data(raw_data, spot_price)

    if df is not None and not df.empty:
        st.subheader("🔹 Option Chain Data")
        st.dataframe(df.sort_values(by=metric, ascending=False).head(top_n))

        st.subheader("🔹 Sankey: Top Active Options Flow")
        fig = build_sankey(df, metric=metric, top_n=top_n)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔹 Heatmap of Volume / OI")
        plot_heatmap(df, metric)

        st.subheader("🔹 Stacked Bar: CE vs PE Volume / OI")
        plot_stacked_bar(df, metric)

        st.subheader("🔹 IV Smile Curve")
        plot_iv_smile(df)

        st.subheader("🔹 Change in OI")
        plot_change_oi(df)

        st.subheader("🔹 Delta / Gamma Exposure")
        plot_delta_gamma(df)

        st.subheader("🔹 Put-Call Ratio by Strike")
        plot_pcr(df)

        st.subheader("🔹 Moneyness Distribution")
        plot_moneyness_distribution(df)

        st.subheader("🔹 Live Price Action with ATM")
        plot_price_action(spot_price, df)

        # Display Analysis & Interpretation
        analysis = generate_analysis(df)
        display_analysis(analysis)

    else:
        st.error("No option data available.")

if __name__ == "__main__":
    main()
