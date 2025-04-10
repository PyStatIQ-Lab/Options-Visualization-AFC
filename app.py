import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# API CONFIGURATION
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}
NIFTY_LOT_SIZE = 75

# FETCH NIFTY SPOT PRICE
@st.cache_data(ttl=60)
def fetch_nifty_price():
    url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()['data']['lastPrice']
    else:
        st.error(f"Failed to fetch Nifty price: {response.status_code}")
        return None

# FETCH OPTIONS DATA
@st.cache_data(ttl=300)
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="24-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch option chain data: {response.status_code}")
        return None

# PROCESS API RESPONSE
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
                    'iv': analytics.get('iv', 0)
                })
    return pd.DataFrame(processed)

# BUILD SANKEY NODES AND LINKS FOR ALLUVIAL FLOW
def build_sankey(df, metric='volume', top_n=10):
    df = df.sort_values(by=metric, ascending=False).head(top_n)
    
    labels = []
    sources = []
    targets = []
    values = []
    
    for i, row in df.iterrows():
        opt_type = row['type']
        strike_label = f"{row['strike']:.0f}"
        label = f"{strike_label}-{opt_type}"
        
        if strike_label not in labels:
            labels.append(strike_label)
        if opt_type not in labels:
            labels.append(opt_type)
        if label not in labels:
            labels.append(label)
        
        source = labels.index(strike_label)
        target = labels.index(label)
        values.append(row[metric])
        sources.append(source)
        targets.append(target)
        
        # Connect to CE/PE from label
        sources.append(target)
        targets.append(labels.index(opt_type))
        values.append(row[metric])

    return go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

# STREAMLIT UI
def main():
    st.set_page_config(page_title="Nifty Option Flow", layout="wide")
    st.title("📊 Real-Time Alluvial Flow of Most Active Nifty Options")
    
    expiry = st.text_input("Expiry Date (DD-MM-YYYY):", "24-04-2025")
    metric = st.selectbox("Select Metric", ["volume", "oi", "ltp"])
    top_n = st.slider("Top N Active Options", 5, 20, 10)

    spot_price = fetch_nifty_price()
    st.markdown(f"**Spot Price:** {spot_price}")

    raw_data = fetch_options_data(expiry=expiry)
    df = process_options_data(raw_data, spot_price)

    if df is not None and not df.empty:
        st.dataframe(df.sort_values(by=metric, ascending=False).head(top_n))
        fig = build_sankey(df, metric=metric, top_n=top_n)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No option data available.")

if __name__ == "__main__":
    main()
