import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# API CONFIGURATION
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}
NIFTY_LOT_SIZE = 75

@st.cache_data(ttl=60)
def fetch_nifty_price():
    url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()['data']['lastPrice']
    else:
        st.error(f"Failed to fetch Nifty price: {response.status_code}")
        return None

@st.cache_data(ttl=300)
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="24-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch option chain data: {response.status_code}")
        return None

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

def plot_heatmap(df, metric):
    heat_df = df.pivot_table(index='type', columns='strike', values=metric, fill_value=0)
    fig = px.imshow(heat_df, aspect="auto", color_continuous_scale='viridis',
                    labels={'x': 'Strike Price', 'y': 'Option Type', 'color': metric.upper()})
    st.plotly_chart(fig, use_container_width=True)

def plot_stacked_bar(df, metric):
    grouped = df.pivot_table(index='strike', columns='type', values=metric, fill_value=0).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grouped['strike'], y=grouped.get('CE', 0), name='CE'))
    fig.add_trace(go.Bar(x=grouped['strike'], y=grouped.get('PE', 0), name='PE'))
    fig.update_layout(barmode='stack', title=f"Stacked Bar: {metric.upper()} Comparison")
    st.plotly_chart(fig, use_container_width=True)

def plot_iv_smile(df):
    fig = px.line(df, x='strike', y='iv', color='type', title="IV Smile Curve")
    st.plotly_chart(fig, use_container_width=True)

def plot_change_oi(df):
    fig = px.bar(df, x='strike', y='change_oi', color='type', barmode='group', title="Change in Open Interest (ΔOI)")
    st.plotly_chart(fig, use_container_width=True)

def plot_delta_gamma(df):
    grouped = df.groupby('strike').agg({'delta': 'sum', 'gamma': 'sum'}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped['strike'], y=grouped['delta'], name="Delta", mode='lines+markers'))
    fig.add_trace(go.Scatter(x=grouped['strike'], y=grouped['gamma'], name="Gamma", mode='lines+markers'))
    fig.update_layout(title="Delta & Gamma Exposure Curve")
    st.plotly_chart(fig, use_container_width=True)

def plot_pcr(df):
    pivot = df.pivot_table(index='strike', columns='type', values='oi', fill_value=0)
    pivot['PCR'] = pivot['PE'] / pivot['CE'].replace(0, np.nan)
    fig = px.bar(pivot.reset_index(), x='strike', y='PCR', title="Put-Call Ratio (PCR)")
    st.plotly_chart(fig, use_container_width=True)

def plot_moneyness_distribution(df):
    counts = df.groupby(['type', 'moneyness']).size().reset_index(name='count')
    fig = px.pie(counts, names='moneyness', values='count', color='type', title="Moneyness Distribution")
    st.plotly_chart(fig, use_container_width=True)

def plot_price_action(spot_price, df):
    atm_strike = df.iloc[(df['strike'] - spot_price).abs().argmin()]['strike']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[spot_price, spot_price], mode='lines', name='Spot Price'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[atm_strike, atm_strike], mode='lines', name='ATM Strike'))
    fig.update_layout(title="Live Price Action with ATM")
    st.plotly_chart(fig, use_container_width=True)

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

    else:
        st.error("No option data available.")

if __name__ == "__main__":
    main()
