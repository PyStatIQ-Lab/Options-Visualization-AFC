import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# API CONFIGURATION
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}
NIFTY_LOT_SIZE = 75

# FETCH NIFTY SPOT PRICE USING YFINANCE
@st.cache_data(ttl=60)
def fetch_nifty_price():
    nifty = yf.Ticker("^NSEI")
    return nifty.history(period="1d")["Close"].iloc[-1]

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
                    'iv': analytics.get('iv', 0),
                    'moneyness': 'ATM' if abs(strike - spot_price) < 1 else 'ITM' if strike < spot_price else 'OTM'
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

# Plot Heatmap of OI or Volume across Strikes
def plot_heatmap(df, metric='volume'):
    # Pivot the DataFrame for plotting
    df_pivot = df.pivot_table(values=metric, index='type', columns='strike', aggfunc='sum', fill_value=0)
    
    # Create the heatmap using Plotly
    fig = px.imshow(df_pivot,
                    labels=dict(x="Strike Price", y="Option Type", color=metric.capitalize()),
                    color_continuous_scale="Viridis",
                    title=f"Heatmap of {metric.capitalize()} across Strikes")
    fig.update_xaxes(title="Strike Price", tickmode="linear")
    fig.update_yaxes(title="Option Type", tickvals=[0, 1], ticktext=["CE", "PE"])
    st.plotly_chart(fig, use_container_width=True)

# Generate stacked bar chart of OI/Volume comparison between CE and PE
def plot_stacked_bar(df, metric='volume'):
    df_grouped = df.groupby(['strike', 'type'])[metric].sum().unstack().fillna(0)
    fig = df_grouped.plot(kind='bar', stacked=True, figsize=(10, 6), color=["blue", "red"])
    st.pyplot(fig)

# IV Smile Curve
def plot_iv_smile(df):
    iv_data = df[['strike', 'type', 'iv']].pivot(index='strike', columns='type', values='iv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iv_data.index, y=iv_data['CE'], mode='lines', name='CE IV'))
    fig.add_trace(go.Scatter(x=iv_data.index, y=iv_data['PE'], mode='lines', name='PE IV'))
    fig.update_layout(title="Implied Volatility Smile", xaxis_title="Strike Price", yaxis_title="Implied Volatility")
    st.plotly_chart(fig, use_container_width=True)

# Change in OI (Δ OI) Bar or Waterfall Chart
def plot_change_oi(df):
    df['oi_change'] = df.groupby('type')['oi'].diff().fillna(0)
    df_grouped = df.groupby(['strike', 'type'])['oi_change'].sum().unstack().fillna(0)
    fig = df_grouped.plot(kind='bar', stacked=True, figsize=(10, 6), color=["green", "orange"])
    st.pyplot(fig)

# Plot Put-Call Ratio by Strike
def plot_pcr(df):
    df_grouped = df.groupby(['strike', 'type'])['oi'].sum().unstack().fillna(0)
    pcr = df_grouped['PE'] / df_grouped['CE']
    fig = go.Figure(go.Scatter(x=pcr.index, y=pcr, mode='lines', name="PCR"))
    fig.update_layout(title="Put-Call Ratio by Strike", xaxis_title="Strike Price", yaxis_title="PCR")
    st.plotly_chart(fig, use_container_width=True)

# Moneyness Distribution (Pie or Histogram)
def plot_moneyness_distribution(df):
    moneyness_dist = df['moneyness'].value_counts()
    fig = go.Figure(go.Pie(labels=moneyness_dist.index, values=moneyness_dist.values, hole=0.3))
    fig.update_layout(title="Moneyness Distribution (ITM/OTM/ATM)")
    st.plotly_chart(fig, use_container_width=True)

# Live Price Action Chart with ATM Highlighted
def plot_price_action(spot_price, df):
    fig = go.Figure()
    strikes = df['strike'].unique()
    fig.add_trace(go.Scatter(x=[spot_price] * 2, y=[min(strikes), max(strikes)], mode='lines', name="Spot Price", line=dict(color="blue", dash="dash")))
    fig.update_layout(title="Live Price Action with ATM Highlighted", xaxis_title="Strike Price", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# Analysis and Interpretation
def generate_analysis(df):
    analysis = """
    📊 **Analysis & Interpretation**:
    - The chart above shows the distribution of Open Interest (OI) and Volume for various strikes.
    - High OI and Volume at a particular strike may indicate heavy market activity or institutional interest.
    - The IV Smile curve indicates implied volatility for both Call and Put options. Higher IV at extreme strikes may suggest market expectation of higher volatility.
    """
    return analysis

# Display Analysis in the app
def display_analysis(analysis):
    st.write(analysis)

# Main Function
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
