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
                    'theta': analytics.get('theta', 0),
                    'vega': analytics.get('vega', 0),
                    'change_oi': market.get('changeOi', 0),
                    'moneyness': 'ATM' if abs(strike - spot_price) < 1 else 'ITM' if (
                        (option_type == 'callOptionData' and strike < spot_price) or
                        (option_type == 'putOptionData' and strike > spot_price)) else 'OTM'
                })
    return pd.DataFrame(processed)

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

# Enhanced Heatmap of OI/Volume
def plot_heatmap(df, metric):
    heat_df = df.pivot_table(index='type', columns='strike', values=metric, fill_value=0)
    
    # Sort columns by strike price
    heat_df = heat_df.reindex(sorted(heat_df.columns), axis=1)
    
    fig = px.imshow(
        heat_df,
        aspect="auto",
        color_continuous_scale='viridis',
        labels={'x': 'Strike Price', 'y': 'Option Type', 'color': metric.upper()},
        title=f"Heatmap of {metric.upper()} Across Strikes"
    )
    
    # Add annotations
    fig.update_traces(
        texttemplate="%{z:.2s}",
        textfont={"size": 10},
        hovertemplate="<b>Strike</b>: %{x}<br><b>Type</b>: %{y}<br><b>" + metric.upper() + "</b>: %{z:,}"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced Stacked Bar Chart for OI/Volume
def plot_stacked_bar(df, metric):
    grouped = df.pivot_table(index='strike', columns='type', values=metric, fill_value=0).reset_index()
    grouped = grouped.sort_values('strike')
    
    fig = go.Figure()
    
    # Add CE and PE bars
    fig.add_trace(go.Bar(
        x=grouped['strike'],
        y=grouped.get('CE', 0),
        name='CE',
        hovertemplate="<b>Strike</b>: %{x}<br><b>CE " + metric.upper() + "</b>: %{y:,}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=grouped['strike'],
        y=grouped.get('PE', 0),
        name='PE',
        hovertemplate="<b>Strike</b>: %{x}<br><b>PE " + metric.upper() + "</b>: %{y:,}<extra></extra>"
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f"Stacked Bar: {metric.upper()} Comparison by Strike",
        xaxis_title="Strike Price",
        yaxis_title=metric.upper(),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced IV Smile Curve
def plot_iv_smile(df):
    # Sort by strike for smooth lines
    df = df.sort_values('strike')
    
    fig = px.line(
        df,
        x='strike',
        y='iv',
        color='type',
        title="IV Smile Curve",
        labels={'iv': 'Implied Volatility', 'strike': 'Strike Price'},
        line_shape='spline'
    )
    
    # Add markers and improve hover info
    fig.update_traces(
        mode='lines+markers',
        hovertemplate="<b>Strike</b>: %{x}<br><b>IV</b>: %{y:.2%}<extra></extra>"
    )
    
    # Format y-axis as percentage
    fig.update_layout(
        yaxis_tickformat=".1%",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced Change in OI (ΔOI) Chart
def plot_change_oi(df):
    # Sort by strike for better visualization
    df = df.sort_values('strike')
    
    fig = px.bar(
        df,
        x='strike',
        y='change_oi',
        color='type',
        barmode='group',
        title="Change in Open Interest (ΔOI)",
        labels={'change_oi': 'Change in OI', 'strike': 'Strike Price'},
        color_discrete_map={'CE': 'blue', 'PE': 'red'}
    )
    
    # Add reference line at 0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    
    # Improve hover info
    fig.update_traces(
        hovertemplate="<b>Strike</b>: %{x}<br><b>ΔOI</b>: %{y:,}<extra></extra>"
    )
    
    fig.update_layout(
        hovermode="x unified",
        yaxis_title="Change in OI"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced Delta / Gamma Exposure Curve
def plot_delta_gamma(df):
    grouped = df.groupby('strike').agg({'delta': 'sum', 'gamma': 'sum'}).reset_index()
    grouped = grouped.sort_values('strike')
    
    fig = go.Figure()
    
    # Add Delta trace
    fig.add_trace(go.Scatter(
        x=grouped['strike'],
        y=grouped['delta'],
        name="Delta Exposure",
        mode='lines+markers',
        line=dict(color='royalblue', width=2),
        hovertemplate="<b>Strike</b>: %{x}<br><b>Delta</b>: %{y:.2f}<extra></extra>"
    ))
    
    # Add Gamma trace
    fig.add_trace(go.Scatter(
        x=grouped['strike'],
        y=grouped['gamma'],
        name="Gamma Exposure",
        mode='lines+markers',
        line=dict(color='firebrick', width=2),
        hovertemplate="<b>Strike</b>: %{x}<br><b>Gamma</b>: %{y:.2f}<extra></extra>"
    ))
    
    # Add reference line at 0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title="Delta & Gamma Exposure Curve",
        xaxis_title="Strike Price",
        yaxis_title="Exposure",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced Put-Call Ratio (PCR) per Strike
def plot_pcr(df):
    pivot = df.pivot_table(index='strike', columns='type', values='oi', fill_value=0).reset_index()
    pivot = pivot.sort_values('strike')
    pivot['PCR'] = pivot['PE'] / pivot['CE'].replace(0, np.nan)
    
    fig = go.Figure()
    
    # Add PCR bars
    fig.add_trace(go.Bar(
        x=pivot['strike'],
        y=pivot['PCR'],
        name='PCR',
        marker_color='purple',
        hovertemplate="<b>Strike</b>: %{x}<br><b>PCR</b>: %{y:.2f}<extra></extra>"
    ))
    
    # Add reference line at 1
    fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="black")
    
    # Add annotations for extreme PCR values
    max_pcr = pivot['PCR'].max()
    min_pcr = pivot['PCR'].min()
    
    if max_pcr > 2:
        extreme_strike = pivot.loc[pivot['PCR'].idxmax(), 'strike']
        fig.add_annotation(
            x=extreme_strike,
            y=max_pcr,
            text="High PCR (Bearish)",
            showarrow=True,
            arrowhead=1
        )
    
    if min_pcr < 0.5:
        extreme_strike = pivot.loc[pivot['PCR'].idxmin(), 'strike']
        fig.add_annotation(
            x=extreme_strike,
            y=min_pcr,
            text="Low PCR (Bullish)",
            showarrow=True,
            arrowhead=1
        )
    
    fig.update_layout(
        title="Put-Call Ratio (PCR) by Strike",
        xaxis_title="Strike Price",
        yaxis_title="PCR (PE OI / CE OI)",
        hovermode="x"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced Moneyness Distribution
def plot_moneyness_distribution(df):
    counts = df.groupby(['type', 'moneyness']).size().reset_index(name='count')
    
    fig = px.sunburst(
        counts,
        path=['type', 'moneyness'],
        values='count',
        title="Moneyness Distribution",
        color='moneyness',
        color_discrete_map={
            'ITM': 'green',
            'OTM': 'red',
            'ATM': 'blue'
        }
    )
    
    fig.update_traces(
        textinfo="label+percent parent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fixed Live Price Action with ATM Highlighted
def plot_price_action(spot_price, df):
    if df.empty:
        return
    
    # Find ATM strike
    atm_strike = df.iloc[(df['strike'] - spot_price).abs().argmin()]['strike']
    
    # Get min and max strikes for the plot range
    min_strike = df['strike'].min()
    max_strike = df['strike'].max()
    
    fig = go.Figure()
    
    # Add spot price line
    fig.add_trace(go.Scatter(
        x=[min_strike, max_strike],
        y=[spot_price, spot_price],
        mode='lines',
        name=f'Spot Price ({spot_price:.2f})',
        line=dict(color='green', width=2, dash='dot')
    ))
    
    # Add ATM strike line
    fig.add_trace(go.Scatter(
        x=[min_strike, max_strike],
        y=[atm_strike, atm_strike],
        mode='lines',
        name=f'ATM Strike ({atm_strike:.2f})',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Add important strikes (highest OI for CE and PE)
    ce_df = df[df['type'] == 'CE']
    pe_df = df[df['type'] == 'PE']
    
    if not ce_df.empty:
        max_ce_oi = ce_df.loc[ce_df['oi'].idxmax()]
        fig.add_trace(go.Scatter(
            x=[min_strike, max_strike],
            y=[max_ce_oi['strike'], max_ce_oi['strike']],
            mode='lines',
            name=f'Max CE OI Strike ({max_ce_oi["strike"]:.2f})',
            line=dict(color='orange', width=1)
        ))
    
    if not pe_df.empty:
        max_pe_oi = pe_df.loc[pe_df['oi'].idxmax()]
        fig.add_trace(go.Scatter(
            x=[min_strike, max_strike],
            y=[max_pe_oi['strike'], max_pe_oi['strike']],
            mode='lines',
            name=f'Max PE OI Strike ({max_pe_oi["strike"]:.2f})',
            line=dict(color='purple', width=1)
        ))
    
    fig.update_layout(
        title="Live Price Action with Key Levels",
        xaxis_title="Strike Price Range",
        yaxis_title="Price",
        showlegend=True,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add Greek Exposure Heatmap
def plot_greek_heatmap(df, greek='delta'):
    heat_df = df.pivot_table(index='type', columns='strike', values=greek, fill_value=0)
    heat_df = heat_df.reindex(sorted(heat_df.columns), axis=1)
    
    fig = px.imshow(
        heat_df,
        aspect="auto",
        color_continuous_scale='RdBu',
        labels={'x': 'Strike Price', 'y': 'Option Type', 'color': greek.capitalize()},
        title=f"{greek.capitalize()} Exposure Across Strikes"
    )
    
    fig.update_traces(
        hovertemplate="<b>Strike</b>: %{x}<br><b>Type</b>: %{y}<br><b>" + greek.capitalize() + "</b>: %{z:.2f}<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Nifty Option Flow", layout="wide")
    st.title("📊 Real-Time Alluvial Flow of Most Active Nifty Options")

    with st.sidebar:
        st.header("Filters")
        expiry = st.text_input("Expiry Date (DD-MM-YYYY):", "24-04-2025")
        metric = st.selectbox("Select Metric", ["volume", "oi", "ltp", "change_oi"])
        top_n = st.slider("Top N Active Options", 5, 20, 10)
        greek = st.selectbox("Select Greek for Heatmap", ["delta", "gamma", "theta", "vega"])

    spot_price = fetch_nifty_price()

    if spot_price is None:
        st.error("Unable to retrieve the Nifty spot price. Please try again later.")
        return

    st.markdown(f"**Current Nifty Spot Price:** ₹{spot_price:,.2f}")

    raw_data = fetch_options_data(expiry=expiry)
    df = process_options_data(raw_data, spot_price)

    if df is not None and not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total CE Volume", f"{df[df['type'] == 'CE']['volume'].sum():,}")
        with col2:
            st.metric("Total PE Volume", f"{df[df['type'] == 'PE']['volume'].sum():,}")

        st.subheader("🔹 Option Chain Data (Top Active)")
        st.dataframe(df.sort_values(by=metric, ascending=False).head(top_n))

        st.subheader("🔹 Sankey: Top Active Options Flow")
        fig = build_sankey(df, metric=metric, top_n=top_n)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔹 Heatmap of OI/Volume Across Strikes")
        plot_heatmap(df, metric)

        st.subheader("🔹 Stacked Bar: CE vs PE OI/Volume Comparison")
        plot_stacked_bar(df, metric)

        st.subheader("🔹 IV Smile Curve")
        plot_iv_smile(df)

        st.subheader("🔹 Change in Open Interest (ΔOI)")
        plot_change_oi(df)

        st.subheader(f"🔹 {greek.capitalize()} Exposure Heatmap")
        plot_greek_heatmap(df, greek)

        st.subheader("🔹 Delta & Gamma Exposure Curve")
        plot_delta_gamma(df)

        st.subheader("🔹 Put-Call Ratio (PCR) by Strike")
        plot_pcr(df)

        st.subheader("🔹 Moneyness Distribution")
        plot_moneyness_distribution(df)

        st.subheader("🔹 Live Price Action with Key Levels")
        plot_price_action(spot_price, df)

    else:
        st.error("No option data available.")

if __name__ == "__main__":
    main()
