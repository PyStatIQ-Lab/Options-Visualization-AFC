import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.ndimage import gaussian_filter1d

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
                    'iv': analytics.get('iv', 0),
                    'change_oi': market.get('changeOi', 0),
                    'delta': analytics.get('delta', 0),
                    'gamma': analytics.get('gamma', 0),
                    'moneyness': 'ITM' if (option_type == 'callOptionData' and strike < spot_price) or 
                                         (option_type == 'putOptionData' and strike > spot_price) 
                                    else 'OTM' if (option_type == 'callOptionData' and strike > spot_price) or 
                                                  (option_type == 'putOptionData' and strike < spot_price)
                                             else 'ATM'
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

# HEATMAP OF OI/VOLUME ACROSS STRIKES
def build_heatmap(df, metric='volume'):
    pivot_df = df.pivot_table(index='strike', columns='type', values=metric, aggfunc='sum').fillna(0)
    pivot_df = pivot_df.sort_index()
    
    fig = go.Figure(go.Heatmap(
        x=pivot_df.index,
        y=pivot_df.columns,
        z=pivot_df.values.T,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title=metric.upper())
    ))
    
    fig.update_layout(
        title=f'Option {metric.upper()} Heatmap',
        xaxis_title='Strike Price',
        yaxis_title='Option Type',
        height=600
    )
    
    return fig

# STACKED BAR CHART OF OI/VOLUME COMPARISON
def build_stacked_bar(df, metric='oi'):
    pivot_df = df.pivot_table(index='strike', columns='type', values=metric, aggfunc='sum').fillna(0)
    pivot_df = pivot_df.sort_index()
    
    fig = go.Figure()
    
    for opt_type in ['CE', 'PE']:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[opt_type],
            name=opt_type,
            hoverinfo='y+name'
        ))
    
    fig.update_layout(
        barmode='stack',
        title=f'Option {metric.upper()} by Strike',
        xaxis_title='Strike Price',
        yaxis_title=metric.upper(),
        height=500
    )
    
    return fig

# IV SMILE CURVE
def build_iv_smile(df, spot_price):
    ce_df = df[df['type'] == 'CE'].sort_values('strike')
    pe_df = df[df['type'] == 'PE'].sort_values('strike')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ce_df['strike'],
        y=ce_df['iv'],
        mode='lines+markers',
        name='CE IV',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=pe_df['strike'],
        y=pe_df['iv'],
        mode='lines+markers',
        name='PE IV',
        line=dict(color='red')
    ))
    
    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text="Spot Price")
    
    fig.update_layout(
        title='Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        height=500
    )
    
    return fig

# CHANGE IN OI WATERFALL CHART
def build_change_oi_chart(df, spot_price):
    df = df.sort_values('strike')
    df['abs_change_oi'] = df['change_oi'].abs()
    
    fig = go.Figure()
    
    for opt_type in ['CE', 'PE']:
        type_df = df[df['type'] == opt_type]
        fig.add_trace(go.Bar(
            x=type_df['strike'],
            y=type_df['change_oi'],
            name=f'{opt_type} ΔOI',
            marker_color='green' if opt_type == 'CE' else 'red',
            hoverinfo='y+name'
        ))
    
    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text="Spot Price")
    
    fig.update_layout(
        barmode='group',
        title='Change in Open Interest by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Change in OI',
        height=500
    )
    
    return fig

# DELTA SKEW / GAMMA EXPOSURE CURVE
def build_gamma_exposure(df, spot_price):
    df = df.sort_values('strike')
    
    # Calculate gamma exposure (simplified)
    df['gamma_exposure'] = df['gamma'] * df['oi'] * NIFTY_LOT_SIZE * 100
    
    fig = go.Figure()
    
    for opt_type in ['CE', 'PE']:
        type_df = df[df['type'] == opt_type]
        fig.add_trace(go.Scatter(
            x=type_df['strike'],
            y=type_df['gamma_exposure'],
            mode='lines',
            name=f'{opt_type} Gamma Exposure',
            line=dict(color='green' if opt_type == 'CE' else 'red')
        ))
    
    # Add smoothed total gamma exposure
    total_gamma = df.groupby('strike')['gamma_exposure'].sum().reset_index()
    total_gamma['smoothed'] = gaussian_filter1d(total_gamma['gamma_exposure'], sigma=1)
    
    fig.add_trace(go.Scatter(
        x=total_gamma['strike'],
        y=total_gamma['smoothed'],
        mode='lines',
        name='Total Gamma Exposure',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text="Spot Price")
    
    fig.update_layout(
        title='Gamma Exposure by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Gamma Exposure',
        height=500
    )
    
    return fig

# PUT-CALL RATIO PER STRIKE
def build_pcr_chart(df, spot_price):
    pivot_df = df.pivot_table(index='strike', columns='type', values='oi', aggfunc='sum').fillna(0)
    pivot_df['PCR'] = pivot_df['PE'] / pivot_df['CE']
    pivot_df = pivot_df.sort_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=pivot_df.index,
        y=pivot_df['PCR'],
        name='PCR',
        marker_color='purple'
    ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="red")
    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text="Spot Price")
    
    fig.update_layout(
        title='Put-Call Ratio by Strike',
        xaxis_title='Strike Price',
        yaxis_title='PCR (PE OI / CE OI)',
        height=500
    )
    
    return fig

# MONEYNESS DISTRIBUTION
def build_moneyness_chart(df):
    moneyness_df = df.groupby(['type', 'moneyness']).size().reset_index(name='count')
    
    fig = px.sunburst(
        moneyness_df,
        path=['type', 'moneyness'],
        values='count',
        title='Option Moneyness Distribution'
    )
    
    fig.update_layout(height=500)
    
    return fig

# STREAMLIT UI
def main():
    st.set_page_config(page_title="Nifty Option Flow", layout="wide")
    st.title("📊 Advanced Nifty Options Analytics Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        expiry = st.text_input("Expiry Date (DD-MM-YYYY):", "24-04-2025")
    with col2:
        metric = st.selectbox("Primary Metric", ["volume", "oi", "ltp"])
    with col3:
        top_n = st.slider("Top N Active Options", 5, 20, 10)

    spot_price = fetch_nifty_price()
    if spot_price:
        st.markdown(f"**Current Nifty Spot Price:** {spot_price:.2f}")
    
    raw_data = fetch_options_data(expiry=expiry)
    df = process_options_data(raw_data, spot_price)

    if df is None or df.empty:
        st.error("No option data available.")
        return
    
    # Main metrics
    total_oi = df['oi'].sum()
    total_volume = df['volume'].sum()
    pcr_total = df[df['type'] == 'PE']['oi'].sum() / df[df['type'] == 'CE']['oi'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Open Interest", f"{total_oi:,}")
    col2.metric("Total Volume", f"{total_volume:,}")
    col3.metric("Total PCR", f"{pcr_total:.2f}")
    
    # Alluvial Flow
    st.subheader("Alluvial Flow of Most Active Options")
    fig = build_sankey(df, metric=metric, top_n=top_n)
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap and Stacked Bar
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("OI/Volume Heatmap")
        heatmap_metric = st.selectbox("Heatmap Metric", ["volume", "oi"], key="heatmap_metric")
        st.plotly_chart(build_heatmap(df, heatmap_metric), use_container_width=True)
    with col2:
        st.subheader("OI/Volume Comparison")
        stacked_metric = st.selectbox("Stacked Bar Metric", ["volume", "oi"], key="stacked_metric")
        st.plotly_chart(build_stacked_bar(df, stacked_metric), use_container_width=True)
    
    # IV Smile and Change in OI
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Implied Volatility Smile")
        st.plotly_chart(build_iv_smile(df, spot_price), use_container_width=True)
    with col2:
        st.subheader("Change in Open Interest")
        st.plotly_chart(build_change_oi_chart(df, spot_price), use_container_width=True)
    
    # Gamma Exposure and PCR
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gamma Exposure")
        st.plotly_chart(build_gamma_exposure(df, spot_price), use_container_width=True)
    with col2:
        st.subheader("Put-Call Ratio by Strike")
        st.plotly_chart(build_pcr_chart(df, spot_price), use_container_width=True)
    
    # Moneyness Distribution
    st.subheader("Option Moneyness Distribution")
    st.plotly_chart(build_moneyness_chart(df), use_container_width=True)

if __name__ == "__main__":
    main()
