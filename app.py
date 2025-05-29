import streamlit as st
import yfinance as yf
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def fetch_data(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker)
            hist = data.history(start=start_date, end=end_date)
            if hist.empty:
                raise ValueError("No data returned from Yahoo Finance")
            return hist['Close']
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # Wait before retrying

# Premium UI with Material Design and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    * {
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    .stTextInput>div>div>input {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        font-size: 16px !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #4285f4 !important;
        box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2) !important;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 16px;
        border-left: 4px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .stMarkdown h1 {
        color: #202124;
        font-weight: 500;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        background: linear-gradient(90deg, #4285f4, #34a853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 8px;
        position: relative;
    }
    
    .stMarkdown h1:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, #4285f4, #34a853);
        border-radius: 2px;
    }
    
    .stMarkdown h2, .stMarkdown h3 {
        color: #3c4043;
        font-weight: 500;
        margin-top: 2rem;
        border-bottom: none;
        position: relative;
        padding-left: 16px;
    }
    
    .stMarkdown h2:before, .stMarkdown h3:before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 60%;
        background: linear-gradient(to bottom, #4285f4, #34a853);
        border-radius: 2px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f3f4;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #dadce0;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #bdc1c6;
    }
    
    /* Animation classes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stMarkdown h1 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main title and input
st.markdown("<h1 style='text-align: center;'>📈 Análise VIX vs Ativos</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Visualize a relação entre o VIX e qualquer ativo listado</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col1:
    ticker_input = st.text_input("Digite o ticker (ex: AAPL, PETR4.SA):", value='AAPL')
with col2:
    st.markdown("<div style='height: 38px; display: flex; align-items: center; justify-content: center;'>🔍</div>", unsafe_allow_html=True)

end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(months=24)  # Increased to 2 years for more data points

try:
    print(f"Fetching VIX data from {start_date} to {end_date}")  # Debug
    vix_data = fetch_data('^VIX', start_date, end_date)
    print(f"VIX data: {len(vix_data)} points")  # Debug
    
    print(f"Fetching {ticker_input} data")  # Debug
    asset_data = fetch_data(ticker_input, start_date, end_date)
    print(f"{ticker_input} data: {len(asset_data)} points")  # Debug
    
    # Debug: Check data ranges and counts
    print(f"VIX data range: {vix_data.index.min()} to {vix_data.index.max()} ({len(vix_data)} points)")
    print(f"Asset data range: {asset_data.index.min()} to {asset_data.index.max()} ({len(asset_data)} points)")
    
    # Convert to naive datetime and normalize dates
    vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None).normalize()
    asset_data.index = pd.to_datetime(asset_data.index).tz_localize(None).normalize()
    
    # Get union of all valid dates
    all_dates = pd.to_datetime(np.union1d(vix_data.index, asset_data.index))
    
    # Reindex both series to common dates
    vix_aligned = vix_data.reindex(all_dates).ffill()
    asset_aligned = asset_data.reindex(all_dates).ffill()
    
    # Combine aligned data
    combined = pd.DataFrame({
        'VIX': vix_aligned,
        'Ativo': asset_aligned
    }).dropna()
    
    if len(combined) < 50:
        st.error(f"""
        ⚠️ Dados insuficientes para análise ⚠️
        
        Possíveis causas:
        1. Ticker inválido ou não encontrado
        2. Período sem dados coincidentes
        3. Problema de conexão com Yahoo Finance
        
        Detalhes:
        - VIX: {len(vix_data)} pontos ({vix_data.index.min()} a {vix_data.index.max()})
        - {ticker_input}: {len(asset_data)} pontos ({asset_data.index.min()} a {asset_data.index.max()})
        - Dados coincidentes: {len(combined)} pontos
        """)
        st.stop()
        
    print(f"Combined data points: {len(combined)}")  # Debug
        
    # Show current values with premium styling
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>📊 Valores Atuais</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #4285f4;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <div style="width: 40px; height: 40px; background: #4285f4; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span class="material-icons" style="color: white; font-size: 24px;">trending_up</span>
                </div>
                <div style="font-size: 1rem; font-weight: 500; color: #5f6368;">VIX ATUAL</div>
            </div>
            <div style="font-size: 2.25rem; font-weight: 700; color: #202124; margin: 8px 0;">${vix_data.iloc[-1]:.2f}</div>
            <div style="display: flex; align-items: center; gap: 4px; color: #5f6368; font-size: 0.875rem;">
                <span class="material-icons" style="font-size: 16px;">event</span>
                {vix_data.index[-1].strftime('%d/%m/%Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ea4335;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <div style="width: 40px; height: 40px; background: #ea4335; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span class="material-icons" style="color: white; font-size: 24px;">show_chart</span>
                </div>
                <div style="font-size: 1rem; font-weight: 500; color: #5f6368;">{ticker_input.upper()}</div>
            </div>
            <div style="font-size: 2.25rem; font-weight: 700; color: #202124; margin: 8px 0;">${asset_data.iloc[-1]:.2f}</div>
            <div style="display: flex; align-items: center; gap: 4px; color: #5f6368; font-size: 0.875rem;">
                <span class="material-icons" style="font-size: 16px;">event</span>
                {asset_data.index[-1].strftime('%d/%m/%Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Improved visualization with synchronized scales
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add VIX trace (left axis)
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined['VIX'],
            name='VIX',
            line=dict(color='#3498db', width=2),
            hovertemplate='VIX: %{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add Asset trace (right axis)
    fig.add_trace(
        go.Scatter(
            x=combined.index,
            y=combined['Ativo'],
            name=ticker_input,
            line=dict(color='#e74c3c', width=2),
            hovertemplate=f'{ticker_input}: %{{y:.2f}}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f'VIX vs {ticker_input} - Análise Comparativa',
        xaxis_title='Data',
        yaxis_title='VIX',
        yaxis2_title=f'{ticker_input} (US$)',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.8)',
        margin=dict(l=50, r=50, b=50, t=50),
        height=500
    )
    
    # Add current value indicators
    fig.add_vline(
        x=combined.index[-1],
        line_width=1,
        line_dash='dash',
        line_color='grey'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate technical indicators with proper alignment
    asset_df = asset_data.to_frame('Close')
    # Ensure we have enough data points for indicators
    if len(asset_df) < 50:
        st.warning(f"Necessário mínimo de 50 dias de dados para análise. Disponível: {len(asset_df)}")
        st.stop()
    
    # Calculate indicators
    asset_df['MA20'] = asset_df['Close'].rolling(20).mean()
    asset_df['MA50'] = asset_df['Close'].rolling(50).mean()
    
    # Calculate RSI only on aligned data
    valid_data = asset_df.dropna().copy()
    delta = valid_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    valid_data['RSI'] = 100 - (100 / (1 + rs))
    asset_df['RSI'] = valid_data['RSI']
    
    # Create technical analysis plot
    fig_ta = px.line(asset_df.reset_index(), 
                    x='Date',
                    y=['Close', 'MA20', 'MA50'],
                    title=f'Análise Técnica - {ticker_input}',
                    labels={'value': 'Preço (US$)', 'variable': 'Indicador'})
    fig_ta.update_layout(
        yaxis_tickprefix='$',
        yaxis_tickformat=',.2f'
    )
    
    # Add RSI subplot
    fig_rsi = px.line(asset_df.reset_index(),
                     x='Date',
                     y=['RSI'],
                     title='RSI (14 dias)')
    
    st.plotly_chart(fig_ta, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Machine Learning Prediction
    st.markdown("### Previsão de Preços (10 dias)")
    st.markdown("""
    **Metodologia Aprimorada:**
    
    1. **Modelo:** Regressão Linear Múltipla considerando:
       - Valor do VIX
       - Médias Móveis (20 e 50 dias)
       - RSI (14 dias)
    2. **Cenário:** Projeção baseada em:
       - Redução de 5% no VIX
       - Evolução das médias móveis
       - Normalização do RSI
    3. **Premissas:**
       - Mantém relações históricas entre indicadores
       - Tendências recentes se mantêm
    4. **Limitações:**
       - Projeção estatística
       - Não considera eventos extraordinários
    """)
    
    # Prepare aligned data for ML with technical indicators
    ml_data = combined[['VIX', 'Ativo']].copy()
    
    # Get aligned technical indicators
    aligned_asset_df = asset_data.loc[combined.index].to_frame('Close')
    aligned_asset_df['MA20'] = aligned_asset_df['Close'].rolling(20).mean()
    aligned_asset_df['MA50'] = aligned_asset_df['Close'].rolling(50).mean()
    
    # Calculate RSI on aligned data
    delta = aligned_asset_df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    aligned_asset_df['RSI'] = 100 - (100 / (1 + rs))
    
    # Add indicators to ML data
    ml_data['MA20'] = aligned_asset_df['MA20'].values
    ml_data['MA50'] = aligned_asset_df['MA50'].values
    ml_data['RSI'] = aligned_asset_df['RSI'].values
    ml_data = ml_data.dropna()
    
    # Features and target
    X = ml_data[['VIX', 'MA20', 'MA50', 'RSI']]
    y = ml_data['Ativo']
    
    # Train multiple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Create future dates
    last_date = ml_data.index[-1]
    future_dates = pd.date_range(last_date, periods=11, freq='B')[1:]  # 10 business days
    
    # Create future scenarios
    last_values = ml_data.iloc[-1]
    vix_scenarios = np.linspace(last_values['VIX'], last_values['VIX']*0.95, 10)
    ma20_scenarios = np.linspace(last_values['MA20'], last_values['MA20']*1.02, 10)  # Slight upward trend
    ma50_scenarios = np.linspace(last_values['MA50'], last_values['MA50']*1.01, 10)  # More stable
    rsi_scenarios = np.linspace(last_values['RSI'], 50, 10)  # Normalizing towards 50
    
    # Predict prices with proper feature names
    predicted_prices = []
    feature_names = ['VIX', 'MA20', 'MA50', 'RSI']
    
    for i in range(10):
        features = pd.DataFrame([[
            vix_scenarios[i],
            ma20_scenarios[i],
            ma50_scenarios[i],
            rsi_scenarios[i]
        ]], columns=feature_names)
        predicted_price = model.predict(features)[0]
        predicted_prices.append(predicted_price)
    
    # Create prediction dataframe with confidence interval
    predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted': predicted_prices,
        'Upper': [p * 1.05 for p in predicted_prices],  # +5% confidence
        'Lower': [p * 0.95 for p in predicted_prices]   # -5% confidence
    })
    
    # Create professional prediction chart
    fig_pred = go.Figure()
    
    # Add confidence interval
    fig_pred.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Upper'],
        fill=None,
        mode='lines',
        line_color='rgba(59, 130, 246, 0.2)',
        name='Intervalo +5%',
        showlegend=False
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(59, 130, 246, 0.2)',
        fillcolor='rgba(59, 130, 246, 0.2)',
        name='Intervalo -5%',
        showlegend=False
    ))
    
    # Add main prediction line
    fig_pred.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Predicted'],
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8),
        name='Previsão',
        hovertemplate='<b>Previsão</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Update layout to match main style
    fig_pred.update_layout(
        title={
            'text': f'Previsão de Preço para {ticker_input} (próximos 10 dias)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title='Data',
        yaxis_title='Preço (US$)',
        hovermode='x unified',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(240,240,240,0.8)',
        margin=dict(l=50, r=50, b=50, t=100),
        height=500,
        yaxis=dict(
            tickprefix='$',
            tickformat=',.2f',
            gridcolor='rgba(0,0,0,0.05)'
        )
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Show predicted values with enhanced styling
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>📅 Valores Previstos</h3>", unsafe_allow_html=True)
    predictions['Predicted'] = predictions['Predicted'].round(2)
    st.dataframe(
        predictions.style
            .format({'Predicted': '${:.2f}'})
            .set_properties(**{
                'background-color': '#f8f9fa',
                'color': '#2c3e50',
                'border': '1px solid #dfe6e9'
            })
            .highlight_max(axis=0, color='#d4edda')
            .highlight_min(axis=0, color='#f8d7da'),
        use_container_width=True
    )
    
except Exception as e:
    st.error(f"Erro: {str(e)}. Verifique o ticker e sua conexão.")
