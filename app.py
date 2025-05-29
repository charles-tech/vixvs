import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

# Função para buscar dados
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
            time.sleep(1)

# Estrutura do Streamlit
st.title("📈 Análise VIX vs Ativos")
ticker_input = st.text_input("Digite o ticker (ex: AAPL, PETR4.SA):", value='AAPL')

end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(months=24)

try:
    vix_data = fetch_data('^VIX', start_date, end_date)
    asset_data = fetch_data(ticker_input, start_date, end_date)
    
    # Garantir índices tz-naive para evitar erros de timezone
    vix_data.index = vix_data.index.tz_localize(None)
    asset_data.index = asset_data.index.tz_localize(None)

    combined = pd.DataFrame({
        'VIX': vix_data,
        'Ativo': asset_data
    }).dropna()

    if len(combined) < 50:
        st.error("⚠️ Dados insuficientes para análise.")
        st.stop()

    # Cálculo dos indicadores
    asset_df = asset_data.to_frame('Close')
    asset_df['MA20'] = asset_df['Close'].rolling(20).mean()
    asset_df['MA50'] = asset_df['Close'].rolling(50).mean()
    
    # RSI
    delta = asset_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    asset_df['RSI'] = 100 - (100 / (1 + rs))
    
    # Garantir que não há nulos
    asset_df.dropna(inplace=True)
    
    # Regressão Linear
    X = asset_df[['RSI', 'MA20', 'MA50']].iloc[-len(combined):]
    y = combined['Ativo']

    model = LinearRegression()
    
    try:
        model.fit(X, y)
    except ValueError as ve:
        st.error("Erro durante o ajuste do modelo: verifique os dados de entrada.")
        st.stop()

    # Previsão
    last_values = asset_df.iloc[-1]
    vix_scenarios = np.linspace(combined['VIX'].iloc[-1], combined['VIX'].iloc[-1]*0.95, 10)
    ma20_scenarios = np.linspace(last_values['MA20'], last_values['MA20']*1.02, 10)
    ma50_scenarios = np.linspace(last_values['MA50'], last_values['MA50']*1.01, 10)
    rsi_scenarios = np.linspace(last_values['RSI'], 50, 10)
    
    predicted_prices = []
    
    for i in range(10):
        features = pd.DataFrame([[
            ma20_scenarios[i],
            ma50_scenarios[i],
            rsi_scenarios[i]
        ]], columns=['MA20', 'MA50', 'RSI'])
        predicted_price = model.predict(features)[0]
        predicted_prices.append(predicted_price)
    
    # Exibindo a previsão
    future_dates = pd.date_range(combined.index[-1], periods=11, freq='B')[1:]
    
    predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted': predicted_prices
    })

    st.line_chart(predictions.set_index('Date'))

except Exception as e:
    st.error(f"Erro: {str(e)}. Verifique o ticker e sua conexão.")
