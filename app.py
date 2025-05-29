import streamlit as st
import yfinance as yf
import time
import pandas as pd

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

def ensure_tz_naive(date):
    return date.tz_localize(None)

def adjust_end_date(data, end_date):
    last_available_date = data.index.max()
    last_available_date = ensure_tz_naive(last_available_date)
    if end_date > last_available_date:
        end_date = last_available_date
        st.warning(f"Ajustando a data de fim para a última data disponível: {end_date.strftime('%d/%m/%Y')}")
    return end_date

st.markdown("<h1 style='text-align: center;'>📈 Análise VIX vs Ativos</h1>", unsafe_allow_html=True)
ticker_input = st.text_input("Digite o ticker (ex: AAPL, PETR4.SA):", value='AAPL')

end_date = pd.Timestamp.now().tz_localize(None)  # Ensure tz-naive
start_date = end_date - pd.DateOffset(months=24)

try:
    vix_data = fetch_data('^VIX', start_date, end_date)
    asset_data = fetch_data(ticker_input, start_date, end_date)
    
    end_date = adjust_end_date(vix_data, end_date)
    end_date = adjust_end_date(asset_data, end_date)

    combined = pd.DataFrame({'VIX': vix_data, 'Ativo': asset_data}).dropna()

    if len(combined) < 50:
        st.error(f"""
        ⚠️ Dados insuficientes para análise ⚠️
        
        Possíveis causas:
        1. Ticker inválido ou não encontrado.
        2. Período sem dados coincidentes.
        3. Problema de conexão com Yahoo Finance.
        """)
        st.stop()
    
    # Continue com sua análise...

except Exception as e:
    st.error(f"Erro: {str(e)}. Verifique o ticker e sua conexão.")
