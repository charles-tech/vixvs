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

def ensure_date_in_index(data, date):
    if date not in data.index:
        st.warning(f"A data {date.strftime('%d/%m/%Y')} não está nos dados disponíveis.")
    return data

# Simplified UI code with added checks

end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(months=24)

try:
    vix_data = fetch_data('^VIX', start_date, end_date)
    asset_data = fetch_data(ticker_input, start_date, end_date)
    
    vix_data = ensure_date_in_index(vix_data, end_date)
    asset_data = ensure_date_in_index(asset_data, end_date)
    
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
    
    # Resto do código para visualizações e previsões...

except Exception as e:
    st.error(f"Erro: {str(e)}. Verifique o ticker e sua conexão.")
