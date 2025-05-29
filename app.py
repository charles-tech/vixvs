import streamlit as st
import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker)
            hist = data.history(start=start_date, end=end_date)
            if hist.empty:
                raise ValueError(f"Nenhuma informação retornada do Yahoo Finance para {ticker}")
            return hist['Close']
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            st.error(f"Erro ao buscar dados: {e}")
            time.sleep(1)

st.title("📈 Análise VIX vs Ativos")
ticker_input = st.text_input("Digite o ticker (ex: AAPL, PETR4.SA):", value='AAPL')

end_date = pd.Timestamp.now().tz_localize(None)
start_date = end_date - pd.DateOffset(months=24)

try:
    vix_data = fetch_data('^VIX', start_date, end_date)
    asset_data = fetch_data(ticker_input, start_date, end_date)

    if vix_data.empty or asset_data.empty:
        st.error("Nenhum dado retornado, por favor verifique o ticker e tente novamente.")
        st.stop()

    combined = pd.DataFrame({
        'VIX': vix_data,
        'Ativo': asset_data
    }).dropna()

    if len(combined) < 50:
        st.error("⚠️ Dados insuficientes para análise. Por favor, tente outro período ou ticker.")
        st.stop()

    st.line_chart(combined)

except Exception as e:
    st.error(f"Erro inesperado: {str(e)}. Verifique o ticker e sua conexão.")
