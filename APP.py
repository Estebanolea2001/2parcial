import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Análisis Financiero", layout="wide")
st.title("📈 Análisis de Acciones con Yahoo Finance")

# Entrada de tickers
tickers_input = st.text_area("Ingresa los símbolos de las acciones separados por coma (ej. AAPL, MSFT, TSLA):", value="AAPL, MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

# Fechas
start_date = st.date_input("Fecha de inicio", datetime.today() - timedelta(days=365 * 3))
end_date = st.date_input("Fecha de fin", datetime.today())

if tickers:
    try:
        sp500 = yf.download("SPY", start=start_date, end=end_date)
        sp500["Daily Return"] = sp500["Close"].pct_change()

        if sp500.empty or "Close" not in sp500.columns:
            st.error("No se pudieron descargar los datos de SPY.")
        else:
            for ticker in tickers:
                st.markdown("---")
                st.header(f"📊 Análisis de {ticker}")

                info = yf.Ticker(ticker).info

                st.subheader("🏢 Información de la empresa")
                st.markdown(f"**Nombre:** {info.get('longName', 'N/D')}")
                st.markdown(f"**Sector:** {info.get('sector', 'N/D')}")
                st.markdown(f"**Industria:** {info.get('industry', 'N/D')}")
                st.markdown(f"**País:** {info.get('country', 'N/D')}")
                st.markdown(f"**Sitio web:** [{info.get('website', 'N/D')}]({info.get('website', '#')})")
                st.markdown(f"**Capitalización de mercado:** {info.get('marketCap', 'N/D'):,}")

                if info.get("longBusinessSummary"):
                    with st.expander("📄 Descripción del negocio"):
                        st.write(info["longBusinessSummary"])

                data = yf.download(ticker, start=start_date, end=end_date)

                if data.empty or "Close" not in data.columns:
                    st.warning(f"No hay datos válidos para {ticker}.")
                    continue

                st.subheader(f"📈 Precio de cierre de {ticker}")
                st.line_chart(data["Close"])

                st.subheader("📄 Datos recientes")
                st.dataframe(data.tail())

                data["Daily Return"] = data["Close"].pct_change()
                combined = pd.concat([data["Daily Return"], sp500["Daily Return"]], axis=1)
                combined.columns = [f"{ticker}_Return", "SP500_Return"]
                combined.dropna(inplace=True)

                X = combined["SP500_Return"]
                Y = combined[f"{ticker}_Return"]
                beta, alpha = np.polyfit(X, Y, 1)
                a2, b2, c2 = np.polyfit(X, Y, 2)

                annualized_return = (1 + data["Daily Return"].mean())**252 - 1
                alpha_annual = (1 + alpha)**252 - 1

                st.subheader("📌 Indicadores Financieros")
                st.table(pd.DataFrame({
                    "Alpha anual": [f"{alpha_annual:.4f}"],
                    "Beta (lineal)": [f"{beta:.4f}"],
                    "Rendimiento anual": [f"{annualized_return:.2%}"]
                }, index=[ticker]))

                st.subheader("📐 Modelo Cuadrático")
                st.code(f"y = {a2:.4f} * x² + {b2:.4f} * x + {c2:.4f}")

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(X, Y, alpha=0.4, label='Datos reales')
                x_vals = np.linspace(X.min(), X.max(), 200)
                ax.plot(x_vals, beta * x_vals + alpha, color='green', label='Lineal', linewidth=2)
                ax.plot(x_vals, a2 * x_vals**2 + b2 * x_vals + c2, color='red', linestyle='--', label='Cuadrática', linewidth=2)

                ax.set_xlabel("Rendimiento diario SPY")
                ax.set_ylabel(f"Rendimiento diario {ticker}")
                ax.set_title(f"Regresión {ticker} vs SPY")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")