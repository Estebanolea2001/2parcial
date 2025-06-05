import streamlit as st
st.set_page_config(page_title="📊 Análisis Financiero de Acciones", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import openai

# ✅ Cargar .env automáticamente
load_dotenv()
file_exists = os.path.exists(".env")
api_key = os.getenv("OPENAI_API_KEY")

# ✅ Verificación en el sidebar
st.sidebar.markdown("### 🔐 Estado de API Key")
if not file_exists:
    st.sidebar.error("❌ El archivo `.env` no existe en la carpeta actual.")
elif not api_key:
    st.sidebar.error("❌ No se encontró la variable OPENAI_API_KEY en `.env`.")
else:
    st.sidebar.success(f"✅ API Key cargada: {api_key[:10]}...")
    openai.api_key = api_key

# Título de la app
st.title("📈 Análisis de Acciones con Yahoo Finance")

# Parámetros de entrada
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    tickers_input = st.text_area("Símbolos de las acciones (ej. AAPL, MSFT, TSLA):", value="AAPL, MSFT")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

    start_date = st.date_input("📅 Fecha de inicio", datetime.today() - timedelta(days=365 * 3))
    end_date = st.date_input("📅 Fecha de fin", datetime.today())

    st.markdown("### Promedios móviles")
    sma_short = st.number_input("📌 Corto plazo (ej. 20-50 días)", value=50, min_value=1, max_value=200)
    sma_long = st.number_input("📌 Largo plazo (ej. 100-200 días)", value=200, min_value=50, max_value=400)

# Lógica principal
if tickers:
    try:
        sp500 = yf.download("SPY", start=start_date, end=end_date)
        sp500["Daily Return"] = sp500["Close"].pct_change()

        if sp500.empty or "Close" not in sp500.columns:
            st.error("No se pudieron descargar los datos del índice SPY.")
        else:
            for ticker in tickers:
                st.markdown("---")
                st.header(f"📊 Análisis de {ticker}")

                info = yf.Ticker(ticker).info
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🏢 Información de la empresa")
                    st.markdown(f"**Nombre:** {info.get('longName', 'N/D')}")
                    st.markdown(f"**Sector:** {info.get('sector', 'N/D')}")
                    st.markdown(f"**Industria:** {info.get('industry', 'N/D')}")
                    st.markdown(f"**País:** {info.get('country', 'N/D')}")
                    st.markdown(f"**Sitio web:** [{info.get('website', 'N/D')}]({info.get('website', '#')})")

                with col2:
                    st.subheader("💰 Datos Financieros")
                    st.markdown(f"**Capitalización de mercado:** {info.get('marketCap', 'N/D'):,}")
                    if info.get("longBusinessSummary"):
                        with st.expander("📄 Descripción del negocio"):
                            st.write(info["longBusinessSummary"])

                data = yf.download(ticker, start=start_date, end=end_date)

                if data.empty or "Close" not in data.columns:
                    st.warning(f"No hay datos válidos para {ticker}.")
                    continue

                # Cálculo de promedios móviles
                data[f"SMA{sma_short}"] = data["Close"].rolling(window=sma_short).mean()
                data[f"SMA{sma_long}"] = data["Close"].rolling(window=sma_long).mean()

                st.subheader(f"📈 Evolución del precio de {ticker}")
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(data.index, data["Close"], label="Precio de cierre", color="blue", linewidth=1.5)
                ax2.plot(data.index, data[f"SMA{sma_short}"], label=f"SMA {sma_short} días", color="orange", linestyle="--")
                ax2.plot(data.index, data[f"SMA{sma_long}"], label=f"SMA {sma_long} días", color="green", linestyle="--")
                ax2.set_title(f"{ticker} - Precio con Promedios Móviles")
                ax2.set_xlabel("Fecha")
                ax2.set_ylabel("USD")
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)

                st.subheader("📄 Datos recientes")
                st.dataframe(data[["Close", f"SMA{sma_short}", f"SMA{sma_long}"]].tail(10))

                # Indicadores financieros
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

                st.subheader("📐 Modelo de Regresión Cuadrática")
                st.code(f"y = {a2:.4f} * x² + {b2:.4f} * x + {c2:.4f}")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(X, Y, alpha=0.4)
                x_vals = np.linspace(X.min(), X.max(), 200)
                ax.plot(x_vals, beta * x_vals + alpha, color='green', label='Lineal')
                ax.plot(x_vals, a2 * x_vals**2 + b2 * x_vals + c2, color='red', linestyle='--', label='Cuadrática')
                ax.set_title("Regresión SP500 vs " + ticker)
                ax.set_xlabel("SP500 Return")
                ax.set_ylabel(f"{ticker} Return")
                ax.legend()
                st.pyplot(fig)

                # 📅 Eventos financieros relevantes
                st.subheader("📆 Próximos Eventos Financieros")
                try:
                    calendar = yf.Ticker(ticker).calendar
                    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                        st.dataframe(calendar.T)
                        next_earnings = calendar.loc["Earnings Date"][0]
                        if isinstance(next_earnings, pd.Timestamp):
                            days_to_earnings = (next_earnings - pd.Timestamp.today()).days
                            if 0 <= days_to_earnings <= 10:
                                st.warning(f"📢 Atención: el próximo reporte de ganancias es en {days_to_earnings} días ({next_earnings.date()})")
                            else:
                                st.info(f"🗓️ Próximo earnings report: {next_earnings.date()}")
                    else:
                        st.info("No se encontraron eventos próximos.")
                except Exception as e:
                    st.warning(f"No se pudieron obtener eventos: {e}")

                # 🧪 Backtesting: Cruce de medias móviles
                st.subheader("🔁 Backtesting: Estrategia de Cruce de Medias")
                data = data.dropna()
                data["Signal"] = 0
                data.loc[data[f"SMA{sma_short}"] > data[f"SMA{sma_long}"], "Signal"] = 1
                data["Position"] = data["Signal"].shift(1)
                data["Strategy Return"] = data["Daily Return"] * data["Position"]
                data["Cumulative Return"] = (1 + data["Daily Return"]).cumprod()
                data["Cumulative Strategy"] = (1 + data["Strategy Return"]).cumprod()

                fig_bt, ax_bt = plt.subplots(figsize=(10, 5))
                ax_bt.plot(data.index, data["Cumulative Return"], label="Buy & Hold", linestyle="--")
                ax_bt.plot(data.index, data["Cumulative Strategy"], label="Estrategia SMA", linewidth=2)
                ax_bt.set_title("Backtest de Estrategia de Cruce de Medias")
                ax_bt.set_ylabel("Crecimiento del Portafolio")
                ax_bt.legend()
                st.pyplot(fig_bt)

                total_return = data["Cumulative Strategy"].iloc[-1] - 1
                st.info(f"📈 Retorno total de la estrategia: **{total_return:.2%}**")

                # 🔮 Análisis con IA
                st.subheader("🧠 Análisis por GPT (OpenAI)")
                if api_key:
                    prompt = f"""
Eres un analista financiero. Evalúa la acción {ticker} con base en los siguientes datos:

- Alpha anual: {alpha_annual:.4f}
- Beta (lineal): {beta:.4f}
- Rendimiento anual estimado: {annualized_return:.2%}
- Sector: {info.get('sector', 'N/D')}
- Industria: {info.get('industry', 'N/D')}
- País: {info.get('country', 'N/D')}

Proporciona un resumen breve (máximo 80 palabras) sobre el comportamiento, riesgo y atractivo de esta acción para un inversionista.
"""
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.success(response["choices"][0]["message"]["content"])
                    except Exception as e:
                        st.error(f"No se pudo obtener el análisis de OpenAI: {e}")
                else:
                    st.warning("⚠️ No se detectó la API Key de OpenAI.")
    except Exception as e:
        st.error(f"❌ Ocurrió un error: {e}")