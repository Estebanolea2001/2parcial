import streamlit as st
st.set_page_config(page_title="📊 Análisis Financiero de Acciones", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import openai

# ✅ Usar clave desde Streamlit secrets
api_key = st.secrets.get("OPENAI_API_KEY")
if api_key:
    openai.api_key = api_key

# Verificación de API Key
st.sidebar.markdown("### 🔐 Estado de API Key")
if not api_key:
    st.sidebar.error("❌ No se encontró OPENAI_API_KEY en secrets.")
else:
    st.sidebar.success(f"✅ API Key cargada: {api_key[:10]}...")

# Título de la app
st.title("📈 Análisis de Acciones con Yahoo Finance")

# Parámetros de entrada
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    tickers_input = st.text_area("Símbolos de acciones (ej. AAPL, MSFT)", value="AAPL, MSFT")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.date_input("📅 Fecha de inicio", datetime.today() - timedelta(days=365 * 3))
    end_date = st.date_input("📅 Fecha de fin", datetime.today())
    sma_short = st.number_input("📉 SMA corto plazo", value=50, min_value=1)
    sma_long = st.number_input("📈 SMA largo plazo", value=200, min_value=10)

# Lógica principal
if tickers:
    try:
        sp500 = yf.download("SPY", start=start_date, end=end_date)
        sp500["Daily Return"] = sp500["Close"].pct_change()

        for ticker in tickers:
            st.markdown("---")
            st.header(f"📊 Análisis de {ticker}")

            info = yf.Ticker(ticker).info
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏢 Información")
                st.markdown(f"**Nombre:** {info.get('longName', 'N/D')}")
                st.markdown(f"**Sector:** {info.get('sector', 'N/D')}")
                st.markdown(f"**Industria:** {info.get('industry', 'N/D')}")
                st.markdown(f"**País:** {info.get('country', 'N/D')}")
                st.markdown(f"**Sitio web:** [{info.get('website', 'N/D')}]({info.get('website', '#')})")

            with col2:
                st.subheader("💰 Financieros")
                st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/D'):,}")
                if info.get("longBusinessSummary"):
                    with st.expander("📄 Descripción del negocio"):
                        st.write(info["longBusinessSummary"])

            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning(f"No hay datos para {ticker}.")
                continue

            data[f"SMA{sma_short}"] = data["Close"].rolling(window=sma_short).mean()
            data[f"SMA{sma_long}"] = data["Close"].rolling(window=sma_long).mean()

            st.subheader("📈 Precio con Promedios Móviles")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(data.index, data["Close"], label="Close", linewidth=1.5)
            ax2.plot(data.index, data[f"SMA{sma_short}"], '--', label=f"SMA {sma_short}")
            ax2.plot(data.index, data[f"SMA{sma_long}"], '--', label=f"SMA {sma_long}")
            ax2.legend(); ax2.grid(); ax2.set_ylabel("USD")
            st.pyplot(fig2)

            st.subheader("📄 Datos recientes")
            st.dataframe(data[["Close", f"SMA{sma_short}", f"SMA{sma_long}"]].tail(10))

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

            st.subheader("📐 Regresión Cuadrática")
            st.code(f"y = {a2:.4f} * x² + {b2:.4f} * x + {c2:.4f}")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, Y, alpha=0.3)
            x_vals = np.linspace(X.min(), X.max(), 200)
            ax.plot(x_vals, beta * x_vals + alpha, label="Lineal", color="green")
            ax.plot(x_vals, a2 * x_vals**2 + b2 * x_vals + c2, '--', label="Cuadrática", color="red")
            ax.legend(); ax.grid()
            st.pyplot(fig)

            st.subheader("📆 Próximos Eventos Financieros")
            try:
                calendar = yf.Ticker(ticker).calendar
                if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                    st.dataframe(calendar.T)
                    next_earnings = calendar.loc["Earnings Date"][0]
                    if isinstance(next_earnings, pd.Timestamp):
                        days_to_earnings = (next_earnings - pd.Timestamp.today()).days
                        if 0 <= days_to_earnings <= 10:
                            st.warning(f"📢 Earnings en {days_to_earnings} días ({next_earnings.date()})")
                        else:
                            st.info(f"🗓️ Próximo earnings: {next_earnings.date()}")
                else:
                    st.info("No hay eventos futuros.")
            except Exception as e:
                st.warning(f"Error al obtener eventos: {e}")

            # 🧪 Backtesting de Cruce de Medias
            st.subheader("🔁 Backtesting: Cruce de Medias")
            data.dropna(inplace=True)
            data["Signal"] = 0
            data.loc[data[f"SMA{sma_short}"] > data[f"SMA{sma_long}"], "Signal"] = 1
            data["Position"] = data["Signal"].shift(1)
            data["Strategy Return"] = data["Daily Return"] * data["Position"]
            data["Cumulative Return"] = (1 + data["Daily Return"]).cumprod()
            data["Cumulative Strategy"] = (1 + data["Strategy Return"]).cumprod()

            fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
            ax_bt.plot(data.index, data["Cumulative Return"], "--", label="Buy & Hold")
            ax_bt.plot(data.index, data["Cumulative Strategy"], label="Estrategia SMA", linewidth=2)
            ax_bt.set_title("Evolución del Portafolio")
            ax_bt.legend(); ax_bt.grid()
            st.pyplot(fig_bt)

            total_return = data["Cumulative Strategy"].iloc[-1] - 1
            st.info(f"📈 Retorno total de la estrategia: **{total_return:.2%}**")

            with st.expander("ℹ️ ¿Por qué este resultado?"):
                st.markdown(f"""
Este resultado se basa en una estrategia de **cruce de medias móviles**:

- ✅ Compra cuando la media corta ({sma_short}) cruza por encima de la larga ({sma_long}).
- ❌ Venta cuando cruza por debajo.

Esta estrategia busca capturar tendencias y evitar caídas grandes, pero puede perderse algunas subidas rápidas.

El rendimiento depende de: calidad de las señales, volatilidad y duración analizada.
""")

            # 🔮 IA con OpenAI
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
                    st.error(f"No se pudo obtener análisis de GPT: {e}")
            else:
                st.warning("⚠️ No se detectó clave API de OpenAI.")
    except Exception as e:
        st.error(f"❌ Error general: {e}")