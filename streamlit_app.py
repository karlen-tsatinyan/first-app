import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ===============================
# CONFIG
# ===============================
class Config:
    BENCHMARK = "SPY"
    YEARS = 10
    END_DATE = datetime.today()
    # Fixed Reference for START_DATE
    START_DATE = (END_DATE - timedelta(days=365*YEARS)).strftime("%Y-%m-%d")

# ===============================
# DATA PIPELINE
# ===============================
@st.cache_data
def load_and_clean_data(ticker):
    stock = yf.download(ticker, start=Config.START_DATE)
    spy = yf.download(Config.BENCHMARK, start=Config.START_DATE)

    # Handle yfinance MultiIndex columns (Fix for v0.2.x)
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    stock = stock.dropna()
    spy = spy.dropna()

    stock["ret_1d"] = stock["Close"].pct_change()
    spy["spy_ret_1d"] = spy["Close"].pct_change()

    df = pd.merge(
        stock.reset_index(),
        spy.reset_index()[["Date", "spy_ret_1d"]],
        on="Date",
        how="inner"
    ).set_index("Date").dropna()
    
    return df

def add_features(df):
    df = df.copy()
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["drawdown"] = df["Close"] / df["Close"].cummax() - 1
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["trend_strength"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-6))))

    # MACD & Bollinger
    df["macd_hist"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["bb_width"] = (2 * df["Close"].rolling(20).std()) / df["sma_20"]

    # ATR & Z-Score
    tr = pd.concat([(df["High"] - df["Low"]), 
                    (df["High"] - df["Close"].shift()).abs(), 
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["zscore"] = (df["Close"] - df["sma_20"]) / (df["Close"].rolling(20).std() + 1e-6)

    # Market Context
    df["corr_spy"] = df["ret_1d"].rolling(30).corr(df["spy_ret_1d"])
    df["rel_strength_20"] = df["ret_20d"] - df["spy_ret_1d"].rolling(20).sum()

    return df.dropna()

def detect_regime(df):
    features = ["vol_20", "drawdown", "trend_strength", "bb_width", "atr"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    gmm = GaussianMixture(n_components=3, random_state=42)
    df["regime"] = gmm.fit_predict(X)
    return df

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Financial Prediction", layout="wide")
st.title("📊 Interactive Financial Prediction Dashboard")

if st.sidebar.button("Clear Cache / Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.popover("Select Ticker"):
    user_input = st.text_input("Input your ticker (e.g. AAPL, BTC-USD)", "AAPL").upper()

if user_input:
    try:
        with st.spinner(f"Analyzing {user_input}..."):
            # 1. Pipeline
            raw_df = load_and_clean_data(user_input)
            df_feat = add_features(raw_df)
            df = detect_regime(df_feat)
            
            # 2. Walk-Forward Prediction Logic
            features = ["ret_1d","ret_5d","ret_20d","vol_20","drawdown",
                        "trend_strength","rsi","macd_hist","bb_width",
                        "atr","zscore","corr_spy","rel_strength_20","regime"]
            
            window = 60 
            df_model = df.copy()
            df_model["target"] = df_model["ret_1d"].shift(-1)
            df_model = df_model[features + ["target", "Close"]].dropna()
            
            dates, actuals, preds, std_devs = [], [], [], []

            for i in range(len(df_model) - window, len(df_model)):
                train = df_model.iloc[:i]
                test = df_model.iloc[i:i+1]
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                model.fit(train[features], train["target"])
                
                # Get uncertainty from tree variance
                per_tree_pred = [t.predict(test[features].values) for t in model.estimators_]
                preds.append(np.mean(per_tree_pred))
                std_devs.append(np.std(per_tree_pred))
                actuals.append(test["target"].values[0])
                dates.append(test.index[0])

            # Convert to price levels
            price_now = df_model.loc[dates, "Close"].values
            y_true = price_now * (1 + np.array(actuals))
            y_pred = price_now * (1 + np.array(preds))
            margin = price_now * (np.array(std_devs) * 2)

            # --- PLOTLY CHART 1: HISTORICAL ACCURACY ---
            st.subheader(f"Historical Prediction Accuracy (Last {window} Days)")
            fig1 = go.Figure()

            # Confidence Band
            fig1.add_trace(go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([y_pred + margin, (y_pred - margin)[::-1]]),
                fill='toself',
                fillcolor='rgba(128,128,128,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="Confidence Range"
            ))

            fig1.add_trace(go.Scatter(x=dates, y=y_true, name="Actual Price", line=dict(color='#1f77b4', width=2)))
            fig1.add_trace(go.Scatter(x=dates, y=y_pred, name="Predicted Price", line=dict(color='#ff7f0e', dash='dash')))

            fig1.update_layout(hovermode="x unified", template="plotly_white", height=500)
            st.plotly_chart(fig1, use_container_width=True)

            # --- PLOTLY CHART 2: NEXT-DAY DETAIL ---
            st.subheader("Next-Day Prediction Detail")
            last_date = df.index[-1]
            hist_df = df.iloc[-100:]
            
            final_pred_price = y_pred[-1]
            final_low = y_pred[-1] - margin[-1]
            final_high = y_pred[-1] + margin[-1]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Close"], name="Recent History", line=dict(color="black")))
            
            # Prediction Point & Error Bar
            fig2.add_trace(go.Scatter(
                x=[last_date], y=[final_pred_price],
                mode='markers',
                marker=dict(color='red', size=12),
                name="Next Day Pred",
                error_y=dict(type='data', symmetric=False, array=[final_high - final_pred_price], arrayminus=[final_pred_price - final_low], color='orange')
            ))

            fig2.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig2, use_container_width=True)

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast", "BULLISH ↑" if preds[-1] > 0 else "BEARISH ↓")
            c2.metric("Predicted Return", f"{preds[-1]:.2%}")
            c3.metric("Expected Price", f"${final_pred_price:.2f}")

    except Exception as e:
        st.error(f"Error processing {user_input}: {e}")
        st.info("Tip: Ensure you have installed requirements: `pip install streamlit yfinance pandas numpy plotly scikit-learn`")
else:
    st.info("Enter a ticker symbol to generate interactive forecasts.")
