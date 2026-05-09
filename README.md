# 📊 Interactive Financial Regime & Price Prediction Dashboard

A Streamlit-based web application that uses Machine Learning to detect market regimes and predict stock prices with built-in confidence intervals.

## 🚀 Live Features
*   **Real-time Data:** Integration with `yfinance` to fetch 10 years of historical data for any ticker (Stocks, Crypto, ETFs).
*   **Unsupervised Regime Detection:** Uses a **Gaussian Mixture Model (GMM)** to classify market states based on volatility, drawdown, and trend strength.
*   **Walk-Forward Forecasting:** Implements a **Random Forest Regressor** that trains on a rolling window to predict the next day's return.
*   **Uncertainty Quantification:** Calculates prediction variance across Random Forest estimators to generate a **Confidence Range** (displayed as a visual band).
*   **Interactive Visuals:** Built with `Plotly` for high-detail tracking of historical accuracy and next-day price targets.

## 🛠️ Tech Stack
*   **Frontend:** Streamlit
*   **Data:** Yahoo Finance (yfinance)
*   **Machine Learning:** Scikit-learn (GaussianMixture, RandomForestRegressor, StandardScaler)
*   **Analysis:** Pandas, NumPy
*   **Visualization:** Plotly Graph Objects

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <(https://github.com/karlen-tsatinyan/first-app)>
   cd <repo-name>
   ```

2. **Install Dependencies:**
   ```bash
   pip install streamlit yfinance pandas numpy plotly scikit-learn
   ```

3. **Run the App:**
   ```bash
   streamlit run your_filename.py
   ```

## 🔍 How It Works

1.  **Data Pipeline:** The app downloads data for your chosen ticker and uses **SPY** as a benchmark to calculate relative strength and correlation.
2.  **Feature Engineering:** It generates 15+ technical indicators including RSI, MACD, Bollinger Band width, ATR, and Z-scores.
3.  **Regime Mapping:** The GMM clusters the data into 3 distinct market regimes (e.g., High Volatility, Trending, Mean Reverting).
4.  **Prediction:**
    *   The model performs a "Walk-Forward" validation on the last 60 days.
    *   It predicts the next day's closing price.
    *   The **Confidence Range** is derived from the standard deviation of the individual trees in the Random Forest ensemble.

## ⚖️ Disclaimer
This software is for **educational purposes only**. It does not constitute financial advice. The machine learning models provided are experimental and historical performance does not guarantee future results.

