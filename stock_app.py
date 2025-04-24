import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# Load preprocessed data (ensure you have this CSV from your ARIMA pipeline)
data = pd.read_csv("stock_new.csv", parse_dates=['date'], index_col='date')

# Sidebar: Forecast settings
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Select forecast horizon (days):", 7, 60, 30)

# Header
st.title("📈 Stock Price Forecasting App - ARIMA")
st.markdown("This app uses an ARIMA model to forecast future stock prices.")

# Show data
st.subheader("📊 Historical Stock Data")
st.dataframe(data.tail(10))

# Prepare train/test split
diff_data = data['close'].diff().dropna()
split_idx = int(len(diff_data) * 0.8)
train_diff = diff_data[:split_idx]
test = data['close'][split_idx:]

# Fit ARIMA
p, d, q = 1, 0, 0  # Based on your previous best_order_aic
model = sm.tsa.ARIMA(train_diff, order=(p, d, q))
results = model.fit()

# Forecast
y_forecast_diff = results.forecast(steps=forecast_days)
last_close = data['close'].iloc[split_idx - 1]
y_forecast = y_forecast_diff.cumsum() + last_close
forecast_index = pd.date_range(start=data.index[split_idx], periods=forecast_days, freq='D')
y_forecast.index = forecast_index

# Evaluation
actual = test[:forecast_days]
actual = actual[actual.index.isin(forecast_index)]
rmse = np.sqrt(mean_squared_error(actual, y_forecast[:len(actual)]))
mae = mean_absolute_error(actual, y_forecast[:len(actual)])
mape = np.mean(np.abs((actual - y_forecast[:len(actual)]) / actual)) * 100

# Metrics
st.subheader("📏 Evaluation Metrics")
st.metric("RMSE", f"{rmse:.4f}")
st.metric("MAE", f"{mae:.4f}")
st.metric("MAPE", f"{mape:.2f}%")

# Plot forecast
st.subheader("🔮 Forecast vs Actual")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['close'], label='Historical')
ax.plot(y_forecast.index, y_forecast, label='Forecast', color='red')
ax.set_title("Stock Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Forecast table
st.subheader("📅 Forecasted Prices")
st.dataframe(y_forecast.rename("Forecasted Price"))

# Download option
st.download_button("Download Forecast", y_forecast.to_csv().encode(), "stock_forecast.csv", "text/csv")

# Footer
st.caption("Model used: ARIMA(1,0,0) — Selected by AIC from grid search")
