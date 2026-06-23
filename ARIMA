import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime

st.set_page_config(
    page_title="ARIMA Stock Forecasting",
    layout="wide"
)

st.title("📈 ARIMA Stock Price Forecasting System")
st.markdown(
    """
Forecast Indian stocks using ARIMA and Yahoo Finance data.

Examples:
- RELIANCE
- TCS
- INFY
- HDFCBANK
- ICICIBANK
- SBIN
"""
)

# --------------------------------
# STOCK INPUT
# --------------------------------

ticker_input = st.text_input(
    "Enter NSE Stock Symbol",
    value="RELIANCE"
)

ticker = ticker_input.upper() + ".NS"

# --------------------------------
# DOWNLOAD DATA
# --------------------------------

if st.button("Generate Forecast"):

    with st.spinner("Downloading data..."):

        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=5)

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

    if data.empty:
        st.error("No data found for this stock.")
        st.stop()

    st.success("Data Downloaded Successfully")

    # Use monthly closing prices
    monthly_data = data["Close"].resample("M").last()

    st.subheader("Last 5 Years Data")

    st.dataframe(
        monthly_data.tail(12)
    )

    # --------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------

    train_size = int(len(monthly_data) * 0.8)

    train = monthly_data[:train_size]
    test = monthly_data[train_size:]

    # --------------------------------
    # ARIMA MODEL
    # --------------------------------

    try:

        model = ARIMA(
            train,
            order=(5, 1, 0)
        )

        model_fit = model.fit()

        predictions = model_fit.forecast(
            steps=len(test)
        )

        rmse = sqrt(
            mean_squared_error(
                test,
                predictions
            )
        )

        st.metric(
            "RMSE",
            round(rmse, 2)
        )

        # --------------------------------
        # FINAL MODEL ON FULL DATA
        # --------------------------------

        final_model = ARIMA(
            monthly_data,
            order=(5, 1, 0)
        )

        final_fit = final_model.fit()

        # Months until June 2027

        last_date = monthly_data.index[-1]

        target_date = pd.Timestamp("2027-06-30")

        forecast_months = (
            (target_date.year - last_date.year) * 12
            + (target_date.month - last_date.month)
        )

        if forecast_months <= 0:
            forecast_months = 12

        forecast_result = final_fit.get_forecast(
            steps=forecast_months
        )

        forecast_values = forecast_result.predicted_mean

        conf_int = forecast_result.conf_int()

        forecast_index = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=forecast_months,
            freq="M"
        )

        forecast_df = pd.DataFrame({
            "Forecast Price": forecast_values.values,
            "Lower CI": conf_int.iloc[:, 0].values,
            "Upper CI": conf_int.iloc[:, 1].values
        }, index=forecast_index)

        # --------------------------------
        # HISTORICAL + FORECAST GRAPH
        # --------------------------------

        st.subheader("Historical and Forecasted Prices")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            monthly_data.index,
            monthly_data.values,
            label="Historical Data"
        )

        ax.plot(
            forecast_df.index,
            forecast_df["Forecast Price"],
            label="Forecast"
        )

        ax.fill_between(
            forecast_df.index,
            forecast_df["Lower CI"],
            forecast_df["Upper CI"],
            alpha=0.2
        )

        ax.set_title(
            f"{ticker_input.upper()} Forecast till June 2027"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        ax.legend()

        st.pyplot(fig)

        # --------------------------------
        # FORECAST TABLE
        # --------------------------------

        st.subheader("Forecast Values")

        display_df = forecast_df.copy()

        display_df.index.name = "Date"

        st.dataframe(
            display_df.style.format({
                "Forecast Price": "₹{:,.2f}",
                "Lower CI": "₹{:,.2f}",
                "Upper CI": "₹{:,.2f}"
            })
        )

        # --------------------------------
        # DOWNLOAD
        # --------------------------------

        csv = display_df.to_csv()

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker_input}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Model Error: {e}"
        )
