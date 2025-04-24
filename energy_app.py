import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load the trained model
model = joblib.load("best_model_energy.pkl")  # Your trained Random Forest model

# Load the original dataset to preserve the feature pipeline
energy_data = pd.read_csv("energy_new.csv", parse_dates=["Datetime"], index_col="Datetime")

# Define columns used in model
columns = ['Global_reactive_power', 'Voltage',
       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3', 'Rolling_Mean_15', 'Rolling_Mean_16',
       'Rolling_Mean_17', 'Rolling_Mean_18', 'Rolling_Mean_19',
       'Rolling_Mean_20', 'Rolling_Mean_21', 'Rolling_Mean_22',
       'Rolling_Mean_23', 'Rolling_Mean_24', 'Rolling_Mean_25',
       'Rolling_Mean_26', 'Rolling_Mean_27', 'Rolling_Mean_28',
       'Rolling_Mean_29', 'Rolling_Mean_30', 'Rolling_Mean_31',
       'Rolling_Mean_32', 'Rolling_Mean_33', 'Rolling_Mean_34',
       'Rolling_Mean_35', 'Rolling_Mean_36', 'Rolling_Mean_37',
       'Rolling_Mean_38', 'Rolling_Mean_39', 'Rolling_Mean_40',
       'Rolling_Mean_41', 'Rolling_Mean_42', 'Rolling_Mean_43',
       'Rolling_Mean_44', 'Rolling_Mean_45', 'Rolling_Mean_46',
       'Rolling_Mean_47', 'Rolling_Mean_48', 'Rolling_Mean_49',
       'Rolling_Mean_50', 'Lag_1', 'Lag_7', 'Lag_14', 'Lag_21', 'Lag_30',
       'Lag_60', 'dayofweek', 'month', 'day', 'is_weekend', 'is_holiday']

st.title("Energy Forecasting App - Random Forest Model")
st.markdown("Enter a date range to receive energy consumption predictions")

# Date range input
date_start = st.date_input("Start Date", datetime(2009, 1, 1))
date_end = st.date_input("End Date", datetime(2009, 1, 30))

if st.button("Predict"):
    try:
        date_range = pd.date_range(start=date_start, end=date_end, freq='D')
        
        # Filter the dataset
        data_filtered = energy_data.loc[energy_data.index.normalize().isin(date_range)]
        
        # Prepare features
        X_input = data_filtered[columns]

        # Apply scaler if used
        # X_input = scaler.transform(X_input)

        # Make predictions
        predictions = model.predict(X_input)

        # Output
        result_df = pd.DataFrame({
            "Date": data_filtered.index,
            "Predicted Global Active Power": predictions
        })

        st.subheader("Forecast Results")
        st.dataframe(result_df.set_index("Date"))

        st.line_chart(result_df.set_index("Date"))

    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")
