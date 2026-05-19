import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATASET

df = pd.read_csv("../data/HHS_Unaccompanied_Alien_Children_Program.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# DATA CLEANING

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Clean HHS column
df['Children in HHS Care'] = (
    df['Children in HHS Care']
    .astype(str)
    .str.replace(',', '')
    .astype(float)
)

# Sort by date
df = df.sort_values('Date')

# Remove missing values
df = df.dropna()

# CREATE METRICS

# Total System Load
df['Total_System_Load'] = (
    df['Children in CBP custody'] +
    df['Children in HHS Care']
)

# Net Intake
df['Net_Intake'] = (
    df['Children transferred out of CBP custody'] -
    df['Children discharged from HHS Care']
)

# Growth Rate
df['Growth_Rate'] = (
    df['Total_System_Load']
    .pct_change() * 100
)

# Backlog
df['Backlog'] = (
    df['Net_Intake']
    .cumsum()
)

# Rolling 7-Day Average
df['Rolling_7D_Load'] = (
    df['Total_System_Load']
    .rolling(7)
    .mean()
)

# KPI SUMMARY

print("\n===== KPI SUMMARY =====")

current_load = df['Total_System_Load'].dropna().iloc[-1]

avg_net_intake = df['Net_Intake'].mean()

volatility = df['Total_System_Load'].std()

backlog = df['Backlog'].iloc[-1]

print("Current Total Load:", int(current_load))

print("Average Net Intake:", round(avg_net_intake, 2))

print("Volatility Index:", round(volatility, 2))

print("Backlog:", round(backlog, 2))

# SAVE CLEANED DATA

df.to_csv(
    "../outputs/cleaned_uac_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

# VISUALIZATION 1
# TOTAL SYSTEM LOAD

plt.figure(figsize=(12,6))

plt.plot(
    df['Date'],
    df['Total_System_Load']
)

plt.title("Total System Load Over Time")
plt.xlabel("Date")
plt.ylabel("Children Under Care")

plt.show()

# VISUALIZATION 2
# CBP vs HHS

plt.figure(figsize=(12,6))

plt.plot(
    df['Date'],
    df['Children in CBP custody'],
    label='CBP Custody'
)

plt.plot(
    df['Date'],
    df['Children in HHS Care'],
    label='HHS Care'
)

plt.title("CBP vs HHS Care Load")
plt.xlabel("Date")
plt.ylabel("Children")

plt.legend()

plt.show()

# VISUALIZATION 3
# NET INTAKE PRESSURE

plt.figure(figsize=(12,6))

plt.bar(
    df['Date'],
    df['Net_Intake']
)

plt.title("Net Intake Pressure")
plt.xlabel("Date")
plt.ylabel("Net Intake")

plt.show()

# VISUALIZATION 4
# ROLLING AVERAGE

plt.figure(figsize=(12,6))

plt.plot(
    df['Date'],
    df['Rolling_7D_Load']
)

plt.title("7-Day Rolling Average of System Load")
plt.xlabel("Date")
plt.ylabel("Average Load")

plt.show()

# FORECASTING

from prophet import Prophet

forecast_df = df[['Date', 'Total_System_Load']]

forecast_df.columns = ['ds', 'y']

# Create model
model = Prophet()

# Train model
model.fit(forecast_df)

# Future dates
future = model.make_future_dataframe(periods=30)

# Predict
forecast = model.predict(future)

# Forecast plot
model.plot(forecast)

plt.title("30-Day Forecast of System Load")

plt.show()

print("\nProject Analysis Completed Successfully!")