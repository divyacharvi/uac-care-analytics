import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="UAC Care Analytics Dashboard",
    layout="wide"
)

# ======================================
# TITLE
# ======================================

st.title("System Capacity & Care Load Analytics Dashboard")

st.markdown("""
This dashboard analyzes healthcare and shelter system load
for the Unaccompanied Alien Children (UAC) Program.
""")

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv("data/HHS_Unaccompanied_Alien_Children_Program.csv")

# ======================================
# DATA CLEANING
# ======================================

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Clean numeric columns
numeric_columns = [
    'Children apprehended and placed in CBP custody',
    'Children in CBP custody',
    'Children transferred out of CBP custody',
    'Children in HHS Care',
    'Children discharged from HHS Care'
]

for col in numeric_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '')
    )

    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove missing values
df = df.dropna()

# ======================================
# CREATE METRICS
# ======================================

df['Total System Load'] = (
    df['Children in CBP custody'] +
    df['Children in HHS Care']
)

df['Net Intake'] = (
    df['Children transferred out of CBP custody'] -
    df['Children discharged from HHS Care']
)

# ======================================
# SIDEBAR FILTERS
# ======================================

st.sidebar.header("Filters")

start_date = st.sidebar.date_input(
    "Start Date",
    value=df['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=df['Date'].max()
)

# Filter dataset
filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# ======================================
# KPI CARDS
# ======================================

current_load = int(filtered_df['Total System Load'].iloc[-1])

avg_net_intake = round(
    filtered_df['Net Intake'].mean(),
    2
)

peak_load = int(
    filtered_df['Total System Load'].max()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Total Load",
        current_load
    )

with col2:
    st.metric(
        "Average Net Intake",
        avg_net_intake
    )

with col3:
    st.metric(
        "Peak System Load",
        peak_load
    )

# ======================================
# CHART 1
# TOTAL SYSTEM LOAD
# ======================================

st.subheader("Total System Load Over Time")

fig1 = px.line(
    filtered_df,
    x='Date',
    y='Total System Load',
    title='Total System Load Trend'
)

fig1.update_layout(
    template='plotly_dark'
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ======================================
# CHART 2
# CBP vs HHS LOAD
# ======================================

st.subheader("CBP vs HHS Care Load")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Children in CBP custody'],
        mode='lines',
        name='CBP Custody'
    )
)

fig2.add_trace(
    go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Children in HHS Care'],
        mode='lines',
        name='HHS Care'
    )
)

fig2.update_layout(
    template='plotly_dark',
    title='CBP vs HHS Care Load'
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ======================================
# CHART 3
# NET INTAKE PRESSURE
# ======================================

st.subheader("Net Intake Pressure")

fig3 = px.bar(
    filtered_df,
    x='Date',
    y='Net Intake',
    title='Net Intake Pressure'
)

fig3.update_layout(
    template='plotly_dark'
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ======================================
# FUTURE LOAD PREDICTION
# ======================================

st.header("Future System Load Prediction")

prediction_df = filtered_df[
    ['Date', 'Total System Load']
].copy()

prediction_df = prediction_df.dropna()

# Convert dates into numbers
prediction_df['Days'] = np.arange(
    len(prediction_df)
)

X = prediction_df[['Days']]

y = prediction_df['Total System Load']

# Train Linear Regression model
model = LinearRegression()

model.fit(X, y)

# Future predictions
future_days = np.arange(
    len(prediction_df) + 90
)

future_predictions = model.predict(
    future_days.reshape(-1, 1)
)

# Future dates
future_dates = pd.date_range(
    start=prediction_df['Date'].min(),
    periods=len(future_days)
)

# Prediction chart
forecast_fig = go.Figure()

# Actual values
forecast_fig.add_trace(
    go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Total System Load'],
        mode='lines',
        name='Actual Load'
    )
)

# Predicted values
forecast_fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        name='Predicted Load'
    )
)

forecast_fig.update_layout(
    title='90-Day Future System Load Prediction',
    xaxis_title='Date',
    yaxis_title='Predicted Load',
    template='plotly_dark'
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)

# ======================================
# PREDICTION TABLE
# ======================================

st.subheader("Prediction Data")

prediction_table = pd.DataFrame({
    'Date': future_dates[-20:],
    'Predicted Load': future_predictions[-20:]
})

st.dataframe(prediction_table)

# ======================================
# DATA PREVIEW
# ======================================

st.subheader("Dataset Preview")

st.dataframe(filtered_df.head(20))