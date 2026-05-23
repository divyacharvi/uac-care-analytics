import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

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
This dashboard analyzes healthcare and shelter system load for the
Unaccompanied Alien Children (UAC) Program.
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
    'Children in HHS Care',
    'New Intakes',
    'Discharges',
    'Total System Load'
]

for col in numeric_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '')
        .replace('nan', None)
    )

    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove missing values
df = df.dropna()

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

# Filter dataframe
filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# ======================================
# KPI METRICS
# ======================================

current_load = int(filtered_df['Total System Load'].iloc[-1])

avg_intake = round(filtered_df['New Intakes'].mean(), 2)

peak_load = int(filtered_df['Total System Load'].max())

# Display KPIs
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Current Total Load",
        value=current_load
    )

with col2:
    st.metric(
        label="Average Net Intake",
        value=avg_intake
    )

with col3:
    st.metric(
        label="Peak System Load",
        value=peak_load
    )

# ======================================
# SYSTEM LOAD GRAPH
# ======================================

st.subheader("Total System Load Over Time")

fig = px.line(
    filtered_df,
    x='Date',
    y='Total System Load',
    title='Total System Load Trend'
)

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Total System Load"
)

st.plotly_chart(fig, use_container_width=True)

# ======================================
# INTAKES VS DISCHARGES
# ======================================

st.subheader("New Intakes vs Discharges")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['New Intakes'],
        mode='lines',
        name='New Intakes'
    )
)

fig2.add_trace(
    go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Discharges'],
        mode='lines',
        name='Discharges'
    )
)

fig2.update_layout(
    template="plotly_dark",
    title="Intakes vs Discharges",
    xaxis_title="Date",
    yaxis_title="Count"
)

st.plotly_chart(fig2, use_container_width=True)

# ======================================
# FORECASTING SECTION
# ======================================

st.header("Future System Load Prediction")

# Prepare forecasting data
forecast_df = filtered_df[['Date', 'Total System Load']].copy()

forecast_df.columns = ['ds', 'y']

forecast_df = forecast_df.dropna()

# Prophet model
model = Prophet()

model.fit(forecast_df)

# Future prediction
future = model.make_future_dataframe(periods=90)

forecast = model.predict(future)

# Forecast graph
forecast_fig = go.Figure()

# Actual
forecast_fig.add_trace(
    go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['y'],
        mode='lines',
        name='Actual Load'
    )
)

# Prediction
forecast_fig.add_trace(
    go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Load'
    )
)

forecast_fig.update_layout(
    template="plotly_dark",
    title="90-Day Future System Load Forecast",
    xaxis_title="Date",
    yaxis_title="Predicted Load"
)

st.plotly_chart(forecast_fig, use_container_width=True)

# ======================================
# FORECAST TABLE
# ======================================

st.subheader("Forecast Data")

forecast_table = forecast[['ds', 'yhat']].tail(20)

forecast_table.columns = ['Date', 'Predicted Load']

st.dataframe(forecast_table)

# ======================================
# RAW DATA
# ======================================

st.subheader("Dataset Preview")

st.dataframe(filtered_df.head(20))