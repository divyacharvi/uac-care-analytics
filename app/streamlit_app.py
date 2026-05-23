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
Interactive analytics dashboard for monitoring
Unaccompanied Alien Children (UAC) healthcare
and shelter system load.
""")

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv(
    "data/HHS_Unaccompanied_Alien_Children_Program.csv"
)

# ======================================
# SHOW COLUMN NAMES
# ======================================

st.subheader("Dataset Columns")

st.write(df.columns.tolist())

# ======================================
# CLEAN DATA
# ======================================

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Detect numeric columns automatically
numeric_cols = df.columns.drop('Date')

# Clean all numeric columns
for col in numeric_cols:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '')
    )

    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

# Remove missing values
df = df.dropna()

# ======================================
# CREATE METRICS
# ======================================

# Use available columns dynamically
cbp_col = 'Children in CBP custody'

hhs_col = 'Children in HHS Care'

transfer_col = 'Children transferred out of CBP custody'

discharge_col = 'Children discharged from HHS Care'

df['Total System Load'] = (
    df[cbp_col] + df[hhs_col]
)

df['Net Intake'] = (
    df[transfer_col] - df[discharge_col]
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

filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# ======================================
# KPI CARDS
# ======================================

current_load = int(
    filtered_df['Total System Load'].iloc[-1]
)

avg_net = round(
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
        avg_net
    )

with col3:
    st.metric(
        "Peak System Load",
        peak_load
    )

# ======================================
# TOTAL LOAD CHART
# ======================================

st.subheader("Total System Load Over Time")

fig1 = px.line(
    filtered_df,
    x='Date',
    y='Total System Load'
)

fig1.update_layout(
    template='plotly_dark'
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ======================================
# CBP vs HHS
# ======================================

st.subheader("CBP vs HHS Care Load")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df[cbp_col],
        mode='lines',
        name='CBP Custody'
    )
)

fig2.add_trace(
    go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df[hhs_col],
        mode='lines',
        name='HHS Care'
    )
)

fig2.update_layout(
    template='plotly_dark'
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ======================================
# NET INTAKE
# ======================================

st.subheader("Net Intake Pressure")

fig3 = px.bar(
    filtered_df,
    x='Date',
    y='Net Intake'
)

fig3.update_layout(
    template='plotly_dark'
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ======================================
# FUTURE PREDICTION
# ======================================

st.header("Future System Load Prediction")

prediction_df = filtered_df[
    ['Date', 'Total System Load']
].copy()

prediction_df['Days'] = np.arange(
    len(prediction_df)
)

X = prediction_df[['Days']]

y = prediction_df['Total System Load']

model = LinearRegression()

model.fit(X, y)

future_days = np.arange(
    len(prediction_df) + 90
)

future_predictions = model.predict(
    future_days.reshape(-1, 1)
)

future_dates = pd.date_range(
    start=prediction_df['Date'].min(),
    periods=len(future_days)
)

forecast_fig = go.Figure()

forecast_fig.add_trace(
    go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Total System Load'],
        mode='lines',
        name='Actual Load'
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        name='Predicted Load'
    )
)

forecast_fig.update_layout(
    template='plotly_dark',
    title='90-Day Future Prediction'
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)

# ======================================
# DATA TABLE
# ======================================

st.subheader("Dataset Preview")

st.dataframe(
    filtered_df.head(20)
)