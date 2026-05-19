import streamlit as st
import pandas as pd
import plotly.express as px

# PAGE SETTINGS

st.set_page_config(
    page_title="UAC Care Analytics Dashboard",
    layout="wide"
)

# LOAD DATASET

df = pd.read_csv("data/HHS_Unaccompanied_Alien_Children_Program.csv")

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])

# Clean HHS column
df['Children in HHS Care'] = (
    df['Children in HHS Care']
    .astype(str)
    .str.replace(',', '')
    .astype(float)
)

# Remove missing values
df = df.dropna()

# CREATE METRICS

df['Total_System_Load'] = (
    df['Children in CBP custody'] +
    df['Children in HHS Care']
)

df['Net_Intake'] = (
    df['Children transferred out of CBP custody'] -
    df['Children discharged from HHS Care']
)

# SIDEBAR FILTERS

st.sidebar.header("Filters")

start_date = st.sidebar.date_input(
    "Start Date",
    df['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    df['Date'].max()
)

filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# DASHBOARD TITLE

st.title("System Capacity & Care Load Analytics Dashboard")

st.markdown("""
This dashboard analyzes healthcare and shelter system load
for the Unaccompanied Alien Children (UAC) Program.
""")

# KPI CARDS

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Total Load",
        int(filtered_df['Total_System_Load'].iloc[-1])
    )

with col2:
    st.metric(
        "Average Net Intake",
        round(filtered_df['Net_Intake'].mean(), 2)
    )

with col3:
    st.metric(
        "Peak System Load",
        int(filtered_df['Total_System_Load'].max())
    )

# CHART 1
# TOTAL SYSTEM LOAD

fig1 = px.line(
    filtered_df,
    x='Date',
    y='Total_System_Load',
    title='Total System Load Over Time'
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# CHART 2
# CBP vs HHS

fig2 = px.line(
    filtered_df,
    x='Date',
    y=[
        'Children in CBP custody',
        'Children in HHS Care'
    ],
    title='CBP vs HHS Care Load'
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# CHART 3
# NET INTAKE

fig3 = px.bar(
    filtered_df,
    x='Date',
    y='Net_Intake',
    title='Net Intake Pressure'
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# DATA PREVIEW

st.subheader("Dataset Preview")

st.dataframe(filtered_df.head(20))