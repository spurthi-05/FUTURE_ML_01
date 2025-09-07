
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("🧠 AI-Powered Sales Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file with Date and Sales columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample dataset (sales_data.csv)")
    df = pd.read_csv("sales_data.csv")

df['Date'] = pd.to_datetime(df['Date'])
st.write("Preview of data:", df.head())

# Prophet expects 'ds' and 'y'
df = df.rename(columns={"Date":"ds","Sales":"y"})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

st.subheader("Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
