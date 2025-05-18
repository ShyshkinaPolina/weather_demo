import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.express as px
from weather_data_processing import preprocess_new_data

df = pd.read_csv("data/weatherAUS.csv")
model_all = joblib.load('models/aussie_rain.joblib')
input_cols = model_all['input_cols']
encoder = model_all['encoder']
scaler = model_all['scaler']
imputer = model_all['imputer']
model = model_all['model']

def predict(data):
    pred, prob = model_all['model'].predict(input_df)
    return pred, prob

st.title('Прогноз погоди в Австралії')
st.markdown('Це проста модель, що показує чи буде дощ в Австралії та ймовірність цього пронозу')
        
st.sidebar.header("Уведіть дані про погоду")

location = st.sidebar.selectbox("Локація", sorted(df['Location'].dropna().unique()))
min_temp = st.sidebar.slider("Мін. температура (°C)", -10.0, 50.0, 15.0)
max_temp = st.sidebar.slider("Макс. температура (°C)", -10.0, 50.0, 25.0)
rainfall = st.sidebar.slider("Опади (мм)", 0.0, 50.0, 0.0)
evaporation = st.sidebar.slider("Випаровуваність", -10.0, 50.0, 15.0)
sunshine = st.sidebar.slider("Сонячні години", 0.0, 14.0, 8.0)
wind_gust_dir = st.sidebar.selectbox("Напрямок вітру", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
wind_gust_speed = st.sidebar.slider("Швидкість вітру", 0.0, 30.0, 5.0)
wind_dir_9am = st.sidebar.selectbox("Напрямок вітру в 9:00", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
wind_dir_3pm = st.sidebar.selectbox("Напрямок вітру в 15:00", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
wind_speed_9am = st.sidebar.slider("Швидкість вітру в 9:00", 0.0, 30.0, 5.0)
wind_speed_3pm = st.sidebar.slider("Швидкість вітру в 15:00", 0.0, 30.0, 5.0)
humidity_9am = st.sidebar.slider("Відносна вологість в 9:00", 0.0, 100.0, 50.0)
humidity_3pm = st.sidebar.slider("Відносна вологість 15:00", 0.0, 100.0, 50.0)
pressure_9am = st.sidebar.slider("Атмосферний тиск в (гПа) 9:00", 0.0, 1200.0, 1013.0)
pressure_3pm = st.sidebar.slider("Атмосферний тиск в (гПа) 15:00", 0.0, 1200.0, 1013.0)
cloud9am = st.sidebar.slider("Хмарність в 9:00", 0.0, 8.0, 4.0)
cloud3pm = st.sidebar.slider("Хмарність в 15:00", 0.0, 8.0, 4.0)
temp9am = st.sidebar.slider("Температура в 9:00", -20.0, 40.0, 18.0)
temp3pm = st.sidebar.slider("Температура в 15:00", -20.0, 45.0, 24.0)
rain_today = st.sidebar.selectbox("Чи йшов дощ сьогодні?", ["No", "Yes"])

input_dict = {
    'Location': [location],
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'WindGustDir': [wind_gust_dir],
    'WindGustSpeed': [wind_gust_speed],
    'WindDir9am': [wind_dir_9am],
    'WindDir3pm': [wind_dir_3pm],
    'WindSpeed9am': [wind_speed_9am],
    'WindSpeed3pm': [wind_speed_3pm],
    'Humidity9am': [humidity_9am],
    'Humidity3pm': [humidity_3pm],
    'Pressure9am': [pressure_9am],
    'Pressure3pm': [pressure_3pm],
    'Cloud9am': [cloud9am],
    'Cloud3pm': [cloud3pm],
    'Temp9am': [temp9am],
    'Temp3pm': [temp3pm],
    'RainToday': [rain_today]
}

input_df = pd.DataFrame(input_dict)
data = preprocess_new_data(input_df, input_cols, encoder, scaler, imputer)

if st.button("Передбачити дощ"):
    
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][list(model.classes_).index(pred)]
    if pred == "Yes":
        st.error(f"Завтра буде дощ з вірогідністю {prob:.2f}")
    else:
        st.success(f"Завтра не буде дощу з вірогідністю {prob:.2f}")

st.subheader("Історичні дані")

if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']) is False:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

st.altair_chart(
    alt.Chart(df.dropna(subset=['Date', 'MaxTemp']))
    .mark_line()
    .encode(x='Date:T', y='MaxTemp:Q', tooltip=['Date', 'MaxTemp'])
    .properties(title="Максимальна температура по датам", height=300),
    use_container_width=True
)

st.subheader("🌧️ Розподіл опадів")
st.bar_chart(df['Rainfall'].dropna())
st.subheader("☔ Статистика дощів")
rain_counts = df['RainTomorrow'].value_counts()
fig = px.pie(names=rain_counts.index, values=rain_counts.values, title="RainTomorrow (Yes/No)")
st.plotly_chart(fig)
