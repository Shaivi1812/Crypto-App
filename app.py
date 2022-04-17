import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2015-01-01'
end = '2021-12-31'

st.title("Cryptocurrency Prediction")

user_input = st.text_input('Enter cryptocurrency', 'BTC-USD')
df = data.DataReader(user_input,'yahoo', start ,end)

#Describing Data
st.subheader('Data from 2015 - 2021')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 MA')
ma100 = df.Close.rolling(100).mean
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 MA and 200 MA')
ma100 = df.Close.rolling(100).mean
ma200 = df.Close.rolling(200).mean
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


