import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2016-01-01'
end = '2022-04-09'

# st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# st.markdown("""
# <nav class="navbar fixed navbar-expand-lg navbar-light" style="background-color: #3498DB;">
#   <a class="navbar-brand" href="#" target="_blank">Cryptocurrency Price Prediction</a>
#   <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
#     <span class="navbar-toggler-icon"></span>
#   </button>
#   <div class="collapse navbar-collapse" id="navbarNav">
#     <ul class="navbar-nav">
#       <li class="nav-item active">
#         <a class="nav-link disabled" href="\Home.py" >Home </a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="#" target="_blank">Graphs<span class="sr-only">(current)</span></a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="" target="_blank">Predictions</a>
#       </li>
#     </ul>
#   </div>
# </nav>
# """, unsafe_allow_html=True)


nav = st.sidebar.radio("Navigation",["Home","About","Trend Prediction","Forecast"])
if nav == "Home":
  st.write("Home")
  st.markdown("""
    <img width="100%" height="380" src="https://images.unsplash.com/photo-1631897641179-82f97def3992?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTEzfHxjcnlwdG9jdXJyZW5jeXxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=500&q=60" alt="Cryptocurrency Price Prediction" />
    <br>
  """, unsafe_allow_html=True)
  st.markdown("<h1 style='text-align: center; font-style: italic '>Crypto Application</h1>", unsafe_allow_html=True)
  st.markdown("<h4 style='text-align: center;'>Help us help you make profits!</h4>", unsafe_allow_html=True)

  st.markdown("""
    <img width="100%" height="380" src="https://images.unsplash.com/photo-1623920813183-ef1b2eb4a562?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTQwfHxjcnlwdG9jdXJyZW5jeXxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=500&q=60" alt="Cryptocurrency Price Prediction" />
    <br>
  """, unsafe_allow_html=True)  
if nav == "About":
  st.title("Crypto Application")
  st.header("What is the application about?")
  st.markdown("""
    <img src="https://images.unsplash.com/photo-1631603090989-93f9ef6f9d80?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8N3x8Y3J5cHRvY3VycmVuY3l8ZW58MHx8MHx8&auto=format&fit=crop&w=500&q=60" alt="Cryptocurrency Price Prediction" />
    <br>
    <br>
  """, unsafe_allow_html=True)
  st.text("""


The application helps in predicting Cryptocurrency Prices for 
3 famous Cryptocurrencies. The graph pages first displays the
data with all its fields from the year 2016 to the year 2022.
Next, it displays various graphs as the result based on the 
calculations that we have performed. The graphs are:
1. Closing Price vs Time Chart
2. Closing Price vs Time chart with 100 MA
3. Closing Price vs Time chart with 100 MA and 200 MA
4. Actual Price vs Predicted Price  
  """)
  st.header("Why do we need it?")
  st.markdown("""
    <img src="https://images.unsplash.com/photo-1640826514546-7d2eab70a4e5?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTR8fGNyeXB0b2N1cnJlbmN5fGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60" alt="Cryptocurrency Price Prediction" />
    <br>
    <br>
  """, unsafe_allow_html=True)
  st.text("""
    ​Cryptocurrency markets have a volatile nature due to the 
    absence of any central authority which is why good predictions 
    can be helpful to decide on the investment. Cryptocurrency 
    is a more secure alternative to the physical currency that 
    we currently use.​ An increasing number of e-commerce business 
    owners are incorporating virtual currency into their operations.​ 
    Because of the increased use of cryptocurrency in recent times, 
    it is necessary to have an application which predicts the price 
    of cryptocurrencies like Bitcoin with good accuracy.



  """)
if nav == "Trend Prediction":
  st.write("Trend Prediction")
  st.title("Cryptocurrency Prediction")

  user_input = st.selectbox('Select cryptocurrency', ['BTC-USD','ETH-USD','LTC-USD'],index = 0)
  df = data.DataReader(user_input,'yahoo', start ,end)

  #Describing Data
  st.subheader('Data from 2016 - 2022')
  st.write(df.describe())

  #Visualizations
  st.subheader('Closing Price vs Time chart')
  fig = plt.figure(figsize = (12,6))
  plt.plot(df.Close)
  st.pyplot(fig)

  st.subheader('Closing Price vs Time chart with 100 MA')
  ma100 = df.Close.rolling(100).mean()
  fig = plt.figure(figsize = (12,6))
  plt.plot(ma100, label = "100 MA")
  plt.plot(df.Close, label = "Closing Price")
  plt.legend()
  st.pyplot(fig)

  st.subheader('Closing Price vs Time chart with 100 MA and 200 MA')
  ma100 = df.Close.rolling(100).mean()
  ma200 = df.Close.rolling(200).mean()
  fig = plt.figure(figsize = (12,6))
  plt.plot(ma100, 'r',label = "100 MA")
  plt.plot(ma200, 'g', label = "200 MA")
  plt.plot(df.Close, 'b', label = "Closing price")
  plt.legend()
  st.pyplot(fig)

  # Splitting Data into Training and Testing

  data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
  data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

  print(data_training.shape)
  print(data_testing.shape)

  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler(feature_range = (0,1))

  data_training_array = scaler.fit_transform(data_training)

  #load model
  model = load_model('keras_model.h5')
  past_100_days = data_training.tail(100)
  final_df = past_100_days.append(data_testing, ignore_index = True)
  input_data = scaler.fit_transform(final_df)

  x_test = []
  y_test = []

  for i in range(100, input_data.shape[0]):
      x_test.append(input_data[i-100: i])
      y_test.append(input_data[i, 0])

  x_test, y_test = np.array(x_test), np.array(y_test)
  # Making Predictions

  y_predicted = model.predict(x_test)
  scaler = scaler.scale_
  scale_factor = 1/scaler[0]
  y_predicted = y_predicted * scale_factor
  y_test = y_test * scale_factor

  st.subheader('Actual Price vs Predicted Price')
  fig2 = plt.figure(figsize = (12,6))
  plt.plot(y_test, 'b', label = 'Original Price')
  plt.plot(y_predicted, 'r', label = 'Predicted Price')
  plt.xlabel('Time')
  plt.ylabel('Price') 
  plt.legend()
  st.pyplot(fig2)
if nav == "Forecast":
  st.markdown("<h1 style='text-align: center;'>Forecasts</h1>", unsafe_allow_html=True)
  st.markdown("<h4 style='text-align: center;'>Ethereum - LSTM</h4>", unsafe_allow_html=True)
  st.markdown("""
    <img width="690px" src="
    https://i.ibb.co/VTCScmq/Screenshot-1.png
    " alt="Cryptocurrency Price Prediction" />
    <br>
    <br>
  """, unsafe_allow_html=True)

  st.markdown("<h4 style='text-align: center;'>Bitcoin - KNN</h4>", unsafe_allow_html=True)
  st.markdown("""
    <img width="700px" src="https://i.ibb.co/sPPXL4x/Bitcoin-BTC-USD.jpg"
    alt="Cryptocurrency Price Prediction" />
    <br>
    <br>
  """, unsafe_allow_html=True)

  st.markdown("<h4 style='text-align: center;'>Litecoin - LSTM </h4>", unsafe_allow_html=True)
  st.markdown("""
    <img width="700px" src="https://i.ibb.co/VvTRXv0/Litecoin.jpg"
    alt="Cryptocurrency Price Prediction" />
    <br>
    <br>
  """, unsafe_allow_html=True)






