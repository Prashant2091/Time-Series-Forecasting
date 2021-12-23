# import pandas as pd
# from PIL import Image
# import streamlit as st
# from datetime import date
# import yfinance as yf
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
# from fbprophet.diagnostics import performance_metrics
# from fbprophet.diagnostics import cross_validation
# from fbprophet.plot import plot_cross_validation_metric
# from plotly import graph_objs as go
# from statsmodels.tsa.arima_model import ARIMA

# START = "2015-01-01"
# TODAY = date.today().strftime("%Y-%m-%d")


# st.title('CO2 EMISSION FORECASTING')
# n_years = st.sidebar.slider('Years of Prediction:', 1, 10)
# period = n_years * 365

# @st.cache
# def load_data(ticker):
#     data = yf.download(ticker, START, TODAY)
#     data.reset_index(inplace=True)
#     return data


# # data_load_state = st.text('Loading data...')
# # data = load_data(selected_stock)
# # data_load_state.text('Loading data... done!')

# data = pd.read_excel("CO2 dataset.xlsx")
# st.subheader('Raw data')
# st.write(data.tail())

# # Plot raw data
# def plot_raw_data():
# 	fig = go.Figure()
# 	fig.add_trace(go.Scatter(x=data['Year'], y=data['CO2'], name="CO2 Emission"))
# 	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
# 	st.plotly_chart(fig)

# plot_raw_data()

# # Predict forecast with Prophet.
# df_train = data[['Year','CO2']]
# df_train = df_train.rename(columns={"Year": "ds", "CO2": "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# # Show and plot forecast
# st.subheader('Forecast data')
# st.write(forecast.tail())
    
# st.write(f'Forecast plot for {n_years} years')
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1)

# # st.write("Forecast components")
# # fig2 = m.plot_components(forecast)
# # st.write(fig2)

##### SWAPNIL #####

import numpy as np
import pandas as pd
import datetime
import warnings
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
import streamlit as st
from plotly import graph_objs as go
import openpyxl


st.title('CO2 EMISSION FORECASTING')
def user_input_features():
    Years = st.number_input('Years of Prediction:', 1, 20)
    return Years 

df = user_input_features()+1
st.subheader('User Input parameters')
st.write(df)

import pandas as pd 
from datetime import datetime


def dateparse(dates):
    return datetime.strptime(dates, '%Y')


data = pd.read_excel("CO2 dataset.xlsx",
                           parse_dates=['Year'],
                           index_col='Year')#engine='openpyxl



#Model Building
future_dates=[data.index[-1]+ DateOffset(years=x)for x in range(0,df)]
future_data=pd.DataFrame(index=future_dates[1:],columns=data.columns)

final_arima = ARIMA(data['CO2'],order = (3,1,4))
final_arima = final_arima.fit()

final_arima.fittedvalues.tail()

future_data['CO2'] = final_arima.predict(start = 215, end = 225, dynamic= True) 


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data.index,y=data['CO2'], name="CO2 Emission"))
	st.plotly_chart(fig)


plot_raw_data()

future_data.tail(df)

st.subheader(f'Forecasting for {df-1} year')
st.write(future_data.tail(df))


# Plot raw data
def plot_result_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=future_data.index,y=future_data['CO2'], name="CO2 Emission"))
	st.plotly_chart(fig)


plot_result_data()