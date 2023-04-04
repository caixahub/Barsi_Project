#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:00:59 2023

@author: elcio

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import seaborn as sns
import yfinance as yf
import tensorflow as tf
from tensorflow import keras

from datetime import datetime,date

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.metrics import mean_squared_error
from keras.layers  import LSTM

# Needed to help our plots look cleaner with plotly 

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

#Declarando as açoes para baixar com o "yf" 

tickers = ['BBAS3.SA', 'IRBR3.SA', 'KLBN11.SA', 'SUZB3.SA', 'UNIP6.SA', 'TASA4.SA', 'TAEE11.SA']
#df = web.get_data_yahoo(tickers, start = '2019-01-01', end = '2022-12-31')

#df

fin_data = yf.download(tickers, start = '2017-01-01', period= '1d')

#fin_data = yf.download(tickers, start = '"2001-01-01"', end = '2021-08-17')

#fin_data.to_csv('/home/elcio/POS_Big_DATA/fin_data.csv')

#check the dimensions of the data
fin_data.shape
#view the first 5 rows of the data
fin_data.head()

#view the last 5 values of the data
fin_data.tail()

# #check if there are missing values for each type of stock

fin_data.isnull().sum()

# some missing values, and we will use the fillna method to resolve the missing values.

# handing missing values
fin_data.fillna(method='ffill', inplace = True) # use front fill method
fin_data.fillna(method='bfill', inplace = True) # use back fill method
#check to see if there are still any missing values
fin_data.isnull().sum()

#view descriptive statistics of adjusted close process of the stocks

fin_data[['Adj Close']].describe()

# view general info

fin_data.info()

# View the maximum close date of stocks
def max_close(stocks,df):
    """ This calculates and returns the maximum closing value of a specific stock"""
    return df['Close'][stocks].max() # computes and returns the maximum closing stock value

# test the above function with specific stocks
def test_max():
    """ This tests the max_close function"""
    for stocks in ['BBAS3.SA', 'IRBR3.SA', 'KLBN11.SA', 'SUZB3.SA', 'UNIP6.SA', 'TASA4.SA', 'TAEE11.SA']:
        print("Maxiumum Closing Value for {} is {}".format(stocks, max_close(stocks,fin_data)))

test_max()        
#if __name__ == "__main__" :
#    test_()


# Plot function for the Adjusted closing value
def plot_adj(df,title,stocks,y=0):
        ax = df['Adj Close'][stocks].plot(title=title, figsize=(16,8), ax=None)
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.axhline(y=y,color='black')
        ax.legend(stocks, loc='upper left')
        plt.show()
        
# View the plot of Adjusted close
stocks = ['BBAS3.SA', 'IRBR3.SA', 'KLBN11.SA', 'SUZB3.SA', 'UNIP6.SA', 'TASA4.SA', 'TAEE11.SA']
plot_adj(fin_data,"Adjusted Close Stock Prices",stocks)    

# calculate the mean volume for the stocks
def mean_vol(stocks,df):
    """ This calculates and returns the minimum volume of a specific stock"""
    return df['Volume'][stocks].mean() # computes and returns the minimum volume of a stock   
# test the above function with specific stocks
def test_mean():
    """ This tests the max_close function"""
    for stocks in ['BBAS3.SA', 'IRBR3.SA', 'KLBN11.SA', 'SUZB3.SA', 'UNIP6.SA', 'TASA4.SA', 'TAEE11.SA']:
        print("Mean Volume for {} is {}".format(stocks, mean_vol(stocks,fin_data)))
test_mean()  

# Plot function for the Adjusted closing value
def plot_adj(df,title,stocks,y=0):
        ax = df['Adj Close'][stocks].plot(title=title, figsize=(16,8), ax=None)
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.axhline(y=y,color='black')
        ax.legend(stocks, loc='upper left')
        plt.show()
        
# View the plot of Adjusted close
stocks = ['BBAS3.SA', 'IRBR3.SA', 'KLBN11.SA', 'SUZB3.SA', 'UNIP6.SA', 'TASA4.SA', 'TAEE11.SA']
plot_adj(fin_data,"Adjusted Close Stock Prices",stocks) 

#view only VIVOSA Adjusted close price
fin_data['Adj Close']['VIVT3.SA'] 

# view only VIVT3SA and BBAS3SA Adjusted close price
fin_data['Adj Close'][[ 'VIVT3.SA','BBAS3.SA' ]]

# view all stocks adjusted close prices for BBAS3.SA and VIVT3SA from Jan 2010 to Aug 2021
fin_data['Adj Close'].loc['2010-01-01':'2021-08-17', ['BBAS3.SA','VIVT3.SA' ]]

 # view all stocks adjusted close price from Jan 2010 to Aug 2021
fin_data['Adj Close'].loc['2010-01-01':'2021-08-17']

# create function to plot data
def plot_data1(df,stocks,title,ylabel='Stock Price',y=0):
    """This funtion plots stock prices"""
    ax = df.plot(title=title, figsize=(16,8), ax=None, fontsize=2)
    ax.set_xlabel("Date")
    ax.set_label(ylabel)
    ax.axhline(y=y,color='black')
    ax.legend(stocks, loc='upper left')
    plt.show()    
    
# create function to plot selected stocks
def selected_plot(df, columns,stocks, start_idx, end_idx):
    """This function plots specific stocks over a given date range"""
    plot_data1(df[columns].loc[start_idx:end_idx, stocks],stocks, title="Plot for selected Stocks")

# create function to plot data based on specific columns, symbols, and date ranges
def test_select():
    """This function plots stock data based on specific columns, symbols, and date ranges """
    # specify columns to plot and stock symbols
    columns = 'Adj Close'
    stocks = ['VIVT3.SA','BBAS3.SA']  
        
    # Get stock data
    df = fin_data

    # Slice and plot
    selected_plot(df, columns, stocks, '2010-01-01', '2021-08-17')
    
test_select()  # run the plot function 

""" Normalizing the data

We want to know how the different types of stocks went up and down with respect to the others. In order to do this, 
we will normalize the data. We do this by 
dividing the values of each column by day one to ensure that each stock starts with 
.
"""

def plot_data2(df,stocks,title='Stock Prices',ylabel="Stock Price",y=0, start='2001-01-01', end ='2021-08-17'):
    
    """ This function creates a plot of adjusted close stock prices
    inputs:
    df - dataframe
    title - plot title
    stocks - the stock symbols of each company
    ylabel - y axis label
    y - horizontal line(integer)
    output: the plot of adjusted close stock prices
    """
    df_new = df[start:end]
    #ax = df_new['Adj Close'][stocks].plot(title=title, figsize=(16,8), ax = None)
    ax = df_new.plot(title=title, figsize=(16,8), ax = None)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.axhline(y=y,color='black')
    ax.legend(stocks, loc='upper left')
    plt.show()
    
    # create function that normalizes the data
def normalize_data(df):
    """ 
    This function normalizes the stock prices using the first row of the dataframe
    input - stock data
    output - normalized stock data
    """
    return df/df.iloc[0,:]  

# plot the data with the new normalized data

stocks = ['SANB4.SA', 'PETR4.SA', 'VIVT3.SA', 'UGPA3.SA', 'BBAS3.SA', 'ITSA4.SA', 'BSBR']

plot_data2(normalize_data(fin_data['Adj Close'][stocks]),stocks,title = "Normalized Stock Prices", ylabel = 'Cummulative return',y=1)

#Cumulative Return
#Let's have a look at how the pandemic affected stock prices for these companies.

stocks = ['BBAS3.SA', 'IRBR3.SA', 'KLBN11.SA', 'SUZB3.SA', 'UNIP6.SA', 'TASA4.SA', 'TAEE11.SA']

plot_data2(normalize_data(fin_data['Adj Close'][stocks]['2019-01-01':'2019-12-31']), stocks,title = '2019', ylabel = 'Cummulative return',y=1, start='2019-01-01', end = '2019-12-31') #2019
plot_data2(normalize_data(fin_data['Adj Close'][stocks]['2020-01-01':'2020-12-31']), stocks,title = '2020', ylabel = 'Cummulative return',y=1, start='2020-01-01', end = '2020-12-31') #2020
plot_data2(normalize_data(fin_data['Adj Close'][stocks]['2021-01-01':'2021-08-17']), stocks,title = '2021', ylabel = 'Cummulative return',y=1, start='2021-01-01', end = '2021-08-17') #2021


"""Computing the Rolling mean and Bollinger Bands
The rolling mean may give us some idea about the true underlying prices of a stock. If there is a significant deviation below or above the rolling mean, it may give us an idea about a potential buying and selling opportunity respectively. The challenge remains to know when this deviation is significant enough to pay attention to it.
 Bollinger Bands is a statistical chart that contains the volatility of a financial instrument over time. Bollinger observed that looking at the recent volatility of the stock, if it is very volatile, we might discard the movement above and below the mean. But if it is not very volatile we may want to pay attention to it. He added a band 
 (2 standard deviations) above and below the mean. We would use the rolling standard deviation to help us achieve this.
 
 """
 
# compute rolling mean, rolling standard deviation, upper and lower bands

def rolling_stats(df, stocks, type_, window = 20):
    """
    This function computes the rolling mean and Bollinger bands
    inputs : 
    df - dataframe
    stocks - the type of stocks we would be analyzing
    type_ - the price type of the rolling calculation
    window - number of days used to calculate the statistics
    output: 
    rolling mean, rolling standard deviation, upper and lower bands of 2 std each
    """
    
    val = df[(type_,stocks)]
    rolling_mean = df[(type_, stocks)].rolling(window=window).mean()
    rolling_std = df[(type_, stocks)].rolling(window=window).std()
    upper_band = rolling_mean + rolling_std*2
    lower_band = rolling_mean - rolling_std*2
    
    return val, rolling_mean, rolling_std, upper_band, lower_band
    
# plot the rolling mean, rolling standard deviation, upper and lower bands

def rolling_plot(stocks, val, rolling_mean, upper_band, lower_band, title='Rolling mean'):
    """
    This function plots the rolling mean and Bollinger bands
    inputs : 
    stocks - the type of stocks we would be analyzing
    val - value of the stock price type
    rolling_mean - rolling mean
    upper_band - stocks upper band
    lower_band - stocks lower band
    title - plot title
    output: 
    plot of rolling mean, rolling standard deviation, upper and lower bands of 2 std each
    """
    ax = rolling_mean.plot(title=title, figsize=(16,8), label='Rolling Mean')
    plt.plot(upper_band, label = 'Upper Band')
    plt.plot(lower_band, label = 'Lower Band')
    plt.plot(val, label = 'Value of Stock')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    plt.show()
    
    return ax

stocks = 'PETR4.SA'
type_ = 'Adj Close'

val, rolling_mean, rolling_std, upper_band, lower_band = rolling_stats(fin_data['2019-01-01':'2021-08-17'], stocks, type_)

rolling_plot(stocks, val, rolling_mean, upper_band, lower_band, title='Rolling mean of {} for {} 20 days window'.format(type_,stocks))
plt.show()

# view rolling statistics for Santader BR
stocks = 'BSBR'
type_ = 'Adj Close'

val, rolling_mean, rolling_std, upper_band, lower_band = rolling_stats(fin_data['2019-01-01':'2021-08-17'], stocks, type_)

rolling_plot(stocks, val, rolling_mean, upper_band, lower_band, title='Rolling mean of {} for {} 20 days window'.format(type_,stocks))
plt.show()

# view rolling statistics for Grupo Ipiranga
stocks = 'UGPA3.SA'
type_ = 'Adj Close'

val, rolling_mean, rolling_std, upper_band, lower_band = rolling_stats(fin_data['2019-01-01':'2021-08-17'], stocks, type_)

rolling_plot(stocks, val, rolling_mean, upper_band, lower_band, title='Rolling mean of {} for {} 20 days window'.format(type_,stocks))
plt.show()

"""Computing Daily Returns
Daily returns tells us how much the stock price go up and down on a particular day. We can compute using the following function
 
where price(t) is the price of today's stock and price(t-1) is the price of yesterday's stock.
"""

def daily_returns_cal(df,stocks):
    """
    This function computes and returns the daily return values
    input: df (dataframe) and stocks
    output: daily return values
    """
    
    daily_returns = (df[('Adj Close', stocks)][1:]/df[('Adj Close', stocks)][:-1].values) - 1
       
    return daily_returns

# Daily return of VIVO
plot_data2(daily_returns_cal(fin_data,'VIVT3.SA'),stocks=['VIVT3.SA'], ylabel = 'Daily returns',title='Stock Prices for VIVTSA',y=0)


# Daily return of Santader ON
plot_data2(daily_returns_cal(fin_data,'SANB4.SA'),stocks=['SANB4.SA'], ylabel = 'Daily returns',title='Stock Prices for Santader' ,y=0)

"""Computing Daily Returns

Daily returns tells us how much the stock price go up and down on a particular day. We can compute using the following function:
    
    
    Daily Returns = price(t)/price(t-1) - 1
    
 
where price(t) is the price of today's stock and price(t-1) is the price of yesterday's stock.

"""

def daily_returns_cal(df,stocks):
    """
    This function computes and returns the daily return values
    input: df (dataframe) and stocks
    output: daily return values
    """
    
    daily_returns = (df[('Adj Close', stocks)][1:]/df[('Adj Close', stocks)][:-1].values) - 1
       
    return daily_returns

# Daily return of Petrobras
plot_data2(daily_returns_cal(fin_data,'PETR4.SA'),stocks=['PETR4.SA'], ylabel = 'Daily returns',title='Stock Prices for Petrobras',y=0)


# Daily return of Satander
plot_data2(daily_returns_cal(fin_data,'SANB4.SA'),stocks=['SANB4.SA'], ylabel = 'Daily returns',title='Stock Prices for Santander',y=0)



"""3. Modelling


In this section I will be trying out some models to predict the Adjusted closing price of a stock.

Predicting Adjusted close value of Petrobras  stocks"""

# Function that get specific stock data and fills in any missing value
def get_data(df, stocks):
    """
    This function gets a specific stock data and fills in any missing values using the fill forward and fill backward methods
    Input: 
    df - dataframe
    stocks - the type of stock
    Output - a cleaned dataset to be used for prediction
    """
    df1 = pd.DataFrame (data = df.iloc[:, df.columns.get_level_values(1)==stocks].values,
                          index = df.iloc[:, df.columns.get_level_values(1)==stocks].index,
                          columns = df.iloc[:, df.columns.get_level_values(1)==stocks].columns.get_level_values(0))
    
    df1.fillna(method='ffill', inplace= True)
    df1.fillna(method='bfill', inplace=True)
    
    return df1
# get Petrobras4 data and view the first 5 rows
msft_data = get_data(fin_data,'PETR4.SA')  
msft_data.head()

# plot showing Petrobras historical Adjusted closing prices
plt.figure(figsize=(16,6))
plt.title('Petrobras Adjusted Close Price History')
plt.plot(msft_data['Adj Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adjusted Close Price USD ($)', fontsize=18)
plt.show()

"""
Predicting using Long Short-Term Memory (LSTM)
LSTM is an artificial recurrent neural network (RNN) architecture used in deep learning that is capable of learning long-term dependencies. It processes data passing on information as it propagates forward and have a chain like structure.

"""

#view the shape
msft_data.shape

# create the variables for prediction and split into training and test sets

y = np.log(msft_data['Adj Close'].astype(int)) # we want to predict the adjusted close price
X = msft_data.drop('Adj Close', axis=1) # predictive variables (removing Adj close from it)

#split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)

# Build the LSTM model for PETROBRAS stock
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# view model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

##Initial parameters used for LSTM
# Train the model - part 1
model.fit(np.array(xtrain).reshape(-1,5,1), ytrain, batch_size =1, epochs=5, verbose=0)

#predictions 
predictions = model.predict(np.array(xtest).reshape(-1,5,1))

#calculation of the mean absolute error
mean_abs_error3 = mean_absolute_error(ytest, predictions)
mean_abs_error3

# Train the model - part 2
model.fit(np.array(xtrain).reshape(-1,5,1), ytrain, batch_size =100, epochs=10, verbose=0)

#predictions 2
predictions = model.predict(np.array(xtest).reshape(-1,5,1))
#calculation of the mean absolute error 2
mean_abs_error3 = mean_absolute_error(ytest, predictions)
mean_abs_error3

#LSTM Refinement
#Final Parameters used to tune LSTM
# Build the LSTM model with the relu activation function
model2 = Sequential()
model2.add(LSTM(128, activation='relu', return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model2.add(LSTM(64, activation='relu', return_sequences=False))
model2.add(Dense(25))
model2.add(Dense(1))

# Compile the model
model2.compile(optimizer='adam', loss='mean_squared_error')
# view model2 summary
model2.summary()

# Train the model - part 3
model2.fit(np.array(xtrain).reshape(-1,5,1), ytrain, batch_size =1, epochs=5, verbose=0)

#predictions 3
predictions2 = model2.predict(np.array(xtest).reshape(-1,5,1))

#calculation of the mean absolute error 2
mean_abs_error2 = mean_absolute_error(ytest, predictions)
mean_abs_error2

# Train the model - part 4 (increasing no. of epochs and batch_size)
model2.fit(np.array(xtrain).reshape(-1,5,1), ytrain, batch_size =100, epochs=10, verbose=0)

#predictions 3
predictions2 = model2.predict(np.array(xtest).reshape(-1,5,1))
#calculation of the mean absolute error 2
mean_abs_error2 = mean_absolute_error(ytest, predictions)
mean_abs_error2

#plot showing the prediction and actual values

fig, axs = plt.subplots(1, 2, figsize=(24, 10), dpi=80)
axs[0].set_title('Predicted vs actual values distribution')
ax1 = sns.kdeplot(data=ytest, color="g", label='Actual values',ax=axs[0])
ax2 = sns.kdeplot(data=predictions2, color="b", label='Predicted values', ax=ax1)
    
sns.regplot(x=ytest, y=predictions2)
plt.title('Predicted vs actual values distribution')
plt.xlabel('Stock Price')
#plt.legend()
ax1.legend()
plt.show()
plt.close()
    
print("Mean absolute error of {0}: {1}".format(model,mean_abs_error3))

# Train the model part 5 (no activation function)
model.fit(np.array(xtrain).reshape(-1,5,1), ytrain, batch_size =800, epochs=50, verbose=0)

# Train the model part 5 (no activation function)
model.fit(np.array(xtrain).reshape(-1,5,1), ytrain, batch_size =800, epochs=50, verbose=0)

#predictions 
predictions = model.predict(np.array(xtest).reshape(-1,5,1))
#calculation of the mean absolute error
mean_abs_error3 = mean_absolute_error(ytest, predictions)
mean_abs_error3

#plot showing the prediction and actual values

fig, axs = plt.subplots(1, 2, figsize=(24, 10), dpi=80)
axs[0].set_title('Predicted vs actual values distribution')
ax1 = sns.kdeplot(data=ytest, color="g", label='Actual values',ax=axs[0])
ax2 = sns.kdeplot(data=predictions, color="b", label='Predicted values', ax=ax1)
    
sns.regplot(x=ytest, y=predictions)
plt.title('Predicted vs actual values distribution')
plt.xlabel('Stock Price')
#plt.legend()
ax1.legend()
plt.show()
plt.close()

print("Mean absolute error of {0}: {1}".format(model,mean_abs_error3))

#view the shape
msft_data.tail()

# Building the model for Adj close prediction

y = np.log(msft_data['Adj Close'].astype(int)) # we want to predict the adjusted close price
X = msft_data.drop('Adj Close', axis=1) # predictive variables (removing Adj close from it)

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)

#creating an instance of a Linear Regressor 
model_lin = LinearRegression()

#fit the model
model_lin.fit(X_train,y_train)

LinearRegression()
# check the score, coef_ and intercept_ of the model
model_lin.score(X_train,y_train)
model_lin.coef_
model_lin.intercept_

print('The score of the model is {}, the coeficients  are {} and the intercept is {}'.format(model_lin.score(X_train,y_train), model_lin.coef_ , model_lin.intercept_ ))

#Predicting using Random Forest Regressor
# Building the model for Adj close prediction

y = np.log(msft_data['Adj Close'].astype(int)) # we want to predict the adjusted close price
X = msft_data.drop('Adj Close', axis=1) # predictive variables (removing Adj close from it)

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)

#creating an instance of a Random Forest Regressor 
model_rf = RandomForestRegressor(n_estimators=100, random_state=47)

#fit the model with the training data
model_rf.fit(X_train,y_train)
RandomForestRegressor(random_state=47)
#prediction
predict = model_rf.predict(X_test)
predict #view some predictions

#calculation of the mean absolute error
mean_abs_error = mean_absolute_error(y_test, predict)
mean_abs_error

# view predictions and actual values
#print(np.c_[y_test,predict])
display_ = pd.DataFrame({'Actual value': y_test, 'Predicted value':predict})
display_.head(10)
#print(y_test,predict)

#plot showing the prediction and actual values

fig, axs = plt.subplots(1, 2, figsize=(24, 10), dpi=80)
axs[0].set_title('Predicted vs actual values distribution')
ax1 = sns.kdeplot(data=y_test, color="g", label='Actual values',ax=axs[0])
ax2 = sns.kdeplot(data=predict, color="b", label='Predicted values', ax=ax1)
    
sns.regplot(x=y_test, y=predict)
plt.title('Predicted vs actual values distribution')
plt.xlabel('Stock Price')
#plt.legend()
ax1.legend()
plt.show()
plt.close()
    
print("Mean absolute error of {0}: {1}".format(model_rf,mean_abs_error))

#Predicting Adjusted close value of VIVO stocks
# get Google data and view the first 5 rows
googl_data = get_data(fin_data, 'VIVT3.SA')  
googl_data.head()

# plot showing Vivo  historical Adjusted closing prices
plt.figure(figsize=(16,6))
plt.title('G Vivo Adjusted Close Price History')
plt.plot(googl_data['Adj Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adjusted Close Price USD ($)', fontsize=18)
plt.show()

# Building the model for Adj close prediction

# create the variables for prediction and split into training and test sets

y = np.log(googl_data['Adj Close'].astype(int)) # we want to predict the adjusted close price
X = googl_data.drop('Adj Close', axis=1) # predictive variables (removing Adj close from it)

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)

##Prediction Using LSTM
#### LSTM Model for Google stocks

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model part 5 (no activation function)
model.fit(np.array(X_train).reshape(-1,5,1), y_train, batch_size =800, epochs=50, verbose=0)

#predictions 
predictions = model.predict(np.array(X_test).reshape(-1,5,1))

#calculation of the mean absolute error
mean_abs_error = mean_absolute_error(y_test, predictions)
mean_abs_error

##Prediction Using Linear Regression
#creating an instance of a Linear Regressor 
model_lin2 = LinearRegression()

#fit the model
model_lin2.fit(X_train,y_train)

#prediction
predict = model_lin2.predict(X_test)

#calculation of the mean absolute error
mean_abs_error = mean_absolute_error(y_test, predict)
mean_abs_error

##Prediction Using Random Forest Regressor
#creating an instance of a Random Forest Regressor 
model_rf2 = RandomForestRegressor(n_estimators=100, random_state=47)

#fit the model with the training data
model_rf2.fit(X_train,y_train)

#prediction
predict2 = model_rf2.predict(X_test)
predict2 #view some predictions

#calculation of the mean absolute error
mean_abs_error2 = mean_absolute_error(y_test, predict2)
mean_abs_error2

#plot showing the prediction and actual values

fig, axs = plt.subplots(1, 2, figsize=(24, 10), dpi=80)
axs[0].set_title('Predicted vs actual values distribution')
ax1 = sns.kdeplot(data=y_test, color="g", label='Actual values',ax=axs[0])
ax2 = sns.kdeplot(data=predict2, color="b", label='Predicted values', ax=ax1)
    
sns.regplot(x=y_test, y=predict2)
plt.title('Predicted vs actual values distribution')
plt.xlabel('Stock Price')
#plt.legend()
ax1.legend()
plt.show()
plt.close()
    
print("Mean absolute error of {0}: {1}".format(model_rf2,mean_abs_error2))

#plot showing the prediction and actual values

fig, axs = plt.subplots(1, 2, figsize=(24, 10), dpi=80)
axs[0].set_title('Actual values distribution')
axs[1].set_title('Predicted values distribution')
ax1 = sns.kdeplot(data=y_test, color="g", label='Actual values',ax=axs[0])
ax2 = sns.kdeplot(data=predict2, color="b", label='Predicted values', ax=axs[1])

ax1.legend()
ax2.legend()
plt.show()
plt.close()
    
print("Mean absolute error of {0}: {1}".format(model_rf2,mean_abs_error2))