#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:31:57 2023

@author: elcio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf

"""

#Declarando as açoes para baixar com o "yf" 

#tickers = ['SUZB3.SA', 'IRBR3.SA', 'KLBN11.SA', 'UNIP6.SA', 'BBAS3.SA', 'TASA4.SA', 'TAEE11.SA']

#DADOS BAIXASDOS COM YF

#fin_data = yf.download(tickers, start = '2023-01-01', end = '2023-02-01')


#fin_data.to_csv('/home/elcio/POS_Aulas/Barsi_data.csv')

#check the dimensions of the data
#fin_data.shape
#view the first 5 rows of the data
#fin_data.head()

#view the last 5 values of the data
#fin_data.tail()

# #check if there are missing values for each type of stock

#fin_data.isnull().sum()

"""
tickers = ['BBAS3.SA']
data = yf.download(tickers, start = '2023-01-01', period= '1d')

data['mm_20p'] = data.Close.rolling(20).mean()
data['max_mm_ant'] = data.mm_20p.rolling(5, closed='left' ).max()
data['min_mm_ant'] = data.mm_20p.rolling(5, closed='left' ).min()
data['cond_alta'] = (data.Close > data.mm_20p) & (data.mm_20p > data.max_mm_ant)
data['cond_baixa'] = (data.Close < data.mm_20p) & (data.mm_20p < data.max_mm_ant)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='sem_tendenia', increasing_line_color='gray', 
    decreasing_line_color='darkgray'))

fig.add_trace(go.Candlestick(x=data.index.where(data['cond_alta']==True),
    open=data['Open'].where(data['cond_alta']==True),
    high=data['High'].where(data['cond_alta']==True),
    low=data['Low'].where(data['cond_alta']==True),
    close=data['Close'].where(data['cond_alta']==True),
    increasing_line_color='limegreen', 
    decreasing_line_color='darkgreen',
    name='Tendencia_de_Alta'))
fig.add_trace(go.Candlestick(x=data.index.where(data['cond_baixa']==True),
    open=data['Open'].where(data['cond_baixa']==True),
    high=data['High'].where(data['cond_baixa']==True),
    low=data['Low'].where(data['cond_alta']==True),
    close=data['Close'].where(data['cond_baixa']==True),
    increasing_line_color='lightcoral', 
    decreasing_line_color='red',
    name='Tendencia_de_Baixa'))
fig.add_trace(go.Scatter(name='Sma_20', x=data.index, y=data['mm_20p'], marker_color='gray'))
fig.update_layout(template='ggplot2', width=1200, height=800 )
plt.show(True)



