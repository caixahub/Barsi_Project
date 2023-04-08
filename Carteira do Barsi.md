**Codigos de Analise de ações (Profit)  Bonus 2**


```python
Projeto Ficando Rico com o Barsi...

```

```python
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
```

    2023-04-04 17:57:18.921704: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
tickers = ['BBAS3.SA', 'IRBR3.SA', 'KLBN11.SA', 'SUZB3.SA', 'UNIP6.SA', 'TASA4.SA', 'TAEE11.SA']
df = yf.download(tickers, start = '2017-01-01', period= '1d')
```

    [*********************100%***********************]  7 of 7 completed



```python
df

```





<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="7" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th>...</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="7" halign="left">Volume</th>
    </tr>
    <tr>
      <th></th>
      <th>BBAS3.SA</th>
      <th>IRBR3.SA</th>
      <th>KLBN11.SA</th>
      <th>SUZB3.SA</th>
      <th>TAEE11.SA</th>
      <th>TASA4.SA</th>
      <th>UNIP6.SA</th>
      <th>BBAS3.SA</th>
      <th>IRBR3.SA</th>
      <th>KLBN11.SA</th>
      <th>...</th>
      <th>TAEE11.SA</th>
      <th>TASA4.SA</th>
      <th>UNIP6.SA</th>
      <th>BBAS3.SA</th>
      <th>IRBR3.SA</th>
      <th>KLBN11.SA</th>
      <th>SUZB3.SA</th>
      <th>TAEE11.SA</th>
      <th>TASA4.SA</th>
      <th>UNIP6.SA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-02</th>
      <td>19.448153</td>
      <td>NaN</td>
      <td>13.230022</td>
      <td>18.122597</td>
      <td>11.511273</td>
      <td>1.716108</td>
      <td>1.991450</td>
      <td>27.540001</td>
      <td>NaN</td>
      <td>17.100000</td>
      <td>...</td>
      <td>20.709999</td>
      <td>1.750000</td>
      <td>4.885713</td>
      <td>1968200</td>
      <td>NaN</td>
      <td>841000.0</td>
      <td>0</td>
      <td>572300.0</td>
      <td>15200</td>
      <td>2660</td>
    </tr>
    <tr>
      <th>2017-01-03</th>
      <td>20.337927</td>
      <td>NaN</td>
      <td>13.562708</td>
      <td>18.122597</td>
      <td>11.532791</td>
      <td>1.744241</td>
      <td>1.968125</td>
      <td>28.799999</td>
      <td>NaN</td>
      <td>17.530001</td>
      <td>...</td>
      <td>21.430000</td>
      <td>1.860000</td>
      <td>4.835713</td>
      <td>7578900</td>
      <td>NaN</td>
      <td>2080700.0</td>
      <td>0</td>
      <td>1292700.0</td>
      <td>11500</td>
      <td>45360</td>
    </tr>
    <tr>
      <th>2017-01-04</th>
      <td>20.232006</td>
      <td>NaN</td>
      <td>13.059813</td>
      <td>18.122597</td>
      <td>11.430589</td>
      <td>1.631709</td>
      <td>1.988534</td>
      <td>28.650000</td>
      <td>NaN</td>
      <td>16.879999</td>
      <td>...</td>
      <td>21.410000</td>
      <td>1.810000</td>
      <td>4.885713</td>
      <td>4156300</td>
      <td>NaN</td>
      <td>1805000.0</td>
      <td>0</td>
      <td>1173400.0</td>
      <td>56900</td>
      <td>1400</td>
    </tr>
    <tr>
      <th>2017-01-05</th>
      <td>20.182573</td>
      <td>NaN</td>
      <td>12.943757</td>
      <td>18.122597</td>
      <td>11.484379</td>
      <td>1.687975</td>
      <td>1.997282</td>
      <td>28.580000</td>
      <td>NaN</td>
      <td>16.730000</td>
      <td>...</td>
      <td>21.209999</td>
      <td>1.780000</td>
      <td>4.885713</td>
      <td>5457100</td>
      <td>NaN</td>
      <td>1837400.0</td>
      <td>0</td>
      <td>1069600.0</td>
      <td>14900</td>
      <td>48860</td>
    </tr>
    <tr>
      <th>2017-01-06</th>
      <td>20.401491</td>
      <td>NaN</td>
      <td>12.881864</td>
      <td>18.122597</td>
      <td>11.387552</td>
      <td>1.641087</td>
      <td>1.976872</td>
      <td>28.889999</td>
      <td>NaN</td>
      <td>16.650000</td>
      <td>...</td>
      <td>21.260000</td>
      <td>1.780000</td>
      <td>4.807141</td>
      <td>3692400</td>
      <td>NaN</td>
      <td>1356700.0</td>
      <td>0</td>
      <td>574100.0</td>
      <td>1000</td>
      <td>1820</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-03-29</th>
      <td>38.310001</td>
      <td>21.129999</td>
      <td>18.180000</td>
      <td>43.070000</td>
      <td>34.919998</td>
      <td>15.500000</td>
      <td>71.639999</td>
      <td>38.310001</td>
      <td>21.129999</td>
      <td>18.180000</td>
      <td>...</td>
      <td>35.230000</td>
      <td>15.450000</td>
      <td>73.660004</td>
      <td>7197800</td>
      <td>1063200.0</td>
      <td>5026000.0</td>
      <td>3955700</td>
      <td>2225500.0</td>
      <td>589300</td>
      <td>230100</td>
    </tr>
    <tr>
      <th>2023-03-30</th>
      <td>39.009998</td>
      <td>22.200001</td>
      <td>18.260000</td>
      <td>43.049999</td>
      <td>34.849998</td>
      <td>16.379999</td>
      <td>71.089996</td>
      <td>39.009998</td>
      <td>22.200001</td>
      <td>18.260000</td>
      <td>...</td>
      <td>35.090000</td>
      <td>15.600000</td>
      <td>71.660004</td>
      <td>13147600</td>
      <td>1717500.0</td>
      <td>5087700.0</td>
      <td>5943600</td>
      <td>2522900.0</td>
      <td>806500</td>
      <td>254100</td>
    </tr>
    <tr>
      <th>2023-03-31</th>
      <td>39.110001</td>
      <td>22.240000</td>
      <td>18.090000</td>
      <td>41.599998</td>
      <td>34.830002</td>
      <td>16.180000</td>
      <td>70.050003</td>
      <td>39.110001</td>
      <td>22.240000</td>
      <td>18.090000</td>
      <td>...</td>
      <td>34.980000</td>
      <td>16.389999</td>
      <td>71.559998</td>
      <td>8450100</td>
      <td>1633100.0</td>
      <td>6252600.0</td>
      <td>14261800</td>
      <td>2057300.0</td>
      <td>1065700</td>
      <td>406100</td>
    </tr>
    <tr>
      <th>2023-04-03</th>
      <td>38.650002</td>
      <td>21.350000</td>
      <td>18.230000</td>
      <td>42.740002</td>
      <td>34.650002</td>
      <td>16.389999</td>
      <td>69.250000</td>
      <td>38.650002</td>
      <td>21.350000</td>
      <td>18.230000</td>
      <td>...</td>
      <td>34.849998</td>
      <td>16.150000</td>
      <td>70.730003</td>
      <td>8960300</td>
      <td>1382600.0</td>
      <td>4081800.0</td>
      <td>6884200</td>
      <td>2448600.0</td>
      <td>642800</td>
      <td>221700</td>
    </tr>
    <tr>
      <th>2023-04-04</th>
      <td>39.290001</td>
      <td>21.980000</td>
      <td>18.309999</td>
      <td>42.349998</td>
      <td>35.180000</td>
      <td>16.270000</td>
      <td>69.970001</td>
      <td>39.290001</td>
      <td>21.980000</td>
      <td>18.309999</td>
      <td>...</td>
      <td>35.000000</td>
      <td>16.440001</td>
      <td>69.650002</td>
      <td>8219000</td>
      <td>1602300.0</td>
      <td>5157000.0</td>
      <td>4730300</td>
      <td>3813500.0</td>
      <td>514700</td>
      <td>200300</td>
    </tr>
  </tbody>
</table>
<p>1558 rows × 42 columns</p>
</div>




```python
# some missing values, and we will use the fillna method to resolve the missing values.

# handing missing values
df.fillna(method='ffill', inplace = True) # use front fill method
df.fillna(method='bfill', inplace = True) # use back fill method
#check to see if there are still any missing values
df.isnull().sum()
```




    Adj Close  BBAS3.SA     0
               IRBR3.SA     0
               KLBN11.SA    0
               SUZB3.SA     0
               TAEE11.SA    0
               TASA4.SA     0
               UNIP6.SA     0
    Close      BBAS3.SA     0
               IRBR3.SA     0
               KLBN11.SA    0
               SUZB3.SA     0
               TAEE11.SA    0
               TASA4.SA     0
               UNIP6.SA     0
    High       BBAS3.SA     0
               IRBR3.SA     0
               KLBN11.SA    0
               SUZB3.SA     0
               TAEE11.SA    0
               TASA4.SA     0
               UNIP6.SA     0
    Low        BBAS3.SA     0
               IRBR3.SA     0
               KLBN11.SA    0
               SUZB3.SA     0
               TAEE11.SA    0
               TASA4.SA     0
               UNIP6.SA     0
    Open       BBAS3.SA     0
               IRBR3.SA     0
               KLBN11.SA    0
               SUZB3.SA     0
               TAEE11.SA    0
               TASA4.SA     0
               UNIP6.SA     0
    Volume     BBAS3.SA     0
               IRBR3.SA     0
               KLBN11.SA    0
               SUZB3.SA     0
               TAEE11.SA    0
               TASA4.SA     0
               UNIP6.SA     0
    dtype: int64




```python
df

```



<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="7" halign="left">Adj Close</th>
      <th colspan="3" halign="left">Close</th>
      <th>...</th>
      <th colspan="3" halign="left">Open</th>
      <th colspan="7" halign="left">Volume</th>
    </tr>
    <tr>
      <th></th>
      <th>BBAS3.SA</th>
      <th>IRBR3.SA</th>
      <th>KLBN11.SA</th>
      <th>SUZB3.SA</th>
      <th>TAEE11.SA</th>
      <th>TASA4.SA</th>
      <th>UNIP6.SA</th>
      <th>BBAS3.SA</th>
      <th>IRBR3.SA</th>
      <th>KLBN11.SA</th>
      <th>...</th>
      <th>TAEE11.SA</th>
      <th>TASA4.SA</th>
      <th>UNIP6.SA</th>
      <th>BBAS3.SA</th>
      <th>IRBR3.SA</th>
      <th>KLBN11.SA</th>
      <th>SUZB3.SA</th>
      <th>TAEE11.SA</th>
      <th>TASA4.SA</th>
      <th>UNIP6.SA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-02</th>
      <td>19.448153</td>
      <td>216.538574</td>
      <td>13.230022</td>
      <td>18.122597</td>
      <td>11.511273</td>
      <td>1.716108</td>
      <td>1.991450</td>
      <td>27.540001</td>
      <td>224.583786</td>
      <td>17.100000</td>
      <td>...</td>
      <td>20.709999</td>
      <td>1.750000</td>
      <td>4.885713</td>
      <td>1968200</td>
      <td>1448107.0</td>
      <td>841000.0</td>
      <td>0</td>
      <td>572300.0</td>
      <td>15200</td>
      <td>2660</td>
    </tr>
    <tr>
      <th>2017-01-03</th>
      <td>20.337927</td>
      <td>216.538574</td>
      <td>13.562708</td>
      <td>18.122597</td>
      <td>11.532791</td>
      <td>1.744241</td>
      <td>1.968125</td>
      <td>28.799999</td>
      <td>224.583786</td>
      <td>17.530001</td>
      <td>...</td>
      <td>21.430000</td>
      <td>1.860000</td>
      <td>4.835713</td>
      <td>7578900</td>
      <td>1448107.0</td>
      <td>2080700.0</td>
      <td>0</td>
      <td>1292700.0</td>
      <td>11500</td>
      <td>45360</td>
    </tr>
    <tr>
      <th>2017-01-04</th>
      <td>20.232006</td>
      <td>216.538574</td>
      <td>13.059813</td>
      <td>18.122597</td>
      <td>11.430589</td>
      <td>1.631709</td>
      <td>1.988534</td>
      <td>28.650000</td>
      <td>224.583786</td>
      <td>16.879999</td>
      <td>...</td>
      <td>21.410000</td>
      <td>1.810000</td>
      <td>4.885713</td>
      <td>4156300</td>
      <td>1448107.0</td>
      <td>1805000.0</td>
      <td>0</td>
      <td>1173400.0</td>
      <td>56900</td>
      <td>1400</td>
    </tr>
    <tr>
      <th>2017-01-05</th>
      <td>20.182573</td>
      <td>216.538574</td>
      <td>12.943757</td>
      <td>18.122597</td>
      <td>11.484379</td>
      <td>1.687975</td>
      <td>1.997282</td>
      <td>28.580000</td>
      <td>224.583786</td>
      <td>16.730000</td>
      <td>...</td>
      <td>21.209999</td>
      <td>1.780000</td>
      <td>4.885713</td>
      <td>5457100</td>
      <td>1448107.0</td>
      <td>1837400.0</td>
      <td>0</td>
      <td>1069600.0</td>
      <td>14900</td>
      <td>48860</td>
    </tr>
    <tr>
      <th>2017-01-06</th>
      <td>20.401491</td>
      <td>216.538574</td>
      <td>12.881864</td>
      <td>18.122597</td>
      <td>11.387552</td>
      <td>1.641087</td>
      <td>1.976872</td>
      <td>28.889999</td>
      <td>224.583786</td>
      <td>16.650000</td>
      <td>...</td>
      <td>21.260000</td>
      <td>1.780000</td>
      <td>4.807141</td>
      <td>3692400</td>
      <td>1448107.0</td>
      <td>1356700.0</td>
      <td>0</td>
      <td>574100.0</td>
      <td>1000</td>
      <td>1820</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-03-29</th>
      <td>38.310001</td>
      <td>21.129999</td>
      <td>18.180000</td>
      <td>43.070000</td>
      <td>34.919998</td>
      <td>15.500000</td>
      <td>71.639999</td>
      <td>38.310001</td>
      <td>21.129999</td>
      <td>18.180000</td>
      <td>...</td>
      <td>35.230000</td>
      <td>15.450000</td>
      <td>73.660004</td>
      <td>7197800</td>
      <td>1063200.0</td>
      <td>5026000.0</td>
      <td>3955700</td>
      <td>2225500.0</td>
      <td>589300</td>
      <td>230100</td>
    </tr>
    <tr>
      <th>2023-03-30</th>
      <td>39.009998</td>
      <td>22.200001</td>
      <td>18.260000</td>
      <td>43.049999</td>
      <td>34.849998</td>
      <td>16.379999</td>
      <td>71.089996</td>
      <td>39.009998</td>
      <td>22.200001</td>
      <td>18.260000</td>
      <td>...</td>
      <td>35.090000</td>
      <td>15.600000</td>
      <td>71.660004</td>
      <td>13147600</td>
      <td>1717500.0</td>
      <td>5087700.0</td>
      <td>5943600</td>
      <td>2522900.0</td>
      <td>806500</td>
      <td>254100</td>
    </tr>
    <tr>
      <th>2023-03-31</th>
      <td>39.110001</td>
      <td>22.240000</td>
      <td>18.090000</td>
      <td>41.599998</td>
      <td>34.830002</td>
      <td>16.180000</td>
      <td>70.050003</td>
      <td>39.110001</td>
      <td>22.240000</td>
      <td>18.090000</td>
      <td>...</td>
      <td>34.980000</td>
      <td>16.389999</td>
      <td>71.559998</td>
      <td>8450100</td>
      <td>1633100.0</td>
      <td>6252600.0</td>
      <td>14261800</td>
      <td>2057300.0</td>
      <td>1065700</td>
      <td>406100</td>
    </tr>
    <tr>
      <th>2023-04-03</th>
      <td>38.650002</td>
      <td>21.350000</td>
      <td>18.230000</td>
      <td>42.740002</td>
      <td>34.650002</td>
      <td>16.389999</td>
      <td>69.250000</td>
      <td>38.650002</td>
      <td>21.350000</td>
      <td>18.230000</td>
      <td>...</td>
      <td>34.849998</td>
      <td>16.150000</td>
      <td>70.730003</td>
      <td>8960300</td>
      <td>1382600.0</td>
      <td>4081800.0</td>
      <td>6884200</td>
      <td>2448600.0</td>
      <td>642800</td>
      <td>221700</td>
    </tr>
    <tr>
      <th>2023-04-04</th>
      <td>39.290001</td>
      <td>21.980000</td>
      <td>18.309999</td>
      <td>42.349998</td>
      <td>35.180000</td>
      <td>16.270000</td>
      <td>69.970001</td>
      <td>39.290001</td>
      <td>21.980000</td>
      <td>18.309999</td>
      <td>...</td>
      <td>35.000000</td>
      <td>16.440001</td>
      <td>69.650002</td>
      <td>8219000</td>
      <td>1602300.0</td>
      <td>5157000.0</td>
      <td>4730300</td>
      <td>3813500.0</td>
      <td>514700</td>
      <td>200300</td>
    </tr>
  </tbody>
</table>
<p>1558 rows × 42 columns</p>
</div>




```python
#view descriptive statistics of adjusted close process of the stocks
df[['Adj Close']].describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="7" halign="left">Adj Close</th>
    </tr>
    <tr>
      <th></th>
      <th>BBAS3.SA</th>
      <th>IRBR3.SA</th>
      <th>KLBN11.SA</th>
      <th>SUZB3.SA</th>
      <th>TAEE11.SA</th>
      <th>TASA4.SA</th>
      <th>UNIP6.SA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1558.000000</td>
      <td>1558.000000</td>
      <td>1558.000000</td>
      <td>1558.000000</td>
      <td>1558.000000</td>
      <td>1558.000000</td>
      <td>1558.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>30.008899</td>
      <td>304.475876</td>
      <td>17.749022</td>
      <td>39.594539</td>
      <td>22.668617</td>
      <td>9.369296</td>
      <td>36.254399</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.404710</td>
      <td>258.272574</td>
      <td>3.947339</td>
      <td>13.762798</td>
      <td>9.435561</td>
      <td>8.002896</td>
      <td>30.104588</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.166691</td>
      <td>18.450001</td>
      <td>10.714169</td>
      <td>16.001627</td>
      <td>11.183148</td>
      <td>1.584821</td>
      <td>1.965209</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.016495</td>
      <td>128.938995</td>
      <td>14.258628</td>
      <td>30.675386</td>
      <td>12.891089</td>
      <td>2.363165</td>
      <td>16.065682</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.693673</td>
      <td>216.538574</td>
      <td>17.219313</td>
      <td>41.355589</td>
      <td>20.502331</td>
      <td>5.087368</td>
      <td>21.225048</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>36.008152</td>
      <td>416.682083</td>
      <td>20.994648</td>
      <td>49.215112</td>
      <td>32.224201</td>
      <td>16.962500</td>
      <td>66.940727</td>
    </tr>
    <tr>
      <th>max</th>
      <td>43.551613</td>
      <td>1030.961914</td>
      <td>27.867046</td>
      <td>74.390312</td>
      <td>40.624340</td>
      <td>26.970087</td>
      <td>109.632782</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_data2(df,stocks,title='Stock Prices',ylabel="Stock Price",y=0, start='2001-01-17', end ='2023-03-01'):
    
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
```


```python
# create function that normalizes the data
def normalize_data(df):
    """ 
    This function normalizes the stock prices using the first row of the dataframe
    input - stock data
    output - normalized stock data
    """
    return df/df.iloc[0,:]    
```


```python
# plot the data with the new normalized data

stocks = ["BBAS3.SA", "IRBR3.SA", "UNIP6.SA", "KLBN11.SA", "TAEE11.SA", "TASA4.SA", "SUZB3.SA"]

plot_data2(normalize_data(df['Adj Close'][stocks]),stocks,title = "Normalized Stock Prices", ylabel = 'Cummulative return',y=1)
```

![output_8_0](https://user-images.githubusercontent.com/118861107/229951280-43decbb0-574f-4ab4-82be-c9de03633c6b.png)
    

    



```python
#Cumulative Return
#Let's have a look at how the pandemic affected stock prices for these companies.

stocks = ["BBAS3.SA", "IRBR3.SA", "UNIP6.SA", "KLBN11.SA", "TAEE11.SA", "TASA4.SA", "SUZB3.SA"]

plot_data2(normalize_data(df['Adj Close'][stocks]['2019-01-01':'2019-12-31']), stocks,title = '2019', ylabel = 'Cummulative return',y=1, start='2019-01-01', end = '2019-12-31') #2019
plot_data2(normalize_data(df['Adj Close'][stocks]['2020-01-01':'2020-12-31']), stocks,title = '2020', ylabel = 'Cummulative return',y=1, start='2020-01-01', end = '2020-12-31') #2020
plot_data2(normalize_data(df['Adj Close'][stocks]['2021-01-01':'2021-08-17']), stocks,title = '2021', ylabel = 'Cummulative return',y=1, start='2021-01-01', end = '2021-08-17') #2021
```
    

![output_9_0](https://user-images.githubusercontent.com/118861107/229951283-d07a12ce-548c-41c4-be33-232ef883f4a5.png)
![output_9_1](https://user-images.githubusercontent.com/118861107/229951284-0e725f69-ad32-40c2-a136-d39910f8a803.png)
![output_9_2](https://user-images.githubusercontent.com/118861107/229951286-0398614a-fb53-4f04-bcc5-a9374dd3fd50.png)
```python




