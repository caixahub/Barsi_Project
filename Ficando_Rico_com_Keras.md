
```python
        Usando as dicas de carteira de investimentos do Barsi propusemos uma questão é possivel ficar
        rico usando um modelo preditor e se é qual o melhor para o projeto ?
        
        Desde as primeiras rodadas o Keras se mostrou melhor preditor que o LSTM 
        embora em varias literaturas o LSTM é indicado com o melhor preditor
        para series temporais, agora fica a criteiro de cada um por ou não 
        dinheiro nas previsões do modelo.
        O projeto tem como base o exercicio do professor  Rdrigo Correia entre outros  
        https://www.linkedin.com/pulse/prevendo-pre%C3%A7o-de-a%C3%A7%C3%B5es-com-deep-learning-lstm-rodrigo-correa/
        foram feitas modificações para o uso do Keras. 
```




```python

# Baixando bibliotecas para analise
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Baixando os dados da ação com yfinance, para quem não esta acostumando com ele 
# é necessario usar o ".SA" junto do ticker origianl da ação.

Stock = pd.DataFrame(yf.Ticker('UNIP6.SA').history(period = '5y'))
```


```python
df = Stock.reset_index()
```


```python
df
```





  <div id="df-1968935e-c5e1-4de3-b2ce-9b0b183d6038">
    <div class="colab-df-container">
      <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-06 00:00:00-03:00</td>
      <td>13.812442</td>
      <td>14.145845</td>
      <td>13.579059</td>
      <td>13.579059</td>
      <td>90580</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-09 00:00:00-03:00</td>
      <td>13.645743</td>
      <td>13.817208</td>
      <td>13.217081</td>
      <td>13.426649</td>
      <td>93660</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-10 00:00:00-03:00</td>
      <td>13.574295</td>
      <td>13.812441</td>
      <td>12.616951</td>
      <td>13.436172</td>
      <td>171220</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-11 00:00:00-03:00</td>
      <td>13.331390</td>
      <td>13.507617</td>
      <td>12.645530</td>
      <td>13.183739</td>
      <td>115640</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-12 00:00:00-03:00</td>
      <td>13.217080</td>
      <td>13.802918</td>
      <td>13.217080</td>
      <td>13.417122</td>
      <td>87780</td>
      <td>0.0</td>
      <td>0.0</td>
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
    </tr>
    <tr>
      <th>1236</th>
      <td>2023-03-31 00:00:00-03:00</td>
      <td>71.559998</td>
      <td>72.389999</td>
      <td>69.599998</td>
      <td>70.050003</td>
      <td>406100</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1237</th>
      <td>2023-04-03 00:00:00-03:00</td>
      <td>70.730003</td>
      <td>70.730003</td>
      <td>69.169998</td>
      <td>69.250000</td>
      <td>221700</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>2023-04-04 00:00:00-03:00</td>
      <td>69.650002</td>
      <td>71.000000</td>
      <td>69.650002</td>
      <td>69.970001</td>
      <td>200900</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1239</th>
      <td>2023-04-05 00:00:00-03:00</td>
      <td>70.199997</td>
      <td>70.879997</td>
      <td>69.320000</td>
      <td>69.930000</td>
      <td>157500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1240</th>
      <td>2023-04-06 00:00:00-03:00</td>
      <td>70.029999</td>
      <td>70.870003</td>
      <td>69.320000</td>
      <td>69.449997</td>
      <td>243500</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1241 rows × 8 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1968935e-c5e1-4de3-b2ce-9b0b183d6038')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">







```python
##Importando as libraries do Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
```


```python
#Criando o Dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    
    new_data['Close'][i] = data['Close'][i]
```


```python
#Colocando data como índice
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
```


```python
#Criando o train e o test set
dataset = new_data.values


train = dataset[0:1100,:]
valid = dataset[1100:,:]
```


```python
train
```





  <div id="df-45cbcd06-400c-4640-a386-9fab2a221d0c">
    <div class="colab-df-container">

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-05 00:00:00-03:00</th>
      <td>13.793394</td>
    </tr>
    <tr>
      <th>2018-04-06 00:00:00-03:00</th>
      <td>13.579059</td>
    </tr>
    <tr>
      <th>2018-04-09 00:00:00-03:00</th>
      <td>13.426649</td>
    </tr>
    <tr>
      <th>2018-04-10 00:00:00-03:00</th>
      <td>13.43617</td>
    </tr>
    <tr>
      <th>2018-04-11 00:00:00-03:00</th>
      <td>13.183738</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-09-05 00:00:00-03:00</th>
      <td>95.246101</td>
    </tr>
    <tr>
      <th>2022-09-06 00:00:00-03:00</th>
      <td>94.253265</td>
    </tr>
    <tr>
      <th>2022-09-08 00:00:00-03:00</th>
      <td>94.873787</td>
    </tr>
    <tr>
      <th>2022-09-09 00:00:00-03:00</th>
      <td>96.038467</td>
    </tr>
    <tr>
      <th>2022-09-12 00:00:00-03:00</th>
      <td>94.167343</td>
    </tr>
  </tbody>
</table>
<p>1100 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-45cbcd06-400c-4640-a386-9fab2a221d0c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">






```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
```


```python
x_train, y_train = [], []
for i in range(90,len(train)):
    x_train.append(scaled_data[i-90:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

```


```python
# Criando a arquitetura da rede neural:
modelo = Sequential()
modelo.add(Dense(units=6, activation="relu", input_dim=x_train.shape[1]))
modelo.add(Dense(units=1, activation="linear")) #neuronio de saida
```


```python
modelo.compile(loss="mse", optimizer="adam", metrics=["mae"])
resultado = modelo.fit(x_train, y_train, epochs=200, batch_size=32, verbose=2 )
```

    Epoch 1/200
    32/32 - 5s - loss: 0.0510 - mae: 0.1482 - 5s/epoch - 155ms/step
    Epoch 2/200
    32/32 - 0s - loss: 0.0090 - mae: 0.0726 - 75ms/epoch - 2ms/step
    Epoch 3/200
    32/32 - 0s - loss: 0.0077 - mae: 0.0674 - 72ms/epoch - 2ms/step
    Epoch 4/200
    32/32 - 0s - loss: 0.0067 - mae: 0.0632 - 70ms/epoch - 2ms/step
    Epoch 5/200
    32/32 - 0s - loss: 0.0058 - mae: 0.0578 - 74ms/epoch - 2ms/step
    Epoch 6/200
    32/32 - 0s - loss: 0.0050 - mae: 0.0541 - 74ms/epoch - 2ms/step
    Epoch 7/200
    32/32 - 0s - loss: 0.0042 - mae: 0.0494 - 73ms/epoch - 2ms/step
    Epoch 8/200
    32/32 - 0s - loss: 0.0037 - mae: 0.0457 - 72ms/epoch - 2ms/step
    Epoch 9/200
    32/32 - 0s - loss: 0.0031 - mae: 0.0424 - 82ms/epoch - 3ms/step
    Epoch 10/200
    32/32 - 0s - loss: 0.0027 - mae: 0.0393 - 73ms/epoch - 2ms/step
    Epoch 11/200
    32/32 - 0s - loss: 0.0023 - mae: 0.0363 - 69ms/epoch - 2ms/step
    Epoch 12/200
    32/32 - 0s - loss: 0.0020 - mae: 0.0338 - 74ms/epoch - 2ms/step
    Epoch 13/200
    32/32 - 0s - loss: 0.0018 - mae: 0.0316 - 72ms/epoch - 2ms/step
    Epoch 14/200
    32/32 - 0s - loss: 0.0016 - mae: 0.0303 - 73ms/epoch - 2ms/step
    Epoch 15/200
    32/32 - 0s - loss: 0.0015 - mae: 0.0287 - 74ms/epoch - 2ms/step
    Epoch 16/200
    32/32 - 0s - loss: 0.0013 - mae: 0.0269 - 74ms/epoch - 2ms/step
    Epoch 17/200
    32/32 - 0s - loss: 0.0012 - mae: 0.0259 - 72ms/epoch - 2ms/step
    Epoch 18/200
    32/32 - 0s - loss: 0.0012 - mae: 0.0254 - 71ms/epoch - 2ms/step
    Epoch 19/200
    32/32 - 0s - loss: 0.0011 - mae: 0.0246 - 78ms/epoch - 2ms/step
    Epoch 20/200
    32/32 - 0s - loss: 0.0010 - mae: 0.0238 - 74ms/epoch - 2ms/step
    Epoch 21/200
    32/32 - 0s - loss: 0.0011 - mae: 0.0236 - 76ms/epoch - 2ms/step
    Epoch 22/200
    32/32 - 0s - loss: 9.7949e-04 - mae: 0.0229 - 81ms/epoch - 3ms/step
    Epoch 23/200
    32/32 - 0s - loss: 9.5524e-04 - mae: 0.0223 - 72ms/epoch - 2ms/step
    Epoch 24/200
    32/32 - 0s - loss: 9.5312e-04 - mae: 0.0223 - 81ms/epoch - 3ms/step
    Epoch 25/200
    32/32 - 0s - loss: 9.1994e-04 - mae: 0.0220 - 74ms/epoch - 2ms/step
    Epoch 26/200
    32/32 - 0s - loss: 9.0176e-04 - mae: 0.0218 - 83ms/epoch - 3ms/step
    Epoch 27/200
    32/32 - 0s - loss: 8.8111e-04 - mae: 0.0214 - 77ms/epoch - 2ms/step
    Epoch 28/200
    32/32 - 0s - loss: 8.5502e-04 - mae: 0.0212 - 72ms/epoch - 2ms/step
    Epoch 29/200
    32/32 - 0s - loss: 8.5395e-04 - mae: 0.0211 - 73ms/epoch - 2ms/step
    Epoch 30/200
    32/32 - 0s - loss: 8.2744e-04 - mae: 0.0207 - 73ms/epoch - 2ms/step
    Epoch 31/200
    32/32 - 0s - loss: 8.2414e-04 - mae: 0.0208 - 75ms/epoch - 2ms/step
    Epoch 32/200
    32/32 - 0s - loss: 8.1951e-04 - mae: 0.0208 - 79ms/epoch - 2ms/step
    Epoch 33/200
    32/32 - 0s - loss: 8.0185e-04 - mae: 0.0204 - 75ms/epoch - 2ms/step
    Epoch 34/200
    32/32 - 0s - loss: 7.6558e-04 - mae: 0.0200 - 70ms/epoch - 2ms/step
    Epoch 35/200
    32/32 - 0s - loss: 7.7159e-04 - mae: 0.0201 - 78ms/epoch - 2ms/step
    Epoch 36/200
    32/32 - 0s - loss: 7.7171e-04 - mae: 0.0200 - 70ms/epoch - 2ms/step
    Epoch 37/200
    32/32 - 0s - loss: 7.4173e-04 - mae: 0.0196 - 71ms/epoch - 2ms/step
    Epoch 38/200
    32/32 - 0s - loss: 7.2582e-04 - mae: 0.0193 - 69ms/epoch - 2ms/step
    Epoch 39/200
    32/32 - 0s - loss: 6.9777e-04 - mae: 0.0189 - 69ms/epoch - 2ms/step
    Epoch 40/200
    32/32 - 0s - loss: 7.3577e-04 - mae: 0.0193 - 73ms/epoch - 2ms/step
    Epoch 41/200
    32/32 - 0s - loss: 7.0137e-04 - mae: 0.0189 - 69ms/epoch - 2ms/step
    Epoch 42/200
    32/32 - 0s - loss: 7.0354e-04 - mae: 0.0189 - 70ms/epoch - 2ms/step
    Epoch 43/200
    32/32 - 0s - loss: 6.9409e-04 - mae: 0.0187 - 71ms/epoch - 2ms/step
    Epoch 44/200
    32/32 - 0s - loss: 6.5812e-04 - mae: 0.0183 - 73ms/epoch - 2ms/step
    Epoch 45/200
    32/32 - 0s - loss: 6.4539e-04 - mae: 0.0181 - 71ms/epoch - 2ms/step
    Epoch 46/200
    32/32 - 0s - loss: 6.3707e-04 - mae: 0.0181 - 75ms/epoch - 2ms/step
    Epoch 47/200
    32/32 - 0s - loss: 6.1640e-04 - mae: 0.0177 - 71ms/epoch - 2ms/step
    Epoch 48/200
    32/32 - 0s - loss: 6.3426e-04 - mae: 0.0179 - 79ms/epoch - 2ms/step
    Epoch 49/200
    32/32 - 0s - loss: 6.2877e-04 - mae: 0.0179 - 75ms/epoch - 2ms/step
    Epoch 50/200
    32/32 - 0s - loss: 5.8969e-04 - mae: 0.0173 - 75ms/epoch - 2ms/step
    Epoch 51/200
    32/32 - 0s - loss: 5.7311e-04 - mae: 0.0171 - 82ms/epoch - 3ms/step
    Epoch 52/200
    32/32 - 0s - loss: 5.6173e-04 - mae: 0.0170 - 82ms/epoch - 3ms/step
    Epoch 53/200
    32/32 - 0s - loss: 5.8723e-04 - mae: 0.0173 - 77ms/epoch - 2ms/step
    Epoch 54/200
    32/32 - 0s - loss: 5.5911e-04 - mae: 0.0169 - 72ms/epoch - 2ms/step
    Epoch 55/200
    32/32 - 0s - loss: 5.6129e-04 - mae: 0.0169 - 72ms/epoch - 2ms/step
    Epoch 56/200
    32/32 - 0s - loss: 5.4532e-04 - mae: 0.0167 - 70ms/epoch - 2ms/step
    Epoch 57/200
    32/32 - 0s - loss: 5.8016e-04 - mae: 0.0171 - 73ms/epoch - 2ms/step
    Epoch 58/200
    32/32 - 0s - loss: 5.4892e-04 - mae: 0.0165 - 73ms/epoch - 2ms/step
    Epoch 59/200
    32/32 - 0s - loss: 5.3085e-04 - mae: 0.0165 - 77ms/epoch - 2ms/step
    Epoch 60/200
    32/32 - 0s - loss: 5.4143e-04 - mae: 0.0163 - 73ms/epoch - 2ms/step
    Epoch 61/200
    32/32 - 0s - loss: 5.3199e-04 - mae: 0.0164 - 83ms/epoch - 3ms/step
    Epoch 62/200
    32/32 - 0s - loss: 5.0193e-04 - mae: 0.0160 - 74ms/epoch - 2ms/step
    Epoch 63/200
    32/32 - 0s - loss: 5.6915e-04 - mae: 0.0170 - 73ms/epoch - 2ms/step
    Epoch 64/200
    32/32 - 0s - loss: 5.8495e-04 - mae: 0.0168 - 80ms/epoch - 2ms/step
    Epoch 65/200
    32/32 - 0s - loss: 4.8947e-04 - mae: 0.0158 - 76ms/epoch - 2ms/step
    Epoch 66/200
    32/32 - 0s - loss: 4.7924e-04 - mae: 0.0155 - 74ms/epoch - 2ms/step
    Epoch 67/200
    32/32 - 0s - loss: 4.8316e-04 - mae: 0.0155 - 72ms/epoch - 2ms/step
    Epoch 68/200
    32/32 - 0s - loss: 4.9431e-04 - mae: 0.0156 - 71ms/epoch - 2ms/step
    Epoch 69/200
    32/32 - 0s - loss: 5.2266e-04 - mae: 0.0161 - 71ms/epoch - 2ms/step
    Epoch 70/200
    32/32 - 0s - loss: 4.8642e-04 - mae: 0.0156 - 77ms/epoch - 2ms/step
    Epoch 71/200
    32/32 - 0s - loss: 4.6464e-04 - mae: 0.0152 - 74ms/epoch - 2ms/step
    Epoch 72/200
    32/32 - 0s - loss: 4.6324e-04 - mae: 0.0152 - 93ms/epoch - 3ms/step
    Epoch 73/200
    32/32 - 0s - loss: 4.4600e-04 - mae: 0.0150 - 91ms/epoch - 3ms/step
    Epoch 74/200
    32/32 - 0s - loss: 4.6823e-04 - mae: 0.0152 - 74ms/epoch - 2ms/step
    Epoch 75/200
    32/32 - 0s - loss: 4.4382e-04 - mae: 0.0149 - 125ms/epoch - 4ms/step
    Epoch 76/200
    32/32 - 0s - loss: 4.5373e-04 - mae: 0.0151 - 118ms/epoch - 4ms/step
    Epoch 77/200
    32/32 - 0s - loss: 4.4012e-04 - mae: 0.0148 - 129ms/epoch - 4ms/step
    Epoch 78/200
    32/32 - 0s - loss: 4.6277e-04 - mae: 0.0152 - 110ms/epoch - 3ms/step
    Epoch 79/200
    32/32 - 0s - loss: 4.5105e-04 - mae: 0.0149 - 106ms/epoch - 3ms/step
    Epoch 80/200
    32/32 - 0s - loss: 4.5840e-04 - mae: 0.0150 - 109ms/epoch - 3ms/step
    Epoch 81/200
    32/32 - 0s - loss: 4.2489e-04 - mae: 0.0144 - 104ms/epoch - 3ms/step
    Epoch 82/200
    32/32 - 0s - loss: 4.4909e-04 - mae: 0.0151 - 107ms/epoch - 3ms/step
    Epoch 83/200
    32/32 - 0s - loss: 4.7092e-04 - mae: 0.0151 - 113ms/epoch - 4ms/step
    Epoch 84/200
    32/32 - 0s - loss: 4.3820e-04 - mae: 0.0146 - 106ms/epoch - 3ms/step
    Epoch 85/200
    32/32 - 0s - loss: 4.1779e-04 - mae: 0.0141 - 110ms/epoch - 3ms/step
    Epoch 86/200
    32/32 - 0s - loss: 4.4816e-04 - mae: 0.0148 - 97ms/epoch - 3ms/step
    Epoch 87/200
    32/32 - 0s - loss: 4.9621e-04 - mae: 0.0152 - 104ms/epoch - 3ms/step
    Epoch 88/200
    32/32 - 0s - loss: 4.9857e-04 - mae: 0.0154 - 102ms/epoch - 3ms/step
    Epoch 89/200
    32/32 - 0s - loss: 3.9763e-04 - mae: 0.0140 - 112ms/epoch - 4ms/step
    Epoch 90/200
    32/32 - 0s - loss: 4.0367e-04 - mae: 0.0141 - 111ms/epoch - 3ms/step
    Epoch 91/200
    32/32 - 0s - loss: 4.0055e-04 - mae: 0.0140 - 117ms/epoch - 4ms/step
    Epoch 92/200
    32/32 - 0s - loss: 4.0395e-04 - mae: 0.0140 - 113ms/epoch - 4ms/step
    Epoch 93/200
    32/32 - 0s - loss: 4.5411e-04 - mae: 0.0150 - 101ms/epoch - 3ms/step
    Epoch 94/200
    32/32 - 0s - loss: 4.9099e-04 - mae: 0.0152 - 75ms/epoch - 2ms/step
    Epoch 95/200
    32/32 - 0s - loss: 4.1149e-04 - mae: 0.0141 - 72ms/epoch - 2ms/step
    Epoch 96/200
    32/32 - 0s - loss: 4.2530e-04 - mae: 0.0144 - 72ms/epoch - 2ms/step
    Epoch 97/200
    32/32 - 0s - loss: 4.1686e-04 - mae: 0.0143 - 73ms/epoch - 2ms/step
    Epoch 98/200
    32/32 - 0s - loss: 4.1313e-04 - mae: 0.0142 - 72ms/epoch - 2ms/step
    Epoch 99/200
    32/32 - 0s - loss: 4.1895e-04 - mae: 0.0144 - 74ms/epoch - 2ms/step
    Epoch 100/200
    32/32 - 0s - loss: 3.8473e-04 - mae: 0.0136 - 72ms/epoch - 2ms/step
    Epoch 101/200
    32/32 - 0s - loss: 3.8215e-04 - mae: 0.0136 - 72ms/epoch - 2ms/step
    Epoch 102/200
    32/32 - 0s - loss: 4.1094e-04 - mae: 0.0141 - 73ms/epoch - 2ms/step
    Epoch 103/200
    32/32 - 0s - loss: 3.7519e-04 - mae: 0.0134 - 81ms/epoch - 3ms/step
    Epoch 104/200
    32/32 - 0s - loss: 3.9728e-04 - mae: 0.0138 - 80ms/epoch - 2ms/step
    Epoch 105/200
    32/32 - 0s - loss: 3.7046e-04 - mae: 0.0134 - 72ms/epoch - 2ms/step
    Epoch 106/200
    32/32 - 0s - loss: 3.9137e-04 - mae: 0.0137 - 72ms/epoch - 2ms/step
    Epoch 107/200
    32/32 - 0s - loss: 4.2256e-04 - mae: 0.0141 - 73ms/epoch - 2ms/step
    Epoch 108/200
    32/32 - 0s - loss: 3.7241e-04 - mae: 0.0133 - 72ms/epoch - 2ms/step
    Epoch 109/200
    32/32 - 0s - loss: 3.5831e-04 - mae: 0.0132 - 73ms/epoch - 2ms/step
    Epoch 110/200
    32/32 - 0s - loss: 3.6610e-04 - mae: 0.0134 - 80ms/epoch - 2ms/step
    Epoch 111/200
    32/32 - 0s - loss: 3.6970e-04 - mae: 0.0133 - 76ms/epoch - 2ms/step
    Epoch 112/200
    32/32 - 0s - loss: 3.8614e-04 - mae: 0.0137 - 74ms/epoch - 2ms/step
    Epoch 113/200
    32/32 - 0s - loss: 3.7456e-04 - mae: 0.0132 - 73ms/epoch - 2ms/step
    Epoch 114/200
    32/32 - 0s - loss: 4.3653e-04 - mae: 0.0143 - 95ms/epoch - 3ms/step
    Epoch 115/200
    32/32 - 0s - loss: 3.7622e-04 - mae: 0.0132 - 73ms/epoch - 2ms/step
    Epoch 116/200
    32/32 - 0s - loss: 3.4178e-04 - mae: 0.0128 - 76ms/epoch - 2ms/step
    Epoch 117/200
    32/32 - 0s - loss: 3.7068e-04 - mae: 0.0133 - 73ms/epoch - 2ms/step
    Epoch 118/200
    32/32 - 0s - loss: 3.5657e-04 - mae: 0.0131 - 76ms/epoch - 2ms/step
    Epoch 119/200
    32/32 - 0s - loss: 3.8231e-04 - mae: 0.0134 - 72ms/epoch - 2ms/step
    Epoch 120/200
    32/32 - 0s - loss: 4.1307e-04 - mae: 0.0139 - 72ms/epoch - 2ms/step
    Epoch 121/200
    32/32 - 0s - loss: 3.6397e-04 - mae: 0.0132 - 75ms/epoch - 2ms/step
    Epoch 122/200
    32/32 - 0s - loss: 3.7095e-04 - mae: 0.0133 - 75ms/epoch - 2ms/step
    Epoch 123/200
    32/32 - 0s - loss: 3.3074e-04 - mae: 0.0126 - 75ms/epoch - 2ms/step
    Epoch 124/200
    32/32 - 0s - loss: 3.3486e-04 - mae: 0.0126 - 75ms/epoch - 2ms/step
    Epoch 125/200
    32/32 - 0s - loss: 3.5071e-04 - mae: 0.0129 - 74ms/epoch - 2ms/step
    Epoch 126/200
    32/32 - 0s - loss: 3.4352e-04 - mae: 0.0129 - 71ms/epoch - 2ms/step
    Epoch 127/200
    32/32 - 0s - loss: 3.4822e-04 - mae: 0.0129 - 74ms/epoch - 2ms/step
    Epoch 128/200
    32/32 - 0s - loss: 3.4543e-04 - mae: 0.0128 - 74ms/epoch - 2ms/step
    Epoch 129/200
    32/32 - 0s - loss: 3.5017e-04 - mae: 0.0130 - 79ms/epoch - 2ms/step
    Epoch 130/200
    32/32 - 0s - loss: 3.4213e-04 - mae: 0.0126 - 70ms/epoch - 2ms/step
    Epoch 131/200
    32/32 - 0s - loss: 3.4565e-04 - mae: 0.0128 - 71ms/epoch - 2ms/step
    Epoch 132/200
    32/32 - 0s - loss: 3.5055e-04 - mae: 0.0129 - 74ms/epoch - 2ms/step
    Epoch 133/200
    32/32 - 0s - loss: 3.4128e-04 - mae: 0.0127 - 77ms/epoch - 2ms/step
    Epoch 134/200
    32/32 - 0s - loss: 3.2988e-04 - mae: 0.0125 - 71ms/epoch - 2ms/step
    Epoch 135/200
    32/32 - 0s - loss: 3.4781e-04 - mae: 0.0128 - 71ms/epoch - 2ms/step
    Epoch 136/200
    32/32 - 0s - loss: 3.5148e-04 - mae: 0.0129 - 72ms/epoch - 2ms/step
    Epoch 137/200
    32/32 - 0s - loss: 3.7935e-04 - mae: 0.0134 - 74ms/epoch - 2ms/step
    Epoch 138/200
    32/32 - 0s - loss: 3.5380e-04 - mae: 0.0130 - 71ms/epoch - 2ms/step
    Epoch 139/200
    32/32 - 0s - loss: 3.2781e-04 - mae: 0.0124 - 71ms/epoch - 2ms/step
    Epoch 140/200
    32/32 - 0s - loss: 3.4731e-04 - mae: 0.0128 - 73ms/epoch - 2ms/step
    Epoch 141/200
    32/32 - 0s - loss: 3.1840e-04 - mae: 0.0123 - 71ms/epoch - 2ms/step
    Epoch 142/200
    32/32 - 0s - loss: 3.6502e-04 - mae: 0.0129 - 82ms/epoch - 3ms/step
    Epoch 143/200
    32/32 - 0s - loss: 3.9574e-04 - mae: 0.0135 - 73ms/epoch - 2ms/step
    Epoch 144/200
    32/32 - 0s - loss: 4.0407e-04 - mae: 0.0138 - 74ms/epoch - 2ms/step
    Epoch 145/200
    32/32 - 0s - loss: 3.1167e-04 - mae: 0.0122 - 72ms/epoch - 2ms/step
    Epoch 146/200
    32/32 - 0s - loss: 3.4077e-04 - mae: 0.0125 - 72ms/epoch - 2ms/step
    Epoch 147/200
    32/32 - 0s - loss: 3.0638e-04 - mae: 0.0121 - 73ms/epoch - 2ms/step
    Epoch 148/200
    32/32 - 0s - loss: 3.2214e-04 - mae: 0.0123 - 74ms/epoch - 2ms/step
    Epoch 149/200
    32/32 - 0s - loss: 3.0699e-04 - mae: 0.0121 - 84ms/epoch - 3ms/step
    Epoch 150/200
    32/32 - 0s - loss: 3.1154e-04 - mae: 0.0120 - 77ms/epoch - 2ms/step
    Epoch 151/200
    32/32 - 0s - loss: 3.0692e-04 - mae: 0.0120 - 79ms/epoch - 2ms/step
    Epoch 152/200
    32/32 - 0s - loss: 3.0605e-04 - mae: 0.0120 - 79ms/epoch - 2ms/step
    Epoch 153/200
    32/32 - 0s - loss: 3.6949e-04 - mae: 0.0133 - 77ms/epoch - 2ms/step
    Epoch 154/200
    32/32 - 0s - loss: 2.9824e-04 - mae: 0.0118 - 84ms/epoch - 3ms/step
    Epoch 155/200
    32/32 - 0s - loss: 3.0567e-04 - mae: 0.0120 - 80ms/epoch - 2ms/step
    Epoch 156/200
    32/32 - 0s - loss: 3.4803e-04 - mae: 0.0127 - 71ms/epoch - 2ms/step
    Epoch 157/200
    32/32 - 0s - loss: 3.3916e-04 - mae: 0.0123 - 72ms/epoch - 2ms/step
    Epoch 158/200
    32/32 - 0s - loss: 3.1384e-04 - mae: 0.0119 - 73ms/epoch - 2ms/step
    Epoch 159/200
    32/32 - 0s - loss: 3.0996e-04 - mae: 0.0120 - 75ms/epoch - 2ms/step
    Epoch 160/200
    32/32 - 0s - loss: 3.0811e-04 - mae: 0.0120 - 77ms/epoch - 2ms/step
    Epoch 161/200
    32/32 - 0s - loss: 3.3993e-04 - mae: 0.0126 - 81ms/epoch - 3ms/step
    Epoch 162/200
    32/32 - 0s - loss: 3.8267e-04 - mae: 0.0132 - 72ms/epoch - 2ms/step
    Epoch 163/200
    32/32 - 0s - loss: 3.3970e-04 - mae: 0.0124 - 71ms/epoch - 2ms/step
    Epoch 164/200
    32/32 - 0s - loss: 3.0189e-04 - mae: 0.0119 - 73ms/epoch - 2ms/step
    Epoch 165/200
    32/32 - 0s - loss: 2.9950e-04 - mae: 0.0118 - 75ms/epoch - 2ms/step
    Epoch 166/200
    32/32 - 0s - loss: 2.8719e-04 - mae: 0.0116 - 73ms/epoch - 2ms/step
    Epoch 167/200
    32/32 - 0s - loss: 2.9255e-04 - mae: 0.0118 - 85ms/epoch - 3ms/step
    Epoch 168/200
    32/32 - 0s - loss: 2.9890e-04 - mae: 0.0118 - 71ms/epoch - 2ms/step
    Epoch 169/200
    32/32 - 0s - loss: 3.1602e-04 - mae: 0.0120 - 73ms/epoch - 2ms/step
    Epoch 170/200
    32/32 - 0s - loss: 3.0281e-04 - mae: 0.0119 - 71ms/epoch - 2ms/step
    Epoch 171/200
    32/32 - 0s - loss: 2.9731e-04 - mae: 0.0118 - 70ms/epoch - 2ms/step
    Epoch 172/200
    32/32 - 0s - loss: 2.9333e-04 - mae: 0.0119 - 75ms/epoch - 2ms/step
    Epoch 173/200
    32/32 - 0s - loss: 3.0321e-04 - mae: 0.0117 - 72ms/epoch - 2ms/step
    Epoch 174/200
    32/32 - 0s - loss: 2.9333e-04 - mae: 0.0116 - 74ms/epoch - 2ms/step
    Epoch 175/200
    32/32 - 0s - loss: 3.0638e-04 - mae: 0.0119 - 76ms/epoch - 2ms/step
    Epoch 176/200
    32/32 - 0s - loss: 2.9807e-04 - mae: 0.0116 - 73ms/epoch - 2ms/step
    Epoch 177/200
    32/32 - 0s - loss: 3.2478e-04 - mae: 0.0124 - 72ms/epoch - 2ms/step
    Epoch 178/200
    32/32 - 0s - loss: 3.6046e-04 - mae: 0.0129 - 73ms/epoch - 2ms/step
    Epoch 179/200
    32/32 - 0s - loss: 3.0372e-04 - mae: 0.0117 - 99ms/epoch - 3ms/step
    Epoch 180/200
    32/32 - 0s - loss: 3.7353e-04 - mae: 0.0129 - 85ms/epoch - 3ms/step
    Epoch 181/200
    32/32 - 0s - loss: 3.5541e-04 - mae: 0.0126 - 74ms/epoch - 2ms/step
    Epoch 182/200
    32/32 - 0s - loss: 2.8848e-04 - mae: 0.0116 - 74ms/epoch - 2ms/step
    Epoch 183/200
    32/32 - 0s - loss: 2.8118e-04 - mae: 0.0114 - 72ms/epoch - 2ms/step
    Epoch 184/200
    32/32 - 0s - loss: 2.8264e-04 - mae: 0.0115 - 72ms/epoch - 2ms/step
    Epoch 185/200
    32/32 - 0s - loss: 3.0003e-04 - mae: 0.0117 - 74ms/epoch - 2ms/step
    Epoch 186/200
    32/32 - 0s - loss: 2.7264e-04 - mae: 0.0113 - 74ms/epoch - 2ms/step
    Epoch 187/200
    32/32 - 0s - loss: 2.8654e-04 - mae: 0.0115 - 72ms/epoch - 2ms/step
    Epoch 188/200
    32/32 - 0s - loss: 2.8908e-04 - mae: 0.0116 - 79ms/epoch - 2ms/step
    Epoch 189/200
    32/32 - 0s - loss: 2.8704e-04 - mae: 0.0114 - 70ms/epoch - 2ms/step
    Epoch 190/200
    32/32 - 0s - loss: 2.7688e-04 - mae: 0.0113 - 72ms/epoch - 2ms/step
    Epoch 191/200
    32/32 - 0s - loss: 2.7375e-04 - mae: 0.0113 - 71ms/epoch - 2ms/step
    Epoch 192/200
    32/32 - 0s - loss: 2.9470e-04 - mae: 0.0114 - 82ms/epoch - 3ms/step
    Epoch 193/200
    32/32 - 0s - loss: 3.0442e-04 - mae: 0.0117 - 73ms/epoch - 2ms/step
    Epoch 194/200
    32/32 - 0s - loss: 2.7821e-04 - mae: 0.0113 - 81ms/epoch - 3ms/step
    Epoch 195/200
    32/32 - 0s - loss: 2.8987e-04 - mae: 0.0114 - 81ms/epoch - 3ms/step
    Epoch 196/200
    32/32 - 0s - loss: 2.8397e-04 - mae: 0.0114 - 83ms/epoch - 3ms/step
    Epoch 197/200
    32/32 - 0s - loss: 2.7643e-04 - mae: 0.0113 - 81ms/epoch - 3ms/step
    Epoch 198/200
    32/32 - 0s - loss: 2.9048e-04 - mae: 0.0116 - 76ms/epoch - 2ms/step
    Epoch 199/200
    32/32 - 0s - loss: 3.7589e-04 - mae: 0.0130 - 71ms/epoch - 2ms/step
    Epoch 200/200
    32/32 - 0s - loss: 3.2895e-04 - mae: 0.0123 - 70ms/epoch - 2ms/step



```python
#Prevendo os 143 últimos preços de ação, baseado nos 90 últimos.

inputs = new_data[len(new_data) - len(valid) - 90:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(90,inputs.shape[0]):
    X_test.append(inputs[i-90:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = modelo.predict(X_test)
```

    5/5 [==============================] - 0s 2ms/step



```python
closing_price = scaler.inverse_transform(closing_price)[:, [0]]
```


```python
#Visualizando a Previsão
plt.rcParams.update({'font.size': 15})


plt.figure(figsize=(15,10))
train = new_data[:1100]
t_2020 = train['2020']
valid = new_data[1100:]
valid['Predictions'] = closing_price
plt.ylabel('Preço da Ação')
plt.xlabel('Data')
plt.plot(train['Close'], label = "Treino")
plt.plot(valid['Close'], label = 'Observado')
plt.plot(valid['Predictions'], label = 'Previsão')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
```

     


![output_15_2](https://user-images.githubusercontent.com/118861107/230731485-c2e5885b-9da3-4a1a-87a2-47dd29af48f0.png)


    

    



