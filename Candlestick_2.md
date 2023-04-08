
**Bonus 1 Codigo CandleStick**

**1) Bibliotecas do codigo**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
```

**2) Regra de Donwload e MMA, basta trocar o Ticker da açao desejada e não esquecer o ".SA".**



```python
tickers = ['SOJA3.SA']
df = yf.download(tickers, start = '2022-01-01', period= '1d')

df['mm_20p'] = df.Close.rolling(20).mean()
df['max_mm_ant'] = df.mm_20p.rolling(2, closed='left' ).max()
df['min_mm_ant'] = df.mm_20p.rolling(2, closed='left' ).min()
df['cond_alta'] = (df.Close > df.mm_20p) & (df.mm_20p > df.max_mm_ant)
df['cond_baixa'] = (df.Close < df.mm_20p) & (df.mm_20p < df.min_mm_ant)

```

    [*********************100%***********************]  1 of 1 completed


**3)CandleStick Plot. Codigo simples para uso de acompanhamento de uma ação listada em bolsa.O grafico é dinâmico com todas as funcionalidades ativas basta ultilizar o menu a direita.**


```python

fig = go.Figure(data=[go.Candlestick(x=df.index,
                
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.add_trace(go.Scatter(name='Sma_20', x=df.index, y=df['mm_20p'], marker_color='gray'))
fig.update_layout(template='ggplot2', plot_bgcolor='black', title='CANDLESTICK PLOT ', width=1200, height=800)
fig.show()
```


![newplot](https://user-images.githubusercontent.com/118861107/230740744-8525094d-a99e-4ef0-a6a6-09bb2cf2815b.png)


