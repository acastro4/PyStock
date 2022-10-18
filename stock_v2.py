import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
import yfinance as yf

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

def get_historical_data(symbol, start_date, end_date, period="1d"):
    df = yf.download(symbol, start_date, end_date, period=period)
    return df
inicio = "2022-01-01"
fin = "2022-09-11"
nombre = "SQM-B.SN"
stock = get_historical_data(nombre, inicio , fin)

def ADX(df: pd.DataFrame(), interval: int=14):
    df['-DM'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = df['High'] - df['High'].shift(1)
    df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM']>0), df['+DM'], 0.0)
    df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM']>0), df['-DM'], 0.0)
    df['TR_TMP1'] = df['High'] - df['Low']
    df['TR_TMP2'] = np.abs(df['High'] - df['Adj Close'].shift(1))
    df['TR_TMP3'] = np.abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR'] = df[['TR_TMP1', 'TR_TMP2', 'TR_TMP3']].max(axis=1)
    df['TR'+str(interval)] = df['TR'].rolling(interval).sum()
    df['+DMI'+str(interval)] = df['+DM'].rolling(interval).sum()
    df['-DMI'+str(interval)] = df['-DM'].rolling(interval).sum()
    df['+DI'+str(interval)] = df['+DMI'+str(interval)] /   df['TR'+str(interval)]*100
    df['-DI'+str(interval)] = df['-DMI'+str(interval)] / df['TR'+str(interval)]*100
    df['DI'+str(interval)+'-'] = abs(df['+DI'+str(interval)] - df['-DI'+str(interval)])
    df['DI'+str(interval)] = df['+DI'+str(interval)] + df['-DI'+str(interval)]
    df['DX'] = (df['DI'+str(interval)+'-'] / df['DI'+str(interval)])*100
    df['ADX'+str(interval)] = df['DX'].rolling(interval).mean()
    df['ADX'+str(interval)] =   df['ADX'+str(interval)].fillna(df['ADX'+str(interval)].mean())
    del df['TR_TMP1'], df['TR_TMP2'], df['TR_TMP3'], df['TR'], df['TR'+str(interval)]
    del df['+DMI'+str(interval)], df['DI'+str(interval)+'-']
    del df['DI'+str(interval)], df['-DMI'+str(interval)]
    del df['+DI'+str(interval)], df['-DI'+str(interval)]
    del df['DX']
    return df

stock = ADX(stock, 14)
print(stock.tail())
ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
ax1.plot(stock['Close'], linewidth = 2, color = '#ff9800')
ax1.set_title('SQM-B.SN CLOSING PRICE')
ax2.plot(stock['ADX14'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')
ax2.legend()
ax2.set_title('SQM-B.SN ADX 14')
plt.show()