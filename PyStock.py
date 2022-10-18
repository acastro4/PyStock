from asyncio import selector_events
from functools import cache
from sqlite3 import Row
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
import yfinance as yf
from scipy.fft import fft
import math
from datetime import datetime
import time

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.floor(math.log2(x))

def get_historical_data(symbol, start_date, end_date, interval="1d"):
    df = yf.download(symbol, start_date, end_date, interval=interval)
    return df

def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    return plus_di, minus_di, adx_smooth

def stop_loss_day(name_stock, maxprice, date_i, date_i1, signal):
    
    my_time = str(date_i)[:10]
    my_time_1 = str(date_i1)[:10]
    data_day = yf.download(name_stock, start=my_time, end=my_time_1, interval="1h")
    max_price = maxprice
    for index, row in data_day.iterrows():
        #print(row)
        max_price = max(max_price, row["Close"])
        if (row["Close"] - max_price) / max_price <= -0.1:
            return True, row["Close"], index
    return False, 0, index

def implement_strategy(prices, open_prices, adx, macd, macd_signal, fibonnaci, threshold):
    buy_price = []
    max_price = 0
    sell_price = []
    adx_signal = []
    signal = -1
    
    for i in range(len(prices)):
        # print(f'PRECIO ACTUAL: {prices[i]}, PRECIO MAX: {max_price}')
        if adx[i] > 25 and macd[i] > macd_signal[i] and macd[i] > 0 and macd_signal[i] > 0:
            if signal != 1:
                # print(f'COMPRAR {prices[i]}')
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                adx_signal.append(signal)
                max_price = prices[i]
            else:
                """ if i + 2 <= len(prices):
                    stoploss, priceloss, date_time_close = stop_loss_day(name_stock, max_price, dates[i+1], dates[i+2], signal)
                    print(stoploss, priceloss, date_time_close)
                    if stoploss:
                        print(f'VENDER POR STOPLOSS {prices[i]}')
                        buy_price.append(np.nan)
                        sell_price.append(priceloss)
                        signal = -1
                        adx_signal.append(signal)
                        max_price = 0
                        continue """
                if (prices[i] - max_price) / max_price <= threshold:
                    # print(f'VENDER POR STOPLOSS {prices[i]}')
                    buy_price.append(np.nan)
                    if i + 1 <= len(prices):
                        sell_price.append(prices[i])
                    else:
                        sell_price.append(open_prices[i + 1])
                    signal = -1
                    adx_signal.append(signal)
                    max_price = 0
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    adx_signal.append(0)
                    max_price = max(max_price, prices[i])
        else:
            if signal != -1:
                # print(f'VENDER {prices[i]}')
                buy_price.append(np.nan)
                sell_price.append(open_prices[i + 1])
                signal = -1
                adx_signal.append(signal)
                max_price = 0
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                adx_signal.append(0)
                max_price = max(max_price, prices[i])
        """ elif adx[i] < 25 and prices[i] < fibonnaci and macd[i] < macd_signal[i]:
            if signal != -1:
                print('VENDER')
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                adx_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                adx_signal.append(0) """
        
        
    return buy_price, sell_price, adx_signal

### MACD CYCLES
def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

def main(sim_stocks=[], threshold=0.05):
    inicio = "2022-01-01"
    fin = "2022-09-11"
    total = 0
    for nombre in sim_stocks:
        try: 
            stock = get_historical_data(nombre, inicio, fin)
            stock['plus_di'] = pd.DataFrame(get_adx(stock['High'], stock['Low'], stock['Close'], 14)[0]).rename(columns = {0:'plus_di'})
            stock['minus_di'] = pd.DataFrame(get_adx(stock['High'], stock['Low'], stock['Close'], 14)[1]).rename(columns = {0:'minus_di'})
            stock['adx'] = pd.DataFrame(get_adx(stock['High'], stock['Low'], stock['Close'], 14)[2]).rename(columns = {0:'adx'})
            stock = stock.dropna()
            stock.tail()
            # print(stock)

            ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
            ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
            ax1.plot(stock['Close'], linewidth = 2, color = '#ff9800')
            ax1.set_title('SQM-B.SN CLOSING PRICE')
            ax2.plot(stock['plus_di'], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)
            ax2.plot(stock['minus_di'], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)
            ax2.plot(stock['adx'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
            ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')
            ax2.legend()
            ax2.set_title('SQM-B.SN ADX 14')
            # plt.show()

            ### DIFERENCIA POR DIA 
            stock['diff'] = stock['Close'] - stock.shift(1)['Close']
            stock = stock[-32:]
            ### APLICAR FAST FOURIER
            fourier_values = np.abs(fft(stock['diff'].values))
            fourier_values = reversed(list(fourier_values))
            diff_values = reversed(stock['diff'].to_list())
            fourier_df = pd.DataFrame(list(zip(diff_values, fourier_values)), columns=['Diff', 'Fourier'])
            fourier_df.index = pd.RangeIndex(start=1, stop=33, step=1)
            fourier_df = fourier_df.sort_values(by='Fourier', ascending=False)
            fourier_indexes = fourier_df.index.to_list()
            selected_indexes = fourier_indexes[:3]
            mac_d_cycles = sorted([math.ceil(x/2) for x in selected_indexes], reverse=True)        
            stock_macd = get_macd(stock['Close'], *mac_d_cycles)

            ### FIBONACCI
            maximum_price = stock['Close'].max()
            minimum_price = stock['Close'].min()
            difference = maximum_price - minimum_price #Get the difference        
            first_level = maximum_price - difference * 0.236   
            second_level = maximum_price - difference * 0.382  
            third_level = maximum_price - difference * 0.5     
            fourth_level = maximum_price - difference * 0.618
            fig_level = first_level
            buy_price, sell_price, adx_signal = implement_strategy(stock['Close'], stock['Open'], stock['adx'], stock_macd['macd'], stock_macd['signal'], fig_level, threshold)
            ax1 = plt.subplot2grid((17,1), (0,0), rowspan = 5, colspan = 1)
            ax2 = plt.subplot2grid((17,1), (6,0), rowspan = 5, colspan = 1)
            ax3 = plt.subplot2grid((17,1), (12,0), rowspan = 5, colspan = 1)
            ax1.plot(stock['Close'], linewidth = 3, color = '#ff9800', alpha = 0.6)
            ax1.axhline(fig_level, color = 'grey', linewidth = 2, linestyle = '--')
            ax1.set_title(f'{nombre} CLOSING PRICE')
            ax1.plot(stock.index, buy_price, marker = '^', color = '#26a69a', markersize = 14, linewidth = 0, label = 'BUY SIGNAL')
            ax1.plot(stock.index, sell_price, marker = 'v', color = '#f44336', markersize = 14, linewidth = 0, label = 'SELL SIGNAL')
            ax2.plot(stock['adx'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
            ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')
            ax2.legend()
            ax2.set_title(f'{nombre} ADX 14')
            ax3.plot(stock_macd['macd'], color = '#2196f3', label = 'MACD', linewidth = 3 )
            ax3.plot(stock_macd['signal'], color = '#C90037', label = 'MACD SIGNAL', linewidth = 3 )
            ax3.axhline(0, color = 'grey', linewidth = 2, linestyle = '--')
            ax3.legend()
            ax3.set_title(f'{nombre} MACD - Soft ({mac_d_cycles[0]}) - Fast ({mac_d_cycles[1]}) - Smooth ({mac_d_cycles[2]})')
            plt.savefig(f'images/spy/STOCKS_{nombre}.png')

            assets = 1000000
            stocks = 0
            position = -1
            for i in range(len(buy_price)):
                if buy_price[i] > 0:
                    position = 1
                    stocks =  assets / buy_price[i]
                elif sell_price[i] > 0 and position == 1:
                    position = -1
                    assets = stocks * sell_price[i]
            print(f'ASSETS: {assets - 1000000}')
            total += assets - 1000000
        except ValueError as error:
            print(error)
    print(f'GANANCIA NETAS {total} EN {len(sim_stocks)} ACTIVOS')

def save_sp500_tickers():
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)
        return tickers

if __name__ == '__main__':
    import bs4 as bs
    import pickle
    import requests

    main(sim_stocks=save_sp500_tickers()[:100], 
         threshold=-0.02)

    # ["SHOP", "CHILE.SN", "COPEC.SN", "BSANTANDER.SN", "ENELAM.SN", "CENCOSUD.SN", "CMPC.SN", "BCI.SN", "FALABELLA.SN", "VAPORES.SN", "ENELCHILE.SN", "QUINENCO.SN", "ANDINA-B.SN", "COLBUN.SN", "PARAUCO.SN", "CCU.SN", "ITAUCORP.SN", "CENCOSHOPP.SN", "CAP.SN", "AGUAS-A.SN", "CONCHATORO.SN", "ENTEL.SN", "MALLPLAZA.SN", "SMU.SN", "IAM.SN", "ECL.SN", "RIPLEY.SN", "SONDA.SN", "SECURITY.SN"]