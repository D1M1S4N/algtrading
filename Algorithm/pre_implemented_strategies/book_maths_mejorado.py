import numpy as np
import pandas as pd
from collections import deque
import scipy.stats as st
from scipy.signal import find_peaks
from scipy.stats import mode
from pandas_ta import *
import talib

class PSAR:
    def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
        self.max_af = max_af
        self.init_af = init_af
        self.af = init_af
        self.af_step = af_step
        self.extreme_point = None
        self.high_price_trend = []
        self.low_price_trend = []
        self.high_price_window = deque(maxlen=2)
        self.low_price_window = deque(maxlen=2)
        
        self.psar_list = []
        self.af_list = []
        self.ep_list = []
        self.high_list = []
        self.low_list = []
        self.trend_list = []
        self._num_days = 0
        
    def calc_psar(self, high, low):
        if len(self.low_price_window) <= 1:
            self.trend = None
            self.extreme_point = high
            return None

        if self.high_price_window[0] < self.high_price_window[1]:
            self.trend = 1
            psar = min(self.low_price_window)
            self.extreme_point = max(self.high_price_window)
        else: 
            self.trend = 0
            psar = max(self.high_price_window)
            self.extreme_point = min(self.low_price_window)

        return psar
    
    def _calcPSAR(self):
        prev_psar = self.psar_list[-1]
        
        if self.trend == 1:
            psar = prev_psar + self.af * (self.extreme_point - prev_psar)
            psar = min(psar, min(self.low_price_window))
        else:
            psar = prev_psar - self.af * (prev_psar - self.extreme_point)
            psar = max(psar, max(self.high_price_window))

        return psar
    
    def _updateCurrentVals(self, psar, high, low):
        if self.trend == 1:
            self.high_price_trend.append(high)
        elif self.trend == 0:
            self.low_price_trend.append(low)

        psar = self._trendReversal(psar, high, low)

        self.psar_list.append(psar)
        self.af_list.append(self.af)
        self.ep_list.append(self.extreme_point)
        self.high_list.append(high)
        self.low_list.append(low)
        self.high_price_window.append(high)
        self.low_price_window.append(low)
        self.trend_list.append(self.trend)

        return psar

    def _trendReversal(self, psar, high, low):
        reversal = False
        if self.trend == 1 and psar > low:
            self.trend = 0
            psar = max(self.high_price_trend)
            self.extreme_point = low
            reversal = True
        elif self.trend == 0 and psar < high:
            self.trend = 1
            psar = min(self.low_price_trend)
            self.extreme_point = high
            reversal = True

        if reversal:
            self.af = self.init_af
            self.high_price_trend.clear()
            self.low_price_trend.clear()
        else:
            if high > self.extreme_point and self.trend == 1:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = high
            elif low < self.extreme_point and self.trend == 0:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = low

        return psar
    
def log_return(price, previous_price) -> float:
    return np.log(price/previous_price)

def average_return(n, price, previous_price) -> float:
    return (1/n) * log_return(price, previous_price)

def variance(data:list):
    return np.var(data)

def momentum(data, n):
    return (data[-1]/data[-1-n])*100

def volatility(data:list):
    np.sqrt(variance(data))

def covariance(x:list, y:list):
    return np.cov(x, y)

def correlation(x, y):
    return np.corrcoef(x, y)

def vwap(price, volume):
    return np.cumsum(price*volume)/np.cumsum(volume)

def sma_function(data, n):
    return np.cumsum(data)/n

def ema_function(precio, anterior_ema, n):
    multiplier = 2/(n + 1)
    
    return (precio * multiplier) + (anterior_ema * (1 - multiplier))

def macd_function(ema12, ema26, anterior_macd, n = 9):
    macd = ema12 - ema26
    signal = ema_function(macd, anterior_macd, n)
    return macd, signal

def rsi(diff,fn_roll):
    change_up = diff.copy()
    change_down = diff.copy()

    change_up[change_up<0] = 0
    change_down[change_down>0] = 0

    diff.equals(change_up + change_down)
    avg_up = change_up.rolling(fn_roll).mean()
    avg_down = change_down.rolling(fn_roll).mean().abs()
    rsi = 100 * avg_up / (avg_up + avg_down)
    
    return rsi

def StochRSI(delta, period=14, smoothK=3, smoothD=3):
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] )
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] )
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)
 
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi, stochrsi_K, stochrsi_D

def StochRSI_EMA(delta, period=14, smoothK=3, smoothD=3):
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI 
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.ewm(span=smoothK).mean()
    stochrsi_D = stochrsi_K.ewm(span=smoothD).mean()

    return stochrsi, stochrsi_K, stochrsi_D

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

def price_volume_trend(data):
    for index, row in data.iterrows():
        if index > 0:
            last_val = data.at[index - 1, 'pvt']
            last_close = data.at[index - 1, 'close']
            today_close = row['close']
            today_vol = row['tick_volume']
            current_val = last_val + (today_vol * (today_close - last_close) / last_close)
        else:
            current_val = row['tick_volume']

        data._set_value(index, 'pvt', current_val)

    return data

def calculate_adaptive_ema(prices, N, smoothing=2):
    ema = [np.cumsum(prices[N:]) / N]

    sf = smoothing/(N + 1)

    for price in prices[1:]:
        volatility=abs(price-ema[-1])
        er=volatility/price
        sf=sf*(er*(sf-1)+1)
        ema.append((price * sf) + ema[-1] * (1 - sf))

    return ema

def mfi(data, window):
    money_flow = data['rmv']
    signal = np.where(money_flow>money_flow.shift(1),1,np.where(money_flow<money_flow.shift(1),-1,0))
    money_flow_s = money_flow*signal

    money_flow_positive = money_flow_s.rolling(window).apply(lambda x:sum(np.where(x>=0.0,x,0.0)),raw=True)
    money_flow_negative = abs(money_flow_s.rolling(window).apply(lambda x:sum(np.where(x<0.0,x,0.0)),raw=True))

    money_flow_ratio = money_flow_positive/money_flow_negative

    return 100 - (100/(1 + money_flow_ratio))

def get_wr(data_5m, lookback):
    high = data_5m['high']
    close = data_5m['close']
    low = data_5m['low']
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr

def get_vroc(data_5m, lookback):
    resultado = []
    for _, i in enumerate(data_5m['tick_volume']):
        if _ > lookback:
            resultado.append(100*((i-data_5m['tick_volume'][_-lookback])/data_5m['tick_volume'][_-lookback]))
        else:
            resultado.append(100*((i-data_5m['tick_volume'][0])/data_5m['tick_volume'][0]))

    return resultado

def get_nvi(data_5m):
    resultado = []

    for _, i in enumerate(data_5m['mean_price']):
        if len(resultado) == 0:
            resultado.append(i)
        else:
            resultado.append(((i-data_5m['mean_price'][_-1])/data_5m['mean_price'][_-1])*resultado[-1])

    return resultado

def get_cci(data_5m,length):
    TP = (data_5m['high'] + data_5m['low'] + data_5m['close']) / 3 
    cci = pd.Series((TP - TP.rolling(length).mean()) / (0.015*TP.rolling(length).std()),name = 'CCI') 
    return cci

def trix(data_5m,length):
    EX1 = data_5m['close'].ewm(span=length, min_periods=length).mean()
    EX2 = EX1.ewm(span=length, min_periods=length).mean()
    EX3 = EX2.ewm(span=length, min_periods=length).mean()
    i = 0
    ROC_l = [np.nan]
    while i + 1 <= data_5m.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1

    Trix = pd.Series(ROC_l, name='Trix_' + str(length))
    return Trix

def vortex(data_5m,length):
    i = 0
    TR = [0]
    while i < data_5m.index[-1]:
        Range = max(data_5m.loc[i + 1, 'high'], data_5m.loc[i, 'close']) - min(data_5m.loc[i + 1, 'low'], data_5m.loc[i, 'close'])
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < data_5m.index[-1]:
        Range = abs(data_5m.loc[i + 1, 'high'] - data_5m.loc[i, 'low']) - abs(data_5m.loc[i + 1, 'low'] - data_5m.loc[i, 'high'])
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.Series(VM).rolling(length).sum() / pd.Series(TR).rolling(length).sum(), name='Vortex_' + str(length))
    return VI

def adr(data_5m):
    high_low=data_5m['high']-data_5m['low']
    ADR=np.cumsum(high_low)/(data_5m.index+1)
    return ADR

def calcular_moda_max(sub_df):
    if len(sub_df) < 1:
        return np.nan
    peaks_max, _ = find_peaks(sub_df, height=np.mean(sub_df))
    max_vals = sub_df.iloc[peaks_max] if len(peaks_max) > 0 else np.array([])
    return mode(max_vals, keepdims=True).mode[0] if len(max_vals) > 0 else np.nan

# Función para calcular la moda de los picos mínimos
def calcular_moda_min(sub_df):
    if len(sub_df) < 1:
        return np.nan
    peaks_min, _ = find_peaks(-sub_df, height=-np.mean(sub_df))
    min_vals = sub_df.iloc[peaks_min] if len(peaks_min) > 0 else np.array([])
    return mode(min_vals, keepdims=True).mode[0] if len(min_vals) > 0 else np.nan

class Maths:
    def __init__(self,ema_length,ema_length2,ema_length3,sma_length_2,sma_length_3,length_macd,rsi_roll,atr_ma,stoch_rsi,psar_parameters,bollinger_sma,adx_window,obv_ema,ema_length4,sma_length_4,sma_length_5,length_macd2,mfi_length,pvt_length,adl_ema,wr_length,vroc,nvi,momentum,cci,bull_bear_power,mass_index,trix,vortex):
        self.hist_price=[]
        
        #-----------MACD-----------
        self.length_macd=length_macd
        self.length_macd2=length_macd2
        self.macd_last=None
        self.macd=[]
        self.macd_signal=[]

        #-----------EMA-----------
        self.ema_length=ema_length
        self.ema_length2=ema_length2
        self.ema_length3=ema_length3
        self.ema_length4=ema_length4

        #-----------SMA2-----------
        self.sma_length_2=sma_length_2
        self.sma_length_3=sma_length_3
        self.sma_length_4=sma_length_4
        self.sma_length_5=sma_length_5

        #-----------VWAP-----------
        self.last_price=0
        self.last_volume=0
        self.vwap=[]

        #-----------TWAP-----------
        self.last_twap_price=0
        self.tamanyo_twap=0
        self.twap=[]

        #-----------RSI-----------
        self.rsi=[]
        self.rsi_roll=rsi_roll
        self.stoch_rsi=stoch_rsi

        #-----------Volume-----------
        self.mean_volume=[]

        #-----------STATS-----------
        self.stdev=[]

        #-----------ATR-----------
        self.last_atr=None
        self.atr_window=14
        self.atr_ma=atr_ma
        self.atr=[]

        #-----------PSAR-----------
        self.indic = PSAR(init_af=psar_parameters[0], max_af=psar_parameters[1], af_step=psar_parameters[2])

        #-----------Bollinger-----------
        self.bollinger_sma=bollinger_sma

        #-----------ADX-----------
        self.adx_window=adx_window

        #-----------OBV-----------  
        self.obv_ema=obv_ema

        #-----------MFI-----------  
        self.mfi_length=mfi_length

        #-----------PVT-----------  
        self.pvt_length=pvt_length

        self.adl_ema=adl_ema

        self.wr_length=wr_length

        self.vroc=vroc

        self.nvi=nvi

        self.momentum=momentum

        self.cci=cci

        self.bull_bear_power=bull_bear_power

        self.mass_index=mass_index

        self.trix=trix
        self.vortex=vortex

        super().__init__()
        
    # Números fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597

    def media_acumulada(self, columna):
        return columna.mean()

    def detectar_outliers(self,df, columna):
        Q1 = df[columna].expanding().quantile(0.9,interpolation='linear')
        Q3 = df[columna].expanding().quantile(0.1,interpolation='linear')

        df['Q3']=Q3
        df['Q1']=Q1

        IQR=Q3-Q1
        
        df['IQR']=IQR
        df['limite_superior'] = Q1 - 2 * IQR
        df['limite_inferior'] = Q1 - 1 * IQR

    def std_acumulada(self,columna):
        return columna.rolling(window = 21).std() # Probar

    def df(self,data_5m):
        #rsi_value,k,d=self.get_rsi(data_5m)   
        #self.detectar_outliers(data_5m,"spread")
        stoch_rsi_df=stochrsi(data_5m['mean_price'],length=self.rsi_roll,rsi_length=self.rsi_roll,k=self.stoch_rsi[0],d=self.stoch_rsi[1])

        data_5m['moda_max'] = data_5m['spread'].expanding().apply(calcular_moda_max, raw=False)
        data_5m['moda_min'] = data_5m['spread'].expanding().apply(calcular_moda_min, raw=False)
        data_5m['moda_diff'] = data_5m['moda_max']-data_5m['moda_min']
        data_5m['moda_inferior'] = data_5m['moda_max']
        data_5m['moda_superior1'] = data_5m['moda_max']+data_5m['moda_min']
        data_5m['moda_superior2'] = data_5m['moda_max']+data_5m['moda_diff']

        data_5m['stochrsi_K'] = stoch_rsi_df[stoch_rsi_df.columns[0]]
        data_5m['stochrsi_D'] = stoch_rsi_df[stoch_rsi_df.columns[1]]
        data_5m['rsi']=stoch_rsi_df[stoch_rsi_df.columns[0]]

        data_5m['ema'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length, adjust=False, min_periods=self.ema_length))
        data_5m['ema2'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length2, adjust=False, min_periods=self.ema_length2))
        data_5m['ema3'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length3, adjust=False, min_periods=self.ema_length3))
        data_5m['ema4'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length4, adjust=False, min_periods=self.ema_length4))

        data_5m['impulso'] = data_5m['ema'].diff()

        data_5m['macd_ma'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_2, adjust=False, min_periods=self.sma_length_2))
        data_5m['macd_ma2'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_3, adjust=False, min_periods=self.sma_length_3))
        data_5m['macd'] = data_5m['macd_ma']-data_5m['macd_ma2']
        data_5m['macd_signal']=self.media_acumulada(data_5m['macd'].ewm(span=self.length_macd, adjust=False, min_periods=self.length_macd))
        data_5m['2macd_ma'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_4, adjust=False, min_periods=self.sma_length_4))
        data_5m['2macd_ma2'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_5, adjust=False, min_periods=self.sma_length_5))
        data_5m['2macd'] = data_5m['2macd_ma']-data_5m['2macd_ma2']
        data_5m['2macd_signal'] = self.media_acumulada(data_5m['2macd'].ewm(span=self.length_macd2, adjust=False, min_periods=self.length_macd2))

        data_5m['vwap'] = np.cumsum(data_5m["mean_price"]*data_5m["tick_volume"])/np.cumsum(data_5m["tick_volume"])
        data_5m['twap'] = np.cumsum(data_5m["mean_price"])/(data_5m.index+1)
        
        data_5m['mean_volume'] = np.cumsum(data_5m["tick_volume"])/(data_5m.index+1)
                
        data_5m['PSAR'] = data_5m.apply(lambda x: self.indic.calcPSAR(x['high'], x['low']), axis=1)
        
        plus_di, minus_di, adx_smooth = get_adx(data_5m["high"],data_5m["low"],data_5m["close"],self.adx_window)
        data_5m['adx'] = adx_smooth
        data_5m['plus_di'] = plus_di #+dmi
        data_5m['minus_di'] = minus_di #-dmi

        data_5m['obv'] = (np.sign(data_5m['close'].diff()) * data_5m['tick_volume']).fillna(0).cumsum()
        data_5m['obv_ema'] = self.media_acumulada(data_5m['obv'].ewm(span=self.obv_ema, adjust=False, min_periods=self.obv_ema))

        data_5m['mfi'] = mfi(data_5m,self.mfi_length)
        data_5m=price_volume_trend(data_5m)
        data_5m['pvt_ema'] = self.media_acumulada(data_5m['pvt'].ewm(span=self.pvt_length, adjust=False, min_periods=self.pvt_length))

        data_5m['momentum'] = np.where(data_5m['close'].diff() > 0, 1, -1)
        data_5m['momentum_ema'] = self.media_acumulada(data_5m['momentum'].ewm(span=self.momentum, adjust=False, min_periods=self.momentum))

        data_5m['mfm'] = ((data_5m['close']-data_5m['low'])-(data_5m['high']-data_5m['close']))/(data_5m['high']-data_5m['low'])
        data_5m['mfv'] = data_5m['mfm']*data_5m['tick_volume']
        
        adl=[]

        for i in range(len(data_5m)):
          
          if i == 0:
            adl.append(data_5m['mfv'][i])
            
          else:
            adl.append(data_5m['mfv'][i]+adl[-1])

        data_5m['adl'] = adl
        data_5m['adl_ema'] = self.media_acumulada(data_5m['adl'].ewm(span=self.adl_ema, adjust=False, min_periods=self.adl_ema))

        data_5m['williams_r'] = get_wr(data_5m,self.wr_length[0])
        data_5m['williams_r_ema'] = self.media_acumulada(data_5m['williams_r'].ewm(span=self.wr_length[1], adjust=False, min_periods=self.wr_length[1]))
        data_5m['vroc'] = get_vroc(data_5m,self.vroc)
        data_5m['nvi'] = get_nvi(data_5m)
        data_5m['nvi_ema'] = self.media_acumulada(data_5m['nvi'].ewm(span=self.nvi, adjust=False, min_periods=self.nvi)) # 255?
        data_5m['bulls_bears_powers_ema'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.bull_bear_power, adjust=False, min_periods=self.bull_bear_power))
        data_5m['bulls_powers'] = data_5m['high']-data_5m['bulls_bears_powers_ema']
        data_5m['bears_powers'] = data_5m['low']-data_5m['bulls_bears_powers_ema']
        data_5m['cci'] = get_cci(data_5m,self.cci[0])
        data_5m['cci_ema'] = self.media_acumulada(data_5m['cci'].ewm(span=self.cci[1], adjust=False, min_periods=self.cci[1]))

        data_5m['trix'] = trix(data_5m,self.trix)
        data_5m['vortex'] = vortex(data_5m,self.vortex)

        data_5m['adr'] = adr(data_5m)

        smi_df=smi(data_5m['close']) #smi, signal, oscillator columns
        data_5m['smi'] = smi_df[smi_df.columns[0]]
        data_5m['smi_signal'] = smi_df[smi_df.columns[1]]

        #data_5m['stc'] = stc(data_5m['close'])
        data_5m['cmo'] = cmo(data_5m['close'])

        keltner_channels = kc(data_5m['high'],data_5m['low'],data_5m['close'])
        data_5m['kc_up'] = keltner_channels[keltner_channels.columns[0]]
        data_5m['kc_basis'] = keltner_channels[keltner_channels.columns[1]]
        data_5m['kc_down'] = keltner_channels[keltner_channels.columns[2]] 

        #data_5m['stdv-z-score'] = (data_5m['diff'] - data_5m['diff'].mean())/data_5m['stdv']
        #data_5m['stdv-p-value'] = st.norm.ppf(data_5m['stdv-z-score'])
        #data_5m['atr-z-score'] = (data_5m['atr'].pct_change() - data_5m['atr'].pct_change().mean())/data_5m['atr'].pct_change().std(ddof=1)
        #data_5m['atr-p-value'] = st.norm.ppf(data_5m['atr-z-score'])

        #statistics
        data_5m["entropy"] = entropy(data_5m['close'])
        data_5m["kurtosis"] = kurtosis(data_5m['close'])
        data_5m["mad"] = mad(data_5m['close'])
        data_5m["median"] = median(data_5m['close'])
        data_5m["skew"] = skew(data_5m['close'])
        data_5m['stdv'] = self.std_acumulada(abs(data_5m['diff']))
        data_5m['stdv_price'] = self.std_acumulada(data_5m['close'])

        data_5m["variance"] = variance(data_5m['close'])
        data_5m["zscore"] = zscore(data_5m['close'])

        # Calcular el coeficiente de variación (CV)
        data_5m['cv_5m'] = (data_5m['stdv'] / data_5m['ema']).where(data_5m['ema'] != 0, 0)

        # Suavizar el CV con una media móvil para reducir ruido
        data_5m['cv_ma_5m'] = self.media_acumulada(data_5m['cv_5m'].rolling(14))

        #volatility indicators
        data_5m['atr'] = self.get_atr(data_5m,data_5m["high"],data_5m["low"],data_5m["close"],self.atr_window)
        data_5m['atr_ma'] = self.media_acumulada(data_5m['atr'].ewm(span=self.atr_ma, adjust=False, min_periods=self.atr_ma))

        exp_factor = 0.5  # Ajusta este valor según la agresividad deseada
        min_atr = data_5m['atr'].cummin()  # Mínimo ATR histórico
        max_atr = data_5m['atr'].cummax()
        normalized_atr = (data_5m['atr'] - min_atr) / (max_atr - min_atr)

        data_5m['adjusted_atr'] = np.clip(normalized_atr,0,1) ** exp_factor
        #data_5m['adjusted_atr'] = normalized_atr ** exp_factor

        data_5m['vw-atr'] = np.cumsum(data_5m["atr"]*data_5m["tick_volume"])/np.cumsum(data_5m["tick_volume"])
        data_5m['vw-atr-ma'] = self.media_acumulada(data_5m['vw-atr'].ewm(span=self.atr_ma, adjust=False, min_periods=self.atr_ma))

        sma = data_5m['mean_price'].ewm(span = self.bollinger_sma, adjust = False, min_periods=self.bollinger_sma)
        std = sma.std()
        sma = self.media_acumulada(sma)

        data_5m['bollinger_sma'] = sma
        data_5m['bollinger_upper'] = sma+(std*2)
        data_5m['bollinger_lower'] = sma-(std*2)
        data_5m['mass_index'] = sum((data_5m['high']-data_5m['low']).ewm(span=self.mass_index[0], adjust = False, min_periods=self.mass_index[0]).mean() / ((data_5m['high']-data_5m['low']).ewm(span=self.mass_index[0], adjust = False, min_periods=self.mass_index[0]).mean()).ewm(span=self.mass_index[0], adjust = False, min_periods=self.mass_index[0]).mean(),self.mass_index[1])
        data_5m['mass_index_ema']=data_5m['mass_index'].ewm(span = self.mass_index[2], adjust = False, min_periods = self.mass_index[2]).mean()

        data_5m["rvi"] = rvi(data_5m['close'])

        thermo_df = thermo(data_5m['high'],data_5m['low'])

        data_5m["thermo"] = thermo_df[thermo_df.columns[0]]
        data_5m['thermo_ma'] = thermo_df[thermo_df.columns[1]]

        data_5m['all_candle_patterns'] = cdl_pattern(
            data_5m['open'], data_5m['high'], data_5m['low'], data_5m['close'], name="all"
        ).sum(axis=1)

        #volume indicators
        """
        data_5m["ad"] = ad()
        data_5m["aobv"] = aobv()
        data_5m["cmf"] = cmf()
        data_5m["efi"] = efi()
        data_5m["eom"] = eom()
        data_5m["kvo"] = kvo()
        data_5m["mfi"] = mfi()
        data_5m["nvi"] = nvi()
        data_5m["obv"] = obv()
        data_5m["pvi"] = pvi()
        data_5m["pvol"] = pvol()
        data_5m["pvr"] = pvr()
        data_5m["pvt"] = pvt()
        data_5m["vp"] = vp()
        """

        #market cycles  
        data_5m["ebsw"] = ebsw(data_5m['close'])

        return data_5m
    
    def df_mejorado(self, df):
        # Crear una copia para no modificar el original
        df = df.copy()
        
        # Números fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
        fibonacci_periods = [21, 34, 55, 89, 144, 233, 377]
        
        # ===== INDICADORES DE TENDENCIA =====
        #print("Calculando indicadores de tendencia...")
        
        # Medias móviles simples y variantes
        for period in fibonacci_periods:
            # SMA - Simple Moving Average
            df[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
            
            # EMA - Exponential Moving Average
            df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
            # DEMA - Double Exponential Moving Average
            df[f'DEMA_{period}'] = talib.DEMA(df['close'], timeperiod=period)
            
            # TEMA - Triple Exponential Moving Average
            df[f'TEMA_{period}'] = talib.TEMA(df['close'], timeperiod=period)
            
            # KAMA - Kaufman Adaptive Moving Average
            df[f'KAMA_{period}'] = talib.KAMA(df['close'], timeperiod=period)
            
            # WMA - Weighted Moving Average
            df[f'WMA_{period}'] = talib.WMA(df['close'], timeperiod=period)
            
            # TRIMA - Triangular Moving Average
            df[f'TRIMA_{period}'] = talib.TRIMA(df['close'], timeperiod=period)
            
            # T3 - Triple Exponential Moving Average with T3 smoothing
            df[f'T3_{period}'] = talib.T3(df['close'], timeperiod=period)
            
            # MIDPOINT - Midpoint over period
            df[f'MIDPOINT_{period}'] = talib.MIDPOINT(df['close'], timeperiod=period)
            
            # MIDPRICE - Midpoint Price over period
            df[f'MIDPRICE_{period}'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=period)
        
        # MAMA - MESA Adaptive Moving Average
        df['MAMA'], df['FAMA'] = talib.MAMA(df['close'])
        
        # MA - Moving Average (con diferentes tipos)
        df['MA_SMA_30'] = talib.MA(df['close'], timeperiod=30, matype=talib.MA_Type.SMA)
        df['MA_EMA_30'] = talib.MA(df['close'], timeperiod=30, matype=talib.MA_Type.EMA)
        df['MA_WMA_30'] = talib.MA(df['close'], timeperiod=30, matype=talib.MA_Type.WMA)
        
        # MAVP - Moving average with variable period
        # Necesita un array de períodos, usando constante para simplificar
        periods = np.full_like(df['close'], 30)
        df['MAVP'] = talib.MAVP(df['close'], periods, minperiod=2, maxperiod=30, matype=0)
        
        # SAR - Parabolic SAR
        df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        
        # SAREXT - Parabolic SAR Extended
        df['SAREXT'] = talib.SAREXT(df['high'], df['low'])
        
        # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close'])
        
        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['close'])
        
        # ===== INDICADORES DE MOMENTUM =====
        #print("Calculando indicadores de momentum...")
        
        # ADX - Average Directional Movement Index
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ADXR - Average Directional Movement Index Rating
        df['ADXR'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # APO - Absolute Price Oscillator
        df['APO'] = talib.APO(df['close'], fastperiod=12, slowperiod=26, matype=0)
        
        # AROON - Aroon
        df['AROON_down'], df['AROON_up'] = talib.AROON(df['high'], df['low'], timeperiod=14)
        
        # AROONOSC - Aroon Oscillator
        df['AROONOSC'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)
        
        # BOP - Balance Of Power
        df['BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
        
        # CCI - Commodity Channel Index
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CMO - Chande Momentum Oscillator
        df['CMO'] = talib.CMO(df['close'], timeperiod=14)
        
        # DX - Directional Movement Index
        df['DX'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MACD - Moving Average Convergence/Divergence
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # MACDEXT - MACD with controllable MA type
        df['MACDEXT'], df['MACDEXT_Signal'], df['MACDEXT_Hist'] = talib.MACDEXT(
            df['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        
        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        df['MACDFIX'], df['MACDFIX_Signal'], df['MACDFIX_Hist'] = talib.MACDFIX(df['close'], signalperiod=9)
        
        # MFI - Money Flow Index
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # MINUS_DI - Minus Directional Indicator
        df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MINUS_DM - Minus Directional Movement
        df['MINUS_DM'] = talib.MINUS_DM(df['high'], df['low'], timeperiod=14)
        
        # MOM - Momentum
        df['MOM'] = talib.MOM(df['close'], timeperiod=10)
        
        # PLUS_DI - Plus Directional Indicator
        df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # PLUS_DM - Plus Directional Movement
        df['PLUS_DM'] = talib.PLUS_DM(df['high'], df['low'], timeperiod=14)
        
        # PPO - Percentage Price Oscillator
        df['PPO'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
        
        # ROC - Rate of change : ((price/prevPrice)-1)*100
        df['ROC'] = talib.ROC(df['close'], timeperiod=10)
        
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        df['ROCP'] = talib.ROCP(df['close'], timeperiod=10)
        
        # ROCR - Rate of change ratio: (price/prevPrice)
        df['ROCR'] = talib.ROCR(df['close'], timeperiod=10)
        
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        df['ROCR100'] = talib.ROCR100(df['close'], timeperiod=10)
        
        # RSI - Relative Strength Index
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        
        # STOCH - Stochastic
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                                fastk_period=5, slowk_period=3, slowk_matype=0, 
                                                slowd_period=3, slowd_matype=0)
        
        # STOCHF - Stochastic Fast
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['high'], df['low'], df['close'], 
                                                    fastk_period=5, fastd_period=3, fastd_matype=0)
        
        # STOCHRSI - Stochastic Relative Strength Index
        df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(df['close'], timeperiod=14, 
                                                        fastk_period=5, fastd_period=3, fastd_matype=0)
        
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        df['TRIX'] = talib.TRIX(df['close'], timeperiod=30)
        
        # ULTOSC - Ultimate Oscillator
        df['ULTOSC'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        
        # WILLR - Williams' %R
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ===== INDICADORES DE VOLATILIDAD =====
        #print("Calculando indicadores de volatilidad...")
        
        # ATR - Average True Range
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # NATR - Normalized Average True Range
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # TRANGE - True Range
        df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])
        
        # Bandas de bollinger (implementación manual como en tu código)
        df['STD_21'] = df['close'].rolling(window=21).std()
        df['Upper_Band'] = df['SMA_21'] + (2 * df['STD_21'])
        df['Lower_Band'] = df['SMA_21'] - (2 * df['STD_21'])
        
        # ATR para diferentes períodos (mantengo tus períodos)
        for period in [14, 21]:
            df[f'ATR_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            df[f'NATR_{period}'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Medidas estadísticas de volatilidad
        df['STDDEV_7'] = talib.STDDEV(df['close'], timeperiod=7, nbdev=1)
        df['STDDEV_14'] = talib.STDDEV(df['close'], timeperiod=14, nbdev=1)
        df['VAR_14'] = talib.VAR(df['close'], timeperiod=14, nbdev=1)
        
        # Canales de Keltner (como en tu código)
        ema_20 = talib.EMA(df['close'], timeperiod=20)
        atr_10 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=10)
        df['KELTNER_UPPER'] = ema_20 + (2 * atr_10)
        df['KELTNER_LOWER'] = ema_20 - (2 * atr_10)
        df['KELTNER_MIDDLE'] = ema_20
        
        # ===== INDICADORES DE VOLUMEN =====
        #print("Calculando indicadores de volumen...")
        
        # Verifica si 'volume' o 'tick_volume' está disponible
        volume_column = 'volume' if 'volume' in df.columns else ('tick_volume' if 'tick_volume' in df.columns else None)
        
        if volume_column:
            # AD - Chaikin A/D Line
            df['AD'] = talib.AD(df['high'], df['low'], df['close'], df[volume_column])
            
            # ADOSC - Chaikin A/D Oscillator
            df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df[volume_column], fastperiod=3, slowperiod=10)
            
            # OBV - On Balance Volume
            df['OBV'] = talib.OBV(df['close'], df[volume_column])
            
            # MFI - Money Flow Index
            df['MFI_14'] = talib.MFI(df['high'], df['low'], df['close'], df[volume_column], timeperiod=14)
            
            # VPT - Volume Price Trend
            df['VPT'] = (df[volume_column] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        # ===== CICLOS E INDICADORES HILBERT TRANSFORM =====
        #print("Calculando indicadores de ciclos...")
        
        # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['close'])
        
        # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
        df['HT_DCPHASE'] = talib.HT_DCPHASE(df['close'])
        
        # HT_PHASOR - Hilbert Transform - Phasor Components
        df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(df['close'])
        
        # HT_SINE - Hilbert Transform - SineWave
        df['HT_SINE'], df['HT_LEADSINE'] = talib.HT_SINE(df['close'])
        
        # ===== TRANSFORMACIONES DE PRECIO =====
        #print("Calculando transformaciones de precio...")
        
        # AVGPRICE - Average Price
        df['AVGPRICE'] = talib.AVGPRICE(df['open'], df['high'], df['low'], df['close'])
        
        # MEDPRICE - Median Price
        df['MEDPRICE'] = talib.MEDPRICE(df['high'], df['low'])
        
        # TYPPRICE - Typical Price
        df['TYPPRICE'] = talib.TYPPRICE(df['high'], df['low'], df['close'])
        
        # WCLPRICE - Weighted Close Price
        df['WCLPRICE'] = talib.WCLPRICE(df['high'], df['low'], df['close'])
        
        # ===== FUNCIONES ESTADÍSTICAS =====
        #print("Calculando indicadores estadísticos...")
        
        # BETA - Beta
        df['BETA'] = talib.BETA(df['high'], df['low'], timeperiod=5)
        
        # CORREL - Correlación de Pearson
        df['CORREL'] = talib.CORREL(df['high'], df['low'], timeperiod=30)
        
        # Regresión Lineal
        df['LINEARREG'] = talib.LINEARREG(df['close'], timeperiod=14)
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['close'], timeperiod=14)
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['close'], timeperiod=14)
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)
        
        # TSF - Time Series Forecast
        df['TSF'] = talib.TSF(df['close'], timeperiod=14)
        
        # VAR - Varianza
        df['VAR'] = talib.VAR(df['close'], timeperiod=5, nbdev=1)
        
        # ===== RECONOCIMIENTO DE PATRONES DE VELAS =====
        #print("Calculando patrones de velas...")
        
        # Patrones de reversión
        df['CDL2CROWS'] = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
        df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        df['CDL3INSIDE'] = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
        df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])
        df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
        df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])
        df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close'], penetration=0.3)
        df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close'])
        df['CDLBELTHOLD'] = talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])
        df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close'])
        df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close'])
        df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close'])
        df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close'])
        df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'], penetration=0.5)
        df['CDLDOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['CDLDOJISTAR'] = talib.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])
        df['CDLENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0.3)
        df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'], penetration=0.3)
        df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close'])
        df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])
        df['CDLHAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
        df['CDLHARAMI'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])
        df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close'])
        df['CDLHIKKAKE'] = talib.CDLHIKKAKE(df['open'], df['high'], df['low'], df['close'])
        df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close'])
        df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close'])
        df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['close'])
        df['CDLINNECK'] = talib.CDLINNECK(df['open'], df['high'], df['low'], df['close'])
        df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['CDLKICKING'] = talib.CDLKICKING(df['open'], df['high'], df['low'], df['close'])
        df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close'])
        df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close'])
        df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])
        df['CDLLONGLINE'] = talib.CDLLONGLINE(df['open'], df['high'], df['low'], df['close'])
        df['CDLMARUBOZU'] = talib.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])
        df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['close'])
        df['CDLMATHOLD'] = talib.CDLMATHOLD(df['open'], df['high'], df['low'], df['close'], penetration=0.5)
        df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration=0.3)
        df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'], penetration=0.3)
        df['CDLONNECK'] = talib.CDLONNECK(df['open'], df['high'], df['low'], df['close'])
        df['CDLPIERCING'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
        df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['close'])
        df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close'])
        df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['close'])
        df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDLSHORTLINE'] = talib.CDLSHORTLINE(df['open'], df['high'], df['low'], df['close'])
        df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
        df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['close'])
        df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['close'])
        df['CDLTAKURI'] = talib.CDLTAKURI(df['open'], df['high'], df['low'], df['close'])
        df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['close'])
        df['CDLTHRUSTING'] = talib.CDLTHRUSTING(df['open'], df['high'], df['low'], df['close'])
        df['CDLTRISTAR'] = talib.CDLTRISTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['close'])
        df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['close'])
        df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['close'])
        
        #print("Cálculo de indicadores completado!")
        return df
    
    def get_rsi(self,data):

        rsi,k,d = StochRSI(data['diff'], period=self.rsi_roll, smoothK = self.stoch_rsi[0], smoothD = self.stoch_rsi[1])

        if len(rsi)<len(data):

            relleno = pd.Series([0 for i in range(len(data)-len(rsi))])
            rsi=relleno._append(rsi)
            k=relleno._append(k)
            d=relleno._append(d)

        return rsi,k,d
    
    def get_standard_deviation(self,data:list):

        self.stdev.append(self.std_acumulada(data))

        return self.stdev
    
    def get_atr(self,data,high,low,close, atr_window=14):

        data_copy = data.copy()

        data_copy['tr0'] = abs(high - low)
        data_copy['tr1'] = abs(high - close.shift())
        data_copy['tr2'] = abs(low - close.shift())

        tr = data_copy[['tr0', 'tr1', 'tr2']].max(axis=1)

        atr=tr.ewm(alpha=1/atr_window, adjust=False).mean()

        return atr
    
if __name__=="__main__":

    import MetaTrader5 as mt
    import matplotlib.pyplot as plt
    from datetime import datetime
    from tqdm import tqdm

    fecha_inicio = datetime(2024,5,9)
    fecha_final = datetime(2024,5,10) 

    mt.initialize()

    login = 80284478
    password = "*iWcMw3k"
    server = "MetaQuotes-Demo"

    mt.login(login, password, server)

    tick_data = pd.DataFrame(mt.copy_ticks_range("EURUSD", fecha_inicio, fecha_final, mt.COPY_TICKS_ALL))

    vwap_twap = []

    market_master_maths = Maths(ema_length = 400,sma_length = 400,length_macd = 9,max_length = 1200)

    for pos,data in tqdm(tick_data.iterrows(),total = len(tick_data)):

        market_master_maths.add(data["bid"],data["ask"])
        market_master_maths.infer(sma_ema=True,macd=False)
        #vwap_twap.append(market_master_maths.vwap_twap())

    sma,ema = market_master_maths.get_sma_ema()
    #macd,signal=market_master_maths.get_macd()

    plt.plot(tick_data["bid"])
    plt.plot(tick_data["ask"])
    plt.plot(sma)
    plt.plot(ema)
    #plt.plot(macd)
    #plt.plot(signal)

    #plt.plot(vwap_twap)

    plt.show()

    mt.shutdown()