import numpy as np
import pandas as pd
from collections import deque
import scipy.stats as st
from scipy.signal import find_peaks
from scipy.stats import mode
from pandas_ta import * # https://tradingstrategy.ai/docs/api/technical-analysis/index.html
# https://github.com/twopirllc/pandas-ta

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

      # Lists to track results
      self.psar_list = []
      self.af_list = []
      self.ep_list = []
      self.high_list = []
      self.low_list = []
      self.trend_list = []
      self._num_days = 0

    def calcPSAR(self, high, low):
      if self._num_days >= 3:
        psar = self._calcPSAR()
      else:
        psar = self._initPSARVals(high, low)

      psar = self._updateCurrentVals(psar, high, low)
      self._num_days += 1

      return psar

    def _initPSARVals(self, high, low):
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
      if self.trend == 1: # Up
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
      # Checks for reversals
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

def log_return(price,previous_price)->float:
    return np.log(price/previous_price)

def average_return(n,price,previous_price)->float:
    return (1/n)*log_return(price,previous_price)

def variance(data:list):
    return np.var(data)

def momentum(data,n):
    # momentum = i/(i-n)*100
    return (data[-1]/data[-1-n])*100

def volatility(data:list):
    np.sqrt(variance(data))

def covariance(x:list,y:list):
    return np.cov(x,y)

def correlation(x,y):
    return np.corrcoef(x,y)

def vwap(price,volume):
    return np.cumsum(price*volume)/np.cumsum(volume)

def sma_function(data,n):
    return np.cumsum(data)/n

def ema_function(precio,anterior_ema,n):
    multiplier=2/(n+1)
    
    return (precio*multiplier)+(anterior_ema*(1-multiplier))

def macd_function(ema12,ema26,anterior_macd,n=9):
    macd=ema12-ema26
    signal=ema_function(macd,anterior_macd,n)
    return macd,signal

def rsi(diff,fn_roll):
    change_up = diff.copy()
    change_down = diff.copy()

    change_up[change_up<0] = 0
    change_down[change_down>0] = 0

    diff.equals(change_up+change_down)
    avg_up = change_up.rolling(fn_roll).mean()
    avg_down = change_down.rolling(fn_roll).mean().abs()
    rsi = 100 * avg_up / (avg_up + avg_down)
    
    return rsi

def StochRSI(delta, period=14, smoothK=3, smoothD=3):
    # Calculate RSI 
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

def calculate_adaptive_ema(prices,N,smoothing=2):
    ema = [np.cumsum(prices[N:]) / N]

    sf=smoothing/(N+1)

    for price in prices[1:]:
        volatility=abs(price-ema[-1])
        er=volatility/price
        sf=sf*(er*(sf-1)+1)
        ema.append((price * sf) + ema[-1] * (1 - sf))

    return ema

def mfi(data,window):
  money_flow=data['rmv']
  signal=np.where(money_flow>money_flow.shift(1),1,np.where(money_flow<money_flow.shift(1),-1,0))
  money_flow_s=money_flow*signal

  money_flow_positive=money_flow_s.rolling(window).apply(lambda x:sum(np.where(x>=0.0,x,0.0)),raw=True)
  money_flow_negative=abs(money_flow_s.rolling(window).apply(lambda x:sum(np.where(x<0.0,x,0.0)),raw=True))

  money_flow_ratio=money_flow_positive/money_flow_negative

  return 100-(100/(1+money_flow_ratio))

def get_wr(data_5m, lookback):
    high=data_5m['high']
    close=data_5m['close']
    low=data_5m['low']
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr

def get_vroc(data_5m,lookback):
   resultado=[]
   for _,i in enumerate(data_5m['tick_volume']):
      if _>lookback:
         resultado.append(100*((i-data_5m['tick_volume'][_-lookback])/data_5m['tick_volume'][_-lookback]))
      else:
         resultado.append(100*((i-data_5m['tick_volume'][0])/data_5m['tick_volume'][0]))

   return resultado

def get_nvi(data_5m):
  resultado=[]

  for _,i in enumerate(data_5m['mean_price']):
    if len(resultado)==0:
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

    def media_acumulada(self,columna):
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
      return columna.std()

    def df(self,data_5m):
        #rsi_value,k,d=self.get_rsi(data_5m)   
        #self.detectar_outliers(data_5m,"spread")
        stoch_rsi_df=stochrsi(data_5m['mean_price'],length=self.rsi_roll,rsi_length=self.rsi_roll,k=self.stoch_rsi[0],d=self.stoch_rsi[1])

        data_5m['moda_max'] = data_5m['spread'].expanding().apply(calcular_moda_max, raw=False)
        data_5m['moda_min'] = data_5m['spread'].expanding().apply(calcular_moda_min, raw=False)
        data_5m['moda_diff'] = data_5m['moda_max']-data_5m['moda_min']
        data_5m['moda_inferior'] = data_5m['moda_max']
        data_5m['moda_superior1']=data_5m['moda_max']+data_5m['moda_min']
        data_5m['moda_superior2']=data_5m['moda_max']+data_5m['moda_diff']

        data_5m['stochrsi_K']=stoch_rsi_df[stoch_rsi_df.columns[0]]
        data_5m['stochrsi_D']=stoch_rsi_df[stoch_rsi_df.columns[1]]
        data_5m['rsi']=stoch_rsi_df[stoch_rsi_df.columns[0]]

        data_5m['ema'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length, adjust=False, min_periods=self.ema_length))
        data_5m['ema2'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length2, adjust=False, min_periods=self.ema_length2))
        data_5m['ema3'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length3, adjust=False, min_periods=self.ema_length3))
        data_5m['ema4'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.ema_length4, adjust=False, min_periods=self.ema_length4))

        data_5m['impulso']=data_5m['ema'].diff()

        data_5m['macd_ma'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_2, adjust=False, min_periods=self.sma_length_2))
        data_5m['macd_ma2'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_3, adjust=False, min_periods=self.sma_length_3))
        data_5m['macd'] = data_5m['macd_ma']-data_5m['macd_ma2']
        data_5m['macd_signal']=self.media_acumulada(data_5m['macd'].ewm(span=self.length_macd, adjust=False, min_periods=self.length_macd))
        data_5m['2macd_ma'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_4, adjust=False, min_periods=self.sma_length_4))
        data_5m['2macd_ma2'] = self.media_acumulada(data_5m['mean_price'].ewm(span=self.sma_length_5, adjust=False, min_periods=self.sma_length_5))
        data_5m['2macd'] = data_5m['2macd_ma']-data_5m['2macd_ma2']
        data_5m['2macd_signal']=self.media_acumulada(data_5m['2macd'].ewm(span=self.length_macd2, adjust=False, min_periods=self.length_macd2))

        data_5m['vwap']=np.cumsum(data_5m["mean_price"]*data_5m["tick_volume"])/np.cumsum(data_5m["tick_volume"])
        data_5m['twap']=np.cumsum(data_5m["mean_price"])/(data_5m.index+1)
        
        data_5m['mean_volume']=np.cumsum(data_5m["tick_volume"])/(data_5m.index+1)
                
        data_5m['PSAR'] = data_5m.apply(lambda x: self.indic.calcPSAR(x['high'], x['low']), axis=1)
        #data_5m['EP'] = self.indic.ep_list
        #data_5m['Trend'] = self.indic.trend_list
        #data_5m['AF'] = self.indic.af_list

        plus_di, minus_di, adx_smooth=get_adx(data_5m["high"],data_5m["low"],data_5m["close"],self.adx_window)
        data_5m['adx']=adx_smooth
        data_5m['plus_di']=plus_di #+dmi
        data_5m['minus_di']=minus_di #-dmi

        data_5m['obv']=(np.sign(data_5m['close'].diff()) * data_5m['tick_volume']).fillna(0).cumsum()
        data_5m['obv_ema']=self.media_acumulada(data_5m['obv'].ewm(span=self.obv_ema, adjust=False, min_periods=self.obv_ema))

        data_5m['mfi']=mfi(data_5m,self.mfi_length)
        data_5m=price_volume_trend(data_5m)
        data_5m['pvt_ema']=self.media_acumulada(data_5m['pvt'].ewm(span=self.pvt_length, adjust=False, min_periods=self.pvt_length))

        data_5m['momentum'] = np.where(data_5m['close'].diff() > 0, 1, -1)
        data_5m['momentum_ema']=self.media_acumulada(data_5m['momentum'].ewm(span=self.momentum, adjust=False, min_periods=self.momentum))

        data_5m['mfm']=((data_5m['close']-data_5m['low'])-(data_5m['high']-data_5m['close']))/(data_5m['high']-data_5m['low'])
        data_5m['mfv']=data_5m['mfm']*data_5m['tick_volume']
        
        adl=[]
        for i in range(len(data_5m)):
          if i==0:
            adl.append(data_5m['mfv'][i])
          else:
            adl.append(data_5m['mfv'][i]+adl[-1])

        data_5m['adl']=adl
        data_5m['adl_ema']=self.media_acumulada(data_5m['adl'].ewm(span=self.adl_ema, adjust=False, min_periods=self.adl_ema))

        data_5m['williams_r']=get_wr(data_5m,self.wr_length[0])
        data_5m['williams_r_ema']=self.media_acumulada(data_5m['williams_r'].ewm(span=self.wr_length[1], adjust=False, min_periods=self.wr_length[1]))
        data_5m['vroc']=get_vroc(data_5m,self.vroc)
        data_5m['nvi']=get_nvi(data_5m)
        data_5m['nvi_ema']=self.media_acumulada(data_5m['nvi'].ewm(span=self.nvi, adjust=False, min_periods=self.nvi)) # 255?
        data_5m['bulls_bears_powers_ema']=self.media_acumulada(data_5m['mean_price'].ewm(span=self.bull_bear_power, adjust=False, min_periods=self.bull_bear_power))
        data_5m['bulls_powers']=data_5m['high']-data_5m['bulls_bears_powers_ema']
        data_5m['bears_powers']=data_5m['low']-data_5m['bulls_bears_powers_ema']
        data_5m['cci']=get_cci(data_5m,self.cci[0])
        data_5m['cci_ema']=self.media_acumulada(data_5m['cci'].ewm(span=self.cci[1], adjust=False, min_periods=self.cci[1]))

        data_5m['trix']=trix(data_5m,self.trix)
        data_5m['vortex']=vortex(data_5m,self.vortex)

        data_5m['adr']=adr(data_5m)

        smi_df=smi(data_5m['close']) #smi, signal, oscillator columns
        data_5m['smi']=smi_df[smi_df.columns[0]]
        data_5m['smi_signal']=smi_df[smi_df.columns[1]]

        #data_5m['stc']=stc(data_5m['close'])
        data_5m['cmo']=cmo(data_5m['close'])

        keltner_channels=kc(data_5m['high'],data_5m['low'],data_5m['close'])
        data_5m['kc_up']=keltner_channels[keltner_channels.columns[0]]
        data_5m['kc_basis']=keltner_channels[keltner_channels.columns[1]]
        data_5m['kc_down']=keltner_channels[keltner_channels.columns[2]] 

        #data_5m['stdv-z-score']=(data_5m['diff'] - data_5m['diff'].mean())/data_5m['stdv']
        #data_5m['stdv-p-value']=st.norm.ppf(data_5m['stdv-z-score'])
        #data_5m['atr-z-score']=(data_5m['atr'].pct_change() - data_5m['atr'].pct_change().mean())/data_5m['atr'].pct_change().std(ddof=1)
        #data_5m['atr-p-value']=st.norm.ppf(data_5m['atr-z-score'])

        #statistics
        data_5m["entropy"]=entropy(data_5m['close'])
        data_5m["kurtosis"]=kurtosis(data_5m['close'])
        data_5m["mad"]=mad(data_5m['close'])
        data_5m["median"]=median(data_5m['close'])
        data_5m["skew"]=skew(data_5m['close'])
        data_5m['stdv']=self.std_acumulada(abs(data_5m['diff']))
        data_5m['stdv_price']=self.std_acumulada(data_5m['close'])

        data_5m["variance"]=variance(data_5m['close'])
        data_5m["zscore"]=zscore(data_5m['close'])

        # Calcular el coeficiente de variación (CV)
        data_5m['cv_5m'] = (data_5m['stdv'] / data_5m['ema']).where(data_5m['ema'] != 0, 0)

        # Suavizar el CV con una media móvil para reducir ruido
        data_5m['cv_ma_5m'] = self.media_acumulada(data_5m['cv_5m'].rolling(14))

        #ma
        """
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        """

        #trend indicators
        """
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        """

        #momentum indicators
        """
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        data_5m[""]=()
        """

        #volatility indicators
        data_5m['atr']=self.get_atr(data_5m,data_5m["high"],data_5m["low"],data_5m["close"],self.atr_window)
        data_5m['atr_ma']=self.media_acumulada(data_5m['atr'].ewm(span=self.atr_ma, adjust=False, min_periods=self.atr_ma))

        exp_factor = 0.5  # Ajusta este valor según la agresividad deseada
        min_atr = data_5m['atr'].cummin()  # Mínimo ATR histórico
        max_atr = data_5m['atr'].cummax()
        normalized_atr = (data_5m['atr'] - min_atr) / (max_atr - min_atr)

        data_5m['adjusted_atr'] = np.clip(normalized_atr,0,1) ** exp_factor
        #data_5m['adjusted_atr'] = normalized_atr ** exp_factor

        data_5m['vw-atr']=np.cumsum(data_5m["atr"]*data_5m["tick_volume"])/np.cumsum(data_5m["tick_volume"])
        data_5m['vw-atr-ma']=self.media_acumulada(data_5m['vw-atr'].ewm(span=self.atr_ma, adjust=False, min_periods=self.atr_ma))
        sma=data_5m['mean_price'].ewm(span=self.bollinger_sma, adjust=False, min_periods=self.bollinger_sma)
        std=sma.std()
        sma=self.media_acumulada(sma)
        data_5m['bollinger_sma']=sma
        data_5m['bollinger_upper']=sma+(std*2)
        data_5m['bollinger_lower']=sma-(std*2)
        data_5m['mass_index']=sum((data_5m['high']-data_5m['low']).ewm(span=self.mass_index[0], adjust=False, min_periods=self.mass_index[0]).mean() / ((data_5m['high']-data_5m['low']).ewm(span=self.mass_index[0], adjust=False, min_periods=self.mass_index[0]).mean()).ewm(span=self.mass_index[0], adjust=False, min_periods=self.mass_index[0]).mean(),self.mass_index[1])
        data_5m['mass_index_ema']=data_5m['mass_index'].ewm(span=self.mass_index[2], adjust=False, min_periods=self.mass_index[2]).mean()
        data_5m["rvi"]=rvi(data_5m['close'])
        thermo_df=thermo(data_5m['high'],data_5m['low'])
        data_5m["thermo"]=thermo_df[thermo_df.columns[0]]
        data_5m['thermo_ma']=thermo_df[thermo_df.columns[1]]
        data_5m['all_candle_patterns'] = cdl_pattern(
            data_5m['open'], data_5m['high'], data_5m['low'], data_5m['close'], name="all"
        ).sum(axis=1)

        #volume indicators
        """
        data_5m["ad"]=ad()
        data_5m["aobv"]=aobv()
        data_5m["cmf"]=cmf()
        data_5m["efi"]=efi()
        data_5m["eom"]=eom()
        data_5m["kvo"]=kvo()
        data_5m["mfi"]=mfi()
        data_5m["nvi"]=nvi()
        data_5m["obv"]=obv()
        data_5m["pvi"]=pvi()
        data_5m["pvol"]=pvol()
        data_5m["pvr"]=pvr()
        data_5m["pvt"]=pvt()
        data_5m["vp"]=vp()
        """

        #market cycles  
        data_5m["ebsw"]=ebsw(data_5m['close'])

        return data_5m
    
    def get_rsi(self,data):
        rsi,k,d = StochRSI(data['diff'], period=self.rsi_roll, smoothK=self.stoch_rsi[0], smoothD=self.stoch_rsi[1])
        if len(rsi)<len(data):
            relleno=pd.Series([0 for i in range(len(data)-len(rsi))])
            rsi=relleno._append(rsi)
            k=relleno._append(k)
            d=relleno._append(d)

        return rsi,k,d
    
    def get_standard_deviation(self,data:list):
        self.stdev.append(self.std_acumulada(data))
        return self.stdev
    
    def get_psar(self,data):
        #https://raposa.trade/blog/the-complete-guide-to-calculating-the-parabolic-sar-in-python/
        return

    def get_atr(self,data,high,low,close, atr_window=14):
        data_copy=data.copy()
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

    fecha_inicio=datetime(2024,5,9)
    fecha_final=datetime(2024,5,10) 

    mt.initialize()

    login=80284478
    password="*iWcMw3k"
    server="MetaQuotes-Demo"

    mt.login(login, password, server)

    tick_data=pd.DataFrame(mt.copy_ticks_range("EURUSD", fecha_inicio, fecha_final, mt.COPY_TICKS_ALL))

    vwap_twap=[]

    market_master_maths=Maths(ema_length=400,sma_length=400,length_macd=9,max_length=1200)

    for pos,data in tqdm(tick_data.iterrows(),total=len(tick_data)):
        market_master_maths.add(data["bid"],data["ask"])
        market_master_maths.infer(sma_ema=True,macd=False)
        #vwap_twap.append(market_master_maths.vwap_twap())

    sma,ema=market_master_maths.get_sma_ema()
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