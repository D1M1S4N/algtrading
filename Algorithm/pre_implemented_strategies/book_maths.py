import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.signal import find_peaks
from scipy.stats import mode
import pandas_ta as pta
import ta.others
import ta.volume
import talib
import ta

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
    def __init__(self):      
        super().__init__()

    def media_acumulada(self, columna):
        return columna.mean()

    def detectar_outliers(self, df, columna):
        Q1 = df[columna].expanding().quantile(0.9, interpolation = 'linear')
        Q3 = df[columna].expanding().quantile(0.1, interpolation = 'linear')

        df['Q3'] = Q3
        df['Q1'] = Q1

        IQR = Q3 - Q1
        
        df['IQR'] = IQR
        df['limite_superior'] = Q1 - 2 * IQR
        df['limite_inferior'] = Q1 - 1 * IQR
        
    def get_atr(self,data,high,low,close, atr_window=14):

        data_copy = data.copy()

        data_copy['tr0'] = abs(high - low)
        data_copy['tr1'] = abs(high - close.shift())
        data_copy['tr2'] = abs(low - close.shift())

        tr = data_copy[['tr0', 'tr1', 'tr2']].max(axis=1)

        atr=tr.ewm(alpha=1/atr_window, adjust=False).mean()

        return atr
    
    def df(self, df):
        # Librerias usadas:
        # talib
        # pandas-ta as pta
        # ta
        # Números fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
        # Preferible usar números fibonacci para los periodos de las medias móviles y otros indicadores
        
        # VALIDACIÓN CRÍTICA: Verificar que hay suficientes datos
        min_required_periods = 144  # El período más largo que usamos
        if len(df) < min_required_periods:
            print(f"⚠️ ADVERTENCIA: Solo hay {len(df)} velas disponibles. Se recomienda al menos {min_required_periods} para calcular todos los indicadores correctamente.")
        
        # Ajustar períodos fibonacci según datos disponibles
        max_available_period = max(8, len(df) // 3)  # Usar como máximo 1/3 de los datos disponibles
        fibonacci_periods = [8, 21, 34, 55, 89, 144]
        fibonacci_periods = [p for p in fibonacci_periods if p <= max_available_period]
        
        if len(fibonacci_periods) == 0:
            fibonacci_periods = [min(8, len(df) - 1)]  # Al menos un período pequeño
        
        # Calculo de los rangos de la distribucion para el spread
        try:
            df['moda_max'] = df['spread'].expanding().apply(calcular_moda_max, raw = False)
            df['moda_min'] = df['spread'].expanding().apply(calcular_moda_min, raw = False)
            df['moda_diff'] = df['moda_max'] - df['moda_min']
            df['moda_inferior'] = df['moda_max']
            df['moda_superior1'] = df['moda_max'] + df['moda_min']
            df['moda_superior2'] = df['moda_max'] + df['moda_diff']
        except Exception as e:
            print(f"Error calculando modas del spread: {e}")
            df['moda_max'] = 0
            df['moda_min'] = 0
            df['moda_diff'] = 0
            df['moda_inferior'] = 0
            df['moda_superior1'] = 0
            df['moda_superior2'] = 0
        
        df['atr'] = self.get_atr(df, df["high"], df["low"], df["close"], min(13, len(df) - 1))
        df['atr_ma'] = self.media_acumulada(df['atr'].ewm(span = min(5, len(df) - 1), adjust = False, min_periods = min(5, len(df) - 1)))

        exp_factor = 0.5  # Ajusta este valor según la agresividad deseada
        min_atr = df['atr'].cummin()  # Mínimo ATR histórico
        max_atr = df['atr'].cummax()
        normalized_atr = (df['atr'] - min_atr) / (max_atr - min_atr + 1e-10)  # Añadir epsilon para evitar división por cero

        df['adjusted_atr'] = np.clip(normalized_atr,0,1) ** exp_factor
        
        # Aqui empieza lo sigma
        
        # ===== INDICADORES DE TENDENCIA =====
        #print("Calculando indicadores de tendencia...")
        
        # Medias móviles simples y variantes
        for period in fibonacci_periods:
            # Validar que el período no exceda los datos disponibles
            if period >= len(df):
                continue
                
            # SMA - Simple Moving Average
            try:
                df[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando SMA_{period}: {e}")
                df[f'SMA_{period}'] = 0
            
            # EMA - Exponential Moving Average
            try:
                df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando EMA_{period}: {e}")
                df[f'EMA_{period}'] = 0

            # DEMA - Double Exponential Moving Average
            """try:
                df[f'DEMA_{period}'] = talib.DEMA(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando DEMA_{period}: {e}")
                df[f'DEMA_{period}'] = 0"""

            # TEMA - Triple Exponential Moving Average
            """try:
                df[f'TEMA_{period}'] = talib.TEMA(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando TEMA_{period}: {e}")
                df[f'TEMA_{period}'] = 0"""

            # KAMA - Kaufman Adaptive Moving Average
            """try:
                df[f'KAMA_{period}'] = talib.KAMA(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando KAMA_{period}: {e}")
                df[f'KAMA_{period}'] = 0"""

            """# WMA - Weighted Moving Average
            try:
                df[f'WMA_{period}'] = talib.WMA(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando WMA_{period}: {e}")
                df[f'WMA_{period}'] = 0

            # TRIMA - Triangular Moving Average
            try:
                df[f'TRIMA_{period}'] = talib.TRIMA(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando TRIMA_{period}: {e}")
                df[f'TRIMA_{period}'] = 0

            # T3 - Triple Exponential Moving Average with T3 smoothing
            try:
                df[f'T3_{period}'] = talib.T3(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando T3_{period}: {e}")
                df[f'T3_{period}'] = 0

            # MIDPOINT - Midpoint over period
            try:
                df[f'MIDPOINT_{period}'] = talib.MIDPOINT(df['close'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando MIDPOINT_{period}: {e}")
                df[f'MIDPOINT_{period}'] = 0

            # MIDPRICE - Midpoint Price over period
            try:
                df[f'MIDPRICE_{period}'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=period)
            except Exception as e:
                print(f"Error calculando MIDPRICE_{period}: {e}")
                df[f'MIDPRICE_{period}'] = 0"""

        # MAMA - MESA Adaptive Moving Average
        #df['MAMA'], df['FAMA'] = talib.MAMA(df['close'])
        
        # MA - Moving Average (con diferentes tipos)
        #df['MA_SMA_30'] = talib.MA(df['close'], timeperiod=34, matype=talib.MA_Type.SMA)
        #df['MA_EMA_30'] = talib.MA(df['close'], timeperiod=34, matype=talib.MA_Type.EMA)
        #df['MA_WMA_30'] = talib.MA(df['close'], timeperiod=34, matype=talib.MA_Type.WMA)
        
        # MAVP - Moving average with variable period
        # Necesita un array de períodos, usando constante para simplificar
        #periods = np.full_like(df['close'], 30)
        #df['MAVP'] = talib.MAVP(df['close'], periods, minperiod=2, maxperiod=34, matype=0)
        
        # SAR - Parabolic SAR
        #df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        
        # SAREXT - Parabolic SAR Extended
        #df['SAREXT'] = talib.SAREXT(df['high'], df['low'], accelerationInitLong=0.01, accelerationLong=0.02, accelerationMaxLong=0.2, accelerationInitShort=0.03, accelerationShort=0.04, accelerationMaxShort=0.25)
        
        # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
        #df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close'])
        
        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        #df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['close'])
        
        # ===== INDICADORES DE MOMENTUM =====
        #print("Calculando indicadores de momentum...")
        
        # ADX - Average Directional Movement Index
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=13)
        
        # ADXR - Average Directional Movement Index Rating
        #df['ADXR'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=13)
        
        # APO - Absolute Price Oscillator
        #df['APO'] = talib.APO(df['close'], fastperiod=13, slowperiod=34, matype=0)
        
        # AROON - Aroon
        #df['AROON_down'], df['AROON_up'] = talib.AROON(df['high'], df['low'], timeperiod=21)
        
        # AROONOSC - Aroon Oscillator
        #df['AROONOSC'] = talib.AROONOSC(df['high'], df['low'], timeperiod=13)
        
        # BOP - Balance Of Power
        #df['BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
        
        # CCI - Commodity Channel Index
        #df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=13)
        
        # CMO - Chande Momentum Oscillator
        #df['CMO'] = talib.CMO(df['close'], timeperiod=13)
        
        # DX - Directional Movement Index
        #df['DX'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=13)
        
        # MACD - Moving Average Convergence/Divergence
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'], fastperiod=13, slowperiod=34, signalperiod=8)
        
        # MACDEXT - MACD with controllable MA type
        #df['MACDEXT'], df['MACDEXT_Signal'], df['MACDEXT_Hist'] = talib.MACDEXT(df['close'], fastperiod=13, fastmatype=0, slowperiod=34, slowmatype=0, signalperiod=8, signalmatype=0)
        
        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        #df['MACDFIX'], df['MACDFIX_Signal'], df['MACDFIX_Hist'] = talib.MACDFIX(df['close'], signalperiod=8)
        
        # MFI - Money Flow Index
        #df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['tick_volume'], timeperiod=13) # Probado en backtest y no va bien
        
        # MINUS_DI - Minus Directional Indicator
        #df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=13)
        
        # MINUS_DM - Minus Directional Movement
        #df['MINUS_DM'] = talib.MINUS_DM(df['high'], df['low'], timeperiod=13)
        
        # MOM - Momentum
        #df['MOM'] = talib.MOM(df['close'], timeperiod=8)
        
        # PLUS_DI - Plus Directional Indicator
        #df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=13)
        
        # PLUS_DM - Plus Directional Movement
        #df['PLUS_DM'] = talib.PLUS_DM(df['high'], df['low'], timeperiod=13)
        
        # PPO - Percentage Price Oscillator
        #df['PPO'] = talib.PPO(df['close'], fastperiod=13, slowperiod=21, matype=0)
        
        # ROC - Rate of change : ((price/prevPrice)-1)*100
        #df['ROC'] = talib.ROC(df['close'], timeperiod=8)
        
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        #df['ROCP'] = talib.ROCP(df['close'], timeperiod=8)
        
        # ROCR - Rate of change ratio: (price/prevPrice)
        #df['ROCR'] = talib.ROCR(df['close'], timeperiod=8)
        
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        #df['ROCR100'] = talib.ROCR100(df['close'], timeperiod=8)
        
        # RSI - Relative Strength Index
        df['RSI'] = talib.RSI(df['close'], timeperiod=13)
        
        # STOCH - Stochastic
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                                fastk_period=5, slowk_period=3, slowk_matype=0, 
                                                slowd_period=3, slowd_matype=0)
        
        # STOCHF - Stochastic Fast
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['high'], df['low'], df['close'], 
                                                    fastk_period=5, fastd_period=3, fastd_matype=0)
        
        # STOCHRSI - Stochastic Relative Strength Index
        df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(df['close'], timeperiod=13, 
                                                        fastk_period=5, fastd_period=3, fastd_matype=0)
        
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        #df['TRIX'] = talib.TRIX(df['close'], timeperiod=34)
        
        # ULTOSC - Ultimate Oscillator
        #df['ULTOSC'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=8, timeperiod2=13, timeperiod3=34)
        
        # WILLR - Williams' %R
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=21)
        
        # ===== INDICADORES DE VOLATILIDAD =====
        #print("Calculando indicadores de volatilidad...")
        
        # ATR - Average True Range
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=13)
        
        # NATR - Normalized Average True Range
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=13)
        
        # TRANGE - True Range
        df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])
        
        # Bandas de bollinger (implementación manual como en tu código)
        df['STD_21'] = df['close'].rolling(window=21).std()
        df['Upper_Band'] = df['SMA_21'] + (2 * df['STD_21'])
        df['Lower_Band'] = df['SMA_21'] - (2 * df['STD_21'])
        
        # ATR para diferentes períodos (mantengo tus períodos)
        for period in [13, 21]:
            df[f'ATR_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            df[f'NATR_{period}'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Medidas estadísticas de volatilidad
        df['STDDEV_8'] = talib.STDDEV(df['close'], timeperiod=8, nbdev=1)
        df['STDDEV_13'] = talib.STDDEV(df['close'], timeperiod=13, nbdev=1)
        df['STDDEV_21'] = talib.STDDEV(df['close'], timeperiod=21, nbdev=1)
        df['VAR_8'] = talib.VAR(df['close'], timeperiod=8, nbdev=1)
        df['VAR_13'] = talib.VAR(df['close'], timeperiod=13, nbdev=1)
        df['VAR_21'] = talib.VAR(df['close'], timeperiod=21, nbdev=1)
        
        # Canales de Keltner (como en tu código)
        ema_20 = talib.EMA(df['close'], timeperiod=21)
        atr_10 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=8)
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
            df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df[volume_column], fastperiod=3, slowperiod=8)
            
            # OBV - On Balance Volume
            df['OBV'] = talib.OBV(df['close'], df[volume_column])
            
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
        
        # BETA - Beta sirve para medir la volatilidad de un activo en relación a un índice de referencia
        """df['BETA_5'] = talib.BETA(df['high'], df['low'], timeperiod=5)
        df['BETA_13'] = talib.BETA(df['high'], df['low'], timeperiod=13)
        df['BETA_21'] = talib.BETA(df['high'], df['low'], timeperiod=34)"""
        
        # CORREL - Correlación de Pearson sirve para medir la relación entre dos series de precios
        #df['CORREL'] = talib.CORREL(df['high'], df['low'], timeperiod=34)
        
        # Regresión Lineal sirve para medir la tendencia de los precios
        """df['LINEARREG'] = talib.LINEARREG(df['close'], timeperiod=13)
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['close'], timeperiod=13)
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['close'], timeperiod=13)
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=13)"""
        
        # TSF - Time Series Forecast sirve para predecir el precio futuro
        #df['TSF'] = talib.TSF(df['close'], timeperiod=13)
        
        # ===== RECONOCIMIENTO DE PATRONES DE VELAS =====
        #print("Calculando patrones de velas...")
        
        # Patrones de reversión
        """df['CDL2CROWS'] = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
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
        df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['close'])"""
        
        # Z-Score para diferentes periodos
        for period in [13, 21, 34, 55, 89]:
            # Z-Score basado en precios de cierre
            try:
                rolling_mean = df['close'].rolling(window=period).mean()
                rolling_std = df['close'].rolling(window=period).std()
                df[f'Z_SCORE_{period}'] = (df['close'] - rolling_mean) / rolling_std
                
                # Z-Score de volumen (si está disponible)
                volume_column = 'volume' if 'volume' in df.columns else ('tick_volume' if 'tick_volume' in df.columns else None)
                if volume_column:
                    vol_rolling_mean = df[volume_column].rolling(window=period).mean()
                    vol_rolling_std = df[volume_column].rolling(window=period).std()
                    df[f'VOL_Z_SCORE_{period}'] = (df[volume_column] - vol_rolling_mean) / vol_rolling_std
            except:
                df[f'Z_SCORE_{period}'] = None
                if volume_column:
                    df[f'VOL_Z_SCORE_{period}'] = None
        
        # Z-Score adaptativo (usando ATR para normalizar)
        atr_14 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        typical_mean = typical_price.rolling(window=21).mean()
        df['ADAPTIVE_Z_SCORE'] = (typical_price - typical_mean) / (atr_14 * 0.5)
        
        # ===== Z-SCORE PARA INDICADORES CLAVE =====
        # Z-Score para RSI
        rsi_mean = df['RSI'].rolling(window=21).mean()
        rsi_std = df['RSI'].rolling(window=21).std()
        df['RSI_Z_SCORE'] = (df['RSI'] - rsi_mean) / rsi_std
        
        # Z-Score para MACD
        macd_mean = df['MACD'].rolling(window=21).mean()
        macd_std = df['MACD'].rolling(window=21).std()
        df['MACD_Z_SCORE'] = (df['MACD'] - macd_mean) / macd_std
        
        # CALCULOS CON PANDAS-TA
        # Cumulative Return
        df['CR'] = ta.others.cumulative_return(df['close'], fillna=True)
        # Daily Log Return
        df['DLR'] = ta.others.daily_log_return(df['close'], fillna=True)
        # Daily Return
        df['DR'] = ta.others.daily_return(df['close'], fillna=True)
        
        # INDICADORES DE TENDENCIA CON TA
        # Arnaud Legoux Moving Average
        df['ALMA'] = pta.alma(df['close'], length=9, offset=0.85, sigma=6)

        # Volume Weighted Average Price (por sesión)
        df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['tick_volume'], window=14, fillna=True)
        
        # Calcula el VWAP por sesión (día) usando pandas-ta
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vp'] = df['typical_price'] * df['tick_volume']

        # Asegúrate de que df['time'] sea tipo datetime
        df['date'] = df['time'].dt.date

        # VWAP por sesión (día)
        df['cum_vp'] = df.groupby('date')['vp'].cumsum()
        df['cum_vol'] = df.groupby('date')['tick_volume'].cumsum()
        df['VWAP_PER_SESSION'] = df['cum_vp'] / df['cum_vol']

        # Even Better Sinewave sirve para detectar tendencias
        df['EBSW'] = pta.ebsw(df['close'], length=14)

        # Zero Lag EMA sirve para eliminar el retraso de la EMA
        df['ZLEMA'] = pta.zlma(df['close'], length=50)
        
        # INDICADORES DE MOMENTUM CON TA
        # Awesome Oscillator sirve para detectar cambios de tendencia
        df['AO'] = pta.ao(df['high'], df['low'], fast=5, slow=34)

        # Center of Gravity
        df['COG'] = pta.cg(df['close'], length=8)

        # Chande Forecast Oscillator
        df['CFO'] = pta.cfo(df['close'], length=13)

        # Correlation Trend Indicator
        df['CTI'] = pta.cti(df['close'], length=13)

        # Fisher Transform
        fisher_df = pta.fisher(df['high'], df['low'], length=9, signal=5)
        df['FISHER'] = fisher_df['FISHERT_9_5']  # Fisher Transform principal
        df['FISHER_SIGNAL'] = fisher_df['FISHERTs_9_5']  # Línea de señal

        # Klinger Volume Oscillator
        """try:
            kvo_df = pta.kvo(df['high'], df['low'], df['close'], df['tick_volume'], fast=34, slow=55, signal=13)
            df['KVO'] = kvo_df['KVO_34_55_13']  # Oscilador principal
            df['KVO_SIGNAL'] = kvo_df['KVOs_34_55_13']  # Línea de señal
        except Exception as e:
            df['KVO'] = None
            df['KVO_SIGNAL'] = None"""

        # Pretty Good Oscillator
        #df['PGO'] = pta.pgo(df['high'], df['low'], df['close'], length=14)

        # Schaff Trend Cycle
        try:
            stc_df = pta.stc(df['close'], length=10, factor=0.5)
            df['STC'] = stc_df['STC_10_12_26_0.5']  # El nombre puede variar según los parámetros
            df['STC_MACD'] = stc_df['STCmacd_10_12_26_0.5']  # El nombre puede variar según los parámetros
            df['STC_STOCH'] = stc_df['STCstoch_10_12_26_0.5']  # El nombre puede variar según los parámetros
        except Exception as e:
            df['STC'] = None
            df['STC_MACD'] = None
            df['STC_STOCH'] = None

        # Squeeze Momentum
        squeeze_df = pta.squeeze(df['high'], df['low'], df['close'], length=20, mult=2)
        df['SQUEEZE_ON'] = squeeze_df['SQZ_ON']
        df['SQUEEZE_OFF'] = squeeze_df['SQZ_OFF']
        df['SQUEEZE_NO_SQUEEZE'] = squeeze_df['SQZ_NO']
        df['SQUEEZE_MOMENTUM'] = squeeze_df['SQZ_20_2.0_20_1.5']

        # Vortex Indicator 
        vortex_df = pta.vortex(df['high'], df['low'], df['close'], length=14)
        df['VI_PLUS'] = vortex_df['VTXP_14']  # Columna de Vortex Indicator Plus
        df['VI_MINUS'] = vortex_df['VTXM_14']  # Columna de Vortex Indicator Minus

        # Know Sure Thing
        try:
            kst_df = pta.kst(df['close'], roc1=10, roc2=15, roc3=20, roc4=30, signal=9)
            df['KST'] = kst_df['KST_10_15_20_30_10_10_10_15']  # El nombre puede ser diferente
            df['KST_SIGNAL'] = kst_df['KSTs_9']    # El nombre puede ser diferente
        except Exception as e:
            df['KST'] = None
            df['KST_SIGNAL'] = None

        # Choppiness Index - Identifica mercados laterales
        df['CHOP'] = pta.chop(df['high'], df['low'], df['close'], length=14)

        # Directional Movement Index
        dmi_df = pta.dm(df['high'], df['low'], length=14)
        df['DMI_PLUS'] = dmi_df['DMP_14']  # Columna suavizada de DM+
        df['DMI_MINUS'] = dmi_df['DMN_14']  # Columna suavizada de DM-
        
        # INDICADORES DE VOLUMEN CON TA
        # Accumulation/Distribution Index
        df['ADI'] = pta.ad(df['high'], df['low'], df['close'], df['tick_volume'])

        # Ease of Movement
        df['EOM'] = pta.eom(df['high'], df['low'], df['close'], df['tick_volume'], length=14)

        # Price Volume Trend
        df['PVT'] = pta.pvt(df['close'], df['tick_volume'])

        # Negative Volume Index
        df['NVI'] = talib.ADOSC(df['high'], df['low'], df['close'], df['tick_volume'], fastperiod=3, slowperiod=8)

        # Positive Volume Index
        df['PVI'] = pta.pvi(df['close'], df['tick_volume'])

        # Elder Force Index
        df['EFI'] = pta.efi(df['close'], df['tick_volume'], length=13)
        
        # CALCULOS ESTADISTICOS
        #df['SKEW'] = pta.skew(df['close'], length=13) # sirve para medir la asimetría de la distribución de precios
        #df['KURT'] = pta.kurtosis(df['close'], length=13) # sirve para medir la "aplanamiento" de la distribución de precios
        # Aqui falta meter alguno mas
        
        # Rellenar NaN con 0 para IAs que no aceptan NaN
        df = df.fillna(0)
        
        # Esto es opcional, pero puede ayudar a reducir el tamaño del dataframe y mejorar el rendimiento
        #df = df.convert_dtypes(dtype_backend="pyarrow")
        
        # 02/05/2025 2:27:00 hay 280 columnas en el dataframe
        
        #print("Cálculo de indicadores completado!")
        return df
    
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