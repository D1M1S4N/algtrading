from estrategia import stats
from estrategia import ai
from estrategia.tecnical import tendencia_actual
from win11toast import toast
import MetaTrader5 as mt
import pandas as pd
from datetime import datetime,timedelta
from pre_implemented_strategies.book_maths import Maths
import random
import traceback
import numpy as np
import pywt
import scipy.stats as st
import statsmodels.api as sm
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from copulas.multivariate import GaussianMultivariate
from smartmoneyconcepts import smc
import math
import mplfinance as mpf
import talib

def calcular_pips(p_i, p_f):
    #con_lotaje = round(abs(p_i - p_f) * p_i / (0.0001*lotaje), 2)
    sin_lotaje = round(abs(p_i - p_f) * p_i / (0.0001), 2)
    return sin_lotaje

class MarketMaster():
    def __init__(self, atr_multipliers, maximo_perdida_diaria, riesgo_por_operacion, dinero_inicial, lista_pesos_confirmaciones, 
                 corr_limit, seed, multiplicador_tp_sl, lista_pesos_estrategias):
        
        self.min_adx_threshold = 20
        self.strong_adx_threshold = 25
        self.very_strong_adx_threshold = 40
        
        self.maximo_perdida_diaria = maximo_perdida_diaria
        self.riesgo_por_operacion = riesgo_por_operacion

        self.multiplicador_tp_sl = multiplicador_tp_sl

        if seed is not None:
            random.seed(seed)

        self.atr_multipliers = atr_multipliers
        
        self.historical_signals_indicators = []
        self.historical_signals_trend = []

        self.MarketMasterStats=stats.MarketMasterStats(corr_limit=corr_limit)

        self.estrategias = [self.estrategia_liquidity_sweep_momentum, self.estrategia_liquidity_smc_advanced, self.estrategia_vwap, self.estrategia_liquidity_momentum_continuation, self.estrategia_trend_continuation, self.estrategia_trend_continuation2]
        self.lista_pesos_estrategias = lista_pesos_estrategias

        self.confirmations = [self.MarketMasterStats.correlation_confirmation, self.liquidity_confirmation]
        self.lista_pesos_confirmaciones = lista_pesos_confirmaciones
        self.lista_pesos_confirmaciones_ordenada = sorted(lista_pesos_confirmaciones)

        self.dinero_inicial = dinero_inicial
        self.total_estrategias = len(self.estrategias)
        super().__init__()

    def valor_pip(self, n_pips,precio):
        #return ((n_pips * 0.0001) / precio) * lotaje
        return ((n_pips * 0.0001) / precio)
    
    def blend(self, estrategias, porcentajes):
        accion = sum(x * porcentajes[pos] for pos, x in enumerate(estrategias))
        return accion

    #--------------------------------------------Estrategias----------------------------------------------------
    
    def debugging(self, args, indicador_valor):
        action = 0
        
        fecha = args["actual_time"]
        
        data_5m_all = args["all_data_5m"][indicador_valor]
        data_5m_all = data_5m_all[data_5m_all["time"] <= fecha]
        
        data = args['all_data']['EURUSD']
        data = data[data["time"] <= fecha].iloc[-1]
        
        return action
    
    def estrategia_liquidity_sweep_momentum(self, args, indicador_valor):
        """
        Estrategia: Liquidity Sweep + Momentum Confirmation
        Solo se ejecuta cuando el precio está en un punto de liquidez.
        Busca barrido (sweep) y confirma con RSI.
        """
        action = 0

        data_5m_all = args["all_data_5m"][indicador_valor]
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        if len(data_5m_all) < 30:
            print("Datos insuficientes para estrategia_liquidity_sweep_momentum")
            return 0

        current = data_5m_all.iloc[-1]
        prev = data_5m_all.iloc[-2]

        # Obtener el punto de liquidez tocado (el más cercano al precio actual)
        precio_actual = current['close']
        puntos_liquidez = list(args['puntos_liquidez'].keys())
        if not puntos_liquidez:
            return 0
        punto_cercano = min(puntos_liquidez, key=lambda x: abs(x - precio_actual))

        # Sweep bajista: el precio hace un mínimo inferior al anterior y cierra por encima del punto de liquidez
        sweep_down = (
            current['low'] < prev['low'] and
            current['low'] <= punto_cercano <= current['close']
        )

        # Sweep alcista: el precio hace un máximo superior al anterior y cierra por debajo del punto de liquidez
        sweep_up = (
            current['high'] > prev['high'] and
            current['high'] >= punto_cercano >= current['close']
        )

        rsi = data_5m_all['RSI']

        rsi_value = rsi.iloc[-1]
        rsi_threshold = 55

        # Señal de compra: sweep bajista + RSI fuerte alcista
        if sweep_down and rsi_value > rsi_threshold:
            action = 1
        # Señal de venta: sweep alcista + RSI fuerte bajista
        elif sweep_up and rsi_value < (100 - rsi_threshold):
            action = -1

        return action
    
    def estrategia_liquidity_momentum_continuation(self, args, indicador_valor):
        """
        Estrategia de continuación de momentum tras manipulación de liquidez.
        Busca el movimiento fuerte después del sweep y el pullback.
        """
        action = 0

        data_5m_all = args["all_data_5m"][indicador_valor]
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        if len(data_5m_all) < 40:
            print("Datos insuficientes para estrategia_liquidity_momentum_continuation")    
            return 0

        current = data_5m_all.iloc[-1]
        prev = data_5m_all.iloc[-2]
        prev2 = data_5m_all.iloc[-3]
        prev3 = data_5m_all.iloc[-4]

        # Punto de liquidez tocado
        precio_actual = current['close']
        puntos_liquidez = list(args['puntos_liquidez'].keys())
        if not puntos_liquidez:
            return 0
        punto_cercano = min(puntos_liquidez, key=lambda x: abs(x - precio_actual))

        # 1. Sweep detection (en las 2 velas anteriores)
        sweep_down = (
            (prev['low'] < prev2['low'] and prev['low'] <= punto_cercano <= prev['close']) or
            (prev2['low'] < prev3['low'] and prev2['low'] <= punto_cercano <= prev2['close'])
        )
        sweep_up = (
            (prev['high'] > prev2['high'] and prev['high'] >= punto_cercano >= prev['close']) or
            (prev2['high'] > prev3['high'] and prev2['high'] >= punto_cercano >= prev2['close'])
        )

        # 2. Pullback: en las últimas 3 velas tras el sweep
        # Para sweep_down, buscamos que el precio baje al menos un 20% de la vela del sweep y luego supere el máximo del sweep
        # Para sweep_up, buscamos que el precio suba al menos un 20% de la vela del sweep y luego caiga por debajo del mínimo del sweep

        # Usamos la vela del sweep más reciente
        if sweep_down:
            sweep_candle = prev if (prev['low'] < prev2['low']) else prev2
            # Buscar pullback en current o prev
            for test_candle in [current, prev]:
                retroceso = sweep_candle['close'] - test_candle['low']
                rango = sweep_candle['close'] - sweep_candle['low']
                if rango > 0 and retroceso > 0.2 * rango:
                    # Confirmación de continuación: ruptura de máximo del sweep y momentum
                    rsi = data_5m_all['RSI'] if 'RSI' in data_5m_all.columns else talib.RSI(data_5m_all['close'], timeperiod=14)
                    rsi_value = rsi.iloc[-1]
                    volumen_creciente = test_candle['tick_volume'] > data_5m_all['tick_volume'].iloc[-6:-1].mean()
                    if test_candle['close'] > sweep_candle['high'] and rsi_value > 57 and volumen_creciente:
                        action = 1
                        break

        elif sweep_up:
            sweep_candle = prev if (prev['high'] > prev2['high']) else prev2
            for test_candle in [current, prev]:
                retroceso = test_candle['high'] - sweep_candle['close']
                rango = sweep_candle['high'] - sweep_candle['close']
                if rango > 0 and retroceso > 0.2 * rango:
                    rsi = data_5m_all['RSI'] if 'RSI' in data_5m_all.columns else talib.RSI(data_5m_all['close'], timeperiod=14)
                    rsi_value = rsi.iloc[-1]
                    volumen_creciente = test_candle['tick_volume'] > data_5m_all['tick_volume'].iloc[-6:-1].mean()
                    if test_candle['close'] < sweep_candle['low'] and rsi_value < 43 and volumen_creciente:
                        action = -1
                        break

        return action
    
    def estrategia_liquidity_smc_advanced(self, args, indicador_valor):
        """
        Estrategia avanzada: Liquidity Sweep + Orderflow + Multi-Timeframe Confirmation
        """
        action = 0

        # --- Datos 5m (para sweep y orderflow) ---
        data_5m_all = args["all_data_5m"][indicador_valor]
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        if len(data_5m_all) < 30:
            print("Datos insuficientes para estrategia_liquidity_smc_advanced")
            return 0

        current = data_5m_all.iloc[-1]
        prev = data_5m_all.iloc[-2]

        # --- Datos 1H (para tendencia superior) ---
        data_1h = args["all_data_1h"][indicador_valor]
        data_1h = data_1h[data_1h["time"] <= args["actual_time"]]
        if len(data_1h) < 20:
            return 0

        # --- Punto de liquidez tocado ---
        precio_actual = current['close']
        puntos_liquidez = list(args['puntos_liquidez'].keys())
        if not puntos_liquidez:
            return 0
        punto_cercano = min(puntos_liquidez, key=lambda x: abs(x - precio_actual))

        # --- 1. Sweep detection ---
        sweep_down = (
            current['low'] < prev['low'] and
            current['low'] <= punto_cercano <= current['close']
        )
        sweep_up = (
            current['high'] > prev['high'] and
            current['high'] >= punto_cercano >= current['close']
        )

        # --- 2. Confirmación de BOS/CHoCH ---
        # Usamos swings: si el sweep rompe el último swing y cierra en dirección contraria, es BOS
        # (Puedes mejorar esto usando tu función de swings si la tienes)
        swing_window = 10
        last_swing_high = data_5m_all['high'].rolling(window=swing_window).max().iloc[-2]
        last_swing_low = data_5m_all['low'].rolling(window=swing_window).min().iloc[-2]
        bos_confirm = False
        if sweep_down and current['close'] > last_swing_low:
            bos_confirm = True
        if sweep_up and current['close'] < last_swing_high:
            bos_confirm = True

        # --- 3. Confirmación de tendencia superior (1H) ---
        sma_fast = data_1h['SMA_8'].iloc[-1]
        sma_slow = data_1h['SMA_21'].iloc[-1]
        tendencia_1h = 1 if sma_fast > sma_slow else -1

        # --- 4. Confirmación de patrón de vela (pinbar, engulfing) ---
        cuerpo = abs(current['close'] - current['open'])
        rango = current['high'] - current['low']
        upper_wick = current['high'] - max(current['close'], current['open'])
        lower_wick = min(current['close'], current['open']) - current['low']
        pinbar_bull = lower_wick > cuerpo * 1.5 and cuerpo < rango * 0.4
        pinbar_bear = upper_wick > cuerpo * 1.5 and cuerpo < rango * 0.4
        engulfing_bull = current['close'] > current['open'] and prev['close'] < prev['open'] and current['close'] > prev['open'] and current['open'] < prev['close']
        engulfing_bear = current['close'] < current['open'] and prev['close'] > prev['open'] and current['close'] < prev['open'] and current['open'] > prev['close']
        vela_confirm = (sweep_down and (pinbar_bull or engulfing_bull)) or (sweep_up and (pinbar_bear or engulfing_bear))

        # --- 5. Confirmación de spike de volumen ---
        avg_vol = data_5m_all['tick_volume'].iloc[-21:-1].mean()
        spike_vol = current['tick_volume'] > 1.5 * avg_vol

        # --- 6. Confirmación de momentum (RSI) ---
        rsi = data_5m_all['RSI'] if 'RSI' in data_5m_all.columns else talib.RSI(data_5m_all['close'], timeperiod=14)
        rsi_value = rsi.iloc[-1]
        rsi_confirm = (sweep_down and rsi_value > 55) or (sweep_up and rsi_value < 45)

        # --- 7. Sumar confirmaciones ---
        confirmaciones = [
            bos_confirm,
            (tendencia_1h == 1 and sweep_down) or (tendencia_1h == -1 and sweep_up),
            vela_confirm,
            spike_vol,
            rsi_confirm
        ]
        n_confirm = sum(confirmaciones)

        # --- 8. Señal solo si hay al menos 3 confirmaciones ---
        if sweep_down and n_confirm >= 3:
            action = 1
        elif sweep_up and n_confirm >= 3:
            action = -1

        return action
    
    # Falta por backtestearla mas meses, pero en enero de 2025 rinde super bien
    def estrategia_vwap(self, args, indicador_valor):
        action = 0
        
        data_5m_all = args["all_data_5m"]['EURUSD']
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        
        # Check if we have enough data
        if len(data_5m_all) < 20:  # Need at least 20 candles for analysis
            print("Datos insuficientes para estrategia_vwap")   
            return action
        
        # Get recent candles for analysis
        recent_data = data_5m_all.tail(20)
        last_candle = recent_data.iloc[-1]  # Current candle
        prev_candle = recent_data.iloc[-2]  # Previous candle
        
        # Verify VWAP column exists
        if 'VWAP_PER_SESSION' not in recent_data.columns:
            return action  # No VWAP data available
        
        # Current price position relative to VWAP
        current_price = last_candle['close']
        current_vwap = last_candle['VWAP_PER_SESSION']
        
        # Count recent VWAP crosses to detect consolidation
        vwap_crosses = 0
        for i in range(1, min(10, len(recent_data))):
            curr = recent_data.iloc[-i]
            prev = recent_data.iloc[-i-1]
            if ((prev['close'] > prev['VWAP_PER_SESSION'] and curr['close'] < curr['VWAP_PER_SESSION']) or
                (prev['close'] < prev['VWAP_PER_SESSION'] and curr['close'] > curr['VWAP_PER_SESSION'])):
                vwap_crosses += 1
        
        # If too many crosses, market is ranging - avoid trading
        if vwap_crosses >= 3:
            return 0
        
        # CONFIRMACIONES NUEVAS
        
        # LONG ENTRY LOGIC
        if current_price > current_vwap:
            # Check for a break above VWAP followed by pullback
            break_above = False
            for i in range(3, min(10, len(recent_data))):
                if (recent_data.iloc[-i-1]['close'] < recent_data.iloc[-i-1]['VWAP_PER_SESSION'] and 
                    recent_data.iloc[-i]['close'] > recent_data.iloc[-i]['VWAP_PER_SESSION']):
                    break_above = True
                    break
            
            # Look for pullback to VWAP without breaking below
            pullback_to_vwap = False
            if break_above:
                for i in range(1, 3):  # Check last 3 candles for pullback
                    candle = recent_data.iloc[-i]
                    if (candle['low'] <= candle['VWAP_PER_SESSION'] * 1.001 and 
                        candle['close'] > candle['VWAP_PER_SESSION']):
                        pullback_to_vwap = True
                        break
            
            # Confirmation patterns
            bullish_rejection = (
                last_candle['close'] > last_candle['open'] and  # Bullish candle
                last_candle['low'] < last_candle['VWAP_PER_SESSION'] and  # Wick touches/goes below VWAP
                last_candle['close'] > last_candle['VWAP_PER_SESSION']  # Close above VWAP
            )
            
            bullish_engulfing = (
                last_candle['close'] > last_candle['open'] and  # Current is bullish
                prev_candle['close'] < prev_candle['open'] and  # Previous was bearish
                last_candle['close'] > prev_candle['open'] and  # Current close higher than previous open
                last_candle['open'] < prev_candle['close']  # Current open lower than previous close
            )
            
            # Long entry signal
            if break_above and pullback_to_vwap and (bullish_rejection or bullish_engulfing):
                action = 1
        
        # SHORT ENTRY LOGIC
        elif current_price < current_vwap:
            # Check for a break below VWAP followed by pullback
            break_below = False
            for i in range(3, min(10, len(recent_data))):
                if (recent_data.iloc[-i-1]['close'] > recent_data.iloc[-i-1]['VWAP_PER_SESSION'] and 
                    recent_data.iloc[-i]['close'] < recent_data.iloc[-i]['VWAP_PER_SESSION']):
                    break_below = True
                    break
            
            # Look for pullback to VWAP without breaking above
            pullback_to_vwap = False
            if break_below:
                for i in range(1, 3):  # Check last 3 candles for pullback
                    candle = recent_data.iloc[-i]
                    if (candle['high'] >= candle['VWAP_PER_SESSION'] * 0.999 and 
                        candle['close'] < candle['VWAP_PER_SESSION']):
                        pullback_to_vwap = True
                        break
            
            # Confirmation patterns
            bearish_rejection = (
                last_candle['close'] < last_candle['open'] and  # Bearish candle
                last_candle['high'] > last_candle['VWAP_PER_SESSION'] and  # Wick touches/goes above VWAP
                last_candle['close'] < last_candle['VWAP_PER_SESSION']  # Close below VWAP
            )
            
            bearish_engulfing = (
                last_candle['close'] < last_candle['open'] and  # Current is bearish
                prev_candle['close'] > prev_candle['open'] and  # Previous was bullish
                last_candle['close'] < prev_candle['open'] and  # Current close lower than previous open
                last_candle['open'] > prev_candle['close']  # Current open higher than previous close
            )
            
            # Short entry signal
            if break_below and pullback_to_vwap and (bearish_rejection or bearish_engulfing):
                action = -1
        
        return action
    
    def estrategia_liquidez(self, args, indicador_valor):
        action = 0  # Valor predeterminado - sin acción
        
        # Obtener datos de mercado y liquidez
        liquidez_final = args["puntos_liquidez"]
        ask = args['ask']
        bid = args['bid']
        
        # Arrays para almacenar diferencias con puntos de liquidez
        liquidez_arriba = []  # Puntos por encima del precio
        liquidez_abajo = []   # Puntos por debajo del precio
        
        # Calcular distancias a puntos de liquidez
        for nivel_precio in liquidez_final.keys():
            if nivel_precio > ask:  # Punto por encima del precio
                liquidez_arriba.append(nivel_precio - ask)
            elif nivel_precio < bid:  # Punto por debajo del precio
                liquidez_abajo.append(bid - nivel_precio)
        
        # Comprobar la distribución de liquidez
        if len(liquidez_arriba) > 0 and len(liquidez_abajo) > 0:
            # Hay liquidez por arriba y por abajo - comparar promedios de distancia
            distancia_media_arriba = sum(liquidez_arriba) / len(liquidez_arriba)
            distancia_media_abajo = sum(liquidez_abajo) / len(liquidez_abajo)
            
            # Liquidez por arriba está más cerca que la de abajo = potencial resistencia
            if distancia_media_arriba < distancia_media_abajo:
                action = -1  # Señal de venta - la liquidez cercana arriba puede actuar como resistencia
                
            # Liquidez por abajo está más cerca que la de arriba = potencial soporte
            elif distancia_media_arriba > distancia_media_abajo:
                action = 1   # Señal de compra - la liquidez cercana abajo puede actuar como soporte
                
        # Casos especiales de distribución de liquidez
        elif len(liquidez_arriba) > 0 and len(liquidez_abajo) == 0:
            # Solo hay liquidez por arriba - posible resistencia fuerte
            action = -1  # Señal de venta
            
        elif len(liquidez_arriba) == 0 and len(liquidez_abajo) > 0:
            # Solo hay liquidez por abajo - posible soporte fuerte
            action = 1   # Señal de compra
        
        # Filtros adicionales (opcional)
        if action != 0:
            # Verificar concentración de liquidez
            if action == 1 and len(liquidez_abajo) < 3:
                # Pocos puntos de soporte - señal débil
                action = 0
            elif action == -1 and len(liquidez_arriba) < 3:
                # Pocos puntos de resistencia - señal débil
                action = 0
        
        return action

    # --------------------------------------------Confirmaciones----------------------------------------------------
    
    def fibonacci_retracement(self, price_sequence, current_price, levels = [0.5, 0.886]):
        high = max(price_sequence['high'])
        low = min(price_sequence['low'])

        # Calcular niveles de Fibonacci
        fib_50 = low + levels[0] * (high - low)
        fib_886 = low + levels[1] * (high - low)

        # Para largos (tendencia alcista)
        if current_price >= fib_50 and current_price <= fib_886:
            signal = 1
        # Para cortos (tendencia bajista)
        elif current_price <= fib_50 and current_price >= fib_886:
            signal = -1
        else:
            signal = 0

        return signal
    
    def session_confirmation(self, args, action, action2, name2, lotaje, peso):
        data_5m_all = args["all_data_5m"]['EURUSD']
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        
        ohlc_5m = data_5m_all[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
        ohlc_5m['time'] = pd.to_datetime(ohlc_5m['time'])
        ohlc_5m.set_index('time', inplace=True)
        
        
        """
        smc.sessions(ohlc, session, start_time, end_time, time_zone = "UTC")
        This method returns which candles are within the session specified

        parameters:
        session: str - the session you want to check (Sydney, Tokyo, London, New York, Asian kill zone, London open kill zone, New York kill zone, london close kill zone, Custom)
        start_time: str - the start time of the session in the format "HH:MM" only required for custom session.
        end_time: str - the end time of the session in the format "HH:MM" only required for custom session.
        time_zone: str - the time zone of the candles can be in the format "UTC+0" or "GMT+0"

        returns:
        Active = 1 if the candle is within the session, 0 if not
        High = the highest point of the session
        Low = the lowest point of the session
        """
        session_london = smc.sessions(ohlc_5m, "London", time_zone="GMT+0")
        session_ny = smc.sessions(ohlc_5m, "New York", time_zone="GMT+0")
        session_london_killzone = smc.sessions(ohlc_5m, "London open kill zone", time_zone="GMT+0")
        session_ny_killzone = smc.sessions(ohlc_5m, "New York kill zone", time_zone="GMT+0")
        session_asian = smc.sessions(ohlc_5m, "Tokyo", time_zone="GMT+0")
        session_sydney = smc.sessions(ohlc_5m, "Sydney", time_zone="GMT+0")
        session_asian_killzone = smc.sessions(ohlc_5m, "Asian kill zone", time_zone="GMT+0")
        session_london_close_killzone = smc.sessions(ohlc_5m, "london close kill zone", time_zone="GMT+0")
        
        is_london = session_london['Active'].iloc[-1] == 1
        is_ny = session_ny['Active'].iloc[-1] == 1
        is_london_killzone = session_london_killzone['Active'].iloc[-1] == 1
        is_ny_killzone = session_ny_killzone['Active'].iloc[-1] == 1
        is_asian = session_asian['Active'].iloc[-1] == 1 or session_sydney['Active'].iloc[-1] == 1
        is_asian_killzone = session_asian_killzone['Active'].iloc[-1] == 1
        is_london_close = session_london_close_killzone['Active'].iloc[-1] == 1
        
        # 1. AJUSTE DEL LOTAJE SEGÚN LA SESIÓN
        session_multiplier = 1.0  # Factor base
        
        # Sesiones de alta volatilidad - aumentamos el tamaño
        if is_london_killzone:
            session_multiplier = 1.2  # 20% más en London Open Killzone
        elif is_ny_killzone:
            session_multiplier = 1.3  # 30% más en NY Killzone
        elif is_london and is_ny:  # Solapamiento Londres-NY (máxima liquidez)
            session_multiplier = 1.5  # 50% más durante solapamiento
        elif is_london_close:
            session_multiplier = 1.1  # 10% más en London Close
        
        # Sesiones de baja volatilidad - reducimos el tamaño
        elif is_asian and not is_asian_killzone:
            session_multiplier = 0.5  # 30% menos durante sesión asiática normal
            
        lotaje *= session_multiplier
        """print(f"Se ha ajustado el lotaje según la sesión.")
        print(f"Multiplicador de sesión: {session_multiplier}")
        print(f"Sesión actual: London={is_london}, NY={is_ny}, London KZ={is_london_killzone}, NY KZ={is_ny_killzone}")"""
        
        return lotaje
    
    def retracement_confirmation(self, args, action, action2, name2, lotaje, peso):
        data_5m_all = args["all_data_5m"]['EURUSD']
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        last_candle = data_5m_all.iloc[-1]
        
        # Se hace un df formato OHLC para el timeframe de 1h (se hace lo mismo para los otros timeframes)
        ohlc_5m = data_5m_all[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
        ohlc_5m['time'] = pd.to_datetime(ohlc_5m['time'])
        ohlc_5m.set_index('time', inplace=True)
        
        # Se calcula el df de swing highs/lows y el de retrocesos
        shl_5m = smc.swing_highs_lows(ohlc_5m, swing_length=21) # Devuelve un df con los swing highs/lows, columna HighLow (1=high, -1=low) y columna Level (precio del swing)
        ret = smc.retracements(ohlc_5m, shl_5m) # Devuelve un df con los retrocesos, columna Direction (1=up, -1=down), columna CurrentRetracement% (porcentaje de retroceso actual) y columna DeepestRetracement% (porcentaje de retroceso más profundo)
        
        """
        En retrocesos alcistas:
        38.2%: Corrección moderada. Buen nivel para continuación si la tendencia es fuerte.

        50% (no es un número oficial de Fibonacci, pero muy usado): Nivel psicológico.

        61.8%: Corrección profunda, nivel clave para rebotes en tendencias fuertes.
        """
        
        """
        En retrocesos bajistas:
        38.2%: Rebote leve dentro de la caída.

        50%: Punto medio, común antes de continuar bajando.

        61.8%: Rebote profundo, ideal para buscar cortos si hay rechazo.
        """
        
        direction = ret['Direction'].iloc[-1]
        retracement = ret['CurrentRetracement%'].iloc[-1] / 100
        deepest_retracement = ret['DeepestRetracement%'].iloc[-1] / 100
        
        bull_trend = 1 if last_candle['low'] > last_candle['EMA_21'] else 0
        bear_trend = -1 if last_candle['high'] < last_candle['EMA_21'] else 0
        
        # Ahora vamos a comprobar la dirección de la acción y la dirección del retroceso
        if action == 1: 
            if direction == 1:
                # Si la dirección del retroceso es alcista, bajamos el lotaje
                #print("La dirección del retroceso es alcista. Ajustando lotaje.")
                lotaje *= 0.7
            if direction == -1 and bull_trend == 1:
                # Si la dirección del retroceso es bajista, entonces comprobamos tambien el porcentaje de retroceso y si esta en buena zona se sube el lotaje
                if retracement >= 0.5 and retracement <= 0.618:
                    #print("El retroceso está en buena zona. Ajustando lotaje.")
                    lotaje *= 1.3  # más confianza
                elif retracement >= 0.382 and retracement < 0.5:
                    #print("El retroceso está en una zona aceptable. Ajustando lotaje.")
                    lotaje *= 1.1  # menos confianza

        elif action == -1:
            if direction == -1:
                # Si la dirección del retroceso es bajista, bajamos el lotaje
                #print("La dirección del retroceso es bajista. Ajustando lotaje.")
                lotaje *= 0.7
            if direction == 1 and bear_trend == -1:
                # Si la dirección del retroceso es alcista, entonces comprobamos tambien el porcentaje de retroceso y si esta en buena zona se sube el lotaje
                if retracement >= 0.5 and retracement <= 0.618:
                    #print("El retroceso está en buena zona. Ajustando lotaje.")
                    lotaje *= 1.3  # más confianza
                elif retracement >= 0.382 and retracement < 0.5:
                    #print("El retroceso está en una zona aceptable. Ajustando lotaje.")
                    lotaje *= 1.1  # menos confianza

        return lotaje
    
    def volume_confirmation(self, args, action, action2, name2, lotaje, peso):
        data_5m_all = args["all_data_5m"]['EURUSD']
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        
        volume_threshold = 1.5  # Define a threshold for significant volume
        
        current = data_5m_all.iloc[-1]
        previous_candles = data_5m_all.iloc[-20:-1]
        
        avg_volume = previous_candles['tick_volume'].mean()
        current_volume = current['tick_volume']
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        significant_volume = volume_ratio >= volume_threshold
        
        if action == 1:
            is_bullish_candle = current['close'] > current['open']
            volume_confirms = is_bullish_candle and significant_volume
            
            last_3_volumes = data_5m_all['tick_volume'].iloc[-4:-1]
            increasing_volume_trend = last_3_volumes.iloc[-1] > last_3_volumes.iloc[-2] and last_3_volumes.iloc[-2] > last_3_volumes.iloc[-3]
            if increasing_volume_trend:
                volume_confirms = True
        elif action == -1:
            is_bearish_candle = current['close'] < current['open']
            volume_confirms = is_bearish_candle and significant_volume
            
            last_3_volumes = data_5m_all['tick_volume'].iloc[-4:-1]
            increasing_volume_trend = last_3_volumes.iloc[-1] > last_3_volumes.iloc[-2] and last_3_volumes.iloc[-2] > last_3_volumes.iloc[-3]
            if increasing_volume_trend:
                volume_confirms = True
        else:
            return lotaje
        
        if not volume_confirms:
            #print("No hay confirmación de volumen. Ajustando lotaje.")
            lotaje *= 0.7  # Reduce lotaje si no hay confirmación de volumen
            
        return lotaje
    
    # Rehacer esta pero dando una señal, para que directamente cancele la orden si no hay confirmacion
    """def candle_confirmation(self, args, action, action2, name2, lotaje, peso):
        # Obtener los datos de diferentes temporalidades
        data_1m_all = args["all_data_1m"]['EURUSD']
        data_1m_all = data_1m_all[data_1m_all["time"] <= args["actual_time"]]
        vela_1m = data_1m_all.iloc[-1]
        
        data_5m_all = args["all_data_5m"]['EURUSD']
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        vela_5m = data_5m_all.iloc[-1]
        
        data_15m_all = args["all_data_15m"]['EURUSD']
        data_15m_all = data_15m_all[data_15m_all["time"] <= args["actual_time"]]
        vela_15m = data_15m_all.iloc[-1]
        
        # Definir patrones alcistas y bajistas
        patrones_alcistas = [
            'MORNINGSTAR', 'MORNINGDOJISTAR', 'HAMMER', 'INVERTEDHAMMER', 'PIERCING', 
            '3WHITESOLDIERS', 'DRAGONFLYDOJI', 'LADDERBOTTOM', 'HOMINGPIGEON',
            'TASUKIGAP', 'BULLISHENGULFING', 'BULLISHHARAMI'
        ]
        
        patrones_bajistas = [
            '3BLACKCROWS', 'EVENINGSTAR', 'EVENINGDOJISTAR', 'HANGINGMAN', 'SHOOTINGSTAR',
            'DARKCLOUDCOVER', 'GRAVESTONEDOJI', 'IDENTITICAL3CROWS', 'ADVANCEBLOCK',
            'UPSIDEGAP2CROWS', 'BEARISHENGULFING', 'BEARISHHARAMI'
        ]
        
        # Diccionario para almacenar valores de temporalidades
        confirmaciones = {
            '1m': {'alcista': False, 'bajista': False, 'vela': vela_1m},
            '5m': {'alcista': False, 'bajista': False, 'vela': vela_5m},
            '15m': {'alcista': False, 'bajista': False, 'vela': vela_15m}
        }
        
        # Comprobar patrones en cada temporalidad
        for temporalidad, datos in confirmaciones.items():
            vela = datos['vela']
            
            # Comprobar patrones alcistas
            datos['alcista'] = any(
                vela.get(f'CDL{patron}', 0) > 0 
                for patron in patrones_alcistas 
                if f'CDL{patron}' in vela
            )
            
            # También comprobar patrones genéricos como CDLENGULFING que pueden ser alcistas
            if 'CDLENGULFING' in vela and vela['CDLENGULFING'] > 0:
                datos['alcista'] = True
            if 'CDLHARAMI' in vela and vela['CDLHARAMI'] > 0:
                datos['alcista'] = True
            if 'CDLMARUBOZU' in vela and vela['CDLMARUBOZU'] > 0 and vela['close'] > vela['open']:
                datos['alcista'] = True
            
            # Comprobar patrones bajistas
            datos['bajista'] = any(
                vela.get(f'CDL{patron}', 0) < 0
                for patron in patrones_bajistas
                if f'CDL{patron}' in vela
            )
            
            # También comprobar patrones genéricos como CDLENGULFING que pueden ser bajistas
            if 'CDLENGULFING' in vela and vela['CDLENGULFING'] < 0:
                datos['bajista'] = True
            if 'CDLHARAMI' in vela and vela['CDLHARAMI'] < 0:
                datos['bajista'] = True
            if 'CDLMARUBOZU' in vela and vela['CDLMARUBOZU'] < 0 and vela['close'] < vela['open']:
                datos['bajista'] = True
        
        # Determinar si hay confirmación según la acción
        confirmacion_encontrada = False
        
        if action == 1:  # Compra (alcista)
            confirmacion_encontrada = any(datos['alcista'] for datos in confirmaciones.values())
        elif action == -1:  # Venta (bajista)
            confirmacion_encontrada = any(datos['bajista'] for datos in confirmaciones.values())
        
        # Ajustar lotaje si no hay confirmación
        if not confirmacion_encontrada:
            #print("No hay confirmación de vela en 1m, 5m o 15m. Ajustando lotaje.")
            lotaje *= 0.05 # Ajustar el lotaje según el peso
            #lotaje = 0 # Cancelar la operación si no hay confirmación
            
        return lotaje"""
    
    def trend_confirmation(self, args, action, action2, name2, lotaje, peso):
        # Obtener datos del timeframe de 1 hora
        data_1h = args["all_data_1h"]['EURUSD']
        data_1h = data_1h[data_1h["time"] <= args["actual_time"]]
        
        # Calcular media móvil rápida (8 períodos) y lenta (21 períodos)
        sma_fast = data_1h['SMA_8'].iloc[-1]
        sma_slow = data_1h['SMA_21'].iloc[-1]
        
        # Determinar tendencia basada en medias móviles y análisis de precio
        trend = 0  # 0 = sin tendencia clara, 1 = alcista, -1 = bajista
        
        # Método 1: Medias móviles
        if sma_fast > sma_slow:
            trend += 1
        elif sma_fast < sma_slow:
            trend -= 1
        
        data_1h_recent = data_1h.tail(20)  # Últimas 20 velas de 1 hora
        
        # Método 2: Análisis de máximos y mínimos
        # Comparar máximos y mínimos de las últimas 10 velas vs las 10 anteriores
        recent_10 = data_1h_recent.tail(10)
        previous_10 = data_1h_recent.head(10)
        
        if recent_10['high'].max() > previous_10['high'].max() and recent_10['low'].min() > previous_10['low'].min():
            trend += 1  # Máximos y mínimos ascendentes = tendencia alcista
        elif recent_10['high'].max() < previous_10['high'].max() and recent_10['low'].min() < previous_10['low'].min():
            trend -= 1  # Máximos y mínimos descendentes = tendencia bajista
        
        # Método 3: Dirección del precio
        # Compara el precio de cierre más reciente con el de hace 15 velas
        if data_1h_recent['close'].iloc[-1] > data_1h_recent['close'].iloc[5]:
            trend += 1
        elif data_1h_recent['close'].iloc[-1] < data_1h_recent['close'].iloc[5]:
            trend -= 1
        
        # Determinar tendencia final (votación de los 3 métodos)
        final_trend = 0
        if trend > 0:
            final_trend = 1  # Tendencia alcista
        elif trend < 0:
            final_trend = -1  # Tendencia bajista
        
        # Comprobar si la operación va a favor o contra tendencia
        if final_trend != 0 and action != 0:  # Si hay tendencia clara y se propone una operación
            if action * final_trend < 0:  # Operación va contra tendencia
                #print("La operación va contra la tendencia. Ajustando lotaje.")
                lotaje *= 0.05
                #lotaje = 0
        
        # Si la operación va a favor de tendencia o no hay tendencia clara, mantener lotaje original
        return lotaje

    def liquidity_confirmation(self,args,action,action2,name2,lotaje,peso):
        mayores_diff = []
        menores_diff = []
        liquidez_final = args["puntos_liquidez"]
        ask = args['ask']
        bid = args['bid']

        for i in liquidez_final.keys():
            if action > 0:
                if i > ask:
                    mayores_diff.append(i - ask)

                elif i < ask:
                    menores_diff.append(ask - i)

            elif action < 0:
                if i < bid:
                    menores_diff.append(bid - i)

                elif i > bid:
                    mayores_diff.append(i - bid)

        if len(mayores_diff) == 0 and len(menores_diff) != 0: #hay liquidez solo por debajo
            if action > 0:
                lotaje = lotaje * peso
        
        elif len(mayores_diff) != 0 and len(menores_diff) == 0: #hay liquidez solo por arriba
            if action < 0:
                lotaje = lotaje * peso

        elif len(mayores_diff) and len(menores_diff): #si hay liquidez por arriba y por abajo se calcula
            mean_mayores_diff = sum(mayores_diff) / len(mayores_diff)
            mean_menores_diff = sum(menores_diff) / len(menores_diff)

            if mean_mayores_diff > mean_menores_diff: # la liquidez de por debajo es mas potente porque están mas cerca
                if action > 0:
                    lotaje = lotaje * peso
            
            elif mean_mayores_diff < mean_menores_diff: # la liquidez de por arriba es mas potente porque están mas cerca
                if action < 0:
                    lotaje = lotaje * peso

        return lotaje

    def liquidity_tp_sl(self, accion, ask, bid, atr, atr_ma, spread, liquidity_levels, sl_multiplier, tp_multiplier):
        if accion > 0: #largo
            entrada = ask

            # Define valores iniciales usando el ATR como referencia
            default_tp = bid + (atr * (atr / atr_ma) * (tp_multiplier * self.multiplicador_tp_sl[0])) + spread
            default_sl = bid - (atr * (atr / atr_ma) * (sl_multiplier * self.multiplicador_tp_sl[0])) - spread

            tp = default_tp
            sl = default_sl
            
            try:
                if sl_multiplier != 1 and tp_multiplier != 1:
                    # Define un rango secundario (más amplio) para buscar puntos de liquidez
                    default_tp_2 = bid + (atr * (atr / atr_ma) * (tp_multiplier * self.multiplicador_tp_sl[1])) + spread
                    default_sl_2 = bid - (atr * (atr / atr_ma) * (sl_multiplier * self.multiplicador_tp_sl[1])) - spread

                    # Separar puntos de liquidez en por encima y por debajo del precio
                    liquidity_above = sorted([lvl for lvl in liquidity_levels if lvl > entrada])
                    liquidity_below = sorted([lvl for lvl in liquidity_levels if lvl < entrada], reverse=True)

                    # Busca puntos de liquidez dentro del rango primario
                    tp = next((lvl for lvl in liquidity_above if default_tp <= lvl <= default_tp_2), tp)
                    sl = next((lvl for lvl in liquidity_below if default_sl_2 <= lvl <= default_sl), sl)

                    # Ajusta TP y SL para que la distancia TP-entrada sea mayor que entrada-SL
                    if (entrada - sl) >= (tp - entrada):
                        tp = entrada + (entrada - sl) * (tp_multiplier/sl_multiplier)  # Aumenta ligeramente el TP para mantener la proporción
            except:
                pass

        elif accion < 0: #corto
            entrada = bid

            # Define valores iniciales usando el ATR como referencia
            default_tp = ask - (atr * (atr / atr_ma) * (tp_multiplier * self.multiplicador_tp_sl[0])) - spread
            default_sl = ask + (atr * (atr / atr_ma) * (sl_multiplier * self.multiplicador_tp_sl[0])) + spread

            tp = default_tp
            sl = default_sl

            try:
                if sl_multiplier != 1 and tp_multiplier != 1:
                    # Define un rango secundario (más amplio) para buscar puntos de liquidez
                    default_tp_2 = ask - (atr * (atr / atr_ma) * (tp_multiplier * self.multiplicador_tp_sl[1])) - spread
                    default_sl_2 = ask + (atr * (atr / atr_ma) * (sl_multiplier * self.multiplicador_tp_sl[1])) + spread

                    # Separar puntos de liquidez en por encima y por debajo del precio
                    liquidity_above = sorted([lvl for lvl in liquidity_levels if lvl > entrada])
                    liquidity_below = sorted([lvl for lvl in liquidity_levels if lvl < entrada], reverse=True)

                    # Busca puntos de liquidez dentro del rango primario
                    tp = next((lvl for lvl in liquidity_below if default_tp_2 <= lvl <= default_tp), tp)
                    sl = next((lvl for lvl in liquidity_above if default_sl <= lvl <= default_sl_2), sl)

                    # Ajusta TP y SL para que la distancia TP-entrada sea mayor que entrada-SL
                    if (sl - entrada) >= (entrada - tp):
                        tp = entrada - (sl - entrada) * (tp_multiplier / sl_multiplier)  # Disminuye ligeramente el TP para mantener la proporción
            except:
                pass

        return tp,sl,entrada

    #-----------------------------------------------------------------------------------------------------

    def calculate_entry(self, args, indicador_valor, pos_liquidez):
        accion, sl, tp, lotaje = 0, 0, 0, 0

        estrategias = [i(args, indicador_valor) for i in self.estrategias]
        accion = self.blend(estrategias, self.lista_pesos_estrategias)

        if accion != 0:
            data = args['actual_data']
            data_5m = args['actual_data_5m'].iloc[-1]
            ask,bid = args['ask'],args['bid']

            dinero = args["dinero"]
            atr_ma = data_5m['atr_ma']
            atr = data_5m['atr']
            sl_multiplier = self.atr_multipliers[args["temporality"]][0]
            tp_multiplier = self.atr_multipliers[args["temporality"]][1]
            spread = ask - bid

            tp, sl, entrada = self.liquidity_tp_sl(accion, ask, bid, atr, atr_ma, spread,args["puntos_liquidez"].keys(), sl_multiplier, tp_multiplier)
            #tp,sl,entrada=self.liquidity_tp_sl(accion,ask,bid, atr, atr_ma,spread,args["puntos_salida"].keys(),sl_multiplier,tp_multiplier)
            
            if accion > 0:
                riesgo = self.riesgo_por_operacion['largo']
                riesgo = riesgo - (0.015 * pos_liquidez) / 100
                riesgo=riesgo*abs(accion)
                
            elif accion < 0:
                riesgo = self.riesgo_por_operacion['corto']
                riesgo = riesgo - (0.015 * pos_liquidez) / 100
                riesgo = riesgo * abs(accion)
                
            lotaje = riesgo / (calcular_pips(entrada, sl) * 10)
            lotaje = lotaje * atr
            lotaje = lotaje * (dinero[1] * 6000)
            lotaje = lotaje * (atr / atr_ma)
            
        else:
            accion = 0

        return accion, tp, sl, lotaje, estrategias

    def run(self, args):
        action, accion, sl, tp, lotaje = 0, 0, 0, 0, 0
        liquidez_final, posiciones_cortos, posiciones_largos = args["puntos_liquidez"], args["posiciones_cortos"], args["posiciones_largos"]

        data = args["all_data"][args["name"]]
        #data_1m_all = args["all_data_1m"][args["name"]]
        data_5m_all = args["all_data_5m"][args["name"]]
        #data_2m_all = args["all_data_2m"][args["name"]]
        data_15m_all = args["all_data_15m"][args["name"]]
        #data_1h_all = args["all_data_1h"][args["name"]]

        if args['mode'] == 'backtesting':
            data = data[data["time"] <=  args['actual_time']].iloc[-1]
            data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
            #data_1m_all = data_1m_all[data_1m_all["time"] <= args["actual_time"]]
            #data_2m_all = data_2m_all[data_2m_all["time"] <= args["actual_time"]]
            data_15m_all = data_15m_all[data_15m_all["time"] <= args["actual_time"]]
            #data_1h_all = data_1h_all[data_1h_all["time"] <= args["actual_time"]]
        else:
            data = data.iloc[-1]

        args['actual_data'] = data
        args['actual_data_5m'] = data_5m_all
        #args['actual_data_1m'] = data_1m_all
        #args['actual_data_2m'] = data_2m_all
        args['actual_data_15m'] = data_15m_all
        #args['actual_data_1h'] = data_1h_all

        ask, bid = args['ask'], args['bid']

        #para comprobar si hay alguna entrada que se repita
        #si sale mal es peligroso porque es como doblar una apuesta
        try:
            action, tp, sl, lotaje, estrategias = self.calculate_entry(args, args['name'], liquidez_final[bid])
        except:
            try:
                action, tp, sl, lotaje, estrategias = self.calculate_entry(args, args['name'], liquidez_final[ask])                
            except:
                action, tp, sl, lotaje, estrategias = self.calculate_entry(args, args['name'], max(liquidez_final.values()))

        # validate signal
        if action != 0 or args['comprobacion_inicial']: # Este if es para confirmaciones
            try:
                try:
                    name2 = max(args["correlation_dict"], key=lambda k: abs(args["correlation_dict"][k][0]))
                    name2 = name2.split("-")[1]
        
                    action2, _, _, _, _ = self.calculate_entry(args, name2, 0)
                except:
                    print("Error getting confirmation entry")
                    name2 = args['name']
                    action2 = 0
            
                action = 1 if action > 0 else -1
                action2 = 1 if action2 > 0 else -1

                lotaje_inicial = lotaje
                lista_confirmaciones_realizadas = []

                # ---- Calcular si se confirma o no la entrada ----
                for _, i in enumerate(self.confirmations):
                    lotaje_inicial_iteracion = lotaje
                    lotaje = i(args, action, action2, name2, lotaje, self.lista_pesos_confirmaciones[_])
                    if lotaje != lotaje_inicial_iteracion:
                        lista_confirmaciones_realizadas.append(1)

                confirmacion_fibonacci = self.fibonacci_retracement(data_5m_all[-21:], ask)
                if confirmacion_fibonacci == 1 and action == -1:
                    lotaje = lotaje * self.lista_pesos_confirmaciones[5]
                elif confirmacion_fibonacci == -1 and action == 1:
                    lotaje = lotaje * self.lista_pesos_confirmaciones[5]
                
                #print(sum(lista_confirmaciones),len(self.confirmations))
                #      3, 5  (significa que ha bajado el lotaje en 3 ocasiones, por lo que el peso para sacara a delante la estrategia es mas pequeño)
                #      (5-3)/5 => 2/5 esto es lo que se baja el lotaje                
                #      (len(self.confirmations)-sum(lista_confirmaciones))/len(self.confirmations)
                
                if lotaje < lotaje_inicial * ((self.lista_pesos_confirmaciones_ordenada[0] * self.lista_pesos_confirmaciones_ordenada[1]) - 0.01):
                    action = 0
                else:
                    print("Calculating confirmation entry for ", name2)
                    print("accion 1: ", action)
                    print("accion 2: ", action2)
                    print()

            except Exception as e: #sale excepcion si coincide que de este accion ese dia no hay datos
                print(f"Error: {e}")
                traceback.print_exc()
                print(f"Operacion cancelada {action} {tp} {sl} {lotaje} {ask} {bid}")
                action2 = 0

        # cobertura con indices, etc
        # condicion comprar indice

        if action > 0:
            action = 1
        elif action < 0:
            action = -1

        #if moneda_activo!=moneda_cuenta:
            # accion contraria como proteccion
            #if action==-1: comprar moneda
            #if action==1: vender moneda

        #return action,sl,tp,round(lotaje,2),accion
        return action, sl, tp, round(lotaje, 2), accion, estrategias
    
    def estrategia_trend_continuation2(self, args, indicador_valor):
        """
        Estrategia mejorada de continuación de tendencia que combina:
        1. Análisis de tendencia multi-timeframe
        2. Confirmación de momentum avanzada
        3. Filtros de volatilidad y volumen
        4. Indicadores de Smart Money Concepts
        """
        action = 0

        # Obtener datos
        data_5m_all = args["all_data_5m"][indicador_valor]
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        if len(data_5m_all) < 30:
            print("No hay suficientes datos para la estrategia de continuación de tendencia avanzada.") 
            return 0

        current = data_5m_all.iloc[-1]
        prev = data_5m_all.iloc[-2]

        # Obtener el punto de liquidez más cercano
        precio_actual = current['close']
        puntos_liquidez = list(args['puntos_liquidez'].keys())
        if not puntos_liquidez:
            return 0
        punto_cercano = min(puntos_liquidez, key=lambda x: abs(x - precio_actual))

        # 1. ANÁLISIS DE TENDENCIA AVANZADO
        # Usando múltiples EMAs y el índice de Choppiness
        trend_up = (
            current['EMA_8'] > current['EMA_21'] > current['EMA_55'] and
            current['close'] > current['EMA_8'] and
            current['CHOP'] < 61.8  # Mercado no lateral
        )
        
        trend_down = (
            current['EMA_8'] < current['EMA_21'] < current['EMA_55'] and
            current['close'] < current['EMA_8'] and
            current['CHOP'] < 61.8  # Mercado no lateral
        )

        # 2. CONFIRMACIÓN DE MOMENTUM AVANZADA
        # Usando Schaff Trend Cycle, Squeeze Momentum y Fisher Transform
        momentum_up = (
            current['STC'] > 75 and
            current['SQUEEZE_MOMENTUM'] > prev['SQUEEZE_MOMENTUM'] and
            current['FISHER'] > current['FISHER_SIGNAL'] and
            current['KST'] > current['KST_SIGNAL']
        )

        momentum_down = (
            current['STC'] < 25 and
            current['SQUEEZE_MOMENTUM'] < prev['SQUEEZE_MOMENTUM'] and
            current['FISHER'] < current['FISHER_SIGNAL'] and
            current['KST'] < current['KST_SIGNAL']
        )

        # 3. CONFIRMACIÓN DE VOLUMEN Y VOLATILIDAD
        # Usando Elder Force Index y ATR
        vol_confirm_up = (
            current['EFI'] > 0 and  # Elder Force Index positivo
            current['ADX'] > 25 and  # Tendencia fuerte
            current['atr'] > current['atr_ma'] * 0.8  # Volatilidad suficiente
        )

        vol_confirm_down = (
            current['EFI'] < 0 and  # Elder Force Index negativo
            current['ADX'] > 25 and  # Tendencia fuerte
            current['atr'] > current['atr_ma'] * 0.8  # Volatilidad suficiente
        )

        # 4. FILTROS ADICIONALES
        # Usando Vortex Indicator y DMI
        additional_up = (
            current['VI_PLUS'] > current['VI_MINUS'] and
            current['DMI_PLUS'] > current['DMI_MINUS'] and
            current['WILLR'] < -80  # Sobrevendido en Williams %R
        )

        additional_down = (
            current['VI_MINUS'] > current['VI_PLUS'] and
            current['DMI_MINUS'] > current['DMI_PLUS'] and
            current['WILLR'] > -20  # Sobrecomprado en Williams %R
        )

        # Señales de entrada solo cuando el precio está cerca de un punto de liquidez
        if abs(precio_actual - punto_cercano) < (current['high'] - current['low']):
            # Señal de compra
            if (trend_up and momentum_up and vol_confirm_up and additional_up and
                not current['SQUEEZE_ON']):  # No estamos en squeeze
                action = 1
                
            # Señal de venta
            elif (trend_down and momentum_down and vol_confirm_down and additional_down and
                not current['SQUEEZE_ON']):  # No estamos en squeeze
                action = -1

        return action
    
    def estrategia_trend_continuation(self, args, indicador_valor):
        """
        Estrategia de continuación de tendencia que busca entradas en puntos de liquidez
        cuando hay una tendencia clara y señales de continuación.
        """
        action = 0

        # Obtener datos
        data_5m_all = args["all_data_5m"][indicador_valor]
        data_5m_all = data_5m_all[data_5m_all["time"] <= args["actual_time"]]
        if len(data_5m_all) < 30:
            print("No hay suficientes datos para la estrategia de continuación de tendencia.")
            return 0

        current = data_5m_all.iloc[-1]
        prev = data_5m_all.iloc[-2]

        # Obtener el punto de liquidez más cercano
        precio_actual = current['close']
        puntos_liquidez = list(args['puntos_liquidez'].keys())
        if not puntos_liquidez:
            return 0
        punto_cercano = min(puntos_liquidez, key=lambda x: abs(x - precio_actual))

        # 1. Identificar tendencia usando EMAs
        trend_up = (current['EMA_8'] > current['EMA_21'] > current['EMA_55'] and 
                   current['close'] > current['EMA_8'])
        trend_down = (current['EMA_8'] < current['EMA_21'] < current['EMA_55'] and 
                     current['close'] < current['EMA_8'])

        # 2. Confirmación de momentum con RSI y MACD
        rsi = current['RSI']
        rsi_prev = data_5m_all['RSI'].iloc[-2]
        macd_increasing = (current['MACD'] > prev['MACD'] and 
                          current['MACD'] > current['MACD_Signal'])
        macd_decreasing = (current['MACD'] < prev['MACD'] and 
                          current['MACD'] < current['MACD_Signal'])

        # 3. Confirmación de volumen
        vol_avg = data_5m_all['tick_volume'].rolling(20).mean().iloc[-1]
        strong_volume = current['tick_volume'] > vol_avg * 1.2

        # 4. ADX para confirmar fuerza de tendencia
        strong_trend = current['ADX'] > 25 and current['ADX'] > prev['ADX']

        # Señales de entrada
        if abs(precio_actual - punto_cercano) < (current['high'] - current['low']):
            # Señal de compra: Tendencia alcista + toca punto de liquidez + confirmaciones
            if (trend_up and strong_trend and 
                rsi > rsi_prev and rsi > 45 and 
                macd_increasing and strong_volume):
                action = 1
                
            # Señal de venta: Tendencia bajista + toca punto de liquidez + confirmaciones
            elif (trend_down and strong_trend and 
                  rsi < rsi_prev and rsi < 55 and 
                  macd_decreasing and strong_volume):
                action = -1

        return action
    