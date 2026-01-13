import MetaTrader5 as mt
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime,timedelta,date
import pandas as pd
import numpy as np
import torch
import ast
from dotenv import dotenv_values
from playsound import playsound
import holidays
import mplfinance as mpf
from utils import *
from estrategia import MarketMaster, liquidity, simplify_liquidity, simplify_liquidity_2, all_liquidity_2, all_liquidity, calcular_pips
from estrategia.gestion_dinamica import MarketMasterManagement
from pre_implemented_strategies.book_maths import Maths
from typing import NamedTuple
from math import exp
import investpy as ivp
from tqdm import tqdm
import time
import warnings
import sys
from data import raw_data_to_df
from collections import ChainMap
import traceback
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import itertools
import polars as pl
import csv
import re

warnings.filterwarnings("ignore")

#https://www.mql5.com/en/docs/python_metatrader5

def actualizar_csv():
    csv_path = "data/registros/registro_backtesting.csv"
    txt_path = "data/registros/registro_backtesting.txt"
    
    if not os.path.exists(txt_path):
        print("El archivo TXT no existe.")
        return
    
    with open(txt_path, "r") as txt_file:
        lines = txt_file.readlines()
    
    if not lines:
        print("El archivo TXT está vacío.")
        return
    
    # Expresión regular para extraer los datos correctamente
    pattern = re.compile(r"entrada:(\d+\.\d+) sl:(\d+\.\d+) tp:(\d+\.\d+) lotaje: (\d+\.\d+) a las (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    
    data = []
    for line in lines:
        match = pattern.search(line)
        if match:
            entrada, sl, tp, lotaje, fecha = match.groups()
            tipo = "Largo" if "largo" in line else "Corto"
            
            # Determinar si la operación alcanzó TP o SL
            resultado = "TP" if float(entrada) < float(tp) else "SL"
            fecha_resultado = fecha  # La fecha de resultado será la misma que la de la entrada
            
            data.append([fecha, tipo, float(entrada), float(lotaje), float(tp), float(sl), resultado, fecha_resultado])
    
    # Escribir los datos en el CSV
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Fecha", "Tipo", "Entrada", "Lotaje", "TP", "SL", "Resultado", "Fecha Resultado"])
        writer.writerows(data)
    
    print("CSV actualizado correctamente desde el TXT con fechas reales de cierre.")

def mostrar_posiciones(posiciones) -> None:
    print(f"\n-------------------- Mostrando {len(posiciones)} Posiciones --------------------\n")
    for i in posiciones:
        print(i)
        try:
            print('Pips tp: ', calcular_pips(i.entrada, i.tp), 'Pips sl: ', calcular_pips(i.entrada, i.sl), 'RR: ', calcular_pips(i.entrada,i.tp) / calcular_pips(i.entrada,i.sl))
        except:
            print('Pips tp: ', calcular_pips(i.entrada, i.tp), 'Pips sl: ', calcular_pips(i.entrada, i.sl), 'RR: 100')

def agregar_puntos_liquidez(args):
    ask, bid, liquidez_final, atr_multipliers, curr_spread, atr, atr_ma, data, pos_dia = args
    entrada = 0
    if bid in liquidez_final:
        tp_corto = ask - (atr * (atr / atr_ma) * atr_multipliers["all_data_5m"][1]) - curr_spread
        tp_largo = bid + (atr * (atr / atr_ma) * atr_multipliers["all_data_5m"][1]) + curr_spread
        resultado_corto, resultado_largo = 0, 0

        for _, dia_para_encontrar_sentido in data.iterrows():
            if dia_para_encontrar_sentido['bid'] >= tp_largo:
                resultado_largo = 'tp'
                break

            elif dia_para_encontrar_sentido['ask'] <= tp_corto:
                resultado_corto = 'tp'
                break
            
        if resultado_largo == 'tp':
            mejor_accion = 1
            entrada = ask
        elif resultado_corto == 'tp':
            mejor_accion = 0
            entrada = bid
        else:
            mejor_accion = None


    elif ask in liquidez_final:
        tp_corto = ask - (atr * (atr / atr_ma) * atr_multipliers["all_data_5m"][1]) - curr_spread
        tp_largo = bid + (atr * (atr / atr_ma) * atr_multipliers["all_data_5m"][1]) + curr_spread

        resultado_corto, resultado_largo = 0, 0

        for _, dia_para_encontrar_sentido in data.iterrows():
            if dia_para_encontrar_sentido['bid'] >= tp_largo:
                resultado_largo = 'tp'
                break
            elif dia_para_encontrar_sentido['ask'] <= tp_corto:
                resultado_corto = 'tp'
                break

        if resultado_largo == 'tp':
            mejor_accion = 1
            entrada = ask
        elif resultado_corto == 'tp':
            mejor_accion = 0
            entrada = bid
        else:
            mejor_accion = None
            
    return mejor_accion, entrada, pos_dia, tp_largo, tp_corto

def calcular_liquidez(liquidez, lista):
    return all_liquidity_2(liquidez, lista)

def recalcular_puntos(list_liquidity_ask, list_liquidity_bid, resultado_liquidez_bid, resultado_liquidez_ask, fecha_busqueda, limite_potencia, first_recalculation, executor):
    puntos_liquidez = {}
    liquidez_final = {}
    simplify_liquidity_tasks = []

    liquidez_bid = resultado_liquidez_bid.get(fecha_busqueda, {})
    liquidez_ask = resultado_liquidez_ask.get(fecha_busqueda, {})

    # Recorremos las listas de atrás hacia adelante, optimizando el acceso
    for pos_liquidity in range(len(list_liquidity_ask)):
        # Usamos `reversed` para acceder a las últimas posiciones sin crear sublistas
        i_bid = list_liquidity_bid[-(pos_liquidity + 1):]
        i_ask = list_liquidity_ask[-(pos_liquidity + 1):]

        # Evitamos el uso de `get` repetidamente, accedemos directamente a los valores
        #all_liquidity_bid = all_liquidity_2(liquidez_bid, i_bid)
        #all_liquidity_ask = all_liquidity_2(liquidez_ask, i_ask)
        future_bid = executor.submit(calcular_liquidez, liquidez_bid, i_bid)
        future_ask = executor.submit(calcular_liquidez, liquidez_ask, i_ask)

        all_liquidity_bid = future_bid.result()  # Bloquea hasta que la tarea termine
        all_liquidity_ask = future_ask.result()  # Bloquea hasta que la tarea termine

        # Simplificamos la liquidez y agregamos los resultados a `puntos_liquidez`
        simplify_liquidity_tasks.append((all_liquidity_bid, all_liquidity_ask, limite_potencia))
        #puntos = simplify_liquidity(all_liquidity_bid, all_liquidity_ask, trigger=limite_potencia)[0]
        #puntos_liquidez.update(puntos)

        #liquidez_final.update(zip(puntos, np.full(len(puntos), pos_liquidity)))

    futures = [executor.submit(simplify_liquidity_2, value) for value in simplify_liquidity_tasks]
    results = [future.result()[0] for future in concurrent.futures.as_completed(futures)]

    for pos_liquidity, result in enumerate(results):
        puntos_liquidez.update(result)
        liquidez_final.update(zip(result, np.full(len(result), pos_liquidity)))

    return liquidez_final, {}, puntos_liquidez

def get_data_temporality(nombre, temporality, dia, days_back):
    # Descargamos los datos de la temporalidad deseada y los convertimos a un DataFrame
    data_5m = pd.DataFrame(mt.copy_rates_range(nombre, temporality, dia - timedelta(days_back), dia + timedelta(1)))
    # No deshacemos de la columna 'real_volume' porque nos devuelve siempre 0 y 'tick_volume' se puede usar como volumen (a parte de que es mejor)
    data_5m = data_5m.drop(["real_volume"], axis = 1)
    data_5m['spread'] = data_5m["spread"] / 100000
    data_5m["mean_price"] = (data_5m["high"] + data_5m["low"] + data_5m["close"]) / 3
    data_5m["rmv"] = data_5m["mean_price"] * data_5m['tick_volume']
    data_5m["time"] = [datetime.fromtimestamp(item) for item in data_5m["time"]]
    data_5m["diff"] = data_5m["mean_price"].diff().fillna(0)
    data_5m.fillna(0)
    data_5m = data_5m.replace('nan', '0')
    data_5m.iloc[:, 1:] = data_5m.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return data_5m

class Backtesting:
    def __init__(self, dinero_inicial, name, RR, divisor_tiempo_limite, tamanyo_break_even, min_spread_trigger, max_spread_trigger, 
                 break_even_size, trigger_trailing_tp, trigger_trailing_sl, enable_break_even, enable_dinamic_sl, enable_dinamic_tp, maximo_perdida_diaria, maximo_beneficio_diario, 
                 lotaje_minimo, longitud_liquidez, ticks_refresco_liquidez, maximo_operaciones_consecutivas, multiplicador_margen_total, limite_potencia, bloquear_noticias, 
                 fecha_inicio, fecha_final, verbose, sounds, grafico, recavar_datos, save_file, maximo_operaciones_diarias, atr_window, atr_ma, atr_multipliers, re_evaluar, 
                 riesgo_por_operacion, dias_retiro, porcentaje_retiro, porcentaje_umbral_ganancias, moneda_cuenta, lista_pesos_confirmaciones, corr_limit, zona_horaria_operable, 
                 multiplicador_tp_sl, lista_pesos_estrategias, *args, **kwargs):
        
        #-------------------Inicialización---------------------------

        account = dotenv_values("account.env")

        mt.initialize()
        
        print("\nInitialized")

        if not mt.initialize():
            raise Error(f'initialize() failed, error code = {mt.last_error()}')

        mt.login(account["login"], password = account["password"], server = account["server"])
        print("Logged in")
        self.timezone = int(int(time.strftime("%z", time.gmtime())) / 100)

        #-------------------Configuraciones--------------------------

        self.name = name
        self.RR = RR
        
        self.multiplicador_margen_total = multiplicador_margen_total
        self.tamanyo_break_even = tamanyo_break_even #pips a partir los cuales se aplica break even
        self.break_even_size = break_even_size #pip despues de break even

        self.enable_break_even = enable_break_even
        self.enable_dinamic_sl = enable_dinamic_sl
        self.enable_dinamic_tp = enable_dinamic_tp

        self.multiplicador_tp_sl = multiplicador_tp_sl

        self.re_evaluar = re_evaluar

        self.trigger_trailing_tp = trigger_trailing_tp #pips
        self.trigger_trailing_sl = trigger_trailing_sl #pips

        self.maximo_perdida_diaria = maximo_perdida_diaria
        #self.maximo_perdida_diaria=riesgo_por_operacion*(maximo_operaciones_consecutivas+1)
        self.maximo_beneficio_diario = maximo_beneficio_diario
        self.lotaje_minimo = lotaje_minimo
        self.longitud_liquidez = longitud_liquidez
        self.ticks_refresco_liquidez = ticks_refresco_liquidez
        self.maximo_operaciones_consecutivas = maximo_operaciones_consecutivas
        self.limite_potencia = limite_potencia #potencia menor que esta no sera tomada en cuenta

        self.maximo_operaciones_diarias = maximo_operaciones_diarias

        self.bloquear_noticias = bloquear_noticias

        #-------------------Configuraciones--------------------------

        self.fecha_inicio = fecha_inicio #2023,3,1
        self.fecha_final = fecha_final #2023,6,1

        self.verbose = verbose
        self.sounds = sounds
        self.grafico = grafico
        self.recavar_datos = recavar_datos
        self.save_file = save_file
        self.moneda_cuenta = moneda_cuenta

        self.comisiones = 3 # 3 x lote
        self.unidad_lote = 100000
        self.multiplicador = 100
        self.margin = 1
        self.dinero_ultimo_retiro = dinero_inicial
        self.dinero_inicial = dinero_inicial
        self.dinero = self.dinero_inicial
        self.dinero_inicial_diario = self.dinero
        self.max_dinero = self.dinero
        self.min_dinero = self.dinero
        self.perdida_maxima_cuenta = 10 / 100
        self.historial_dinero = [self.dinero_inicial]
        self.historial_dinero_con_beneficio = [self.dinero_inicial]
        self.historial_velas = []
        self.operaciones_cerradas = []
        self.operaciones_cerradas_ids = set()
        self.positivos_negativos = [0, 0]
        self.n_prints_comprobacion_inicial = 0
        self.original_min_spread_trigger = min_spread_trigger
        self.original_max_spread_trigger = max_spread_trigger
        self.min_spread_trigger = min_spread_trigger
        self.max_spread_trigger = max_spread_trigger
        self.n_cortos = 0
        self.n_largos = 0
        self.n_tp = 0
        self.n_paradas = 0
        self.n_sl = 0
        self.n_tp_diario = 0
        self.n_sl_diario = 0
        self.n_p_diario = 0
        self.n_dias = 0

        self.retiros = []
        self.dias_retiro = dias_retiro
        self.porcentaje_retiro = porcentaje_retiro
        self.porcentaje_umbral_ganancias = porcentaje_umbral_ganancias

        self.comision_retiro = 95 / 100
        self.estadisticas_operaciones = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        self.posiciones = []

        #----------MarketMasterMaths----------
        
        self.atr_window = atr_window
        self.atr_ma = atr_ma
        self.atr_multipliers = atr_multipliers
        
        self.lista_pesos_confirmaciones = lista_pesos_confirmaciones
        self.zona_horaria_operable_original = zona_horaria_operable
        self.divisor_tiempo_limite = divisor_tiempo_limite

        self.corr_limit = corr_limit

        self.lista_pesos_estrategias = lista_pesos_estrategias

        self.riesgo_por_operacion = riesgo_por_operacion
        
        self.historial_total_puntos_liquidez = {}

        super().__init__(*args, **kwargs)

    def _registrar_operaciones_cerradas(self, posiciones):
        for posicion in posiciones:
            if getattr(posicion, "resultado") != 0 and getattr(posicion, "id") not in self.operaciones_cerradas_ids:
                self.operaciones_cerradas.append(posicion)
                self.operaciones_cerradas_ids.add(getattr(posicion, "id"))

    def _construir_historial_velas(self):
        if not self.historial_velas:
            return pd.DataFrame()
        historial = pd.concat(self.historial_velas, ignore_index=True)
        historial = historial.drop_duplicates(subset=["time"]).sort_values("time")
        return historial

    def _mostrar_grafico_velas_operaciones(self):
        if not self.grafico:
            return
        historial = self._construir_historial_velas()
        if historial.empty or not self.operaciones_cerradas:
            return

        historial = historial.copy()
        historial = historial.set_index("time")
        historial = historial.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        )

        num_rows = len(historial)
        if num_rows == 0:
            return
        entry_long_win = np.full(num_rows, np.nan)
        entry_long_loss = np.full(num_rows, np.nan)
        entry_short_win = np.full(num_rows, np.nan)
        entry_short_loss = np.full(num_rows, np.nan)
        exit_win = np.full(num_rows, np.nan)
        exit_loss = np.full(num_rows, np.nan)

        for operacion in self.operaciones_cerradas:
            fecha_entrada = getattr(operacion, "fecha")
            fecha_salida = getattr(operacion, "fecha_resultado")
            entrada = getattr(operacion, "entrada")
            tipo = getattr(operacion, "typo")
            resultado = getattr(operacion, "resultado")

            if resultado == 1:
                salida = getattr(operacion, "tp")
            elif resultado == -1:
                salida = getattr(operacion, "sl")
            else:
                salida = resultado

            entrada_idx = historial.index.get_indexer([fecha_entrada], method="nearest")[0]
            salida_idx = historial.index.get_indexer([fecha_salida], method="nearest")[0]

            beneficio = (salida - entrada) * (1 if tipo == 1 else -1)
            gano = beneficio >= 0

            if tipo == 1 and gano:
                entry_long_win[entrada_idx] = entrada
            elif tipo == 1:
                entry_long_loss[entrada_idx] = entrada
            elif gano:
                entry_short_win[entrada_idx] = entrada
            else:
                entry_short_loss[entrada_idx] = entrada

            if gano:
                exit_win[salida_idx] = salida
            else:
                exit_loss[salida_idx] = salida

        addplots = []
        if np.isfinite(entry_long_win).any():
            addplots.append(mpf.make_addplot(entry_long_win, type="scatter", marker="^", markersize=70, color="green"))
        if np.isfinite(entry_long_loss).any():
            addplots.append(mpf.make_addplot(entry_long_loss, type="scatter", marker="^", markersize=70, color="red"))
        if np.isfinite(entry_short_win).any():
            addplots.append(mpf.make_addplot(entry_short_win, type="scatter", marker="v", markersize=70, color="green"))
        if np.isfinite(entry_short_loss).any():
            addplots.append(mpf.make_addplot(entry_short_loss, type="scatter", marker="v", markersize=70, color="red"))
        if np.isfinite(exit_win).any():
            addplots.append(mpf.make_addplot(exit_win, type="scatter", marker="o", markersize=50, color="green"))
        if np.isfinite(exit_loss).any():
            addplots.append(mpf.make_addplot(exit_loss, type="scatter", marker="o", markersize=50, color="red"))

        fig, axlist = mpf.plot(
            historial,
            type="candle",
            addplot=addplots,
            volume=False,
            returnfig=True,
            style="yahoo",
            title="Backtesting - Velas y operaciones (5m)",
        )
        ax = axlist[0]

        for operacion in self.operaciones_cerradas:
            fecha_entrada = getattr(operacion, "fecha")
            fecha_salida = getattr(operacion, "fecha_resultado")
            entrada = getattr(operacion, "entrada")
            tipo = getattr(operacion, "typo")
            resultado = getattr(operacion, "resultado")

            if resultado == 1:
                salida = getattr(operacion, "tp")
            elif resultado == -1:
                salida = getattr(operacion, "sl")
            else:
                salida = resultado

            entrada_idx = historial.index.get_indexer([fecha_entrada], method="nearest")[0]
            salida_idx = historial.index.get_indexer([fecha_salida], method="nearest")[0]

            entrada_time = historial.index[entrada_idx]
            salida_time = historial.index[salida_idx]
            beneficio = (salida - entrada) * (1 if tipo == 1 else -1)
            color = "green" if beneficio >= 0 else "red"
            ax.plot([entrada_time, salida_time], [entrada, salida], linestyle="--", color=color, linewidth=1.0)

        legend_items = [
            mpatches.Patch(color="green", label="Ganadora"),
            mpatches.Patch(color="red", label="Perdedora"),
        ]
        ax.legend(handles=legend_items, loc="upper left")
        plt.tight_layout()
        plt.show()

    def _guardar_csv_operaciones(self):
        if not self.operaciones_cerradas:
            return
        csv_path = "data/registros/registro_backtesting.csv"
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Fecha", "Tipo", "Entrada", "Lotaje", "TP", "SL", "Resultado", "Fecha Resultado"])
            for operacion in self.operaciones_cerradas:
                tipo = "Largo" if getattr(operacion, "typo") == 1 else "Corto"
                resultado = getattr(operacion, "resultado")
                if resultado == 1:
                    resultado_label = "TP"
                elif resultado == -1:
                    resultado_label = "SL"
                else:
                    resultado_label = "PARADA"
                writer.writerow(
                    [
                        getattr(operacion, "fecha"),
                        tipo,
                        getattr(operacion, "entrada"),
                        getattr(operacion, "lotaje"),
                        getattr(operacion, "tp"),
                        getattr(operacion, "sl"),
                        resultado_label,
                        getattr(operacion, "fecha_resultado"),
                    ]
                )

    def run(self) -> None:

        simbolos = mt.symbols_get()
        simbolos = [symbol for symbol in simbolos if symbol.visible]

        self.lista_activos = []

        for i in simbolos:
            self.lista_activos.append(i.name)

        self.lista_activos=['EURUSD', 'GBPUSD', 'USDJPY'] #quitar esto cuando esté listo

        print()

        self.correlation_dict={} # name-name:correlation

        combinaciones = itertools.combinations(self.lista_activos, 2)

        print("Activos invertibles: ", self.lista_activos)
        print("Activos invertibles: ", len(self.lista_activos))

        file = open("data/registros/registro_backtesting.txt", "w")
        try:
            if self.save_file:
                orig_stdout = sys.stdout
                f = open('out.txt', 'w')
                sys.stdout = f

            if self.break_even_size >= self.tamanyo_break_even:
                raise Error("Tamaño de break even superior al de break even")

            start = time.time()

            list_days = []
            list_days_5m = []
            list_liquidity_bid = []
            list_liquidity_ask = []
            anterior_mayor_spread = None

            marketmaster = MarketMaster(self.atr_multipliers, self.maximo_perdida_diaria, self.riesgo_por_operacion, self.dinero_inicial, 
                                        self.lista_pesos_confirmaciones, self.corr_limit, generate_seed(), self.multiplicador_tp_sl, lista_pesos_estrategias = self.lista_pesos_estrategias)
            
            marketmastermanagement = MarketMasterManagement(marketmaster)

            # Recorrer los días entre la fecha de inicio y la fecha final
            for n in range(int((self.fecha_final - self.fecha_inicio).days)):
                dia = self.fecha_inicio + timedelta(n)
                self.historial_puntos_liquidez = []
                self.correlation_dict = {}
                combinaciones = itertools.combinations(self.lista_activos, 2)
                if dia.weekday() not in [5, 6]:
                    first_recalculation = False
                    all_data = {}

                    if (self.historial_dinero[-1] - (self.dinero_inicial)) / (self.dinero_inicial) >- self.perdida_maxima_cuenta:
                        for name in tqdm(self.lista_activos, total = len(self.lista_activos)):
                            data = pd.DataFrame(mt.copy_ticks_range(name, dia, dia + timedelta(1), mt.COPY_TICKS_ALL)) # coger datos hasta el dia actual sin incluirlo para que no sea trampa
                            data['ask_dif'] = data['ask'].diff()
                            data['bid_dif'] = data['bid'].diff()

                            data['power'] = np.select(
                                [data["bid_dif"] > data["ask_dif"], data["bid_dif"] < data["ask_dif"]],
                                [1, -1],
                                default = 0
                            )

                            data['spread'] = data['ask'] - data['bid']
 
                            data = data.drop(columns = ["last", "volume", "time_msc", "volume_real"], axis = 1)
                            data["time"] = [datetime.fromtimestamp(item) for item in data["time"]]

                            all_data[name] = data

                        data = all_data[self.name]

                    if len(data) != 0:
                        if es_horario_de_verano(dia): # es verano entonces se usa el segundo trigger
                            #if anterior_mayor_spread!=None:
                                #self.max_spread_trigger=convertir_spread(data['ask'].iloc[0],anterior_mayor_spread)
                                #self.min_spread_trigger=convertir_spread(data['ask'].iloc[0],anterior_mayor_spread*0.8)
                            #else:
                            self.max_spread_trigger = convertir_spread(data['ask'].iloc[0], self.original_max_spread_trigger[1])
                            self.min_spread_trigger = convertir_spread(data['ask'].iloc[0], self.original_min_spread_trigger[1])

                            self.temporada = 'verano'
                            self.zona_horaria_operable = self.zona_horaria_operable_original[1]
                        else: # es invierno entonces se usa el primer trigger
                            #if anterior_mayor_spread!=None:
                                #self.max_spread_trigger=convertir_spread(data['ask'].iloc[0],anterior_mayor_spread)
                                #self.min_spread_trigger=convertir_spread(data['ask'].iloc[0],anterior_mayor_spread*0.8)
                            #else:
                            self.max_spread_trigger = convertir_spread(data['ask'].iloc[0], self.original_max_spread_trigger[0])
                            self.min_spread_trigger = convertir_spread(data['ask'].iloc[0], self.original_min_spread_trigger[0])

                            self.temporada = 'invierno'
                            self.zona_horaria_operable = self.zona_horaria_operable_original[0]

                    if self.bloquear_noticias and len(data) != 0:
                        if es_horario_de_verano(dia):
                            calendario_base = ivp.economic_calendar(time_zone=f"GMT +1:00",importances=["high"],from_date=f"{(dia-timedelta(days=1)).strftime("%d/%m/%Y")}",to_date=f"{dia.strftime("%d/%m/%Y")}")
                        else:
                            calendario_base = ivp.economic_calendar(time_zone=f"GMT",importances=["high"],from_date=f"{(dia-timedelta(days=1)).strftime("%d/%m/%Y")}",to_date=f"{dia.strftime("%d/%m/%Y")}")

                        try:                        
                            calendario_base = calendario_base.drop(["id","actual","forecast","previous"],axis=1)
                            calendario_base = calendario_base.iloc[np.where(calendario_base["importance"]=="high")]
                            calendario_base = calendario_base.drop(calendario_base[calendario_base['time'] == 'All Day'].index)
                            calendario_base["time"] = [datetime.strptime(item,"%H:%M") for item in calendario_base["time"]]                    
                            calendario = calendario_base
                        except:
                            calendario = None

                    n_prints_comprobacion = 0
                    if self.n_prints_comprobacion_inicial != 0:
                        n_prints_comprobacion = len(data) // self.n_prints_comprobacion_inicial

                    if len(data) > 0 and (self.historial_dinero[-1] - (self.dinero_inicial)) / (self.dinero_inicial) >- self.perdida_maxima_cuenta:
                        market_master_maths = Maths()

                        all_data_1m = {}
                        all_data_2m = {}
                        all_data_5m = {}
                        all_data_15m = {}
                        all_data_1h = {}
                        saltar = False

                        for nombre in tqdm(self.lista_activos, total = len(self.lista_activos)):                            
                            data_1m = get_data_temporality(nombre, mt.TIMEFRAME_M1, dia, 0)
                            data_2m = get_data_temporality(nombre, mt.TIMEFRAME_M2, dia, 0)
                            data_5m = get_data_temporality(nombre, mt.TIMEFRAME_M5, dia, 10)
                            data_15m = get_data_temporality(nombre, mt.TIMEFRAME_M15, dia, 7)
                            data_1h = get_data_temporality(nombre, mt.TIMEFRAME_H1, dia, 10)
                            data_5m_time = data_5m["time"]

                            try:
                                data_1m = market_master_maths.df(data_1m)
                                data_2m = market_master_maths.df(data_2m)
                                data_5m = market_master_maths.df(data_5m)
                                data_1h = market_master_maths.df(data_1h)
                                data_15m = market_master_maths.df(data_15m)
                            except Exception as e:
                                print(f"Error: {e}")
                                traceback.print_exc()
                                saltar = True

                            all_data_1m[nombre] = data_1m
                            all_data_2m[nombre] = data_2m
                            all_data_5m[nombre] = data_5m
                            all_data_15m[nombre] = data_15m
                            all_data_1h[nombre] = data_1h

                        data_1m = all_data_1m[self.name]
                        data_2m = all_data_2m[self.name]
                        data_5m = all_data_5m[self.name]
                        data_15m = all_data_15m[self.name]
                        data_1h = all_data_1h[self.name]

                        if not data_5m.empty:
                            self.historial_velas.append(data_5m[["time", "open", "high", "low", "close"]].copy())

                        if not data_5m.empty:
                            self.historial_velas.append(data_5m[["time", "open", "high", "low", "close"]].copy())

                        #print(data_5m.columns)
                        #filtered_df = data_5m[(data_5m['all_candle_patterns'] != 0)]
                        #print(filtered_df)

                        if len(list_days) < self.longitud_liquidez or saltar == True:
                            saltar = True
                        else:
                            saltar = False

                        prints_noticia = [0, 0]

                        print()
                        print(f"Start: {data["time"][0]} | End: {data["time"][len(data) - 1]}")
                        print(f"Backtesting con {len(data):,} datos y {round(self.dinero, 2)}€\n")
                        print("-" * 100)
                        print()

                        #if self.bloquear_noticias:
                            #calendario=calendario_base.iloc[np.where(calendario_base["date"]==f"{data["time"][100].date().strftime("%d/%m/%Y")}")]
                        
                        resultado_liquidez_bid = {} #fecha->dict(liquidez)
                        resultado_liquidez_ask = {}
                        puntos_liquidez = {}
                        operaciones_diarias_puntos = []

                        anterior_bid, anterior_ask = 0, 0

                        self.n_dias += 1
                        recalcular = False

                        try:                        
                            contador_data_5m = data_5m.index[data_5m['time'] == pd.to_datetime(dia.strftime("%Y-%m-%d %H:%M:%S"))]
                            if len(contador_data_5m) == 0:
                                contador_data_5m = data_5m.index[data_5m['time'] == pd.to_datetime((dia + timedelta(hours = 1)).strftime("%Y-%m-%d %H:%M:%S"))]

                            contador_data_5m = contador_data_5m.values[0]
                            anterior_contador_5m = None
                        except:
                            print("Error con contador_data_5m")
                            contador_data_5m = 0

                        datos = {}

                        for i in self.lista_activos:
                            datos[i] = pd.DataFrame(mt.copy_rates_range(i, mt.TIMEFRAME_M5, dia - timedelta(4), dia)) # coger datos hasta el dia actual sin incluirlo para que no sea trampa
                            datos[i]["time"] = [datetime.fromtimestamp(item) for item in datos[i]["time"]]

                        #print(self.correlation_dict,combinaciones,self.lista_activos)
                        total_size_combinations = 0
                        for activo1, activo2 in combinaciones:
                            total_size_combinations += 1
                            #print(activo1,activo2,len(activo1),len(activo2))
                            #if len(datos[activo1])==len(datos[activo2]):
                            serie1 = datos[activo1]['close']
                            serie2 = datos[activo2]['close']
                            serie1_normalizada = ((serie1 - serie1.min()) / (serie1.max() - serie1.min()))
                            serie2_normalizada = ((serie2 - serie2.min()) / (serie2.max() - serie2.min()))
                            correlacion = serie1_normalizada.corr(serie2_normalizada)
                            self.correlation_dict[f"{activo1}-{activo2}"] = (correlacion, np.var(serie1_normalizada), sum(serie1_normalizada) / len(serie1_normalizada), np.var(serie2_normalizada), sum(serie2_normalizada) / len(serie2_normalizada))
                            #if self.verbose:
                                #print(self.correlation_dict[f"{activo1}-{activo2}"])

                        self.correlation_dict = {clave: valor for clave, valor in self.correlation_dict.items() if clave.startswith(self.name)}
                        args={"correlation_dict":self.correlation_dict, "recalcular":False, "all_data":all_data, "all_data_1m":all_data_1m, "all_data_2m":all_data_2m, "all_data_5m":all_data_5m, 
                              "all_data_1h":all_data_1h, "all_data_15m":all_data_15m, "name":self.name, "temporada":self.temporada, "mode":"backtesting", "comprobacion_inicial":False}

                        #print(data)
                        #print(f"{data.loc[data['spread'] > 0, 'spread'].min():.10f},{max(data['spread']):.10f},{data['spread'].mean():.10f}")
                        
                        valor_limite_superior = 0.06
                        valor_limite_inferior = 0.03

                        data['limite_superior_izq'] = data['spread'].expanding().quantile(valor_limite_superior)
                        data['limite_inferior_izq'] = data['spread'].expanding().quantile(valor_limite_inferior)
                        data['limite_superior_der'] = data['spread'].expanding().quantile(1 - valor_limite_superior)
                        data['limite_inferior_der'] = data['spread'].expanding().quantile(1 - valor_limite_inferior)

                        data_iterate = pl.from_pandas(data)
                        anterior_data_5m = None                    

                        #for pos_dia,i in enumerate(tqdm(data.iterrows(),total=len(data))):
                        # Aqui se usa ThreadPoolExecutor para ejecutar las tareas en paralelo
                        # En esta parte del codigo se calcula la liquidez y se ejecutan las operaciones
                        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
                            for pos_dia, i in enumerate(tqdm(data_iterate.iter_rows(named = True), total = len(data))):                            
                                if n_prints_comprobacion != 0 and pos_dia != 0 and pos_dia % n_prints_comprobacion == 0:
                                    print(f"\nHora: {i["time"].time()} | stop loss: {self.n_sl_diario} | take profits: {self.n_tp_diario} | Número operaciones por cerrar: {len(self.posiciones_sl_cortos) + len(self.posiciones_sl_largos)}")

                                fecha = i["time"]
                                hora = fecha.time()

                                fecha_busqueda = f"{fecha.day}/{fecha.month}/{fecha.year}"
                                bid = float("%.5f" % i["bid"])
                                ask = float("%.5f" % i["ask"])
                                limite_superior_izq = i['limite_superior_izq']
                                limite_inferior_izq = i['limite_inferior_izq']
                                limite_superior_der = i['limite_superior_der']
                                limite_inferior_der = i['limite_inferior_der']
                                
                                precios = {bid, ask}
                                nuevas_entradas = []

                                curr_spread = ask - bid

                                if fecha_busqueda not in resultado_liquidez_ask and fecha_busqueda not in resultado_liquidez_bid:
                                    resultado_liquidez_bid = {str(fecha_busqueda):{}}
                                    resultado_liquidez_ask = {str(fecha_busqueda):{}}

                                if not saltar:
                                    try:
                                        actual, siguiente = data_5m_time[contador_data_5m], data_5m_time[contador_data_5m+1]
                                        
                                        if not (actual.hour == hora.hour and actual.minute <= hora.minute < siguiente.minute):    
                                            if siguiente.minute == 0:
                                                if not (actual.hour == hora.hour and actual.minute <= hora.minute < (siguiente.minute+60)):
                                                    contador_data_5m += 1
                                            else:
                                                contador_data_5m += 1
                                    except:
                                        pass

                                    if anterior_contador_5m != contador_data_5m:
                                        data_5m_actual = data_5m[:contador_data_5m + 1]
                                        #print("\n",data_5m_actual.iloc[-1],fecha,"\n")
                                        #limite_spread = (self.max_spread_trigger - data_5m_actual['adjusted_atr'] * (self.max_spread_trigger - self.min_spread_trigger)).values[-1]
                                        #limite_spread = (data_5m_actual['limite_superior'] - data_5m_actual['adjusted_atr'] * (data_5m_actual['limite_superior'] - data_5m_actual['limite_inferior'])).values[-1]
                                        limite_spread = (data_5m_actual['moda_superior1'] - data_5m_actual['adjusted_atr'] * (data_5m_actual['moda_superior1'] - data_5m_actual['moda_inferior'])).values[-1]
                                        anterior_contador_5m = contador_data_5m

                                        for indice_posicion, posicion in enumerate(self.posiciones):
                                            if getattr(posicion, "resultado") == 0: #operacion sin cerrar
                                                if getattr(posicion, "typo") == 1 and not bid >= getattr(posicion, "tp") and not bid <= getattr(posicion, "sl"):  #largo
                                                    try:
                                                        tp, sl = marketmastermanagement.profit_management(args, 1, ask, bid, data_5m_actual.iloc[-1], curr_spread)
                                                        if sl < bid < tp:
                                                            if tp < getattr(posicion, "tp"):
                                                                self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(tp = tp)
                                                                print(f"\nTp/sl modificados")
                                                            #if sl>getattr(posicion,"sl"):
                                                                #self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(sl=sl)
                                                                #print(f"\nTp/sl modificados")

                                                    except:
                                                        pass
                                                elif getattr(posicion, "typo") == -1 and not ask <= getattr(posicion, "tp") and not ask >= getattr(posicion, "sl"): #corto
                                                    try:
                                                        tp, sl = marketmastermanagement.profit_management(args, -1, ask, bid, data_5m_actual.iloc[-1], curr_spread)
                                                        if tp < ask < sl:
                                                            if tp > getattr(posicion, "tp"):
                                                                self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(tp = tp)
                                                                print(f"\nTp/sl modificados")
                                                            #if sl<getattr(posicion,"sl"):
                                                                #self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(sl=sl)
                                                                #print(f"\nTp/sl modificados")
                                                    except:
                                                        pass

                                    for indice_posicion, posicion in enumerate(self.posiciones):
                                        if getattr(posicion, "resultado") == 0: #operacion sin cerrar
                                            valor_parciales_profit = ajustar_intervalo(getattr(posicion, "anterioridad"), 0, self.longitud_liquidez, 50, 70)
                                            if getattr(posicion, "typo") == 1: #largo
                                                if bid >= getattr(posicion, "tp"): #tp
                                                    recalcular = True
                                                    if getattr(posicion, "parciales_retirados"):
                                                        self.n_tp += valor_parciales_profit
                                                        self.n_tp_diario += valor_parciales_profit
                                                    else:
                                                        self.n_tp += 1
                                                        self.n_tp_diario += 1
                                                    #self.dinero+=(((bid*self.unidad_lote*getattr(posicion,"lotaje")))-(self.comisiones*getattr(posicion,"lotaje")))
                                                    self.dinero += ((bid - getattr(posicion, "entrada")) * self.unidad_lote * getattr(posicion, "lotaje")) - (self.comisiones * getattr(posicion, "lotaje"))
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(resultado = 1)
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(fecha_resultado = fecha)
                                                    
                                                    if self.verbose:
                                                        print()
                                                        print(bid, " tp largo hora: ", fecha.hour, fecha.minute, fecha.second, self.dinero)

                                                elif bid <= getattr(posicion, "sl"): #sl
                                                    recalcular = True
                                                    if posicion.entrada - posicion.sl > 0:
                                                        if getattr(posicion, "parciales_retirados"):
                                                            self.n_sl += 1 - valor_parciales_profit
                                                            self.n_sl_diario += 1 - valor_parciales_profit
                                                        else:
                                                            self.n_sl += 1
                                                            self.n_sl_diario += 1
                                                    else:
                                                        if getattr(posicion, "parciales_retirados"):
                                                            self.n_tp += valor_parciales_profit
                                                            self.n_tp_diario += valor_parciales_profit
                                                        else:
                                                            self.n_tp += 1
                                                            self.n_tp_diario += 1
                                                    
                                                    #self.dinero+=(((bid*self.unidad_lote*getattr(posicion,"lotaje")))-(getattr(posicion,"lotaje")*self.comisiones))
                                                    self.dinero += ((bid - getattr(posicion, "entrada")) * self.unidad_lote * getattr(posicion, "lotaje")) - (self.comisiones * getattr(posicion, "lotaje"))
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(resultado = -1)
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(fecha_resultado = fecha)

                                                    if self.verbose:
                                                        print()
                                                        print(bid, " sl largo hora: ", fecha.hour, fecha.minute, fecha.second, self.dinero)

                                            else: #corto
                                                if ask <= getattr(posicion,"tp"): #tp
                                                    recalcular = True
                                                    if getattr(posicion, "parciales_retirados"):
                                                        self.n_tp += valor_parciales_profit
                                                        self.n_tp_diario += valor_parciales_profit
                                                    else:
                                                        self.n_tp += 1
                                                        self.n_tp_diario += 1
                                                    #self.dinero-=(((ask*self.unidad_lote*getattr(posicion,"lotaje")))+(getattr(posicion,"lotaje")*self.comisiones))
                                                    self.dinero += ((getattr(posicion, "entrada") - ask) * self.unidad_lote * getattr(posicion, "lotaje")) - (self.comisiones * getattr(posicion, "lotaje"))
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(resultado = 1)
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(fecha_resultado = fecha)

                                                    if self.verbose:
                                                        print()
                                                        print(ask, " tp corto hora: ", fecha.hour, fecha.minute, fecha.second, self.dinero)

                                                elif ask >= getattr(posicion, "sl"): #sl
                                                    #print(self.dinero)
                                                    recalcular = True
                                                    if posicion.entrada - posicion.sl > 0:
                                                        if getattr(posicion, "parciales_retirados"):
                                                            self.n_sl += 1 - valor_parciales_profit
                                                            self.n_sl_diario += 1 - valor_parciales_profit
                                                        else:
                                                            self.n_sl += 1
                                                            self.n_sl_diario += 1
                                                        
                                                    else:
                                                        if getattr(posicion, "parciales_retirados"):
                                                            self.n_tp += valor_parciales_profit
                                                            self.n_tp_diario += valor_parciales_profit
                                                        else:
                                                            self.n_tp += 1
                                                            self.n_tp_diario += 1

                                                    #self.dinero-=(((ask*self.unidad_lote*getattr(posicion,"lotaje")))+(getattr(posicion,"lotaje")*self.comisiones))
                                                    self.dinero += ((getattr(posicion, "entrada") - ask) * self.unidad_lote * getattr(posicion, "lotaje")) - (self.comisiones * getattr(posicion, "lotaje"))
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(resultado = -1)
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(fecha_resultado = fecha)

                                                    if self.verbose:
                                                        print()
                                                        print(ask, " sl corto hora: ", fecha.hour, fecha.minute, fecha.second, self.dinero)

                                    valor_dinero = 0
                                    margen_total = 0
                                    contador = 0
                                    posiciones_cortos = list(i for i in self.posiciones if getattr(i, "resultado") == 0 and getattr(i, "typo") == -1)
                                    posiciones_largos = list(i for i in self.posiciones if getattr(i, "resultado") == 0 and getattr(i, "typo") == 1)

                                    for posicion in self.posiciones:
                                        if getattr(posicion, "resultado") == 0:
                                            contador += 1
                                            if getattr(posicion, "typo") == 1: #largo
                                                valor_dinero += ((bid - getattr(posicion, "entrada")) * self.unidad_lote * getattr(posicion, "lotaje")) - (self.comisiones * getattr(posicion, "lotaje"))
                                                margen_total += mt.order_calc_margin(mt.ORDER_TYPE_BUY, self.name, getattr(posicion, "lotaje"), getattr(posicion, "entrada")) * 10
                                            else: #corto
                                                valor_dinero += ((getattr(posicion, "entrada") - ask) * self.unidad_lote * getattr(posicion, "lotaje")) - (self.comisiones * getattr(posicion, "lotaje"))
                                                margen_total += mt.order_calc_margin(mt.ORDER_TYPE_SELL, self.name, getattr(posicion, "lotaje"), getattr(posicion, "entrada")) * 10

                                    valor_dinero += self.dinero

                                    #if (((valor_dinero-self.dinero_inicial_diario)/(self.dinero_inicial_diario/self.multiplicador))>-self.maximo_perdida_diaria and ((valor_dinero-self.dinero_inicial_diario)/(self.dinero_inicial_diario/self.multiplicador))<self.maximo_beneficio_diario and len(list_days)>=self.longitud_liquidez) and fecha.hour<=20:
                                    if (((valor_dinero - self.dinero_inicial_diario) / (self.dinero_inicial_diario)) > -self.maximo_perdida_diaria 
                                        and ((valor_dinero - self.dinero_inicial_diario) / (self.dinero_inicial_diario)) < self.maximo_beneficio_diario
                                        and len(list_days) >= self.longitud_liquidez):
                                        operar = True
                                        for _i, _i_ in enumerate(self.posiciones):
                                            if getattr(_i_, "resultado") == 0:
                                                valor_trailing_tp = marketmaster.valor_pip(self.trigger_trailing_tp, bid)
                                                valor_trailing_sl = marketmaster.valor_pip(self.trigger_trailing_sl, bid)
                                                valor_parciales_profit = ajustar_intervalo(getattr(_i_, "anterioridad"), 0, self.longitud_liquidez, 50, 70)
                                                if getattr(_i_, "typo") == 1: # largo
                                                    if self.enable_break_even and getattr(_i_, "sl") < getattr(_i_, "entrada") and bid >= getattr(_i_, "entrada") + (getattr(_i_, "tp") - getattr(_i_, "entrada")) / 2:
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_, "entrada") + marketmaster.valor_pip(self.break_even_size, getattr(_i_, "entrada")))
                                                        if self.verbose:
                                                            print("\nbe largo")

                                                    #elif getattr(_i_,"sl")>=getattr(_i_,"entrada") and self.enable_dinamic_sl and anterior_bid<bid:
                                                    elif self.enable_dinamic_sl and anterior_bid < bid and bid > (_i_.sl + valor_trailing_sl) and not (_i_.tp - valor_trailing_tp) <= bid <= _i_.tp: # añadir condicion para sl trigger
                                                        if self.verbose:
                                                            print("dinamic sl largo")

                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_, "sl") + (bid - anterior_bid))

                                                    #elif self.enable_dinamic_tp and anterior_bid<bid and getattr(_i_,"sl")>=getattr(_i_,"entrada") and (_i_.tp-valor_trailing_tp)<=bid<=_i_.tp:
                                                    #elif self.enable_dinamic_tp and getattr(_i_,"sl")>=getattr(_i_,"entrada") and (_i_.tp-valor_trailing_tp)<=bid<=_i_.tp:
                                                    elif self.enable_dinamic_tp and (_i_.tp - valor_trailing_tp) <= bid <= _i_.tp:
                                                        if self.verbose:
                                                            print("dinamic tp largo")
                                                            
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(tp = getattr(_i_, "tp") + (bid - anterior_bid))
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_, "tp") - valor_trailing_tp)

                                                    elif getattr(_i_, "parciales_retirados") == False and bid >= getattr(_i_, "entrada") + (getattr(_i_, "tp") - getattr(_i_, "entrada")) * valor_parciales_profit: #el precio se encuentra en el medio
                                                        lotaje_a_liquidar = round(getattr(_i_, "lotaje") * valor_parciales_profit, 2)
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(lotaje = getattr(_i_, "lotaje") - lotaje_a_liquidar)
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(parciales_retirados = True)
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_, "entrada"))
                                                        self.n_tp += valor_parciales_profit
                                                        self.n_tp_diario += valor_parciales_profit
                                                        self.dinero += ((bid - getattr(_i_, "entrada")) * self.unidad_lote * lotaje_a_liquidar) - (self.comisiones * lotaje_a_liquidar)
                                                        print("\nRetiro de parciales largo")

                                                    """elif getattr(_i_,"parciales_retirados")==False and bid<=getattr(_i_,"entrada")-(getattr(_i_,"entrada")-getattr(_i_,"sl"))*(1-valor_parciales_profit): #el precio se encuentra en el medio
                                                        lotaje_a_liquidar=round(getattr(_i_,"lotaje")*(1-valor_parciales_profit),2)
                                                        #self.posiciones[_i]=self.posiciones[_i]._replace(lotaje=getattr(_i_,"lotaje")-lotaje_a_liquidar,parciales_retirados=True,sl=_i_.entrada-(_i_.entrada-_i_.sl)*(valor_parciales_profit))
                                                        self.posiciones[_i]=self.posiciones[_i]._replace(lotaje=getattr(_i_,"lotaje")-lotaje_a_liquidar,parciales_retirados=True)
                                                        self.n_sl+=(1-valor_parciales_profit)
                                                        self.n_sl_diario+=(1-valor_parciales_profit)
                                                        self.dinero+=((bid-getattr(_i_,"entrada"))*self.unidad_lote*lotaje_a_liquidar)-(self.comisiones*lotaje_a_liquidar)
                                                        print("\nRetiro de parciales largo")"""

                                                    """elif fecha>getattr(_i_,"fecha")+timedelta(hours=getattr(_i_,"duracion_estimada")): # se pasa de la duracion estimada
                                                        if bid>=getattr(_i_,"entrada")+marketmaster.valor_pip(self.break_even_size,getattr(_i_,"entrada")): #esta en la zona de tp por lo que se pone be
                                                            self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"entrada")+marketmaster.valor_pip(self.break_even_size,getattr(_i_,"entrada")))
                                                            if self.verbose:
                                                                print("\nbe largo por tiempo")
                                                        else: #esta en la zona de sl se cierra la operacion
                                                            self.dinero+=((bid-getattr(_i_,"entrada"))*self.unidad_lote*getattr(_i_,"lotaje"))-(self.comisiones*getattr(_i_,"lotaje"))
                                                            self.posiciones[_i]=self.posiciones[_i]._replace(resultado=-1)
                                                            self.posiciones[_i]=self.posiciones[_i]._replace(fecha_resultado=fecha)
                                                            if self.verbose:
                                                                print("\nsl largo por tiempo")"""
                                                    
                                                else: #corto
                                                    valor_trailing_tp = marketmaster.valor_pip(self.trigger_trailing_tp, ask)
                                                    valor_trailing_sl = marketmaster.valor_pip(self.trigger_trailing_sl, ask)
                                                    valor_parciales_profit = ajustar_intervalo(getattr(_i_, "anterioridad"), 0, self.longitud_liquidez, 50, 70)
                                                    if self.enable_break_even and getattr(_i_, "sl") > getattr(_i_, "entrada") and ask <= getattr(_i_, "entrada") - (getattr(_i_, "entrada") - getattr(_i_, "tp")) / 2:
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_, "entrada") - marketmaster.valor_pip(self.break_even_size, getattr(_i_, "entrada")))
                                                        if self.verbose:
                                                            print("\nbe corto")
                                                    
                                                    #elif getattr(_i_,"sl")<=getattr(_i_,"entrada") and self.enable_dinamic_sl and anterior_ask>ask:
                                                    elif self.enable_dinamic_sl and anterior_ask > ask and ask < (_i_.sl - valor_trailing_sl) and not (_i_.tp <= ask <= (_i_.tp + valor_trailing_tp)):
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_, "sl") + (ask - anterior_ask))
                                                        if self.verbose:
                                                            print("dinamic sl corto")

                                                    #elif self.enable_dinamic_tp and anterior_bid<bid and getattr(_i_,"sl")<=getattr(_i_,"entrada") and (_i_.tp<=ask<=(_i_.tp+valor_trailing_tp)):
                                                    #elif self.enable_dinamic_tp and getattr(_i_,"sl")<=getattr(_i_,"entrada") and (_i_.tp<=ask<=(_i_.tp+valor_trailing_tp)):
                                                    elif self.enable_dinamic_tp and (_i_.tp <= ask <= (_i_.tp + valor_trailing_tp)):
                                                        if self.verbose:
                                                            print("dinamic tp corto")
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(tp = getattr(_i_, "tp") + (ask - anterior_ask))
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_, "tp") + valor_trailing_tp)

                                                    elif getattr(_i_, "parciales_retirados") == False and ask <= getattr(_i_, "entrada") - (getattr(_i_, "entrada") - getattr(_i_, "tp")) * valor_parciales_profit: #va por la mitad
                                                        lotaje_a_liquidar = round(getattr(_i_, "lotaje") * valor_parciales_profit, 2)
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(lotaje = getattr(_i_, "lotaje") - lotaje_a_liquidar)
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(parciales_retirados = True)
                                                        self.posiciones[_i] = self.posiciones[_i]._replace(sl = getattr(_i_,"entrada"))
                                                        self.n_tp += valor_parciales_profit
                                                        self.n_tp_diario += valor_parciales_profit
                                                        self.dinero += ((getattr(_i_, "entrada") - ask) * self.unidad_lote * lotaje_a_liquidar) - (self.comisiones * lotaje_a_liquidar)
                                                        print("\nRetiro de parciales corto")

                                                    """elif getattr(_i_,"parciales_retirados")==False and ask>=getattr(_i_,"entrada")+(getattr(_i_,"sl")-getattr(_i_,"entrada"))*(1-valor_parciales_profit): #va por la mitad
                                                        lotaje_a_liquidar=round(getattr(_i_,"lotaje")*(1-valor_parciales_profit),2)
                                                        #self.posiciones[_i]=self.posiciones[_i]._replace(lotaje=getattr(_i_,"lotaje")-lotaje_a_liquidar,parciales_retirados=True,sl=_i_.entrada+(_i_.sl-_i_.entrada)*(valor_parciales_profit))
                                                        self.posiciones[_i]=self.posiciones[_i]._replace(lotaje=getattr(_i_,"lotaje")-lotaje_a_liquidar,parciales_retirados=True)
                                                        self.n_sl+=(1-valor_parciales_profit)
                                                        self.n_sl_diario+=(1-valor_parciales_profit)
                                                        self.dinero+=((getattr(_i_,"entrada")-ask)*self.unidad_lote*lotaje_a_liquidar)-(self.comisiones*lotaje_a_liquidar)
                                                        print("\nRetiro de parciales corto")"""

                                                    """elif fecha>getattr(_i_,"fecha")+timedelta(hours=getattr(_i_,"duracion_estimada")):
                                                        if ask<=getattr(_i_,"entrada")-marketmaster.valor_pip(self.break_even_size,getattr(_i_,"entrada")): # esta en la zona de tp se pone be
                                                            self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"entrada")-marketmaster.valor_pip(self.break_even_size,getattr(_i_,"entrada")))
                                                            if self.verbose:    
                                                                print("\nbe corto por tiempo")
                                                        else: #esta en la zona de sl se cierra la operacion
                                                            self.dinero+=((getattr(_i_,"entrada")-ask)*self.unidad_lote*getattr(_i_,"lotaje"))-(self.comisiones*getattr(_i_,"lotaje"))
                                                            self.posiciones[_i]=self.posiciones[_i]._replace(resultado=-1)
                                                            self.posiciones[_i]=self.posiciones[_i]._replace(fecha_resultado=fecha)
                                                            if self.verbose:
                                                                print("\nsl corto por tiempo")"""

                                    else:
                                        if len(posiciones_largos) > 0 or len(posiciones_cortos) > 0:
                                            self.dinero = valor_dinero
                                            
                                            if self.verbose:
                                                #print()
                                                #print(self.dinero)
                                                #print(valor_dinero)
                                                #print(self.dinero_inicial_diario)
                                                #print((((valor_dinero-self.dinero_inicial_diario)/(self.dinero_inicial_diario/self.multiplicador))))

                                                #print(len(list_days)>=self.longitud_liquidez)

                                                #print((valor_dinero-self.dinero_inicial_diario)/(self.dinero_inicial_diario/self.multiplicador))
                                                print(f"\n\nLímite perdida o horario alcanzado, liquidando posiciones {valor_dinero}\n") #es beneficio o perdida, pero el de beneficio es muy alto
                                                #time.sleep(10)
                                                #break

                                            self.n_p_diario = contador
                                            self.n_paradas += self.n_p_diario

                                            for pos_entrada,entrada in enumerate(self.posiciones):
                                                if getattr(entrada, "resultado") == 0:
                                                    if getattr(entrada, "typo") == 1 and getattr(entrada, "entrada") <= bid <= getattr(entrada, "tp"):
                                                        self.n_tp += 1
                                                    elif getattr(entrada, "typo") == 1 and getattr(entrada, "sl") <= bid <= getattr(entrada, "entrada"):
                                                        self.n_sl += 1
                                                    elif getattr(entrada, "typo") == -1 and getattr(entrada, "tp") <= ask <= getattr(entrada, "entrada"):
                                                        self.n_tp += 1
                                                    elif getattr(entrada, "typo") == -1 and getattr(entrada, "entrada") <= ask <= getattr(entrada, "sl"):
                                                        self.n_sl += 1

                                                    self.posiciones[pos_entrada] = self.posiciones[pos_entrada]._replace(resultado = ask)
                                                    self.posiciones[pos_entrada] = self.posiciones[pos_entrada]._replace(fecha_resultado = fecha)

                                            operar = False
                                            #saltar=True
                                    #print(data_5m_actual['limite_inferior'].iloc[-1],limite_spread)
                                    if not (curr_spread <= limite_spread and self.zona_horaria_operable[0] <= fecha.hour <= self.zona_horaria_operable[1] and not saltar): #163.62
                                    #if not (curr_spread<=limite_superior_izq and self.zona_horaria_operable[0]<=fecha.hour<=self.zona_horaria_operable[1] and not saltar): #163.62
                                        operar = False

                                    if ((pos_dia % self.ticks_refresco_liquidez == 0 and operar) or len(puntos_liquidez) == 0 or recalcular):
                                    #if (pos_dia%self.ticks_refresco_liquidez==0 or len(puntos_liquidez)==0 or recalcular):
                                        recalcular = False

                                        liquidez_final, salidas_final, puntos_liquidez = recalcular_puntos(list_liquidity_ask, list_liquidity_bid, resultado_liquidez_bid, resultado_liquidez_ask, 
                                                                                                           fecha_busqueda, self.limite_potencia, first_recalculation,executor)

                                        if not first_recalculation:
                                            first_recalculation = True

                                        #if len(liquidez_final)!=0:
                                            #if self.verbose:
                                                #print("\nRecalculando liquidez")
                                                #print(liquidez_final)

                                if not saltar and operar and len(puntos_liquidez) != 0 and (len(posiciones_cortos) + len(posiciones_largos)) <= self.maximo_operaciones_consecutivas and len(self.posiciones) <= self.maximo_operaciones_diarias and (bid in liquidez_final or ask in liquidez_final):
                                #if not saltar and operar and len(puntos_liquidez)!=0 and (len(posiciones_cortos)+len(posiciones_largos))<=self.maximo_operaciones_consecutivas and len(self.posiciones)<=self.maximo_operaciones_diarias and ((bid in liquidez_final or ask in liquidez_final) or 
                                    #(not precios & set(resultado_liquidez_bid[fecha_busqueda]) and not precios & set(resultado_liquidez_ask[fecha_busqueda]))): 

                                    #comprobar si operaciones contrarias
                                    #if len(posiciones_cortos+posiciones_largos)!=0:
                                        #if any(x.typo==1 and x.resultado==0 for x in posiciones_cortos+posiciones_largos) and any(x.typo==-1 and x.resultado==0 for x in posiciones_cortos+posiciones_largos):
                                            #raise Error('Posiciones contrarias')

                                    #comprobar si noticia
                                    if self.bloquear_noticias and type(calendario) != type(None) and len(calendario) > 0:
                                        coincidencias = list(i for i in calendario["time"] if (i - timedelta(minutes = 15)).time() <= hora <= (i + timedelta(minutes = 5)).time())
                                        #no operar 15 minutos antes ni 15 minutos despues

                                        if len(coincidencias) > 0:
                                            for coincidencia in coincidencias:
                                                calendario = calendario[calendario.time != coincidencia]

                                            if self.verbose and not prints_noticia[0]:
                                                print("\nRango noticia, no operar")
                                                prints_noticia[0] = 1

                                            operar = False

                                        else:
                                            if self.verbose and not prints_noticia[1] and prints_noticia[0]:
                                                prints_noticia[1] = 1
                                                print("\nFuera noticia operar")

                                            operar = True

                                    if self.recavar_datos:
                                        action, sl, tp, lotaje, accion = 0, 0, 0, 0, 0
                                    else:
                                        if not (type(anterior_data_5m) != None and len(data_5m_actual) == anterior_data_5m):
                                            """
                                            args={"dinero":[self.dinero_inicial,valor_dinero,self.dinero_inicial_diario], "puntos_liquidez":liquidez_final,
                                            "puntos_salida":salidas_final,"posiciones_cortos":posiciones_cortos,"posiciones_largos":posiciones_largos,
                                            "correlation_dict":self.correlation_dict,"recalcular":False,
                                            "pos_dia":pos_dia,"all_data":all_data,"all_data_5m":all_data_5m,"actual_time":fecha}
                                            """
                                            
                                            args["dinero"] = [self.dinero_inicial, valor_dinero, self.dinero_inicial_diario]
                                            args["puntos_liquidez"] = liquidez_final
                                            args["puntos_salida"] = salidas_final
                                            args["posiciones_cortos"] = posiciones_cortos
                                            args["posiciones_largos"] = posiciones_largos
                                            args["pos_dia"] = pos_dia
                                            args["actual_time"] = fecha
                                            args["ask"] = ask
                                            args["bid"] = bid

                                            try:
                                                #es mejor poner las operaciones de menor temporalidad en la lista primero
                                                args['temporality'] = "all_data_5m"
                                                action, sl, tp, lotaje, accion, estrategias = marketmaster.run(args)

                                                if action != 0:
                                                    if ask in operaciones_diarias_puntos or bid in operaciones_diarias_puntos:
                                                        lotaje = lotaje / 2

                                                    if action == 1:
                                                        margin_necesary = mt.order_calc_margin(mt.ORDER_TYPE_BUY, self.name, lotaje, ask) * 10

                                                    elif action == -1:
                                                        margin_necesary = mt.order_calc_margin(mt.ORDER_TYPE_SELL, self.name, lotaje, bid) * 10

                                                    if puede_abrir_orden(margin_necesary, margen_total, self.multiplicador_margen_total, self.dinero_inicial_diario, valor_dinero):
                                                        nuevas_entradas.append((action, sl, tp, lotaje, "all_data_5m"))
                                                        anterior_data_5m = len(data_5m_actual)
                                                    else:
                                                        action = 0

                                            except Exception as e:
                                                print("Error en marketmaster")
                                                print(f"Error: {e}")
                                                traceback.print_exc()
                                                action, sl, tp, lotaje, accion = 0, 0, 0, 0, 0

                                    if self.re_evaluar and accion != 0 and (len(posiciones_cortos) != 0 or len(posiciones_largos) != 0):
                                        for indice_posicion,posicion_re_evaluar in enumerate(self.posiciones):
                                            if posicion_re_evaluar.typo == -1 and accion == 1 and posicion_re_evaluar.resultado == 0:
                                                #recalcular=True
                                                if posicion_re_evaluar.entrada - ask > 0:
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(tp = posicion_re_evaluar.entrada)
                                                else:
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(sl = posicion_re_evaluar.entrada)
                                                
                                                self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(fecha_resultado = fecha)

                                                if self.verbose:
                                                    print('posición corta cerrada por re evaluación')

                                            elif posicion_re_evaluar.typo == 1 and accion == -1 and posicion_re_evaluar.resultado == 0:
                                                #recalcular=True
                                                if posicion_re_evaluar.entrada - ask > 0:
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(sl = posicion_re_evaluar.entrada)
                                                else:
                                                    self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(tp = posicion_re_evaluar.entrada)
                                                
                                                self.posiciones[indice_posicion] = self.posiciones[indice_posicion]._replace(fecha_resultado = fecha)

                                                if self.verbose:
                                                    print('posición larga cerrada por re evaluación')

                                    for action, sl, tp, lotaje, data_temporality in nuevas_entradas:
                                        #if action==1 and lotaje>=(self.lotaje_minimo*valor_dinero)/self.dinero_inicial:
                                        if action == 1 and lotaje >= self.lotaje_minimo:
                                            operaciones_diarias_puntos.append(ask)
                                            recalcular = True
                                            self.n_largos += 1
                                            self.estadisticas_operaciones[fecha.weekday()] += 1

                                            tp = float('%.5f' % tp)
                                            sl = float('%.5f' % sl) 
                                        
                                            limite_tiempo = calcular_tiempo_limite(tp, ask, data_5m_actual['atr'].iloc[-1]) / self.divisor_tiempo_limite
                                            #self.posiciones.append(Posicion(self.name,fecha,action,ask,lotaje,tp,sl,tp,sl,0,0,limite_tiempo))
                                            try:
                                                self.posiciones.append(Posicion(self.name, fecha, action, ask, lotaje, tp, sl, tp, sl, 0, 0, data_temporality, False, liquidez_final[ask], 1))
                                            except:
                                                try:
                                                    self.posiciones.append(Posicion(self.name, fecha, action, ask, lotaje, tp, sl, tp, sl, 0, 0, data_temporality, False, liquidez_final[bid], 1))
                                                except:
                                                    self.posiciones.append(Posicion(self.name, fecha, action, ask, lotaje, tp, sl, tp, sl, 0, 0, data_temporality, False, 0, 1))
                                            file.write(f"\ncrear largo entrada:{round(ask, 4)} sl:{round(sl, 4)} tp:{round(tp, 4)} lotaje: {lotaje} a las {fecha} | Señales: {estrategias}")

                                            if self.verbose:
                                                print(f"\ncrear largo entrada:{ask} sl:{sl} tp:{tp} lotaje: {lotaje} a las {fecha} | Señales: {estrategias}")
                            
                                        #if action==-1 and lotaje>=(self.lotaje_minimo*valor_dinero)/self.dinero_inicial:
                                        if action == -1 and lotaje >= self.lotaje_minimo:
                                            operaciones_diarias_puntos.append(bid)
                                            recalcular = True
                                            self.n_cortos += 1
                                            self.estadisticas_operaciones[fecha.weekday()] += 1

                                            tp = float('%.5f' % tp)
                                            sl = float('%.5f' % sl)
                                        
                                            limite_tiempo = calcular_tiempo_limite(tp, bid, data_5m_actual['atr'].iloc[-1]) / self.divisor_tiempo_limite
                                            #self.posiciones.append(Posicion(self.name,fecha,action,bid,lotaje,tp,sl,tp,sl,0,0,limite_tiempo))
                                            try:
                                                self.posiciones.append(Posicion(self.name, fecha, action, bid, lotaje, tp, sl, tp, sl, 0, 0, data_temporality, False, liquidez_final[bid], 1))
                                            except:
                                                try:
                                                    self.posiciones.append(Posicion(self.name, fecha, action, bid, lotaje, tp, sl, tp, sl, 0, 0, data_temporality, False, liquidez_final[ask], 1))
                                                except:
                                                    self.posiciones.append(Posicion(self.name, fecha, action, bid, lotaje, tp, sl, tp, sl, 0, 0, data_temporality, False, 0, 1))

                                            file.write(f"\ncrear corto entrada:{round(bid, 4)} sl:{round(sl, 4)} tp:{round(tp, 4)} lotaje: {lotaje} a las {fecha} | Señales: {estrategias}")

                                            if self.verbose:
                                                print(f"\ncrear corto entrada:{bid} sl:{sl} tp:{tp} lotaje: {lotaje} a las {fecha} | Señales: {estrategias}")

                                liquidity(bid, resultado_liquidez_bid[fecha_busqueda])
                                liquidity(ask, resultado_liquidez_ask[fecha_busqueda])

                                """if first_recalculation:
                                    # Recorremos las listas de atrás hacia adelante
                                    for pos_liquidity in range(len(list_liquidity_ask)):
                                        liquidity(bid,list_liquidity_ask[pos_liquidity])
                                        liquidity(ask,list_liquidity_bid[pos_liquidity])"""

                                if (not saltar and operar and len(puntos_liquidez) != 0) and (bid in liquidez_final or ask in liquidez_final):
                                #if (not saltar and operar and len(puntos_liquidez)!=0) and ((bid in liquidez_final or ask in liquidez_final) or 
                                    #(not precios & set(resultado_liquidez_bid[fecha_busqueda]) and not precios & set(resultado_liquidez_ask[fecha_busqueda]))): 

                                    recalcular = True
                                    self.historial_puntos_liquidez.append((ask, bid, liquidez_final, self.atr_multipliers, curr_spread, data_5m_actual.iloc[-1]['atr'], data_5m_actual.iloc[-1]['atr_ma'], data[pos_dia + 1:],pos_dia))

                                if bid != anterior_bid:
                                    anterior_bid = bid

                                if ask != anterior_ask:
                                    anterior_ask = ask

                        """dinero_a_retirar=self.comision_por_retiro*self.dinero*self.cantidad_retiro
                        if (n-self.longitud_liquidez+1)%self.dias_retiro==0 and (self.dinero-dinero_a_retirar)>self.dinero_ultimo_retiro:
                            print("Retiro de ",dinero_a_retirar)
                            self.retiros.append(dinero_a_retirar)
                            self.dinero-=dinero_a_retirar
                            self.dinero_ultimo_retiro=self.dinero"""
                        
                        if not saltar and (dia.weekday() == 4 or ((dia + timedelta(days = 1)).date() in holidays.US())) and len(self.posiciones) != 0:
                            print("\nLiquidando las posiciones antes del fin de semana o festivo\n")
                            self.dinero = valor_dinero
                            for pos_entrada, entrada in enumerate(self.posiciones):
                                if getattr(entrada, "resultado") == 0:
                                    if getattr(entrada, "typo") == 1 and getattr(entrada, "entrada") <= bid <= getattr(entrada, "tp"):
                                        self.n_tp += 1
                                    elif getattr(entrada,"typo") == 1 and getattr(entrada, "sl") <= bid <= getattr(entrada, "entrada"):
                                        self.n_sl += 1
                                    elif getattr(entrada, "typo") == -1 and getattr(entrada, "tp") <= ask <= getattr(entrada, "entrada"):
                                        self.n_tp += 1
                                    elif getattr(entrada, "typo") == -1 and getattr(entrada, "entrada") <= ask <= getattr(entrada, "sl"):
                                        self.n_sl += 1

                                    self.posiciones[pos_entrada] = self.posiciones[pos_entrada]._replace(resultado = ask)
                                    self.posiciones[pos_entrada] = self.posiciones[pos_entrada]._replace(fecha_resultado = fecha)

                        dinero_a_retirar = 0
                        if not saltar and (n - self.longitud_liquidez + 1) % self.dias_retiro == 0:
                            #dinero_a_retirar = retiro(self.dinero,self.dinero_ultimo_retiro,self.porcentaje_retiro,self.porcentaje_umbral_ganancias, self.comision_retiro) #comparado con retiro respecto al ultimo saldo tras retiro
                            dinero_a_retirar = retiro(self.dinero, self.dinero_inicial, self.porcentaje_retiro, self.porcentaje_umbral_ganancias, self.comision_retiro) #comparado con saldo inicial fijo 
                            if dinero_a_retirar != 0:
                                print("Retiro de ", dinero_a_retirar)
                                self.retiros.append(dinero_a_retirar)
                                self.dinero -= dinero_a_retirar                       #LOS RETIROS SE LOS RESTA COMO BENEFICIO NEGATIVO...
                                self.dinero_ultimo_retiro = self.dinero
                            else:
                                print("No se cumplen condiciones para retirar")

                        rentabilidad_diaria = (self.dinero - self.dinero_inicial_diario) + dinero_a_retirar
                        self.historial_dinero.append(self.dinero)
                        self.historial_dinero_con_beneficio.append(self.dinero + np.sum(self.retiros))

                        if len(self.posiciones) != 0:
                            mostrar_posiciones(self.posiciones)

                        print(f"\nEn el día {dia.date()} se han tenido {self.n_tp_diario} tp y {self.n_sl_diario} sl y {self.n_p_diario} paradas | Dinero: {round(self.historial_dinero[-1], 2)} | Beneficio: {round(rentabilidad_diaria, 2)} | Número dia: {self.n_dias - self.longitud_liquidez}")

                        try:
                            print(f"RR actual diario: {round(self.positivos_negativos[0] / self.positivos_negativos[1], 2)} | RR actual operaciones: {self.n_tp / self.n_sl}\n")
                        except:
                            pass

                        self.dinero_inicial_diario = self.historial_dinero[-1]
                        self.dinero = self.dinero_inicial_diario

                        self.n_tp_diario = 0
                        self.n_sl_diario = 0
                        self.n_p_diario = 0

                        if len(self.historial_dinero_con_beneficio) > 1:
                            if self.historial_dinero_con_beneficio[-1] > self.historial_dinero_con_beneficio[-2]:
                                self.positivos_negativos[0] += 1
                            elif self.historial_dinero_con_beneficio[-1] < self.historial_dinero_con_beneficio[-2]:
                                self.positivos_negativos[1] += 1
                        else:
                            if self.historial_dinero_con_beneficio[-1] > self.dinero_inicial:
                                self.positivos_negativos[0] += 1
                            elif self.historial_dinero_con_beneficio[-1] < self.dinero_inicial:
                                self.positivos_negativos[1] += 1

                        list_days.append(1)
                        list_days_5m.append(1)

                        list_liquidity_bid.append(resultado_liquidez_bid[fecha_busqueda])
                        list_liquidity_ask.append(resultado_liquidez_ask[fecha_busqueda])

                        resultado_liquidez_ask.clear()
                        resultado_liquidez_bid.clear()

                        if len(list_liquidity_bid) > self.longitud_liquidez:
                            del list_liquidity_bid[0]
                            del list_liquidity_ask[0]
                            del list_days[0]

                        if len(list_days_5m) > 1:
                            del list_days_5m[0]

                        if not (len(list_days) == len(list_liquidity_ask) == len(list_liquidity_bid)):
                            raise Error("")

                        if len(self.historial_puntos_liquidez) != 0 and self.recavar_datos:
                            pool = multiprocessing.Pool(5)
                            resultados = pool.map(agregar_puntos_liquidez, self.historial_puntos_liquidez)
                            for i in resultados:
                                mejor_accion, entrada, pos_dia, tp_largo, tp_corto = i
                                if entrada != 0:
                                    if str(dia.date()) not in self.historial_total_puntos_liquidez:
                                        self.historial_total_puntos_liquidez[str(dia.date())] = [(mejor_accion, entrada, pos_dia, tp_largo, tp_corto, data_5m_actual.iloc[-1].tolist()[1:])]
                                    else:
                                        self.historial_total_puntos_liquidez[str(dia.date())].append((mejor_accion, entrada, pos_dia, tp_largo, tp_corto, data_5m_actual.iloc[-1].tolist()[1:]))

                        lista_borrar = []
                        for posicion in self.posiciones:
                            if posicion.resultado != 0:
                                lista_borrar.append(posicion)

                        self._registrar_operaciones_cerradas(lista_borrar)

                        for borrar_posicion in lista_borrar:
                            self.posiciones.remove(borrar_posicion)

                    else:
                        pass

                try:
                    anterior_mayor_spread = data['spread'].mean()
                except:
                    pass

            if self.recavar_datos:
                dataframe = raw_data_to_df(self.historial_total_puntos_liquidez)
                dataframe.to_csv(f'data/datasets/dataframe_{self.name}_{self.fecha_inicio.date()}_{self.fecha_final.date()}.csv')

            mt.shutdown()

            rentabilidad_diaria = (self.dinero - self.dinero_inicial_diario)

            self.historial_dinero.append(self.dinero)

            self.dinero = self.historial_dinero[-1]

            if self.n_dias != 0:
                print()
                print("-" * 100)
                print()

                if  ((self.dinero - self.dinero_inicial) / self.dinero_inicial) <= - self.perdida_maxima_cuenta:
                    print("Cuenta quemada :(")
                    print()

                print(f"Tiempo de ejecucion: {round((time.time()-start) / 60, 2)} minutos")
                print(f"{self.n_dias} días analizados | Dias positivos: {self.positivos_negativos[0]} | Dias negativos: {self.positivos_negativos[1]} | Dias neutros: {self.n_dias-sum(self.positivos_negativos)}")
                print(f"Maximo dinero: {round(max(self.historial_dinero),2)} | Minimo dinero: {round(min(self.historial_dinero),2)}")
                #print(f"Dinero inicial: {self.dinero_inicial}€ | dinero final: {round(self.dinero,2)}€ | Beneficio: {round(self.dinero-self.dinero_inicial,2)} ({round(((self.dinero-self.dinero_inicial)/self.dinero_inicial)*100,2)}%)")
                print(f"Dinero inicial: {self.dinero_inicial}€ | dinero final: {round(self.dinero, 2)}€ | Beneficio: {round((self.dinero - self.dinero_inicial) + np.sum(self.retiros), 2)} ({round(((self.dinero + np.sum((self.retiros)) - self.dinero_inicial) / self.dinero_inicial) * 100, 2)}%)")
                print(f"Numero operaciones: {self.n_cortos + self.n_largos} | Media operaciones por día: {int(round((self.n_cortos+self.n_largos) / self.n_dias, 0))} | numero cortos: {self.n_cortos} | numero largos: {self.n_largos}")

                print(f"Total profits: {self.n_tp} | Total stops: {self.n_sl} | Total paradas: {self.n_paradas}")
                print("Retiros: ",self.retiros)
                print(f"{round(sum(self.retiros),2)}€ ganado en {len(self.retiros)} retiros")

                keys=["Lunes","Martes","Miercoles","Jueves","Viernes","Sábado","Domingo"]
                values = list(self.estadisticas_operaciones.values())
                self.estadisticas_operaciones = dict(zip(keys, values))
                print(f"Relación dia semana y numero operaciones: {self.estadisticas_operaciones}")

                mejor,dinero_inicio_perdida = maximo_perdida(self.historial_dinero_con_beneficio)
                try:
                    print('Máximo pérdida consecutiva: ', round(100 * (mejor / dinero_inicio_perdida), 2), '%')
                    print(f"RR (operaciones): {self.n_tp / self.n_sl} | Winrate: {(self.n_tp / self.n_sl) / ((self.n_tp / self.n_sl) + 1)}%")
                except:
                    print(f"No loss {self.n_tp}, {self.n_sl}")

                # Sumar los valores positivos
                suma_positivos = sum(x for x in self.historial_dinero_con_beneficio if x > 0)

                # Sumar los valores negativos y obtener su valor absoluto
                suma_negativos_abs = abs(sum(x for x in self.historial_dinero_con_beneficio if x < 0))

                # Realizar la división
                resultado = suma_positivos / suma_negativos_abs if suma_negativos_abs != 0 else 0
                print("RR diario teórico: ", resultado)

                print()

                if self.sounds:
                    playsound('data\\ok.mp3')

                self._guardar_csv_operaciones()

                self._mostrar_grafico_velas_operaciones()

                if len(self.historial_dinero) > 1:
                    if self.dinero>self.dinero_inicial:
                        color = "green"
                    else:
                        color = "red"
                    
                    self.historial_dinero = self.historial_dinero[self.longitud_liquidez:-1]
                    poly_fn = np.poly1d(np.polyfit(list(range(1, len(self.historial_dinero_con_beneficio) + 1)), self.historial_dinero_con_beneficio, 1))
                    coeficiente_reescalado = self.dinero_inicial - poly_fn(0)
                    plt.plot(self.historial_dinero, "o", color = color)
                    plt.plot(list(range(0, len(self.historial_dinero))), self.historial_dinero, coeficiente_reescalado + poly_fn(list(range(0, len(self.historial_dinero)))), "--k", color = color)
                    try:
                        red_patch = mpatches.Patch(color=color, label=f'Money Plot (Inclinación: {round(np.polyfit(list(range(1,len(self.historial_dinero)+1)), self.historial_dinero, 1)[0], 2)}) Rendimiento: {round(((self.dinero-self.dinero_inicial)/self.dinero_inicial)*100,2)}% Ratio: {self.positivos_negativos[0]} positivos - {self.positivos_negativos[1]} negativos ({round(self.positivos_negativos[0]/self.positivos_negativos[1],2)}) RR (operaciones): {self.n_tp/self.n_sl}')
                        plt.legend(handles = [red_patch], loc = 'upper center', shadow = True, fancybox = True)
                    except:
                        pass

                    plt.savefig('data/resultados/resultados_backtest.png')
                    plt.show()

                    plt.plot(self.historial_dinero_con_beneficio, "o", color = color)
                    plt.plot(list(range(0, len(self.historial_dinero_con_beneficio))), self.historial_dinero_con_beneficio, coeficiente_reescalado + poly_fn(list(range(0, len(self.historial_dinero_con_beneficio)))), "--k", color = color)
                    try:
                        red_patch = mpatches.Patch(color = color, label = f'Beneficio Plot (Inclinación: {round(np.polyfit(list(range(1, len(self.historial_dinero_con_beneficio) + 1)), self.historial_dinero_con_beneficio, 1)[0], 2)}) Rendimiento: {round((((self.dinero-self.dinero_inicial)+np.sum(self.retiros))/self.dinero_inicial)*100,2)}% Ratio: {self.positivos_negativos[0]} positivos - {self.positivos_negativos[1]} negativos ({round(self.positivos_negativos[0]/self.positivos_negativos[1],2)}) RR (operaciones): {self.n_tp/self.n_sl}')
                        plt.legend(handles = [red_patch], loc = 'upper center', shadow = True, fancybox = True)
                    except:
                        pass

                    plt.savefig('data/resultados/resultados_backtest_beneficio.png')
                    plt.show()

                    plt.plot(np.diff(np.array(self.historial_dinero)))
                    plt.show()

        except BaseException as e:
            if self.sounds:
                playsound('data\\error.mp3')

            print(f"\nError: [{e!r}]\n")

            raise e

        finally:
            file.close()
            try:
                mt.shutdown()
                torch.cuda.empty_cache()
                pool.close()
            except:
                pass

            if self.save_file:
                sys.stdout = orig_stdout
                f.close()
    
def main() -> None:
    backtest = Backtesting(
        # Configuracion general
        dinero_inicial = 100000, name = "EURUSD", moneda_cuenta = "USD", zona_horaria_operable = [[4, 22], [5, 23]],
        
        # Configuracion gestión de riesgo
        RR = 1.5, tamanyo_break_even = 16, break_even_size = 1, enable_break_even = False,
        enable_dinamic_sl = False, enable_dinamic_tp = True,
        maximo_perdida_diaria = 1.2/100, riesgo_por_operacion = {'corto':0.46/100, 'largo':0.46/100, 'invierno':0.68/100, 'verano':0.68/100},
        trigger_trailing_tp = 0.5, trigger_trailing_sl = 6, 
        maximo_beneficio_diario = 80 / 100, lotaje_minimo = 0.02,
        maximo_operaciones_consecutivas = 100, multiplicador_margen_total = 1.5,
        multiplicador_tp_sl = [1, 1],
        min_spread_trigger = [0.00006, 0.00005], max_spread_trigger = [0.00009, 0.00008],
        maximo_operaciones_diarias = 40,
        atr_window = 13, atr_ma = 5,
        
        # Configuracion de la liquidez
        longitud_liquidez = 7, ticks_refresco_liquidez = 100000, limite_potencia = 1.0,
        atr_multipliers = {"all_data_1m":(2, 3), "all_data_2m":(2, 3), "all_data_5m":(2, 3), "all_data_1h":(2.5, 3.75), "all_data_15m":(2, 3)},
        re_evaluar = False,
        
        # Configuracion noticias
        bloquear_noticias = False,
        
        # Configuracion estrategias y confirmaciones
        lista_pesos_confirmaciones = [0.7, 0.7, 0.6, 0.9, 0.8, 0.8, 0.8],
        lista_pesos_estrategias = [1, 1, 1, 1, 1, 1],
        
        # Configuracion de la correlacion
        corr_limit = 0.5,
        
        # Configuracion retiros
        dias_retiro = 1, porcentaje_retiro = 50, divisor_tiempo_limite = 3, porcentaje_umbral_ganancias = 13 / 100,
        
        # Ajustes backtesting
        fecha_inicio = datetime(2020, 1, 1), 
        fecha_final = datetime(2025, 11, 5), 
        verbose = True, sounds = False, grafico = False, recavar_datos = False, save_file = False
    )
    
    backtest.run()
    
    actualizar_csv()
    
if __name__ == "__main__":
    main()
