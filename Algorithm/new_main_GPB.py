import MetaTrader5 as mt
from estrategia.gestion_dinamica import MarketMasterManagement
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime,timedelta,date
import pandas as pd
import numpy as np
import holidays
from dotenv import dotenv_values
from playsound import playsound
from utils import *
from estrategia import MarketMaster, liquidity, simplify_liquidity, all_liquidity_2, calcular_pips
from pre_implemented_strategies.book_maths import Maths
from typing import NamedTuple
import investpy as ivp
from tqdm import tqdm
import telebot
import time
import warnings
import sys
import subprocess
import json
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import itertools
from concurrent.futures import ThreadPoolExecutor
import polars as pl

colorama_init()
warnings.filterwarnings("ignore")

#https://www.mql5.com/en/docs/python_metatrader5

def iniciar_mt5():
    account = dotenv_values("account.env")
    if not mt.initialize():
        raise RuntimeError(f"MT5 init failed, error: {mt.last_error()}")
    mt.login(account["login"], account["password"], account["server"])
    return account

def get_tick_data(args):
    name_par, hoy, siguiente = args
    """if hoy + timedelta(hours=(datetime.now()-timedelta(minutes=5)).hour) < siguiente:
        hoy = hoy + timedelta(hours=(datetime.now()-timedelta(minutes=5)).hour)

    data = pd.DataFrame(mt.copy_ticks_range(name_par, hoy+timedelta(hours=(datetime.now()-timedelta(minutes=1)).hour), siguiente, mt.COPY_TICKS_ALL))
    while len(data) < 10000:
        hoy = hoy - timedelta(minutes=5)
        data = pd.DataFrame(mt.copy_ticks_range(name_par, hoy,siguiente, mt.COPY_TICKS_ALL))"""

    """data['ask_dif'] = data['ask'].diff()
    data['bid_dif'] = data['bid'].diff()

    data['power'] = np.select(
        [data["bid_dif"] > data["ask_dif"], data["bid_dif"] < data["ask_dif"]],
        ["Compradores", "Vendedores"],
        default = "Equilibrado"
    )"""

    data = pd.DataFrame([mt.symbol_info_tick(name_par)._asdict()])

    data['spread'] = data['ask'] - data['bid']
    data['original_time'] = data['time']

    #data = data.drop(columns=["last", "volume", "time_msc", "volume_real"], axis=1)
    data["time"] = [datetime.fromtimestamp(item) for item in data["time"]]

    return name_par, data

def get_data(args):
    nombre, hoy, siguiente, market_master_maths, temporality = args

    if temporality == mt.TIMEFRAME_M1:
        fecha_inicio = hoy
    if temporality == mt.TIMEFRAME_M2:
        fecha_inicio = hoy
    if temporality == mt.TIMEFRAME_M5: 
        fecha_inicio = hoy - timedelta(days=1)
    if temporality == mt.TIMEFRAME_M15:
        fecha_inicio = hoy - timedelta(days=2)
    if temporality == mt.TIMEFRAME_H1:
        fecha_inicio = hoy - timedelta(days=8)

    data_5m = pd.DataFrame(mt.copy_rates_range(nombre, temporality, fecha_inicio, siguiente))
    #data_5m = pd.DataFrame(mt.copy_rates_range(nombre, temporality, hoy, siguiente))

    #data_5m=data_5m.drop(["real_volume","spread"],axis=1)
    data_5m["mean_price"] = (data_5m["high"] + data_5m["low"] + data_5m["close"]) / 3
    data_5m["rmv"] = data_5m["mean_price"] * data_5m['tick_volume']
    data_5m["time"] = [datetime.fromtimestamp(item) for item in data_5m["time"]]
    data_5m["diff"] = data_5m["mean_price"].diff().fillna(0)

    try:
        data_5m = market_master_maths.df(data_5m)
    except:
        print(f"Error al calcular indicadores para {nombre}{temporality}")
        pass

    return nombre, data_5m

def recalcular_liquidez(list_liquidity_ask, list_liquidity_bid, resultado_liquidez_bid, resultado_liquidez_ask, limite_potencia):
    puntos_liquidez = {}
    liquidez_final = {}

    # Recorremos las listas de atrás hacia adelante, optimizando el acceso
    for pos_liquidity in range(len(list_liquidity_ask)):
        # Usamos `reversed` para acceder a las últimas posiciones sin crear sublistas
        i_bid = list_liquidity_bid[-(pos_liquidity + 1):]
        i_ask = list_liquidity_ask[-(pos_liquidity + 1):]

        # Evitamos el uso de `get` repetidamente, accedemos directamente a los valores
        all_liquidity_bid = all_liquidity_2(resultado_liquidez_bid, i_bid)
        all_liquidity_ask = all_liquidity_2(resultado_liquidez_ask, i_ask)

        # Simplificamos la liquidez y agregamos los resultados a `puntos_liquidez`
        puntos = simplify_liquidity(all_liquidity_bid, all_liquidity_ask, trigger = limite_potencia)[0]
        puntos_liquidez.update(puntos)

        liquidez_final.update(zip(puntos, np.full(len(puntos), pos_liquidity)))

    return liquidez_final, {}, puntos_liquidez

def leer_saldo(archivo_saldo):
    try:
        with open(archivo_saldo, "r") as f:
            datos = json.load(f)
            return datos.get("fecha"), datos.get("saldo")
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None  # Si el archivo no existe o está vacío

# Función para guardar saldo con la fecha actual
def guardar_saldo(archivo_saldo,saldo):
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    datos = {"fecha": fecha_actual, "saldo": saldo}
    with open(archivo_saldo, "w") as f:
        json.dump(datos, f, indent=4)

def run(dinero_inicial, zona_horaria_operable, name, porcentaje_retiro, porcentaje_umbral_ganancias, tamanyo_break_even, min_spread_trigger, max_spread_trigger, 
        break_even_size, trigger_trailing_tp, trigger_trailing_sl, enable_break_even, enable_dinamic_sl, enable_dinamic_tp, maximo_perdida_diaria, 
        longitud_liquidez, ticks_refresco_liquidez, limite_potencia, bloquear_noticias, verbose, sounds, save_file, ema_length, ema_length2, 
        sma_length_2, sma_length_3, length_macd, rsi_roll, rsi_values, atr_window, atr_ma, stoch_rsi, ema_length3, psar_parameters, bollinger_sma, 
        adx_window, obv_ema, ema_length4, sma_length_4, sma_length_5, length_macd2, mfi_length, mfi_values, pvt_length, adl_ema, wr_length, vroc, 
        nvi, momentum, cci, bull_bear_power, mass_index, trix, vortex, z_score_levels, atr_multipliers, re_evaluar, riesgo_por_operacion, moneda_cuenta, 
        lista_pesos_confirmaciones, var_multiplier, corr_limit, multiplicador_tp_sl, lista_pesos_estrategias):
    #-------------------Inicialización---------------------------

    account = iniciar_mt5()

    bot = telebot.TeleBot(account["token_telebot"])
    chanel = account["chanel_telebot"]

    """@bot.message_handler(func=lambda message: True)
    def get_chat_id(message):
        #Esta funcion sirve para mostrar el id del grupo
        print(f"Chat ID: {message.chat.id}")
        bot.reply_to(message, f"El ID de este chat es: {message.chat.id}")

    bot.polling()"""

    #-------------------Utilidades-------------------------------

    account_info = mt.account_info()
    #account_info.balance
    #account_info.equity

    dinero_inicial = account_info.equity
    dinero_inicial_diario = dinero_inicial

    fecha_guardada, saldo_guardado = leer_saldo("data/registros/saldo.json")
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")  # Solo la fecha sin hora

    if fecha_guardada:
        fecha_guardada = fecha_guardada.split()[0]  # Obtener solo la parte de la fecha

    print()
    if not saldo_guardado or fecha_guardada != fecha_hoy:
        if dinero_inicial is not None:
            guardar_saldo("data/registros/saldo.json", dinero_inicial)
            print(f"Saldo actualizado: {dinero_inicial}")
        else:
            print("Error al obtener saldo de MetaTrader.")
    else:
        print(f"Saldo de hoy ya registrado: {saldo_guardado}")

    symbols = mt.symbols_get() # cosas operables
    symbols_list = []
    for i in symbols:
        symbols_list.append(i.name)

    #-------------------Configuraciones--------------------------

    verbose = True
    lista_pares = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD']
    correlation_dict = {}
    combinaciones = itertools.combinations(lista_pares, 2)
    original_min_spread_trigger = min_spread_trigger
    original_max_spread_trigger = max_spread_trigger
    min_spread_trigger = 0
    max_spread_trigger = 0

    print("""
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%,    %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %(  *%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   %%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%  .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#  ,%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%   #%%%%%%%%%%%%%%%%%%%%%%%%%%%%(  *%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%*  /%%%%%%%%%%%%%%%%%%%%%%%%,  #%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%#  .%%%%%%%%%%%%%%%%%%%%   %%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*   %%%%%%%%%%%%.  %%%%%%%%%%%%%%%%  .%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  .  (%%%%%%%%%%%*  (%%%%%%%%%%(  *%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%/  *%%%%%%%%%%%(  ,%%%%%%*  (%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%   %%%%%%%%%%%%   %%.  %%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%   %%%%%%%%%%%%    %%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%/  /%%%%%%%#  *%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%(  .%%%/  /%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%*  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%/..%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%,,%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Market Master  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """)

    print("Activos invertibles: ", lista_pares)
    print("Activos invertibles: ", len(lista_pares))

    deviation = 1.0
    max_slipage = 99.95/100
    counter = 0
    operar = True

    if len(lista_pares) > 0: 
        #comprobar si existen
        for i in lista_pares:
            if i not in symbols_list:
                mt.shutdown()
                raise Error(f'{i} not in valid symbols')

    #-------------------Código Principal-------------------------

    try:
        print()

        if save_file:
            orig_stdout = sys.stdout
            f = open('out.txt', 'w')
            sys.stdout = f

        resultado_liquidez_bid = []
        resultado_liquidez_ask = []

        hoy = datetime.fromordinal(datetime.today().toordinal())
        siguiente = hoy + timedelta(1)
        dinero_inicial = mt.account_info()._asdict()['balance']

        if es_horario_de_verano(hoy): # es verano entonces se usa el segundo trigger
            zona_horaria_operable = zona_horaria_operable[1]

        else: # es invierno entonces se usa el primer trigger
            zona_horaria_operable = zona_horaria_operable[0]

        liquidity_bid = {}
        liquidity_ask = {}
        final_liquidity = {}

        max_workers = os.cpu_count() * 2

        #-------------------Comprobar que no haya maxima perdida diaria-------------------------
        #deals=mt.history_deals_get(hoy, siguiente, group=name)
        #acumulado_total_anterior_operaciones=0
        #for i in deals:
            #acumulado_total_anterior_operaciones+=i.profit

        #if ((dinero_inicial+acumulado_total_anterior_operaciones)/dinero_inicial)-1 < -maximo_perdida_diaria:
            #raise Error('Maximo perdida diaria ya alcanzado')        

        #-------------------Buscar puntos liquidez-------------------------
        if longitud_liquidez != 0:
            contador = 1
            while len(resultado_liquidez_bid) < longitud_liquidez:
                tick_data = pd.DataFrame(mt.copy_ticks_range(name, hoy - timedelta(contador - 1), hoy - timedelta(contador - 2), mt.COPY_TICKS_ALL))
                tick_data['time'] = [datetime.fromtimestamp(item) for item in tick_data["time"]]
                tick_data = pl.from_pandas(tick_data)

                if len(tick_data) != 0 and (hoy - timedelta(contador - 1)).weekday() not in (5, 6):
                    liquidez_bid = {}
                    liquidez_ask = {}

                    if contador - 1 != 0:    
                        for pos, i in tqdm(enumerate(tick_data.iter_rows(named = True)), total = len(tick_data)):
                            bid = float("%.5f" % i["bid"])
                            ask = float("%.5f" % i["ask"])

                            liquidity(ask, liquidez_ask)
                            liquidity(bid, liquidez_bid)

                        resultado_liquidez_ask.insert(0, liquidez_ask)
                        resultado_liquidez_bid.insert(0, liquidez_bid)
                    else:
                        pass
                
                contador += 1

            print('\nTamaño liquidez precalculada: ', len(resultado_liquidez_ask))
        print()

        marketmaster = MarketMaster(atr_multipliers, maximo_perdida_diaria, riesgo_por_operacion, dinero_inicial, lista_pesos_confirmaciones, 
                                    corr_limit, generate_seed(), multiplicador_tp_sl, lista_pesos_estrategias)
        marketmastermanagement = MarketMasterManagement(marketmaster)
        market_master_maths = Maths()

        tick_data = pd.DataFrame(mt.copy_ticks_range(name, hoy,siguiente, mt.COPY_TICKS_ALL))
        tick_data["time"] = [datetime.fromtimestamp(item) for item in tick_data["time"]]  

        _, data_5m = get_data((name, hoy, siguiente, market_master_maths, mt.TIMEFRAME_M5))
        tick_data = pl.from_pandas(tick_data)

        liquidity_ask = {}
        liquidity_bid = {}
        puntos_liquidez = {}
        contador = 1

        # iterar todos los datos haciendo como que acaba de empezar el bot por el principio del dia
        for pos_dia, i in enumerate(tqdm(tick_data.iter_rows(named = True), total = len(tick_data))):
            fecha = i["time"]
            bid = float("%.5f" % i["bid"])
            ask = float("%.5f" % i["ask"])
            curr_spread = ask - bid

            #data_5m_actual=data_5m[data_5m["time"] <= fecha]
            
            #limite_spread = (max_spread_trigger +data_5m_actual['adjusted_atr'] * (min_spread_trigger - max_spread_trigger)).values[-1]

            limite_spread = max_spread_trigger
            
            if not (curr_spread <= limite_spread and zona_horaria_operable[0] <= fecha.hour <= zona_horaria_operable[1]):
                operar = False
            else:
                operar = True

            if ((pos_dia % ticks_refresco_liquidez == 0 and operar) or len(puntos_liquidez) == 0 or recalcular):
            #if (pos_dia%self.ticks_refresco_liquidez==0 or len(puntos_liquidez)==0 or recalcular):
                recalcular = False
                final_liquidity, final_exit, puntos_liquidez = recalcular_liquidez(resultado_liquidez_bid, resultado_liquidez_ask, liquidity_ask, liquidity_bid, limite_potencia)

                #if verbose:
                    #print("\nRecalculando liquidez")
                    #print(final_liquidity)
                    #print()

            if (operar and len(puntos_liquidez) != 0) and (bid in final_liquidity or ask in final_liquidity):
                recalcular = True
                print("\nrecalculando\n")

            liquidity(bid, liquidity_bid)
            liquidity(ask, liquidity_ask)

            contador += 1

        print(f"\nLiquidez precalculada correctamente y datos hasta el momento actual tambien\nLiquidez: {final_liquidity}\n")
        # continuar por donde lo había dejado
        last_element = tick_data[-1, 'time']

        if sounds:
            playsound('data\\ok.mp3')

        print(f'Running on {name}...')
        
        dinero_inicial = mt.account_info()._asdict()['balance']
        recalcular = False
        numero_posiciones = 0

        #-------------------Diccionario correlacion-------------------------

        datos = {}

        for i in lista_pares:
            datos[i] = pd.DataFrame(mt.copy_rates_range(i, mt.TIMEFRAME_M5, hoy - timedelta(4), hoy)) # coger datos hasta el dia actual sin incluirlo para que no sea trampa
            datos[i]["time"] = [datetime.fromtimestamp(item) for item in datos[i]["time"]]

        total_size_combinations = 0

        for activo1, activo2 in combinaciones:
            total_size_combinations += 1
            # Evitar combinaciones de un par consigo mismo
            if activo1 == activo2:
                continue
                
            if len(datos[activo1]) == len(datos[activo2]):
                serie1 = datos[activo1]['close']
                serie2 = datos[activo2]['close']
                serie1_normalizada = ((serie1 - serie1.min()) / (serie1.max() - serie1.min()))
                serie2_normalizada = ((serie2 - serie2.min()) / (serie2.max() - serie2.min()))
                correlacion = serie1_normalizada.corr(serie2_normalizada)
                correlation_dict[f"{activo1}-{activo2}"] = (correlacion, np.var(serie1_normalizada), sum(serie1_normalizada) / len(serie1_normalizada), np.var(serie2_normalizada), sum(serie2_normalizada) / len(serie2_normalizada))
                #if self.verbose:
                    #print(self.correlation_dict[f"{activo1}-{activo2}"])

        # Filtrar solo las correlaciones que incluyen el par actual
        correlation_dict = {clave: valor for clave, valor in correlation_dict.items() if clave.startswith(name) or clave.endswith(name)}
        #name2 = max(correlation_dict, key=lambda k: abs(correlation_dict[k][0]))
        #print(correlation_dict,name2)

        prints_noticia = [0, 0]

        if bloquear_noticias:
            if es_horario_de_verano(hoy):
                calendario_base = ivp.economic_calendar(
                    time_zone = "GMT +1:00",
                    importances = ["high"],
                    from_date = "{}".format((hoy - timedelta(days = 1)).strftime("%d/%m/%Y")),
                    to_date = "{}".format(hoy.strftime("%d/%m/%Y"))
                )
            else:
                calendario_base = ivp.economic_calendar(
                    time_zone = "GMT",
                    importances = ["high"],
                    from_date = "{}".format((hoy - timedelta(days = 1)).strftime("%d/%m/%Y")),
                    to_date = "{}".format(hoy.strftime("%d/%m/%Y"))
                )

            try:                        
                calendario_base = calendario_base.drop(["id", "actual", "forecast", "previous"], axis=1)
                calendario_base = calendario_base.iloc[np.where(calendario_base["importance"] == "high")]
                calendario_base = calendario_base.drop(calendario_base[calendario_base['time'] == 'All Day'].index)
                calendario_base["time"] = [datetime.strptime(item, "%H:%M") for item in calendario_base["time"]]                    
                calendario = calendario_base
            except:
                calendario = None

        anterior = None
        dia_de_la_semana = hoy.weekday()
        operaciones_diarias_puntos = []
        operaciones_diarias = []
        comprobacion_inicial = False
        anterior_data_5m = None

        if chanel is not None:
            bot.send_message(chanel, "Recalculated liquidity points: {}".format(list(final_liquidity.keys())))

        while True:
            start = time.time()
            all_data = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                futures = {executor.submit(get_tick_data, (name_par, hoy, siguiente)): name_par for name_par in lista_pares}
                for future in futures:
                    name_par, data = future.result()
                    #assert (len(data)>=10000)
                    all_data[name_par] = data

            tick_data = all_data[name]

            if min_spread_trigger == 0 and max_spread_trigger == 0:
                all_data_5m = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                    futures = {executor.submit(get_data, (nombre, hoy, siguiente, market_master_maths, mt.TIMEFRAME_M5)): nombre for nombre in lista_pares}
                    for future in futures:
                        nombre, data = future.result()
                        all_data_5m[nombre] = data

                data_5m = all_data_5m[name]

                if es_horario_de_verano(hoy): # es verano entonces se usa el segundo trigger
                    max_spread_trigger = convertir_spread(tick_data['ask'].iloc[0], original_max_spread_trigger[1])
                    min_spread_trigger = convertir_spread(tick_data['ask'].iloc[0], original_min_spread_trigger[1])
                else: # es invierno entonces se usa el primer trigger
                    max_spread_trigger = convertir_spread(tick_data['ask'].iloc[0], original_max_spread_trigger[0])
                    min_spread_trigger = convertir_spread(tick_data['ask'].iloc[0], original_min_spread_trigger[0])

                #print(f"Usando spread_trigger: {spread_trigger}")

            """
            tick_data=pd.DataFrame(mt.copy_ticks_range(name, hoy,siguiente, mt.COPY_TICKS_ALL))
            tick_data=tick_data.drop(columns=["last","volume","time_msc","volume_real","flags"],axis=1)
            tick_data["time"]=[datetime.fromtimestamp(item) for item in tick_data["time"]]
            """

            actual = tick_data.iloc[-1]
            if anterior is None:
                anterior = tick_data.iloc[-1]

            #print(time.time()-start)

            if actual['original_time'] != last_element:
                if dia_de_la_semana == 4 and actual['time'].hour == 21:
                    break

                bid = float('%.5f' % actual['bid'])
                ask = float('%.5f' % actual['ask'])
                anterior_bid = float('%.5f' % anterior['bid'])
                anterior_ask = float('%.5f' % anterior['ask'])

                account_info_dict = mt.account_info()._asdict()
                dinero = account_info_dict['equity']
                posiciones = list(mt.positions_get())
                #  0        1    2      3        4          5       6         7         8     9       10         11     12      13     14
                #ticket   time  type  magic  identifier  reason  volume  price_open    sl    tp  price_current  swap  profit  symbol comment

                if ((dinero - dinero_inicial) / dinero_inicial) <= -maximo_perdida_diaria:
                    print("\n\nLiquidando posiciones\n")
                    raise Error('Maximo perdida diaria ya alcanzado')        

                action = 0
                accion = 0
                posiciones_cortos = []
                posiciones_largos = []
                margen_total = 0

                #print(posiciones)

                if len(posiciones) > 0:
                    if actual['time'].minute % 5 == 0 and actual['time'].second in [0, 1, 2]:
                        all_data_5m = {}
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                            futures = {executor.submit(get_data, (nombre, hoy, siguiente, market_master_maths, mt.TIMEFRAME_M5)): nombre for nombre in lista_pares}
                            for future in futures:
                                nombre, data = future.result()
                                all_data_5m[nombre] = data

                        data_5m = all_data_5m[name]

                    #print(posiciones,operaciones_diarias)

                    for _i_ in posiciones:
                        valor_trailing_tp = marketmaster.valor_pip(trigger_trailing_tp, bid)
                        valor_trailing_sl = marketmaster.valor_pip(trigger_trailing_sl, bid)
                        posicion_almacenada_localmente = [(pos, i) for pos, i in enumerate(operaciones_diarias) if i.id == _i_.comment and i.entrada == _i_.price_open]
                        if len(posicion_almacenada_localmente) != 0:
                            posicion_almacenada_localmente = posicion_almacenada_localmente[0]
                            #print(posicion_almacenada_localmente)
                            _i_local = posicion_almacenada_localmente[1]
                            valor_parciales_profit = ajustar_intervalo(getattr(_i_local, "anterioridad"), 0, longitud_liquidez, 50, 70)

                        if _i_.type == 0:  # largo
                            margen_total += mt.order_calc_margin(mt.ORDER_TYPE_BUY, _i_.symbol, _i_.volume, _i_.price_open) * 10
                            posiciones_largos.append(_i_)
                            if _i_.sl < _i_.price_open and bid >= _i_.price_open + marketmaster.valor_pip(tamanyo_break_even, _i_.price_open) and enable_break_even:
                                mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.price_open + marketmaster.valor_pip(break_even_size, _i_.price_open), "tp": _i_.tp})
                                if verbose:
                                    print("\nbe largo")

                            elif enable_dinamic_sl and anterior_bid < bid and bid > (_i_.sl + valor_trailing_sl) and not (_i_.tp - valor_trailing_tp) <= bid <= _i_.tp:  # añadir condicion para sl trigger
                                mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.sl + (bid - anterior_bid), "tp": _i_.tp})
                                if verbose:
                                    print("dinamic sl largo")

                            elif enable_dinamic_tp and (_i_.tp - valor_trailing_tp) <= bid <= _i_.tp:
                                mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.tp - valor_trailing_tp, "tp": _i_.tp + (bid - anterior_bid)})
                                if verbose:
                                    print("dinamic tp largo")

                            elif len(posicion_almacenada_localmente) != 0 and getattr(_i_local, "parciales_retirados") == False and bid >= _i_.price_open + (_i_.tp - _i_.price_open) * valor_parciales_profit:  # el precio se encuentra en el medio
                                lotaje_a_liquidar = round(_i_.volume * valor_parciales_profit, 2)
                                # cambiar lotaje
                                resultado, _ = order(_i_.symbol, mt.symbol_info_tick(_i_.symbol).bid, 0.0, 0.0, 0, mt.TRADE_ACTION_DEAL, mt.ORDER_TYPE_SELL, deviation, lotaje_a_liquidar, position=_i_.ticket)
                                if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                    print(f'Error parciales no completado {resultado.retcode}')
                                else:
                                    # cambiar sl
                                    resultado = mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.price_open, "tp": _i_.tp})
                                    if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                        print(f'Error parciales no completado {resultado.retcode}')
                                    else:
                                        operaciones_diarias[posicion_almacenada_localmente[0]] = _i_local._replace(sl=_i_.price_open, parciales_retirados=True, lotaje=_i_.volume - lotaje_a_liquidar)

                                        print("\nRetiro de parciales (largo)")
                                        bot.send_message(chanel,"Retiro de parciales (largo)\n{}\nWithdrawal percentage: {}%\nPlease set be!".format(_i_local.id,valor_parciales_profit*100))

                                """elif len(posicion_almacenada_localmente)!=0 and getattr(_i_local,"parciales_retirados")==False and bid<=_i_.price_open-(_i_.price_open-_i_.sl)*(1-valor_parciales_profit): #el precio se encuentra en el medio
                                    lotaje_a_liquidar=round(_i_.volume*(1-valor_parciales_profit),2)
                                    #cambiar lotaje
                                    operaciones_diarias[posicion_almacenada_localmente[0]]=_i_local._replace(lotaje=_i_.volume-lotaje_a_liquidar,parciales_retirados=True)
                                    resultado,_=order(_i_.symbol,mt.symbol_info_tick(_i_.symbol).bid,0.0,0.0,0,mt.TRADE_ACTION_DEAL,mt.ORDER_TYPE_SELL,deviation,lotaje_a_liquidar,position=_i_.ticket)
                                    print("\nRetiro de parciales en perdida (largo)")
                                    bot.send_message(chanel,"Retiro de parciales en perdida (largo)\n{}\nWithdrawal percentage: {}%".format(_i_local.id,valor_parciales_profit*100))"""

                            elif re_evaluar and accion == -1:
                                order(_i_.symbol, mt.symbol_info_tick(_i_.symbol).bid, 0.0, 0.0, 0, mt.TRADE_ACTION_DEAL, mt.ORDER_TYPE_SELL, 20, _i_.volume, position=_i_.ticket)
                            
                            if actual['time'].minute % 5 == 0 and actual['time'].second in [0, 1, 2]:
                                try:
                                    tp, sl = marketmastermanagement.profit_management(args, 1, ask, bid, data_5m.iloc[-1], curr_spread)
                                    if sl < bid < tp:
                                        if tp < _i_.tp:
                                            resultado = mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.sl, "tp": tp})
                                            if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                                print(f'Error modificacion tp no completado {resultado.retcode}')
                                            else:
                                                operaciones_diarias[posicion_almacenada_localmente[0]] = _i_local._replace(tp=tp)
                                                print(f"\nTp/sl modificados")
                                                bot.send_message(chanel,"Tp modificado\n{}\nNew tp: {}".format(_i_local.id,tp))
                                        #if sl>getattr(posicion,"sl"):
                                            #self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(sl=sl)
                                            #print(f"\nTp/sl modificados")

                                except:
                                    pass

                        else:  # corto
                            margen_total += mt.order_calc_margin(mt.ORDER_TYPE_SELL, _i_.symbol, _i_.volume, _i_.price_open) * 10
                            posiciones_cortos.append(_i_)
                            if _i_.sl > _i_.price_open and ask <= _i_.price_open - marketmaster.valor_pip(tamanyo_break_even, _i_.price_open) and enable_break_even:
                                mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.price_open - marketmaster.valor_pip(break_even_size, _i_.price_open), "tp": _i_.tp})
                                if verbose:
                                    print("\nbe corto")

                            elif enable_dinamic_sl and anterior_ask > ask and ask < (_i_.sl - valor_trailing_sl) and not (_i_.tp <= ask <= (_i_.tp + valor_trailing_tp)):
                                mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.sl + (ask - anterior_ask), "tp": _i_.tp})
                                if verbose:
                                    print("dinamic sl corto")

                            elif enable_dinamic_tp and (_i_.tp <= ask <= (_i_.tp + valor_trailing_tp)):
                                mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.tp + valor_trailing_tp, "tp": _i_.tp + (ask - anterior_ask)})
                                if verbose:
                                    print("dinamic tp corto")

                            elif len(posicion_almacenada_localmente) != 0 and getattr(_i_local, "parciales_retirados") == False and ask <= _i_.price_open - (_i_.price_open - getattr(_i_, "tp")) * valor_parciales_profit:  # va por la mitad
                                lotaje_a_liquidar = round(_i_.volume * valor_parciales_profit, 2)
                                # cambiar lotaje
                                resultado, _ = order(_i_.symbol, mt.symbol_info_tick(_i_.symbol).ask, 0.0, 0.0, 0, mt.TRADE_ACTION_DEAL, mt.ORDER_TYPE_BUY, deviation, lotaje_a_liquidar, position=_i_.ticket)
                                if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                    print(f'Error parciales no completado {resultado.retcode}')
                                else:
                                    # cambiar sl
                                    resultado = mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.price_open, "tp": _i_.tp})
                                    if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                        print(f'Error parciales no completado {resultado.retcode}')
                                    else:
                                        operaciones_diarias[posicion_almacenada_localmente[0]] = _i_local._replace(sl=_i_.price_open, parciales_retirados=True, lotaje=_i_.volume - lotaje_a_liquidar)

                                        print("\nRetiro de parciales (corto)")
                                        bot.send_message(chanel,"Retiro de parciales (corto)\n{}\nWithdrawal percentage: {}%\nPlease set be!".format(_i_local.id,valor_parciales_profit*100))

                                """elif len(posicion_almacenada_localmente)!=0 and getattr(_i_local,"parciales_retirados")==False and ask>=_i_.price_open+(_i_.sl-_i_.price_open)*(1-valor_parciales_profit): #va por la mitad
                                    lotaje_a_liquidar=round(_i_.volume*(1-valor_parciales_profit),2)
                                    #cambiar lotaje
                                    operaciones_diarias[posicion_almacenada_localmente[0]]=_i_local._replace(lotaje=_i_.volume-lotaje_a_liquidar,parciales_retirados=True)
                                    resultado,_=order(_i_.symbol,mt.symbol_info_tick(_i_.symbol).ask,0.0,0.0,0,mt.TRADE_ACTION_DEAL,mt.ORDER_TYPE_BUY,deviation,lotaje_a_liquidar,position=_i_.ticket)
                                    print("\nRetiro de parciales en perdida (corto)")
                                    bot.send_message(chanel,"Retiro de parciales en perdida (corto)\n{}\nWithdrawal percentage: {}%".format(_i_local.id,valor_parciales_profit*100))"""

                            elif re_evaluar and accion == 1:
                                order(_i_.symbol, mt.symbol_info_tick(_i_.symbol).ask, 0.0, 0.0, 0, mt.TRADE_ACTION_DEAL, mt.ORDER_TYPE_BUY, 20, _i_.volume, position=_i_.ticket)

                            if actual['time'].minute % 5 == 0 and actual['time'].second in [0, 1, 2]:
                                try:
                                    tp, sl = marketmastermanagement.profit_management(args, -1, ask, bid, data_5m.iloc[-1], curr_spread)
                                    if tp < ask < sl:
                                        if tp > _i_.tp:
                                            resultado = mt.order_send({"action": mt.TRADE_ACTION_SLTP, "position": _i_.ticket, "symbol": _i_.symbol, "sl": _i_.sl, "tp": tp})
                                            if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                                print(f'Error modificacion tp no completado {resultado.retcode}')
                                            else:
                                                operaciones_diarias[posicion_almacenada_localmente[0]] = _i_local._replace(tp=tp)
                                                print(f"\nTp/sl modificados")
                                                bot.send_message(chanel,"Tp modificado\n{}\nNew tp: {}".format(_i_local.id,tp))
                                        #if sl<getattr(posicion,"sl"):
                                            #self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(sl=sl)
                                            #print(f"\nTp/sl modificados")
                                except:
                                    pass

                positions = mt.positions_total()
                if positions != numero_posiciones:
                    recalcular = True
                    numero_posiciones = positions

                #spread_trigger = (max_spread_trigger +data_5m['adjusted_atr'].iloc[-1] * (min_spread_trigger - max_spread_trigger))
                spread_trigger = (data_5m['moda_superior1'] - data_5m['adjusted_atr'] * (data_5m['moda_superior1'] - data_5m['moda_inferior'])).values[-1]

                if (ask - bid) <= spread_trigger and ((dinero - dinero_inicial) / dinero_inicial) >= -maximo_perdida_diaria and zona_horaria_operable[0] <= actual["time"].hour <= zona_horaria_operable[1]:
                    operar = True
                    if bloquear_noticias and type(calendario) != type(None) and len(calendario) > 0:
                        coincidencias = list(i for i in calendario["time"] if (i - timedelta(minutes=15)).time() <= actual['time'].time() <= (i + timedelta(minutes=5)).time())
                        #no operar 15 minutos antes ni 15 minutos despues

                        if len(coincidencias) > 0:
                            for coincidencia in coincidencias:
                                calendario = calendario[calendario.time != coincidencia]

                            if verbose and not prints_noticia[0]:
                                print("\nRango noticia, no operar")
                                prints_noticia[0] = 1

                            operar = False

                        else:
                            if verbose and not prints_noticia[1] and prints_noticia[0]:
                                prints_noticia[1] = 1
                                print("\nFuera noticia operar")
                            operar = True
                else:
                    operar = False
                    if ((dinero - -dinero_inicial) / dinero_inicial) < -maximo_perdida_diaria:
                        print("\n\nLiquidando posiciones\n")
                        break

                if len(final_liquidity) == 0 or (contador % ticks_refresco_liquidez == 0 and operar) or recalcular:
                    recalcular = False
                    puntos_liquidez_anterior = puntos_liquidez.copy()

                    final_liquidity, final_exit, puntos_liquidez = recalcular_liquidez(resultado_liquidez_bid, resultado_liquidez_ask, liquidity_ask, liquidity_bid, limite_potencia)

                    if puntos_liquidez.keys() != puntos_liquidez_anterior.keys():
                        bot.send_message(chanel, "Recalculated liquidity points: {}".format(list(final_liquidity.keys())))

                    puntos_liquidez_anterior.clear()

                    if verbose:
                        print("\nRecalculando liquidez")
                        #print(final_liquidity)
                        #print()

                if account_info_dict['profit'] == 0:
                    color_profit = Fore.BLUE
                elif account_info_dict['profit'] > 0:
                    color_profit = Fore.GREEN
                elif account_info_dict['profit'] < 0:
                    color_profit = Fore.RED

                # reevaluar posiciones y recalcular cuando tp/sl

                if operar:
                #if True:
                    color_operar = Fore.GREEN

                    if (bid in final_liquidity or ask in final_liquidity) or not comprobacion_inicial:
                    #if len(posiciones)<=3 or not comprobacion_inicial:
                        if comprobacion_inicial:
                            recalcular = True

                        if not (type(anterior_data_5m) != None and len(data_5m) == anterior_data_5m):
                        #if True:
                            #all_data_1m = {}
                            #all_data_2m = {}
                            all_data_5m = {}
                            all_data_15m = {}
                            all_data_1h = {}

                            """with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                                futures = {executor.submit(get_data, (nombre, hoy, siguiente, market_master_maths, mt.TIMEFRAME_M1)): nombre for nombre in lista_pares}
                                for future in futures:
                                    nombre, data = future.result()
                                    all_data_1m[nombre] = data"""

                            """with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                                futures = {executor.submit(get_data, (nombre,hoy,siguiente,market_master_maths,mt.TIMEFRAME_M2)): nombre for nombre in lista_pares}
                                for future in futures:
                                    nombre,data=future.result()
                                    all_data_2m[nombre] = data"""

                            with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                                futures = {executor.submit(get_data, (nombre, hoy, siguiente, market_master_maths, mt.TIMEFRAME_M5)): nombre for nombre in lista_pares}
                                for future in futures:
                                    nombre, data = future.result()
                                    all_data_5m[nombre] = data

                            with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                                futures = {executor.submit(get_data, (nombre, hoy, siguiente, market_master_maths, mt.TIMEFRAME_M15)): nombre for nombre in lista_pares}
                                for future in futures:
                                    nombre, data = future.result()
                                    all_data_15m[nombre] = data

                            with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Ajusta max_workers según tu hardware
                                futures = {executor.submit(get_data, (nombre,hoy,siguiente,market_master_maths,mt.TIMEFRAME_H1)): nombre for nombre in lista_pares}
                                for future in futures:
                                    nombre,data=future.result()
                                    all_data_1h[nombre] = data

                            #data_1m = all_data_1m[name]
                            #data_2m = all_data_2m[name]
                            data_5m = all_data_5m[name]
                            data_15m = all_data_15m[name]
                            data_1h = all_data_1h[name]

                            args = {"correlation_dict": correlation_dict, "recalcular": False, "all_data": all_data, "all_data_15m": all_data_15m, "all_data_5m": all_data_5m, "all_data_1h": all_data_1h, "name": name, "mode": "real"} #"all_data_1m": all_data_1m, "all_data_2m": all_data_2m
                            args["dinero"] = [dinero_inicial, dinero, dinero_inicial_diario]
                            args["puntos_liquidez"] = final_liquidity
                            args["puntos_salida"] = final_exit
                            args["posiciones_cortos"] = posiciones_cortos
                            args["posiciones_largos"] = posiciones_largos
                            args["pos_dia"] = None
                            args["actual_time"] = actual['time']
                            args["ask"] = ask
                            args["bid"] = bid
                            args['temporality'] = "all_data_5m"
                            args["comprobacion_inicial"] = not(comprobacion_inicial)

                            action, sl, tp, lotaje, accion, estrategias = marketmaster.run(args)
                            
                            """action = 1
                            estrategias = [1,0]
                            #tp,sl=marketmastermanagement.profit_management(args,action,ask,bid,data_5m.iloc[-1],curr_spread)
                            if action == -1:
                                tp = bid-0.0003
                                sl = bid+0.0003
                            else:
                                tp = ask+0.0003                                
                                sl = ask-0.0003

                            lotaje = 0.02

                            comprobacion_inicial = True"""

                            print(f'Evaluar punto liquidez {bid} - {ask} | {action} | Señales: {estrategias}')
                            if action != 0 and comprobacion_inicial:
                                #if ask in posiciones_largos or bid in posiciones_cortos:
                                if ask in operaciones_diarias_puntos or bid in operaciones_diarias_puntos:
                                    lotaje = round(lotaje / 2, 2)

                                anterior_data_5m = len(data_5m)

                    if action == 1 and comprobacion_inicial:  # comprar
                        margin_necesary = mt.order_calc_margin(mt.ORDER_TYPE_BUY, name, lotaje, ask) * 10
                        if puede_abrir_orden(margin_necesary, margen_total, 1.5, dinero_inicial_diario, dinero):
                            operaciones_diarias_puntos.append(ask)
                            last_tick_to_buy = mt.symbol_info_tick(name)
                            recalcular = True
                            diferencia = last_tick_to_buy.ask - ask
                            if abs(diferencia) < abs(last_tick_to_buy.ask - last_tick_to_buy.ask * max_slipage):
                                tp = tp + diferencia
                                sl = sl + diferencia
                                sl, tp = round(sl, 5), round(tp, 5)
                                resultado, _ = order(name, last_tick_to_buy.ask, sl, tp, contador % 100, mt.TRADE_ACTION_DEAL, mt.ORDER_TYPE_BUY, deviation, lotaje, position=None)

                                if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                    bot.send_message(f'Operación no completada {resultado.retcode}')
                                else:
                                    try:
                                        operaciones_diarias.append(Posicion(name, fecha, action, last_tick_to_buy.ask, lotaje, tp, sl, tp, sl, 0, 0, "all_data_5m", False, final_liquidity[ask], f"Order number: {contador % 100}"))
                                    except:
                                        try:
                                            operaciones_diarias.append(Posicion(name, fecha, action, last_tick_to_buy.ask, lotaje, tp, sl, tp, sl, 0, 0, "all_data_5m", False, final_liquidity[bid], f"Order number: {contador % 100}"))
                                        except:
                                            operaciones_diarias.append(Posicion(name, fecha, action, last_tick_to_buy.ask, lotaje, tp, sl, tp, sl, 0, 0, "all_data_5m", False, 1, f"Order number: {contador % 100}"))

                                if verbose:
                                    print("\nCrear largo entrada:{} sl:{} tp:{} lotaje:{} | Señales: {}".format(
                                        last_tick_to_buy.ask, sl, tp, lotaje, estrategias
                                    ))
                                    
                                    bot.send_message(chanel, "Crear largo ({})\nSeñales: {}\nFecha (servidor): {}\nOrder number: {}\nEntrada:{}\nSl:{}\nTp:{}\nLotaje: {}".format(
                                        name, estrategias, actual["time"], contador,last_tick_to_buy.ask, sl, tp, lotaje
                                    ))

                                if sounds:
                                    playsound('data\\operacion.mp3')

                    elif comprobacion_inicial == False:
                        comprobacion_inicial = True
                        action = 0
                        print("Comprobacion inicial realizada con exito ✅\n")

                    if action == -1 and comprobacion_inicial: #vender
                        margin_necesary = mt.order_calc_margin(mt.ORDER_TYPE_BUY, name, lotaje, ask) * 10
                        if puede_abrir_orden(margin_necesary, margen_total, 1.5, dinero_inicial_diario, dinero):
                            operaciones_diarias_puntos.append(bid)
                            last_tick_to_sell = mt.symbol_info_tick(name)
                            recalcular = True
                            diferencia = last_tick_to_sell.bid - bid
                            if abs(diferencia) < abs(last_tick_to_sell.ask - last_tick_to_sell.ask * max_slipage):
                                tp = tp - diferencia
                                sl = sl - diferencia
                                sl, tp = round(sl, 5), round(tp, 5)
                                resultado, _ = order(name, last_tick_to_sell.bid, sl, tp, contador % 100, mt.TRADE_ACTION_DEAL, mt.ORDER_TYPE_SELL, deviation, lotaje, position=None)
                                if resultado.retcode != mt.TRADE_RETCODE_DONE:
                                    bot.send_message(f'Operación no completada {resultado.retcode}')
                                else:
                                    try:
                                        operaciones_diarias.append(Posicion(name, fecha, action, last_tick_to_sell.bid, lotaje, tp, sl, tp, sl, 0, 0, "all_data_5m", False, final_liquidity[bid], f"Order number: {contador % 100}"))
                                    except:
                                        try:
                                            operaciones_diarias.append(Posicion(name, fecha, action, last_tick_to_sell.bid, lotaje, tp, sl, tp, sl, 0, 0, "all_data_5m", False, final_liquidity[ask], f"Order number: {contador % 100}"))
                                        except:
                                            operaciones_diarias.append(Posicion(name, fecha, action, last_tick_to_sell.bid, lotaje, tp, sl, tp, sl, 0, 0, "all_data_5m", False, 1, f"Order number: {contador % 100}"))
                                
                                if verbose:
                                    print("\nCrear corto entrada:{} sl:{} tp:{} lotaje:{} | Señales: {}".format(
                                        last_tick_to_sell.bid, sl, tp, lotaje, estrategias
                                    ))

                                    bot.send_message(chanel, "Crear corto ({})\nSeñales: {}\nFecha (servidor): {}\nOrder number: {}\nEntrada:{}\nSl:{}\nTp:{}\nLotaje: {}".format(
                                        name, estrategias, actual["time"], contador,last_tick_to_sell.bid, sl, tp, lotaje
                                    ))
                                
                                if sounds:
                                    playsound('data\\operacion.mp3')
                    
                    elif comprobacion_inicial == False:
                        comprobacion_inicial = True
                        print("Comprobacion inicial realizada con exito ✅\n")

                else:
                    color_operar = Fore.RED

                print("{}●{} | Hora: {} | Tiempo ciclo: {} /s | Tick: {}{} - {}{} | Puntos liquidez cercanos: {}{} - {}{} | Balance: {:.1f} | Número Posiciones: {} | Profit: {}{}{} | Porcentaje Profit: {}{:.4f}%{}".format(
                    color_operar, Style.RESET_ALL, 
                    (actual["time"]).strftime('%H:%M:%S'), 
                    format(time.time() - start, '.5f'), 
                    Fore.LIGHTMAGENTA_EX, format(bid, '.5f'), format(ask, '.5f'), Style.RESET_ALL, 
                    Fore.GREEN, sorted(final_liquidity.keys(), key=lambda x: abs(bid - x))[0], 
                    sorted(final_liquidity.keys(), key=lambda x: abs(bid - x))[1], Style.RESET_ALL, 
                    dinero, numero_posiciones, 
                    color_profit, account_info_dict['profit'], Style.RESET_ALL, 
                    color_profit, round(account_info_dict['profit'] / dinero_inicial, 4) * 100, Style.RESET_ALL
                ))

                liquidity(ask, liquidity_ask)
                liquidity(bid, liquidity_bid)

                last_element = actual['original_time']
                contador += 1

            else:
                start = time.time()

            anterior = actual

    except BaseException as exc:
        if sounds:
            playsound('data\\error.mp3')

        print("\nApagando bot...\n")

        if not isinstance(exc, KeyboardInterrupt):
            print(f"Error: [{exc!r}]\n")  # Imprime el error solo si no es KeyboardInterrupt

        raise exc

    finally:
        if save_file:
            sys.stdout = orig_stdout
            f.close()

        try:
            liquidar = input("\nDesea liquidar las posiciones pendientes? (y/n): ")
            #liquidate and close connections
            if liquidar == "y":
                liquidate(20)
        except:
            pass

        if hoy.weekday() == 4 or (siguiente.date() in holidays.US()):
            liquidate(20)

        #show things
        dinero = mt.account_info()._asdict()['equity']

        print(f'Beneficio del día {hoy} => {round(dinero-dinero_inicial, 2)} ({round((dinero-dinero_inicial)/dinero_inicial*100, 2)}%)\n')

        comision_retiro = 100/100
        dinero_retirar = retiro(dinero, dinero_inicial, porcentaje_retiro, porcentaje_umbral_ganancias, comision_retiro)
        if dinero_retirar > 0:
            if verbose:
                print(f"Dinero a retirar: {dinero_retirar:.2f}")

                bot.send_message(chanel, f"Dinero a retirar: {dinero_retirar:.2f}, {actual["time"]}, {last_tick_to_buy.ask}") 
        else:
            if verbose:
                print("\nNo se cumplen condiciones de retiro en el {}\n".format(
                    actual["time"]
                    ))

                bot.send_message(chanel, "No se cumplen condiciones de retiro en el {}".format(
                    actual["time"]
                ))

        mt.shutdown()

        """
        if actual['time'].hour>20:
            print('\nPreparando para apagar ordenador...\n')
            time.sleep(60*5)
            subprocess.run(["shutdown", "-s"])
        """

if __name__=="__main__":
    run(
        dinero_inicial = 100000, name = "GBPUSD", tamanyo_break_even = 16, break_even_size = -15, enable_break_even = False,
        enable_dinamic_sl = False, enable_dinamic_tp = True, maximo_perdida_diaria = 1.2/100, riesgo_por_operacion = {'corto': 0.58/100, 'largo': 0.68/100, 'invierno': 0.68/100, 'verano': 0.68/100}, #maximo_perdida_diaria=3/100 riesgo_por_operacion=0.7/100
        trigger_trailing_tp = 0.5, trigger_trailing_sl = 6, #lotaje_maximo=1.4, lotaje_minimo=0.2
        longitud_liquidez = 7, #cuanto mayor mejor, 30 va bien, pero 10 es mas rápido para hacer las pruebas
        ticks_refresco_liquidez = 10000, limite_potencia = 1, bloquear_noticias = False, #maximo_operaciones_consecutivas=2
        atr_multipliers = {"all_data_1m":(2, 3), "all_data_2m":(2, 3), "all_data_5m":(2, 3), "all_data_1h":(2.5, 3.75), "all_data_15m":(2, 3)}, re_evaluar = False,
        moneda_cuenta = "USD", lista_pesos_confirmaciones = [0.7, 0.7, 0.6, 0.9, 0.8, 0.8, 0.8], var_multiplier = 1, corr_limit = 0.5,

        zona_horaria_operable = [[4, 22], [5, 23]], verbose = False, sounds = False,
        multiplicador_tp_sl = [1, 1], lista_pesos_estrategias = [1, 1, 1, 1, 1, 1], porcentaje_retiro = 50, porcentaje_umbral_ganancias = 13/100,

        save_file = False, min_spread_trigger = [0.00006, 0.00005], max_spread_trigger = [0.00009, 0.00008],

        ema_length = 8, ema_length2 = 55, ema_length3 = 55, ema_length4 = 144,

        sma_length_2 = 21, sma_length_3 = 34, length_macd = 13,
        sma_length_4 = 34, sma_length_5 = 55, length_macd2 = 21,

        rsi_roll = 13, stoch_rsi = (3, 3), rsi_values = (20, 80),
        bollinger_sma = 89,
        obv_ema = 5,

        adx_window = 2,
        psar_parameters = (0.02, 0.2, 0.02),
        atr_window = 13,    atr_ma = 21,
        mfi_length = 13, mfi_values = (20, 80),
        pvt_length = 144, # sin probar bien
        adl_ema = 144, #sin probar bien
        wr_length = (21, 89),
        vroc = 13,
        nvi = 34,
        momentum = 144,
        cci = (21, 5),
        bull_bear_power = 55,
        mass_index = (9, 25, 55),
        trix = 13,
        vortex = 13,
        z_score_levels = (-2, 2)
        )