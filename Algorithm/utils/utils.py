import MetaTrader5 as mt
from datetime import datetime
import time
import numpy as np
import investpy as ivp
from datetime import datetime,timedelta
import os
from typing import NamedTuple

class Error(RuntimeError): # Hereda todas las características de RuntimeError
    """Explicacion ..."""
    pass 

class Posicion(NamedTuple):
    name:str
    fecha:str
    typo:bool # 1=> largo, -1=> corto
    entrada:float
    lotaje:float
    tp:float
    sl:float
    original_tp:float
    original_sl:float
    resultado:float
    fecha_resultado:str
    #prioridad: int #duracion estimada
    duracion_estimada:float
    parciales_retirados:bool
    anterioridad:int
    id: int

def generate_seed():
    # Combina la hora actual y el PID del proceso para obtener un seed único
    timestamp = time.time()
    process_id = os.getpid()
    unique_seed = int((timestamp + process_id) * 1000) % 1000000
    return unique_seed

def convertir_spread(numero,spread_normal):
    if numero>1:
        return spread_normal*int(numero)
    else:
        return spread_normal*numero

# Función para determinar si es horario de verano (DST) en MetaTrader
def es_horario_de_verano(fecha):
    # El horario de verano en MetaTrader empieza el último domingo de marzo y termina el último domingo de octubre.
    year = fecha.year
    # El último domingo de marzo
    marzo = datetime(year, 3, 31)
    marzo = marzo - timedelta(days=marzo.weekday() + 1)
    
    # El último domingo de octubre
    octubre = datetime(year, 10, 31)
    octubre = octubre - timedelta(days=octubre.weekday() + 1)

    return marzo <= fecha <= octubre

def maximo_perdida(lista):
    contador=0
    mejor=0
    dinero_inicial=0
    for _,i in enumerate(lista):
        if _!=len(lista)-1:
            if i>=lista[_+1]:
                contador+=i-lista[_+1]
                
                if contador>mejor:
                    mejor=contador
                    dinero_inicial=lista[_+1]+mejor

            else:
                contador=0
    #print("Mejor: ", mejor, " Dinero incial: ",dinero_inicial)
    
    return mejor,dinero_inicial

def puede_abrir_orden(margin_necesary,margen_ocupado,max_margen_porcentaje,dinero_inicial_diario,valor_dinero):
    profit_diario=(valor_dinero-dinero_inicial_diario)/dinero_inicial_diario
    margen_libre = valor_dinero - margen_ocupado
    margen_permitido = margen_libre * (max_margen_porcentaje+profit_diario)
    return margin_necesary <= margen_permitido

def diferencia(n1, n2):
    return (n1-n2)/n2

def riesgo_beneficio(entrada,stop,profit):
    return abs(diferencia(profit,entrada)) / abs(diferencia(stop,entrada))

def numero_acciones(dinero,valor):
    return int(dinero//valor)

def horario():
    hora=datetime.datetime.now().time()
    operar=True
    #si no esta en killzone:
        #operar=False
    
    return operar

def calcular_tiempo_limite(tp,entrada,atr):
    distancia_tp=abs(tp-entrada)
    # Calcula el tiempo límite
    tiempo_limite = distancia_tp / atr
    return tiempo_limite

def calculate_zscore(spread, window=30):
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    return (spread - mean) / std

def retiro(saldo_actual,saldo_base,porcentaje_retiro,porcentaje_umbral_ganancias,comision_retiro):
    dinero_retirar=0
    """
    Input:

    Output: 0 si no se retira, y !=0 es lo que se retira
    """
    #print(f" Valor saldo base: {saldo_base}, tipo: {type(saldo_base)}, Valor porcentaje umbral ganancias: {porcentaje_umbral_ganancias}, tipo: {type(porcentaje_umbral_ganancias)}")
    umbral_ganancias = saldo_base * porcentaje_umbral_ganancias
    if saldo_actual > saldo_base + umbral_ganancias:
        ganancia_excedente = saldo_actual - (saldo_base + umbral_ganancias)
        dinero_retirar = ganancia_excedente * porcentaje_retiro
        comision = dinero_retirar * comision_retiro
        dinero_retirar -= comision

    return dinero_retirar

def get_events():
    """
    devuelve False si no se puede operar
    devuelve True si se puede operar
    """
    today=datetime.datetime.today().date()
    timezone=int(int(time.strftime("%z", time.gmtime()))/100)

    hora=datetime.datetime.now().time()

    calendario=ivp.economic_calendar(time_zone=f"GMT +{timezone}:00",importances=["high","medium"],from_date=f"{(today-datetime.timedelta(days=1)).strftime("%d/%m/%Y")}",to_date=f"{today.strftime("%d/%m/%Y")}")
    calendario=calendario.drop(["id","actual","forecast","previous"],axis=1)
    calendario=calendario.iloc[np.where(calendario["date"]==f"{today.strftime("%d/%m/%Y")}")]
    calendario=calendario.drop(calendario[calendario['time'] == 'All Day'].index)

    operar=True

    if len(calendario)!=0:
        for i in calendario.iterrows():
            if i[1]["importance"]=="medium":
                hora_noticia=datetime.datetime.strptime(i[1]["time"],"%H:%M").time()
                hora_min=(datetime.datetime.combine(datetime.date(1,1,1),hora_noticia) - datetime.timedelta(minutes=5)).time()
                hora_max=(datetime.datetime.combine(datetime.date(1,1,1),hora_noticia) + datetime.timedelta(minutes=5)).time()
                
                if hora_min<=hora<=hora_max:
                    operar=False
                
            elif i[1]["importance"]=="high":
                hora_noticia=datetime.datetime.strptime(i[1]["time"],"%H:%M").time()
                hora_min=(datetime.datetime.combine(datetime.date(1,1,1),hora_noticia) - datetime.timedelta(minutes=15)).time()
                hora_max=(datetime.datetime.combine(datetime.date(1,1,1),hora_noticia) + datetime.timedelta(minutes=15)).time()

                if hora_min<=hora<=hora_max:
                    operar=False

    return operar

def dias_no_operables():
    pass
    #print(data.earnings_dates) # no operar este dia
    #print(data.dividends) # no operar este dia
    #print(data.splits) # no operar este dia
    
def get_institutions(data):
    return round(((float(data.major_holders.iloc[1][0][:-1]) + float(data.major_holders.iloc[2][0][:-1])) / 2)+float(data.major_holders.iloc[0][0][:-1]),2) #cuanto menor participacion de instituciones, el precio se mueve menos tecnico

def place_test_order(self, symbol, direction):
    """Coloca una orden de prueba y analiza el slippage y spread."""
    test_lot = 0.01  # Tamaño mínimo de la orden de prueba
    entry_price = self.api.get_current_price(symbol)
    spread_before = self.api.get_spread(symbol)  # Medir spread antes
    
    order_id = self.api.place_order(symbol, direction, test_lot, entry_price)
    time.sleep(0.5)  # Pequeña espera para ver reacción del mercado
    
    execution_price = self.api.get_order_execution_price(order_id)
    spread_after = self.api.get_spread(symbol)  # Medir spread después
    
    slippage = abs(execution_price - entry_price)
    spread_change = abs(spread_after - spread_before)
    
    return slippage, spread_change

def detect_hft_activity(spread_before2,spread_before,spread_change):
    """Evalúa si hay actividad sospechosa de HFT según slippage y spread."""
    if spread_change < spread_before+(spread_before-spread_before2)*3:
        return True  # Hay posible actividad HFT
    return False

def ajustar_intervalo(x, min_x, max_x, min_y, max_y):
    y = (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y
    return y/100

def execute_trade(spread_before2,spread_before,spread_change):
    """
    Ejecuta la operación real si no se detecta actividad HFT.
    """
    
    if detect_hft_activity(spread_before2,spread_before,spread_change):
        print("\n⚠️ Posible actividad HFT detectada. Ajustando estrategia...",spread_before2,spread_before,spread_change)
        return False  # Evita ejecutar la orden real
    
    # Si no hay actividad sospechosa, ejecuta la orden real
    print("\n✅ Orden real ejecutada sin actividad sospechosa.")

    return True

def order(symbol,entry_price,stop_loss,take_profit,counter,action,type,deviation,lots,position=None,position_by=None):
    """
    https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py

    TRADE_ACTION_DEAL:          Place an order for an instant deal with the specified parameters (set a market order)
    TRADE_ACTION_PENDING:       Place an order for performing a deal at specified conditions (pending order)
    TRADE_ACTION_SLTP:          Change open position Stop Loss and Take Profit
    TRADE_ACTION_MODIFY:        Change parameters of the previously placed trading order
    TRADE_ACTION_REMOVE:        Remove previously placed pending order
    TRADE_ACTION_CLOSE_BY:      Close a position by an opposite one

    deviation (pips)= 1.2
    """
    request = {
        "action": action,
        "symbol": symbol,
        "volume": lots, # lots
        "type": type,
        "price": entry_price,
        "sl": stop_loss, # FLOAT
        "tp": take_profit, # FLOAT
        "deviation": int(deviation*10), # INTERGER
        "magic": 0, # INTERGER
        "comment": f"Order number: {counter}",
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_FOK,
        #"type_filling": mt.ORDER_FILLING_FOK
    }

    if position!=None:
        request["position"] = position

    if position_by!=None:
        request["position_by"] = position_by

    print()
    print(request)
    print()
    order = mt.order_send(request)

    return order,counter+1

def liquidate(deviation):
    for i in mt.positions_get():
        if i.type==1: # venta
            resultado,contador=order(i.symbol,mt.symbol_info_tick(i.symbol).ask,0.0,0.0,0,mt.TRADE_ACTION_DEAL,mt.ORDER_TYPE_BUY,deviation,i.volume,position=i.ticket)
        elif i.type==0: #compra
            resultado,contador=order(i.symbol,mt.symbol_info_tick(i.symbol).bid,0.0,0.0,0,mt.TRADE_ACTION_DEAL,mt.ORDER_TYPE_SELL,deviation,i.volume,position=i.ticket)

def penalty(seconds):
    """Penalizacion para cuando falle"""
    time.sleep(seconds)

def number_to_datetime(number):
    return datetime.fromtimestamp(number / 1e3)

if __name__ == '__main__':
    #print(riesgo_beneficio(8,7,10))
    print(numero_acciones(1000, 7))