import pandas as pd
import numpy as np
import polars as pl
from collections import Counter
from functools import reduce
import heapq

def liquidity(dato, resultados):
    if dato not in resultados:
        resultados[dato] = 1
    else:
        resultados[dato] += 1

def union_liquidez(a, b):
    """for i,j in b.items():
        if i in a:
            a[i]+=j

        else:
            a[i]=j

    return a"""
    a.update((Counter(a) + Counter(b)))
    return a  # Retorna a, pero también lo modifica en memoria

def union_liquidez_2(a, b):
    # Convertimos los diccionarios a DataFrames de Polars
    df_a = pl.DataFrame({'precio': list(a.keys()), 'liquidez': list(a.values())})
    df_b = pl.DataFrame({'precio': list(b.keys()), 'liquidez': list(b.values())})

    # Concatenamos los DataFrames
    df_combined = pl.concat([df_a, df_b])

    # Agrupamos por la columna "precio" y sumamos los valores de "liquidez"
    df_grouped = (
        df_combined.group_by('precio')
        .agg(pl.col('liquidez').sum().alias('liquidez'))
    )

    # Convertimos el DataFrame agrupado de vuelta a un diccionario
    result_dict = dict(zip(df_grouped['precio'], df_grouped['liquidez']))

    # Actualizamos el diccionario original 'a' con los nuevos valores sumados
    a.update(result_dict)

    return a

def all_liquidity(liquidez_diaria, list_days):
    for d in list_days:
        union_liquidez(liquidez_diaria, d)
    return liquidez_diaria
    """liquidez_diaria.update(sum(map(Counter, [liquidez_diaria] + list_days), Counter()))
    return liquidez_diaria"""

"""
def all_liquidity(liquidez_diaria,list_days):
    #la liquidez diaria se calcula con todos los ticks
    #la liquidez semanal se calcula juntando las liquidezes diarias finales

    if len(list_days)>0:
        resultado=list_days.copy()
        for i in list_days:
            union_liquidez(resultado,i)

        union_liquidez(resultado,list_days)
    else:
        resultado=liquidez_diaria

    return resultado
"""

"""def all_liquidity_2(liquidez_diaria,list_days):
    if len(list_days)>0:
        resultado=list_days[0]
        #list_days.append(liquidez_diaria)

        for i in list_days[1:]:
            union_liquidez(resultado,i)
            #resultado=liquidity_to_percentage(resultado)

        #resultado=liquidity_to_percentage(resultado)
        union_liquidez(resultado,liquidez_diaria)
        #resultado=resultado
    else:
        resultado=liquidez_diaria

    return resultado

def simplify_liquidity(liquidity_bid, liquidity_ask, trigger):
    final = {}
    final2 = {}

    for i in set(liquidity_ask) | set(liquidity_bid):
        bid = liquidity_bid.get(i, 0)
        ask = liquidity_ask.get(i, 0)
        valor = abs(bid - ask)
        final[i] = valor
        final2[i] = bid + ask
    
    mayor = max(final.values(), default=1)
    resultado = {punto: valor / mayor for punto, valor in sorted(final.items(), key=lambda x: x[1], reverse=True) if valor / mayor >= trigger}
    
    menor = min(final2.values(), default=1)
    resultado.update({punto: valor for punto, valor in sorted(final2.items(), key=lambda x: x[1]) if menor / valor >= trigger and punto not in resultado})
    
    return resultado, {}"""

def all_liquidity_2(liquidez_diaria, list_days):
    if not list_days:
        return liquidez_diaria

    # Usamos reduce para evitar múltiples llamadas a `union_liquidez`
    resultado = reduce(union_liquidez, list_days)

    union_liquidez(resultado, liquidez_diaria)
    
    return resultado

def simplify_liquidity(liquidity_bid, liquidity_ask, trigger):
    keys = liquidity_bid.keys() | liquidity_ask.keys()

    final = {i: abs((b := liquidity_bid.get(i, 0)) - (a := liquidity_ask.get(i, 0))) for i in keys}
    final2 = {i: b + a for i, (b, a) in zip(keys, zip(map(liquidity_bid.get, keys, [0]*len(keys)), map(liquidity_ask.get, keys, [0]*len(keys))))}

    mayor = max(final.values(), default = 1)
    menor = min(final2.values(), default = 1)

    resultado = {punto: valor / mayor for punto, valor in heapq.nlargest(len(final), final.items(), key=lambda x: x[1]) if valor / mayor >= trigger}

    resultado.update({
        punto: valor for punto, valor in sorted(final2.items(), key=lambda x: x[1])
        if menor / valor >= trigger and punto not in resultado
    })

    return resultado, {}

def simplify_liquidity_2(args):
    liquidity_bid, liquidity_ask, trigger = args

    final = {}
    final2 = {}

    # Unificar claves sin usar `keys = liquidity_bid.keys() | liquidity_ask.keys()`
    for i in liquidity_bid:
        b = liquidity_bid[i]
        a = liquidity_ask.get(i, 0)  # Si no existe en ask, asumimos 0
        final[i] = abs(b - a)
        final2[i] = b + a

    for i in liquidity_ask:
        if i not in final:
            a = liquidity_ask[i]
            b = 0  # No estaba en liquidity_bid, asumimos 0
            final[i] = abs(b - a)
            final2[i] = b + a

    mayor = max(1,max(final.values(), default=1))
    menor = min(final2.values(), default=1)

    resultado = {
        punto: valor / mayor
        for punto, valor in heapq.nlargest(len(final), final.items(), key=lambda x: x[1])
        if valor / mayor >= trigger
    }

    resultado.update({
        punto: valor
        for punto, valor in sorted(final2.items(), key=lambda x: x[1])
        if menor / valor >= trigger and punto not in resultado
    })

    return resultado, {}

def tendencia_actual(args, signal1, signal2, name2, lotaje, peso):
    data_5m = args["all_data_5m"][args['name']]
    data_5m = data_5m[data_5m["time"] <= args["actual_time"]]
    
    estructura = {}
    
    def encontrar_max_min(data, n):
        if len(data) < n:
            return 0
        highs = data['high'].iloc[-n:]
        lows = data['low'].iloc[-n:]
        max_local = highs.max()
        min_local = lows.min()
        prev_high, prev_low = data.iloc[-n-1]['high'], data.iloc[-n-1]['low']
        return 1 if max_local > prev_high else -1 if min_local < prev_low else 0
    
    estructura['5m'] = encontrar_max_min(data_5m, 50)
    
    if -estructura['5m'] != signal1:
        lotaje = lotaje * peso

    return lotaje

def impulse_cv(args, indicador_valor):
    fecha = args['actual_time']
    data_5m_all = args["all_data_5m"][indicador_valor]
    data_5m_all = data_5m_all[data_5m_all["time"] <= fecha]
    data_5m = data_5m_all.iloc[-1]

    data = args["all_data"][indicador_valor]
    data = data[data["time"] <= fecha].iloc[-1]

    umbral_dinamico = 0.5 * data_5m['ATR']

    # Señal de impulso fuerte (1 = alcista, -1 = bajista)
    signal = 0 
    if data_5m['impulso'] > umbral_dinamico:
        signal=1
    elif data_5m['impulso'] < -umbral_dinamico:
        signal=-1

    if signal == 1 and data['power'] == 1 and data_5m['cv_ma_5m'] < 1 and data_5m['cv_ma'] < 1:
        return 1  # Compra fuerte
    elif signal == -1 and data['power'] == -1 and data_5m['cv_ma_5m'] < 1 and data_5m['cv_ma'] < 1:
        return -1  # Venta fuerte
    else:
        return 0

def main():
    import MetaTrader5 as mt
    import matplotlib.pyplot as plt
    from datetime import datetime
    from datetime import timedelta
    from tqdm import tqdm

    mt.initialize()

    login=80284478
    password="*iWcMw3k"
    server="MetaQuotes-Demo"

    mt.login(login, password, server)
    print("loged in")
    
    #today=datetime.today().date()
    fecha_inicio=datetime(2024,5,10)
    fecha_final=datetime(2024,5,12)
    intervalo=2
    trigger_potencia=1.0

    datas=[pd.DataFrame(mt.copy_ticks_range("EURUSD", fecha_inicio+timedelta(i), fecha_final+timedelta(i), mt.COPY_TICKS_ALL)) for i in range(intervalo)]

    tick_data=pd.DataFrame(mt.copy_ticks_range("EURUSD", fecha_inicio, fecha_final+timedelta(intervalo), mt.COPY_TICKS_ALL))

    plt.plot(tick_data["bid"])
    plt.plot(tick_data["ask"])

    tick_data=datas[-1]
    resultado_bid={}
    resultado_ask={}
    print(len(tick_data))
    for _,i in enumerate(tqdm(tick_data.iterrows(),total=len(tick_data))):
        bid=float("%.5f" % i[1]["bid"])
        ask=float("%.5f" % i[1]["ask"])
        liquidity(ask,resultado_ask)
        liquidity(bid,resultado_bid)

    resultado_liquidez_bid=[]
    resultado_liquidez_ask=[]

    inicial,mayor=simplify_liquidity(resultado_bid,resultado_ask,trigger_potencia)
    resultados=[inicial]
    resultados_2=[mayor]
    
    for i in range(len(datas[:-1])):
        datas_list=datas[-(i+1):-1]

        for k in datas_list:
            tick_data=k
            resultado_bid_sub={}
            resultado_ask_sub={}
            print(len(tick_data))
            for _,i in enumerate(tqdm(tick_data.iterrows(),total=len(tick_data))):
                bid=float("%.5f" % i[1]["bid"])
                ask=float("%.5f" % i[1]["ask"])
                liquidity(ask,resultado_ask_sub)
                liquidity(bid,resultado_bid_sub)

            resultado_liquidez_bid.append(resultado_bid_sub)
            resultado_liquidez_ask.append(resultado_ask_sub)

        if len(resultado_liquidez_bid)!=0 or len(resultado_liquidez_ask)!=0:
            final,mayor=simplify_liquidity(all_liquidity_2(resultado_bid,resultado_liquidez_bid),all_liquidity_2(resultado_ask,resultado_liquidez_ask),trigger_potencia)
            print(final)
            resultados.append(final)
            resultados_2.append(mayor)

    liquidez_final={}
    liquidez_final_2={}
    
    for i in resultados:
        for _,j in i.items():
                liquidez_final[_]=j

    for i in resultados_2:
        for _,j in i.items():
                liquidez_final_2[_]=j

    print(liquidez_final)
    print(liquidez_final_2)

    for i,j in liquidez_final.items():
        if j<0:
            plt.axhline(y=i, color='red', linestyle='-', alpha = -j)
        else:
            plt.axhline(y=i, color='green', linestyle='-', alpha = 1.0)

    for i,j in liquidez_final_2.items():
        if j<0:
            plt.axhline(y=i, color='red', linestyle='-', alpha = -j)
        else:
            plt.axhline(y=i, color='red', linestyle='-', alpha = 1.0)

    mt.shutdown()

    plt.show()

if __name__ == '__main__':
    main()

# TODO:
#identificar quien tiene el poder viendo cual ha subido y cual ha bajado