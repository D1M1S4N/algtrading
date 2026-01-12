import MetaTrader5 as mt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime,timedelta,date
import pandas as pd
import numpy as np
import torch
import ast
from dotenv import dotenv_values
from playsound import playsound
from utils import *
from estrategia import MarketMaster,liquidity,simplify_liquidity,all_liquidity_2,calcular_pips
from pre_implemented_strategies.book_maths import Maths
from typing import NamedTuple
import investpy as ivp
from tqdm import tqdm
import time
import warnings
import sys
from data import raw_data_to_df
import traceback
import multiprocessing
import itertools
import optuna
import gc

warnings.filterwarnings("ignore")

#https://www.mql5.com/en/docs/python_metatrader5

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

def mostrar_posiciones(posiciones)->None:
    print(f"\n-------------------- Mostrando {len(posiciones)} Posiciones --------------------\n")
    for i in posiciones:
        print(i)
        try:
            print('Pips tp: ',calcular_pips(i.entrada,i.tp),'Pips sl: ',calcular_pips(i.entrada,i.sl),'RR: ',calcular_pips(i.entrada,i.tp)/calcular_pips(i.entrada,i.sl))
        except:
            print('Pips tp: ',calcular_pips(i.entrada,i.tp),'Pips sl: ',calcular_pips(i.entrada,i.sl),'RR: 100')

def agregar_puntos_liquidez(args):
    ask,bid,liquidez_final,atr_multipliers,curr_spread,atr,atr_ma,data,pos_dia=args
    entrada=0
    if bid in liquidez_final:
        tp_corto=ask-(atr*(atr/atr_ma)*atr_multipliers["all_data_5m"][1])-curr_spread
        tp_largo=bid+(atr*(atr/atr_ma)*atr_multipliers["all_data_5m"][1])+curr_spread
        resultado_corto,resultado_largo=0,0

        for _,dia_para_encontrar_sentido in data.iterrows():
            if dia_para_encontrar_sentido['bid']>=tp_largo:
                resultado_largo='tp'
                break

            elif dia_para_encontrar_sentido['ask']<=tp_corto:
                resultado_corto='tp'
                break
            
        if resultado_largo=='tp':
            mejor_accion=1
            entrada=ask
        elif resultado_corto=='tp':
            mejor_accion=0
            entrada=bid
        else:
            mejor_accion=None


    elif ask in liquidez_final:
        tp_corto=ask-(atr*(atr/atr_ma)*atr_multipliers["all_data_5m"][1])-curr_spread
        tp_largo=bid+(atr*(atr/atr_ma)*atr_multipliers["all_data_5m"][1])+curr_spread

        resultado_corto,resultado_largo=0,0

        for _,dia_para_encontrar_sentido in data.iterrows():
            if dia_para_encontrar_sentido['bid']>=tp_largo:
                resultado_largo='tp'
                break
            elif dia_para_encontrar_sentido['ask']<=tp_corto:
                resultado_corto='tp'
                break

        if resultado_largo=='tp':
            mejor_accion=1
            entrada=ask
        elif resultado_corto=='tp':
            mejor_accion=0
            entrada=bid
        else:
            mejor_accion=None
            
    return mejor_accion,entrada,pos_dia,tp_largo,tp_corto

def recalcular_puntos_paralelismo(list_liquidity_ask,list_liquidity_bid,resultado_liquidez_bid,resultado_liquidez_ask,fecha_busqueda,limite_potencia,pos_liquidity):
    i_bid=list_liquidity_bid[-(pos_liquidity+1):]
    i_ask=list_liquidity_ask[-(pos_liquidity+1):]

    all_liquidity_bid=all_liquidity_2(resultado_liquidez_bid[fecha_busqueda],i_bid)
    all_liquidity_ask=all_liquidity_2(resultado_liquidez_ask[fecha_busqueda],i_ask)
    
    #for pos_liquidity in range(len(list_liquidity_ask)):
        #i_bid=list_liquidity_bid[-(pos_liquidity+1):]
        #i_ask=list_liquidity_ask[-(pos_liquidity+1):]

        #all_liquidity_bid=all_liquidity_2(resultado_liquidez_bid[fecha_busqueda],i_bid)
        #all_liquidity_ask=all_liquidity_2(resultado_liquidez_ask[fecha_busqueda],i_ask)
        #puntos_liquidez.append(simplify_liquidity(all_liquidity_bid,all_liquidity_ask,trigger=limite_potencia))

    return simplify_liquidity(all_liquidity_bid,all_liquidity_ask,trigger=limite_potencia)

def recalcular_puntos(list_liquidity_ask,list_liquidity_bid,resultado_liquidez_bid,resultado_liquidez_ask,fecha_busqueda,limite_potencia):
    puntos_liquidez=[]

    for i in range(len(list_liquidity_ask)):
        puntos_liquidez.append(recalcular_puntos_paralelismo(list_liquidity_ask,list_liquidity_bid,resultado_liquidez_bid,resultado_liquidez_ask,fecha_busqueda,limite_potencia,i))
        
    liquidez_final={}
    salidas_final={}

    for punto_liquidez,puntos_salida in puntos_liquidez:
        for _j,j in punto_liquidez.items():
            #calcular el valor de los indicadores en este punto
            if _j not in liquidez_final:
                liquidez_final[_j]=0

        for _j,j in puntos_salida.items():
            if _j not in salidas_final:
                salidas_final[_j]=j

    return liquidez_final,salidas_final,puntos_liquidez

def get_data_temporality(nombre,temporality,dia,days_back):
    data_5m=pd.DataFrame(mt.copy_rates_range(nombre, temporality, dia-timedelta(days_back),dia+timedelta(1)))
    data_5m=data_5m.drop(["real_volume"],axis=1)
    data_5m['spread']=data_5m["spread"]/100000
    data_5m["mean_price"]=(data_5m["high"]+data_5m["low"]+data_5m["close"])/3
    data_5m["rmv"]=data_5m["mean_price"]*data_5m['tick_volume']
    data_5m["time"]=[datetime.fromtimestamp(item) for item in data_5m["time"]]
    data_5m["diff"]=data_5m["mean_price"].diff().fillna(0)
    data_5m.fillna(0)
    data_5m=data_5m.replace('nan', '0')
    data_5m.iloc[:, 1:] = data_5m.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return data_5m

class Backtesting:
    def __init__(self,dinero_inicial,name,RR,sl_minimo_pips,sl_maximo_pips,divisor_tiempo_limite,tamanyo_break_even,min_spread_trigger,max_spread_trigger,break_even_size,trigger_trailing_tp,trigger_trailing_sl,enable_break_even,enable_dinamic_sl,enable_dinamic_tp,maximo_perdida_diaria,maximo_beneficio_diario,lotaje_maximo,lotaje_minimo,longitud_liquidez,ticks_refresco_liquidez,maximo_operaciones_consecutivas,limite_potencia,bloquear_noticias,fecha_inicio,fecha_final,verbose,sounds,grafico,recavar_datos,save_file,maximo_operaciones_diarias,ema_length,ema_length2,sma_length_2,sma_length_3,length_macd,rsi_roll,rsi_values,atr_window,atr_ma,stoch_rsi,ema_length3,psar_parameters,bollinger_sma,adx_window,obv_ema,ema_length4,sma_length_4,sma_length_5,length_macd2,mfi_length,mfi_values,pvt_length,adl_ema,wr_length,vroc,nvi,momentum,cci,bull_bear_power,mass_index,trix,vortex,z_score_levels,atr_multipliers,re_evaluar,riesgo_por_operacion,dias_retiro,cantidad_retiro,moneda_cuenta,lista_pesos_confirmaciones,var_multiplier,corr_limit,zona_horaria_operable,multiplicador_tp_sl,*args,**kwargs):
        #-------------------Inicialización---------------------------

        account=dotenv_values("account.env")

        mt.initialize()
        
        print("\nInitialized")

        if not mt.initialize():
            raise Error(f'initialize() failed, error code = {mt.last_error()}')

        mt.login(account["login"],password=account["password"],server=account["server"])
        print("Logged in")
        self.timezone=int(int(time.strftime("%z", time.gmtime()))/100)

        #-------------------Configuraciones--------------------------

        self.name=name
        self.RR=RR
        self.sl_minimo_pips=sl_minimo_pips
        self.sl_maximo_pips=sl_maximo_pips
        self.tp_minimo_pips=self.RR*self.sl_minimo_pips
        self.tp_maximo_pips=self.RR*self.sl_maximo_pips
        
        self.tamanyo_break_even=tamanyo_break_even #pips a partir los cuales se aplica break even
        self.break_even_size=break_even_size #pip despues de break even

        self.enable_break_even=enable_break_even
        self.enable_dinamic_sl=enable_dinamic_sl
        self.enable_dinamic_tp=enable_dinamic_tp

        self.multiplicador_tp_sl=multiplicador_tp_sl

        self.re_evaluar=re_evaluar

        self.trigger_trailing_tp=trigger_trailing_tp #pips
        self.trigger_trailing_sl=trigger_trailing_sl #pips

        self.maximo_perdida_diaria=maximo_perdida_diaria
        #self.maximo_perdida_diaria=riesgo_por_operacion*(maximo_operaciones_consecutivas+1)
        self.maximo_beneficio_diario=maximo_beneficio_diario
        self.lotaje_maximo=lotaje_maximo
        self.lotaje_minimo=lotaje_minimo
        self.longitud_liquidez=longitud_liquidez
        self.ticks_refresco_liquidez=ticks_refresco_liquidez
        self.maximo_operaciones_consecutivas=maximo_operaciones_consecutivas
        self.limite_potencia=limite_potencia #potencia menor que esta no sera tomada en cuenta

        self.maximo_operaciones_diarias=maximo_operaciones_diarias

        self.bloquear_noticias=bloquear_noticias

        #-------------------Configuraciones--------------------------

        self.fecha_inicio=fecha_inicio #2023,3,1
        self.fecha_final=fecha_final #2023,6,1

        self.verbose=verbose
        self.sounds=sounds
        self.grafico=grafico
        self.recavar_datos=recavar_datos
        self.save_file=save_file
        self.moneda_cuenta=moneda_cuenta

        self.comisiones=3 # 3 x lote
        self.unidad_lote=100000
        self.multiplicador=100
        self.margin=1
        self.dinero_ultimo_retiro=dinero_inicial
        self.dinero_inicial=dinero_inicial
        self.dinero=self.dinero_inicial
        self.dinero_inicial_diario=self.dinero
        self.max_dinero=self.dinero
        self.min_dinero=self.dinero
        self.perdida_maxima_cuenta=10/100
        self.historial_dinero=[self.dinero_inicial]
        self.positivos_negativos=[0,0]
        self.n_prints_comprobacion_inicial=0
        self.original_min_spread_trigger=min_spread_trigger
        self.original_max_spread_trigger=max_spread_trigger
        self.min_spread_trigger=min_spread_trigger
        self.max_spread_trigger=max_spread_trigger
        self.n_cortos=0
        self.n_largos=0
        self.n_tp=0
        self.n_paradas=0
        self.n_sl=0
        self.n_tp_diario=0
        self.n_sl_diario=0
        self.n_p_diario=0
        self.n_dias=0

        self.retiros=[]
        self.dias_retiro=dias_retiro
        self.cantidad_retiro=cantidad_retiro

        self.comision_por_retiro=95/100
        self.estadisticas_operaciones={0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        self.posiciones=[]

        #----------MarketMasterMaths----------
        self.ema_length=ema_length
        self.ema_length2=ema_length2
        self.ema_length3=ema_length3
        self.ema_length4=ema_length4
        self.sma_length_2=sma_length_2
        self.sma_length_3=sma_length_3
        self.sma_length_4=sma_length_4
        self.sma_length_5=sma_length_5
        self.length_macd=length_macd
        self.length_macd2=length_macd2
        self.rsi_roll=rsi_roll
        self.rsi_values=rsi_values
        self.atr_window=atr_window
        self.atr_ma=atr_ma
        self.stoch_rsi=stoch_rsi
        self.psar_parameters=psar_parameters
        self.bollinger_sma=bollinger_sma
        self.adx_window=adx_window
        self.obv_ema=obv_ema
        self.mfi_length=mfi_length
        self.mfi_values=mfi_values
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
        self.z_score_levels=z_score_levels
        self.atr_multipliers=atr_multipliers
        self.lista_pesos_confirmaciones=lista_pesos_confirmaciones
        self.zona_horaria_operable_original=zona_horaria_operable
        self.divisor_tiempo_limite=divisor_tiempo_limite

        self.var_multiplier=var_multiplier
        self.corr_limit=corr_limit

        #self.riesgo_por_operacion=(dinero_inicial*riesgo_por_operacion)/6000
        self.riesgo_por_operacion=riesgo_por_operacion
        
        self.historial_total_puntos_liquidez={} #fecha,(punto,posicion,dinero,atr,atr_ma) # con estos datos se calcula lotaje,tp,sl en cada uno de los sentidos

        #self.lotaje_maximo=(dinero_inicial*self.lotaje_maximo)/6000
        #self.lotaje_minimo=(dinero_inicial*self.lotaje_minimo)/6000

        super().__init__(*args,**kwargs)

    def run(self)->None:

        simbolos=mt.symbols_get()
        simbolos = [symbol for symbol in simbolos if symbol.visible]

        self.lista_activos=[]

        for i in simbolos:
            self.lista_activos.append(i.name)

        self.lista_activos=['EURUSD', 'GBPUSD', 'USDJPY', 'USDSEK']

        #self.lista_activos = ["EURUSD","USDSEK"] #quitar esto cuando esté listo

        print()

        self.correlation_dict={} # name-name:correlation

        combinaciones = itertools.combinations(self.lista_activos, 2)

        print("Activos invertibles: ",self.lista_activos)
        print("Activos invertibles: ",len(self.lista_activos))

        """
        datos={}

        for i in self.lista_activos:
            datos[i]=pd.DataFrame(mt.copy_rates_range(i, mt.TIMEFRAME_M5, datetime(2024,5, 1),datetime(2024,6,1)))

        
        print("Activos invertibles: ",self.lista_activos)
        print("Activos invertibles: ",len(self.lista_activos))

        total_size_combinations=0
        for activo1, activo2 in combinaciones:
            total_size_combinations+=1
            if len(datos[activo1])==len(datos[activo2]):
                serie1=datos[activo1]['close']
                serie2=datos[activo2]['close']
                serie1_normalizada = ((serie1 - serie1.min()) / (serie1.max() - serie1.min()))
                serie2_normalizada = ((serie2 - serie2.min()) / (serie2.max() - serie2.min()))
                correlacion = serie1_normalizada.corr(serie2_normalizada)
                self.correlation_dict[f"{activo1}-{activo2}"]=(correlacion,np.var(serie1_normalizada),sum(serie1_normalizada)/len(serie1_normalizada),np.var(serie2_normalizada),sum(serie2_normalizada)/len(serie2_normalizada))
                if self.verbose:
                    print(activo1,activo2,self.correlation_dict[f"{activo1}-{activo2}"])

        print(f"\nTamaño diccionario correlaciones {len(self.correlation_dict)}/{total_size_combinations}")

        serie1=datos['EURUSD']['close']
        serie2=datos['USDSEK']['close']
        correlacion,var1,media1,var2,media2=self.correlation_dict[f"EURUSD-USDSEK"]
        print(correlacion,var1,media1,var2,media2)
        serie1_normalizada = ((serie1 - serie1.min()) / (serie1.max() - serie1.min()))
        serie2_normalizada = ((serie2 - serie2.min()) / (serie2.max() - serie2.min()))
        plt.plot(serie1_normalizada,color='blue')
        plt.plot(serie2_normalizada,color='blue')
        plt.axhline(media1+(5*var1),color='red')
        plt.axhline(media2-(5*var2),color='red')
        plt.grid()
        plt.show()"""

        file = open("data/registros/registro_backtesting.txt","w")
        try:
            if self.save_file:
                orig_stdout = sys.stdout
                f = open('out.txt', 'w')
                sys.stdout = f

            if self.break_even_size>=self.tamanyo_break_even:
                raise Error("Tamaño de break even superior al de break even")

            start=time.time()

            list_days=[]
            list_days_5m=[]
            list_liquidity_bid=[]
            list_liquidity_ask=[]

            marketmaster=MarketMaster(self.RR,self.sl_minimo_pips,self.sl_maximo_pips,self.tp_minimo_pips,self.tp_maximo_pips,self.lotaje_maximo,self.lotaje_minimo,self.rsi_values,self.mfi_values,self.z_score_levels,self.atr_multipliers,self.maximo_perdida_diaria,self.maximo_operaciones_consecutivas,self.riesgo_por_operacion,self.dinero_inicial,self.lista_pesos_confirmaciones,self.var_multiplier,self.corr_limit,generate_seed(),self.multiplicador_tp_sl)

            for n in range(int((self.fecha_final-self.fecha_inicio).days)):
                dia=self.fecha_inicio+timedelta(n)
                self.historial_puntos_liquidez=[]
                self.correlation_dict={}
                combinaciones = itertools.combinations(self.lista_activos, 2)
                if dia.weekday() not in [5,6]:
                    all_data={}

                    if (self.historial_dinero[-1]-(self.dinero_inicial))/(self.dinero_inicial) >- self.perdida_maxima_cuenta:
                        for name in tqdm(self.lista_activos,total=len(self.lista_activos)):
                            data=pd.DataFrame(mt.copy_ticks_range(name, dia,dia+timedelta(1), mt.COPY_TICKS_ALL))
                            data['ask_dif']=data['ask'].diff()
                            data['bid_dif']=data['bid'].diff()

                            data['power']=np.select(
                                [data["bid_dif"] > data["ask_dif"], data["bid_dif"] < data["ask_dif"]],
                                ["Compradores", "Vendedores"],
                                default="Equilibrado"
                            )

                            data['spread'] = data['ask'] - data['bid']

                            data=data.drop(columns=["last","volume","time_msc","volume_real"],axis=1)
                            data["time"]=[datetime.fromtimestamp(item) for item in data["time"]]

                            all_data[name] = data

                        data=all_data[self.name]

                    if len(data)!=0:
                        if es_horario_de_verano(dia): # es verano entonces se usa el segundo trigger
                            self.max_spread_trigger=convertir_spread(data['ask'].iloc[0],self.original_max_spread_trigger[1])
                            self.min_spread_trigger=convertir_spread(data['ask'].iloc[0],self.original_min_spread_trigger[1])
                            self.temporada='verano'
                            self.zona_horaria_operable=self.zona_horaria_operable_original[1]
                        else: # es invierno entonces se usa el primer trigger
                            self.max_spread_trigger=convertir_spread(data['ask'].iloc[0],self.original_max_spread_trigger[0])
                            self.min_spread_trigger=convertir_spread(data['ask'].iloc[0],self.original_min_spread_trigger[0])
                            self.temporada='invierno'
                            self.zona_horaria_operable=self.zona_horaria_operable_original[0]

                        print(f"Backtesting con min_spread: {self.min_spread_trigger} y max_spread: {self.max_spread_trigger} y zona horaria: {self.zona_horaria_operable} | Pérdida maxima diaria: {self.maximo_perdida_diaria}")

                    if self.bloquear_noticias and len(data)!=0:
                        if es_horario_de_verano(dia):
                            calendario_base=ivp.economic_calendar(time_zone=f"GMT +1:00",importances=["high"],from_date=f"{(dia-timedelta(days=1)).strftime("%d/%m/%Y")}",to_date=f"{dia.strftime("%d/%m/%Y")}")
                        else:
                            calendario_base=ivp.economic_calendar(time_zone=f"GMT",importances=["high"],from_date=f"{(dia-timedelta(days=1)).strftime("%d/%m/%Y")}",to_date=f"{dia.strftime("%d/%m/%Y")}")

                        try:                        
                            calendario_base=calendario_base.drop(["id","actual","forecast","previous"],axis=1)
                            calendario_base=calendario_base.iloc[np.where(calendario_base["importance"]=="high")]
                            calendario_base=calendario_base.drop(calendario_base[calendario_base['time'] == 'All Day'].index)
                            calendario_base["time"]=[datetime.strptime(item,"%H:%M") for item in calendario_base["time"]]                    
                            calendario=calendario_base
                        except:
                            calendario=None

                    n_prints_comprobacion=0
                    if self.n_prints_comprobacion_inicial!=0:
                        n_prints_comprobacion=len(data)//self.n_prints_comprobacion_inicial

                    if len(data)>0 and (self.historial_dinero[-1]-(self.dinero_inicial))/(self.dinero_inicial) >- self.perdida_maxima_cuenta:
                        market_master_maths=Maths(ema_length=self.ema_length,ema_length2=self.ema_length2,ema_length3=self.ema_length3,ema_length4=self.ema_length4,sma_length_2=self.sma_length_2,sma_length_3=self.sma_length_3,length_macd=self.length_macd,rsi_roll=self.rsi_roll,atr_ma=self.atr_ma,stoch_rsi=self.stoch_rsi,psar_parameters=self.psar_parameters,bollinger_sma=self.bollinger_sma,adx_window=self.adx_window,obv_ema=self.obv_ema,sma_length_4=self.sma_length_4,sma_length_5=self.sma_length_5,length_macd2=self.length_macd2,mfi_length=self.mfi_length,pvt_length=self.pvt_length,adl_ema=self.adl_ema,wr_length=self.wr_length,vroc=self.vroc,nvi=self.nvi,momentum=self.momentum,cci=self.cci,bull_bear_power=self.bull_bear_power,mass_index=self.mass_index,trix=self.trix,vortex=self.vortex)

                        all_data_5m={}
                        all_data_15m={}
                        all_data_1h={}
                        saltar=False

                        for nombre in tqdm(self.lista_activos,total=len(self.lista_activos)):
                            data_5m=get_data_temporality(nombre,mt.TIMEFRAME_M5,dia,0)
                            data_15m=get_data_temporality(nombre,mt.TIMEFRAME_M15,dia,0)
                            data_1h=get_data_temporality(nombre,mt.TIMEFRAME_H1,dia,1)
                            data_5m_time=data_5m["time"]
                            data_1h_time=data_1h["time"]

                            try:
                                data_5m=market_master_maths.df(data_5m)
                                data_1h=market_master_maths.df(data_1h)
                                data_15m=market_master_maths.df(data_15m)
                            except:
                                saltar=True

                            all_data_5m[nombre] = data_5m
                            all_data_15m[nombre] = data_15m
                            all_data_1h[nombre] = data_1h

                        data_5m = all_data_5m[self.name]
                        data_15m = all_data_15m[self.name]
                        data_1h = all_data_1h[self.name]

                        if len(list_days)<self.longitud_liquidez or saltar==True:
                            saltar=True
                        else:
                            saltar=False

                        prints_noticia=[0,0]

                        print()
                        print(f"Start: {data["time"][0]} | End: {data["time"][len(data)-1]}")
                        print(f"Backtesting con {len(data):,} datos y {round(self.dinero,2)}€\n")
                        print("-"*100)
                        print()

                        #if self.bloquear_noticias:
                            #calendario=calendario_base.iloc[np.where(calendario_base["date"]==f"{data["time"][100].date().strftime("%d/%m/%Y")}")]
                        
                        resultado_liquidez_bid={} #fecha->dict(liquidez)
                        resultado_liquidez_ask={}
                        puntos_liquidez=[]

                        anterior_bid,anterior_ask=0,0

                        self.n_dias+=1
                        recalcular=False
                        contador_data_5m=0
                        anterior_contador_5m=None

                        datos={}

                        for i in self.lista_activos:
                            datos[i]=pd.DataFrame(mt.copy_rates_range(i, mt.TIMEFRAME_M5, dia-timedelta(4),dia)) # coger datos hasta el dia actual sin incluirlo para que no sea trampa
                            datos[i]["time"]=[datetime.fromtimestamp(item) for item in datos[i]["time"]]

                        #print(self.correlation_dict,combinaciones,self.lista_activos)
                        total_size_combinations=0
                        for activo1, activo2 in combinaciones:
                            total_size_combinations+=1
                            if len(datos[activo1])==len(datos[activo2]):
                                serie1=datos[activo1]['close']
                                serie2=datos[activo2]['close']
                                serie1_normalizada = ((serie1 - serie1.min()) / (serie1.max() - serie1.min()))
                                serie2_normalizada = ((serie2 - serie2.min()) / (serie2.max() - serie2.min()))
                                correlacion = serie1_normalizada.corr(serie2_normalizada)
                                self.correlation_dict[f"{activo1}-{activo2}"]=(correlacion,np.var(serie1_normalizada),sum(serie1_normalizada)/len(serie1_normalizada),np.var(serie2_normalizada),sum(serie2_normalizada)/len(serie2_normalizada))
                                #if self.verbose:
                                    #print(self.correlation_dict[f"{activo1}-{activo2}"])

                        self.correlation_dict = {clave: valor for clave, valor in self.correlation_dict.items() if clave.startswith(self.name)}
                        #print(self.correlation_dict)
                        args={"correlation_dict":self.correlation_dict,"recalcular":False,"all_data":all_data,"all_data_5m":all_data_5m,"all_data_1h":all_data_1h,"all_data_15m":all_data_15m,"name":self.name,"temporada":self.temporada}

                        for pos_dia,i in enumerate(tqdm(data.iterrows(),total=len(data))):
                            if n_prints_comprobacion!=0 and pos_dia!=0 and pos_dia%n_prints_comprobacion==0:
                                print(f"\nHora: {i[1]["time"].time()} | stop loss: {self.n_sl_diario} | take profits: {self.n_tp_diario} | Número operaciones por cerrar: {len(self.posiciones_sl_cortos)+len(self.posiciones_sl_largos)}")

                            fecha=i[1]["time"]
                            hora=fecha.time()

                            fecha_busqueda=f"{fecha.day}/{fecha.month}/{fecha.year}"
                            bid=float("%.5f" % i[1]["bid"])
                            ask=float("%.5f" % i[1]["ask"])
                            nuevas_entradas=[]

                            curr_spread=ask-bid

                            if fecha_busqueda not in resultado_liquidez_ask and fecha_busqueda not in resultado_liquidez_bid:
                                resultado_liquidez_bid={str(fecha_busqueda):{}}
                                resultado_liquidez_ask={str(fecha_busqueda):{}}

                            if not saltar:
                                try:
                                    actual,siguiente=data_5m_time[contador_data_5m],data_5m_time[contador_data_5m+1]
                                    
                                    if not (actual.hour==hora.hour and actual.minute<=hora.minute<siguiente.minute):    
                                        if siguiente.minute==0:
                                            if not (actual.hour==hora.hour and actual.minute<=hora.minute<(siguiente.minute+60)):
                                                contador_data_5m+=1

                                        else:
                                            contador_data_5m+=1
                                except:
                                    pass

                                if anterior_contador_5m!=contador_data_5m:
                                    data_5m_actual=data_5m[:contador_data_5m+1]
                                    limite_spread = (self.max_spread_trigger +data_5m_actual['adjusted_atr'] * (self.min_spread_trigger - self.max_spread_trigger)).values[-1]
                                    anterior_contador_5m=contador_data_5m

                                for indice_posicion,posicion in enumerate(self.posiciones):
                                    if getattr(posicion,"resultado")==0: #operacion sin cerrar
                                        if getattr(posicion,"typo")==1: #largo
                                            if bid>=getattr(posicion,"tp"): #tp
                                                recalcular=True
                                                self.n_tp+=1
                                                self.n_tp_diario+=1
                                                #self.dinero+=(((bid*self.unidad_lote*getattr(posicion,"lotaje")))-(self.comisiones*getattr(posicion,"lotaje")))
                                                self.dinero+=((bid-getattr(posicion,"entrada"))*self.unidad_lote*getattr(posicion,"lotaje"))-(self.comisiones*getattr(posicion,"lotaje"))
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(resultado=1)
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(fecha_resultado=fecha)
                                                
                                                if self.verbose:
                                                    print()
                                                    print(bid," tp largo hora: ",fecha.hour,fecha.minute,fecha.second,self.dinero)

                                            elif bid<=getattr(posicion,"sl"): #sl
                                                recalcular=True
                                                if posicion.entrada-posicion.sl>0:
                                                    self.n_sl+=1
                                                    self.n_sl_diario+=1
                                                else:
                                                    self.n_tp+=1
                                                    self.n_tp_diario+=1
                                                
                                                #self.dinero+=(((bid*self.unidad_lote*getattr(posicion,"lotaje")))-(getattr(posicion,"lotaje")*self.comisiones))
                                                self.dinero+=((bid-getattr(posicion,"entrada"))*self.unidad_lote*getattr(posicion,"lotaje"))-(self.comisiones*getattr(posicion,"lotaje"))
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(resultado=-1)
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(fecha_resultado=fecha)

                                                if self.verbose:
                                                    print()
                                                    print(bid," sl largo hora: ",fecha.hour,fecha.minute,fecha.second,self.dinero)

                                        else: #corto
                                            if ask<=getattr(posicion,"tp"): #tp
                                                recalcular=True
                                                self.n_tp+=1
                                                self.n_tp_diario+=1
                                                #self.dinero-=(((ask*self.unidad_lote*getattr(posicion,"lotaje")))+(getattr(posicion,"lotaje")*self.comisiones))
                                                self.dinero+=((getattr(posicion,"entrada")-ask)*self.unidad_lote*getattr(posicion,"lotaje"))-(self.comisiones*getattr(posicion,"lotaje"))
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(resultado=1)
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(fecha_resultado=fecha)

                                                if self.verbose:
                                                    print()
                                                    print(ask," tp corto hora: ",fecha.hour,fecha.minute,fecha.second,self.dinero)

                                            elif ask>=getattr(posicion,"sl"): #sl
                                                #print(self.dinero)
                                                recalcular=True
                                                if posicion.entrada-posicion.sl>0:
                                                    self.n_tp+=1
                                                    self.n_tp_diario+=1
                                                else:
                                                    self.n_sl+=1
                                                    self.n_sl_diario+=1

                                                #self.dinero-=(((ask*self.unidad_lote*getattr(posicion,"lotaje")))+(getattr(posicion,"lotaje")*self.comisiones))
                                                self.dinero+=((getattr(posicion,"entrada")-ask)*self.unidad_lote*getattr(posicion,"lotaje"))-(self.comisiones*getattr(posicion,"lotaje"))
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(resultado=-1)
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(fecha_resultado=fecha)

                                                if self.verbose:
                                                    print()
                                                    print(ask," sl corto hora: ",fecha.hour,fecha.minute,fecha.second,self.dinero)

                                valor_dinero=0
                                margen_total=0
                                contador=0
                                posiciones_cortos=list(i for i in self.posiciones if getattr(i,"resultado")==0 and getattr(i,"typo")==-1)
                                posiciones_largos=list(i for i in self.posiciones if getattr(i,"resultado")==0 and getattr(i,"typo")==1)

                                for posicion in self.posiciones:
                                    if getattr(posicion,"resultado")==0:
                                        contador+=1
                                        if getattr(posicion,"typo")==1: #largo
                                            valor_dinero+=((bid-getattr(posicion,"entrada"))*self.unidad_lote*getattr(posicion,"lotaje"))-(self.comisiones*getattr(posicion,"lotaje"))
                                            margen_total+=mt.order_calc_margin(mt.ORDER_TYPE_BUY, self.name, getattr(posicion,"lotaje"), getattr(posicion,"entrada"))
                                        else: #corto
                                            valor_dinero+=((getattr(posicion,"entrada")-ask)*self.unidad_lote*getattr(posicion,"lotaje"))-(self.comisiones*getattr(posicion,"lotaje"))
                                            margen_total+=mt.order_calc_margin(mt.ORDER_TYPE_SELL, self.name, getattr(posicion,"lotaje"), getattr(posicion,"entrada"))

                                valor_dinero+=self.dinero

                                #if (((valor_dinero-self.dinero_inicial_diario)/(self.dinero_inicial_diario/self.multiplicador))>-self.maximo_perdida_diaria and ((valor_dinero-self.dinero_inicial_diario)/(self.dinero_inicial_diario/self.multiplicador))<self.maximo_beneficio_diario and len(list_days)>=self.longitud_liquidez) and fecha.hour<=20:
                                if (((valor_dinero-self.dinero_inicial_diario)/(self.dinero_inicial_diario))>-self.maximo_perdida_diaria and len(list_days)>=self.longitud_liquidez):
                                    operar=True
                                    for _i,_i_ in enumerate(self.posiciones):
                                        if getattr(_i_,"resultado")==0:
                                            valor_trailing_tp=marketmaster.valor_pip(self.trigger_trailing_tp,bid)
                                            valor_trailing_sl=marketmaster.valor_pip(self.trigger_trailing_sl,bid)
                                            if getattr(_i_,"typo")==1: # largo
                                                if self.enable_break_even and getattr(_i_,"sl")<getattr(_i_,"entrada") and bid>=getattr(_i_,"entrada")+(getattr(_i_,"tp")-getattr(_i_,"entrada"))/2:
                                                    self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"entrada")+marketmaster.valor_pip(self.break_even_size,getattr(_i_,"entrada")))
                                                    if self.verbose:
                                                        print("\nbe largo")

                                                #elif getattr(_i_,"sl")>=getattr(_i_,"entrada") and self.enable_dinamic_sl and anterior_bid<bid:
                                                elif self.enable_dinamic_sl and anterior_bid<bid and bid>(_i_.sl+valor_trailing_sl) and not (_i_.tp-valor_trailing_tp)<=bid<=_i_.tp: # añadir condicion para sl trigger
                                                    if self.verbose:
                                                        print("dinamic sl largo")

                                                    self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"sl")+(bid-anterior_bid))

                                                #elif self.enable_dinamic_tp and anterior_bid<bid and getattr(_i_,"sl")>=getattr(_i_,"entrada") and (_i_.tp-valor_trailing_tp)<=bid<=_i_.tp:
                                                #elif self.enable_dinamic_tp and getattr(_i_,"sl")>=getattr(_i_,"entrada") and (_i_.tp-valor_trailing_tp)<=bid<=_i_.tp:
                                                elif self.enable_dinamic_tp and (_i_.tp-valor_trailing_tp)<=bid<=_i_.tp:
                                                    if self.verbose:
                                                        print("dinamic tp largo")
                                                        
                                                    self.posiciones[_i]=self.posiciones[_i]._replace(tp=getattr(_i_,"tp")+(bid-anterior_bid))
                                                    self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"tp")-valor_trailing_tp)

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
                                                if self.enable_break_even and getattr(_i_,"sl")>getattr(_i_,"entrada") and ask<=getattr(_i_,"entrada")-(getattr(_i_,"entrada")-getattr(_i_,"tp"))/2:
                                                    self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"entrada")-marketmaster.valor_pip(self.break_even_size,getattr(_i_,"entrada")))
                                                    if self.verbose:
                                                        print("\nbe corto")
                                                
                                                #elif getattr(_i_,"sl")<=getattr(_i_,"entrada") and self.enable_dinamic_sl and anterior_ask>ask:
                                                elif self.enable_dinamic_sl and anterior_ask>ask and ask<(_i_.sl-valor_trailing_sl) and not (_i_.tp<=ask<=(_i_.tp+valor_trailing_tp)):
                                                    self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"sl")+(ask-anterior_ask))
                                                    if self.verbose:
                                                        print("dinamic sl corto")

                                                #elif self.enable_dinamic_tp and anterior_bid<bid and getattr(_i_,"sl")<=getattr(_i_,"entrada") and (_i_.tp<=ask<=(_i_.tp+valor_trailing_tp)):
                                                #elif self.enable_dinamic_tp and getattr(_i_,"sl")<=getattr(_i_,"entrada") and (_i_.tp<=ask<=(_i_.tp+valor_trailing_tp)):
                                                elif self.enable_dinamic_tp and (_i_.tp<=ask<=(_i_.tp+valor_trailing_tp)):
                                                    if self.verbose:
                                                        print("dinamic tp corto")
                                                    self.posiciones[_i]=self.posiciones[_i]._replace(tp=getattr(_i_,"tp")+(ask-anterior_ask))
                                                    self.posiciones[_i]=self.posiciones[_i]._replace(sl=getattr(_i_,"tp")+valor_trailing_tp)

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
                                    if len(posiciones_largos)>0 or len(posiciones_cortos)>0:
                                        self.dinero=valor_dinero
                                        
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

                                        self.n_p_diario=contador
                                        self.n_paradas+=self.n_p_diario

                                        for pos_entrada,entrada in enumerate(self.posiciones):
                                            if getattr(entrada,"resultado")==0:
                                                self.posiciones[pos_entrada]=self.posiciones[pos_entrada]._replace(resultado=ask)
                                                self.posiciones[pos_entrada]=self.posiciones[pos_entrada]._replace(fecha_resultado=fecha)

                                        operar=False
                                        #saltar=True
                                
                                if not (curr_spread<=limite_spread and self.zona_horaria_operable[0]<=fecha.hour<=self.zona_horaria_operable[1] and not saltar): #163.62
                                    operar=False

                                if pos_dia%self.ticks_refresco_liquidez==0 or len(puntos_liquidez)==0 or recalcular:
                                    recalcular=False

                                    liquidez_final,salidas_final,puntos_liquidez=recalcular_puntos(list_liquidity_ask,list_liquidity_bid,resultado_liquidez_bid,resultado_liquidez_ask,fecha_busqueda,self.limite_potencia)

                                    if len(liquidez_final)!=0:
                                        if self.verbose:
                                            print("\nRecalculando liquidez")
                                            print(liquidez_final)

                            if not saltar and operar and (len(posiciones_cortos)+len(posiciones_largos))<=self.maximo_operaciones_consecutivas and len(self.posiciones)<=self.maximo_operaciones_diarias and (bid in liquidez_final or ask in liquidez_final):
                                #comprobar si operaciones contrarias
                                #if len(posiciones_cortos+posiciones_largos)!=0:
                                    #if any(x.typo==1 and x.resultado==0 for x in posiciones_cortos+posiciones_largos) and any(x.typo==-1 and x.resultado==0 for x in posiciones_cortos+posiciones_largos):
                                        #raise Error('Posiciones contrarias')

                                #comprobar si noticia
                                if self.bloquear_noticias and type(calendario)!=type(None) and len(calendario)>0:
                                    coincidencias=list(i for i in calendario["time"] if (i-timedelta(minutes=15)).time()<=hora<=(i+timedelta(minutes=5)).time())
                                    #no operar 15 minutos antes ni 15 minutos despues

                                    if len(coincidencias)>0:
                                        for coincidencia in coincidencias:
                                            calendario = calendario[calendario.time != coincidencia]

                                        if self.verbose and not prints_noticia[0]:
                                            print("\nRango noticia, no operar")
                                            prints_noticia[0]=1

                                        operar=False

                                    else:
                                        if self.verbose and not prints_noticia[1] and prints_noticia[0]:
                                            prints_noticia[1]=1
                                            print("\nFuera noticia operar")

                                        operar=True

                                if self.recavar_datos:
                                    action,sl,tp,lotaje,accion=0,0,0,0,0
                                else:
                                    """
                                    args={"dinero":[self.dinero_inicial,valor_dinero,self.dinero_inicial_diario], "puntos_liquidez":liquidez_final,
                                    "puntos_salida":salidas_final,"posiciones_cortos":posiciones_cortos,"posiciones_largos":posiciones_largos,
                                    "correlation_dict":self.correlation_dict,"recalcular":False,
                                    "pos_dia":pos_dia,"all_data":all_data,"all_data_5m":all_data_5m,"actual_time":fecha}
                                    """
                                    
                                    args["dinero"]=[self.dinero_inicial,valor_dinero,self.dinero_inicial_diario]
                                    args["puntos_liquidez"]=liquidez_final
                                    args["puntos_salida"]=salidas_final
                                    args["posiciones_cortos"]=posiciones_cortos
                                    args["posiciones_largos"]=posiciones_largos
                                    args["pos_dia"]=pos_dia
                                    args["actual_time"]=fecha
                                    args["ask"] = ask
                                    args["bid"] = bid

                                    try:
                                        #es mejor poner las operaciones de menor temporalidad en la lista primero
                                        args['temporality']="all_data_5m"
                                        action,sl,tp,lotaje,accion=marketmaster.run(args)
                                        if action!=0:
                                            entradas_cortos=list(i.entrada for i in posiciones_cortos)
                                            entradas_largos=list(i.entrada for i in posiciones_largos)
                                            if ask in entradas_largos or bid in entradas_cortos:
                                                lotaje=lotaje/2

                                            nuevas_entradas.append((action,sl,tp,lotaje,"all_data_5m"))
                                        
                                        """args['temporality']="all_data_15m"
                                        action,sl,tp,lotaje,accion=marketmaster.run(args)
                                        if action!=0:
                                            #print("1 hora action: ",action,ask,sl,tp,lotaje)
                                            entradas_cortos=list(i.entrada for i in posiciones_cortos)
                                            entradas_largos=list(i.entrada for i in posiciones_largos)
                                            if ask in entradas_largos or bid in entradas_cortos:
                                                lotaje=lotaje/2

                                            nuevas_entradas.append((action,sl,tp,lotaje,"all_data_15m"))"""

                                        """args['temporality']="all_data_1h"
                                        action,sl,tp,lotaje,accion=marketmaster.run(args)
                                        if action!=0:
                                            #print("1 hora action: ",action,ask,sl,tp,lotaje)
                                            entradas_cortos=list(i.entrada for i in posiciones_cortos)
                                            entradas_largos=list(i.entrada for i in posiciones_largos)
                                            if ask in entradas_largos or bid in entradas_cortos:
                                                lotaje=lotaje/2

                                            nuevas_entradas.append((action,sl,tp,lotaje,"all_data_1h"))"""

                                    except Exception as e:
                                        print("Error en marketmaster")
                                        print(f"Error: {e}")
                                        traceback.print_exc()
                                        action,sl,tp,lotaje,accion=0,0,0,0,0

                                if self.re_evaluar and accion!=0 and (len(posiciones_cortos)!=0 or len(posiciones_largos)!=0):
                                    for indice_posicion,posicion_re_evaluar in enumerate(self.posiciones):
                                        if posicion_re_evaluar.typo==-1 and accion==1 and posicion_re_evaluar.resultado==0:
                                            #recalcular=True
                                            if posicion_re_evaluar.entrada-ask>0:
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(tp=posicion_re_evaluar.entrada)
                                            else:
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(sl=posicion_re_evaluar.entrada)
                                            
                                            self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(fecha_resultado=fecha)

                                            if self.verbose:
                                                print('posición corta cerrada por re evaluación')

                                        elif posicion_re_evaluar.typo==1 and accion==-1 and posicion_re_evaluar.resultado==0:
                                            #recalcular=True
                                            if posicion_re_evaluar.entrada-ask>0:
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(sl=posicion_re_evaluar.entrada)
                                            else:
                                                self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(tp=posicion_re_evaluar.entrada)
                                            
                                            self.posiciones[indice_posicion]=self.posiciones[indice_posicion]._replace(fecha_resultado=fecha)

                                            if self.verbose:
                                                print('posición larga cerrada por re evaluación')

                                for action,sl,tp,lotaje,data_temporality in nuevas_entradas:
                                    #if action==1 and lotaje>=(self.lotaje_minimo*valor_dinero)/self.dinero_inicial:
                                    if action==1 and lotaje>=self.lotaje_minimo:
                                        recalcular=True
                                        self.n_largos+=1
                                        margin_necesary = mt.order_calc_margin(mt.ORDER_TYPE_BUY, self.name, lotaje, ask)
                                        self.estadisticas_operaciones[fecha.weekday()]+=1

                                        tp=float('%.5f' % tp)
                                        sl=float('%.5f' % sl)

                                        if margin_necesary<(self.dinero_inicial_diario-margen_total):
                                            limite_tiempo=calcular_tiempo_limite(tp,ask,data_5m_actual['atr'].iloc[-1])/self.divisor_tiempo_limite
                                            #self.posiciones.append(Posicion(self.name,fecha,action,ask,lotaje,tp,sl,tp,sl,0,0,limite_tiempo))
                                            self.posiciones.append(Posicion(self.name,fecha,action,ask,lotaje,tp,sl,tp,sl,0,0,data_temporality))
                                            file.write(f"\ncrear largo entrada:{round(ask,4)} sl:{round(sl,4)} tp:{round(tp,4)} lotaje: {lotaje} a las {fecha}")

                                            if self.verbose:
                                                print(f"\ncrear largo entrada:{ask} sl:{sl} tp:{tp} lotaje: {lotaje} a las {fecha}")
                        
                                    #if action==-1 and lotaje>=(self.lotaje_minimo*valor_dinero)/self.dinero_inicial:
                                    if action==-1 and lotaje>=self.lotaje_minimo:
                                        recalcular=True
                                        self.n_cortos+=1
                                        margin_necesary = mt.order_calc_margin(mt.ORDER_TYPE_SELL, self.name, lotaje, bid)
                                        self.estadisticas_operaciones[fecha.weekday()]+=1

                                        tp=float('%.5f' % tp)
                                        sl=float('%.5f' % sl)

                                        if margin_necesary<(self.dinero_inicial_diario-margen_total):
                                            limite_tiempo=calcular_tiempo_limite(tp,bid,data_5m_actual['atr'].iloc[-1])/self.divisor_tiempo_limite
                                            #self.posiciones.append(Posicion(self.name,fecha,action,bid,lotaje,tp,sl,tp,sl,0,0,limite_tiempo))
                                            self.posiciones.append(Posicion(self.name,fecha,action,bid,lotaje,tp,sl,tp,sl,0,0,data_temporality))
                                            file.write(f"\ncrear corto entrada:{round(bid,4)} sl:{round(sl,4)} tp:{round(tp,4)} lotaje: {lotaje} a las {fecha}")

                                            if self.verbose:
                                                print(f"\ncrear corto entrada:{bid} sl:{sl} tp:{tp} lotaje: {lotaje} a las {fecha}")

                            liquidity(bid,resultado_liquidez_bid[fecha_busqueda])
                            liquidity(ask,resultado_liquidez_ask[fecha_busqueda])

                            if (not saltar and operar) and (bid in liquidez_final or ask in liquidez_final):
                                recalcular=True
                                self.historial_puntos_liquidez.append((ask,bid,liquidez_final,self.atr_multipliers,curr_spread,data_5m_actual.iloc[-1]['atr'],data_5m_actual.iloc[-1]['atr_ma'],data[pos_dia+1:],pos_dia))

                            if bid!=anterior_bid:
                                anterior_bid=bid

                            if ask!=anterior_ask:
                                anterior_ask=ask

                        """dinero_a_retirar=self.comision_por_retiro*self.dinero*self.cantidad_retiro
                        if (n-self.longitud_liquidez+1)%self.dias_retiro==0 and (self.dinero-dinero_a_retirar)>self.dinero_ultimo_retiro:
                            print("Retiro de ",dinero_a_retirar)
                            self.retiros.append(dinero_a_retirar)
                            self.dinero-=dinero_a_retirar
                            self.dinero_ultimo_retiro=self.dinero"""
                        
                        if (n-self.longitud_liquidez+1)%self.dias_retiro==0:
                            dinero_a_retirar = retiro(self.dinero,self.dinero_ultimo_retiro,self.comision_por_retiro,self.cantidad_retiro)
                            if dinero_a_retirar!=0:
                                print("Retiro de ",dinero_a_retirar)
                                self.retiros.append(dinero_a_retirar)
                                self.dinero-=dinero_a_retirar
                                self.dinero_ultimo_retiro=self.dinero

                        rentabilidad_diaria=(self.dinero-self.dinero_inicial_diario)
                        self.historial_dinero.append(self.dinero)

                        if len(self.posiciones)!=0:
                            mostrar_posiciones(self.posiciones)

                        print(f"\nEn el día {dia.date()} se han tenido {self.n_tp_diario} tp y {self.n_sl_diario} sl y {self.n_p_diario} paradas | Dinero: {round(self.historial_dinero[-1],2)} | Beneficio: {round(rentabilidad_diaria,2)} | Número dia: {self.n_dias-self.longitud_liquidez}")

                        self.dinero_inicial_diario=self.historial_dinero[-1]
                        self.dinero=self.dinero_inicial_diario

                        self.n_tp_diario=0
                        self.n_sl_diario=0
                        self.n_p_diario=0

                        if len(self.historial_dinero)>1:
                            if self.historial_dinero[-1]>self.historial_dinero[-2]:
                                self.positivos_negativos[0]+=1
                            elif self.historial_dinero[-1]<self.historial_dinero[-2]:
                                self.positivos_negativos[1]+=1
                        else:
                            if self.historial_dinero[-1]>self.dinero_inicial:
                                self.positivos_negativos[0]+=1
                            elif self.historial_dinero[-1]<self.dinero_inicial:
                                self.positivos_negativos[1]+=1

                        list_days.append(data)
                        list_days_5m.append(data_5m)
                        list_liquidity_bid.append(resultado_liquidez_bid[fecha_busqueda])
                        list_liquidity_ask.append(resultado_liquidez_ask[fecha_busqueda])

                        if len(list_liquidity_bid)>self.longitud_liquidez:
                            del list_liquidity_bid[0]
                            del list_liquidity_ask[0]
                            del list_days[0]

                        if len(list_days_5m)>1:
                            del list_days_5m[0]

                        if not (len(list_days)==len(list_liquidity_ask)==len(list_liquidity_bid)):
                            raise Error("")

                        if len(self.historial_puntos_liquidez)!=0 and self.recavar_datos:
                            pool=multiprocessing.Pool(5)
                            resultados=pool.map(agregar_puntos_liquidez,self.historial_puntos_liquidez)
                            for i in resultados:
                                mejor_accion,entrada,pos_dia,tp_largo,tp_corto=i
                                if entrada!=0:
                                    if str(dia.date()) not in self.historial_total_puntos_liquidez:
                                        self.historial_total_puntos_liquidez[str(dia.date())]=[(mejor_accion,entrada,pos_dia,tp_largo,tp_corto,data_5m_actual.iloc[-1].tolist()[1:])]
                                    else:
                                        self.historial_total_puntos_liquidez[str(dia.date())].append((mejor_accion,entrada,pos_dia,tp_largo,tp_corto,data_5m_actual.iloc[-1].tolist()[1:]))

                        #if len(self.posiciones)!=0 and self.grafico:
                        if self.grafico:
                            try:
                                plt.plot(data[["ask","bid"]])

                                for i in self.posiciones:
                                    x=[data.index[data['time']==getattr(i,"fecha")].tolist()[0],data.index[data['time']==getattr(i,"fecha_resultado")].tolist()[0]]
                                    y=[getattr(i,"entrada")]    

                                    if getattr(i,"resultado")==1:
                                        y.append(getattr(i,"tp"))

                                    elif getattr(i,"resultado")==-1:
                                        y.append(getattr(i,"sl"))

                                    else:
                                        y.append(getattr(i,"resultado"))

                                    plt.plot(x,y,'--bo',color="green" if i.typo==1 else "red",marker='^' if i.typo==1 else 'v')         

                                plt.show()

                            except:
                                pass

                        lista_borrar=[]
                        for posicion in self.posiciones:
                            if posicion.resultado!=0:
                                lista_borrar.append(posicion)

                        for borrar_posicion in lista_borrar:
                            self.posiciones.remove(borrar_posicion)

                    else:
                        pass

            if self.recavar_datos:
                dataframe=raw_data_to_df(self.historial_total_puntos_liquidez)
                dataframe.to_csv(f'data/datasets/dataframe_{self.name}_{self.fecha_inicio.date()}_{self.fecha_final.date()}.csv')

            mt.shutdown()

            rentabilidad_diaria=(self.dinero-self.dinero_inicial_diario)

            self.historial_dinero.append(self.dinero)

            self.dinero=self.historial_dinero[-1]

            if self.n_dias!=0:
                print()
                print("-"*100)
                print()

                if  ((self.dinero-self.dinero_inicial)/self.dinero_inicial) <= - self.perdida_maxima_cuenta:
                    print("Cuenta quemada :(")
                    print()

                print(f"Tiempo de ejecucion: {round((time.time()-start)/60,2)} minutos")
                print(f"{self.n_dias} días analizados | Dias positivos: {self.positivos_negativos[0]} | Dias negativos: {self.positivos_negativos[1]} | Dias neutros: {self.n_dias-sum(self.positivos_negativos)}")
                print(f"Maximo dinero: {round(max(self.historial_dinero),2)} | Minimo dinero: {round(min(self.historial_dinero),2)}")
                print(f"Dinero inicial: {self.dinero_inicial}€ | dinero final: {round(self.dinero,2)}€ | Beneficio: {round(self.dinero-self.dinero_inicial,2)} ({round(((self.dinero-self.dinero_inicial)/self.dinero_inicial)*100,2)}%)")
                print(f"Numero operaciones: {self.n_cortos+self.n_largos} | Media operaciones por día: {int(round((self.n_cortos+self.n_largos)/self.n_dias,0))} | numero cortos: {self.n_cortos} | numero largos: {self.n_largos}")

                print(f"Total profits: {self.n_tp} | Total stops: {self.n_sl} | Total paradas: {self.n_paradas}")
                print("Retiros: ",self.retiros)
                print(f"{round(sum(self.retiros),2)}€ ganado en {len(self.retiros)} retiros")

                keys=["Lunes","Martes","Miercoles","Jueves","Viernes","Sábado","Domingo"]
                values=list(self.estadisticas_operaciones.values())
                self.estadisticas_operaciones=dict(zip(keys, values))
                print(f"Relación dia semana y numero operaciones: {self.estadisticas_operaciones}")

                mejor,dinero_inicio_perdida=maximo_perdida(self.historial_dinero)
                print('Máximo pérdida consecutiva: ',round(100*(mejor/dinero_inicio_perdida),2),'%')
                print(f"RR (operaciones): {self.n_tp/self.n_sl} | Winrate: {(self.n_tp/self.n_sl)/((self.n_tp/self.n_sl)+1)}%")

                print()

                if self.sounds:
                    playsound('data\\ok.mp3')

                if len(self.historial_dinero)>1:
                    if self.dinero>self.dinero_inicial:
                        color="green"
                    else:
                        color="red"
                    
                    self.historial_dinero=self.historial_dinero[self.longitud_liquidez:-1]
                    poly_fn=np.poly1d(np.polyfit(list(range(1,len(self.historial_dinero)+1)),self.historial_dinero,1))
                    coeficiente_reescalado=self.dinero_inicial-poly_fn(0)
                    plt.plot(self.historial_dinero,"o",color=color)
                    plt.plot(list(range(0,len(self.historial_dinero))),self.historial_dinero,coeficiente_reescalado+poly_fn(list(range(0,len(self.historial_dinero)))),"--k",color=color)
                    try:
                        red_patch = mpatches.Patch(color=color, label=f'Money Plot (Inclinación: {round(np.polyfit(list(range(1,len(self.historial_dinero)+1)),self.historial_dinero,1)[0],2)}) Rendimiento: {round(((self.dinero-self.dinero_inicial)/self.dinero_inicial)*100,2)}% Ratio: {self.positivos_negativos[0]} positivos - {self.positivos_negativos[1]} negativos ({round(self.positivos_negativos[0]/self.positivos_negativos[1],2)}) RR (operaciones): {self.n_tp/self.n_sl}')
                    except:
                        pass

                    plt.legend(handles=[red_patch],loc='upper center',shadow=True,fancybox=True)
                    plt.savefig('data/resultados/resultados_backtest.png')
                    plt.show()

                    plt.plot(np.diff(np.array(self.historial_dinero)))
                    plt.show()

        except BaseException as e:
            if self.sounds:
                playsound('data\\error.mp3')

            print(f"\nError: [{e!r}]\n")
            traceback.print_exc()

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

            try:
                rr=self.n_tp/self.n_sl
                return 0.7*(rr/(rr+1))+0.3*((self.dinero-self.dinero_inicial)/self.dinero_inicial)
            except:
                return 0

def main(trial):

    #maximo_perdida_diaria = trial.suggest_float('maximo_perdida_diaria', 0.01, 0.04)
    #riesgo_por_operacion = trial.suggest_float('riesgo_por_operacion', 0.001, 0.01)
    maximo_perdida_diaria=1.2/100
    riesgo_por_operacion={'corto':0.68/100,'largo':0.68/100,'invierno':0.68/100,'verano':0.68/100}
    lotaje_minimo = 0.2#trial.suggest_float('lotaje_minimo', 0.1, 2.0)
    lotaje_maximo = 1.4#trial.suggest_float('lotaje_maximo', lotaje_minimo, 2.0)
    #maximo_operaciones_consecutivas = trial.suggest_int('maximo_operaciones_consecutivas', 1, 6)
    maximo_operaciones_consecutivas=1
    var_multiplier = 1#trial.suggest_float('var_multiplier', 0.5, 2.0)
    corr_limit = 0.5#trial.suggest_float('corr_limit', 0.3, 0.6)
    #longitud_liquidez = trial.suggest_int('longitud_liquidez', 1, 30)
    #atr_multipliers = trial.suggest_categorical('atr_multipliers', [(2, 3), (3, 4), (4, 5)])
    mutiplicador_tp_sl_0=trial.suggest_float('multiplicador_tp_sl_0', 0.5, 1) #,initial_value=
    mutiplicador_tp_sl_1=trial.suggest_float('multiplicador_tp_sl_1', 1.5, 3)

    backtest=Backtesting(dinero_inicial=6000,name="EURUSD",RR=1.5,sl_minimo_pips=6.5,sl_maximo_pips=6.5,tamanyo_break_even=16,break_even_size=-15,enable_break_even=False,
                        enable_dinamic_sl=False,enable_dinamic_tp=True,maximo_perdida_diaria=maximo_perdida_diaria,riesgo_por_operacion=riesgo_por_operacion,
                        trigger_trailing_tp=0.5,trigger_trailing_sl=6,maximo_beneficio_diario=80/100,lotaje_maximo=lotaje_maximo,lotaje_minimo=lotaje_minimo,
                        longitud_liquidez=4, #cuanto mayor mejor, 30 va bien, pero 10 es mas rápido para hacer las pruebas #lotaje_maximo=1.4
                        ticks_refresco_liquidez=20000,maximo_operaciones_consecutivas=maximo_operaciones_consecutivas,limite_potencia=1,bloquear_noticias=False,
                        atr_multipliers={"all_data_5m":(2,3),"all_data_1h":(2.5,3.75),"all_data_15m":(2,3)},re_evaluar=True, moneda_cuenta="USD",lista_pesos_confirmaciones=[0.5,0.6],
                        var_multiplier=var_multiplier,corr_limit=corr_limit,multiplicador_tp_sl=[mutiplicador_tp_sl_0,mutiplicador_tp_sl_1],

                        dias_retiro=1000,cantidad_retiro=0/100,zona_horaria_operable=[[4,22],[5,23]], divisor_tiempo_limite=3, # la cantidad de retiro tiene que ser menor que la perdida maxima consecutiva

                        #--------------- 2024 ---------------
                        #fecha_inicio=datetime(2024,11,1),fecha_final=datetime(2024,12,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,10,1),fecha_final=datetime(2024,11,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,9,1),fecha_final=datetime(2024,10,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,8,1),fecha_final=datetime(2024,9,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,7,1),fecha_final=datetime(2024,8,11),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,6,1),fecha_final=datetime(2024,7,11),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,5,1),fecha_final=datetime(2024,6,11),verbose=False,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,4,1),fecha_final=datetime(2024,5,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,3,1),fecha_final=datetime(2024,4,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,2,1),fecha_final=datetime(2024,3,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,1,1),fecha_final=datetime(2024,2,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,
                        #fecha_inicio=datetime(2024,1,1),fecha_final=datetime(2024,12,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,

                        fecha_inicio=datetime(2024,6,1),fecha_final=datetime(2024,9,1),verbose=False,sounds=True,grafico=False,recavar_datos=False,

                        #--------------- 2023 ---------------
                        #fecha_inicio=datetime(2023,3,1),fecha_final=datetime(2024,8,1),verbose=True,sounds=True,grafico=False,recavar_datos=False,

                        #--------------- 2022 ---------------
                        #fecha_inicio=datetime(2022,1,1),fecha_final=datetime(2023,1,1),verbose=True,sounds=True,grafico=False,recavar_datos=True,

                        #--------------- 2021 ---------------
                        #fecha_inicio=datetime(2021,1,1),fecha_final=datetime(2022,1,1),verbose=True,sounds=True,grafico=False,recavar_datos=True,
                        
                        #--------------- 2020 ---------------
                        #fecha_inicio=datetime(2020,1,1),fecha_final=datetime(2021,1,1),verbose=True,sounds=True,grafico=False,recavar_datos=True,

                        save_file=False,maximo_operaciones_diarias=40,min_spread_trigger=[0.00006,0.00005],max_spread_trigger=[0.00009,0.00008],

                        ema_length=8,ema_length2=55,ema_length3=55,ema_length4=144,

                        sma_length_2=21,sma_length_3=34,length_macd=13,
                        sma_length_4=34,sma_length_5=55,length_macd2=21,

                        rsi_roll=13,stoch_rsi=(3,3),rsi_values=(20,80),
                        bollinger_sma=89,
                        obv_ema=5,

                        adx_window=2,
                        psar_parameters=(0.02,0.2,0.02),
                        atr_window=13,atr_ma=5,
                        mfi_length=13,mfi_values=(20,80),
                        pvt_length=144, # sin probar bien
                        adl_ema=144, #sin probar bien
                        wr_length=(21,89),
                        vroc=13,
                        nvi=34,
                        momentum=144,
                        cci=(21,5),
                        bull_bear_power=55,
                        mass_index=(9,25,55),
                        trix=13,
                        vortex=13,
                        z_score_levels=(-2,2))

                        # Números fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597

    performance = backtest.run()

    gc.collect()
    return performance

if __name__=="__main__":
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_contour,
        plot_slice,
        plot_edf,
    )

    start_time=time.time()
    #study = optuna.create_study(direction="maximize")
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=10,n_jobs=-1)

    print()
    print("Best parameters: ", study.best_params)
    print("Best performance: ", study.best_value)
    print("Tiempo necesario: ",(time.time()-start_time)/60," minutos")

    # Mostrar gráficos
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
    plot_parallel_coordinate(study).show()
    plot_contour(study).show()
    plot_slice(study).show()
    plot_edf(study).show()