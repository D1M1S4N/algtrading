import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import warnings

#datos, noticias, resultados, etc

warnings.filterwarnings('ignore')

def obtener_datos_financieros(ticker):
    """
    Obtiene los datos financieros de una acción específica desde Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    
    # Obtener datos financieros básicos
    info = stock.info
    
    # Obtener estados financieros
    balance_sheet = stock.balance_sheet
    financials = stock.financials
    cashflow = stock.cashflow
    
    return info, balance_sheet, financials, cashflow

def calcular_ratios(info, balance_sheet, financials, cashflow):
    """
    Calcula ratios financieros clave para el análisis fundamental.
    """
    ratios = {}
    
    # Rentabilidad
    ratios['Margen de Beneficio'] = info.get('profitMargins', None)  # Margen de beneficio
    ratios['ROE'] = info.get('returnOnEquity', None)  # Retorno sobre el capital (ROE)
    
    # Crecimiento (usando ingresos anuales)
    
    ingresos = financials.loc['Total Revenue']
    growth_rate = (ingresos.iloc[0] - ingresos.iloc[1]) / ingresos.iloc[1]  # Crecimiento anual
    ratios['Tasa de Crecimiento de Ingresos'] = growth_rate

    # Deuda
    
    total_deuda = balance_sheet.loc['Total Liabilities Net Minority Interest'][0]  # Total de pasivos
    patrimonio = balance_sheet.loc['Stockholders Equity'][0]  # Patrimonio
    ratios['Deuda/Patrimonio'] = total_deuda / patrimonio
    
    
    # Flujo de caja libre (aproximado)
    
    # Aproximación de Capital Expenditures
    capex = (balance_sheet.loc['Net PPE'][0] - balance_sheet.loc['Net PPE'][1]) + \
            balance_sheet.loc['Accumulated Depreciation'][0]
    
    # Flujo de caja operativo (se necesita ajustar si no está disponible)
    flujo_caja_operativo = cashflow.loc['Operating Cash Flow'][0]  # Ajusta aquí si usas otra métrica
    
    # Cálculo de flujo de caja libre
    flujo_caja_libre = flujo_caja_operativo - capex
    ratios['Flujo de Caja Libre'] = int(flujo_caja_libre)
    
    # Current Ratio

    current_assets = balance_sheet.loc['Current Assets'][0]
    current_liabilities = balance_sheet.loc['Current Liabilities'][0]
    ratios['Current Ratio'] = current_assets / current_liabilities

    # Debt-to-Equity

    total_debt = balance_sheet.loc['Total Debt'][0]
    stockholders_equity = balance_sheet.loc['Stockholders Equity'][0]
    ratios['Debt-to-Equity'] = total_debt / stockholders_equity

    return ratios

def mostrar_resultados(ratios):
    """
    Muestra los resultados del análisis fundamental de manera ordenada.
    """
    print("\n=== Análisis Fundamental al Estilo Warren Buffett ===\n")
    for key, value in ratios.items():
        if value is None:
            print(f"{key}: Datos no disponibles")
        else:
            print(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")

    print()

def run():
    pass 

if __name__ == "__main__":
    ticker = input("Introduce el ticker de la acción (ej: AAPL): ")
    info, balance_sheet, financials, cashflow = obtener_datos_financieros(ticker)
    
    # Imprimir los encabezados disponibles en balance_sheet (para depurar si falta algo)
    #print("\nEncabezados de balance_sheet disponibles:")
    #print(balance_sheet.index.tolist())
    
    ratios = calcular_ratios(info, balance_sheet, financials, cashflow)
    mostrar_resultados(ratios)