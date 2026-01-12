import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# === 1. Cargar los datos ===
df = pd.read_csv('datos_trading.csv', parse_dates=['Fecha'], index_col='Fecha')
# === 2. Calcular el spread entre las dos acciones ===
df['Spread'] = df['Accion1'] - df['Accion2']

# === 3. Calcular el Z-Score del spread ===
def calculate_zscore(spread, window=30):
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    return (spread - mean) / std
df['Z-Score'] = calculate_zscore(df['Spread'])
# === 4. Análisis del Índice ===
df['Indice_SMA_50'] = df['Indice'].rolling(window=50).mean()
df['Indice_SMA_200'] = df['Indice'].rolling(window=200).mean()
df['Market_Trend'] = np.where(df['Indice_SMA_50'] > df['Indice_SMA_200'], 1, -1)
# === 5. Señales de Trading ===
df['Signal'] = np.where((df['Z-Score'] > 2) & (df['Market_Trend'] == 1), -1,
                        np.where((df['Z-Score'] < -2) & (df['Market_Trend'] == 1), 1,
0))
# === 6. Cobertura con Divisas ===
df['Hedge'] = np.where(df['Signal'] == 1, -df['ParDivisas'].pct_change().shift(-1),
                       np.where(df['Signal'] == -1,
df['ParDivisas'].pct_change().shift(-1), 0))
# === 7. Evaluar el rendimiento ===
df['Strategy_Return'] = df['Signal'] * (df['Accion1'].pct_change() - df['Accion2'].pct_change())
df['Hedged_Return'] = df['Strategy_Return'] - df['Hedge']
df['Cumulative_Return'] = (1 + df['Hedged_Return']).cumprod()
# === 8. Visualizar el rendimiento ===
plt.plot(df['Cumulative_Return'], label='Rendimiento Acumulado')
plt.legend()
plt.show()

def run():
    pass

"""

    Acciones influenciadas hoy por la tendencia del indice
        Indice tendencia alcista
            Accion alcista
                action=1

            Accion bajista
                action=0 (esperar a cambio de tendencia)


        Indice tendencia bajista        
            Accion alcista
                action=0 (esperar a cambio de tendencia)

            Accion bajista
                action=-1

        
    Acciones no influenciadas hoy por la tendencia del indice
        Accion alcista
            action=1

        Accion bajista
            action=-1

----------------------------------------------------------------------------------------

    action=1
        Compra de 1000 EUR de acciones.
        
        Si la compra es en moneda distinta de la cuenta
        Vender 1000 EUR en el par EUR/USD para obtener 1100 USD (a un tipo de cambio de 1 EUR = 1.10 USD).

    action=-1
        Venta de 1000 EUR de acciones.

        Si la compra es en moneda distinta de la cuenta
        Compra 1000 EUR en el par EUR/USD para obtener 1100 USD (a un tipo de cambio de 1 EUR = 1.10 USD).

"""