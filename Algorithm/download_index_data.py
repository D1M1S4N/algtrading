import pandas as pd
import MetaTrader5 as mt5

# Descargar datos de SP500 desde MetaTrader 5
def download_sp500_data(symbol, timeframe, start_date, end_date):
    # Inicializar MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return None

    # Descargar datos históricos
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    # Finalizar MetaTrader 5
    mt5.shutdown()

    # Convertir a DataFrame de pandas
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    else:
        print("No data retrieved")
        return None
    
# Parámetros de descarga
symbol = "SPX500"
timeframe = mt5.TIMEFRAME_M5
start_date = pd.Timestamp("2018-06-06")
end_date = pd.Timestamp("2025-11-04")

# Descargar los datos
sp500_data = download_sp500_data(symbol, timeframe, start_date, end_date)

# Mostrar los datos
print(sp500_data.head())
print(sp500_data.tail())

# Guardar los datos en un archivo CSV
sp500_data.to_csv("sp500_data_m5_fn.csv", index=False)