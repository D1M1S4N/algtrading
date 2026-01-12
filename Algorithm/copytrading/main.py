from MetaTrader5 import MT5
import time

# Definir las cuentas y servidores
cuenta_maestra = 123456
cuenta_esclava = 789012
servidor_maestro = "Broker1"
servidor_esclavo = "Broker2"
contraseña_maestra = "pass_maestra"
contraseña_esclava = "pass_esclava"

# Inicializar solo una vez
MT5.initialize()

# Conectar a las cuentas
if not MT5.login(cuenta_maestra, password = contraseña_maestra, server = servidor_maestro):
    print("Error al conectar con la cuenta Maestra.")
    exit()

if not MT5.login(cuenta_esclava, password = contraseña_esclava, server = servidor_esclavo):
    print("Error al conectar con la cuenta Esclava.")
    exit()

# Función para copiar la operación
def copiar_orden(orden_maestra):
    # Solicitud de la orden a copiar
    request = {
        "action": MT5.TRADE_ACTION_DEAL,
        "symbol": orden_maestra.symbol,
        "volume": orden_maestra.volume,
        "type": orden_maestra.type,
        "price": MT5.symbol_info_tick(orden_maestra.symbol).ask,
        "deviation": 10,
        "magic": 123456,
        "comment": "Operación copiada de la cuenta Maestra",
    }

    # Enviar la orden a la cuenta esclava
    result = MT5.order_send(request)
    if result.retcode != MT5.TRADE_RETCODE_DONE:
        print(f"Error al copiar la orden en la cuenta Esclava.")
    else:
        print(f"Orden copiada con éxito en la cuenta Esclava.")

# Bucle para monitorear las operaciones de la cuenta Maestra
while True:
    # Obtener las operaciones abiertas de la cuenta Maestra
    posiciones_maestra = MT5.positions_get()

    if posiciones_maestra is None:
        print("No se pudo obtener las operaciones.")
        break

    for pos in posiciones_maestra:
        # Verificar si esta operación no está ya copiada (por ejemplo, comparando el magic number)
        if pos.magic != 123456:  # Puedes añadir una lógica para evitar copiar órdenes ya copiadas
            copiar_orden(pos)

    # Esperar antes de revisar nuevamente las operaciones
    time.sleep(5)  # Espera 5 segundos antes de verificar de nuevo

# Cerrar la conexión al final
MT5.shutdown()