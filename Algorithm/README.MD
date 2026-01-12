# Market Master Algorithmic Trading System (Market Master © 2023-06-26)

## Descripción General
Este proyecto implementa un sistema de trading algorítmico diseñado para operar utilizando MetaTrader 5. El sistema incluye capacidades de backtesting y ejecución en vivo.

## Archivos Principales

### backtesting.py
Este archivo contiene la implementación del sistema de backtesting.

#### Clase Backtesting
- **Propósito**: Simular estrategias de trading en datos históricos.
- **Parámetros principales**:
  - `dinero_inicial`: Capital inicial para el backtesting.
  - `name`: Nombre del instrumento financiero.
  - `RR`: Ratio de riesgo/recompensa.
  - `lotaje_maximo`, `lotaje_minimo`: Límites de tamaño de lote.
  - `maximo_perdida_diaria`: Límite de pérdida diaria.
  - `riesgo_por_operacion`: Porcentaje de riesgo por operación.

#### Métodos Principales
- `run()`: Ejecuta la simulación de backtesting.

### main.py
Este archivo implementa la ejecución en vivo del sistema de trading en MetaTrader 5.

#### Clase Principal (no nombrada en el snippet)
- **Propósito**: Ejecutar la estrategia de trading en tiempo real.
- **Parámetros principales**: Similares a los de la clase Backtesting.

#### Métodos Principales
- `run()`: Ejecuta la estrategia de trading en vivo.
- `get_data()`,`get_tick_data()`: Obtiene datos en tiempo real de MetaTrader 5.

## Características Clave del Sistema
1. **Gestión de Riesgo**: 
   - Control de riesgo por operación y pérdida máxima diaria.
   - Cálculo dinámico de tamaño de posición.

2. **Estrategias**: 
   - Las estrategias implementadas se encuentran en estrategia/main.py

3. **Optimización de Entrada/Salida**: 
   - Cálculo dinámico de niveles de entrada, Stop Loss y Take Profit.
   - Implementación de trailing stop y break-even.

4. **Gestión de Tiempo**: 
   - Considera horarios de mercado y eventos económicos.

6. **Ejecución en Vivo**: 
   - Integración directa con MetaTrader 5 para operaciones en tiempo real.
   - Manejo de errores y excepciones para operación robusta.

## Uso
1. Configurar parámetros en `main.py` o `backtesting.py`.
2. Ejecutar backtesting para probar la estrategia.
3. Ajustar parámetros según resultados del backtesting.
4. Ejecutar `main.py` para operar en vivo con MetaTrader 5.

## Notas Importantes
- Asegurar que la lógica implementada en `backtesting.py` se refleje fielmente en `main.py`.
- Realizar pruebas exhaustivas en cuentas demo antes de usar fondos reales.
- Monitorear regularmente el rendimiento del sistema y ajustar según sea necesario.
