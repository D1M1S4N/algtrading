#-----------------------------------------------------------------------------------------

EL IPDA lo que hace es buscar el mejor precio al que una institución puede comprar, para que no sea ni muy alto, ni muy bajo, ni muy cerca, ni muy lejos, y que no modifique mucho la tendencia del mercado, para esto necesitan mucha gente.

Para buscar el mejor precio a veces solo con buscar encuentran un punto, y si no lo encuentran, lo crean, en mi caso, solo tengo que ver donde ha avido puntos
inducidos,para meterme en el sentido del punto.

cuando una institución hace una operación, tiene que estar un tiempo acumulando, creando compradores o vendedores, y una vez que los tiene realiza la acción,
produciendo un movimiento brutal.

Donde mas liquidez hay, es donde la mayoría de la gente ha puesto los sl.

#-----------------------------------------------------------------------------------------

-Puedo buscar tambien los puntos a los que se manipula el precio para las instituciones (para ir al precio ese, para entrar en la manipulación no, ya que ellos lo camuflan para que no se pueda saber el sentido, sería gambling).

-Puedo buscar el punto donde mas sl hay para así entrar en ese punto y aprovechar la liquidez

#-----------------------------------------------------------------------------------------

Trading de pares con reversiones

Ejemplo:

Ejemplo práctico
    Supongamos que estás operando con dos acciones del sector bancario, como Banco A y Banco B:

    Históricamente, sus precios tienden a moverse de manera similar.
    Un día, Banco A sube un 5% mientras que Banco B cae un 3%.

    Aplicando la estrategia:
        Vendes en corto Banco A (porque esperas que su precio baje).
        Compras Banco B (porque esperas que su precio suba).

    Cierras ambas posiciones cuando los precios vuelven a sus niveles de correlación histórica.

Ventajas de esta estrategia
    Riesgo limitado: Al operar con activos correlacionados, estás protegido en cierta medida contra movimientos bruscos del mercado, ya que se compensan mutuamente.
    Diversificación: Al operar con múltiples activos, puedes diversificar tus posiciones y reducir el riesgo general.

Esto se puede usar con acciones distintas, o con una misma accion en 2 indices distintos

#-----------------------------------------------------------------------------------------

Estrategia cobertura divisas / acciones / indices

Ejemplo de aplicación
    Supongamos que eres un inversor europeo que quiere comprar acciones de Apple Inc. (cotizadas en EE. UU.):

        Comprar acciones de Apple (que cotizan en USD).
        Comprar un ETF o contrato de futuros del índice S&P 500 para diversificar la exposición al mercado estadounidense en general.
        Cubrir la exposición al USD/EUR mediante un contrato de futuros en el par de divisas USD/EUR, para protegerte de una posible depreciación del dólar frente al euro.

#-----------------------------------------------------------------------------------------

1. Análisis de Profundidad del LOB
Observación de Niveles de Precio: En el LOB, los niveles de precio muestran las cantidades de órdenes de compra (bid) y venta (ask) en cada precio. La profundidad del mercado indica el volumen de órdenes en distintos niveles y puede sugerir resistencias o soportes de precio a corto plazo.
Imbalance de Órdenes: Se calcula la diferencia entre el volumen total de órdenes de compra y el de órdenes de venta en los niveles más cercanos al precio actual (por ejemplo, los primeros 5 o 10 niveles). Un desequilibrio grande puede indicar una presión de compra o venta inminente, que podría empujar el precio en una dirección.

2. Modelos de Predicción de Microestructura del Mercado
Procesos Estocásticos: Algunos traders usan modelos estocásticos (por ejemplo, el modelo de Hawkes) que simulan la actividad del LOB basándose en patrones históricos de órdenes. Estos modelos pueden predecir la aparición de picos de actividad, como una fuerte ola de compras o ventas.
Redes Neuronales y Deep Learning: Con el aprendizaje profundo, se pueden entrenar redes neuronales para detectar patrones complejos en el flujo de órdenes del LOB. Las redes neuronales recurrentes (RNN) y redes convolucionales (CNN) son comunes para predecir la dirección del próximo movimiento de precios al analizar secuencias de datos en el LOB.

3. Detección de Spoofing y Manipulación de Mercado
Análisis de Cambios Rápidos en el LOB: En algunos casos, traders intentan manipular el mercado colocando grandes órdenes sin intención de ejecutarlas (spoofing), lo cual puede influir en el comportamiento de otros participantes. Se puede usar machine learning para identificar patrones de entrada y retirada rápida de órdenes grandes.
Seguimiento de Órdenes Falsas y Velocidad de Cancelación: Algoritmos que monitorizan la tasa de colocación y cancelación de órdenes en tiempo real pueden detectar intenciones de manipulación. Si una gran orden de venta o compra se retira frecuentemente en momentos críticos, podría ser una señal de actividad engañosa.

4. Creación de Indicadores de Microestructura
Spread Bid-Ask: El spread entre las órdenes de compra y venta más cercanas refleja la liquidez del mercado en tiempo real. Un spread pequeño indica alta liquidez y baja volatilidad, mientras que un spread grande sugiere mayor volatilidad y menor liquidez.
Momentum del LOB: Se puede calcular el momentum del LOB al observar cómo cambian los volúmenes en distintos niveles de precio. Por ejemplo, un aumento rápido en órdenes de compra en los primeros niveles puede indicar presión alcista inminente.
Volatilidad Implied en el LOB: Algunas técnicas calculan una volatilidad implícita a partir del volumen y frecuencia de cambios en las órdenes. Un aumento en esta volatilidad puede señalar una mayor probabilidad de cambio brusco en el precio.

5. Análisis del "Order Flow" o Flujo de Órdenes
Flujo de Ordenes Direccional: Monitorear si el flujo de órdenes (la cantidad de órdenes de compra o venta que se ejecutan) es mayor en una dirección. Si la mayoría de las órdenes se ejecutan al precio de compra (bid) o al precio de venta (ask), eso puede señalar una presión de compra o venta, respectivamente.
Intensidad de las Ejecuciones: La cantidad de órdenes ejecutadas en cada dirección, junto con su frecuencia, indica la intensidad del mercado. Un volumen alto de órdenes de compra ejecutadas rápidamente podría indicar un alza en el precio, mientras que un flujo de ventas podría indicar lo contrario.

6. Construcción de Indicadores de Desequilibrio
Desequilibrio del LOB en Tiempo Real: Una técnica común es calcular un "desequilibrio" del LOB basado en la diferencia de volumen entre los primeros niveles de compra y venta. Un gran desequilibrio en los niveles más cercanos al precio puede sugerir que el precio se moverá en dirección del lado con menor volumen.
Desequilibrio Dinámico: Los algoritmos pueden ajustar el cálculo del desequilibrio a medida que el mercado cambia. Un aumento repentino en el volumen de órdenes de venta puede dar una señal de venta, mientras que un aumento en las órdenes de compra puede indicar presión alcista.

7. Predicción de Movimientos de Precios a Muy Corto Plazo
Machine Learning Supervisado: Se puede entrenar un modelo de aprendizaje automático (como un modelo de árboles de decisión o una red neuronal) para predecir el movimiento del precio en base a snapshots (instantáneas) del LOB, utilizando una gran cantidad de datos históricos del LOB.
Secuencias Temporales: Utilizando modelos como Long Short-Term Memory (LSTM), que están diseñados para trabajar con secuencias de datos, es posible analizar cómo el cambio en órdenes de compra y venta en los primeros niveles afecta el precio en los próximos segundos o milisegundos.

---------------------------- NOTICIAS ----------------------------

GERMAN: Yo no opero hasta que sale la noticia importante, cuando sale me voy a forex factory el web para ver las noticias y veo como han salido para el euro o el dolar si positivas y negativas, si busco por ejemplo compras en Ny (dolar) y la noticia ha sido negativa para el dolar es bueno porque significa que el euro puede subir y me beneficia en las compras

Y en base a eso meto mas o menos riesgo, con el mismo ejemplo, si busco compras y hubiera salido positiva para el euro me puede perjudicar ya que puede seguir bajando

---------------------------- Lista cosas ----------------------------

comprobar liquidez de arriba o abajo para decir que hacer (confirmacion) ✅

si hay mucha liquidez en sentido contrario es malo, y si tiene poca liquidez en sentido bueno tambien es malo (confirmacion) ✅

ia optimización parametros ✅

noticias ✅

poner limite operaciones consecutivas en real ✅

calcular el resto de tick_data y data_5m solo si se el precio esta en un punto de liquidez para asi ir mas rapido ✅

calcular bien perdida diaria en real ✅

arreglar lo de no hay suficientes datos en demo ✅

probar a cambiar dinero[1] en la formula del riesgo por el dinero inicial diario en vez de por el valor del dinero actual ✅ (esto va peor mejor con el valor del dinero actual)

poner diferencia de spread entre horario verano e invierno ✅

he probado a cambiar maximo perdida diaria a 1.4,2 y 3, y el que mejor rr da y mejor estabilidad en el grafico es el rr=3 ✅

Una vez el bot esté metido dentro de una operación, hay ocasiones en las que el precio no se mueve y empieza a rangear, en esos casos que recalcule la liquidez para ver si es bueno cerrar la operación sin que llegue al tp ya sea a mitad de recorrido o en be o parciales 
^ esto de aqui arriba lo podria arreglar poniendo un limite de tiempo teniendo en cuenta el numero de confirmaciones, y para mas precisión podria poner que si el precio se mueve menos de x% en x tiempo  desde la entrada que se cierre la operacion. ✅

si la entrada es repetida se baja el lotaje a la mitad ✅

hay pecheada dura en data_5m=get_data_temporality(self.name) en vez de nombre ✅

cambiar lotaje minimo dinamico (esto es mejor dejarlo tal cual 0.2, ya que es el lotaje minimo para que una operacion te de beneficio) ✅

ia sin datos de tick para que vaya mas rapido ✅

cambiar tamaño tp y sl ✅

asignar peso por estrategia en vez de sumarlas ✅

confirmaciones de entrada: Cambio de estructura, formación de velas + imbalance y confirmación de imbalance (ver que lo respeta ofreciendo el mercado una vela envolvente, o vela martillo de rechazo) ✅

volver a pensar algoritmo de liquidez ✅

mirar lotaje minimo en real ✅

poner en los mensajes las estrategias utilizadas ✅

arreglar lo de las horas ✅

probar lo de que el punto de liquidez esté entre medias (esto va peor) ✅

probar a poner un copy en la liquidez de los dias antes de modificar nada (esto va peor) ✅

poner lo de mas riesgo cuando la liquidez es mas antigua (probarlo lo del lotaje en vez de riesgo)  (va mejor quitando riesgo)✅

probar a bajar el lotaje/2 tambien si el punto esta en las operaciones anteriores ✅

arreglar el main, poner lo de lotaje/2 pero con el precio de entrada bien, guardandolo en una lista ✅

mirar al final del año hay mas paradas porque arriesga mas porque tiene mas dinero pero la perdida diaria es una mierda (solucionado poniendo retiros 2x1) ✅

cerrar operaciones los viernes ✅

optimizar algoritmo liquidez ✅

probar a hacer una entrada en el mismo punto solo si data_5m es distinto del anterior ✅

retiros en main ✅

quitar operaciones los dias festivos ✅

arreglar lo de las operaciones consecutivas ✅
poner que si ha perdido mucho, que no meta 2 operaciones, es decir, si esta cerca de la maxima perdida diaria que meta solo 1 ✅

revisar en el main lo de no entrar en la misma vela parece que no funciona muy bien ✅

probar fibonacci ✅

poner nuevo sistema de estrategias? ❓

poner +0.5 con parciales en vez de 1 (backtesting copy) ✅ (va mejor sube 0.18 de rr)

probar modificar sl ademas de tp (backtesting) ❌

probar a poner be despues de parciales (main, comparando con backtesting copy) ✅

1- probar recalculacion de tp y parciales ❌

2- probar recalculacion de tp y parciales poniendo be ✅

3- probar recalculacion de tp y retirar parciales en sl tambien ✅

4- probar recalculacion de tp y retirar parciales en sl tambien poniendo tp/2 ❌

de todas estas pruebas anteriores, la mejor es combinar la opcion 2 y 3, la 4 es demasiado, y la 1 es peor que la 2

---------------------------------------------------------------------------------------------

poner las fotos de los backtestings de whatsapp en una carpeta del bot ✅

poner para que meta mas lotaje y gane mas beneficio, arriesgando mas, pero sin cambiar el riesgo para que los tp/sl no cambien mucho (la solucion para esto es hacer mas operaciones, meter mas lotaje no es buena idea) ❌

probar a poner retiros en numeros distintos de 50, como por ejemplo 60 o 70 ✅

añadir en el main un test, para que cuando carge el bot al principio haga una operacion de coña para ver si funciona todo bien (hay que probarlo) ✅

arreglar lo de que no opere en la misma vela de 5m (hay que probarlo) ✅

poner en el main, modificacion de tp, retirar parciales y poner be, y retirar parciales en sl ✅

poner limite de deslizamiento en las entradas ✅

arreglar el backtesting que con 200000 no opera ✅

poner que el bot mande mensaje a tg cuando recalcule liquidez sobre los puntos ✅

arreglar el error en la linea 494 estrategias/main ✅

probar porque si /(atr/atr_ma) estaba mal, donde se usa *(atr/atr_ma) igual esta mal (en los tp me refiero) ❌

arreglar los parciales en real ⚠️

formula que encuentre spread maximo y minimo de verano y de invierno para probar con datos de tickstory ✅

arreglar parciales ✅

arreglar error en dia 2023-08-01 ✅

arreglar la ñapa del backtesting de los parciales de sl ✅

hacer script que copie las operaciones de una cuenta a otra

arreglar mensaje telegram id

poner anterioridad liquidez en telegram

arreglar error en columnas de datos 5m

poner rr teorico tambien cada dia

probar a poner limite de beneficio

arreglar la ñapa de std y en general los indicadores

arreglar estrategia mezcla

probar estrategia random como bias

datos 5 min de varios dias?

quitar los maximos valores fijos posibles (como de la estrategia estadistica por ejemplo)

preparar para que todo este listo para multiples cosas operables, quitar las metricas de pips, /10000 del spread,etc

optimizar estrategias un poco

hacer que vaya mas rapido el backtesting cuando hay mucha liquidez al recalcular pierde mucho tiempo 

probar a poner min o max en las entradas que sean de except (no aparece en ningun punto de liquidez)

poner, que teniendo el archivo resultado del backtesting, se pueda ver un grafico interactivo con las velas y las entradas

operar indices y confirmaciones (para reducir riesgo o para doble beneficio)

arreglar lo de las operaciones contrarias

probar maximo perdida diaria

arreglar lo de las noticias otra vez

optimizar las estrategias con ia y todo

https://github.com/joshyattridge/smart-money-concepts

optimizar main y marketmaster.run

arreglar para que en el main los ticks vayan bien, que no haya ticks fantasma (lo puedo arreglar poniendo que se reinicie cada cierto tiempo)⚠️

probar para poner una operacion muy pequeña antes de la operacion real para ver como reacciona (Simula órdenes falsas o pequeños trades para detectar si hay HFTs (High-Frequency Traders) tratando de manipular el precio)

Coloca órdenes a ambos lados del order book con spreads estratégicos, actuando como market maker en mercados ilíquidos.

Incluye métricas avanzadas, como proyecciones de ganancias, análisis de flujo de caja y evaluaciones de riesgo

modelo black scholes

ia para noticias ademas de arreglar que no opere en noticias

simulaciones monte carlo

cambiar riesgo dependiendo de corto o largo 
cambiar riesgo dependiendo de verano o invierno 
cambiar la temporalidad 

Parámetros adaptativos con IA

IA para blending

cuando esten puestos de varios activos pueda rebalancear las posiciones como mejor le convenga

añadir ia que tome toda la informacion de la cuenta para tomar decisiones de operaciones, es decir que mire lo que ha ganado y perdido y el riesgo