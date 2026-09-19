# El motor nativo (C++ / CUDA)

> El sustrato sobre el que correrá el cerebro de un millón de neuronas. Este documento
> dice qué hay, cómo se compila, qué mide y qué falta. Los números son de este Mac
> (M3 Max, 14 núcleos) salvo que se diga otra cosa.

## Qué es

`native/` contiene un núcleo de red de impulsos escrito para escalar:

- **Neuronas de Izhikevich en arrays** (v, u, a, b, c, d), integradas exactamente como
  el modelo TypeScript (`src/core/snn/neuron.ts`): dos medios pasos de Euler para v, un
  paso para u, umbral 30 mV, reinicio a c, u += d. El test de paridad dispara la neurona
  nativa y la de TypeScript con la misma corriente y comprueba que disparan en los mismos
  ticks.
- **Sinapsis dispersas** en formato CSR por neurona presináptica (destinos + pesos), con la
  transpuesta como mapa de índices para recorrer las sinapsis que llegan a una neurona.
  Un millón de neuronas con 100 sinapsis cada una son 100 millones de sinapsis: 800 MB de
  índices y pesos más la transpuesta.
- **Interneuronas inhibitorias reales** (una fracción, de disparo rápido, con pesos de
  salida negativos) en lugar de la competición algorítmica k-ganadores: la competencia es
  lo que la población inhibitoria hace a la excitatoria.
- **Plasticidad STDP por trazas** (Morrison, Diesmann & Gerstner 2008) solo en sinapsis
  excitatorias, con un factor de modulación global (lo que dirán los neuromoduladores) y
  pesos acotados en [0, wmax].
- **Retardo de un tick**: lo que dispara en t excita a sus destinos en t+1.
- **Hilos** en CPU sobre rangos de neuronas; la entrega de impulsos acumula por hilo y
  reduce. Los mismos kernels existen para **CUDA** (`engine_cuda.cu`): integración por
  neurona, compactación de impulsos, entrega con `atomicAdd`, plasticidad. Se compilan con
  `-DGBRAIN_CUDA=ON`. **No se han compilado ni ejecutado aún**: esta máquina no tiene
  GPU NVIDIA. Lo primero al alquilar una es correr `gbrain-bench` con ese backend y
  comparar tasas de disparo y ticks por segundo con la CPU.

El motor no sabe de regiones, buses ni memorias: es el sustrato. El cerebro (TypeScript)
lo maneja por el addon de Node (`native/addon.cc`, cargado por `src/core/snn/native.ts`).

## Compilar y medir

```bash
npm run build:native      # el addon de Node (node-gyp; necesita clang/gcc y Python)
npm run test:engine       # paridad, red de 10 000 con inhibición y STDP, escala
npm run bench:native      # el ejecutable gbrain-bench (CMake), 10 000 neuronas por defecto
native/build/gbrain-bench 1000000 100 100     # neuronas, sinapsis por neurona, ticks [hilos] [corriente]
GBRAIN_BENCH_1M=1 npm run test:engine         # añade la medida a un millón (~2 GB)
```

## Medidas (CPU, M3 Max, 14 hilos, 100 sinapsis por neurona, ~0,3 % de neuronas disparando por tick)

| Neuronas | Sinapsis | ms por tick | ticks/s | Tiempo real a 10 Hz |
|---|---|---|---|---|
| 10 000 | 1 M | 0,7 | 1 370 | 137× |
| 100 000 | 10 M | 2,6 | 390 | 39× |
| 1 000 000 | 100 M | 29 | 34 | 3,4× |
| 1 000 000, 1 hilo | 100 M | 39 | 26 | 2,6× |

Un millón de neuronas corre en tiempo real en este Mac con esa actividad. Tres avisos
honestos: la actividad medida es baja (0,3 %) y el coste de la entrega de impulsos y de
la plasticidad crece con ella; la entrega y la plasticidad solo se reparten entre hilos
cuando hay muchos impulsos por tick, así que 14 hilos apenas dan 1,3× sobre un hilo a esa
actividad (el reparto de la plasticidad está pendiente); y con el estado de reposo
heterogéneo (sin él la red disparaba en bloque) el tick cuesta el doble que la primera
medida en silencio. Con actividad del 2–5 % hay que medir de nuevo.

## La primera región sobre el motor: la corteza nativa

`src/regions/native-cortex/native-cortex.ts` es una `BrainRegion` cuyas neuronas viven en
el motor: 10 000 neuronas de impulsos (5 000 dentro del cerebro), 100 sinapsis recurrentes
al azar por neurona (excitatorias débiles, 0–0,2; ocho veces más fuertes sobre las
interneuronas, que es lo que hace que la inhibición de retroalimentación siga a la
actividad), un 20 % de interneuronas con peso −2, y una proyección aferente dispersa (cada
neurona excitatoria escucha 50 canales de entrada, una interneurona 15) por la que le llega
lo que el tálamo relé. No hay competición algorítmica ni plantilla: una entrada excita a las
neuronas cuyas aferentes la muestrean, las interneuronas mantienen la respuesta dispersa, y
la STDP recurrente (con LTP dominante: lo que dispara junto se cablea junto) estampa la
asamblea. Calibrar esto costó: con recurrentes fuertes todas las neuronas acababan
disparando, con ganancia aferente alta la entrada sola las disparaba, y con interneuronas
poco acopladas la inhibición no se enteraba de nada.

Se activa con `GBRAIN_NATIVE=1` (necesita el addon compilado): el cerebro añade la región
`nativeCortex` alimentada por el relé visual; sale en el dashboard como "Native Ctx (C++)".
Sus pesos recurrentes persisten en el estado del cerebro (2 MB a 5 000 neuronas).

Lo que el test (`tests/native-region.test.ts`, en `npm run test:engine`) comprueba a
10 000 neuronas, sobre los recuentos de disparo de las neuronas excitatorias (las
interneuronas disparan a todo y no representan nada): la misma entrada dos veces da
respuestas correlacionadas (r = 0,60) y dos entradas distintas no (r = 0,10), el mismo
criterio que la suite usa para el hipocampo y la corteza prefrontal; la respuesta es
dispersa (un 27 % de las excitatorias dispara alguna vez en media presentación); tras 40
repeticiones las sinapsis entre las neuronas de la asamblea (el 5 % más sostenido) crecen
más que el resto (+0,21 frente a +0,12); dentro del cerebro dispara al ver algo y calla si
no. Lo que **no** hace aún: completar la respuesta desde media entrada (r 0,13 → 0,17).
Con 100 sinapsis recurrentes al azar, cada miembro de la asamblea recibe un puñado de
sinapsis de su asamblea: pocas para completarla aunque estén al máximo. Hace falta
conectividad más densa dentro de la asamblea, es decir, plasticidad estructural (crear
sinapsis entre las neuronas que disparan juntas), que es lo siguiente.

## Lo que falta para que el cerebro corra encima

1. **Regiones sobre el motor**: hecha la primera (la corteza nativa, arriba). Las cortezas
   actuales (visual, auditiva, de color, de partes) siguen siendo plantillas densas en
   TypeScript; pasarlas al motor significa que sus categorías nazcan de asambleas, no de
   engramas k-WTA, y eso pide primero la plasticidad estructural.
2. **Plasticidad estructural y conectividad con estructura**: crear sinapsis entre las
   neuronas que disparan juntas (y podar las inútiles), para que las asambleas se completen;
   y proyecciones entre regiones (tálamo → corteza) como bloques del CSR con pesos iniciales
   por proyección.
3. **Neuromoduladores regionales**: el factor de modulación es global; debe ser por
   región (dopamina en el estriado, acetilcolina en la corteza…).
4. **Retardos axonales** por sinapsis (hoy un tick para todas) y **oscilaciones**: con
   interneuronas reales aparecen ritmos gamma; hay que medirlos y usarlos (la fase como
   ventana de plasticidad).
5. **Cómputo dendrítico**: compartimentos apicales para la predicción descendente.
6. **Persistencia**: el estado del motor (pesos CSR) en el formato binario del cerebro.
7. **GPU**: compilar y medir el backend CUDA en una máquina alquilada; después decidir
   si el millón vive en la GPU o en CPU.
