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
- **Retardo de un tick y corrientes sinápticas exponenciales**: lo que dispara en t llega a
  sus destinos desde t+1 y se apaga con una constante de tiempo (AMPA ≈ 5 ms para las
  excitatorias, GABA_A ≈ 10 ms para las inhibitorias). Sin eso cada impulso era una corriente
  de un solo tick y nada sumaba en el tiempo: una asamblea no podía sostenerse.
- **Depresión sináptica a corto plazo** (Tsodyks & Markram 1997) por neurona presináptica
  excitatoria: cada impulso gasta una fracción (0,3) de los recursos sinápticos, que se
  recuperan en ~200 ms. Es lo que hace que una asamblea se encienda y se apague en vez de
  convertirse en un atractor que sobrevive a su entrada y responde a todo (medido: sin ella,
  la asamblea aprendida seguía disparando en silencio y la respuesta a media entrada
  correlacionaba 0,99 con la respuesta a otra entrada).
- **Plasticidad inhibitoria** (Vogels et al. 2011) en las sinapsis interneurona → excitatoria:
  la inhibición aprende a equilibrar la excitación de cada neurona hacia una tasa objetivo.
- **Plasticidad estructural**: cada cierto número de ticks, las neuronas excitatorias
  coactivas cambian sus sinapsis más débiles por sinapsis nuevas hacia otras del núcleo
  coactivo (sinaptogénesis entre neuronas coactivas; Holtmaat & Svoboda 2009), con el abanico
  fijo para que la memoria no crezca.
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

## Medidas (CPU, M3 Max, 14 hilos, 100 sinapsis por neurona, `gbrain-bench`)

| Neuronas | Sinapsis | ms por tick | ticks/s | Tiempo real a 10 Hz | Actividad |
|---|---|---|---|---|---|
| 10 000 | 1 M | 0,19 | 5 300 | 530× | 0,1 % |
| 100 000 | 10 M | 0,6 | 1 660 | 166× | 0,2 % |
| 1 000 000 | 100 M | 17 | 58 | 5,8× | 0,2 % |
| 1 000 000 (medida anterior, sin depresión sináptica) | 100 M | 29 | 34 | 3,4× | 0,3 % |

Un millón de neuronas corre en tiempo real en este Mac con esa actividad. Avisos honestos:
la actividad medida es baja (con la depresión sináptica a corto plazo el banco dispara menos
que antes, de ahí que el tick salga más barato: no es una optimización) y el coste de la
entrega de impulsos y de la plasticidad crece con ella; la entrega y la plasticidad solo se
reparten entre hilos cuando hay muchos impulsos por tick (el reparto de la plasticidad está
pendiente). Con actividad del 2–5 % hay que medir de nuevo.

## La primera región sobre el motor: la corteza nativa

`src/regions/native-cortex/native-cortex.ts` es una `BrainRegion` cuyas neuronas viven en
el motor: 10 000 neuronas de impulsos (5 000 dentro del cerebro), 100 sinapsis recurrentes
al azar por neurona (excitatorias débiles, 0–0,2; ocho veces más fuertes sobre las
interneuronas, que es lo que hace que la inhibición de retroalimentación siga a la
actividad), un 20 % de interneuronas con peso −2, y una proyección aferente dispersa y
**topográfica**: cada neurona excitatoria escucha 50 canales de entrada dentro de un campo
receptivo (una quinta parte de los canales, centrado en su lugar en la lámina; una
interneurona escucha 15), como la retinotopía o la tonotopía. Por ahí le llega lo que el
tálamo relé. No hay competición algorítmica ni plantilla: una entrada excita a las neuronas
cuyas aferentes la muestrean, las interneuronas mantienen la respuesta dispersa, y la STDP
recurrente (con LTP dominante: lo que dispara junto se cablea junto) estampa la asamblea.
Calibrar esto costó: con recurrentes fuertes todas las neuronas acababan disparando, con
ganancia aferente alta la entrada sola las disparaba, y con interneuronas poco acopladas la
inhibición no se enteraba de nada.

Se activa con `GBRAIN_NATIVE=1` (necesita el addon compilado): el cerebro añade la región
`nativeCortex` alimentada por el relé visual; sale en el dashboard como "Native Ctx (C++)".
Sus pesos recurrentes persisten en el estado del cerebro (2 MB a 5 000 neuronas).

Lo que el test (`tests/native-region.test.ts`, en `npm run test:engine`) comprueba a
10 000 neuronas, sobre los recuentos de disparo de las neuronas excitatorias durante toda la
presentación (las interneuronas disparan a todo y no representan nada): la misma entrada
dos veces da respuestas correlacionadas (r = 0,88) y dos entradas distintas mucho menos
(r = 0,27; con aferentes topográficas, donde dos entradas al azar iluminan el mismo tramo de
canales responden las mismas neuronas), el mismo criterio que la suite usa para el hipocampo
y la corteza prefrontal; la respuesta es dispersa (un 1,5 % de las excitatorias dispara por
tick y solo un 0,4 % se sostiene tres impulsos o más; el arranque hace disparar una vez a
muchas antes de que las interneuronas las alcancen); tras 40 repeticiones las sinapsis entre
las neuronas de la asamblea (el 5 % más sostenido) crecen más que el resto (+0,61 frente a
+0,11) y la plasticidad estructural (cada 60 ticks, las neuronas excitatorias que dispararon
al menos tres veces forman el núcleo coactivo; cada una cambia hasta cuatro de sus sinapsis
más débiles, por debajo de 0,15, por sinapsis nuevas hacia otras del núcleo, nacidas a 1,0;
el abanico se mantiene fijo, la transpuesta se reconstruye) recablea cientos de sinapsis;
dentro del cerebro dispara al ver algo y calla si no.

**Compleción** (la medida honesta, con la topografía): media entrada excita a la mitad de la
lámina que la escucha; las neuronas de la asamblea que solo escuchan la mitad apagada no
reciben nada de la pista, y únicamente las sinapsis recurrentes pueden traerlas de vuelta.
Antes de aprender, con media entrada dispara entre el 0 y el 20 % de ese lado; después, el
**85–100 %** (12 de 12 combinaciones de semilla de red y par de patrones, con patrones
equilibrados por mitades), y solo un 3–8 % de las demás neuronas de ese lado; la respuesta
completada es la propia de esa entrada (r ≤ 0,22 con la respuesta a otra) y nada sigue
disparando en silencio. Lo que hizo falta, por orden de descubrimiento: (1) corrientes
sinápticas con constante de tiempo, porque con impulsos de un tick la recurrencia nunca se
encendía (con media entrada disparaban 8 neuronas por tick frente a 100); (2) con eso, la
asamblea aprendida se convertía en un atractor permanente (600 neuronas por tick en silencio,
respuesta idéntica a cualquier entrada): la depresión sináptica a corto plazo la hace
transitoria; (3) con aferentes al azar "media entrada" era la entrada entera a media
intensidad y nadie superaba el umbral: la topografía convierte la pista en lo que es en la
corteza, una parte del patrón que excita una parte de la asamblea. Límite conocido: la
asamblea medida es pequeña (30–60 neuronas sostenidas de 8 000) y con sinapsis nuevas más
fuertes (2,0) completa más pero pierde la especificidad (otra entrada también la enciende).

## Lo que falta para que el cerebro corra encima

1. **Regiones sobre el motor**: hecha la primera (la corteza nativa, arriba). Las cortezas
   actuales (visual, auditiva, de color, de partes) siguen siendo plantillas densas en
   TypeScript; pasarlas al motor significa que sus categorías nazcan de asambleas, no de
   engramas k-WTA, y eso pide primero la plasticidad estructural.
2. **Conectividad con estructura**: la compleción funciona con aferentes topográficas y
   recurrentes al azar; faltan recurrentes con estructura (más densas entre vecinas) y
   proyecciones entre regiones (tálamo → corteza) como bloques del CSR con pesos iniciales
   por proyección; y que la depresión y la facilitación a corto plazo sean por tipo de
   sinapsis (hoy la depresión es por neurona presináptica excitatoria).
3. **Neuromoduladores regionales**: el factor de modulación es global; debe ser por
   región (dopamina en el estriado, acetilcolina en la corteza…).
4. **Retardos axonales** por sinapsis (hoy un tick para todas) y **oscilaciones**: con
   interneuronas reales aparecen ritmos gamma; hay que medirlos y usarlos (la fase como
   ventana de plasticidad).
5. **Cómputo dendrítico**: compartimentos apicales para la predicción descendente.
6. **Persistencia**: el estado del motor (pesos CSR) en el formato binario del cerebro.
7. **GPU**: compilar y medir el backend CUDA en una máquina alquilada; después decidir
   si el millón vive en la GPU o en CPU.
