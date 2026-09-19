# gBrain — plan de referencia: un cerebro que aprende como un bebé

> Documento vivo. Cada bloque nombra el mecanismo, su base en neurociencia, qué hay
> hoy en el código y el test que puede fallar. Los bloques van en orden: cada uno
> se apoya en el anterior. Los cuatro primeros viven en la capa de orquestación
> (TypeScript) y no esperan al motor en C++.

## Principio

Fidelidad antes que atajo. El cerebro no trae de serie significados (ni un
diccionario de palabras afectivas, ni un léxico), trae **reacciones a rasgos
sensoriales, impulsos y sesgos de atención**. Todo lo demás lo aprende de la
experiencia, con reglas locales y neuromodulación. Lo que hoy es un atajo se marca
como *andamio del maestro* y tiene fecha de retirada.

## Lo que ya está resuelto

Neuronas de Izhikevich con STDP; codificación dispersa por competición (k-WTA);
depresión sináptica a corto plazo en el bus; escalado homeostático; tálamo que
filtra por saliencia con ganancia por acetilcolina y norepinefrina; hipocampo con
separación en el giro dentado, compleción en CA3, fronteras de evento y repetición
en el sueño hacia la corteza; amígdala que valora y libera neuromoduladores;
categorías sensoriales por vigilancia; memoria de asociación entre modalidades con
lectura contrastiva; mapas motores (voz y mano) que aprenden oyéndose y viéndose;
copia de eferencia de la voz; modo enseñanza con curva de aprendizaje; constantes de
escala humana en segundos reales.

## Bloque 0 — Lo innato

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Vía rápida tálamo→amígdala** con detectores fijos de prosodia: volumen, brusquedad del arranque, aspereza, altura y contorno del tono, duración | LeDoux (vía baja); Fernald 1993: bebés de 5 meses responden a aprobación/prohibición en idiomas desconocidos; Arnal 2015: la aspereza alarma sin aprendizaje | La amígdala solo recibe de las cortezas; la emoción venía de un diccionario de 364 palabras | Una voz suave, aguda y ascendente sube la valencia; una voz seca, grave y brusca la baja, digan lo que digan |
| **Sobresalto** ante un sonido súbito y fuerte | Reflejo de sobresalto acústico (tronco) | No existe | La norepinefrina y el cortisol suben en el mismo tick en que llega el sonido, antes de que la corteza auditiva lo clasifique |
| **Lo que se acerca** (looming) | Respuesta defensiva a expansión retiniana, presente en recién nacidos | No existe | Una mancha que crece deprisa alarma; la misma mancha que se aleja, no |
| **Sesgo hacia caras** | Johnson & Morton (CONSPEC): los recién nacidos siguen más un patrón de cara que el mismo invertido | No existe | Entre dos dibujos con los mismos trazos, el que tiene forma de cara recibe más atención (acetilcolina, oxitocina) |
| **Condicionamiento de miedo** de un ensayo, extinción lenta, recuperación espontánea | LTP en la amígdala lateral; la extinción es aprendizaje prefrontal, no borrado | `conditionResponse` existe pero nadie lo llama | Una figura vista durante un sobresalto asusta después; tras exposiciones seguras deja de hacerlo; con cortisol alto reaparece |
| **Presión de sueño** | Adenosina: se acumula con la actividad y se disipa durmiendo | Duerme por temporizador | Un cerebro muy estimulado duerme antes que uno en reposo |
| **Contacto social como recompensa** | La oxitocina responde a la voz y al contacto | Solo el 👍 | Oír una voz sube la oxitocina; una voz cálida más |
| **La voz como aprobación** | Referencia social: el bebé lee la voz del adulto para saber si algo está bien | Botones 👍 / 👎 | Una voz cálida tras un recuerdo lo refuerza como el 👍; una seca lo debilita |
| **Palabras afectivas aprendidas** | Condicionamiento clásico: la palabra oída junto a algo que ya asusta o calma adquiere su emoción | Diccionario precargado | "miedo" leída sola no produce nada; oída junto a una voz áspera, después sí |

Fuera: `affective-lexicon.ts`.

## Bloque 1 — Motivación y cuerpo

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Error de predicción de recompensa**: la dopamina es recompensa recibida menos esperada; cada categoría, palabra y asociación lleva un valor aprendido | Schultz 1997 | Hecho (`core/motivation`): cada cosa lleva lo que suele traer; la omisión de lo esperado es una bajada | Enseñar lo mismo diez veces: el pulso de dopamina cae hacia cero; algo nuevo lo devuelve |
| **Curiosidad** como recompensa intrínseca: bono por novedad y por progreso de aprendizaje | Kakade & Dayan 2002; Oudeyer | Hecho: bono de novedad que se habitúa; progreso al recordar mejor; progreso de los mapas motores (crecimiento de lo que saben) | Prefiere lo que le está enseñando algo y abandona lo que ya domina |
| **Adaptación hedónica**: la línea base de cada neuromodulador se desplaza hacia su media reciente | Desensibilización de receptores | Hecho: deriva hacia el nivel reciente dentro de una banda y vuelve al punto de ajuste | Un estado alto sostenido se normaliza |
| **Selección de acción** al estilo de los ganglios basales: balbucear, imitar o callar; garabatear, copiar o dibujar de memoria; dormir | Actor-crítico en estriado | Hecho para balbucear/garabatear: cada actividad vale el progreso que trae, elección softmax | Con dos actividades disponibles, dedica más tiempo a la que le hace progresar |
| **Impulsos**: aburrimiento, cansancio, contacto | Alostasis (Damasio, Sterling) | Hecho: aburrimiento acorta la pausa, soledad → llamada, presión de sueño | Sin interacción busca estímulos; con ella aprende más de esa sesión |
| **Hábitos**: lo repetido pasa de dirigido a objetivo (valor) a estímulo-respuesta (estriado), más rápido y sin atención | Dickinson: devaluación de la recompensa; Yin & Knowlton 2006 | **Hecho** (`core/motivation/habits.ts`): cada respuesta a una señal estampa un enlace estímulo-respuesta (0,12 por ejecución, +0,08 con elogio, −0,06·fuerza con reprimenda); pasado 0,6 el hábito responde a la señal por sí mismo, sin recuerdo ni valor de por medio, y se extingue despacio. El sistema dirigido a objetivo calla cuando la señal predice reprimenda (valor < −0,3) o el recuerdo se debilita | `habits.test.ts` | Una respuesta muy practicada sigue ejecutándose aunque su recompensa ya no valga; una poco practicada se detiene |

## Bloque 2 — El tiempo

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Secuencias**: el oído recibe la palabra como sucesión de sonidos; la mano dibuja a trazos | Células de tiempo, precesión de fase theta, cadenas sinápticas | Hecho en el oído: el micrófono manda cuadros cada 200 ms mientras se habla y la ventana del oído (10 cuadros) guarda el orden; dos vocales en un orden y en el otro son dos sonidos. Entre perceptos: `core/memory/sequence-memory` aprende qué sigue a qué y genera una expectativa (lo esperado no sorprende). Pendiente: la mano a trazos | Distingue "pa-ta" de "ta-pa" |
| **Fases de codificación y recuperación** separadas (ritmo theta) en el hipocampo | Hasselmo 2002 | Pendiente (umbrales ajustados a mano); se hará cuando el motor nuevo permita recalibrar sin romper los tests | Menos falsos reconocimientos con los mismos umbrales |
| **Memoria de trabajo con compuerta** | Ganglios basales → prefrontal (O'Reilly & Frank 2006) | Hecho: entra lo que sorprende o se recompensa (|error de predicción| ≥ 0,2), con esa prioridad; lo rutinario pasa de largo y lo retenido se desvanece sin refresco | Retiene lo relevante de una secuencia y suelta lo demás |
| **Orden de palabras** | Estadísticas de secuencia (Saffran 1996), aprendizaje de transiciones | **Hecho** (`core/language/word-order.ts`): cada frase leída vota el orden de los *tipos* de sus palabras (forma, color, sonido, según el registro palabra–referente); al producir, las palabras que un percepto trae de vuelta esperan un aliento (600 ms) y salen juntas en el orden ganador. Enseñado "coche verde", el coche azul es "coche azul"; enseñado "verde coche", "azul coche" | `word-order.test.ts`: dice "coche azul", no "azul coche"; al revés si se le enseña al revés |

A partir de aquí las **palabras entran por el oído** y **leer es ver**: el canal de
texto, el léxico precargado y el flujo de pensamiento en palabras (andamios) se
retiran, y el lenguaje se reconstruye desde el sonido.

## Bloque 3 — Predicción y jerarquía

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Codificación predictiva**: la corteza predice su entrada y aprende del error; la sorpresa es ese error | Rao & Ballard 1999; Friston | Hecho en su forma mínima: cada corteza sensorial mide la **sorpresa** de un percepto, 1 − coseno entre la entrada y lo que reconstruyen las neuronas que respondieron, antes de aprender de ella (la visual, en el primer tick con respuesta). La sorpresa alimenta la atención (norepinefrina, acetilcolina) y descuenta el bono de novedad; la expectativa de secuencia descuenta lo anunciado. Pendiente: que el error corrija la predicción capa a capa | La sorpresa cae con la repetición y sube con lo inesperado |
| **Retina con color** (canales oponentes) y **rasgos separados**: forma, color, tamaño, posición en poblaciones distintas | V4 color, corteza temporal forma | Hecho para el color: la retina emite además un código de color invariante a forma y posición (histograma tono × saturación) que va a una **corteza del color (V4)** propia con categorías por exposición y su modalidad en la asociación; la forma va por contraste centro-periferia (células ganglionares: un rectángulo relleno y un coche ya no son lo mismo). Pendiente: tamaño y posición como rasgos | "azul" dicho sobre tres objetos azules se aplica a un cuarto |
| **Composición** | Aprendizaje cross-situacional (Smith & Yu 2008) | Hecho: cada percepto (forma, color) recuerda por su lado; enseñado coche, verde, azul y coche verde, el coche azul nunca visto escribe "coche azul" (`composition.test.ts`) | coche, verde, azul, coche verde → el coche azul evoca "coche" y "azul" |
| **Dos niveles** con agrupamiento: invariancia a posición y tamaño | V1→V2→IT | **Hecho** (`regions/visual-cortex/parts-cortex.ts`): sobre V1 (plantillas de imagen entera) hay una corteza de partes: un diccionario de 24 partes locales compartidas por toda la retina (ventanas de 6×6 celdas del mapa de contraste, binarizadas y agrupadas 2×2, así un trazo grueso y uno fino son la misma parte), aplicado en una rejilla densa de posiciones; agrupamiento por cuadrantes y global (células complejas); neuronas de objeto por competición con vigilancia 0,5 y compromiso, categorías 'Object-n'. El engrama del objeto se une al de V1 en el código visual de la memoria de asociación (misma latencia talámica que V1) y al recordar manda la ruta más fuerte. Media figura de cuadrado: V1 ve un corchete nuevo, el nivel de objeto ve el cuadrado y vuelve «cuadrado». Límite honesto: media cruz (una T) es ambigua a este nivel; el tamaño se sigue resolviendo en el frente (zoom) | `hierarchy.test.ts`; `sensory-learning.test.ts` (hueco cerrado a nivel de objeto) |. Pendiente: la jerarquía de partes (hueco conocido: media figura recentrada no se reconoce como la figura) | Reconoce el mismo objeto desplazado y a otro tamaño |
| **Preguntas** como señal asociada a una dimensión, con atención por rasgo desde la prefrontal | Atención basada en rasgos (Treue) | Hecho: el registro palabra–referente (cross-situacional) dice a qué dimensión pertenece cada palabra; una pregunta seguida de respuestas de una misma dimensión pasa a **pedir esa dimensión**, y al oírla el cerebro responde con el nombre más específico de lo que hay delante en esa dimensión | "¿de qué color es?" → "azul"; "¿qué es?" → "coche" |

## Bloque 4 — Sentidos activos

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Ojo que se mueve**: sacadas sobre una imagen mayor que la fóvea, dirigidas por saliencia y por atención descendente | Colículo superior, campos oculares frontales | **Hecho** (saliencia ascendente): la retina segmenta la escena en regiones de contenido (rejilla gruesa 32×32), el ojo fija la mayor, la corteza la ve sola (forma y color) y, cerrada esa presentación, una sacada (1,2 s) lleva el ojo a la siguiente; cada objeto se visita una vez, máximo 4. Pendiente: dirección descendente por atención (la pregunta o la memoria de trabajo eligen dónde mirar) | `active-senses.test.ts` §1: una escena con la cruz y el cuadrado se mira en dos fijaciones y los nombra uno a uno; una cosa sola, una mirada |
| **Trazos**: modelo interno tipo cerebelo que aprende trayectorias y tiempos | Cerebelo (Wolpert, Kawato) | **Hecho**: el dashboard envía los trazos (puntos y duración) con el dibujo; la mano guarda por dibujo los gestos vistos (orden, dirección, tiempos; el más repetido manda) y al copiar tiende sus propias celdas a lo largo del gesto del modelo que tiene delante, trazo a trazo y con su tiempo; el dashboard anima la copia. Lo nunca visto hacer se dibuja de golpe | `active-senses.test.ts` §3: vista la cruz trazarse horizontal→vertical, la copia así, con esos tiempos; vista más veces al revés, cambia; el cuadrado, de golpe |
| **Consonantes**: dinámica temporal del tracto vocal | Articuladores | **Hecho**: una emisión se despliega en el tiempo, una trama de 200 ms de inicio (murmullo nasal /m/: formante nasal 300 Hz y resonancias débiles; o ráfaga de liberación /p/: banda ancha) y luego la vocal, en la misma ventana del oído. Los labios son tres unidades del mapa motor (abiertos, nasal, oclusiva) que aprenden y leen solo los primeros ticks de lo oído; la mitad del balbuceo empieza con los labios cerrados | `active-senses.test.ts` §2: tras balbucear, repite m+vocal con los labios cerrados y la nariz abierta, p+vocal con liberación y la vocal sola con los labios abiertos (la /i/ sola puede confundirse con /m/: comparten el primer formante bajo) |

## Bloque 5 — Desarrollo

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Periodos críticos**: la plasticidad de una categoría baja con su edad | Estrechamiento perceptivo (Werker & Tees 1984); imán perceptivo (Kuhl 1991) | **Hecho**: cada categoría se atrinchera con sus exposiciones (τ = 10) y sus neuronas relajan la vigilancia hasta un 35 %: capturan lo cercano en vez de dejar que funde categoría propia | `development.test.ts` §6: dos sonidos cercanos oídos desde el principio son dos; el segundo, oído solo cuando el primero está gastado, se asimila |
| **Poda** de lo que no se usa | Poda sináptica postnatal | **Hecho**: al dormir, una categoría vista menos de 2 veces y no vista en 2 sueños desaparece y sus neuronas se liberan (sinapsis al 20 %) | `development.test.ts` §7: el cuadrado visto una vez se poda; la cruz vista cuatro veces queda; el cuadrado vuelve a ser nuevo |
| **Fases de sueño**: repetición (lento) y reactivación al azar (REM) | Diekelmann & Born 2010 | **Hecho**: NREM (repetición de episodios hacia la corteza prefrontal, rebaja sináptica) y REM (acetilcolina arriba; CA3 completa quimeras de dos episodios que la corteza ve con plasticidad 0,3; lo visual conocido se repite desplazado y con ruido como variantes) | `development.test.ts` §5. Pendiente medir «generaliza mejor tras dormir» con un test propio |
| **Imaginación despierta** (red por defecto): sin estímulo externo la dopamina baja y el aburrimiento sube; en vez de esperar, el cerebro reactiva recuerdos por su cuenta (repetición despierta en el hipocampo), los recombina en pistas parciales que CA3 completa, pasa el resultado por las cortezas y el léxico como imagen y palabras, y el mismo sistema de valor lo recompensa si lo imaginado es nuevo y luego resulta útil (progreso), con descuento para que no se autoestimule en bucle. Lo imaginado lleva marca de origen interno (monitorización de realidad: Johnson) y no se guarda como episodio del mundo | Red por defecto (Raichle 2001); divagación mental (Smallwood & Schooler 2015); repetición despierta (Foster & Wilson 2006); recombinación (Schacter & Addis) | **Hecho**: a los 5 s sin percibir nada, «soñar despierto» entra como actividad elegible (softmax con balbucear y garabatear, valor aprendido). Dos cosas conocidas al azar → pista mixta (mitad y mitad si son de un tipo; lado a lado si no) → la memoria de asociación la completa en palabras y en los otros sentidos, la corteza visual la dibuja en el ojo de la mente; la mano la dibuja si está encendida. Recompensa 0,4 × novedad / veces imaginado; 0,5 cuando lo imaginado aparece de verdad. Marca de origen: no funda categoría, ni episodio, ni asociación | Sin estímulos durante un rato, el flujo de pensamiento y la pizarra muestran combinaciones nuevas de lo aprendido; algunas aparecen después en lo que dice o dibuja; nunca las confunde con cosas vistas |
| **Sueños**: en REM (acetilcolina alta, norepinefrina y serotonina bajas, sin control prefrontal) se mezclan fragmentos de varios episodios y CA3 los completa en quimeras que pasan por la corteza con plasticidad baja; el dashboard muestra lo que "ve" y "dice" dormido | Hoel 2021 (cerebro sobreajustado); generative replay (van de Ven 2020); Stickgold & Walker | **Hecho**: 3 sueños por noche con el mismo motor; el dashboard muestra lo soñado (💭/🌙 en el flujo, ojo de la mente); al despertar el sueño queda en el pensamiento | Tras dormir dibuja o nombra algo que combina dos cosas aprendidas y nunca vio juntas, y reconoce mejor variantes nuevas de lo aprendido |
| **Reconsolidación y olvido activo** | Nader 2000 | Parcial: una re-experiencia refuerza y refecha el episodio; las señales condicionadas se extinguen; el trazo no se modifica al reactivarse | Un recuerdo reactivado se puede modificar |

## Después: el motor en C++ (CPU y CUDA)

Sinapsis dispersas, interneuronas reales y oscilaciones en lugar de la competición
algorítmica, neuromoduladores regionales, cómputo dendrítico. Hitos: paridad con los
tests actuales a 10 000 neuronas; luego 100 000 y 1 000 000 midiendo ticks por
segundo en CPU y en una GPU alquilada. Ver `README.md` para `BRAIN_TICK_HZ`.

**Estado (2026-09-19)**: el núcleo existe (`native/`, ver `docs/ENGINE.md`): neuronas de
Izhikevich en arrays con paridad probada contra el modelo TypeScript, sinapsis CSR con
transpuesta, interneuronas inhibitorias reales, STDP por trazas modulada, hilos en CPU,
kernels CUDA escritos (sin compilar: no hay GPU aquí), addon de Node y `gbrain-bench`.
En este Mac un millón de neuronas con 100 sinapsis corre a ~58 ticks/s (5,8× tiempo real a
10 Hz) con ~0,2 % de actividad. Primera región sobre el motor hecha: la corteza nativa
(`GBRAIN_NATIVE=1`; asambleas excitatorias reproducibles y dispersas sin competición
algorítmica). El motor tiene corrientes sinápticas exponenciales, depresión a corto plazo,
plasticidad estructural e inhibitoria, y la corteza nativa aferentes topográficas: con eso la
**compleción de patrón es real** (media entrada trae de vuelta el 85–100 % del lado de la
asamblea que la pista no alcanza, con especificidad y sin atractores permanentes).
Falta: recurrentes con estructura y proyecciones entre regiones en el CSR, neuromoduladores
por región, retardos y oscilaciones, dendritas, y medir en GPU.

## Escalera de hitos (criterio de éxito global)

1. Palabras de atributo generalizan a un objeto nuevo.
2. Composición: el coche azul evoca "coche" y "azul".
3. Lo dice en orden: "coche azul". **Hecho.**
4. Responde "¿de qué color es?" y "¿qué es?".
5. Un objeto nuevo de un color conocido: responde el color aunque no sepa el nombre.
