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
| **Hábitos**: lo repetido pasa de dirigido a objetivo (valor) a estímulo-respuesta (estriado), más rápido y sin atención | Dickinson: devaluación de la recompensa; Yin & Knowlton 2006 | La asociación sube de confianza, pero nada cambia de sistema | Una respuesta muy practicada sigue ejecutándose aunque su recompensa ya no valga; una poco practicada se detiene |

## Bloque 2 — El tiempo

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Secuencias**: el oído recibe la palabra como sucesión de sonidos; la mano dibuja a trazos | Células de tiempo, precesión de fase theta, cadenas sinápticas | Hecho en el oído: el micrófono manda cuadros cada 200 ms mientras se habla y la ventana del oído (10 cuadros) guarda el orden; dos vocales en un orden y en el otro son dos sonidos. Entre perceptos: `core/memory/sequence-memory` aprende qué sigue a qué y genera una expectativa (lo esperado no sorprende). Pendiente: la mano a trazos | Distingue "pa-ta" de "ta-pa" |
| **Fases de codificación y recuperación** separadas (ritmo theta) en el hipocampo | Hasselmo 2002 | Pendiente (umbrales ajustados a mano); se hará cuando el motor nuevo permita recalibrar sin romper los tests | Menos falsos reconocimientos con los mismos umbrales |
| **Memoria de trabajo con compuerta** | Ganglios basales → prefrontal (O'Reilly & Frank 2006) | Hecho: entra lo que sorprende o se recompensa (|error de predicción| ≥ 0,2), con esa prioridad; lo rutinario pasa de largo y lo retenido se desvanece sin refresco | Retiene lo relevante de una secuencia y suelta lo demás |
| **Orden de palabras** | Estadísticas de secuencia, aprendizaje de transiciones | La memoria de secuencias ya guarda transiciones entre palabras leídas; producir en orden espera a que las palabras entren por el oído | Dice "coche azul", no "azul coche" |

A partir de aquí las **palabras entran por el oído** y **leer es ver**: el canal de
texto, el léxico precargado y el flujo de pensamiento en palabras (andamios) se
retiran, y el lenguaje se reconstruye desde el sonido.

## Bloque 3 — Predicción y jerarquía

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Codificación predictiva**: la corteza predice su entrada y aprende del error; la sorpresa es ese error | Rao & Ballard 1999; Friston | La novedad se mide a posteriori como cambio de patrón | La sorpresa cae con la repetición y sube con lo inesperado |
| **Retina con color** (canales oponentes) y **rasgos separados**: forma, color, tamaño, posición en poblaciones distintas | V4 color, corteza temporal forma | Retina en gris, categorías por imagen completa | "azul" dicho sobre tres objetos azules se aplica a un cuarto |
| **Composición** | Aprendizaje cross-situacional (Smith & Yu 2008) | Lectura contrastiva de un solo ganador | coche, verde, azul, coche verde → el coche azul evoca "coche" y "azul" |
| **Dos niveles** con agrupamiento: invariancia a posición y tamaño | V1→V2→IT | Una capa | Reconoce el mismo objeto desplazado y a otro tamaño |
| **Preguntas** como señal asociada a una dimensión, con atención por rasgo desde la prefrontal | Atención basada en rasgos (Treue) | Ninguno | "¿de qué color es?" → "azul"; "¿qué es?" → "coche" |

## Bloque 4 — Sentidos activos

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Ojo que se mueve**: sacadas sobre una imagen mayor que la fóvea, dirigidas por saliencia y por atención descendente | Colículo superior, campos oculares frontales | Foveación fija en el centro del contenido | Explora una escena con varios objetos y los nombra uno a uno |
| **Trazos**: modelo interno tipo cerebelo que aprende trayectorias y tiempos | Cerebelo (Wolpert, Kawato) | Dibuja de golpe | Copia un dibujo con el orden de trazos de una persona |
| **Consonantes**: dinámica temporal del tracto vocal | Articuladores | Dos formantes fijos | Imita "ma" y "pa" |

## Bloque 5 — Desarrollo

| Mecanismo | Base | Hoy | Test |
|---|---|---|---|
| **Periodos críticos**: la plasticidad de una categoría baja con su edad | Estrechamiento perceptivo (Werker & Tees 1984) | Vigilancia sin edad | Un contraste no oído en los primeros meses cuesta más aprenderlo después |
| **Poda** de lo que no se usa | Poda sináptica postnatal | Solo rebaja en el sueño | Las categorías que nunca se repiten desaparecen |
| **Fases de sueño**: repetición (lento) y reactivación al azar (REM) | Diekelmann & Born 2010 | Solo repetición y rebaja | Generaliza mejor tras dormir |
| **Imaginación despierta** (red por defecto): sin estímulo externo la dopamina baja y el aburrimiento sube; en vez de esperar, el cerebro reactiva recuerdos por su cuenta (repetición despierta en el hipocampo), los recombina en pistas parciales que CA3 completa, pasa el resultado por las cortezas y el léxico como imagen y palabras, y el mismo sistema de valor lo recompensa si lo imaginado es nuevo y luego resulta útil (progreso), con descuento para que no se autoestimule en bucle. Lo imaginado lleva marca de origen interno (monitorización de realidad: Johnson) y no se guarda como episodio del mundo | Red por defecto (Raichle 2001); divagación mental (Smallwood & Schooler 2015); repetición despierta (Foster & Wilson 2006); recombinación (Schacter & Addis) | Nada: sin entrada el pensamiento se vacía | Sin estímulos durante un rato, el flujo de pensamiento y la pizarra muestran combinaciones nuevas de lo aprendido; algunas aparecen después en lo que dice o dibuja; nunca las confunde con cosas vistas |
| **Sueños**: en REM (acetilcolina alta, norepinefrina y serotonina bajas, sin control prefrontal) se mezclan fragmentos de varios episodios y CA3 los completa en quimeras que pasan por la corteza con plasticidad baja; el dashboard muestra lo que "ve" y "dice" dormido | Hoel 2021 (cerebro sobreajustado); generative replay (van de Ven 2020); Stickgold & Walker | Nada | Tras dormir dibuja o nombra algo que combina dos cosas aprendidas y nunca vio juntas, y reconoce mejor variantes nuevas de lo aprendido |
| **Reconsolidación y olvido activo** | Nader 2000 | Ninguno | Un recuerdo reactivado se puede modificar |

## Después: el motor en C++ (CPU y CUDA)

Sinapsis dispersas, interneuronas reales y oscilaciones en lugar de la competición
algorítmica, neuromoduladores regionales, cómputo dendrítico. Hitos: paridad con los
tests actuales a 10 000 neuronas; luego 100 000 y 1 000 000 midiendo ticks por
segundo en CPU y en una GPU alquilada. Ver `README.md` para `BRAIN_TICK_HZ`.

## Escalera de hitos (criterio de éxito global)

1. Palabras de atributo generalizan a un objeto nuevo.
2. Composición: el coche azul evoca "coche" y "azul".
3. Lo dice en orden: "coche azul".
4. Responde "¿de qué color es?" y "¿qué es?".
5. Un objeto nuevo de un color conocido: responde el color aunque no sepa el nombre.
