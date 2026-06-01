# gBrain (Cerebro Digital)

Simulación bio-inspirada de un cerebro humano implementado de forma artificial. Nombre de producto: **gBrain**; nombre técnico del paquete: `digital-brain`. Usuarios externos envían estímulos sensoriales; un proceso servidor los integra, simula regiones cerebrales y acumula experiencia mediante aprendizaje.

## Language

**Brain simulation**:
El núcleo del sistema: redes de neuronas pulsantes, regiones funcionales, bus de spikes, neuromoduladores y memoria. Todo lo que ocurre tras convertir un estímulo en spikes hasta que el estado cerebral cambia.
_Avoid_: Cerebro, modelo, AI brain (demasiado genérico o implica LLM)

**Sensory client**:
Interfaz que captura estímulos del usuario (texto, imagen, audio), los preprocesa en el dispositivo y los envía al servidor para su procesamiento neural.
_Avoid_: Frontend, dashboard (el dashboard es una implementación concreta del sensory client)

**Brain server**:
Proceso que aloja la simulación cerebral, recibe estímulos ya preprocesados, los codifica a spikes si hace falta, ejecuta ticks y persiste el aprendizaje entre sesiones.
_Avoid_: API, backend (son mecanismos de transporte, no el concepto)

**Stimulus**:
Cualquier entrada sensorial que un usuario envía al sistema (texto leído, imagen vista, sonido oído).
_Avoid_: Input, mensaje (demasiado genérico)

**Learning**:
Desarrollo por experiencia: los estímulos entran, se agrupan y se vinculan entre sí; la repetición refuerza esos vínculos hasta que el sistema puede actuar (responder) cuando ya reconoce asociaciones suficientes.
_Avoid_: Training, fine-tuning (implica ML clásico o LLM)

**Classification**:
Agrupar experiencias similares y distinguirlas de otras, sin exigir todavía una respuesta verbal.
_Avoid_: Labeling, categorización supervisada (implica etiquetas dadas de antemano)

**Reinforcement**:
Fortalecimiento de un trazo o asociación cada vez que el mismo estímulo (o uno muy parecido) vuelve a aparecer.
_Avoid_: Reward, gradient (lenguaje de RL o backprop)

**Response**:
Salida del sistema cuando la activación de asociaciones alcanza umbral (p. ej. lenguaje, emoción visible, acción) — no en el primer contacto con un estímulo nuevo.
_Avoid_: Output, reply (demasiado genérico; no implica que hubo aprendizaje previo)

**Word form**:
La forma observable de una unidad lingüística (sonido, grafema, secuencia de letras como «luna»).
_Avoid_: Token, palabra (sin distinguir de sentido)

**Sense**:
Un significado posible de una misma word form en un contexto concreto (p. ej. astro nocturno vs nombre de mascota).
_Avoid_: Definición, concepto (demasiado abstracto o de diccionario)

**Association**:
Vínculo entre un patrón neural y el contexto simultáneo (otros estímulos, emoción, región activa) en el momento de la experiencia.
_Avoid_: Embedding, feature (lenguaje de ML)
