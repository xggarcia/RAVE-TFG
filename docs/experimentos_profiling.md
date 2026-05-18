# Catálogo de Experimentos de Profiling — RAVE-TFG

Cada experimento produce **una sesión CSV** + **8 gráficas PNG/PDF**.  
Procedimiento estándar para todos:
1. Iniciar la app (`python -m app`)
2. Configurar según los parámetros indicados
3. Pulsar **Start stream** (el profiler arranca automáticamente)
4. Esperar la duración indicada
5. Pulsar **Stop**
6. Ejecutar `python tools/plot_profiling.py` (auto-detecta la sesión más reciente)
7. Guardar las gráficas relevantes con el nombre del experimento

---

## Grupo A — Línea Base y Escalabilidad

### EXP-01 · Baseline mínimo
**Objetivo:** Establecer el rendimiento de referencia con carga mínima.  
**Configuración:**
- 1 modelo activo (demo_model o cualquier modelo entrenado)
- Input mode: `random`
- Temperatura: 0.6 · Smooth: 0.4 · Gain: 0.6
- Sin prior, sin recording, sin gesture

**Duración:** 3 minutos  
**Gráficas clave:** `_rtf`, `_inference`, `_ram`  
**Hipótesis:** RTF estable por debajo de 0.4 después del spike inicial de warmup. RAM plana. Jitter < 2 ms en estado estacionario.  
**Uso en TFG:** Punto de referencia para todos los demás experimentos.

---

### EXP-02 · Escalado de modelos — carga incremental
**Objetivo:** Medir cómo escala el RTF al añadir modelos concurrentes.  
**Configuración:**
- Empieza con 1 modelo activo
- Cada 20 segundos, activa un slot adicional (máximo 4)
- Todos en input mode `random`, parámetros por defecto

**Duración:** 90 segundos (4 etapas × ~20s)  
**Gráficas clave:** `_rtf`, `_inference`, `_cpu`, `_dashboard`  
**Hipótesis:** RTF escala aproximadamente lineal con el número de modelos. El algoritmo Adaptive Stride debería incrementar `decode_stride` cuando se supere el 95 % del budget.  
**Métricas a anotar:** RTF en cada etapa, `inference_mean_ms` por modelo añadido.

---

### EXP-03 · Punto de ruptura — ¿cuántos modelos aguanta?
**Objetivo:** Encontrar el límite práctico de modelos simultáneos antes de underruns persistentes.  
**Configuración:**
- Añade 1 modelo cada 15 segundos hasta que el sistema produzca underruns continuos o el audio sea claramente defectuoso
- Anota el número de modelos al que empieza la degradación

**Duración:** Hasta degradación o máximo de slots disponibles  
**Gráficas clave:** `_rtf`, `_queue`, `_stride`, `_dashboard`  
**Hipótesis:** El límite estará determinado por el número de cores físicos disponibles y el tamaño del modelo. Adaptive Stride extenderá la vida útil del sistema.  
**Métricas a anotar:** N modelos al primer underrun, N al primer salto de stride, RTF máximo.

---

### EXP-04 · Comparación de modos de rendimiento --> Ya no tenemos estos modos. !!!!!
**Objetivo:** Comparar los tres perfiles de `StreamingEngine` (Quality / Balanced / Max Stability).  
**Configuración:** Repetir EXP-01 tres veces, una por cada performance_mode  
**Nota:** Requiere modificar el parámetro en `_stream_builders.py` o añadir un selector en la UI.  
**Duración:** 2 minutos por modo (6 min total)  
**Gráficas clave:** `_rtf`, `_queue` (superponer las tres sesiones manualmente)  
**Hipótesis:** Max Stability aumenta la cola pero tolera más modelos; Quality tiene menos latencia pero más underruns bajo carga.

---

## Grupo B — Algoritmo Adaptive Stride

### EXP-05 · Activación del Adaptive Stride bajo presión artificial
**Objetivo:** Demostrar el algoritmo en acción provocando overload.  
**Configuración:**
- 2 modelos activos
- Mientras graba, abrir otras aplicaciones pesadas (compilador, editor de vídeo) para saturar la CPU durante 20-30 s y luego cerrarlas

**Duración:** 3 minutos  
**Gráficas clave:** `_stride`, `_rtf`, `_inference`, `_cpu`  
**Hipótesis:** `decode_stride` sube durante el overload artificial y baja cuando la CPU se libera. El RTF debería mantenerse por debajo de 1.0 gracias al stride.

---

### EXP-06 · Stride forzado vs adaptativo — comparación de calidad percibida
**Objetivo:** Documentar el trade-off latencia/calidad del stride.  
**Configuración:** Tres sub-sesiones con `base_decode_stride` fijado a 1, 2 y 4 respectivamente (editar `engine_core.py` temporalmente)  
**Duración:** 2 minutos por configuración  
**Gráficas clave:** `_inference`, `_rtf`, `_queue`  
**Hipótesis:** Stride alto reduce RTF pero degrada la textura del audio (repetición de chunks cacheados). Las gráficas mostrarán la diferencia cuantitativa aunque la degradación perceptual sea subjetiva.

---

### EXP-07 · Recovery time — velocidad de bajada de stride
**Objetivo:** Medir cuántos segundos tarda el sistema en recuperar stride=1 tras un overload.  
**Configuración:**
- Empieza con 3 modelos (para que stride suba a 2 o 3)
- A los 60 s, desactiva 2 slots
- Mide el tiempo hasta stride=1

**Duración:** 3 minutos  
**Gráficas clave:** `_stride` (inspección visual del tiempo de bajada), `_rtf`  
**Hipótesis:** La condición de bajada es `_stable_cycles >= 25`, lo que a ~10-20 chunks/s implica 1-2.5 segundos. Verificar empíricamente.

---

## Grupo C — Warmup y Compilación JIT

### EXP-08 · Cold start vs warm start (caché JIT)
**Objetivo:** Cuantificar el spike de primera inferencia (TorchScript JIT compilation).  
**Configuración:**
- **Cold start:** Arrancar la app desde cero, stream inmediato con 1 modelo
- **Warm start:** Detener y volver a iniciar stream **sin** cerrar la app

**Duración:** 30 segundos por sub-sesión  
**Gráficas clave:** `_inference` (primeros 10 s), `_rtf` (primeros 10 s)  
**Hipótesis:** El cold start muestra un spike de `inference_peak_ms` > 200 ms en el primer chunk; el warm start empieza directamente en el rango estacionario (~15-25 ms).  
**Uso en TFG:** Justifica el diseño de la etapa de carga con `QThread` y los stages de progreso.

---

### EXP-09 · Warmup por tamaño de modelo
**Objetivo:** Comparar el spike JIT entre un modelo pequeño y uno grande.  
**Configuración:** Repetir EXP-08 cold start con dos modelos de distinto tamaño  
**Duración:** 30 segundos cada uno  
**Gráficas clave:** `_inference` (primeros 15 s de cada sesión)  
**Hipótesis:** Modelos más grandes producen spikes iniciales más largos. Documentar los valores exactos.

---

## Grupo D — Recursos del Sistema

### EXP-10 · Test de memory leak — sesión larga
**Objetivo:** Demostrar que la aplicación no tiene fugas de memoria.  
**Configuración:**
- 2 modelos activos, parámetros por defecto
- No realizar ninguna interacción durante la sesión

**Duración:** 15 minutos (o más si el tiempo lo permite)  
**Gráficas clave:** `_ram` (lo más importante), `_rtf`  
**Hipótesis:** RAM RSS plana después de la carga inicial. Cualquier pendiente positiva sostenida indicaría un leak. Resultado esperado: pendiente ≈ 0 MB/min en estado estacionario.  
**Uso en TFG:** Prueba directa de correctness del pipeline de audio con numpy.

---

### EXP-11 · Uso de CPU por número de modelos (muestreo fino)
**Objetivo:** Construir una curva CPU% vs N modelos para la sección de análisis.  
**Configuración:** 5 sub-sesiones de 2 minutos con N = 1, 2, 3, 4, (5 si hay slots)  
**Duración:** 10 minutos total  
**Gráficas clave:** `_cpu` de cada sesión + gráfica manual de dispersión CPU vs N  
**Métricas a anotar:** `cpu_process_pct` promedio de los últimos 60 s de cada sesión.

---

### EXP-12 · Presión de memoria del sistema
**Objetivo:** Ver si el sistema se degrada cuando la RAM del sistema está casi llena.  
**Configuración:**
- Abrir otras aplicaciones hasta usar >70% de la RAM total del sistema
- Lanzar 2 modelos y medir

**Duración:** 3 minutos  
**Gráficas clave:** `_ram`, `_rtf`, `_inference`  
**Hipótesis:** Si hay page faults, `inference_peak_ms` aumentará esporádicamente. Un RTF estable indicaría que el working set de PyTorch cabe en RAM física.

---

## Grupo E — Parámetros de Audio

### EXP-13 · Efecto del block size en latencia y RTF
**Objetivo:** Cuantificar el trade-off latencia/estabilidad para distintos tamaños de bloque.  
**Configuración:** Cambiar `block_size` en la llamada a `StreamWorker` a 256, 512, 1024, 2048 samples  
**Duración:** 2 minutos por configuración  
**Gráficas clave:** `_inference`, `_rtf`, `_queue` de cada sub-sesión  
**Hipótesis:** Bloques pequeños (256) dan menos latencia pero más underruns porque el budget_ms es solo 5.8 ms. Bloques grandes (2048) tienen 46 ms de budget y mayor estabilidad.  
**Uso en TFG:** Justifica la elección de block_size por defecto.

---

### EXP-14 · Input mode: random vs audio file
**Objetivo:** Comparar la carga computacional entre los dos modos de entrada.  
**Configuración:**
- Sub-sesión A: input mode `random`, 1 modelo
- Sub-sesión B: input mode `audio`, mismo modelo, archivo WAV de 30 s

**Duración:** 2 minutos cada uno  
**Gráficas clave:** `_inference`, `_cpu`  
**Hipótesis:** El modo `audio` añade el coste de la codificación (encode) más la lectura del buffer de latents, pero la inferencia debería ser similar ya que `model.decode(z)` es el mismo paso.

---

### EXP-15 · Overhead del modelo Prior
**Objetivo:** Medir el coste del prior en RTF e inferencia.  
**Configuración:**
- Sub-sesión A: 1 modelo, prior OFF
- Sub-sesión B: 1 modelo, prior ON (mismo checkpoint)

**Duración:** 2 minutos cada uno  
**Gráficas clave:** `_inference`, `_rtf`  
**Hipótesis:** El prior añade un segundo forward pass. Se espera que `inference_mean_ms` se duplique aproximadamente.

---

### EXP-16 · Parámetro Temperature — ¿afecta al tiempo de inferencia?
**Objetivo:** Verificar que los parámetros de síntesis (temperatura) no alteran el tiempo de cómputo.  
**Configuración:** 5 sub-sesiones de 60 s con temperatura = 0.1, 0.5, 1.0, 2.0, 3.0  
**Duración:** 5 minutos total  
**Gráficas clave:** `_inference` de cada sub-sesión (solo los valores medios)  
**Hipótesis:** Temperatura solo escala el tensor `z` antes del decode; el tiempo de `model.decode(z)` es idéntico para cualquier temperatura.

---

### EXP-17 · Overhead del Recording
**Objetivo:** Medir si activar la grabación de audio afecta al rendimiento en tiempo real.  
**Configuración:**
- Sub-sesión A: 2 modelos, recording OFF
- Sub-sesión B: 2 modelos, recording ON (grabando a disco)

**Duración:** 3 minutos cada uno  
**Gráficas clave:** `_rtf`, `_inference`, `_cpu`  
**Hipótesis:** El recording usa un `deque` de chunks + escritura asíncrona. Se espera un overhead mínimo en RTF pero visible en `cpu_process_pct`.

---

## Grupo F — Estabilidad y Condiciones Extremas

### EXP-18 · Sesión continua larga — deriva temporal
**Objetivo:** Detectar si el rendimiento se degrada con el tiempo (fragmentation, cache eviction, thermal throttling).  
**Configuración:**
- 2 modelos activos, parámetros por defecto
- No interactuar

**Duración:** 20-30 minutos  
**Gráficas clave:** `_rtf`, `_inference`, `_ram`, `_dashboard`  
**Hipótesis:** Si el sistema está bien diseñado, todas las métricas deben ser estacionarias. Cualquier pendiente creciente en `inference_mean_ms` o `ram_rss_mb` indica un problema.

---

### EXP-19 · Overload deliberado y recuperación
**Objetivo:** Verificar que el sistema no se corrompe bajo condiciones de overload severo y se recupera limpiamente.  
**Configuración:**
- Inicia con el máximo de modelos soportados + 1 (para garantizar underruns)
- Deja 30 s en overload
- Desactiva todos los modelos excepto 1
- Espera 60 s más y observa la recuperación

**Duración:** 2 minutos  
**Gráficas clave:** `_rtf`, `_stride`, `_queue`, `_inference`  
**Hipótesis:** `underruns_delta` sube durante el overload, luego cae a 0. `decode_stride` sube y baja. RAM no crece durante el overload (no hay leak por overrun).

---

### EXP-20 · Hot-swap de modelo — cambio en caliente
**Objetivo:** Medir el impacto en rendimiento de cambiar un modelo mientras se hace streaming.  
**Configuración:**
- 1 modelo activo durante 60 s
- Cambiar el modelo del slot 1 por otro diferente (sin detener el stream)
- Esperar 60 s más

**Duración:** 2 minutos  
**Gráficas clave:** `_inference` (buscar spike en el momento del swap), `_rtf`, `_ram`  
**Hipótesis:** Un spike de RTF en el momento del swap (carga del nuevo modelo). RAM aumenta temporalmente si ambos modelos están en memoria durante la transición.

---

### EXP-21 · Gesture control — overhead del control de fase
**Objetivo:** Medir si el procesamiento de la curva de gestos añade carga al audio thread.  
**Configuración:**
- Sub-sesión A: 2 modelos, gesture OFF
- Sub-sesión B: 2 modelos, gesture ON con curva activa

**Duración:** 2 minutos cada uno  
**Gráficas clave:** `_cpu`, `_inference`  
**Hipótesis:** El gesture control opera sobre `latent_bias` en el producer loop. Overhead esperado < 0.5 ms de `producer_mean_ms`.

---

### EXP-22 · Smoothing extremo — coste del filtro de latents
**Objetivo:** Medir si el smoothing alto (interpolación de z) tiene coste computacional.  
**Configuración:**
- Sub-sesión A: smooth = 0.0 (sin interpolación)
- Sub-sesión B: smooth = 0.95 (interpolación pesada)

**Duración:** 2 minutos cada uno  
**Gráficas clave:** `_inference` (solo `inference_mean_ms`)  
**Hipótesis:** Smoothing es solo `z = a*z_prev + (1-a)*z`, una operación en tensor pequeño. Diferencia esperada < 0.1 ms (no significativa).

---

### EXP-23 · Throttling térmico — sesión de máxima carga prolongada
**Objetivo:** Detectar si la CPU reduce su frecuencia por temperatura después de varios minutos de carga máxima.  
**Configuración:**
- Máximo de modelos soportados (o 3+)
- Sin intervención

**Duración:** 15 minutos  
**Gráficas clave:** `_inference` (buscar tendencia creciente lenta), `_cpu`, `_rtf`  
**Hipótesis:** En portátiles o sistemas sin buena disipación, `inference_mean_ms` aumenta gradualmente a partir del minuto 5-8 por throttling. En sistemas con buena refrigeración, permanece estable.

---

### EXP-24 · Arranque múltiple — variabilidad entre sesiones
**Objetivo:** Medir la variabilidad run-to-run para evaluar la reproducibilidad del sistema.  
**Configuración:** Repetir EXP-01 (baseline) 5 veces, reiniciando la app entre cada ejecución  
**Duración:** 3 min × 5 = 15 minutos  
**Gráficas clave:** `_rtf` de las 5 sesiones (superpuestas manualmente)  
**Hipótesis:** Las sesiones deberían ser muy similares después del warmup JIT. La variabilidad de `inference_mean_ms` en estado estacionario debería ser < 5 %.

---

### EXP-25 · Comparativa con y sin psutil (overhead del profiler)
**Objetivo:** Verificar que el propio profiler no altera las métricas que mide.  
**Configuración:**
- Sub-sesión A: `RAVE_PROFILE=1` (profiler activo)
- Sub-sesión B: `RAVE_PROFILE=0` (profiler desactivado), medir CPU con monitor externo

**Duración:** 3 minutos cada uno  
**Gráficas clave:** `_cpu`, `_inference` de la sesión A  
**Hipótesis:** El profiler duerme ~1 s entre muestras y solo hace lecturas no bloqueantes. Overhead esperado < 0.1 % CPU. Si hay diferencia medible, documentarla como limitación del sistema de medición.

---

## Resumen de prioridades para el TFG

| Prioridad | Experimentos | Sección del TFG |
|---|---|---|
| **Esencial** | EXP-01, 02, 03, 08, 10 | Evaluación de rendimiento |
| **Alta** | EXP-05, 06, 07, 13, 18 | Algoritmo Adaptive Stride + elección de parámetros |
| **Media** | EXP-09, 11, 14, 15, 19 | Análisis de recursos y trade-offs |
| **Complementaria** | EXP-04, 16, 17, 20–25 | Robustez, reproducibilidad, limitaciones |

## Convención de nombres de sesiones

Al terminar cada experimento, renombrar el CSV antes de ejecutar el plot:

```
session_YYYYMMDD_HHMMSS.csv  →  exp01_baseline_01.csv
                                  exp02_scaling_4models.csv
                                  exp05_stride_cpu_stress.csv
```

```bash
# Ejemplo: plotear un experimento renombrado
python tools/plot_profiling.py tools/profiling_sessions/exp02_scaling_4models.csv
```

Las gráficas se guardarán con el mismo stem (`exp02_scaling_4models_rtf.png`, etc.).
