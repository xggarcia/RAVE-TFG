# Guía: Entrenar y Usar Priors con MSPrior

## ¿Qué es un Prior?

Un **prior** es un modelo autoregressivo que aprende los patrones temporales en el espacio latente de RAVE. A diferencia del muestreo aleatorio, el prior genera secuencias latentes estructuradas basadas en lo que aprendió del dataset.

## ¿Por Qué Usar un Prior?

### Ventajas:
- ✅ **Generación más coherente**: Patrones temporales aprendidos del dataset
- ✅ **Similitud con datos de entrenamiento**: Sonidos más cercanos al estilo original
- ✅ **Estructura musical**: Mejor continuidad y desarrollo temporal
- ✅ **Control en Max/MSP**: Integración nativa con nn~

### Cuándo NO usar Prior:
- ❌ Si quieres máxima creatividad/exploración aleatoria
- ❌ Si no tienes el dataset original del RAVE
- ❌ Si solo usarás streaming en Python (random latent funciona excelente)

## Requisitos

1. **Modelo RAVE entrenado por TI** con acceso a checkpoints (.ckpt)
   - ⚠️ **NO funciona con modelos descargados o .ts exportados**
   - ⚠️ **Debes tener la carpeta de checkpoints del entrenamiento**
2. **Dataset de audio ORIGINAL** usado para entrenar RAVE
3. **MSPrior instalado**:
   ```bash
   pip install acids-msprior
   ```

### ⚠️ Limitación Importante

**MSPrior SOLO funciona con checkpoints de entrenamiento (.ckpt), NO con archivos exportados (.ts)**

- ✅ Correcto: `models/user_model/checkpoints/last.ckpt`
- ❌ Incorrecto: `models/user_model/exported_model/model.ts`

Si solo tienes un modelo .ts descargado, **no puedes entrenar un prior** para él. Necesitas haber entrenado el modelo RAVE tú mismo.

## Paso a Paso

### Método 1: Menú Interactivo (Recomendado)

```bash
python main.py
```

1. Selecciona **Opción D: Entrenar Prior para modelo RAVE**
2. Presiona Enter para auto-buscar checkpoint (o ingresa ruta manualmente)
3. Ingresa la ruta al dataset de audio
4. Elige un nombre para el prior (ej: `my_prior`)
5. Selecciona configuración (`decoder_only` recomendado)
6. Confirma y espera (puede tardar horas)

**Nota:** El código buscará automáticamente en `models/user_model/checkpoints/last.ckpt`

### Método 2: Línea de Comandos

```bash
python main.py train_prior \
  --rave models/user_model/exported_model/my_model.ts \
  --audio input_data/user_data \
  --name my_prior \
  --config decoder_only \
  --output models/user_model/prior
```

## Proceso de Entrenamiento

El proceso tiene 3 etapas:

### 1. Preprocesamiento (Encoding)
```
Audio Files → RAVE Encoder → Latent Representations
```
- Codifica todo el dataset a vectores latentes
- Puede tardar minutos-horas según tamaño del dataset
- Salida: carpeta con latents preprocesados

### 2. Entrenamiento del Prior
```
Latent Sequences → Autoregressive Model → Trained Prior
```
- Aprende patrones temporales en el espacio latente
- Usa Transformer (decoder_only) o GRU (recurrent)
- Monitorea pérdida en TensorBoard
- **Presiona Ctrl+C cuando estés satisfecho** (no necesitas entrenar hasta el final)

### 3. Exportación
```
Trained Prior → TorchScript (.ts) → Ready for Max/MSP
```
- Exporta a formato .ts para uso en Max/MSP
- Archivo listo para nn~ external

## Configuraciones Disponibles

| Configuración | Descripción | Uso Recomendado |
|--------------|-------------|-----------------|
| `decoder_only` | Transformer autoregressivo (mejor calidad) | Datasets grandes, GPU potente |
| `recurrent` | GRU (más ligero y rápido) | Datasets pequeños, CPU/GPU limitada |
| `encoder_decoder` | Seq2seq con entrada externa | Experimental (requiere rave2vec) |
| `encoder_decoder_continuous` | Seq2seq continuo | Experimental (requiere rave2vec) |

**Recomendación**: Empieza con `decoder_only` si tienes GPU, o `recurrent` si tienes limitaciones de hardware.

## Detener el Entrenamiento

**No necesitas entrenar hasta el final**. Puedes detener cuando:

1. La pérdida se estabiliza
2. Has alcanzado un punto que consideras suficiente
3. Quieres probar el prior

**Para detener:**
- Presiona `Ctrl+C` durante el entrenamiento
- El modelo se exportará automáticamente desde el último checkpoint

## Uso del Prior Entrenado

### Ubicación del Prior Exportado

Después del entrenamiento encontrarás:

```
models/user_model/prior/
├── preprocessed_latents/    # Latents codificados (puedes borrar después)
└── training/
    └── my_prior/
        ├── my_prior.ts      # ← ARCHIVO EXPORTADO
        ├── checkpoints/     # Checkpoints de entrenamiento
        └── logs/           # TensorBoard logs
```

### Uso en Max/MSP con nn~

El prior se usa junto con RAVE en Max/MSP:

```
1. RAVE solo (random latents):
   [nn~ rave_model.ts]
        |
    [decode]
        |
    [dac~]

2. RAVE + Prior (structured generation):
   [nn~ prior.ts @prior 1]
        |
    [generate latents]
        |
   [nn~ rave_model.ts]
        |
    [decode]
        |
    [dac~]
```

### ¿Por qué NO en Python Streaming?

Los priors exportados (.ts) tienen una API diseñada para Max/MSP, no para Python. Para usar el prior en Python necesitarías:
- Acceso a los checkpoints originales (.ckpt), no el .ts exportado
- Integración personalizada con la biblioteca MSPrior
- Gestión manual del estado autoregressivo

**Alternativa**: El streaming de Python usa random latent sampling que funciona excelente para exploración creativa.

## Monitoreo con TensorBoard

Durante el entrenamiento puedes monitorear el progreso:

```bash
tensorboard --logdir models/user_model/prior/training
```

Abre: `http://localhost:6006`

**Métricas importantes:**
- **Loss**: Debe bajar y estabilizarse
- **Perplexity**: Indica qué tan "seguro" está el modelo
- Menor loss = mejor aprendizaje de patrones

## Solución de Problemas

### Error: "MSPrior no está instalado"
```bash
pip install acids-msprior
```

### Error: "RecursiveScriptModule object has no attribute 'sr'"
MSPrior no puede leer archivos .ts exportados. Usa el checkpoint:
```bash
# ❌ Incorrecto
--rave models/user_model/exported_model/model.ts

# ✅ Correcto
--rave models/user_model/checkpoints/last.ckpt
```

### Error: "Model file not found"
Verifica que:
- El modelo RAVE existe y es un archivo .ts
- La ruta es absoluta o relativa correcta
- Exportaste el modelo después de entrenar RAVE

### Error: "Audio path not found"
- Verifica que la carpeta existe
- Debe contener archivos de audio (.wav, .mp3, etc.)
- **Debe ser el MISMO dataset usado para entrenar RAVE**

### Entrenamiento muy lento
- Usa `--config recurrent` (más rápido que decoder_only)
- Reduce el tamaño del dataset
- Verifica que estás usando GPU si está disponible

### Out of Memory (OOM)
- Usa configuración `recurrent` en lugar de `decoder_only`
- Reduce batch size (no ajustable desde CLI, edita configs de MSPrior)
- Usa chunks más pequeños de audio en preprocesamiento

## Comparación: Random vs Prior

| Aspecto | Random Latent Sampling | Prior Generation |
|---------|------------------------|------------------|
| **Setup** | Ninguno (funciona directo) | Requiere entrenamiento |
| **Tiempo** | Instantáneo | Horas de entrenamiento |
| **Creatividad** | Alta exploración | Más conservador |
| **Coherencia** | Buena con smoothing | Muy estructurado |
| **Similitud al dataset** | Media | Alta |
| **Uso** | Python streaming, Max/MSP | Solo Max/MSP (export .ts) |
| **Control real-time** | Total (temp, smoothing, gain) | Limitado (temperatura) |

## Ejemplo Completo

```bash
# 1. Entrenar RAVE (si aún no lo hiciste)
python main.py workflow \
  --audio input_data/user_data \
  --name vintage_synth \
  --config v2_small

# 2. Exportar RAVE (opcional, para streaming/Max)
python main.py export

# 3. Entrenar Prior (usa CHECKPOINT, no .ts)
python main.py train_prior \
  --rave models/user_model/checkpoints/last.ckpt \
  --audio input_data/user_data \
  --name vintage_prior \
  --config decoder_only

# 4. Encuentra el prior exportado en:
# models/user_model/prior/training/vintage_prior/vintage_prior.ts
```

## Recursos Adicionales

- **MSPrior GitHub**: https://github.com/caillonantoine/msprior
- **nn~ External**: https://github.com/acids-ircam/nn_tilde
- **RAVE Documentation**: https://github.com/acids-ircam/RAVE
- **Max/MSP Tutorials**: https://forum.ircam.fr/

## Consejos Avanzados

1. **Experimenta con temperaturas**: En Max/MSP puedes ajustar la temperatura del prior en tiempo real

2. **Combina múltiples priors**: Entrena priors con diferentes datasets para diferentes estilos

3. **Fine-tuning**: Puedes re-entrenar un prior sobre un dataset más específico

4. **Interpolación**: En Max/MSP puedes interpolar entre outputs de múltiples priors

5. **Checkpoint selection**: Si el entrenamiento sobreajusta, usa un checkpoint anterior de `models/user_model/prior/training/my_prior/checkpoints/`
