# 🎵 Opción C: Generación en Tiempo Real (Streaming)

## 📋 Descripción

La nueva **Opción C** permite generar audio en tiempo real usando modelos RAVE. El sistema crea un flujo continuo de audio sintetizado a partir de vectores latentes aleatorios, perfecto para performances en vivo, instalaciones sonoras o experimentación creativa.

## 🚀 Instalación

Primero, instala las dependencias actualizadas:

```bash
pip install -r requirements.txt
```

**Nueva dependencia añadida:** `sounddevice>=0.4.6` para streaming de audio en tiempo real.

## 🎮 Formas de Uso

### 1. Menú Interactivo (Recomendado)

Simplemente ejecuta:

```bash
python main.py
```

Se abrirá un menú interactivo donde puedes seleccionar:

```
  C) Generación en Tiempo Real (Streaming) ⭐ NUEVO
```

El menú te guiará paso a paso:
1. Seleccionar modelo DEMO (1) o PROPIO (2)
2. Configurar sample rate (por defecto 44100 Hz)
3. Configurar duración de chunks (por defecto 0.5 segundos)

### 2. Línea de Comandos (CLI)

Para usuarios avanzados:

```bash
# Usar modelo DEMO
python main.py stream

# Usar modelo personalizado
python main.py stream --model models/user_model/exported_model/mi_modelo.ts

# Configuración avanzada
python main.py stream --model models/user_model/exported_model/mi_modelo.ts --sr 48000 --chunk-duration 0.3
```

#### Parámetros disponibles:

- `--model`: Ruta al archivo `.ts` del modelo (por defecto: modelo DEMO)
- `--sr`: Sample rate en Hz (por defecto: 44100)
- `--latent-size`: Tamaño del vector latente (por defecto: 128)
- `--chunk-duration`: Duración de cada chunk en segundos (por defecto: 0.5)

## ⚠️ Requisitos Importantes

### Modelo en Formato .ts (TorchScript)

Para un rendimiento óptimo en tiempo real **SIN CORTES**, debes usar un modelo exportado en formato `.ts`:

1. **Si usas el modelo DEMO:** Ya está listo, no necesitas hacer nada.

2. **Si usas tu PROPIO modelo:**
   - Primero debes exportarlo usando la opción 3 del menú o:
     ```bash
     python main.py export
     ```
   - Esto crea un archivo `.ts` optimizado en `models/user_model/exported_model/`

### ¿Qué pasa si NO existe el .ts?

Si intentas usar un modelo que no está exportado:

```
❌ Error: Model file not found: models/user_model/exported_model/mi_modelo.ts

⚠️  Aviso: Para mejor rendimiento en tiempo real, usa la opción 'Exportar' primero.
   No se encontró el archivo .ts optimizado.
```

**Solución:** Exporta tu modelo primero (Opción 3 del menú).

## 🎹 Cómo Funciona

1. **Carga del modelo:** El script carga el modelo `.ts` exportado usando `torch.jit.load`

2. **Generación continua:**
   - En cada iteración, genera un tensor de ruido aleatorio (latent walk)
   - El modelo RAVE decodifica ese ruido a audio
   - El audio se normaliza para evitar clipping
   - Se envía al stream de salida en chunks

3. **Streaming en vivo:**
   - Usa `sounddevice` para enviar audio al sistema
   - Sin latencia perceptible
   - Calidad profesional (44.1kHz por defecto)

## 🛑 Detener el Streaming

Presiona **Ctrl+C** en cualquier momento para detener el streaming de forma segura:

```
⏹️  Streaming detenido por el usuario

✅ Stream cerrado correctamente
```

El programa maneja la interrupción correctamente y vuelve al menú principal (si usas el modo interactivo).

## 🔧 Troubleshooting

### Error: "No audio output detected"

Asegúrate de que tu sistema tiene un dispositivo de salida de audio configurado.

### Audio entrecortado o con clics

1. Aumenta `--chunk-duration` a 0.7 o 1.0 segundos
2. Verifica que estás usando un archivo `.ts` exportado (no `.ckpt`)
3. Cierra otros programas que usen audio

### "Model file not found"

1. Si usas modelo DEMO: Verifica que existe `models/demo_model/demo_model.ts`
2. Si usas modelo PROPIO: Primero exporta tu modelo (Opción 3 o `python main.py export`)

## 🎨 Casos de Uso

- **Instalaciones sonoras:** Genera paisajes sonoros infinitos y únicos
- **Performance en vivo:** Control creativo en tiempo real
- **Experimentación:** Explora el espacio latente de tu modelo
- **Meditación/Ambient:** Genera texturas sonoras continuas

## 📝 Ejemplo Completo

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Abrir menú interactivo
python main.py

# 3. Seleccionar opción C
# 4. Elegir modelo DEMO (1)
# 5. Presionar Enter para usar configuración por defecto
# 6. ¡Escuchar audio generado en tiempo real!
# 7. Presionar Ctrl+C para detener
```

## 🆕 Comparación de Opciones

| Opción | Propósito | Salida |
|--------|-----------|--------|
| **A** | Generar archivo de audio | Archivo `.wav` guardado |
| **B** | Entrenar modelo completo | Modelo entrenado y exportado |
| **C** ⭐ | Streaming en tiempo real | Audio en vivo por altavoces |

---

**¡Disfruta creando audio en tiempo real con RAVE!** 🎵✨
