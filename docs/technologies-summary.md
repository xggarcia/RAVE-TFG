# Resumen de Tecnologías Utilizadas en RAVE-TFG

## 1. Machine Learning y Deep Learning

| Tecnología | Uso en el Proyecto |
|---|---|
| **PyTorch (torch)** | Framework principal de deep learning. Se utiliza para cargar los modelos RAVE exportados como TorchScript (`.ts`) mediante `torch.jit.load()`, ejecutar inferencia (`encode()`/`decode()`), manipulación de tensores, detección de GPU con `torch.cuda.is_available()`, y gestión de memoria CUDA. Es la columna vertebral de todo el pipeline de generación de audio. |
| **RAVE (acids-rave)** | Modelo de autoencoder variacional de audio en tiempo real. Se invoca como CLI externo (`subprocess.run`) para las fases de preprocesado (`rave preprocess`), entrenamiento (`rave train`) y exportación (`rave export --streaming`). Los modelos exportados se cargan posteriormente como TorchScript para inferencia. |
| **MSPrior (acids-msprior)** | Modelo autorregresivo de prior sobre el espacio latente de RAVE. Se utiliza para entrenar priors que generan trayectorias latentes más estructuradas. Se invoca como CLI externo para preprocesado, entrenamiento y exportación de priors. |
| **scikit-learn (PCA)** | Reducción de dimensionalidad mediante Análisis de Componentes Principales (`sklearn.decomposition.PCA`) para proyectar el espacio latente de alta dimensión a 2D, permitiendo su visualización en el scatter plot de la interfaz de control de fases. |
| **pytorch-lightning** | Framework de entrenamiento que utiliza RAVE internamente para gestionar el ciclo de entrenamiento (epochs, validation, checkpointing). Versión 1.9.x para compatibilidad con la API de acids-rave. |

## 2. Procesamiento de Audio

| Tecnología | Uso en el Proyecto |
|---|---|
| **librosa** | Carga de archivos de audio (`librosa.load`) y resampleo (`librosa.resample`). Se usa como fallback para decodificar formatos MP3/OGG que `soundfile` no puede leer directamente. También se utiliza para cálculos de RMS y normalización de audio. |
| **soundfile (libsndfile)** | Lectura y escritura de archivos de audio en formatos WAV, FLAC y OGG. Permite controlar el subtipo de salida (ej. `PCM_16`) y el dtype de los datos. Se utiliza tanto en el pipeline de preprocesado como en la grabación de audio durante el streaming. |
| **sounddevice** | Motor de audio en tiempo real. Gestiona streams de salida (`sd.OutputStream`) con tamaños de bloque tan bajos como 64-256 muestras para latencia mínima. Permite consultar dispositivos de audio disponibles (`sd.query_devices()`) y reproducir audio directamente. |
| **torchaudio.transforms.Resample** | Resampleo de audio GPU-acelerado. Se utiliza cuando la tasa de muestreo del archivo de audio no coincide con la del modelo RAVE cargado, convirtiendo automáticamente las señales antes de la inferencia. |
| **pyloudnorm** | Normalización de sonoridad según estándar EBU R128. Se utiliza en el pipeline de creación de datasets para normalizar el volumen de los clips de audio descargados de Freesound. |

## 3. Interfaz Gráfica de Usuario (GUI)

| Tecnología | Uso en el Proyecto |
|---|---|
| **PySide6 (Qt 6)** | Framework principal de la interfaz gráfica moderna. Proporciona todos los widgets de la aplicación de escritorio: `QMainWindow`, `QStackedWidget` para navegación entre páginas, `QVBoxLayout`/`QHBoxLayout` para layouts, señales/slots para comunicación entre componentes, `QThread` para workers en segundo plano, y `QPainter` para dibujar widgets personalizados (knobs, medidores VU, waveform, phase pad). |
| **pyqtgraph** | Biblioteca de visualización GPU-accelerada basada en Qt. Se utiliza para gráficos en tiempo real: curva de pérdida durante el entrenamiento (`PlotWidget` + `PlotCurveItem`), espectrograma rolling (`ImageItem` + `ColorMap`), y scatter plot de PCA para el mapa latente (`ScatterPlotItem`). Mucho más rápido que matplotlib para actualizaciones en tiempo real. |
| **customtkinter** | Framework de GUI alternativo (usado en la versión legacy `streaming_gui/`). Wrapper moderno de Tkinter con tema oscuro. Proporciona widgets como `CTkFrame`, `CTkButton`, `CTkSlider`, `CTkOptionMenu`, `CTkTextbox` para la interfaz de streaming multi-modelo anterior a la migración a PySide6. |
| **tkinter** | Framework de GUI estándar de Python. Se utiliza en la versión legacy de streaming GUI para variables de estado (`tk.IntVar`, `tk.DoubleVar`, `tk.StringVar`), diálogos de archivo (`filedialog`), cajas de mensaje (`messagebox`), y canvas para scroll. |

## 4. Cómputo Numérico y Científico

| Tecnología | Uso en el Proyecto |
|---|---|
| **NumPy** | Biblioteca fundamental de computación numérica. Se utiliza en todo el proyecto para: operaciones con arrays de audio, cálculo de RMS (`np.mean`, `np.square`, `np.sqrt`), cálculo de espectrograma (FFT con `np.fft.rfft()`, ventana Hanning con `np.hanning()`), conversión a decibelios (`np.log10()`), buffer circular (`np.roll()`), padding (`np.pad`), concatenación, y operaciones estadísticas. |
| **SciPy** | Biblioteca de algoritmos científicos. RAVE la utiliza internamente para diseño de filtros (Kaiser, FIR). El proyecto incluye un patch (`install/patch_rave.py`) para corregir incompatibilidades con scipy >= 1.12 (cambio de ubicación de `kaiser` y deprecación del argumento `nyq` en `firwin`). |
| **gin-config** | Sistema de configuración utilizado por RAVE internamente para gestionar hiperparámetros de entrenamiento mediante archivos de configuración `.gin`. |

## 5. Comunicación HTTP y APIs Externas

| Tecnología | Uso en el Proyecto |
|---|---|
| **requests** | Cliente HTTP para comunicarse con la API de Freesound v2. Se utiliza para: búsqueda de audio por texto (`/search/text/`), obtención de metadatos de sonidos (`/sounds/{id}/`), recuperación de descriptores de análisis (`/sounds/{id}/analysis/`), descarga de previsualizaciones MP3, y descarga de audio completo. |
| **Freesound API v2** | Servicio externo de Freesound.org que proporciona acceso a una base de datos de sonidos con metadatos ricos y descriptores acústicos (centroide espectral, MFCC, disonancia, etc.). Requiere clave API para búsqueda y OAuth2 para descargas completas. |
| **OAuth2** | Protocolo de autorización implementado para el flujo de código de autorización con Freesound. Incluye redirección del navegador para autorización, intercambio de código por token, refresco de tokens, y almacenamiento en caché de credenciales en `.freesound_tokens.json`. |

## 6. Concurrencia y Paralelismo

| Tecnología | Uso en el Proyecto |
|---|---|
| **QThread (PySide6)** | Workers en segundo plano para la GUI: entrenamiento (`QProcess`), preprocesado, exportación, generación, limpieza de datos, y dataset creation. Mantiene la interfaz responsive mientras las operaciones pesadas se ejecutan fuera del hilo principal. |
| **QProcess (PySide6)** | Ejecución del entrenamiento de RAVE como proceso separado. Aísla la memoria CUDA del proceso GUI, permitiendo matar el proceso limpiamente si falla sin crashear la interfaz. Parsea stdout línea por línea con regex para extraer progreso y métricas. |
| **threading (stdlib)** | Hilos daemon para control de teclado en CLI streaming, arquitectura productor/consumidor en el motor de streaming, y seguridad de hilos en la GUI. Se utiliza con `threading.Event` para señales de parada y `threading.Lock` para acceso seguro a estado compartido. |
| **queue (stdlib)** | Cola thread-safe (`queue.Queue`) para el pipeline productor-consumidor del motor de streaming en tiempo real. El productor genera chunks de audio y los pone en la cola; el consumidor los escribe en el dispositivo de salida. |
| **collections.deque** | Ventana deslizante para métricas de rendimiento en el motor de streaming (producer_ms, decode_ms, write_ms), permitiendo calcular promedios móviles de latencia. |
| **multiprocessing** | Utilizado por RAVE internamente para el preprocesado de datasets. El proyecto aplica un patch para limitar el número de workers a 4 (por defecto) evitando el agotamiento de memoria virtual en Windows. |

## 7. Almacenamiento y Persistencia de Datos

| Tecnología | Uso en el Proyecto |
|---|---|
| **LMDB** | Base de datos embebida clave-valor de alto rendimiento. RAVE la utiliza para almacenar el dataset preprocesado de forma eficiente, permitiendo lazy loading de clips de audio durante el entrenamiento sin cargar todo en memoria. |
| **CSV** | Formato de archivos para gestionar la creación de datasets: CSV de queries para búsqueda en Freesound, CSV de IDs seleccionados para descarga final, y CSV de metadatos de sonidos curados. |
| **JSON** | Serialización de datos de anclas de fase (phase anchors), respuestas de la API de Freesound, caché de tokens OAuth2, y configuración de slots del streaming GUI. |

## 8. Frameworks de Entrenamiento y Logging

| Tecnología | Uso en el Proyecto |
|---|---|
| **TensorBoard** | Sistema de visualización de métricas de entrenamiento. RAVE escribe logs de pérdida, epoch, y métricas de validación que se pueden visualizar con `tensorboard --logdir models/user_model/checkpoints`. |
| **tqdm** | Barras de progreso para operaciones largas como descargas de Freesound, preprocesado de datasets, y entrenamiento de modelos. |
| **Flask** | Framework web ligero. Utilizado internamente por acids-rave para servir TensorBoard o interfaces de monitorización durante el entrenamiento. |

## 9. Construcción, Empaquetado y Distribución

| Tecnología | Uso en el Proyecto |
|---|---|
| **setuptools** | Sistema de build de Python. Definido en `pyproject.toml` para empaquetar el proyecto como paquete instalable con `pip install .`. |
| **uv** | Gestor de paquetes y entornos virtuales ultrarrápido. Se utiliza en los scripts de instalación (`install.bat`/`install.sh`) y en el instalador Windows para crear el entorno virtual e instalar dependencias. |
| **PyInstaller** | Herramienta de empaquetado para crear ejecutables standalone. El hook personalizado (`hooks/hook-torch.py`) optimiza el bundle excluyendo submódulos de torch innecesarios para inferencia (training, distributed, profiling, etc.), reduciendo significativamente el tamaño del ejecutable. |
| **Inno Setup** | Sistema de instalación para Windows. El script `installer/setup.iss` define el instalador gráfico que instala la aplicación, crea el entorno virtual, descarga PyTorch con soporte CUDA, y crea accesos directos. |

## 10. Utilidades y Herramientas Auxiliares

| Tecnología | Uso en el Proyecto |
|---|---|
| **argparse** | Parser de argumentos de línea de comandos. Define todos los subcomandos (`preprocess`, `train`, `export`, `workflow`, `generate`, `stream`, `train_prior`, `clean`) con sus respectivos argumentos y opciones. |
| **pathlib** | Manejo moderno de rutas de archivo. Utilizado en todo el proyecto para construcción de rutas cross-platform, patrones glob, y resolución de paths relativos/absolutos. |
| **shutil** | Operaciones de sistema de archivos de alto nivel: eliminación de directorios (`rmtree`), movimiento y copia de archivos, utilizado en la operación de limpieza y gestión de datasets. |
| **re (regex)** | Expresiones regulares para parsear la salida del entrenamiento (step, loss, it/s, val_loss), extraer sample rate de nombres de archivo de modelo (ej. `_r48000_`), y validar rangos de duración. |
| **dataclasses** | Definición de estructuras de datos ligeras como `DownloadJob`, `SoundCandidate`, y `_SlotState` para mantener estado organizado con type hints. |
| **ctypes** | Llamadas a APIs nativas de Windows. Se utiliza para establecer el AppUserModelID de la aplicación (`SetCurrentProcessExplicitAppUserModelID`) para que el icono de la taskbar se muestre correctamente. |
| **msvcrt** | Módulo específico de Windows para captura de teclado en tiempo real durante el streaming CLI (`kbhit()`, `getch()`). |
| **webbrowser** | Apertura del navegador del usuario para el flujo de autorización OAuth2 de Freesound. |

## 11. Dependencias de Frameworks ML (transitivas)

| Tecnología | Uso en el Proyecto |
|---|---|
| **einops** | Manipulación de tensores con operaciones de reorganización legibles. Utilizado internamente por RAVE para reshapes de tensores en el modelo. |
| **nn-tilde** | Extensiones de módulos de PyTorch. Utilizado por RAVE para capas de convolución y normalización especializadas. |
| **cached-conv** | Convolución con caché para inferencia en tiempo real. Crítico para RAVE streaming, permite mantener estado entre chunks de audio sin recalcular convoluciones completas. |
| **absl-py** | Biblioteca de utilidades de Google. Utilizada internamente por RAVE para logging y configuración. |
| **GPUtil** | Detección de GPU y memoria VRAM. Utilizado por RAVE para auto-detectar capacidades de GPU. El proyecto incluye una detección alternativa con `torch.cuda.is_available()` para evitar problemas de GPUtil en Windows. |
| **udls** | Utilidades de deep learning. Dependencia transitiva de acids-rave. |
| **pathos** | Paralelismo avanzado con serialización mejorada. Utilizado por RAVE para operaciones paralelas que requieren serializar objetos complejos. |

---

**Nota:** Este resumen cubre los 107 archivos Python del proyecto distribuidos en `src/` (45 archivos), `app/` (59 archivos), `hooks/` (1 archivo), `install/` (1 archivo), y `tools/` (1 archivo), así como los scripts de instalación (`installer/`) y configuración (`pyproject.toml`, `requirements.txt`).
