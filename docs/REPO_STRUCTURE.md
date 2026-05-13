# Estructura del repositorio RAVE-TFG

Este documento describe cómo está organizado el repositorio tras la reorganización,
qué hace cada archivo y por qué se eliminó parte del código antiguo.

## 1. Visión general

El repo combina dos consumidores que comparten el mismo backend:

- **CLI** (`main.py` → `src/cli/`) — menú interactivo y subcomandos `argparse`.
- **App de escritorio PySide6** (`app/`) — UI que ejecuta el backend en hilos/procesos.

Ambos importan la misma lógica desde `src/`, así no hay duplicación.

```
RAVE-TFG/
├── main.py                ← Entry CLI
├── app/                   ← UI PySide6 (workers + pages + widgets)
├── src/
│   ├── cli/               ← Menú interactivo y argparse
│   ├── core/              ← Lógica de procesado (preprocess, train, export, …)
│   ├── streaming/         ← Motor de inferencia en tiempo real
│   └── database/          ← Ingesta Freesound + normalización/conversión audio
├── tools/                 ← Utilidades dev (check_file_lengths.py)
├── hooks/                 ← Hooks PyInstaller
├── install/, installer/   ← Scripts y bundler de instalación
├── docs/                  ← Documentación
├── tests/                 ← Tests (por crear)
├── design-mock/           ← Prototipo HTML (referencia UI)
├── input_data/, models/,  ← Carpetas de datos de usuario y modelos exportados
│   preprocessed_data/,
│   outputs/, database/
└── pyproject.toml, requirements.txt, README.md, installer.iss
```

## 2. Por qué esta organización

- **`src/` por dominio, no por tipo.** Los módulos viven en `core/`, `streaming/`,
  `database/`, `cli/` según *qué hacen*, no según si son "utilidades" o "modelos".
  Así es fácil saber dónde mirar para cualquier funcionalidad.
- **CLI y GUI comparten `src/`.** El requisito del proyecto (`CLAUDE.md`) es que la
  app no rescriba el core de RAVE: lo envuelve. `app/workers/` solo lanza funciones
  importadas de `src/core/` en `QThread`/`QProcess`.
- **`src/streaming/` separado de la UI.** El motor de tiempo real es independiente
  del framework de UI; lo consume `app/workers/stream_worker.py`, pero podría
  reutilizarse desde otro frontend sin tocar código.
- **`src/database/` aislado.** La parte de descarga Freesound es opcional y trae
  dependencias propias (API key, OAuth); mantenerlo separado permite importarlo
  sólo cuando se usa.

## 3. Archivos por carpeta

### `main.py`
Entry CLI. Sin argumentos lanza el menú interactivo; con argumentos parsea
subcomandos. Configura UTF-8 en consolas de Windows.

### `src/cli/`
| Archivo | Descripción |
|---|---|
| `interactive_menu.py` | Bucle del menú principal; construye `MenuContext` y delega en submenús. |
| `commands.py` | `argparse` para `preprocess`, `train`, `export`, `workflow`, `generate`, `clean`. |
| `menu_helpers.py` | `MenuContext`: helpers `ask_*`, selección de modelos/checkpoints. |
| `menu_actions_generate.py` | Submenú Generate (audio). El streaming se lanza vía `python -m app`. |
| `menu_actions_training.py` | Submenú Data & Training: workflow completo y paso a paso. |
| `menu_actions_dataset.py` | Submenú de creación de dataset (Freesound). |
| `menu_actions_maintenance.py` | Submenú de mantenimiento (clean). |

### `src/core/`
| Archivo | Descripción |
|---|---|
| `preprocess.py` | `PreprocessDataset`: invoca `rave preprocess` (LMDB, canales, lazy). |
| `train.py` | `TrainModel` + `find_latest_run`; detecta CUDA/CPU. |
| `export.py` | `ExportModel`: convierte checkpoint a TorchScript (`.ts`). |
| `generate.py` | `UseModel`: invoca `rave generate` con modelo y audio de referencia. |
| `workflow.py` | `train_workflow`: encadena preprocess → train → export. |
| `clean.py` | `CleanUserData`: borra preprocesados, checkpoints, exports, outputs. |

### `src/streaming/`
| Archivo | Descripción |
|---|---|
| `engine.py` | Re-export de `StreamingEngine` para imports cortos. |
| `engine_core.py` | `StreamingEngine`: gestión de slots, buffers, mezcla. |
| `engine_loops.py` | Bucle de inferencia (encode → modificar latentes → decode). |
| `models.py` | Dataclasses `ModelSlot` y estado por slot. |
| `calibration.py` | `QuickCalibrator`: RMS objetivo, auto-gain. |
| `phase_control.py` | Anclajes de fase + `generate_anchors_from_folders`, PCA. |

### `src/database/`
| Archivo | Descripción |
|---|---|
| `_freesound_api.py` | Cliente de bajo nivel de la API de Freesound. |
| `freesound_auth.py` | Carga credenciales OAuth2 y obtiene `access_token`. |
| `create_csv.py` | Selección interactiva de previews → CSV de IDs. |
| `download_csv.py` | Descarga de sonidos por ID desde CSV. |
| `first_download_freesound.py` | Descarga inicial de previews para criba. |
| `create_database.py` | Orquesta el flujo completo: query → previews → selección → descarga → normalize → convert. |
| `merge_selected_csv.py` | Combina varios CSV de IDs seleccionados. |
| `normalize_volume.py` | Normalización RMS de un directorio de audio. |
| `convert_format.py` | Resampleo + cambio de canales/subtype (WAV/FLAC/OGG). |

### `app/` (UI PySide6)
| Ruta | Descripción |
|---|---|
| `__main__.py` | Entry de la app: `QApplication`, fuentes, QSS, `MainWindow`. |
| `_paths.py` | Helpers de rutas. |
| `ui/main_window.py` | `QMainWindow` con sidebar + stack de páginas + status rail. |
| `ui/tokens.qss` | Tokens visuales (colores, radios, tipografía) portados del mock. |
| `ui/shell/sidebar.py` | Barra lateral con navegación. |
| `ui/shell/titlebar.py` | Barra de título personalizada (sin marco del SO). |
| `ui/shell/status_rail.py` | Barra inferior de estado/progreso. |
| `ui/widgets/form.py` | Helpers de formulario (`PageHeader`, `Panel`, `Field`, `FileInput`). |
| `ui/widgets/knob.py` | Knob rotatorio con `QPainter`. |
| `ui/widgets/vu.py` | VU meter. |
| `ui/widgets/waveform.py` | Visor de forma de onda. |
| `ui/widgets/spectrogram.py` | Espectrograma en tiempo real con `pyqtgraph`. |
| `ui/widgets/phase_pad.py` | Pad XY de fase. |
| `ui/widgets/latent_radar.py` | Visualización del espacio latente. |
| `ui/widgets/progress_panel.py` | Panel de progreso/logs. |
| `ui/pages/home.py` + `_home_widgets.py` | Dashboard. |
| `ui/pages/preprocess.py` | Formulario de preprocess. |
| `ui/pages/train.py` + `_train_*.py` | Página de entrenamiento (form, monitor, live, loss). |
| `ui/pages/export.py` | Exportar a TorchScript. |
| `ui/pages/generate.py` | Generar audio. |
| `ui/pages/stream.py` + `_stream_*.py` | GUI de streaming multi-slot, controles avanzados. |
| `ui/pages/dataset.py` + `_dataset_*.py` | Wizard de dataset (Freesound). |
| `ui/pages/workflow.py` + `_workflow_*.py` | Workflow encadenado. |
| `ui/pages/clean.py` | Limpiar datos de usuario. |
| `ui/pages/placeholder.py` | Stub para páginas sin implementar. |
| `workers/preprocess_worker.py` | `QThread` que ejecuta `PreprocessDataset`. |
| `workers/train_worker.py` | `QProcess` para entrenar (aislamiento CUDA). |
| `workers/export_worker.py` | `QThread` para `ExportModel`. |
| `workers/generate_worker.py` | `QThread` para `UseModel`. |
| `workers/clean_worker.py` | `QThread` para `CleanUserData`. |
| `workers/dataset_worker.py` | `QThread` genérico para tareas del wizard de dataset. |
| `workers/stream_worker.py` | Hilo de inferencia que consume `src/streaming`. |
| `workers/_stream_loader.py`, `_stream_models.py` | Helpers internos del stream worker. |

### `tools/`, `hooks/`, `install/`, `installer/`
| Ruta | Descripción |
|---|---|
| `tools/check_file_lengths.py` | Lint: avisa si un archivo Python excede 250/350 líneas. |
| `hooks/hook-torch.py` | Hook PyInstaller para recopilar submódulos de Torch (sin training/dist). |
| `install/install.sh`, `install/install.bat` | Scripts de instalación por SO. |
| `install/patch_rave.py` | Parches post-install de compatibilidad scipy. |
| `installer/`, `installer.iss` | Bundler Windows (Inno Setup + uv embebido). |

### Carpetas de datos
| Ruta | Descripción |
|---|---|
| `input_data/demo_data/` | Audio de demo incluido. |
| `input_data/user_data/` | Audio del usuario (subido o descargado). |
| `preprocessed_data/` | LMDB activo + `metadata.yaml`. **No versionar** (tamaño >> repo). |
| `models/demo_model/` | Modelo demo `.ts`. |
| `models/user_model/` | Checkpoints y exports del usuario. |
| `outputs/recordings/` | Audio generado o grabado por la GUI. |
| `database/database_creation/`, `database/database_download/` | CSV de queries e IDs seleccionados (demo y user). Se mantienen en la raíz porque los flujos del CLI/GUI los referencian con esas rutas. |

## 4. Código eliminado

| Eliminado | Motivo |
|---|---|
| `src/stream_gui.py` | Wrapper del GUI antiguo (customtkinter); reemplazado por `app/ui/pages/stream.py`. |
| `src/streaming_gui/` (carpeta entera) | GUI antigua en customtkinter (app, ui, slots, model_io, runtime, mixins, …). Sustituida por la app PySide6 en `app/`. |
| `src/stream.py` | Streaming CLI legacy basado en `sounddevice` + `threading`. La funcionalidad vive ahora en `src/streaming/` + `app/workers/stream_worker.py`. |
| `src/train_prior.py` | Wrapper de MSPrior. Estaba deshabilitado en el menú (`train_prior_fn=None`) y nunca se llamaba. |
| `src/phase_workflow.py` | Workflow de entrenamiento por fases y anclajes. Las entradas de menú estaban comentadas; no había uso real desde la app principal. |
| `app/ui/pages/anchors.py`, `phase.py`, `prior.py` | Páginas PySide6 que dependían de los módulos anteriores; quedaban como código muerto y se retiraron del sidebar/`main_window`. |
| `tools/fix_home_widgets.py` | Script de refactor de un solo uso; ya aplicado, sin valor histórico en runtime. |

Comandos CLI retirados (no tenían backend tras la limpieza): `stream`, `phase_train`,
`gen_anchors`. Para streaming la entrada única es `python -m app`.

## 5. Duplicados que existían (ya resueltos)

- **Dos implementaciones de streaming.** El repo tenía a la vez `src/streaming_gui/`
  (customtkinter) y `app/ui/pages/stream.py` + `app/workers/stream_worker.py`
  (PySide6) cubriendo el mismo caso de uso. Se eliminó la versión customtkinter
  y se conservó la PySide6, que es la apuntada por `pyproject.toml`/`CLAUDE.md`.
- **Dos formas de invocar el core.** Antes había imports planos
  (`from src.preprocess import …`) y rutas opcionales `database_creation/…`.
  Ahora todo el código importa desde `src.core.*` y `src.database.*`.

## 6. Cómo verificar

1. `python main.py` → menú interactivo arranca sin errores; no hay opciones de
   streaming CLI ni `Train prior` / `Phase-aware`.
2. `python main.py preprocess --help` (idem para `train`, `export`, `workflow`,
   `generate`, `clean`).
3. `python -m app` → la GUI PySide6 arranca; la sidebar solo muestra
   *Generate / Stream / Dataset / Train / Workflow / Clean*.
4. `python tools/check_file_lengths.py` → ningún módulo de `src/` o `app/`
   excede 350 líneas.
