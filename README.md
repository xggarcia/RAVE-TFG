# RAVE-TFG

Desktop application for training, exporting, and using [RAVE](https://github.com/acids-icml/RAVE) neural audio models.
Includes a PySide6 GUI and an optional CLI for scripting.

---

## Requirements

- Windows 10/11
- Python 3.10
- CUDA-capable GPU recommended for training (CUDA 12.1)
- ~3 GB disk space for the Python environment

---

## Installation

### Option A — Windows installer (recommended)

Download `RAVE-TFG-Setup.exe` from the [Releases](https://github.com/xggarcia/RAVE-TFG/releases) page and run it.

The installer will:
1. Create a Python 3.10 virtual environment under the install folder
2. Install RAVE core (`acids-rave`, `acids-msprior`) and all dependencies — **this downloads ~2 GB of PyTorch CUDA packages and takes 5–10 minutes**
3. Apply compatibility patches
4. Create Start Menu and optional desktop shortcut

To launch after install: use the **RAVE-TFG** shortcut or `Start Menu → RAVE-TFG`.

### Option B — Manual (from source)

```bash
git clone https://github.com/xggarcia/RAVE-TFG
cd RAVE-TFG
```

**Windows:**
```bat
install\install.bat
```

The install scripts handle the `acids-rave` / `scipy` version conflict automatically.

---

## Running the desktop app

```bash
python main.py
```

The app opens a sidebar with all workflows:

| Section | Pages |
|---|---|
| Generate & Stream | Generate Audio · Streaming GUI |
| Core Workflow | Dataset Creation · Preprocess · Train Model · Export Model |
| Training Extras | Phase Anchors |
| Maintenance | Clean User Data |

---

## CLI reference

Pass any command directly to skip the GUI:

```bash
python main.py <command> [options]
```

### `preprocess`

Chunk and encode an audio folder into an LMDB dataset.

```bash
python main.py preprocess <audio_path> [--channels 1] [--lazy] [--max-db-size 10]
```

Output: `preprocessed_data/`

### `train`

Train a RAVE model from a preprocessed dataset.

```bash
python main.py train [--name my_model] [--config v2_small] [--db-path preprocessed_data] \
  [--channels 1] [--val-every 1000] [--save-every 10000] [--max-steps 6000000] [--batch-size 8] \
  [--extra-config noise] [--gin-override KEY=VALUE]
```

Configs: `v1`, `v2`, `v2_small` *(default)*, `v3`, `discrete`, `onnx`, `raspberry`

Output: `models/user_model/checkpoints/<name>/`

### `export`

Convert a training checkpoint to a deployable TorchScript `.ts` file.

```bash
python main.py export [--run-path models/user_model/checkpoints/<name>/version_0]
```

If `--run-path` is omitted the latest run is auto-detected.
Output: `models/user_model/exported_model/<name>.ts`

### `workflow`

Run preprocess → train → export in one shot.

```bash
python main.py workflow <audio_path> [--name my_model] [--config v2_small] \
  [--channels 1] [--val-every 1000] [--max-steps 6000000]
```

### `generate`

Render audio from a trained model.

```bash
python main.py generate [--model demo/model/demo_model.ts] [--audio demo/audio/audio1.wav] \
  [--output generated] [--no-random]
```

- Default mode samples random latent vectors.
- `--no-random` re-encodes the reference audio instead.

Output: `outputs/<output>.wav`

### `database`

Download and curate audio from Freesound using a query CSV.

```bash
python main.py database <jobs.csv> [--selected-csv-dir database/database_download/user] \
  [--final-root input_data/user_data]
```

Set `FREESOUND_API_KEY` in a `.env` file or in the CSV's `API_Key` column.

### `clean`

Delete all user-generated data (preprocessed dataset, checkpoints, exports, outputs) after confirmation.

```bash
python main.py clean
```

---

## Demo assets

The repository includes ready-to-use assets in `demo/` so you can try every feature without needing your own files:

| Path | Use for |
|---|---|
| `demo/model/demo_model.ts` | Streaming GUI · Generate audio |
| `demo/model/demo_model_2.ts` | Streaming GUI — second slot, for the multi-model demo |
| `demo/audio/audio1,2,4,5.wav` | Generate audio · reference input |
| `demo/database/creation/*.csv` | Database → first download |
| `demo/database/download/rain_audio.csv` | Database → final download |

> **Model provenance:** `demo_model.ts` is this project's own model (RAVE v2, trained on rain audio).
> `demo_model_2.ts` is **not** part of this project's work — it is the pretrained *percussion* model
> released by ACIDS-IRCAM (`rave_percussion_rt`), bundled only to demonstrate real-time multi-model streaming.

---

## Data folders

```
demo/               ← included demo assets (model + audio samples)
input_data/
  user_data/        ← put your training audio here
preprocessed_data/  ← LMDB dataset (created by preprocess)
models/
  user_model/
    checkpoints/    ← training runs
    exported_model/ ← .ts files ready for streaming
outputs/            ← generated audio
database/
  database_download/user/   ← Freesound selected-IDs CSVs
```

---

## TensorBoard

```bash
tensorboard --logdir models/user_model/checkpoints
```

---

## License

MIT License — Copyright (c) 2026 Guillem Garcia

## Acknowledgments

[ACIDS-IRCAM / RAVE](https://github.com/acids-ircam/RAVE) — the RAVE architecture and training code.

The bundled `demo/model/demo_model_2.ts` is the pretrained *percussion* RAVE model (`rave_percussion_rt`)
released by ACIDS-IRCAM, used here under their terms purely for demonstration purposes.
