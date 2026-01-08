# RAVE-TFG

## Project Structure

```
RAVE-TFG/
├── main.py              # Main CLI and functions
├── requirements.txt     # Python dependencies
├── configs/             # Configuration files
├── input_data/          # Audio datasets
│   ├── demo_data/       # Demo audio files
│   └── user_data/       # Place your own audio here
├── preprocessed_data/   # Preprocessed datasets (generated)
├── models/
│   ├── demo_model/      # Pre-trained demo models
│   └── user_model/      # User trained models
│       ├── checkpoints/ # Training checkpoints
│       └── exported_model/ # Exported models (.ts files)
└── outputs/             # Generated audio files
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Minimum GPU memory depends on config:
  - `v2_small`: 8 GB
  - `v2`: 16 GB
  - `discrete`: 18 GB
  - `v3`: 32 GB

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/xggarcia/RAVE-TFG
cd RAVE-TFG
```

### 2. Create and activate virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The project provides a full CLI with the following commands:

```bash
python main.py <command> [options]
```

#### Available Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Preprocess audio dataset |
| `train` | Train a RAVE model |
| `export` | Export trained model to TorchScript |
| `workflow` | Run complete pipeline (preprocess → train → export) |
| `generate` | Generate audio using a trained model |
| `clean` | Delete all user data (with double confirmation) |

---

### Preprocess Audio Dataset

Prepare your audio files for training.

```bash
python main.py preprocess <audio_path> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--channels` | 1 | Number of audio channels (1=mono, 2=stereo) |
| `--no-lazy` | False | Disable lazy loading (pre-processes all audio) |
| `--max-db-size` | 10 | Maximum database size in GB |

**Example:**
```bash
python main.py preprocess input_data/my_audio --channels 1
```

---

### Train a Model

Train a RAVE model on your preprocessed dataset.

```bash
python main.py train [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--name` | my_model | Name for your model |
| `--config` | v2_small | Architecture (see table below) |
| `--db-path` | preprocessed_data | Path to preprocessed dataset |
| `--channels` | 1 | Audio channels |
| `--val-every` | 1000 | Checkpoint every N steps |
| `--save-every` | 10000 | Save model every N steps |
| `--max-steps` | 6000000 | Maximum training steps |
| `--batch-size` | 8 | Batch size |

**Architecture Configurations:**

| Config | Description | Min GPU Memory |
|--------|-------------|----------------|
| `v1` | Original continuous model | 8 GB |
| `v2` | Improved continuous model (faster, higher quality) | 16 GB |
| `v2_small` | Smaller v2, good for timbre transfer | 8 GB |
| `v3` | v2 with style transfer capabilities | 32 GB |
| `discrete` | Similar to SoundStream/EnCodec | 18 GB |
| `onnx` | Noiseless v1 for ONNX export | 6 GB |
| `raspberry` | Lightweight for Raspberry Pi 4 | 5 GB |

**Example:**
```bash
python main.py train --name my_guitar_model --config v2_small --val-every 500
```

---

### Export Model

Export a trained model to TorchScript format for use in Max/MSP, PureData, etc.

```bash
python main.py export [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--run-path` | Auto-detect | Path to training run folder |



**Example:**
```bash
python main.py export
python main.py export --run-path models/user_model/checkpoints/my_model/version_0
```

---

### Complete Workflow

Run the entire pipeline in one command: preprocess → train → export.

```bash
python main.py workflow <audio_path> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--name` | my_model | Model name |
| `--config` | v2_small | Architecture config |
| `--channels` | 1 | Audio channels |
| `--val-every` | 1000 | Checkpoint frequency |
| `--max-steps` | 6000000 | Max training steps |

**Example:**
```bash
python main.py workflow input_data/my_audio --name my_model --config v2_small
```

---

### Generate Audio

Generate new audio using a trained RAVE model. When using random mode, generates 30 seconds of audio by default.

```bash
python main.py generate [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--model` | models/demo_model/demo_model.ts | Path to model file |
| `--audio` | input_data/demo_data/audio1.wav | Path to sample audio file (used to determine latent dimensions) |
| `--output` | generated | Output filename (without extension) |
| `--no-random` | False | Use input audio's latent instead of random |

**Example:**
```bash
# Generate 30s of random audio (uses demo model and demo audio by default)
python main.py generate

# Generate random audio with custom model
python main.py generate --model models/user_model/exported_model/my_model.ts --audio input.wav --output my_output

# Reconstruct input audio through the model (keeps original length)
python main.py generate --model models/user_model/exported_model/my_model.ts --audio input.wav --no-random
```

---

### Clean User Data

Delete all user-generated data (preprocessed datasets, checkpoints, exported models, and outputs). This command requires **double confirmation** for safety.

```bash
python main.py clean
```

**What gets deleted:**
- `preprocessed_data/` - Preprocessed datasets
- `models/user_model/checkpoints/` - Training checkpoints
- `models/user_model/exported_model/` - Exported .ts models
- `outputs/` - Generated audio files

**Example:**
```bash
python main.py clean
```

**Confirmation process:**
1. First prompt: Type `yes` to confirm
2. Second prompt: Type `DELETE ALL USER DATA` exactly

---


## Training Tips

### How Long Does Training Take?

Training time depends on:
- Dataset size
- GPU power
- Configuration used

Expect **several hours to days** for a fully trained model. You can stop training at any time and export the latest checkpoint.

### Recommended Settings for Quick Testing

```bash
python main.py train --name test_model --config v2_small --val-every 500 --max-steps 10000
```

### Resume Training

Training automatically resumes from the latest checkpoint if one exists.

### Monitor Training

Training progress is logged to TensorBoard. View with:

```bash
tensorboard --logdir models/user_model/checkpoints
```

---

## Troubleshooting

### "Insufficient disk space" error on Windows

RAVE's LMDB database pre-allocates space. Reduce `--max-db-size`:

```bash
python main.py preprocess input_data/my_audio --max-db-size 5
```

### "No checkpoint found" when exporting

You need to train the model first and let it run until at least one checkpoint is saved (every `--val-every` steps).

### Clicking artifacts in Max/MSP

Make sure to export with streaming mode (default). If you used `--no-streaming`, re-export:

```bash
python main.py export
```

```

---


## License

TBD

## Acknowledgments

- [ACIDS-IRCAM/RAVE](https://github.com/acids-ircam/RAVE) - Original RAVE implementation
