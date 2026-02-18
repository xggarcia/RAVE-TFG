# RAVE-TFG

Real-time Audio Variational autoEncoder wrapper for easy training, inference, and **live streaming** with interactive controls.

## Features

- 🎵 **Interactive Menu**: Easy-to-use interface for all operations
- 🖥️ **Dynamic Multi-Model Streaming**: Load unlimited models with add/delete slot management
- 🎛️ **Independent Parameters**: Each model has its own gain, temperature, and smoothing
- 🔀 **Real-Time Audio Mixing**: Blend multiple models together or toggle individually
- 🎨 **Dual Input Modes**: Random generation OR audio file transformation per slot
- 🔁 **Audio Looping**: Use audio files as input with optional looping
- 🎚️ **Random Intensity Control**: Adjust noise characteristics for generative mode
- 🎮 **Live Controls**: Modify parameters and activate/deactivate models during playback
- ➕ **Dynamic Slots**: Add or remove model slots on the fly (starts with 1 slot)
- 🚀 **Complete Pipeline**: Preprocess, train, export, and generate in one workflow
- 🧹 **Safe Data Management**: Clean user data with double confirmation
- 💻 **Cross-Platform**: Works on Windows, Linux, and macOS

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

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Test with demo model

```bash
# Launch interactive menu
python main.py

# Or test streaming directly
python main.py stream
```

---

## CLI Commands Reference

## Usage

### Interactive Menu (Recommended)

Simply run the script without arguments to launch the interactive menu:

```bash
python main.py
```

**Menu Options:**

```
============================================================
  RAVE-TFG - Menu Interactivo
============================================================

[*] Generacion de Audio:
  A) Generar audio desde archivo (UseModel)
  B) Entrenar modelo completo (Workflow)
  C) Streaming con GUI (Multi-Model Interface) ** UPDATED **
  D) Streaming con Teclado (Controles por teclas)
  E) Entrenar Prior para modelo RAVE

[+] Operaciones Avanzadas:
  1) Preprocesar dataset
  2) Entrenar modelo
  3) Exportar modelo

[#] Utilidades:
  4) Limpiar datos de usuario
  0) Salir
```

The menu guides you through each step with prompts and default values.

---

### Command Line Interface

For advanced users and automation, all features are available via CLI:

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
| `stream` | **NEW** Real-time audio streaming with live controls |
| `clean` | Delete all user data (with double confirmation) |

---

## 🎛️ Real-Time Streaming

### Two Streaming Modes

**Option C: Multi-Model GUI Streaming** - Recommended for creative sound design
- **Load up to 4 models simultaneously**
- **Independent parameters** for each model (gain, temperature, smoothing)
- **Activate/deactivate** models individually in real-time
- **Automatic audio mixing** when multiple models are active
- Visual sliders and color-coded slots
- Model discovery and selection dropdowns
- Real-time status monitoring

**Option D: Keyboard Streaming (Single Model)** - Best for focused experimentation
- Runs in terminal with keyboard shortcuts
- Faster startup, lower resource usage
- Keyboard controls for parameters
- Preferred for live performances

### 🖥️ Multi-Model GUI Streaming (Option C)

**Features:**
- 🎛️ **4 Independent Model Slots**: Load different models in each slot
- 🎨 **Color-Coded Interface**: Blue, Red, Orange, Purple slots for easy identification
- 📊 **Individual Parameter Sliders**: Gain, Temperature, Smoothing for each model
- 🎲 **Input Source Selection**: Random generation OR audio file per slot
- 🔁 **Audio File Support**: Load .wav/.mp3/.flac files with optional looping
- 🎚️ **Random Intensity**: Control noise characteristics in random mode
- ✓ **Activation Checkboxes**: Toggle models on/off during streaming
- 🔀 **Real-Time Mixing**: Blend multiple models or use solo
- 📁 **Model Discovery**: Auto-finds .ts files in your models folder
- 📝 **Status Log**: Monitor loading and streaming activity

**How to Use:**

```bash
python main.py
# Select Option C
# A window will open with 4 model slots
```

**Multi-Model Workflow:**
1. **Load Models**: For each slot:
   - Select model from dropdown OR click "Browse..."
   - Click "Load" button
   - Wait for "Loaded (Inactive)" status

2. **Configure Parameters**: Adjust sliders for each model:
   - **Gain** (0.0-1.0): Volume in the mix (lower for multiple models)
   - **Temperature** (0.1-3.0): Sound variation/chaos
   - **Smoothing** (0.0-0.95): Transition smoothness

3. **Choose Input Source**: For each slot:
   - **Random**: Generate from noise (adjust Intensity slider)
   - **Audio File**: Click "Select Audio", choose file, check "Loop" if desired

4. **Activate Models**: Check "ACTIVE" for models you want to hear

5. **Start Streaming**: Click "START STREAMING"

6. **Live Mixing**: While streaming:
   - Toggle models on/off with checkboxes
   - Adjust sliders to change parameters
   - Changes apply immediately

7. **Stop**: Click "STOP STREAMING" when done

**Example Multi-Model Setup:**
```
Slot 1: Drums model    (Random, Intensity: 1.2, Gain: 0.5, Active: ✓)
Slot 2: Melody model   (Audio: melody.wav, Loop: ON, Gain: 0.4, Active: ✓)
Slot 3: Ambient model  (Random, Intensity: 0.8, Gain: 0.3, Active: ✓)
Slot 4: (Empty)

→ Result: Generative drums + looping melody + ambient textures
```

**📖 Documentation:**
- [Multi-Model Guide](docs/MULTI_MODEL_GUIDE.md) - Advanced techniques and mixing tips
- [Input Source Guide](docs/INPUT_SOURCE_GUIDE.md) - Random vs Audio modes, intensity control
- [Quick Reference](docs/MULTI_MODEL_QUICKREF.md) - Visual layout and quick actions

### 🎮 Keyboard Streaming (Option D)

**Quick Start:**

**From Interactive Menu:**
```bash
python main.py
# Select option D (not C)
# Choose DEMO (1) or your own model (2)
# Press Enter to use defaults
```

**From Command Line:**
```bash
# Use demo model with default settings
python main.py stream

# Use custom model
python main.py stream --model models/user_model/exported_model/my_model.ts

# Disable interactive controls
python main.py stream --no-interactive
```

### 🎮 Live Parameter Controls

When streaming starts, you can modify the sound in real-time using keyboard controls:

#### Volume / Gain
- **Q**: Increase volume (+5%)
- **A**: Decrease volume (-5%)
- Range: 0.0 (silent) to 1.0 (max)

#### Temperature (Sound Variation)
- **W**: More variation (+0.1)
- **S**: Less variation (-0.1)
- Range: 0.1 to 3.0
  - **Low (0.1-0.5)**: Soft, predictable sounds
  - **Medium (0.8-1.2)**: Balanced variety
  - **High (1.5-3.0)**: Chaotic, experimental sounds

#### Smoothing (Interpolation)
- **E**: More smoothing (+5%)
- **D**: Less smoothing (-5%)
- Range: 0.0 to 0.95
  - **0.0**: No smoothing (can sound choppy)
  - **0.5**: 50% blend (smooth transitions)
  - **0.9**: Very gradual changes (fluid evolution)

#### Other Controls
- **R**: Reset to default values (Gain: 0.9, Temp: 1.0, Smooth: 0.0)
- **SPACE**: Show current parameters
- **X** or **ESC**: Exit streaming

### Stream Command Options

```bash
python main.py stream [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | demo_model.ts | Path to .ts model file |
| `--sr` | 44100 | Sample rate in Hz |
| `--latent-size` | Auto-detect | Latent vector size |
| `--chunk-duration` | 1.0 | Audio chunk duration in seconds |
| `--no-interactive` | False | Disable real-time controls |

### Creative Examples

**Soft Ambient Soundscape:**
```
1. Start streaming
2. Press 'S' multiple times (Temperature ~0.3)
3. Press 'E' multiple times (Smoothing ~0.8)
4. Adjust volume with Q/A
```

**Glitch/Experimental:**
```
1. Start streaming
2. Press 'W' multiple times (Temperature ~2.5)
3. Keep Smoothing low
4. Moderate volume
```

**Gradual Evolution:**
```
1. Start with defaults
2. Press 'E' until Smoothing ~0.9
3. Gradually increase Temperature with 'W'
4. Observe slow sound evolution
```

### Requirements for Streaming

**Model Format:** Streaming requires a `.ts` (TorchScript) file for optimal real-time performance.

- **Demo model**: Already included, ready to use
- **Your own model**: Export first using Option 3 or:
  ```bash
  python main.py export
  ```

If `.ts` file is missing, you'll see:
```
[!] Aviso: Para mejor rendimiento en tiempo real, usa la opcion 'Exportar' primero.
    No se encontro el archivo .ts optimizado.
```

---

### Command Line Interface

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

### Recommended Settings for Quick Testing

```bash
python main.py train --name test_model --config v2_small --val-every 50 --max-steps 1000
```

### Resume Training

Training automatically resumes from the latest checkpoint if one exists.

### Monitor Training

Training progress is logged to TensorBoard. View with:

```bash
tensorboard --logdir models/user_model/checkpoints
```

---

---

## Troubleshooting

### Streaming Issues

**No audio output:**
- Check system volume and audio device settings
- Verify the `.ts` model file exists
- Try increasing gain with 'Q' key

**Audio is choppy/glitchy:**
- Increase `--chunk-duration` to 1.5 or 2.0
- Reduce Temperature (press 'S')
- Increase Smoothing (press 'E')

**Controls not responding:**
- Make sure terminal window has focus
- On Windows, controls use `msvcrt` (native keyboard input)

**Unicode/Encoding errors on Windows:**
- The script automatically configures UTF-8 encoding
- If issues persist, run: `chcp 65001` before running Python

### Training Issues

**"Insufficient disk space" error on Windows**

RAVE's LMDB database pre-allocates space. Reduce `--max-db-size`:

```bash
python main.py preprocess input_data/my_audio --max-db-size 5
```

**"No checkpoint found" when exporting**

You need to train the model first and let it run until at least one checkpoint is saved (every `--val-every` steps).

**Clicking artifacts in Max/MSP**

Make sure to export with streaming mode (default). If you used `--no-streaming`, re-export:

```bash
python main.py export
```

---

## Training Priors with MSPrior

### What is a Prior?

A **prior** is an autoregressive model that learns temporal patterns in RAVE's latent space. It can generate more coherent and musically structured sequences than random sampling.

### When to Use Priors

- You want more structured, less random generation
- You're using the model in Max/MSP or PureData
- You want generation that closely follows training data patterns

### Training a Prior (Option D)

**Requirements:**
- A trained and exported RAVE model (.ts file)
- The **original audio dataset** used to train the RAVE model
- MSPrior installed (`pip install acids-msprior`)

**Interactive Menu:**

```bash
python main.py
# Select Option D: Entrenar Prior para modelo RAVE
```

**Command Line:**

```bash
python main.py train_prior \
  --rave models/user_model/exported_model/my_model.ts \
  --audio input_data/user_data \
  --name my_prior \
  --config decoder_only
```

**Steps:**
1. **Preprocess**: Encodes audio to latent representations using RAVE
2. **Train**: Trains autoregressive prior on latent sequences
3. **Export**: Exports prior as `.ts` file for use in Max/MSP

**Configurations:**
- `decoder_only` (recommended): Transformer-based autoregressive model
- `recurrent`: Lighter GRU-based model for limited compute

**Using the Trained Prior:**

The prior can be used in Max/MSP with nn~:

```
[nn~ rave_model.ts]  [nn~ prior.ts]
       |                    |
   [encode]          [generate latents]
       |                    |
       +----------+---------+
                  |
              [decode]
                  |
              [audio out]
```

**Note:** Priors in exported .ts files are designed for Max/MSP/PureData, not direct Python usage. For Python streaming, the random latent sampling (default in Option C) produces excellent quality.

---

## Performance Tips

### For Real-Time Streaming
- Use exported `.ts` files (not `.ckpt` checkpoints)
- Start with `chunk_duration=1.0` (default)
- If CPU is struggling, increase chunk duration to reduce overhead
- Lower `sr` to 22050 if needed for faster processing

### For Training
- Use `v2_small` config for faster training on limited GPU memory
- Start with `--max-steps 10000` to test before long training
- Monitor with TensorBoard: `tensorboard --logdir models/user_model/checkpoints`

---

## License

TBD

## Acknowledgments

- [ACIDS-IRCAM/RAVE](https://github.com/acids-ircam/RAVE) - Original RAVE implementation
- Real-time streaming and interactive controls implementation by RAVE-TFG team
