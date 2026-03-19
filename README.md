# RAVE-TFG

Real-time audio generation and model training toolkit built around RAVE, with an interactive menu, command-line workflows, and multi-model GUI streaming.

This README is the single source of user documentation for this repository.

## Quick Start (5 minutes)

1. Clone and enter the repository:

```bash
git clone https://github.com/xggarcia/RAVE-TFG
cd RAVE-TFG
```

2. Create and activate a virtual environment:

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

Windows:

```bash
install\install.bat
```

Linux/macOS:

```bash
chmod +x install/install.sh
./install/install.sh
```

4. Launch the interactive menu:

```bash
python main.py
```

5. First test:
- Go to Generate & Stream.
- Choose Multi-model GUI streaming.
- Use the included demo model first.

## Why install scripts are recommended

The install scripts handle version compatibility around acids-rave and scipy. They install core packages in a way that avoids common Python 3.12 dependency conflicts.

## Project Structure

```text
RAVE-TFG/
  main.py
  requirements.txt
  install/
    install.bat
    install.sh
    patch_rave.py
  input_data/
    demo_data/
    user_data/
  preprocessed_data/
  models/
    demo_model/
      demo_model.ts
    user_model/
      checkpoints/
      exported_model/
  outputs/
  src/
```

Where to put your files:
- Put your training audio in input_data/user_data.
- User checkpoints are created in models/user_model/checkpoints.
- Exported TorchScript models are created in models/user_model/exported_model.
- Generated WAV files are written to outputs.

## Main Usage Modes

## 1) Interactive Menu (recommended)

Run:

```bash
python main.py
```

Menu options:
- Main menu:
  - 1) Generate & Stream
  - 2) Data & Training
  - 3) Maintenance
  - 0) Exit
- Generate & Stream:
  - Generate audio from model
  - Multi-model GUI streaming
- Data & Training:
  - Full workflow (preprocess -> train -> export)
  - Preprocess dataset
  - Train model
  - Export model
  - Train prior (advanced)
- Maintenance:
  - Clean user data
  - Help / About

Navigation and setup behavior:
- Numeric choices only.
- Submenus include Back, Home, and Exit.
- Invalid input retries in place until valid.
- Most operations offer Quick mode (safe defaults) and Advanced mode (full parameter control).

This is the easiest path for external users because each option prompts for required inputs.

## 2) Command-line interface

Run commands as:

```bash
python main.py <command> [options]
```

Available commands:
- preprocess
- train
- export
- workflow
- generate
- stream
- train_prior
- clean

## Commands Reference

## preprocess

```bash
python main.py preprocess <audio_path> [--channels 1] [--no-lazy] [--max-db-size 10]
```

Arguments:
- audio_path (required): folder containing audio files
- --channels: 1 or 2 (default 1)
- --no-lazy: disable lazy loading
- --max-db-size: max LMDB size in GB (default 10)

Output:
- preprocessed_data/

## train

```bash
python main.py train [--name my_model] [--config v2_small] [--db-path preprocessed_data] [--channels 1] [--val-every 1000] [--save-every 10000] [--max-steps 6000000] [--batch-size 8]
```

Output:
- models/user_model/checkpoints/<name>/...

## export

```bash
python main.py export [--run-path models/user_model/checkpoints/<model>/version_<n>]
```

Behavior:
- If run-path is omitted, latest run is auto-detected.

Output:
- models/user_model/exported_model/<name>.ts

## workflow

```bash
python main.py workflow <audio_path> [--name my_model] [--config v2_small] [--channels 1] [--val-every 1000] [--max-steps 6000000]
```

Runs preprocess, train, and export in one sequence.

## generate

```bash
python main.py generate [--model models/demo_model/demo_model.ts] [--audio input_data/demo_data/audio1.wav] [--output generated] [--no-random]
```

Notes:
- Random mode generates new audio using sampled latent vectors.
- Non-random mode reconstructs from input audio latents.

Output:
- outputs/<output>.wav

## stream

```bash
python main.py stream
```

Launches the multi-model streaming GUI.

## train_prior

```bash
python main.py train_prior --rave <checkpoint.ckpt> --audio <audio_folder> [--name my_prior] [--config decoder_only] [--output models/user_model/prior]
```

Important:
- MSPrior requires checkpoint files (.ckpt). Exported .ts files are not the native training input for MSPrior.
- If you pass a .ts file, the script attempts to locate its original checkpoint.

Outputs:
- models/user_model/prior/preprocessed_latents/
- models/user_model/prior/training/<name>/
- exported prior .ts in the training run folder

## clean

```bash
python main.py clean
```

Deletes user-generated data after double confirmation:
- preprocessed_data/
- models/user_model/checkpoints/
- models/user_model/exported_model/
- outputs/

## Real-Time Streaming Guide

Streaming is provided through a GUI experience.

## Option C: Multi-model GUI streaming

How to open:
- Run python main.py
- Select Generate & Stream
- Select Multi-model GUI streaming

Core capabilities:
- Dynamic model slots (starts with 1, you can add/remove slots)
- One model per slot
- Per-slot controls:
  - Gain: 0.0 to 1.0
  - Temperature: 0.1 to 3.0
  - Smoothing: 0.0 to 0.95
- Per-slot input source:
  - Random mode with intensity control
  - Audio file mode with optional looping
- Global audio settings:
  - Sample rate: 22050, 44100, 48000
  - Chunk duration: 0.5 to 3.0 seconds
- Live model activation/deactivation while streaming
- Mixed output from all active slots

Input mode behavior:
- Random mode: uses sampled latent noise. Higher intensity and temperature increase variation.
- Audio mode: encodes selected audio and re-synthesizes via model latent space; loop keeps playback continuous.

Typical supported audio file formats for input mode:
- wav, mp3, flac, ogg

Quick workflow:
1. Add one or more slots.
2. Load model files (.ts) for each slot.
3. Configure source mode and parameters per slot.
4. Activate desired slots.
5. Start streaming.
6. Adjust controls live.

Mixing tip:
- As you activate more models, reduce per-slot gain to avoid clipping and keep headroom.

## Priors: What is supported and what is not

A prior is an autoregressive model over RAVE latent space that can produce more structured latent trajectories.

What is supported in this repository:
- Training priors using train_prior (MSPrior workflow)
- Exporting prior artifacts for downstream use

Important limitations to understand:
- Real-time streaming in this repo is focused on GUI-based model playback and mixing.
- Practical prior-centric generation is typically done with official RAVE/MSPrior tooling and environments such as Max/MSP nn~.

For external users: this means priors are useful and trainable here, but do not expect the streaming GUI to behave as full prior-driven generation.

## Model Configurations

Common architecture options for training:

| Config | Typical use | Approx minimum GPU memory |
|---|---|---|
| v1 | legacy continuous model | 8 GB |
| v2 | improved continuous quality | 16 GB |
| v2_small | lighter and practical default | 8 GB |
| v3 | style-transfer oriented variant | 32 GB |
| discrete | SoundStream/EnCodec-like discrete mode | 18 GB |
| onnx | noiseless v1 intended for ONNX export | 6 GB |
| raspberry | lightweight profile | 5 GB |

Recommendation:
- Start with v2_small for first experiments.
- Move to v2 after validating your full pipeline.

## Troubleshooting

No sound during streaming:
- Check OS output device and system volume.
- Confirm model path exists and points to a valid .ts file.
- Increase gain.

Audio sounds choppy:
- Increase chunk duration (for example 1.5 or 2.0).
- Lower temperature.
- Increase smoothing.
- Reduce number of active GUI slots.

Export fails with no checkpoint found:
- Train long enough to produce checkpoints first.
- Verify run-path points to a valid training run directory.

Preprocess fails due disk/memory pressure:
- Lower --max-db-size.
- Use a smaller dataset slice to validate pipeline first.

## Performance Tips

For streaming:
- Use exported .ts models.
- Keep chunk duration around 1.0 to 2.0 when CPU is limited.
- Use lower sample rate if needed for stability.

For training:
- Validate quickly with lower max-steps before long runs.
- Use TensorBoard:

```bash
tensorboard --logdir models/user_model/checkpoints
```

## External Integrations

The exported models are designed for interoperability with RAVE ecosystems and can be used in external environments such as Max/MSP or PureData depending on the workflow.

## FAQ

Q: Do I need to train my own model to test the project?
A: No. A demo model is included at models/demo_model/demo_model.ts.

Q: What streaming mode is supported?
A: Streaming is GUI-only, with multi-model mixing controls.

Q: Can I do everything from CLI without menu?
A: Yes. All main operations have CLI commands.

Q: Are priors mandatory for good output quality?
A: No. Random latent sampling with a well-trained model can already produce high-quality results.

## License

TBD

## Acknowledgments

- ACIDS-IRCAM/RAVE
- Contributors and maintainers of this repository
