# Input Source Control Guide

## Overview
Each model slot now supports **two input modes** for generating audio:

1. **Random Mode**: Generate from random noise (traditional neural synthesis)
2. **Audio File Mode**: Use real audio as input (latent space manipulation)

## Input Modes

### 🎲 Random Mode (Default)

**What it does:** Generates audio from random noise, creating unpredictable, generative sounds.

**Controls:**
- **Intensity** (0.1 - 3.0): Controls the "strength" or "amplitude" of the random noise
  - Low (0.1-0.5): Subtle, quiet variations
  - Medium (0.5-1.5): Standard generation
  - High (1.5-3.0): Intense, chaotic output

**Use cases:**
- Pure generative synthesis
- Exploratory sound design
- Creating unpredictable textures
- Live performance improvisation

**Example:**
```
Slot 1: Random mode, Intensity: 1.0, Temp: 1.2
→ Generates continuous random audio variations
```

### 🎵 Audio File Mode

**What it does:** Encodes a real audio file into the model's latent space, then decodes it back (with modifications).

**Controls:**
- **Select Audio** button: Choose a .wav, .mp3, .flac, or .ogg file
- **Loop** checkbox: Restart from beginning when audio finishes
  - ✓ Checked: Loops infinitely
  - ☐ Unchecked: Stops when audio ends (slot deactivates)

**Use cases:**
- Audio-to-audio transformation
- Style transfer (e.g., make drums sound like your trained model)
- Timbre modification
- Audio "remixing" through the model

**Example:**
```
Slot 1: Audio mode, Input: drums.wav, Loop: ON
→ Continuously plays drums through the model's transformation
```

## How to Use

### Setting Up Random Mode

1. Load a model in a slot
2. **Input mode**: Select "Random" (default)
3. **Adjust Intensity slider**: Set noise strength
   - Try 1.0 first, then experiment
4. **Activate** the slot
5. **Start streaming**

### Setting Up Audio File Mode

1. **Load a model** in a slot first (important!)
2. **Input mode**: Select "Audio File"
3. **Click "Select Audio"** button
4. Choose an audio file from the dialog
5. Wait for encoding (status log shows progress)
6. **Check "Loop"** if you want it to repeat
7. **Activate** the slot
8. **Start streaming**

**Note:** The audio must be encoded before streaming starts. This happens automatically when you select the file.

## Parameter Interactions

### Temperature (applies to both modes)
- **Random mode**: Scales noise after intensity
  - Intensity=1.0, Temp=2.0 → Very chaotic
  - Intensity=2.0, Temp=0.5 → Intense but controlled
  
- **Audio mode**: Modifies the encoded latents
  - Temp < 1.0: More faithful to original audio
  - Temp > 1.0: More variation/distortion

### Smoothing (applies to both modes)
- Blends current latent with previous
- Creates temporal continuity
- Works identically for random and audio modes

### Gain (applies to both modes)
- Final volume control after decoding
- Use lower values when mixing multiple slots

## Multi-Slot Combinations

### Random + Random
Mix multiple random generators for complex textures:
```
Slot 1: Random, Intensity: 0.8, Temp: 1.0, Smooth: 0.3
Slot 2: Random, Intensity: 1.5, Temp: 0.5, Smooth: 0.7
→ Blend of smooth and chaotic elements
```

### Audio + Audio
Layer multiple audio transformations:
```
Slot 1: Audio (drums.wav), Loop: ON, Gain: 0.5
Slot 2: Audio (melody.wav), Loop: ON, Gain: 0.4
→ Both files play through models simultaneously
```

### Random + Audio
Combine generative and transformative:
```
Slot 1: Random, Intensity: 1.0 (generative texture)
Slot 2: Audio (bass.wav), Loop: ON (rhythmic element)
→ Generative background with transformed rhythm
```

## Audio File Requirements

### Supported Formats
- ✅ WAV (.wav)
- ✅ MP3 (.mp3)
- ✅ FLAC (.flac)
- ✅ OGG (.ogg)

### Processing
- Audio is automatically resampled to match the current sample rate (22050/44100/48000 Hz)
- Converted to mono (single channel)
- Encoded using the loaded model's encoder

### File Length
- **Short files** (< 1 second): May loop very quickly if Loop is ON
- **Long files** (minutes): Encoded in full, may take a moment to process
- **Recommended**: 5-30 seconds for most use cases

## Tips & Tricks

### Random Mode Tips
1. **Start with Intensity = 1.0**, adjust from there
2. **Lower intensity** (0.3-0.7) for ambient/background textures
3. **Higher intensity** (1.5-3.0) for aggressive/chaotic sounds
4. **Combine with high smoothing** (0.7-0.9) to tame high intensity

### Audio Mode Tips
1. **Use Loop = ON** for continuous playback during experimentation
2. **Try different audio types**: percussive, melodic, ambient, vocals
3. **Temperature exploration**:
   - Temp = 0.5: Subtle transformation
   - Temp = 1.0: Balanced
   - Temp = 2.0: Heavy transformation
4. **Non-looping mode** useful for:
   - One-shot samples
   - Timed performances
   - Rendering specific audio segments

### Performance Optimization
- **Audio encoding** happens when you select the file (not during streaming)
- **Long audio files** may take a few seconds to encode
- **Multiple audio slots** use more memory (each stores encoded latents)
- **Random mode** is more CPU-efficient (no encoding needed)

## Workflow Examples

### Example 1: Generative Ambient Layer
```
Slot 1: Random mode
- Intensity: 0.6
- Temperature: 1.2
- Smoothing: 0.8
- Gain: 0.7

Result: Smooth, slowly evolving ambient texture
```

### Example 2: Audio Transformation
```
Slot 1: Audio mode (input: vocal_phrase.wav)
- Loop: OFF
- Temperature: 1.5
- Smoothing: 0.3
- Gain: 0.9

Result: One-shot vocal transformation with variation
```

### Example 3: Hybrid Mix
```
Slot 1: Random, Intensity: 1.0, Gain: 0.4 (generative pad)
Slot 2: Audio (drums.wav), Loop: ON, Gain: 0.5 (rhythm)
Slot 3: Audio (bass.wav), Loop: ON, Gain: 0.3 (bass line)

Result: Generative pad with looping rhythmic elements
```

### Example 4: A/B Comparison
```
Slot 1: Random, Intensity: 1.0, Temp: 1.0
Slot 2: Audio (input.wav), Loop: ON, Temp: 1.0

Toggle Active checkboxes to compare:
- Pure generation vs audio transformation
- Same model, different input sources
```

## Troubleshooting

### "Please load a model first before selecting audio input"
→ You must load a model before selecting audio files (model is needed for encoding)

### Audio encoding takes too long
→ Long files (> 1 minute) take time to encode. Use shorter clips or be patient

### Audio sounds wrong/distorted
→ Try adjusting Temperature (lower = more faithful to original)
→ Check that Loop is set correctly (unintended looping can sound odd)

### Slot deactivates during streaming
→ Audio file finished and Loop = OFF
→ This is normal behavior; re-activate or check the Loop box

### No difference between Random and Audio mode
→ Ensure you clicked "Select Audio" and saw the encoding confirmation
→ Check status log for "Audio encoded" message

### Audio loops too fast
→ Your audio file is very short
→ Use a longer file or adjust expectations

## Advanced: Understanding Latent Space

### Random Mode
```
Random Noise → Scale by Intensity → Scale by Temperature → Decoder → Audio
```

### Audio Mode
```
Audio File → Encoder → Latents → Scale by Temperature → Decoder → Audio
```

**Key difference:** 
- Random: Starts from random distribution
- Audio: Starts from encoded audio structure

Both pass through the same decoder, but start from different points in latent space.

## Status Log Messages

### Random Mode
```
"Slot 1: Loaded successfully (latent: 2x64, output: 8192 samples)"
"Starting streaming with 1 active model(s)"
"  Slot 1: demo_model.ts [Random]"
```

### Audio Mode
```
"Slot 1: Loading audio file drums.wav..."
"Slot 1: Audio encoded (3.2s, 25 chunks)"
"Starting streaming with 1 active model(s)"
"  Slot 1: demo_model.ts [Audio: drums.wav]"
"Slot 1: Audio playback finished"  // If Loop = OFF
```

---

**Experiment and have fun!** The combination of random and audio modes across multiple slots opens up countless creative possibilities.
