# Input Source Feature - Quick Start

## What's New?

Each model slot now has **two input modes**:

### 🎲 Random Mode (Generative)
```
Random Noise → Model Decoder → Unique Audio
```
- **Control:** Intensity slider (0.1-3.0)
- **Use:** Pure synthesis, unpredictable sounds
- **Best for:** Ambient, experimental, generative music

### 🎵 Audio File Mode (Transformative)
```
Your Audio → Model Encoder → Latents → Model Decoder → Transformed Audio
```
- **Control:** Select audio file, Loop checkbox
- **Use:** Transform existing audio through the model
- **Best for:** Style transfer, remixing, audio-to-audio processing

---

## Quick Examples

### Example 1: Pure Generation (Random Only)
```
┌─ Slot 1 ─────────────────────┐
│ ☑ ACTIVE                      │
│ Model: demo_model.ts          │
│ ⦿ Random  ○ Audio File        │
│ Intensity: ████████ 1.5       │
│ Gain: 0.8, Temp: 1.2          │
└───────────────────────────────┘

→ Continuous generative audio
```

### Example 2: Audio Transformation
```
┌─ Slot 1 ─────────────────────┐
│ ☑ ACTIVE                      │
│ Model: user_model.ts          │
│ ○ Random  ⦿ Audio File        │
│ [Select Audio] ☑ Loop         │
│ File: drums.wav               │
│ Gain: 0.9, Temp: 1.0          │
└───────────────────────────────┘

→ Drums continuously transformed through model
```

### Example 3: Hybrid Mix
```
┌─ Slot 1 ─────────────────────┐
│ ☑ ACTIVE                      │
│ ⦿ Random  ○ Audio File        │
│ Intensity: ████ 0.8           │
│ Gain: 0.4                     │
└───────────────────────────────┘

┌─ Slot 2 ─────────────────────┐
│ ☑ ACTIVE                      │
│ ○ Random  ⦿ Audio File        │
│ File: melody.wav  ☑ Loop      │
│ Gain: 0.5                     │
└───────────────────────────────┘

→ Random texture + looping melody
```

---

## Step-by-Step: Using Audio Files

1. **Load a model** in the slot
   ```
   Select model → Click "Load"
   ```

2. **Switch to Audio File mode**
   ```
   Click "Audio File" radio button
   ```

3. **Select your audio**
   ```
   Click "Select Audio" → Choose .wav/.mp3/.flac
   Wait for "Audio encoded" message
   ```

4. **Configure looping**
   ```
   ☑ Loop = Repeats indefinitely
   ☐ Loop = Plays once then stops
   ```

5. **Activate and stream**
   ```
   Check "ACTIVE" → Click "START STREAMING"
   ```

---

## Parameter Guide

| Parameter | Random Mode | Audio Mode |
|-----------|-------------|------------|
| **Intensity** | Controls noise strength | (Not used) |
| **Temperature** | Scales random values | Modifies encoded latents |
| **Smoothing** | Temporal blending | Temporal blending |
| **Gain** | Final volume | Final volume |

---

## When to Use Each Mode

### Use Random Mode When:
- ✅ Creating original generative music
- ✅ Exploring latent space randomly
- ✅ Need unpredictable variations
- ✅ Live improvisation
- ✅ No specific audio input available

### Use Audio File Mode When:
- ✅ Transforming existing recordings
- ✅ Style transfer (make X sound like Y)
- ✅ Audio remixing through AI
- ✅ Need rhythmic/structured output
- ✅ Processing specific audio material

---

## Common Workflows

### Workflow 1: Ambient Soundscape
```
Slot 1: Random, Intensity 0.6, Smooth 0.8, Gain 0.5
Slot 2: Random, Intensity 1.2, Smooth 0.3, Gain 0.4
Slot 3: Audio (pad.wav, Loop), Temp 0.8, Gain 0.3

= Layered ambient with audio foundation
```

### Workflow 2: Rhythm + Melody
```
Slot 1: Audio (drums.wav, Loop), Gain 0.6
Slot 2: Audio (bass.wav, Loop), Gain 0.5
Slot 3: Random, Intensity 1.0, Gain 0.3

= Structured rhythm with generative melody
```

### Workflow 3: Sound Design Exploration
```
Slot 1: Audio (source.wav, No Loop), Temp varies
→ Experiment with different Temperature values
→ Render one-shot transformations
```

---

## Troubleshooting

**Q: Audio doesn't sound different from original?**
- Increase Temperature (try 1.5-2.0)
- Model might be trained on similar content

**Q: Slot deactivates during playback?**
- Audio finished and Loop is OFF
- This is normal; re-enable or turn on Loop

**Q: Can't select audio file?**
- Must load model first
- Model encoder needed for audio processing

**Q: What's the difference between Intensity and Temperature?**
- **Intensity**: Only for Random mode, scales the noise input
- **Temperature**: For both modes, affects variation/chaos

**Q: Audio loops too fast?**
- File is very short
- Use longer audio or embrace the fast loop!

---

## Tips for Best Results

### Random Mode Tips
- **Low Intensity (0.3-0.7)** = Gentle, ambient
- **High Intensity (1.5-3.0)** = Chaotic, aggressive
- **Combine High Intensity + High Smoothing** = Controlled chaos

### Audio Mode Tips
- **Temperature < 1.0** = More faithful to original
- **Temperature > 1.5** = Heavy transformation
- **Use percussive sources** for rhythmic output
- **Use melodic sources** for tonal output
- **Try vocal samples** for unique textures

### Mixing Tips
- **Random + Audio** = Structure + Chaos
- **Multiple Audio files** = Layered transformations
- **Multiple Random** = Complex generative textures
- **Lower gains** when mixing many slots (0.3-0.5 each)

---

## Status Log Examples

### Random Mode Streaming:
```
Slot 1: Loaded successfully (latent: 2x64, output: 8192 samples)
Starting streaming with 1 active model(s)
  Slot 1: demo_model.ts [Random]
    Gain=0.80, Temp=1.20, Smooth=0.30
STREAMING STARTED - Mixing active models in real-time
```

### Audio Mode Streaming:
```
Slot 1: Loading audio file melody.wav...
Slot 1: Audio encoded (5.3s, 42 chunks)
Starting streaming with 1 active model(s)
  Slot 1: user_model.ts [Audio: melody.wav]
    Gain=0.70, Temp=1.00, Smooth=0.50
STREAMING STARTED - Mixing active models in real-time
[After file finishes with Loop OFF:]
Slot 1: Audio playback finished
```

---

**For detailed information, see [INPUT_SOURCE_GUIDE.md](INPUT_SOURCE_GUIDE.md)**
