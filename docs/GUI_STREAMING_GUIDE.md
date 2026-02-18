# GUI Streaming - Quick Start Guide

## Opening the GUI

**Option 1 - From Menu:**
```bash
python main.py
# Select: C) Streaming con GUI (Interfaz Visual)
```

**Option 2 - Direct Launch:**
```bash
python -m src.stream_gui
```

## GUI Layout

```
┌─────────────────────────────────────────────┐
│    🎵 RAVE Audio Streaming                  │
├─────────────────────────────────────────────┤
│                                             │
│  MODEL SELECTION                            │
│  ┌─────────────────────────────┐ [Browse]  │
│  │ Select model from dropdown  │           │
│  └─────────────────────────────┘           │
│                                             │
│  AUDIO SETTINGS                             │
│  Sample Rate:    [44100 Hz ▼]              │
│  Chunk Duration: [1.0 seconds]              │
│                                             │
│  REAL-TIME PARAMETERS                       │
│  Gain:        ━━━━━━○━━━━━  0.90           │
│  Temperature: ━━━━━○━━━━━━  1.00           │
│  Smoothing:   ○━━━━━━━━━━━  0.00           │
│                                             │
│  STATUS                                     │
│  ┌─────────────────────────────────────┐   │
│  │ Ready. Select model and START...    │   │
│  │                                      │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  [  ▶ START STREAMING  ]  [  ⬛ STOP  ]    │
└─────────────────────────────────────────────┘
```

## Step-by-Step Usage

### 1. Select a Model

**Auto-Discovery:**
- GUI automatically finds models in:
  - `models/demo_model/*.ts`
  - `models/user_model/exported_model/*.ts`
- Select from dropdown

**Manual Selection:**
- Click "Browse..." button
- Navigate to your .ts model file
- Select and open

### 2. Configure Audio Settings

**Sample Rate:**
- 22050 Hz: Lower quality, faster processing
- 44100 Hz: CD quality (default)
- 48000 Hz: Professional quality

**Chunk Duration:**
- 0.1-0.5s: Lower latency, more CPU usage
- 1.0s: Balanced (default)
- 2.0-5.0s: Higher latency, less CPU usage

### 3. Start Streaming

1. Click **"▶ START STREAMING"** button
2. Wait for model to load (status will show progress)
3. Audio begins playing automatically

### 4. Adjust Parameters in Real-Time

**While streaming, move the sliders:**

**Gain Slider (Volume):**
- Far left (0.0): Silent
- Middle (0.5): Half volume
- Far right (1.0): Maximum volume
- **Tip:** Start at 0.9 and adjust to taste

**Temperature Slider (Variation):**
- Low (0.1-0.5): Gentle, predictable sounds
- Medium (0.8-1.2): Natural variation
- High (1.5-3.0): Wild, experimental sounds
- **Tip:** Try 1.0 first, then explore

**Smoothing Slider (Coherence):**
- 0.0: No interpolation (can sound choppy)
- 0.3-0.5: Smooth transitions
- 0.7-0.9: Very fluid evolution
- **Tip:** Increase if sound is too "jumpy"

### 5. Monitor Status

The status box shows:
- Model loading progress
- Detected latent dimensions
- Streaming activity
- Parameter changes
- Any errors

### 6. Stop Streaming

Click **"⬛ STOP"** button to end streaming

## Tips & Tricks

### Best Practices

1. **Start Conservative:**
   - Temperature: 1.0
   - Smoothing: 0.0
   - Gain: 0.9
   - Then experiment!

2. **For Ambient Sounds:**
   - Temperature: 0.5-0.8
   - Smoothing: 0.6-0.9
   - Creates slow, evolving textures

3. **For Rhythmic/Glitchy:**
   - Temperature: 1.5-2.5
   - Smoothing: 0.0-0.2
   - Creates more chaotic variations

4. **For Smooth Melodies:**
   - Temperature: 0.8-1.2
   - Smoothing: 0.4-0.7
   - Balanced variation with continuity

### Performance Optimization

**If audio is glitching:**
- Increase chunk duration to 2.0s
- Lower sample rate to 22050 Hz
- Close other applications

**If latency is too high:**
- Decrease chunk duration to 0.5s
- Use a more powerful CPU

**If CPU usage is too high:**
- Increase chunk duration
- Lower sample rate
- Use simpler model (if available)

## Keyboard Shortcuts

While GUI is focused:
- **Space**: Update status log
- **Esc**: (Same as clicking STOP)

## Advanced Features

### Model Switching

To use a different model:
1. Click "⬛ STOP"
2. Select new model from dropdown
3. Click "▶ START STREAMING"

### Recording Output

Use external audio recording software:
- **Windows**: Audacity, OBS Studio
- **macOS**: QuickTime, Audacity
- **Linux**: Audacity, JACK

Set input to "System Audio" or "Loopback"

### Multiple Instances

You can run multiple GUI windows:
```bash
# Terminal 1
python main.py  # Select C

# Terminal 2  
python main.py  # Select C
```

Each will stream independently (but watch CPU usage!)

## Troubleshooting

### "No models found"
**Solution:** 
- Click "Browse..." and manually select a .ts file
- Or train/export a model first (Option B)

### Model loads but no sound
**Check:**
- System volume is up
- Correct output device selected
- Gain slider is above 0.0
- Look for errors in status box

### GUI freezes
**Try:**
- Wait a few seconds (loading can take time)
- Check terminal for error messages
- Restart the application

### Audio sounds distorted
**Fix:**
- Lower temperature (<1.0)
- Increase gain if too quiet
- Check sample rate matches model

### "Error loading model"
**Solutions:**
- Verify .ts file exists and is valid
- Check file wasn't corrupted
- Try re-exporting the model

## Comparison: GUI vs Keyboard

| Feature | GUI (Option C) | Keyboard (Option D) |
|---------|----------------|---------------------|
| Ease of use | ⭐⭐⭐⭐⭐ Very easy | ⭐⭐⭐ Moderate |
| Visual feedback | ✅ Sliders & status | ❌ Terminal only |
| Parameter precision | ✅ Exact values | ⭐ Stepped changes |
| Model selection | ✅ Dropdown menu | ⭐ Command line only |
| Resource usage | ⭐ Slightly higher | ✅ Lower |
| Startup time | ⭐ Slightly slower | ✅ Instant |
| Best for | Learning, experimentation | Performances, automation |

## Next Steps

Once comfortable with GUI streaming:
- Try **Keyboard Streaming (Option D)** for performances
- **Train your own model (Option B)** for custom sounds
- **Train a Prior (Option E)** for more structured generation
- Explore different temperature/smoothing combinations
- Record your favorite generations!

## Questions?

Check the main [README.md](../README.md) for:
- Training models
- Exporting models
- Prior generation
- Troubleshooting
- Performance tips
