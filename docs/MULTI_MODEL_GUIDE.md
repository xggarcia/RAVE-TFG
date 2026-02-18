# Multi-Model RAVE Streaming Guide

## Overview
The updated RAVE GUI now supports **simultaneous loading and streaming of up to 4 models**, with independent parameter control and real-time audio mixing.

## Features

### Multi-Model Architecture
- **4 Independent Model Slots**: Each slot can hold a different RAVE model
- **Individual Parameters**: Every model has its own gain, temperature, and smoothing controls
- **Selective Activation**: Enable/disable models individually while streaming
- **Real-Time Mixing**: Audio from active models is combined automatically

### Visual Interface Components

#### Global Audio Settings
- **Sample Rate**: 22050 / 44100 / 48000 Hz (applies to all models)
- **Chunk Duration**: 0.5 to 3.0 seconds (buffer size)

#### Model Slots (Color-Coded)
Each slot features:
- **ACTIVE Checkbox**: Toggle model on/off during streaming
- **Model Dropdown**: Select from discovered models
- **Load Button**: Load the selected model into this slot
- **Browse Button**: Navigate to any .ts model file
- **Status Indicator**: Shows model state (Inactive / Loaded / Active)
- **Individual Sliders**:
  - **Gain** (0.0 - 1.0): Volume level for this model in the mix
  - **Temperature** (0.1 - 3.0): Variation/randomness in generated audio
  - **Smoothing** (0.0 - 0.95): Temporal continuity between chunks

## Usage

### Basic Workflow

#### 1. Launch GUI
```bash
python main.py
# Select Option C
```

#### 2. Load Models
For each slot you want to use:
1. Select a model from the dropdown OR click "Browse..." to find a .ts file
2. Click the **Load** button
3. Wait for "Loaded (Inactive)" status

#### 3. Configure Parameters
Adjust the sliders for each loaded model:
- **Lower gain values** (0.3-0.6) work well for mixing multiple models
- **Higher gain** (0.7-1.0) if using a single model
- Experiment with temperature and smoothing to taste

#### 4. Activate Models
- Check the **ACTIVE** box for each model you want to hear
- You can activate 1, 2, 3, or all 4 models simultaneously
- Status changes to "Active"

#### 5. Start Streaming
- Click **START STREAMING**
- Audio from all active models will mix in real-time
- View the status log for confirmation

### Advanced Techniques

#### Layering Sounds
Mix complementary models for complex textures:
- **Model 1**: Percussion-trained model (Gain: 0.6, Temp: 0.8)
- **Model 2**: Melodic model (Gain: 0.4, Temp: 1.2)
- **Model 3**: Ambient/texture model (Gain: 0.3, Temp: 1.5)

Result: Layered generative composition

#### Dynamic Mixing
While streaming is active:
1. **Activate/deactivate** models in real-time using checkboxes
2. **Adjust sliders** to change parameters on-the-fly
3. Mix changes take effect immediately (no restart needed)

#### Gain Staging
When mixing multiple models:
- Start with **all gains at 0.3-0.4**
- Listen for clipping/distortion in the mix
- Adjust individual gains to balance
- The system auto-normalizes if the mix exceeds 1.0, but manual balance sounds better

#### Parameter Exploration
- **Temperature**: Higher = more variation, lower = more stable
- **Smoothing**: Higher = smoother transitions, lower = more chaotic
- Try extreme values (Temp: 2.5+, Smooth: 0.9+) for experimental sounds

### Model Discovery
The GUI automatically scans:
- `models/demo_model/*.ts`
- `models/user_model/exported_model/*.ts`

Models found in these directories populate all dropdowns automatically.

### Solo Mode
To audition a single model:
1. Activate only that model's checkbox
2. Set its gain to 0.8-1.0
3. Other loaded models remain silent

### A/B Comparison
Load the same model in multiple slots with different parameters:
- **Slot 1**: Temp 0.5, Smooth 0.0 (chaotic)
- **Slot 2**: Temp 1.5, Smooth 0.8 (smooth, varied)

Toggle between them or mix both.

## Audio Mixing Details

### Mixing Algorithm
When multiple models are active:
1. Each model generates audio independently
2. Individual gain is applied to each output
3. Outputs are summed: `mixed = (audio1 * gain1) + (audio2 * gain2) + ...`
4. If peak exceeds 1.0, the mix is normalized: `mixed / max(abs(mixed))`
5. Result is sent to audio output

### Recommended Gain Settings

| # Active Models | Individual Gain Range |
|----------------|-----------------------|
| 1 model        | 0.7 - 1.0            |
| 2 models       | 0.4 - 0.7            |
| 3 models       | 0.3 - 0.5            |
| 4 models       | 0.25 - 0.4           |

These are starting points; adjust by ear.

## Troubleshooting

### "Model Not Loaded" Warning
- You clicked ACTIVE without loading a model
- **Solution**: Select a model and click "Load" first

### No Sound Output
- **Check**: Is at least one model activated?
- **Check**: Are gain sliders above 0?
- **Check**: System audio not muted?
- **Check**: Status log for error messages

### Distorted/Clipping Audio
- **Cause**: Sum of gains too high when mixing
- **Solution**: Reduce individual gain values
- Auto-normalization prevents true clipping but can sound compressed

### Model Loading Fails
- **Check**: File is a valid .ts TorchScript model
- **Check**: Model was exported correctly (see export documentation)
- **Error message** in status log provides details

### Streaming Stops Unexpectedly
- Check status log for error messages
- Restart the GUI and try again
- Ensure models are compatible (exported with same RAVE version)

## Performance Considerations

### CPU Usage
- More active models = higher CPU usage
- Each model runs inference independently
- **Optimization**: Reduce chunk duration (0.5s) for lower latency, but uses more CPU

### Latency
- Latency = chunk duration + audio device buffer
- **Low latency setup**: 0.5s chunk, 44100 Hz
- **Stable setup**: 1.0s chunk, 44100 Hz
- **Safe setup**: 1.5s chunk, 22050 Hz (for slower systems)

### Memory Usage
- Each loaded model consumes ~100-500 MB (depends on model architecture)
- 4 large models may require 2+ GB RAM
- Inactive models still occupy memory (loaded but not generating audio)

## Comparison: Multi-Model vs Single-Model

### Multi-Model Advantages
✅ Mix different sonic characteristics  
✅ Layer complementary textures  
✅ Real-time A/B comparison  
✅ Dynamic switching without stopping  
✅ Creative sound design possibilities  

### Single-Model Advantages (Option D)
✅ Lower CPU/memory usage  
✅ Keyboard shortcuts for parameters  
✅ Simpler interface  
✅ Better for focused experimentation  

## Tips for Creative Use

1. **Timbral Blending**: Load models trained on different instruments, mix at equal gains
2. **Controlled Chaos**: Mix a low-temp model (stable) with high-temp model (chaotic)
3. **Evolving Textures**: Activate models sequentially while streaming
4. **Parameter Automation** (manual): Slowly adjust sliders during streaming for evolving sounds
5. **Model Morphing**: Load similar models with varied temperatures to explore the latent space

## Keyboard Shortcuts
- **Space**: (When focused on window) Toggle streaming on/off
- **Tab**: Navigate between controls
- **Arrow Keys**: Adjust selected slider
- **Enter**: Click focused button

(Note: These are standard Tkinter shortcuts, not custom bindings)

## Example Session

```
1. Load drum-trained model into Slot 1
   - Set Gain: 0.5, Temp: 0.8, Smooth: 0.2
   
2. Load melody-trained model into Slot 2
   - Set Gain: 0.4, Temp: 1.2, Smooth: 0.6
   
3. Activate both slots

4. Start streaming → Hear rhythmic + melodic blend

5. While streaming:
   - Deactivate Slot 1 → Solo melody
   - Reactivate Slot 1 → Back to full mix
   - Increase Slot 2 Temp to 2.0 → Melody gets wilder
   - Adjust Slot 1 Gain to 0.7 → Drums more prominent

6. Load experimental model into Slot 3
   - Set Gain: 0.3, Temp: 2.5, Smooth: 0.0
   - Activate → Add chaotic layer

7. Mix and balance all three in real-time
```

---

**Enjoy exploring multi-model generative audio with RAVE!**
