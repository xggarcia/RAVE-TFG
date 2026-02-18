# Multi-Model GUI Quick Reference

## Visual Layout

```
┌─────────────────────────────────────────────────────────────────┐
│         RAVE Multi-Model Streaming                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ⚙️ Global Audio Settings                                        │
│   Sample Rate: [44100 ▼]    Chunk Duration: [1.0 s]            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 🎛️ Model Slots (Activate multiple to mix audio)                │
│                                                                  │
│ ┌─ Model Slot 1 ────────────────────────────────────────────┐  │
│ │ [✓] ACTIVE   Model: [demo_model.ts ▼]  [Load] [Browse...]│  │
│ │ Status: Active                                            │  │
│ │ Gain: [█████░░░░░] 0.50  Temp: [███████░░] 1.00           │  │
│ │ Smooth: [░░░░░░░░░] 0.00                                  │  │
│ └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│ ┌─ Model Slot 2 ────────────────────────────────────────────┐  │
│ │ [ ] ACTIVE   Model: [user_model.ts ▼]  [Load] [Browse...] │  │
│ │ Status: Loaded (Inactive)                                 │  │
│ │ Gain: [████░░░░░░] 0.40  Temp: [████████░] 1.20           │  │
│ │ Smooth: [██████░░░] 0.60                                  │  │
│ └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│ ┌─ Model Slot 3 ────────────────────────────────────────────┐  │
│ │ [ ] ACTIVE   Model: [No Model Loaded ▼]  [Load] [Browse..]│  │
│ │ Status: Inactive                                          │  │
│ │ Gain: [████░░░░░░] 0.40  Temp: [███████░░] 1.00           │  │
│ │ Smooth: [░░░░░░░░░] 0.00                                  │  │
│ └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│ ┌─ Model Slot 4 ────────────────────────────────────────────┐  │
│ │ [ ] ACTIVE   Model: [No Model Loaded ▼]  [Load] [Browse..]│  │
│ │ Status: Inactive                                          │  │
│ │ Gain: [████░░░░░░] 0.40  Temp: [███████░░] 1.00           │  │
│ │ Smooth: [░░░░░░░░░] 0.00                                  │  │
│ └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 📊 Status Log                                                   │
│ Found 2 available models                                        │
│ Loading demo_model.ts into Slot 1...                            │
│ Slot 1: Loaded successfully (latent: 2x64)                     │
│ Slot 1 activated: demo_model.ts                                │
│ Loading user_model.ts into Slot 2...                            │
│ Slot 2: Loaded successfully (latent: 2x128)                    │
│ ==================================================              │
│ Starting streaming with 1 active model(s)                       │
│ Sample Rate: 44100 Hz                                           │
│ Chunk: 1.0s (44100 samples)                                     │
│   Slot 1: demo_model.ts                                         │
│     Gain=0.50, Temp=1.00, Smooth=0.00                          │
│ Audio stream initialized                                        │
│ STREAMING STARTED - Mixing active models in real-time          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│    [  ▶ START STREAMING  ]        [  ⏹ STOP STREAMING  ]      │
└─────────────────────────────────────────────────────────────────┘
```

## Color Coding

Each slot has a colored border for easy identification:
- **Slot 1**: Blue (#3498db)
- **Slot 2**: Red (#e74c3c)
- **Slot 3**: Orange (#f39c12)
- **Slot 4**: Purple (#9b59b6)

## Workflow States

### Slot States
1. **Inactive**: No model loaded
   - Status: "Inactive"
   - ACTIVE checkbox: disabled (grayed out)
   - Can select and load a model

2. **Loaded (Inactive)**: Model loaded but not active
   - Status: "Loaded (Inactive)"
   - ACTIVE checkbox: unchecked, enabled
   - Ready to activate

3. **Active**: Model loaded and generating audio
   - Status: "Active"
   - ACTIVE checkbox: checked
   - Contributing to the audio mix

### Streaming States
- **Stopped**: Before clicking START or after clicking STOP
  - Can load/unload models
  - Can change global settings
  - Cannot activate models without loading first

- **Streaming**: While audio is playing
  - Can activate/deactivate models (checkboxes)
  - Can adjust sliders for all models
  - Cannot change global audio settings
  - Cannot load new models (stop streaming first)

## Quick Actions

### Load a Model
1. Click dropdown in any slot → Select model
2. Click "Load" button
3. Wait for "Loaded (Inactive)" status

### Activate/Mix Multiple Models
1. Load models into 2+ slots
2. Check "ACTIVE" for each slot you want to hear
3. Adjust gain sliders (start with 0.3-0.5 each)
4. Click "START STREAMING"
5. Listen to the blend!

### Solo a Model
1. Load desired model in any slot
2. Ensure other slots are NOT checked as ACTIVE
3. Check ACTIVE for your solo slot
4. Set gain to 0.7-1.0 for full volume
5. Start streaming

### A/B Compare
1. Load Model A in Slot 1
2. Load Model B in Slot 2
3. Activate Slot 1, deactivate Slot 2 → Hear Model A
4. Deactivate Slot 1, activate Slot 2 → Hear Model B
5. Toggle between them by clicking checkboxes

### Dynamic Layering (While Streaming)
1. Start with Slot 1 active
2. Check Slot 2 ACTIVE → Adds layer
3. Adjust Slot 2 gain → Balance mix
4. Check Slot 3 ACTIVE → Adds another layer
5. Uncheck Slot 1 → Removes that layer

## Slider Ranges

| Parameter   | Min  | Max  | Default | Description                          |
|-------------|------|------|---------|--------------------------------------|
| Gain        | 0.0  | 1.0  | 0.5     | Volume level in mix                  |
| Temperature | 0.1  | 3.0  | 1.0     | Variation/randomness                 |
| Smoothing   | 0.0  | 0.95 | 0.0     | Temporal smoothing between chunks    |

**Multi-Model Gain Recommendations:**
- 1 model active: 0.7-1.0
- 2 models active: 0.4-0.7 each
- 3 models active: 0.3-0.5 each
- 4 models active: 0.25-0.4 each

## Button States

| Button          | Normal State        | During Streaming    |
|-----------------|---------------------|---------------------|
| START STREAMING | Enabled (green)     | Disabled (grayed)   |
| STOP STREAMING  | Disabled (grayed)   | Enabled (red)       |
| Load            | Enabled             | Disabled            |
| Browse...       | Enabled             | Disabled            |
| ACTIVE checkbox | Enabled if loaded   | Enabled (can toggle)|

## Tips

✅ **DO:**
- Load models before clicking START
- Use lower gain values when mixing multiple models
- Experiment with different parameter combinations
- Check the status log for error messages
- Save your favorite parameter combinations (write them down)

❌ **DON'T:**
- Try to load models while streaming (stop first)
- Set all gains to 1.0 when mixing (will cause clipping/compression)
- Activate a slot without loading a model first (checkbox will auto-uncheck)
- Change global audio settings during streaming (stop first)

## Keyboard Navigation

- **Tab**: Move between controls
- **Space**: Toggle focused checkbox/button
- **Arrow Keys**: Adjust focused slider
- **Enter**: Click focused button

## Troubleshooting

**"Please load a model in Slot X before activating"**
→ Select a model from dropdown and click "Load" first

**No sound when streaming**
→ Check that at least one slot has ACTIVE checked
→ Check that gain sliders are above 0
→ Check system audio volume

**Distorted/compressed sound**
→ Lower the gain on all active models
→ If 3 models active with gain=1.0 each, reduce to 0.3-0.4

**Can't click Load button while streaming**
→ Click STOP STREAMING first, then load new models

---

For more detailed information, see the full [Multi-Model Guide](MULTI_MODEL_GUIDE.md).
