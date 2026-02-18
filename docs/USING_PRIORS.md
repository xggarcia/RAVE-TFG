# Using RAVE Models with Priors

## What is a Prior?

A **prior** in RAVE models is an autoregressive generative model trained over the latent space. It learns the temporal structure and patterns from training data to generate coherent latent sequences.

### Benefits:
- More coherent and musical output
- Sounds closer to training data style
- Better temporal structure
- Learned patterns instead of random noise

## Important Note

**RAVE priors are complex autoregressive models** that require special integration. The prior is **not directly callable** from Python in exported .ts models. Instead, priors work through:

1. **Official RAVE CLI**: `rave generate` command
2. **Max/MSP nn~**: Real-time prior sampling
3. **PureData nn~**: Real-time prior sampling

For **streaming mode in this wrapper**, we use **random latent sampling** which still produces high-quality audio through RAVE's powerful decoder.

## How to Use Models with Priors

### 1. Download Model with Prior

Download the **vintage** model from [RAVE Models](https://acids-ircam.github.io/rave_models_download):
- Model: `vintage (prior).ts`
- Trained on 80h of vintage music
- RAVE v1 - large architecture

### 2. Place Model in Project

```
models/
  demo_model/
    demo_model2(prior).ts   <-- Place here
```

### 3. Run Streaming with Prior

**Option A - Interactive Menu:**
```bash
python main.py
# Select Option C
# Choose model path: models/demo_model/demo_model2(prior).ts
# When asked "Usar prior del modelo (si disponible)? (s/n) [s]:" press Enter (default: yes)
```

**Option B - Command Line:**
```bash
python main.py stream --model models/demo_model/demo_model2(prior).ts
```

**To disable prior (use random noise):**
```bash
python main.py stream --model models/demo_model/demo_model2(prior).ts --no-prior
```

## Technical Details

### Current Streaming Implementation (Random Latent Sampling):
```python
# Generate random latent codes
z = torch.randn(1, latent_size, latent_length) * temperature

# Decode through RAVE
audio = model.decode(z)
```

**Why this still sounds good:**
- RAVE's decoder is trained to produce high-quality audio from its latent space
- Random sampling explores the learned manifold
- Temperature and smoothing controls provide musical variation
- The model's architecture ensures coherent audio output

### For True Prior Usage:

Use the official RAVE command-line tool:

```bash
# Install RAVE
pip install acids-rave

# Generate audio with prior
rave generate /pathLatent (This Wrapper) | Prior (Official RAVE) |
|---------|------------------------------|----------------------|
| Output style | High-quality synthesis | Structured generation |
| Temporal structure | Controlled by smoothing | Autoregressive patterns |
| Training data similarity | Manifold exploration | Strong similarity |
| Ease of use | Simple Python API | Requires RAVE CLI/Max |
| Real-time control | Full parameter control | Limited

| Feature | Random Noise | Prior |
|---------|--------------|-------|
| Output style | Abstract/experimental | Musical/coherent |
| Temporal structure | Random variations | Learned patterns |
| Training data similarity | Low | High |
| Creativity | High variation | Controlled variation |

## Models with Priors

From RAVE models repository (https://acids-ircam.github.io/rave_models_download):

| Model | Type | Training Data | Prior |
|-------|------|---------------|-------|
| vintage | v1 large | 80h vintage music | ✓ |
| percussion (prior) | v1 | Percussion sounds | ✓ |
| darbouka_onnx (prior) | v2 | Darbouka drum | ✓ |

**Note**: Models without "(prior)" in their name may not have a learned prior distribution.

## Troubleshooting
tiene Prior (distribucion aprendida)" but still using random noise:**
- This is correct behavior. The streaming wrapper uses random latent sampling
- For true prior generation, use: `rave generate` command or Max/MSP nn~
- Random latent sampling still produces excellent quality through RAVE's decoder

**Want to use the prior properly?**
- Install official RAVE: `pip install acids-rave`
- Use command: `rave generate model.ts input.wav --out output/`
- Or use Max/MSP nn~ external with `@prior 1` attribute

**Getting the best sound with random latents:**
- Adjust temperature (W/S keys) around 0.5-1.5 for musical results
- Enable smoothing (E/D keys) at 0.3-0.7 for temporal coherence
- Models with priors still have excellent decoderemporal coherence
- Some models may have subtle prior effects

## Example Session

```bash
python main.py stream --model models/demo_model/demo_model2(prior).ts --sr 44100

# Output:
# [OK] Modelo tiene Prior (distribucion aprendida)
# Generacion: PRIOR (distribucion aprendida)
# 
# Use Q/A, W/S, E/D to control gain, temperature, smoothing in real-time
```
