# -*- coding: utf-8 -*-
"""
Generate audio using trained RAVE models
"""
import os
import torch
import librosa as li
import soundfile as sf
import numpy as np

torch.set_grad_enabled(False)


def UseModel(model_path="demo/model/demo_model.ts", audio_path='demo/audio/audio1.wav', sr=44100, random=True, output_name="Output_audio", duration=30):
    """
    Generate audio from random latent vectors using a RAVE model.
    
    Args:
        model_path: Path to .ts model file
        audio_path: Path to sample audio (used to determine latent shape)
        sr: Sample rate
        random: Generate from random latents (True) or reconstruct input (False)
        output_name: Output filename (without extension)
        duration: Duration in seconds (only for random mode)
    
    Returns:
        Path to generated audio file
    """
    model = torch.jit.load(model_path).eval()

    # Load a sample audio to get the latent shape
    x, sr = li.load(audio_path, sr=sr)
    x = torch.from_numpy(x).reshape(1, 1, -1)
    
    # Encode to get latent representation and its shape
    z = model.encode(x)
    
    # Generate audio from random numbers
    if random:
        # Calculate latent size for desired duration (30s default)
        latent_per_second = z.shape[-1] / (len(x.reshape(-1)) / sr)
        target_latent_length = int(latent_per_second * duration)
        z_shape = (z.shape[0], z.shape[1], target_latent_length)
        z_random = torch.from_numpy(np.random.randn(*z_shape).astype(np.float32))
    else:
        z_random = z
    x_hat = model.decode(z_random).numpy().reshape(-1)
    
    # Save the generated audio
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"{output_name}.wav")
    sf.write(output_path, x_hat, sr)
    print(f"Audio generated and saved to: {output_path}")
    return output_path
