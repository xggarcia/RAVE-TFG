# -*- coding: utf-8 -*-
"""
Real-time audio streaming with RAVE models
"""
import os
import sys
import torch
import numpy as np
import sounddevice as sd
import threading
import time

torch.set_grad_enabled(False)


def StreamAudio(model_path="models/demo_model/demo_model.ts", sr=44100, latent_size=None, chunk_duration=1.0, interactive=True, use_prior=False):
    """
    Generate and stream audio in real-time using a RAVE model.
    The model continuously generates audio by sampling random latent vectors and decoding them.
    
    Args:
        model_path: Path to the .ts model file (TorchScript exported model)
        sr: Sample rate (default: 44100 Hz)
        latent_size: Size of the latent vector (default: auto-detect from model)
        chunk_duration: Duration of each audio chunk in seconds (default: 1.0s)
        interactive: Enable real-time parameter control (default: True)
        use_prior: (Ignored - kept for API compatibility. Priors require official RAVE CLI)
    
    Note:
        This function uses random latent sampling which produces high-quality audio
        through RAVE's powerful decoder. For true prior-based generation, use the
        official RAVE command: `rave generate model.ts input.wav --out output/`
    """
    print("=" * 50)
    print("RAVE Real-Time Audio Streaming")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"[X] Error: Model file not found: {model_path}")
        
        # Check if checkpoint exists as fallback
        checkpoint_dir = os.path.dirname(model_path).replace("exported_model", "checkpoints")
        if os.path.exists(checkpoint_dir):
            print("\n[!] Aviso: Para mejor rendimiento en tiempo real, usa la opcion 'Exportar' primero.")
            print("    No se encontro el archivo .ts optimizado.")
            print(f"    Busca checkpoints en: {checkpoint_dir}")
        return None
    
    try:
        # Load the model
        print(f"\nCargando modelo: {model_path}")
        model = torch.jit.load(model_path).eval()
        print("[OK] Modelo cargado exitosamente")
        
        # Get latent dimensions from model by encoding a dummy input
        print("\n[*] Detectando dimensiones del modelo...")
        dummy_length = sr * 2  # 2 seconds of audio
        dummy_audio = torch.randn(1, 1, dummy_length)
        
        with torch.no_grad():
            dummy_z = model.encode(dummy_audio)
        
        # Get the actual latent dimensions
        detected_latent_size = dummy_z.shape[1]
        compression_ratio = dummy_length / dummy_z.shape[2]
        
        if latent_size is None:
            latent_size = detected_latent_size
        
        print(f"    Latent dims detectados: {detected_latent_size}")
        print(f"    Compression ratio: ~{compression_ratio:.0f}x")
        
        # Check if model has a prior
        has_prior = hasattr(model, 'prior')
        if has_prior:
            print(f"[OK] Modelo tiene Prior (distribucion aprendida)")
            print(f"    Nota: El prior no es directamente accesible en Python")
            print(f"    Este modo usa muestreo aleatorio del espacio latente,")
            print(f"    que aun produce audio de alta calidad via el decoder.")
            if use_prior:
                print(f"    Para usar el prior completo: 'rave generate' (CLI oficial)")
        else:
            print(f"[!] Modelo SIN Prior (muestreo aleatorio del espacio latente)")
        
        # Streaming always uses random latent sampling
        # (Prior generation requires official RAVE CLI or Max/MSP integration)
        use_prior = False
        
        # Calculate chunk size
        chunk_samples = int(sr * chunk_duration)
        latent_length = max(1, int(chunk_samples / compression_ratio))
        
        # Real-time control parameters
        params = {
            'gain': 0.9,           # Volume/gain (0.0 to 1.0)
            'temperature': 1.0,    # Latent noise temperature (0.1 to 3.0)
            'smoothing': 0.0,      # Latent smoothing/interpolation (0.0 to 0.95)
            'running': True        # Control flag
        }
        
        # Previous latent for smoothing
        prev_z = None
        
        def control_thread():
            """Thread to handle keyboard input for real-time parameter control"""
            print("\n" + "=" * 60)
            print("  CONTROLES EN TIEMPO REAL")
            print("=" * 60)
            print("  [Q/A] Gain:        Subir/Bajar volumen")
            print("  [W/S] Temperature: Mas/Menos variacion")
            print("  [E/D] Smoothing:   Mas/Menos suavizado")
            print("  [R]   Reset:       Valores por defecto")
            print("  [SPACE] Info:      Mostrar parametros actuales")
            print("  [ESC/X] Salir:     Detener streaming")
            print("=" * 60 + "\n")
            
            while params['running']:
                try:
                    if sys.platform == 'win32':
                        import msvcrt
                        if msvcrt.kbhit():
                            key = msvcrt.getch().decode('utf-8').lower()
                            
                            if key == 'q':
                                params['gain'] = min(1.0, params['gain'] + 0.05)
                                print(f"\n[+] Gain: {params['gain']:.2f}")
                            elif key == 'a':
                                params['gain'] = max(0.0, params['gain'] - 0.05)
                                print(f"\n[-] Gain: {params['gain']:.2f}")
                            elif key == 'w':
                                params['temperature'] = min(3.0, params['temperature'] + 0.1)
                                print(f"\n[+] Temperature: {params['temperature']:.2f}")
                            elif key == 's':
                                params['temperature'] = max(0.1, params['temperature'] - 0.1)
                                print(f"\n[-] Temperature: {params['temperature']:.2f}")
                            elif key == 'e':
                                params['smoothing'] = min(0.95, params['smoothing'] + 0.05)
                                print(f"\n[+] Smoothing: {params['smoothing']:.2f}")
                            elif key == 'd':
                                params['smoothing'] = max(0.0, params['smoothing'] - 0.05)
                                print(f"\n[-] Smoothing: {params['smoothing']:.2f}")
                            elif key == 'r':
                                params['gain'] = 0.9
                                params['temperature'] = 1.0
                                params['smoothing'] = 0.0
                                print(f"\n[!] Reset: Gain={params['gain']:.2f}, Temp={params['temperature']:.2f}, Smooth={params['smoothing']:.2f}")
                            elif key == ' ':
                                print(f"\n[i] Gain={params['gain']:.2f} | Temp={params['temperature']:.2f} | Smooth={params['smoothing']:.2f}")
                            elif key in ['x', '\x1b']:  # x or ESC
                                print("\n[!] Deteniendo...")
                                params['running'] = False
                                break
                    time.sleep(0.05)
                except:
                    break
        
        print(f"\nConfiguracion:")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Chunk duration: {chunk_duration}s ({chunk_samples} samples)")
        print(f"  Latent shape: (1, {latent_size}, {latent_length})")
        print(f"  Generacion: Muestreo aleatorio del espacio latente")
        
        if interactive:
            print(f"  Modo: INTERACTIVO (controles habilitados)")
        else:
            print(f"  Modo: AUTOMATICO (sin controles)")
        
        print(f"\n[>] Iniciando streaming de audio...")
        
        if interactive:
            # Start control thread
            control = threading.Thread(target=control_thread, daemon=True)
            control.start()
        else:
            print("    Presiona Ctrl+C para detener\n")
        
        # Initialize audio stream
        stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype='float32'
        )
        
        stream.start()
        
        try:
            iteration = 0
            while params['running']:
                # Generate latent vector via random sampling
                # Note: RAVE's decoder is trained to produce high-quality audio
                # from random latent samples within its learned manifold
                with torch.no_grad():
                    # Sample random latent with temperature control
                    z_random = torch.randn(1, latent_size, latent_length).float() * params['temperature']
                    
                    # Apply smoothing/interpolation with previous latent
                    if params['smoothing'] > 0 and prev_z is not None:
                        z_random = params['smoothing'] * prev_z + (1 - params['smoothing']) * z_random
                    
                    prev_z = z_random.clone()
                    
                    # Decode to audio
                    audio_chunk = model.decode(z_random).numpy().reshape(-1)
                
                # Check if audio was generated
                if iteration == 0:
                    print(f"[DEBUG] Audio chunk generado: min={audio_chunk.min():.3f}, max={audio_chunk.max():.3f}, mean={np.abs(audio_chunk).mean():.3f}")
                
                # Normalize and apply gain
                max_val = np.abs(audio_chunk).max()
                if max_val > 0:
                    audio_chunk = (audio_chunk / max_val) * params['gain']
                
                # Ensure correct size
                if len(audio_chunk) > chunk_samples:
                    audio_chunk = audio_chunk[:chunk_samples]
                elif len(audio_chunk) < chunk_samples:
                    audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
                
                # Write to audio stream
                stream.write(audio_chunk.astype(np.float32))
                
                # Progress indicator
                iteration += 1
                if iteration % 5 == 0 and not interactive:
                    print(f"  [>>] Streaming... ({iteration * chunk_duration:.1f}s generados) | RMS: {np.sqrt(np.mean(audio_chunk**2)):.3f}", end='\r')
                
        except KeyboardInterrupt:
            print("\n\n[STOP] Streaming detenido por el usuario")
        
        finally:
            # Clean up
            stream.stop()
            stream.close()
            print("\n[OK] Stream cerrado correctamente")
            print("=" * 50)
    
    except Exception as e:
        print(f"\n[X] Error durante el streaming: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return True
