# -*- coding: utf-8 -*-
import os
import sys
import shutil
from IPython.display import Audio, display
import librosa as li
import soundfile as sf
import numpy as np
import requests
import torch
import subprocess
import sounddevice as sd
import time

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

torch.set_grad_enabled(False)


def UseModel(model_path = "models/demo_model/demo_model.ts", audio_path='data/demo/audio1.wav', sr=44100, random = True, output_name = "Output_audio", duration=30):
    """
    Generate audio from random latent vectors using a RAVE model.
    
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



def StreamAudio(model_path="models/demo_model/demo_model.ts", sr=44100, latent_size=None, chunk_duration=1.0):
    """
    Generate and stream audio in real-time using a RAVE model.
    The model continuously generates audio from random latent vectors.
    
    Args:
        model_path: Path to the .ts model file (TorchScript exported model)
        sr: Sample rate (default: 44100 Hz)
        latent_size: Size of the latent vector (default: auto-detect from model)
        chunk_duration: Duration of each audio chunk in seconds (default: 1.0s)
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
        
        # Calculate chunk size
        chunk_samples = int(sr * chunk_duration)
        latent_length = max(1, int(chunk_samples / compression_ratio))
        
        print(f"\nConfiguracion:")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Chunk duration: {chunk_duration}s ({chunk_samples} samples)")
        print(f"  Latent shape: (1, {latent_size}, {latent_length})")
        print(f"\n[>] Iniciando streaming de audio...")
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
            while True:
                # Generate random latent vector (latent walk/random noise)
                z_random = torch.randn(1, latent_size, latent_length).float()
                
                # Decode to audio
                with torch.no_grad():
                    audio_chunk = model.decode(z_random).numpy().reshape(-1)
                
                # Check if audio was generated
                if iteration == 0:
                    print(f"[DEBUG] Audio chunk generado: min={audio_chunk.min():.3f}, max={audio_chunk.max():.3f}, mean={np.abs(audio_chunk).mean():.3f}")
                
                # Normalize and amplify
                max_val = np.abs(audio_chunk).max()
                if max_val > 0:
                    # Normalize to full range and apply gain
                    audio_chunk = (audio_chunk / max_val) * 0.9  # 90% of full scale
                
                # Ensure correct size
                if len(audio_chunk) > chunk_samples:
                    audio_chunk = audio_chunk[:chunk_samples]
                elif len(audio_chunk) < chunk_samples:
                    audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
                
                # Write to audio stream
                stream.write(audio_chunk.astype(np.float32))
                
                # Progress indicator
                iteration += 1
                if iteration % 5 == 0:
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


def PreprocessDataset(audio_path, channels=1, lazy=True, max_db_size=10):
    """
    Preprocess the dataset.
    
    """
    # Use absolute path to avoid RAVE bug with relative paths
    data_path = os.path.abspath("preprocessed_data")
    
    # Create directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    cmd = [
        "rave",
        "preprocess",
        "--input_path", audio_path,
        "--output_path", data_path,
        "--channels", str(channels),
        "--max_db_size", str(max_db_size)
    ]
    
    if lazy:
        cmd.append("--lazy")
    
    print(f"Preprocessing: {audio_path} -> {data_path}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Use input="y\n" to automatically confirm the lazy dataset prompt on Windows/macOS
    if(lazy):
        subprocess.run(cmd, check=True, input="y\n", text=True)

    print(f"Preprocessing completed!")
    return data_path


def TrainModel(
    name="my_model",
    config="v2_small",
    db_path="preprocessed_data",
    out_path="models/user_model/checkpoints",
    channels=1,
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8
):
    """
    Train a RAVE model.
    
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    
    # Build command
    cmd = [
        "rave",
        "train",
        "--config", config,
        "--db_path", db_path,
        "--out_path", out_path,
        "--name", name,
        "--channels", str(channels),
        "--val_every", str(val_every),
        "--save_every", str(save_every),
        "--max_steps", str(max_steps),
        "--batch", str(batch_size)
    ]
    
    print(f"Training model: {name}")
    print(f"Config: {config}")
    print(f"Dataset: {db_path}")
    print(f"Output: {out_path}")
    print(f"Checkpoint every: {val_every} steps")
    print(f"Save every: {save_every} steps")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, check=True)
    
    print(f"Training completed!")
    return out_path


def ExportModel(run_path=None, output_path="models/user_model/exported_model", streaming=True):
    """
    Export a trained RAVE model to a TorchScript file.

    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # If no run path provided, try to find the latest one
    if run_path is None:
        checkpoints_dir = os.path.abspath("models/user_model/checkpoints")
        if os.path.exists(checkpoints_dir):
            # Find the most recent run
            runs = []
            for model_dir in os.listdir(checkpoints_dir):
                model_path = os.path.join(checkpoints_dir, model_dir)
                if os.path.isdir(model_path):
                    for version_dir in os.listdir(model_path):
                        version_path = os.path.join(model_path, version_dir)
                        if os.path.isdir(version_path) and version_dir.startswith("version_"):
                            runs.append(version_path)
            
            if runs:
                # Sort by modification time, get the latest
                run_path = max(runs, key=os.path.getmtime)
                print(f"Auto-detected run: {run_path}")
            else:
                raise FileNotFoundError("No training runs found in models/user_model/checkpoints")
        else:
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Convert to absolute path (RAVE has issues with relative paths)
    run_path = os.path.abspath(run_path)
    
    # Check if checkpoint files exist in the run directory
    # PyTorch Lightning saves checkpoints in a 'checkpoints' subfolder
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    else:
        # Fallback: check directly in run_path (for older RAVE versions or manual setups)
        checkpoint_files = [f for f in os.listdir(run_path) if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files (.ckpt) found in {run_path} or {run_path}/checkpoints.\n"
            "You need to train the model first. Run TrainModel() and let it train "
            "for at least a few epochs until a checkpoint is saved."
        )
    
    # Build command
    cmd = [
        "rave",
        "export",
        "--run", run_path,
    ]
    
    # Add streaming flag for realtime processing (avoids clicking in Max/MSP)
    if streaming:
        cmd.append("--streaming")
    
    print(f"Exporting model from: {run_path}")
    print(f"Checkpoints found: {checkpoint_files}")
    print(f"Streaming mode: {streaming}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, check=True)
    
    # Find the exported .ts file and move it to the output directory
    # RAVE exports to the checkpoints directory by default
    export_search_dir = checkpoint_dir if os.path.exists(checkpoint_dir) else run_path
    exported_files = [f for f in os.listdir(export_search_dir) if f.endswith('.ts')]
    
    if exported_files:
        # Get the most recently created .ts file
        exported_files_full = [os.path.join(export_search_dir, f) for f in exported_files]
        latest_export = max(exported_files_full, key=os.path.getmtime)
        
        # Move to output directory
        dest_path = os.path.join(output_path, os.path.basename(latest_export))
        shutil.move(latest_export, dest_path)
        print(f"Model moved to: {dest_path}")
    
    print(f"Export completed!")
    return output_path


def train_workflow(
    audio_path,
    model_name="my_model",
    channels=1,
    lazy=True,
    max_db_size=10,
    config="v2_small",
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8,
    streaming=True
):
    """
    Complete training workflow: preprocess → train → export
    
    """
    print("=" * 50)
    print("RAVE Training Workflow")
    print("=" * 50)
    
    # Step 1: Preprocess
    print("\n[Step 1/3] Preprocessing dataset...")
    data_path = PreprocessDataset(audio_path, channels, lazy, max_db_size)
    
    # Step 2: Train
    print("\n[Step 2/3] Training model...")
    out_path = TrainModel(
        name=model_name,
        config=config,
        db_path=data_path,
        out_path="models/user_model/checkpoints",
        channels=channels,
        val_every=val_every,
        save_every=save_every,
        max_steps=max_steps,
        batch_size=batch_size
    )
    
    # Step 3: Export
    print("\n[Step 3/3] Exporting model...")
    exported_path = ExportModel(streaming=streaming)
    
    print("\n" + "=" * 50)
    print("Workflow completed!")
    print(f"Exported model saved to: {exported_path}")
    print("=" * 50)
    
    return exported_path


def CleanUserData():
    """
    Delete all user-related data: preprocessed data, checkpoints, exported models, and outputs.
    Requires double confirmation for safety.
    """
    
    # Directories to clean
    dirs_to_clean = [
        ("preprocessed_data", "Preprocessed dataset"),
        ("models/user_model/checkpoints", "Training checkpoints"),
        ("models/user_model/exported_model", "Exported models"),
        ("outputs", "Generated outputs"),
    ]
    
    # Check what exists (exclude .gitkeep files from count)
    existing_dirs = []
    for dir_path, description in dirs_to_clean:
        if os.path.exists(dir_path):
            contents = [f for f in os.listdir(dir_path) if f != '.gitkeep']
            if contents:
                existing_dirs.append((dir_path, description))
    
    if not existing_dirs:
        print("Nothing to clean. All user data directories are already empty.")
        return False
    
    # Show what will be deleted
    print("=" * 50)
    print("⚠️  WARNING: USER DATA CLEANUP ⚠️")
    print("=" * 50)
    print("\nThe following directories will be PERMANENTLY DELETED:\n")
    
    for dir_path, description in existing_dirs:
        # Count files excluding .gitkeep
        file_count = sum(1 for _, _, files in os.walk(dir_path) for f in files if f != '.gitkeep')
        print(f"  📁 {dir_path}")
        print(f"     └── {description} ({file_count} files)")
    
    print("\n" + "=" * 50)
    
    # First confirmation
    print("\n🔴 FIRST CONFIRMATION:")
    confirm1 = input("Are you sure you want to delete all user data? (yes/no): ").strip().lower()
    
    if confirm1 != "yes":
        print("\n❌ Cleanup cancelled.")
        return False
    
    # Second confirmation
    print("\n🔴 SECOND CONFIRMATION:")
    print("Type 'DELETE ALL USER DATA' to confirm:")
    confirm2 = input("> ").strip()
    
    if confirm2 != "DELETE ALL USER DATA":
        print("\n❌ Cleanup cancelled. Confirmation text did not match.")
        return False
    
    # Perform cleanup
    print("\n  Deleting user data...")
    
    for dir_path, description in existing_dirs:
        try:
            # Remove all contents but keep the directory and .gitkeep files
            for item in os.listdir(dir_path):
                if item == '.gitkeep':
                    continue  # Preserve .gitkeep files
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"  ✅ Cleaned: {dir_path}")
        except Exception as e:
            print(f"  ❌ Error cleaning {dir_path}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ User data cleanup completed!")
    print("=" * 50)
    
    return True


def interactive_menu():
    """
    Interactive menu for easy RAVE workflow navigation.
    """
    while True:
        print("\n" + "=" * 60)
        print("  RAVE-TFG - Menu Interactivo")
        print("=" * 60)
        print("\n[*] Generacion de Audio:")
        print("  A) Generar audio desde archivo (UseModel)")
        print("  B) Entrenar modelo completo (Workflow)")
        print("  C) Generacion en Tiempo Real (Streaming) ** NUEVO **")
        print("\n[+] Operaciones Avanzadas:")
        print("  1) Preprocesar dataset")
        print("  2) Entrenar modelo")
        print("  3) Exportar modelo")
        print("\n[#] Utilidades:")
        print("  4) Limpiar datos de usuario")
        print("  0) Salir")
        print("\n" + "=" * 60)
        
        choice = input("\nSelecciona una opcion: ").strip().upper()
        
        if choice == "A":
            print("\n--- Opcion A: Generar Audio desde Archivo ---")
            model_choice = input("Usar modelo DEMO (1) o PROPIO (2)? [1]: ").strip() or "1"
            
            if model_choice == "1":
                model_path = "models/demo_model/demo_model.ts"
                audio_path = "input_data/demo_data/audio1.wav"
            else:
                model_path = input("Ruta al modelo .ts [models/user_model/exported_model/*.ts]: ").strip()
                if not model_path:
                    # Try to find exported model
                    export_dir = "models/user_model/exported_model"
                    if os.path.exists(export_dir):
                        ts_files = [f for f in os.listdir(export_dir) if f.endswith('.ts')]
                        if ts_files:
                            model_path = os.path.join(export_dir, ts_files[0])
                            print(f"Usando: {model_path}")
                        else:
                            print("[X] No se encontraron modelos .ts exportados")
                            continue
                    else:
                        print("[X] Directorio de modelos exportados no existe")
                        continue
                
                audio_path = input("Ruta al audio de muestra: ").strip()
            
            output_name = input("Nombre de salida [generated]: ").strip() or "generated"
            duration = input("Duraci\u00f3n en segundos [30]: ").strip()
            duration = int(duration) if duration else 30
            
            UseModel(model_path=model_path, audio_path=audio_path, output_name=output_name, duration=duration)
            input("\nPresiona Enter para continuar...")
        
        elif choice == "B":
            print("\n--- Opcion B: Workflow Completo de Entrenamiento ---")
            audio_path = input("Ruta a la carpeta con audios: ").strip()
            if not audio_path or not os.path.exists(audio_path):
                print("[X] Ruta invalida")
                continue
            
            model_name = input("Nombre del modelo [my_model]: ").strip() or "my_model"
            config = input("Configuracion [v2_small/v2/v3]: ").strip() or "v2_small"
            max_steps = input("Maximo de pasos [6000000]: ").strip()
            max_steps = int(max_steps) if max_steps else 6000000
            
            train_workflow(
                audio_path=audio_path,
                model_name=model_name,
                config=config,
                max_steps=max_steps
            )
            input("\nPresiona Enter para continuar...")
        
        elif choice == "C":
            print("\n--- Opcion C: Generacion en Tiempo Real (Streaming) ---")
            model_choice = input("Usar modelo DEMO (1) o PROPIO (2)? [1]: ").strip() or "1"
            
            if model_choice == "1":
                model_path = "models/demo_model/demo_model.ts"
            else:
                # Check for exported .ts file
                export_dir = "models/user_model/exported_model"
                if os.path.exists(export_dir):
                    ts_files = [f for f in os.listdir(export_dir) if f.endswith('.ts')]
                    if ts_files:
                        model_path = os.path.join(export_dir, ts_files[0])
                        print(f"[OK] Modelo .ts encontrado: {model_path}")
                    else:
                        print("\n[!] No se encontro archivo .ts exportado.")
                        print("    Para mejor rendimiento en tiempo real, exporta tu modelo primero.")
                        
                        # Check for checkpoint as fallback
                        checkpoint_dir = "models/user_model/checkpoints"
                        if os.path.exists(checkpoint_dir):
                            print(f"    Puedes exportar tu modelo usando la opcion 3 del menu.")
                        continue
                else:
                    print("[X] No existe el directorio de modelos exportados")
                    continue
            
            print(f"\nUsando modelo: {model_path}")
            sr = input("Sample rate [44100]: ").strip()
            sr = int(sr) if sr else 44100
            
            chunk_duration = input("Duracion de chunk en segundos [1.0]: ").strip()
            chunk_duration = float(chunk_duration) if chunk_duration else 1.0
            
            StreamAudio(model_path=model_path, sr=sr, chunk_duration=chunk_duration)
            input("\\nPresiona Enter para continuar...")
        
        elif choice == "1":
            print("\n--- Opcion 1: Preprocesar Dataset ---")
            audio_path = input("Ruta a la carpeta con audios: ").strip()
            if not audio_path or not os.path.exists(audio_path):
                print("[X] Ruta invalida")
                continue
            
            PreprocessDataset(audio_path=audio_path)
            input("\nPresiona Enter para continuar...")
        
        elif choice == "2":
            print("\n--- Opcion 2: Entrenar Modelo ---")
            model_name = input("Nombre del modelo [my_model]: ").strip() or "my_model"
            config = input("Configuracion [v2_small/v2/v3]: ").strip() or "v2_small"
            
            TrainModel(name=model_name, config=config)
            input("\nPresiona Enter para continuar...")
        
        elif choice == "3":
            print("\n--- Opcion 3: Exportar Modelo ---")
            run_path = input("Ruta al run (dejar vacio para auto-detectar): ").strip() or None
            
            ExportModel(run_path=run_path)
            input("\nPresiona Enter para continuar...")
        
        elif choice == "4":
            print("\n--- Opcion 4: Limpiar Datos de Usuario ---")
            CleanUserData()
            input("\nPresiona Enter para continuar...")
        
        elif choice == "0":
            print("\n¡Hasta luego!")
            break
        
        else:
            print("\n[X] Opcion invalida. Por favor, selecciona una opcion valida.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    import argparse
    import sys
    
    # If no arguments provided, launch interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description="RAVE Training and Inference CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess audio dataset")
    preprocess_parser.add_argument("audio_path", help="Path to folder containing audio files")
    preprocess_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    preprocess_parser.add_argument("--no-lazy", action="store_true", help="Disable lazy loading")
    preprocess_parser.add_argument("--max-db-size", type=int, default=10, help="Max database size in GB (default: 10)")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a RAVE model")
    train_parser.add_argument("--name", default="my_model", help="Model name (default: my_model)")
    train_parser.add_argument("--config", default="v2_small", help="Architecture config (default: v2_small)")
    train_parser.add_argument("--db-path", default="preprocessed_data", help="Path to preprocessed dataset")
    train_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    train_parser.add_argument("--val-every", type=int, default=1000, help="Checkpoint every N steps (default: 1000)")
    train_parser.add_argument("--save-every", type=int, default=10000, help="Save every N steps (default: 10000)")
    train_parser.add_argument("--max-steps", type=int, default=6000000, help="Max training steps (default: 6000000)")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export trained model to TorchScript")
    export_parser.add_argument("--run-path", help="Path to training run folder (auto-detects if not provided)")
    
    # Workflow command (full pipeline)
    workflow_parser = subparsers.add_parser("workflow", help="Run complete workflow: preprocess → train → export")
    workflow_parser.add_argument("audio_path", help="Path to folder containing audio files")
    workflow_parser.add_argument("--name", default="my_model", help="Model name (default: my_model)")
    workflow_parser.add_argument("--config", default="v2_small", help="Architecture config (default: v2_small)")
    workflow_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    workflow_parser.add_argument("--val-every", type=int, default=1000, help="Checkpoint every N steps (default: 1000)")
    workflow_parser.add_argument("--max-steps", type=int, default=6000000, help="Max training steps (default: 6000000)")
    
    # Generate command (use model)
    generate_parser = subparsers.add_parser("generate", help="Generate audio using a trained model")
    generate_parser.add_argument("--model", default="models/demo_model/demo_model.ts", help="Path to model file")
    generate_parser.add_argument("--audio", default="input_data/demo_data/audio1.wav", help="Path to sample audio file")
    generate_parser.add_argument("--output", default="generated", help="Output filename (without extension)")
    generate_parser.add_argument("--no-random", action="store_true", help="Use input audio's latent instead of random")
    
    # Clean command (delete all user data)
    clean_parser = subparsers.add_parser("clean", help="Delete all user data (preprocessed, checkpoints, exports, outputs)")
    
    # Stream command (real-time audio generation)
    stream_parser = subparsers.add_parser("stream", help="Generate and stream audio in real-time (Opción C)")
    stream_parser.add_argument("--model", default="models/demo_model/demo_model.ts", help="Path to .ts model file (default: demo model)")
    stream_parser.add_argument("--sr", type=int, default=44100, help="Sample rate in Hz (default: 44100)")
    stream_parser.add_argument("--latent-size", type=int, default=None, help="Latent vector size (default: auto-detect)")
    stream_parser.add_argument("--chunk-duration", type=float, default=1.0, help="Audio chunk duration in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        PreprocessDataset(
            audio_path=args.audio_path,
            channels=args.channels,
            lazy=not args.no_lazy,
            max_db_size=args.max_db_size
        )
    
    elif args.command == "train":
        TrainModel(
            name=args.name,
            config=args.config,
            db_path=args.db_path,
            channels=args.channels,
            val_every=args.val_every,
            save_every=args.save_every,
            max_steps=args.max_steps,
            batch_size=args.batch_size
        )
    
    elif args.command == "export":
        ExportModel(
            run_path=args.run_path
        )
    
    elif args.command == "workflow":
        train_workflow(
            audio_path=args.audio_path,
            model_name=args.name,
            config=args.config,
            channels=args.channels,
            val_every=args.val_every,
            max_steps=args.max_steps
        )
    
    elif args.command == "generate":
        UseModel(
            model_path=args.model,
            audio_path=args.audio,
            output_name=args.output,
            random=not args.no_random
        )
    
    elif args.command == "clean":
        CleanUserData()
    
    elif args.command == "stream":
        StreamAudio(
            model_path=args.model,
            sr=args.sr,
            latent_size=args.latent_size,
            chunk_duration=args.chunk_duration
        )