# -*- coding: utf-8 -*-
"""
RAVE-TFG Main CLI and Interactive Menu
Real-time Audio Variational autoEncoder wrapper
"""
import os
import sys

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import module functions
from src.preprocess import PreprocessDataset
from src.train import TrainModel
from src.export import ExportModel
from src.generate import UseModel
from src.stream import StreamAudio
from src.stream_gui import launch_gui
from src.workflow import train_workflow
from src.clean import CleanUserData
from src.train_prior import TrainPrior


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
        print("  C) Streaming con GUI (Interfaz Visual) ** NUEVO **")
        print("  D) Streaming con Teclado (Controles por teclas)")
        print("  E) Entrenar Prior para modelo RAVE")
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
            duration = input("Duracion en segundos [30]: ").strip()
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
            print("\n--- Opcion C: Streaming con GUI (Interfaz Visual) ---")
            print("\nAbriendo ventana de streaming...")
            print("Cierra la ventana cuando termines.\n")
            
            try:
                launch_gui()
            except Exception as e:
                print(f"\n[X] Error al abrir GUI: {e}")
                import traceback
                traceback.print_exc()
            
            input("\nPresiona Enter para continuar...")
        
        elif choice == "D":
            print("\n--- Opcion D: Streaming con Teclado (Controles por teclas) ---")
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
            
            # Sample rate input
            sr_input = input("\nSample rate en Hz (22050/44100/48000) [44100]: ").strip()
            sr = int(sr_input) if sr_input else 44100
            
            chunk_duration = input("Duracion de chunk en segundos [1.0]: ").strip()
            chunk_duration = float(chunk_duration) if chunk_duration else 1.0
            
            interactive_mode = input("Habilitar controles en tiempo real? (s/n) [s]: ").strip().lower()
            interactive = interactive_mode != 'n'
            
            use_prior_mode = input("Usar prior del modelo (si disponible)? (s/n) [s]: ").strip().lower()
            use_prior = use_prior_mode != 'n'
            
            StreamAudio(model_path=model_path, sr=sr, chunk_duration=chunk_duration, interactive=interactive, use_prior=use_prior)
            input("\nPresiona Enter para continuar...")
        
        elif choice == "E":
            print("\n--- Opcion E: Entrenar Prior para Modelo RAVE ---")
            print("\n[!] IMPORTANTE: MSPrior requiere el CHECKPOINT original de RAVE (.ckpt)")
            print("    NO funciona con archivos .ts exportados")
            print("    Necesitas haber entrenado tu propio modelo RAVE (Opcion B)\n")
            
            # Get RAVE checkpoint path
            rave_checkpoint = input("Ruta al checkpoint RAVE (.ckpt) [auto-buscar]: ").strip()
            
            if not rave_checkpoint:
                # Try to find checkpoint automatically
                checkpoint_dir = "models/user_model/checkpoints"
                if os.path.exists(checkpoint_dir):
                    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
                    if ckpt_files:
                        # Prefer last.ckpt
                        if 'last.ckpt' in ckpt_files:
                            rave_checkpoint = os.path.join(checkpoint_dir, 'last.ckpt')
                        else:
                            rave_checkpoint = os.path.join(checkpoint_dir, ckpt_files[0])
                        print(f"[OK] Checkpoint encontrado: {rave_checkpoint}")
                    else:
                        print("[X] No se encontraron checkpoints en models/user_model/checkpoints")
                        print("    Primero entrena un modelo RAVE (Opcion B)")
                        continue
                else:
                    print("[X] No existe directorio de checkpoints")
                    print("    Primero entrena un modelo RAVE (Opcion B)")
                    continue
            
            if not os.path.exists(rave_checkpoint):
                print(f"[X] Checkpoint no encontrado: {rave_checkpoint}")
                continue
            
            # Get audio dataset path
            audio_path = input("Ruta al dataset de audio (mismos datos que usaste para entrenar RAVE): ").strip()
            if not audio_path or not os.path.exists(audio_path):
                print("[X] Ruta invalida o no existe")
                continue
            
            # Get prior name
            prior_name = input("Nombre para el prior [my_prior]: ").strip() or "my_prior"
            
            # Get configuration
            print("\nConfiguraciones disponibles:")
            print("  decoder_only (recomendado): Modelo autoregressivo incondicional")
            print("  recurrent: Mas ligero, usa GRU en lugar de Transformer")
            config = input("Configuracion [decoder_only]: ").strip() or "decoder_only"
            
            print("\n[!] El entrenamiento puede tardar horas. Presiona Ctrl+C cuando quieras detenerlo.")
            confirm = input("Continuar? (s/n) [s]: ").strip().lower()
            if confirm == 'n':
                continue
            
            TrainPrior(
                rave_model_path=rave_checkpoint,
                audio_path=audio_path,
                prior_name=prior_name,
                config=config
            )
            input("\nPresiona Enter para continuar...")
        
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
            print("\nHasta luego!")
            break
        
        else:
            print("\n[X] Opcion invalida. Por favor, selecciona una opcion valida.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    import argparse
    
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
    stream_parser.add_argument("--no-interactive", action="store_true", help="Disable real-time parameter controls")
    stream_parser.add_argument("--no-prior", action="store_true", help="Disable prior (use random noise instead of learned distribution)")
    
    # Train Prior command (MSPrior integration)
    train_prior_parser = subparsers.add_parser("train_prior", help="Train a prior for a RAVE model using MSPrior")
    train_prior_parser.add_argument("--rave", required=True, help="Path to RAVE checkpoint (.ckpt) - NOT .ts file")
    train_prior_parser.add_argument("--audio", required=True, help="Path to audio dataset folder (same data used to train RAVE)")
    train_prior_parser.add_argument("--name", default="my_prior", help="Name for the prior model (default: my_prior)")
    train_prior_parser.add_argument("--config", default="decoder_only", 
                                   choices=["decoder_only", "recurrent", "encoder_decoder", "encoder_decoder_continuous"],
                                   help="MSPrior configuration (default: decoder_only)")
    train_prior_parser.add_argument("--output", default="models/user_model/prior", help="Output path for prior (default: models/user_model/prior)")
    
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
            chunk_duration=args.chunk_duration,
            interactive=not args.no_interactive,
            use_prior=not args.no_prior
        )
    
    elif args.command == "train_prior":
        TrainPrior(
            rave_model_path=args.rave,
            audio_path=args.audio,
            prior_name=args.name,
            config=args.config,
            output_path=args.output
        )
