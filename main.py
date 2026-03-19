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
from src.stream_gui import launch_gui
from src.workflow import train_workflow
from src.clean import CleanUserData
from src.train_prior import TrainPrior


def interactive_menu():
    """Interactive menu for RAVE workflows with category-based navigation."""

    def ask_choice(prompt, valid_choices):
        while True:
            choice = input(prompt).strip()
            if choice in valid_choices:
                return choice
            print(f"[X] Invalid choice. Valid options: {', '.join(valid_choices)}")

    def ask_int(prompt, default):
        while True:
            raw = input(prompt).strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                print("[X] Please enter an integer value.")

    def ask_yes_no(prompt, default_yes=True):
        default_label = "Y/n" if default_yes else "y/N"
        while True:
            raw = input(f"{prompt} [{default_label}]: ").strip().lower()
            if not raw:
                return default_yes
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            print("[X] Please answer with y or n.")

    def pause():
        input("\nPress Enter to continue...")

    def model_mode_prompt():
        print("\nSelect setup mode:")
        print("  1) Quick (recommended defaults)")
        print("  2) Advanced (custom parameters)")
        print("  9) Cancel")
        return ask_choice("Choose: ", {"1", "2", "9"})

    def find_exported_model():
        export_dir = "models/user_model/exported_model"
        if not os.path.exists(export_dir):
            return None
        ts_files = [f for f in os.listdir(export_dir) if f.endswith(".ts")]
        if not ts_files:
            return None
        return os.path.join(export_dir, ts_files[0])

    def find_rave_checkpoint():
        checkpoint_dir = "models/user_model/checkpoints"
        if not os.path.exists(checkpoint_dir):
            return None
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            return None
        if "last.ckpt" in ckpt_files:
            return os.path.join(checkpoint_dir, "last.ckpt")
        return os.path.join(checkpoint_dir, ckpt_files[0])

    def pick_model_path(default_demo=True):
        print("\nModel source:")
        print("  1) Demo model")
        print("  2) User exported model (.ts)")
        default = "1" if default_demo else "2"
        choice = input(f"Choose [default {default}]: ").strip() or default
        if choice == "1":
            return "models/demo_model/demo_model.ts"

        custom = input("Model .ts path (leave empty for auto-detect): ").strip()
        if custom:
            if os.path.exists(custom):
                return custom
            print(f"[X] Model not found: {custom}")
            return None

        model_path = find_exported_model()
        if model_path:
            print(f"[OK] Using detected exported model: {model_path}")
            return model_path

        print("[X] No exported .ts model found in models/user_model/exported_model")
        return None

    def run_generate_audio():
        print("\n--- Generate Audio from Model ---")
        mode = model_mode_prompt()
        if mode == "9":
            return

        if mode == "1":
            model_path = pick_model_path(default_demo=True)
            if not model_path:
                pause()
                return

            audio_path = "input_data/demo_data/audio1.wav"
            if model_path != "models/demo_model/demo_model.ts":
                audio_path = input("Reference audio path (required for user model): ").strip()
                if not audio_path or not os.path.exists(audio_path):
                    print("[X] Invalid audio path.")
                    pause()
                    return

            output_name = input("Output name [generated]: ").strip() or "generated"
            UseModel(
                model_path=model_path,
                audio_path=audio_path,
                output_name=output_name,
                duration=30
            )
            pause()
            return

        model_path = pick_model_path(default_demo=True)
        if not model_path:
            pause()
            return

        default_audio = "input_data/demo_data/audio1.wav" if model_path == "models/demo_model/demo_model.ts" else ""
        audio_prompt = f"Reference audio path [{default_audio}]: " if default_audio else "Reference audio path: "
        audio_path = input(audio_prompt).strip() or default_audio
        if not audio_path or not os.path.exists(audio_path):
            print("[X] Invalid audio path.")
            pause()
            return

        output_name = input("Output name [generated]: ").strip() or "generated"
        duration = ask_int("Duration in seconds [30]: ", 30)
        UseModel(model_path=model_path, audio_path=audio_path, output_name=output_name, duration=duration)
        pause()

    def run_gui_stream():
        print("\n--- Multi-Model GUI Streaming ---")
        print("Launching GUI window. Close it when done.\n")
        try:
            launch_gui()
        except Exception as e:
            print(f"[X] Failed to open GUI: {e}")
            import traceback
            traceback.print_exc()
        pause()

    def run_full_workflow():
        print("\n--- Full Training Workflow (preprocess -> train -> export) ---")
        mode = model_mode_prompt()
        if mode == "9":
            return

        audio_path = input("Audio folder path: ").strip()
        if not audio_path or not os.path.exists(audio_path):
            print("[X] Invalid audio folder path.")
            pause()
            return

        if mode == "1":
            train_workflow(
                audio_path=audio_path,
                model_name="my_model",
                config="v2_small",
                max_steps=6000000
            )
            pause()
            return

        model_name = input("Model name [my_model]: ").strip() or "my_model"
        config = input("Config [v2_small/v2/v3] [v2_small]: ").strip() or "v2_small"
        channels = ask_int("Channels [1]: ", 1)
        val_every = ask_int("Validation every N steps [1000]: ", 1000)
        max_steps = ask_int("Max steps [6000000]: ", 6000000)

        train_workflow(
            audio_path=audio_path,
            model_name=model_name,
            config=config,
            channels=channels,
            val_every=val_every,
            max_steps=max_steps
        )
        pause()

    def run_preprocess():
        print("\n--- Preprocess Dataset ---")
        mode = model_mode_prompt()
        if mode == "9":
            return

        audio_path = input("Audio folder path: ").strip()
        if not audio_path or not os.path.exists(audio_path):
            print("[X] Invalid audio folder path.")
            pause()
            return

        if mode == "1":
            PreprocessDataset(audio_path=audio_path)
            pause()
            return

        channels = ask_int("Number of channels [1]: ", 1)
        lazy = ask_yes_no("Enable lazy loading", default_yes=True)
        max_db_size = ask_int("Max DB size in GB [10]: ", 10)
        PreprocessDataset(audio_path=audio_path, channels=channels, lazy=lazy, max_db_size=max_db_size)
        pause()

    def run_train_model():
        print("\n--- Train Model ---")
        mode = model_mode_prompt()
        if mode == "9":
            return

        if mode == "1":
            model_name = input("Model name [my_model]: ").strip() or "my_model"
            TrainModel(name=model_name, config="v2_small")
            pause()
            return

        model_name = input("Model name [my_model]: ").strip() or "my_model"
        config = input("Config [v2_small/v2/v3] [v2_small]: ").strip() or "v2_small"
        db_path = input("Preprocessed dataset path [preprocessed_data]: ").strip() or "preprocessed_data"
        channels = ask_int("Channels [1]: ", 1)
        val_every = ask_int("Validation every N steps [1000]: ", 1000)
        save_every = ask_int("Save every N steps [10000]: ", 10000)
        max_steps = ask_int("Max steps [6000000]: ", 6000000)
        batch_size = ask_int("Batch size [8]: ", 8)

        TrainModel(
            name=model_name,
            config=config,
            db_path=db_path,
            channels=channels,
            val_every=val_every,
            save_every=save_every,
            max_steps=max_steps,
            batch_size=batch_size
        )
        pause()

    def run_export_model():
        print("\n--- Export Model to TorchScript ---")
        mode = model_mode_prompt()
        if mode == "9":
            return

        if mode == "1":
            ExportModel(run_path=None)
            pause()
            return

        run_path = input("Run folder path (leave empty for auto-detect): ").strip() or None
        ExportModel(run_path=run_path)
        pause()

    def run_train_prior():
        print("\n--- Train Prior for RAVE Model ---")
        print("[!] MSPrior requires a RAVE checkpoint (.ckpt), not an exported .ts file.")
        mode = model_mode_prompt()
        if mode == "9":
            return

        if mode == "1":
            rave_checkpoint = find_rave_checkpoint()
            if not rave_checkpoint:
                print("[X] No RAVE checkpoint found in models/user_model/checkpoints")
                pause()
                return
            print(f"[OK] Using checkpoint: {rave_checkpoint}")

            audio_path = input("Audio dataset path: ").strip()
            if not audio_path or not os.path.exists(audio_path):
                print("[X] Invalid dataset path.")
                pause()
                return

            prior_name = input("Prior name [my_prior]: ").strip() or "my_prior"
            if not ask_yes_no("Training can take hours. Continue", default_yes=True):
                return

            TrainPrior(
                rave_model_path=rave_checkpoint,
                audio_path=audio_path,
                prior_name=prior_name,
                config="decoder_only"
            )
            pause()
            return

        rave_checkpoint = input("RAVE checkpoint path (.ckpt) [auto-detect]: ").strip()
        if not rave_checkpoint:
            rave_checkpoint = find_rave_checkpoint()

        if not rave_checkpoint or not os.path.exists(rave_checkpoint):
            print("[X] Checkpoint not found.")
            pause()
            return

        audio_path = input("Audio dataset path: ").strip()
        if not audio_path or not os.path.exists(audio_path):
            print("[X] Invalid dataset path.")
            pause()
            return

        prior_name = input("Prior name [my_prior]: ").strip() or "my_prior"
        print("\nAvailable configs: decoder_only, recurrent, encoder_decoder, encoder_decoder_continuous")
        config = input("Config [decoder_only]: ").strip() or "decoder_only"

        if not ask_yes_no("Training can take hours. Continue", default_yes=True):
            return

        TrainPrior(
            rave_model_path=rave_checkpoint,
            audio_path=audio_path,
            prior_name=prior_name,
            config=config
        )
        pause()

    def run_clean_user_data():
        print("\n--- Clean User Data ---")
        if ask_yes_no("This will remove generated user data. Continue", default_yes=False):
            CleanUserData()
        pause()

    def show_help_about():
        print("\n--- Help / About ---")
        print("RAVE-TFG interactive menu:")
        print("- Generate & Stream: run inference and real-time tools")
        print("- Data & Training: preprocess, train, export, full workflow, train prior")
        print("- Maintenance: cleanup and utility information")
        print("\nTip: Use Quick mode for safe defaults, Advanced mode for full control.")
        pause()

    def generate_stream_menu():
        while True:
            print("\n" + "=" * 60)
            print("  Generate & Stream")
            print("=" * 60)
            print("  1) Generate audio from model")
            print("  2) Multi-model GUI streaming")
            print("  8) Back")
            print("  9) Home")
            print("  0) Exit")

            choice = ask_choice("\nChoose: ", {"1", "2", "8", "9", "0"})
            if choice == "1":
                run_generate_audio()
            elif choice == "2":
                run_gui_stream()
            elif choice == "8":
                return "BACK"
            elif choice == "9":
                return "HOME"
            elif choice == "0":
                return "EXIT"

    def data_training_steps_menu():
        while True:
            print("\n" + "=" * 60)
            print("  Data & Training - Step-by-Step")
            print("=" * 60)
            print("  1) Preprocess dataset")
            print("  2) Train model")
            print("  3) Export model")
            print("  8) Back")
            print("  9) Home")

            choice = ask_choice("\nChoose: ", {"1", "2", "3", "8", "9"})
            if choice == "1":
                run_preprocess()
            elif choice == "2":
                run_train_model()
            elif choice == "3":
                run_export_model()
            elif choice == "8":
                return "BACK"
            elif choice == "9":
                return "HOME"

    def data_training_menu():
        while True:
            print("\n" + "=" * 60)
            print("  Data & Training")
            print("=" * 60)
            print("  1) Full workflow (recommended)")
            print("  2) Step-by-step tools")
            print("  3) Train prior (advanced)")
            print("  8) Back")
            print("  9) Home")

            choice = ask_choice("\nChoose: ", {"1", "2", "3", "8", "9"})
            if choice == "1":
                run_full_workflow()
            elif choice == "2":
                nested = data_training_steps_menu()
                if nested == "HOME":
                    return "HOME"
            elif choice == "3":
                run_train_prior()
            elif choice == "8":
                return "BACK"
            elif choice == "9":
                return "HOME"

    def maintenance_menu():
        while True:
            print("\n" + "=" * 60)
            print("  Maintenance")
            print("=" * 60)
            print("  1) Clean user data")
            print("  2) Help / About")
            print("  8) Back")
            print("  9) Home")

            choice = ask_choice("\nChoose: ", {"1", "2", "8", "9"})
            if choice == "1":
                run_clean_user_data()
            elif choice == "2":
                show_help_about()
            elif choice == "8":
                return "BACK"
            elif choice == "9":
                return "HOME"

    while True:
        print("\n" + "=" * 60)
        print("  RAVE-TFG - Main Menu")
        print("=" * 60)
        print("  1) Generate & Stream")
        print("  2) Data & Training")
        print("  3) Maintenance")
        print("  0) Exit")

        choice = ask_choice("\nChoose: ", {"1", "2", "3", "0"})
        if choice == "1":
            result = generate_stream_menu()
            if result == "EXIT":
                print("\nGoodbye!")
                break
        elif choice == "2":
            result = data_training_menu()
            if result == "EXIT":
                print("\nGoodbye!")
                break
        elif choice == "3":
            result = maintenance_menu()
            if result == "EXIT":
                print("\nGoodbye!")
                break
        elif choice == "0":
            print("\nGoodbye!")
            break


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
    
    # Stream command (GUI only)
    subparsers.add_parser("stream", help="Launch multi-model GUI streaming")
    
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
        launch_gui()
    
    elif args.command == "train_prior":
        TrainPrior(
            rave_model_path=args.rave,
            audio_path=args.audio,
            prior_name=args.name,
            config=args.config,
            output_path=args.output
        )
