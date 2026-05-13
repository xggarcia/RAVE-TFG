import argparse


def build_parser():
    """Create the command-line parser for non-interactive execution."""
    parser = argparse.ArgumentParser(description="RAVE Training and Inference CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess audio dataset")
    preprocess_parser.add_argument("audio_path", help="Path to folder containing audio files")
    preprocess_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    preprocess_parser.add_argument("--lazy", action="store_true", help="Enable lazy loading (slower training, smaller DB)")
    preprocess_parser.add_argument("--max-db-size", type=int, default=10, help="Max database size in GB (default: 10)")

    train_parser = subparsers.add_parser("train", help="Train a RAVE model")
    train_parser.add_argument("--name", default="my_model", help="Model name (default: my_model)")
    train_parser.add_argument("--config", default="v2_small", help="Architecture config (default: v2_small)")
    train_parser.add_argument("--db-path", default="preprocessed_data", help="Path to preprocessed dataset")
    train_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    train_parser.add_argument("--val-every", type=int, default=1000, help="Checkpoint every N steps (default: 1000)")
    train_parser.add_argument("--save-every", type=int, default=10000, help="Save every N steps (default: 10000)")
    train_parser.add_argument("--max-steps", type=int, default=6000000, help="Max training steps (default: 6000000)")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    train_parser.add_argument("--gin-override", action="append", default=[], metavar="KEY=VALUE", help="Override gin config bindings (e.g. PHASE_1_DURATION=350000)")
    train_parser.add_argument("--extra-config", action="append", default=[], metavar="CONFIG", help="Additional RAVE configs to stack (e.g. noise, causal)")

    export_parser = subparsers.add_parser("export", help="Export trained model to TorchScript")
    export_parser.add_argument("--run-path", help="Path to training run folder (auto-detects if not provided)")

    workflow_parser = subparsers.add_parser("workflow", help="Run complete workflow: preprocess -> train -> export")
    workflow_parser.add_argument("audio_path", help="Path to folder containing audio files")
    workflow_parser.add_argument("--name", default="my_model", help="Model name (default: my_model)")
    workflow_parser.add_argument("--config", default="v2_small", help="Architecture config (default: v2_small)")
    workflow_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    workflow_parser.add_argument("--val-every", type=int, default=1000, help="Checkpoint every N steps (default: 1000)")
    workflow_parser.add_argument("--max-steps", type=int, default=6000000, help="Max training steps (default: 6000000)")

    generate_parser = subparsers.add_parser("generate", help="Generate audio using a trained model")
    generate_parser.add_argument("--model", default="models/demo_model/demo_model.ts", help="Path to model file")
    generate_parser.add_argument("--audio", default="input_data/demo_data/audio1.wav", help="Path to sample audio file")
    generate_parser.add_argument("--output", default="generated", help="Output filename (without extension)")
    generate_parser.add_argument("--no-random", action="store_true", help="Use input audio's latent instead of random")

    subparsers.add_parser("clean", help="Delete all user data (preprocessed, checkpoints, exports, outputs)")

    return parser


def run_command(args):
    """Execute a parsed CLI command."""
    if args.command == "preprocess":
        from src.core.preprocess import PreprocessDataset

        PreprocessDataset(
            audio_path=args.audio_path,
            channels=args.channels,
            lazy=args.lazy,
            max_db_size=args.max_db_size,
        )

    elif args.command == "train":
        from src.core.train import TrainModel

        TrainModel(
            name=args.name,
            config=args.config,
            extra_configs=args.extra_config,
            db_path=args.db_path,
            channels=args.channels,
            val_every=args.val_every,
            save_every=args.save_every,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            overrides=args.gin_override,
        )

    elif args.command == "export":
        from src.core.export import ExportModel

        ExportModel(run_path=args.run_path)

    elif args.command == "workflow":
        from src.core.workflow import train_workflow

        train_workflow(
            audio_path=args.audio_path,
            model_name=args.name,
            config=args.config,
            channels=args.channels,
            val_every=args.val_every,
            max_steps=args.max_steps,
        )

    elif args.command == "generate":
        from src.core.generate import UseModel

        UseModel(
            model_path=args.model,
            audio_path=args.audio,
            output_name=args.output,
            random=not args.no_random,
        )

    elif args.command == "clean":
        from src.core.clean import CleanUserData

        CleanUserData()
