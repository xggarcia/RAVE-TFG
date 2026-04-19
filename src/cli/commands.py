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
    subparsers.add_parser("stream", help="Launch multi-model GUI streaming")

    train_prior_parser = subparsers.add_parser("train_prior", help="Train a prior for a RAVE model using MSPrior")
    train_prior_parser.add_argument("--rave", required=True, help="Path to RAVE checkpoint (.ckpt) - NOT .ts file")
    train_prior_parser.add_argument("--audio", required=True, help="Path to audio dataset folder (same data used to train RAVE)")
    train_prior_parser.add_argument("--name", default="my_prior", help="Name for the prior model (default: my_prior)")
    train_prior_parser.add_argument(
        "--config",
        default="decoder_only",
        choices=["decoder_only", "recurrent", "encoder_decoder", "encoder_decoder_continuous"],
        help="MSPrior configuration (default: decoder_only)",
    )
    train_prior_parser.add_argument("--output", default="models/user_model/prior", help="Output path for prior (default: models/user_model/prior)")

    phase_train_parser = subparsers.add_parser("phase_train", help="Phase-aware training: train + generate phase anchors")
    phase_train_parser.add_argument("audio_base", help="Base folder with phase subfolders (e.g. soft_rain/, rain/, storm/)")
    phase_train_parser.add_argument("--name", default="my_phase_model", help="Model name (default: my_phase_model)")
    phase_train_parser.add_argument("--config", default="v2_small", help="Architecture config (default: v2_small)")
    phase_train_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    phase_train_parser.add_argument("--val-every", type=int, default=1000, help="Checkpoint every N steps (default: 1000)")
    phase_train_parser.add_argument("--max-steps", type=int, default=6000000, help="Max training steps (default: 6000000)")
    phase_train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    phase_train_parser.add_argument("--phases", help="Comma-separated phase order (default: alphabetical)")

    gen_anchors_parser = subparsers.add_parser("gen_anchors", help="Generate phase anchors for an existing model")
    gen_anchors_parser.add_argument("--model", required=True, help="Path to exported .ts model")
    gen_anchors_parser.add_argument("--audio-base", required=True, help="Base folder with phase subfolders")
    gen_anchors_parser.add_argument("--phases", help="Comma-separated phase order (default: alphabetical)")

    return parser


def run_command(args):
    """Execute a parsed CLI command."""
    if args.command == "preprocess":
        from src.preprocess import PreprocessDataset

        PreprocessDataset(
            audio_path=args.audio_path,
            channels=args.channels,
            lazy=args.lazy,
            max_db_size=args.max_db_size,
        )

    elif args.command == "train":
        from src.train import TrainModel

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
        from src.export import ExportModel

        ExportModel(run_path=args.run_path)

    elif args.command == "workflow":
        from src.workflow import train_workflow

        train_workflow(
            audio_path=args.audio_path,
            model_name=args.name,
            config=args.config,
            channels=args.channels,
            val_every=args.val_every,
            max_steps=args.max_steps,
        )

    elif args.command == "generate":
        from src.generate import UseModel

        UseModel(
            model_path=args.model,
            audio_path=args.audio,
            output_name=args.output,
            random=not args.no_random,
        )

    elif args.command == "clean":
        from src.clean import CleanUserData

        CleanUserData()

    elif args.command == "stream":
        from src.stream_gui import launch_gui

        launch_gui()

    elif args.command == "train_prior":
        from src.train_prior import TrainPrior

        TrainPrior(
            rave_model_path=args.rave,
            audio_path=args.audio,
            prior_name=args.name,
            config=args.config,
            output_path=args.output,
        )

    elif args.command == "phase_train":
        from src.phase_workflow import phase_train_workflow

        phase_labels = [l.strip() for l in args.phases.split(",")] if args.phases else None
        phase_train_workflow(
            audio_base_path=args.audio_base,
            phase_labels=phase_labels,
            model_name=args.name,
            config=args.config,
            channels=args.channels,
            val_every=args.val_every,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
        )

    elif args.command == "gen_anchors":
        from src.phase_workflow import generate_anchors_only

        phase_labels = [l.strip() for l in args.phases.split(",")] if args.phases else None
        generate_anchors_only(
            model_path=args.model,
            audio_base_path=args.audio_base,
            phase_labels=phase_labels,
        )
