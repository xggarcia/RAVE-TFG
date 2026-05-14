import os

from .menu_actions_dataset import dataset_creation_menu


def run_full_workflow(ctx):
    print("\n--- Full Training Workflow (preprocess -> train -> export) ---")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    audio_path = ctx.ask_path("Audio folder path: ")
    if not audio_path or not os.path.exists(audio_path):
        print("[X] Invalid audio folder path.")
        ctx.pause()
        return

    if mode == "1":
        ctx.workflow_fn(audio_path=audio_path, model_name="my_model", config="v2_small", max_steps=6000000)
        ctx.pause()
        return

    model_name = input("Model name [my_model]: ").strip() or "my_model"
    config = input("Config [v2_small/v2/v3] [v2_small]: ").strip() or "v2_small"
    extra_raw = input("Extra configs, comma-separated (e.g. noise,causal) [none]: ").strip()
    extra_configs = [c.strip() for c in extra_raw.split(",") if c.strip()] if extra_raw else None
    channels = ctx.ask_int("Channels [1]: ", 1)
    val_every = ctx.ask_int("Validation every N steps [1000]: ", 1000)
    max_steps = ctx.ask_int("Max steps [6000000]: ", 6000000)
    batch_size = ctx.ask_int("Batch size [8] (try 16/32 on 24GB+ GPUs): ", 8)

    from src.core.train import find_latest_run
    existing_run = find_latest_run("models/user_model/checkpoints", model_name)
    resume = False
    if existing_run:
        print(f"[i] Found previous run: {existing_run}")
        resume = ctx.ask_yes_no("Resume from latest checkpoint", default_yes=True)

    ctx.workflow_fn(
        audio_path=audio_path,
        model_name=model_name,
        config=config,
        extra_configs=extra_configs,
        channels=channels,
        val_every=val_every,
        max_steps=max_steps,
        batch_size=batch_size,
        ckpt=True if resume else None,
    )
    ctx.pause()


def run_preprocess(ctx):
    print("\n--- Preprocess Dataset ---")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    audio_path = ctx.ask_path("Audio folder path: ")
    if not audio_path or not os.path.exists(audio_path):
        print("[X] Invalid audio folder path.")
        ctx.pause()
        return

    if mode == "1":
        ctx.preprocess_fn(audio_path=audio_path)
        ctx.pause()
        return

    channels = ctx.ask_int("Number of channels [1]: ", 1)
    lazy = ctx.ask_yes_no("Enable lazy loading (slower training, smaller DB)", default_yes=False)
    max_db_size = ctx.ask_int("Max DB size in GB [10]: ", 10)
    ctx.preprocess_fn(audio_path=audio_path, channels=channels, lazy=lazy, max_db_size=max_db_size)
    ctx.pause()


def run_train_model(ctx):
    print("\n--- Train Model ---")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    if mode == "1":
        model_name = input("Model name [my_model]: ").strip() or "my_model"
        ctx.train_fn(name=model_name, config="v2_small")
        ctx.pause()
        return

    model_name = input("Model name [my_model]: ").strip() or "my_model"
    config = input("Config [v2_small/v2/v3] [v2_small]: ").strip() or "v2_small"
    extra_raw = input("Extra configs, comma-separated (e.g. noise,causal) [none]: ").strip()
    extra_configs = [c.strip() for c in extra_raw.split(",") if c.strip()] if extra_raw else None
    db_path = input("Preprocessed dataset path [preprocessed_data]: ").strip() or "preprocessed_data"
    channels = ctx.ask_int("Channels [1]: ", 1)
    val_every = ctx.ask_int("Validation every N steps [1000]: ", 1000)
    save_every = ctx.ask_int("Save every N steps [10000]: ", 10000)
    max_steps = ctx.ask_int("Max steps [6000000]: ", 6000000)
    batch_size = ctx.ask_int("Batch size [8]: ", 8)

    from src.core.train import find_latest_run
    existing_run = find_latest_run("models/user_model/checkpoints", model_name)
    resume = False
    if existing_run:
        print(f"[i] Found previous run: {existing_run}")
        resume = ctx.ask_yes_no("Resume from latest checkpoint", default_yes=True)

    ctx.train_fn(
        name=model_name,
        config=config,
        extra_configs=extra_configs,
        db_path=db_path,
        channels=channels,
        val_every=val_every,
        save_every=save_every,
        max_steps=max_steps,
        batch_size=batch_size,
        ckpt=True if resume else None,
    )
    ctx.pause()


def run_export_model(ctx):
    print("\n--- Export Model to TorchScript ---")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    if mode == "1":
        ctx.export_fn(run_path=None)
        ctx.pause()
        return

    run_path = input("Run folder path (leave empty for auto-detect): ").strip() or None
    ctx.export_fn(run_path=run_path)
    ctx.pause()


def data_training_steps_menu(ctx):
    while True:
        print("\n" + "=" * 60)
        print("  Data & Training - Step-by-Step")
        print("=" * 60)
        print("  1) Preprocess dataset")
        print("  2) Train model")
        print("  3) Export model")
        print("  8) Back")
        print("  9) Home")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "3", "8", "9"})
        if choice == "1":
            run_preprocess(ctx)
        elif choice == "2":
            run_train_model(ctx)
        elif choice == "3":
            run_export_model(ctx)
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"


def data_training_menu(ctx):
    while True:
        print("\n" + "=" * 60)
        print("  Data & Training")
        print("=" * 60)
        print("  1) Full workflow (recommended)")
        print("  2) Step-by-step tools")
        print("  6) Dataset creation")
        print("  8) Back")
        print("  9) Home")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "6", "8", "9"})
        if choice == "1":
            run_full_workflow(ctx)
        elif choice == "2":
            nested = data_training_steps_menu(ctx)
            if nested == "HOME":
                return "HOME"
        elif choice == "6":
            nested = dataset_creation_menu(ctx)
            if nested == "HOME":
                return "HOME"
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"
