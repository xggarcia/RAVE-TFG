import os


def run_full_workflow(ctx):
    print("\n--- Full Training Workflow (preprocess -> train -> export) ---")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    audio_path = input("Audio folder path: ").strip()
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
    channels = ctx.ask_int("Channels [1]: ", 1)
    val_every = ctx.ask_int("Validation every N steps [1000]: ", 1000)
    max_steps = ctx.ask_int("Max steps [6000000]: ", 6000000)

    ctx.workflow_fn(
        audio_path=audio_path,
        model_name=model_name,
        config=config,
        channels=channels,
        val_every=val_every,
        max_steps=max_steps,
    )
    ctx.pause()


def run_preprocess(ctx):
    print("\n--- Preprocess Dataset ---")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    audio_path = input("Audio folder path: ").strip()
    if not audio_path or not os.path.exists(audio_path):
        print("[X] Invalid audio folder path.")
        ctx.pause()
        return

    if mode == "1":
        ctx.preprocess_fn(audio_path=audio_path)
        ctx.pause()
        return

    channels = ctx.ask_int("Number of channels [1]: ", 1)
    lazy = ctx.ask_yes_no("Enable lazy loading", default_yes=True)
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
    db_path = input("Preprocessed dataset path [preprocessed_data]: ").strip() or "preprocessed_data"
    channels = ctx.ask_int("Channels [1]: ", 1)
    val_every = ctx.ask_int("Validation every N steps [1000]: ", 1000)
    save_every = ctx.ask_int("Save every N steps [10000]: ", 10000)
    max_steps = ctx.ask_int("Max steps [6000000]: ", 6000000)
    batch_size = ctx.ask_int("Batch size [8]: ", 8)

    ctx.train_fn(
        name=model_name,
        config=config,
        db_path=db_path,
        channels=channels,
        val_every=val_every,
        save_every=save_every,
        max_steps=max_steps,
        batch_size=batch_size,
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


def run_train_prior(ctx):
    print("\n--- Train Prior for RAVE Model ---")
    print("[!] MSPrior requires a RAVE checkpoint (.ckpt), not an exported .ts file.")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    if mode == "1":
        rave_checkpoint = ctx.find_rave_checkpoint()
        if not rave_checkpoint:
            print("[X] No RAVE checkpoint found in models/user_model/checkpoints")
            ctx.pause()
            return
        print(f"[OK] Using checkpoint: {rave_checkpoint}")

        audio_path = input("Audio dataset path: ").strip()
        if not audio_path or not os.path.exists(audio_path):
            print("[X] Invalid dataset path.")
            ctx.pause()
            return

        prior_name = input("Prior name [my_prior]: ").strip() or "my_prior"
        if not ctx.ask_yes_no("Training can take hours. Continue", default_yes=True):
            return

        ctx.train_prior_fn(rave_model_path=rave_checkpoint, audio_path=audio_path, prior_name=prior_name, config="decoder_only")
        ctx.pause()
        return

    rave_checkpoint = input("RAVE checkpoint path (.ckpt) [auto-detect]: ").strip()
    if not rave_checkpoint:
        rave_checkpoint = ctx.find_rave_checkpoint()

    if not rave_checkpoint or not os.path.exists(rave_checkpoint):
        print("[X] Checkpoint not found.")
        ctx.pause()
        return

    audio_path = input("Audio dataset path: ").strip()
    if not audio_path or not os.path.exists(audio_path):
        print("[X] Invalid dataset path.")
        ctx.pause()
        return

    prior_name = input("Prior name [my_prior]: ").strip() or "my_prior"
    print("\nAvailable configs: decoder_only, recurrent, encoder_decoder, encoder_decoder_continuous")
    config = input("Config [decoder_only]: ").strip() or "decoder_only"

    if not ctx.ask_yes_no("Training can take hours. Continue", default_yes=True):
        return

    ctx.train_prior_fn(rave_model_path=rave_checkpoint, audio_path=audio_path, prior_name=prior_name, config=config)
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


def run_phase_training(ctx):
    print("\n--- Phase-Aware Training ---")
    print("[*] Train a model on phase-organized audio for phase interpolation.")
    print("    Organize your audio in subfolders by phase, e.g.:")
    print("      input_data/user_data/soft_rain/")
    print("      input_data/user_data/rain/")
    print("      input_data/user_data/storm/")
    print("    Subfolders are sorted alphabetically (phase order).")
    mode = ctx.model_mode_prompt()
    if mode == "9":
        return

    audio_base = input("Base folder with phase subfolders [input_data/user_data]: ").strip()
    if not audio_base:
        audio_base = "input_data/user_data"
    if not os.path.exists(audio_base):
        print(f"[X] Folder not found: {audio_base}")
        ctx.pause()
        return

    if mode == "1":
        model_name = input("Model name [my_phase_model]: ").strip() or "my_phase_model"
        ctx.phase_train_fn(audio_base_path=audio_base, model_name=model_name)
        ctx.pause()
        return

    model_name = input("Model name [my_phase_model]: ").strip() or "my_phase_model"
    config = input("Config [v2_small/v2/v3] [v2_small]: ").strip() or "v2_small"
    channels = ctx.ask_int("Channels [1]: ", 1)
    val_every = ctx.ask_int("Validation every N steps [1000]: ", 1000)
    max_steps = ctx.ask_int("Max steps [6000000]: ", 6000000)
    batch_size = ctx.ask_int("Batch size [8]: ", 8)

    custom_order = input("Custom phase order (comma-separated folder names, or empty for alphabetical): ").strip()
    phase_labels = [l.strip() for l in custom_order.split(",") if l.strip()] if custom_order else None

    ctx.phase_train_fn(
        audio_base_path=audio_base,
        phase_labels=phase_labels,
        model_name=model_name,
        config=config,
        channels=channels,
        val_every=val_every,
        max_steps=max_steps,
        batch_size=batch_size,
    )
    ctx.pause()


def run_generate_anchors(ctx):
    print("\n--- Generate Phase Anchors for Existing Model ---")
    print("[*] Create phase anchors from an already-trained model.")

    model_path = ctx.pick_model_path(default_demo=False)
    if not model_path or not os.path.exists(model_path):
        print("[X] Model not found.")
        ctx.pause()
        return

    audio_base = input("Base folder with phase subfolders: ").strip()
    if not audio_base or not os.path.exists(audio_base):
        print("[X] Folder not found.")
        ctx.pause()
        return

    custom_order = input("Custom phase order (comma-separated, or empty for alphabetical): ").strip()
    phase_labels = [l.strip() for l in custom_order.split(",") if l.strip()] if custom_order else None

    ctx.generate_anchors_fn(
        model_path=model_path,
        audio_base_path=audio_base,
        phase_labels=phase_labels,
    )
    ctx.pause()


def data_training_menu(ctx):
    while True:
        print("\n" + "=" * 60)
        print("  Data & Training")
        print("=" * 60)
        print("  1) Full workflow (recommended)")
        print("  2) Step-by-step tools")
        print("  3) Train prior (advanced)")
        print("  4) Phase-aware training")
        print("  5) Generate phase anchors (existing model)")
        print("  8) Back")
        print("  9) Home")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "3", "4", "5", "8", "9"})
        if choice == "1":
            run_full_workflow(ctx)
        elif choice == "2":
            nested = data_training_steps_menu(ctx)
            if nested == "HOME":
                return "HOME"
        elif choice == "3":
            run_train_prior(ctx)
        elif choice == "4":
            run_phase_training(ctx)
        elif choice == "5":
            run_generate_anchors(ctx)
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"
