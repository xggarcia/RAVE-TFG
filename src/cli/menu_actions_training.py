import os
from pathlib import Path


def run_dataset_do_all(ctx):
    print("\n--- Dataset Creation: DO ALL ---")
    csv_path = input("Query CSV path: ").strip()
    if not csv_path or not os.path.exists(csv_path):
        print("[X] Invalid query CSV path.")
        ctx.pause()
        return

    temp_preview_root = input("Temporary preview folder [descSounds]: ").strip() or "descSounds"
    selected_csv_dir = input("Selected IDs CSV output folder [database/database_download/user]: ").strip() or "database/database_download/user"
    final_root = input("Final audio output root [input_data/user_data]: ").strip() or "input_data/user_data"

    try:
        from src.database_creation.create_database import run_create_database_workflow
        from src.database_creation.download_csv import _load_dotenv
    except ModuleNotFoundError:
        from database_creation.create_database import run_create_database_workflow
        from database_creation.download_csv import _load_dotenv

    _load_dotenv()
    run_create_database_workflow(
        jobs_csv_path=Path(csv_path),
        selected_csv_dir=Path(selected_csv_dir),
        final_root=Path(final_root),
        temp_preview_root=Path(temp_preview_root),
    )
    ctx.pause()


def run_dataset_first_download_only(ctx):
    print("\n--- Dataset Creation: FIRST DOWNLOAD ONLY ---")
    csv_path = input("Query CSV path: ").strip()
    if not csv_path or not os.path.exists(csv_path):
        print("[X] Invalid query CSV path.")
        ctx.pause()
        return

    temp_preview_root = input("Folder to save all preview candidates [descSounds]: ").strip() or "descSounds"

    try:
        from dataclasses import replace
        from pathlib import Path

        from src.database_creation.download_csv import _load_dotenv
        from src.database_creation.first_download_freesound import download_sounds_freesound, read_jobs_from_csv
    except ModuleNotFoundError:
        from dataclasses import replace
        from pathlib import Path

        from database_creation.download_csv import _load_dotenv
        from database_creation.first_download_freesound import download_sounds_freesound, read_jobs_from_csv

    _load_dotenv()

    try:
        jobs = read_jobs_from_csv(Path(csv_path))
    except Exception as exc:
        print(f"[X] Could not read jobs CSV: {exc}")
        ctx.pause()
        return

    if not jobs:
        print("No valid query jobs found in CSV.")
        ctx.pause()
        return

    jobs = [replace(job, output_dir=Path(temp_preview_root)) for job in jobs]
    total_saved = 0
    for idx, job in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] First download for query: {job.query_text}")
        total_saved += download_sounds_freesound(job)

    print(f"\n[OK] First download completed. Saved {total_saved} preview candidate(s).")
    ctx.pause()


def run_dataset_select_only(ctx):
    print("\n--- Dataset Creation: SELECT FROM DOWNLOADED PREVIEWS ---")
    preview_root = input("Folder containing preview candidates [descSounds]: ").strip() or "descSounds"
    if not os.path.exists(preview_root):
        print("[X] Preview folder does not exist.")
        ctx.pause()
        return

    output_dir = input("Folder to save selected IDs CSV [database/database_download/user]: ").strip() or "database/database_download/user"
    csv_name = input("Selected IDs CSV filename [selected_sound_ids.csv]: ").strip() or "selected_sound_ids.csv"

    try:
        from pathlib import Path

        from src.database_creation.create_csv import gather_candidates, run_selection, write_selected_ids_csv
    except ModuleNotFoundError:
        from pathlib import Path

        from database_creation.create_csv import gather_candidates, run_selection, write_selected_ids_csv

    candidates = gather_candidates(Path(preview_root))
    if not candidates:
        print("No playable preview candidates found.")
        ctx.pause()
        return

    selected_ids = run_selection(candidates)
    output_csv = Path(output_dir) / csv_name
    write_selected_ids_csv(selected_ids, output_csv)
    print(f"\n[OK] Saved {len(selected_ids)} selected ID(s) to: {output_csv}")
    ctx.pause()


def run_dataset_download_only(ctx):
    print("\n--- Dataset Creation: DOWNLOAD SELECTED IDS ---")
    selected_csv_path = input("Selected IDs CSV path: ").strip()
    if not selected_csv_path or not os.path.exists(selected_csv_path):
        print("[X] Invalid selected IDs CSV path.")
        ctx.pause()
        return

    final_output_dir = input("Final audio output folder: ").strip()
    if not final_output_dir:
        print("[X] Output folder path is required.")
        ctx.pause()
        return

    try:
        from pathlib import Path

        from src.database_creation.download_csv import _load_dotenv, _read_sound_ids, download_sound_by_id
    except ModuleNotFoundError:
        from pathlib import Path

        from database_creation.download_csv import _load_dotenv, _read_sound_ids, download_sound_by_id

    _load_dotenv()
    api_key = os.getenv("FREESOUND_API_KEY", "").strip()
    if not api_key:
        print("[X] Missing FREESOUND_API_KEY in environment/.env")
        ctx.pause()
        return

    try:
        sound_ids = _read_sound_ids(Path(selected_csv_path))
    except Exception as exc:
        print(f"[X] Could not read selected IDs CSV: {exc}")
        ctx.pause()
        return

    output_dir = Path(final_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sound_ids:
        print("No sound IDs found in selected CSV.")
        ctx.pause()
        return

    success = 0
    for idx, sound_id in enumerate(sound_ids, start=1):
        print(f"[{idx}/{len(sound_ids)}] Downloading {sound_id}")
        if download_sound_by_id(sound_id, api_key, output_dir, skip_existing=True):
            success += 1

    print(f"\n[OK] Downloaded {success}/{len(sound_ids)} selected sound(s) into: {output_dir}")
    ctx.pause()


def run_normalize_volume(ctx):
    print("\n--- Normalize Volume (single folder) ---")
    folder = input("Folder containing audio files: ").strip()
    if not folder or not os.path.exists(folder):
        print("[X] Folder not found.")
        ctx.pause()
        return

    target_db_str = input("Target RMS in dBFS [-20.0]: ").strip()
    try:
        target_db = float(target_db_str) if target_db_str else -20.0
    except ValueError:
        print("[X] Invalid dB value.")
        ctx.pause()
        return

    try:
        from src.database_creation.normalize_volume import normalize_directory
    except ModuleNotFoundError:
        from database_creation.normalize_volume import normalize_directory

    from pathlib import Path
    print(f"\nNormalizing audio in: {folder}  (target: {target_db:.1f} dBFS)")
    ok, total = normalize_directory(Path(folder), target_db)
    if total == 0:
        print("[!] No audio files found.")
    else:
        print(f"\n[OK] Normalized {ok}/{total} file(s).")
    ctx.pause()


def run_merge_selected_csv(ctx):
    print("\n--- Merge Selected IDs into Combined CSV ---")
    input_dir = input("Folder with per-query selected-IDs CSVs [database/database_download/user]: ").strip()
    input_dir = input_dir or "database/database_download/user"
    if not os.path.isdir(input_dir):
        print("[X] Folder not found.")
        ctx.pause()
        return

    output_path = input("Output CSV path [database/database_download/user/combined_ids.csv]: ").strip()
    output_path = output_path or "database/database_download/user/combined_ids.csv"

    try:
        from src.database_creation.merge_selected_csv import merge_selected_csvs
    except ModuleNotFoundError:
        from database_creation.merge_selected_csv import merge_selected_csvs

    from pathlib import Path
    total = merge_selected_csvs(Path(input_dir), Path(output_path))
    if total:
        print(f"\n[OK] Written {total} sound ID(s) to: {output_path}")
    ctx.pause()


def dataset_creation_menu(ctx):
    while True:
        print("\n" + "=" * 60)
        print("  Data & Training - Dataset Creation")
        print("=" * 60)
        print("  1) DO ALL (first download -> select -> final download)")
        print("  2) First download only (all preview candidates)")
        print("  3) Select only (choose what to keep from previews)")
        print("  4) Final download only (from selected IDs CSV)")
        print("  5) Normalize volume (single folder)")
        print("  6) Merge selected IDs into combined CSV")
        print("  8) Back")
        print("  9) Home")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "3", "4", "5", "6", "8", "9"})
        if choice == "1":
            run_dataset_do_all(ctx)
        elif choice == "2":
            run_dataset_first_download_only(ctx)
        elif choice == "3":
            run_dataset_select_only(ctx)
        elif choice == "4":
            run_dataset_download_only(ctx)
        elif choice == "5":
            run_normalize_volume(ctx)
        elif choice == "6":
            run_merge_selected_csv(ctx)
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"


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
        print("  6) Dataset creation")
        print("  8) Back")
        print("  9) Home")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "3", "4", "5", "6", "8", "9"})
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
        elif choice == "6":
            nested = dataset_creation_menu(ctx)
            if nested == "HOME":
                return "HOME"
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"
