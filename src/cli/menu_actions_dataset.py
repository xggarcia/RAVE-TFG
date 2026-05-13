import os
from pathlib import Path


def run_dataset_do_all(ctx):
    print("\n--- Dataset Creation: DO ALL ---")
    csv_path = ctx.ask_path("Query CSV path: ")
    if not csv_path or not os.path.exists(csv_path):
        print("[X] Invalid query CSV path.")
        ctx.pause()
        return

    temp_preview_root = input("Temporary preview folder [descSounds]: ").strip() or "descSounds"
    selected_csv_dir = input("Selected IDs CSV output folder [database/database_download/user]: ").strip() or "database/database_download/user"
    final_root = input("Final audio output root [input_data/user_data]: ").strip() or "input_data/user_data"

    try:
        from src.database.create_database import run_create_database_workflow
        from src.database.download_csv import _load_dotenv
    except ModuleNotFoundError:
        from database.create_database import run_create_database_workflow
        from database.download_csv import _load_dotenv

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
    csv_path = ctx.ask_path("Query CSV path: ")
    if not csv_path or not os.path.exists(csv_path):
        print("[X] Invalid query CSV path.")
        ctx.pause()
        return

    temp_preview_root = input("Folder to save all preview candidates [descSounds]: ").strip() or "descSounds"

    try:
        from dataclasses import replace

        from src.database.download_csv import _load_dotenv
        from src.database.first_download_freesound import download_sounds_freesound, read_jobs_from_csv
    except ModuleNotFoundError:
        from dataclasses import replace

        from database.download_csv import _load_dotenv
        from database.first_download_freesound import download_sounds_freesound, read_jobs_from_csv

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
        from src.database.create_csv import gather_candidates, run_selection, write_selected_ids_csv
    except ModuleNotFoundError:
        from database.create_csv import gather_candidates, run_selection, write_selected_ids_csv

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
    selected_csv_path = os.path.normpath(input("Selected IDs CSV path: ").strip())
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
        from src.database.download_csv import _load_dotenv, _read_sound_ids, download_sound_by_id
    except ModuleNotFoundError:
        from database.download_csv import _load_dotenv, _read_sound_ids, download_sound_by_id

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
        from src.database.normalize_volume import normalize_directory
    except ModuleNotFoundError:
        from database.normalize_volume import normalize_directory

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
        from src.database.merge_selected_csv import merge_selected_csvs
    except ModuleNotFoundError:
        from database.merge_selected_csv import merge_selected_csvs

    total = merge_selected_csvs(Path(input_dir), Path(output_path))
    if total:
        print(f"\n[OK] Written {total} sound ID(s) to: {output_path}")
    ctx.pause()


def run_convert_format(ctx):
    print("\n--- Convert Format (single folder) ---")
    folder = input("Folder containing audio files: ").strip()
    if not folder or not os.path.exists(folder):
        print("[X] Folder not found.")
        ctx.pause()
        return

    target_sr_str = input("Target sample rate [44100]: ").strip()
    target_channels_str = input("Target channels [1]: ").strip()
    target_subtype = input("Target WAV subtype [PCM_16]: ").strip() or "PCM_16"

    try:
        target_sr = int(target_sr_str) if target_sr_str else 44100
        target_channels = int(target_channels_str) if target_channels_str else 1
    except ValueError:
        print("[X] Sample rate and channels must be integers.")
        ctx.pause()
        return

    if target_sr < 1 or target_channels < 1:
        print("[X] Sample rate and channels must be >= 1.")
        ctx.pause()
        return

    try:
        from src.database.convert_format import convert_directory
    except ModuleNotFoundError:
        from database.convert_format import convert_directory

    print(
        f"\nConverting audio in: {folder}  "
        f"(target: {target_sr} Hz, {target_channels} channel(s), {target_subtype})"
    )
    ok, total = convert_directory(
        root=Path(folder),
        target_sr=target_sr,
        target_channels=target_channels,
        target_subtype=target_subtype,
    )

    if total == 0:
        print("[!] No audio files found.")
    else:
        print(f"\n[OK] Converted {ok}/{total} file(s).")
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
        print("  7) Convert format/sample rate (single folder)")
        print("  8) Back")
        print("  9) Home")

        choice = ctx.ask_choice("\nChoose: ", {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
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
        elif choice == "7":
            run_convert_format(ctx)
        elif choice == "8":
            return "BACK"
        elif choice == "9":
            return "HOME"
