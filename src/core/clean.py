"""Clean user data with a single confirmation prompt."""

import os
import shutil


DIRS_TO_CLEAN = [
    ("preprocessed_data", "Preprocessed dataset"),
    ("models/user_model/checkpoints", "Training checkpoints"),
    ("models/user_model/exported_model", "Exported models"),
    ("outputs", "Generated outputs"),
    ("descSounds", "Temporary Freesound preview downloads"),
    ("input_data/user_data", "Downloaded user dataset audio"),
    ("database/database_download/user", "User-selected sound ID CSV files"),
]


def _non_gitkeep_contents(dir_path):
    return [f for f in os.listdir(dir_path) if f != ".gitkeep"]


def _count_files(dir_path):
    return sum(1 for _, _, files in os.walk(dir_path) for f in files if f != ".gitkeep")


def _clean_dir(dir_path):
    for item in os.listdir(dir_path):
        if item == ".gitkeep":
            continue
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)


def CleanUserData():
    """Delete all user-related data after a single yes/no confirmation."""
    existing_dirs = [
        (path, desc) for path, desc in DIRS_TO_CLEAN
        if os.path.exists(path) and _non_gitkeep_contents(path)
    ]

    if not existing_dirs:
        print("Nothing to clean. All user data directories are already empty.")
        return False

    print("=" * 50)
    print("WARNING: USER DATA CLEANUP")
    print("=" * 50)
    print("\nThe following directories will be PERMANENTLY DELETED:\n")
    for dir_path, description in existing_dirs:
        print(f"  {dir_path}")
        print(f"     -> {description} ({_count_files(dir_path)} files)")
    print("\n" + "=" * 50)

    confirm = input("\nDelete all user data? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("\n[X] Cleanup cancelled.")
        return False

    print("\n  Deleting user data...")
    for dir_path, _ in existing_dirs:
        _clean_dir(dir_path)
        print(f"  [OK] Cleaned: {dir_path}")

    print("\n" + "=" * 50)
    print("[OK] User data cleanup completed!")
    print("=" * 50)
    return True
