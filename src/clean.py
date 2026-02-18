# -*- coding: utf-8 -*-
"""
Clean user data with safety confirmations
"""
import os
import shutil


def CleanUserData():
    """
    Delete all user-related data: preprocessed data, checkpoints, exported models, and outputs.
    Requires double confirmation for safety.
    
    Returns:
        True if cleanup was performed, False if cancelled
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
    print("WARNING: USER DATA CLEANUP")
    print("=" * 50)
    print("\nThe following directories will be PERMANENTLY DELETED:\n")
    
    for dir_path, description in existing_dirs:
        # Count files excluding .gitkeep
        file_count = sum(1 for _, _, files in os.walk(dir_path) for f in files if f != '.gitkeep')
        print(f"  {dir_path}")
        print(f"     -> {description} ({file_count} files)")
    
    print("\n" + "=" * 50)
    
    # First confirmation
    print("\nFIRST CONFIRMATION:")
    confirm1 = input("Are you sure you want to delete all user data? (yes/no): ").strip().lower()
    
    if confirm1 != "yes":
        print("\n[X] Cleanup cancelled.")
        return False
    
    # Second confirmation
    print("\nSECOND CONFIRMATION:")
    print("Type 'DELETE ALL USER DATA' to confirm:")
    confirm2 = input("> ").strip()
    
    if confirm2 != "DELETE ALL USER DATA":
        print("\n[X] Cleanup cancelled. Confirmation text did not match.")
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
            print(f"  [OK] Cleaned: {dir_path}")
        except Exception as e:
            print(f"  [X] Error cleaning {dir_path}: {e}")
    
    print("\n" + "=" * 50)
    print("[OK] User data cleanup completed!")
    print("=" * 50)
    
    return True
