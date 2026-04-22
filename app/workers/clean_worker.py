import os
import shutil
from PySide6.QtCore import QThread, Signal

# Mirrors the bucket definitions in src/clean.py
BUCKETS = [
    ("preprocessed_data",                    "Preprocessed LMDBs"),
    ("models/user_model/checkpoints",         "Training checkpoints"),
    ("models/user_model/exported_model",      "Exported models"),
    ("outputs",                              "Generated outputs"),
    ("descSounds",                           "Temp Freesound downloads"),
    ("input_data/user_data",                 "Downloaded audio"),
    ("database/database_download/user",      "User CSV files"),
]
_DELETE_ENTIRE = {"descSounds"}


def scan_buckets() -> list[dict]:
    """Return info about each bucket that has content."""
    result = []
    for path, label in BUCKETS:
        if not os.path.exists(path):
            continue
        if path in _DELETE_ENTIRE:
            size = _dir_size(path)
            count = _dir_count(path)
        else:
            contents = [f for f in os.listdir(path) if f != ".gitkeep"]
            if not contents:
                continue
            size = _dir_size(path)
            count = sum(1 for _, _, files in os.walk(path) for f in files if f != ".gitkeep")
        result.append({"path": path, "label": label, "size": size, "count": count})
    return result


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _dir_count(path: str) -> int:
    return sum(1 for _, _, files in os.walk(path) for _ in files)


def fmt_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


class CleanWorker(QThread):
    log      = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, paths: list[str]):
        super().__init__()
        self._paths = paths

    def run(self):
        errors = []
        for path in self._paths:
            try:
                if path in _DELETE_ENTIRE:
                    shutil.rmtree(path, ignore_errors=True)
                    self.log.emit(f"Deleted: {path}")
                else:
                    for item in os.listdir(path):
                        if item == ".gitkeep":
                            continue
                        item_path = os.path.join(path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    self.log.emit(f"Cleaned: {path}")
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                self.log.emit(f"Error cleaning {path}: {exc}")

        if errors:
            self.finished.emit(False, f"Completed with {len(errors)} error(s)")
        else:
            self.finished.emit(True, "All selected data deleted.")
