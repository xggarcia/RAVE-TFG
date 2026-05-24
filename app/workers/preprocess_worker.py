import sys
import io
from PySide6.QtCore import QThread, Signal


class _Tee(io.TextIOBase):
    """Redirect writes to a callback while also writing to the original stream."""
    def __init__(self, original, callback):
        self._orig = original
        self._cb = callback

    def write(self, text):
        if text.strip():
            self._cb(text.rstrip())
        return self._orig.write(text)

    def flush(self):
        self._orig.flush()


class PreprocessWorker(QThread):
    log      = Signal(str)           # one line per emit
    finished = Signal(bool, str)     # success, short message

    def __init__(self, audio_path: str, channels: int = 1,
                 lazy: bool = False, max_db_size: int = 8,
                 output_path: str | None = None):
        super().__init__()
        self._audio_path  = audio_path
        self._channels    = channels
        self._lazy        = lazy
        self._max_db_size = max_db_size
        self._output_path = output_path

    def run(self):
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, self.log.emit)
        sys.stderr = _Tee(old_err, self.log.emit)
        try:
            from src.core.preprocess import PreprocessDataset
            out = PreprocessDataset(
                audio_path=self._audio_path,
                channels=self._channels,
                lazy=self._lazy,
                max_db_size=self._max_db_size,
                output_path=self._output_path,
            )
            self.finished.emit(True, f"Done → {out}")
        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            sys.stdout, sys.stderr = old_out, old_err
