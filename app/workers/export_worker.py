import sys
import io
from PySide6.QtCore import QThread, Signal


class _Tee(io.TextIOBase):
    def __init__(self, original, callback):
        self._orig = original
        self._cb = callback

    def write(self, text):
        if text.strip():
            self._cb(text.rstrip())
        return self._orig.write(text)

    def flush(self):
        self._orig.flush()


class ExportWorker(QThread):
    log      = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, run_path: str, output_name: str = "",
                 destination: str = "", streaming: bool = True):
        super().__init__()
        self._run_path    = run_path
        self._output_name = output_name
        self._destination = destination or "models/user_model/exported_model"
        self._streaming   = streaming

    def run(self):
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, self.log.emit)
        sys.stderr = _Tee(old_err, self.log.emit)
        try:
            from src.core.export import ExportModel
            out = ExportModel(
                run_path=self._run_path or None,
                output_path=self._destination,
                streaming=self._streaming,
            )
            self.finished.emit(True, f"Done → {out}")
        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            sys.stdout, sys.stderr = old_out, old_err
