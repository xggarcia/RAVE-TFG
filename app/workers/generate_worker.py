import io
import os
import sys

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


class GenerateWorker(QThread):
    log = Signal(str)
    finished = Signal(bool, str, str)  # success, message, output_wav

    def __init__(
        self,
        model_path: str,
        audio_path: str,
        output_name: str,
        duration: int,
        random_mode: bool = True,
    ):
        super().__init__()
        self._model_path = model_path
        self._audio_path = audio_path
        self._output_name = output_name
        self._duration = duration
        self._random_mode = random_mode

    def run(self):
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, self.log.emit)
        sys.stderr = _Tee(old_err, self.log.emit)
        try:
            from src.core.generate import UseModel

            out = UseModel(
                model_path=self._model_path,
                audio_path=self._audio_path,
                random=self._random_mode,
                output_name=self._output_name,
                duration=self._duration,
            )
            self.finished.emit(True, f"Render complete: {os.path.basename(out)}", out)
        except Exception as exc:
            self.finished.emit(False, str(exc), "")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
