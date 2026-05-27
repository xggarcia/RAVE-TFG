"""Generic worker for dataset-creation backend calls."""
import sys
from PySide6.QtCore import QThread, Signal

from app.workers._common import _Tee


class DatasetWorker(QThread):
    log      = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, self.log.emit)
        sys.stderr = _Tee(old_err, self.log.emit)
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(True, f"Done → {result}" if result else "Done")
        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            sys.stdout, sys.stderr = old_out, old_err
