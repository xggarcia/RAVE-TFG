"""TrainWorker — runs `rave train` via QProcess and parses stdout live."""
import re
import sys
from PySide6.QtCore import QObject, QProcess, Signal

# Strip ANSI escape codes before parsing
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]|\r")

# Patterns found in PyTorch Lightning / RAVE stdout
_STEP_RE    = re.compile(r"step[=\s](\d+)", re.I)
_LOSS_RE    = re.compile(r"\bloss[=\s]([\d.]+)", re.I)
_ITS_RE     = re.compile(r"([\d.]+)\s*it/s|it/s[=\s]([\d.]+)", re.I)
_VAL_RE     = re.compile(r"v(?:al(?:id)?)?_?loss[=\s]([\d.]+)", re.I)
_EPOCH_RE   = re.compile(r"Epoch\s+(\d+)")

CUDA_OOM_MARKERS = ["OutOfMemoryError", "out of memory", "CUDA error"]


def _eta_str(remaining: int, it_s: float) -> str:
    if it_s <= 0 or remaining <= 0:
        return "—"
    secs = int(remaining / it_s)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class TrainWorker(QObject):
    progress = Signal(dict)       # {step, loss, it_s, val_loss, eta}
    log      = Signal(str, str)   # (level, message)  level: INFO|WARN|ERR
    failed   = Signal(str, str)   # (short_msg, traceback)
    finished = Signal(dict)       # {steps_trained}

    def __init__(self, cmd: list[str], max_steps: int = 0):
        super().__init__()
        self._cmd      = cmd
        self._max_steps = max_steps
        self._buf      = ""
        self._stderr_buf = ""
        self._state: dict = {"step": 0, "loss": 0.0, "it_s": 0.0, "val_loss": 0.0}
        self._oom_lines: list[str] = []
        self._collecting_oom = False

        self._proc = QProcess()
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)

    def start(self):
        exe = self._cmd[0]
        args = self._cmd[1:]
        self._proc.start(exe, args)
        if not self._proc.waitForStarted(3000):
            self.failed.emit("Process failed to start", f"Command: {' '.join(self._cmd)}")

    def stop(self):
        if self._proc.state() != QProcess.NotRunning:
            self._proc.terminate()

    def _on_output(self):
        raw = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buf += raw
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line)

    def _handle_line(self, raw_line: str):
        line = _ANSI_RE.sub("", raw_line).strip()
        if not line:
            return

        # OOM detection
        if any(m in line for m in CUDA_OOM_MARKERS):
            self._collecting_oom = True
        if self._collecting_oom:
            self._oom_lines.append(line)
            self.log.emit("ERR", line)
            return

        # Level heuristic
        level = "WARN" if any(w in line for w in ("warn", "WARN", "Warning")) else \
                "ERR"  if any(w in line for w in ("error", "Error", "ERROR", "failed", "Failed")) else \
                "INFO"
        self.log.emit(level, line)

        # Extract metrics
        step_m  = _STEP_RE.search(line)
        loss_m  = _LOSS_RE.search(line)
        its_m   = _ITS_RE.search(line)
        val_m   = _VAL_RE.search(line)

        updated = False
        if step_m:
            self._state["step"] = int(step_m.group(1)); updated = True
        if loss_m and not val_m:
            self._state["loss"] = float(loss_m.group(1)); updated = True
        if its_m:
            self._state["it_s"] = float(its_m.group(1) or its_m.group(2)); updated = True
        if val_m:
            self._state["val_loss"] = float(val_m.group(1)); updated = True

        if updated and self._state["step"] > 0:
            remaining = max(0, self._max_steps - self._state["step"])
            self._state["eta"] = _eta_str(remaining, self._state["it_s"])
            self.progress.emit(dict(self._state))

    def _on_finished(self, exit_code: int, exit_status):
        if self._collecting_oom and self._oom_lines:
            tb = "\n".join(self._oom_lines)
            self.failed.emit("CUDA out of memory", tb)
        elif exit_code != 0:
            self.failed.emit(f"Process exited with code {exit_code}", "")
        else:
            self.finished.emit({"steps_trained": self._state["step"]})
