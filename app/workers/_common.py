import io


class _Tee(io.TextIOBase):
    """Redirect writes to a Qt signal callback; suppress terminal output."""

    def __init__(self, original, callback):
        self._orig = original
        self._cb = callback

    def write(self, text):
        if text.strip():
            self._cb(text.rstrip())
        return len(text)

    def flush(self):
        pass
