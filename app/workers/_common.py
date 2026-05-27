import io


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
