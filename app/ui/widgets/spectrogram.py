"""Rolling waterfall spectrogram widget using pyqtgraph ImageItem."""
import numpy as np

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from app.ui.widgets.form import BG0, FG3, MONO

_N_COLS  = 120   # history columns (time axis)
_N_ROWS  = 48    # frequency bins (0 → ~Nyquist/3, where most energy lives)
_DB_MIN  = -70.0 # noise floor
_DB_MAX  =   0.0 # 0 dBFS ceiling (spectrum normalized to window sum)
_HEIGHT  = 38    # fixed pixel height


def _make_colormap():
    stops = np.array([0.0, 0.4, 0.75, 1.0])
    colors = np.array([
        [24,  30,  26, 255],   # near-black  (silence)
        [20,  70,  40, 255],   # dark green  (low energy)
        [100, 210, 50, 255],   # acid green  (mid energy)
        [220, 180, 40, 255],   # amber       (peaks only)
    ], dtype=np.ubyte)
    return pg.ColorMap(stops, colors)


class SpectrogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(_HEIGHT)
        self._buf = np.full((_N_ROWS, _N_COLS), _DB_MIN, dtype=np.float32)

        if not _HAS_PG:
            lbl = QLabel("pyqtgraph missing")
            lbl.setStyleSheet(f"color:{FG3}; font-size:10px; {MONO} background:transparent;")
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(lbl)
            return

        pg.setConfigOptions(antialias=False, useOpenGL=True)

        self._plot = pg.PlotWidget(background=BG0)
        self._plot.setFixedHeight(_HEIGHT)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self._plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)

        self._img = pg.ImageItem()
        self._img.setLookupTable(_make_colormap().getLookupTable(nPts=256))
        self._img.setLevels([_DB_MIN, _DB_MAX])
        self._plot.addItem(self._img)
        self._img.setImage(self._buf.T, autoLevels=False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._plot)

    def push(self, audio_chunk: np.ndarray):
        if not _HAS_PG:
            return
        n = len(audio_chunk)
        window = np.hanning(n)
        win_sum = window.sum() or 1.0          # normalise → 0 dBFS = clipping
        spectrum = np.abs(np.fft.rfft(audio_chunk * window)) / win_sum
        spectrum = spectrum[:_N_ROWS]
        col = 20.0 * np.log10(spectrum.clip(1e-9))
        col = col.clip(_DB_MIN, _DB_MAX).astype(np.float32)
        self._buf = np.roll(self._buf, -1, axis=1)
        self._buf[:, -1] = col
        self._img.setImage(self._buf.T, autoLevels=False)
