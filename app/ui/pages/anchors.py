"""Phase anchors page — generate latent anchors + PCA scatter preview."""
import math
import random
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QPushButton, QLabel, QLineEdit,
)
from PySide6.QtCore import Qt, Slot

from app.ui.widgets.form import (
    PageHeader, Panel, Field, FileInput, section_title, _lbl,
    BG0, BG1, BG2, FG0, FG1, FG2, FG3, LINE0, LINE1, ACID, AMBER, MAG, BLUE, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.dataset_worker import DatasetWorker

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

# Named anchor colours matching the mock hues
_ANCHOR_STYLES = [
    ("#a8e63d", "intro"),
    ("#40d0d8", "verse"),
    ("#d840b8", "chorus"),
    ("#e8c040", "bridge"),
    ("#e0406a", "outro"),
]


def _placeholder_scatter():
    """Return (x, y, color, size) for deterministic placeholder points."""
    rng = random.Random(42)
    centers = [(-1.2, 0.8), (1.1, 0.9), (1.2, -0.9), (-1.1, -0.9), (0.0, 0.0)]
    brushes, xs, ys = [], [], []
    for (cx, cy), (color, _) in zip(centers, _ANCHOR_STYLES):
        for _ in range(40):
            xs.append(cx + rng.gauss(0, 0.28))
            ys.append(cy + rng.gauss(0, 0.28))
            brushes.append(pg.mkBrush(color + "70"))
    return xs, ys, brushes


class _ScatterPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(38)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2f3832, stop:1 #28302b);"
            f"border-bottom:1px solid {LINE0};"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.addWidget(section_title("Preview · latent space"))
        layout.addWidget(header)

        if _HAS_PG:
            self._plot = pg.PlotWidget(background=BG0)
            self._plot.setMinimumHeight(280)
            self._plot.setMinimumWidth(340)
            self._plot.hideAxis("bottom")
            self._plot.hideAxis("left")
            self._plot.setMouseEnabled(x=False, y=False)
            self._plot.setMenuEnabled(False)

            # Placeholder scatter cloud
            xs, ys, brushes = _placeholder_scatter()
            scatter = pg.ScatterPlotItem(
                x=xs, y=ys,
                size=5, pen=pg.mkPen(None), brush=brushes,
            )
            self._plot.addItem(scatter)

            # Anchor dots + labels
            for (cx, cy), (color, label) in zip(
                [(-1.2, 0.8), (1.1, 0.9), (1.2, -0.9), (-1.1, -0.9), (0.0, 0.0)],
                _ANCHOR_STYLES,
            ):
                dot = pg.ScatterPlotItem(
                    x=[cx], y=[cy],
                    size=12,
                    pen=pg.mkPen(color, width=1),
                    brush=pg.mkBrush(color),
                )
                self._plot.addItem(dot)
                txt = pg.TextItem(text=label, color=color,
                                  anchor=(0, 1))
                txt.setFont(pg.QtGui.QFont("JetBrains Mono", 8))
                txt.setPos(cx + 0.06, cy + 0.06)
                self._plot.addItem(txt)

            layout.addWidget(self._plot)
        else:
            fb = _lbl("pyqtgraph not installed — install it to see the scatter plot.",
                       size=11, color=FG3)
            fb.setAlignment(Qt.AlignCenter)
            fb.setContentsMargins(14, 60, 14, 60)
            layout.addWidget(fb)

        caption = _lbl("PCA projection · latent dim → 2D", size=10, color=FG3, mono=True)
        caption.setAlignment(Qt.AlignCenter)
        caption.setContentsMargins(0, 6, 0, 10)
        layout.addWidget(caption)

    def update_from_anchors(self, anchors: list[dict]):
        """Refresh plot with real anchor data after generation."""
        if not _HAS_PG or not anchors:
            return
        self._plot.clear()
        colors = [s[0] for s in _ANCHOR_STYLES]
        rng = random.Random(99)
        for i, anchor in enumerate(anchors):
            color = colors[i % len(colors)]
            cx = anchor.get("pca_x", 0.0)
            cy = anchor.get("pca_y", 0.0)
            xs = [cx + rng.gauss(0, 0.25) for _ in range(30)]
            ys = [cy + rng.gauss(0, 0.25) for _ in range(30)]
            sc = pg.ScatterPlotItem(x=xs, y=ys, size=4,
                                    pen=pg.mkPen(None),
                                    brush=pg.mkBrush(color + "60"))
            self._plot.addItem(sc)
            dot = pg.ScatterPlotItem(x=[cx], y=[cy], size=12,
                                     pen=pg.mkPen(color, width=1),
                                     brush=pg.mkBrush(color))
            self._plot.addItem(dot)
            txt = pg.TextItem(text=anchor.get("label", f"phase{i}"),
                               color=color, anchor=(0, 1))
            txt.setPos(cx + 0.06, cy + 0.06)
            self._plot.addItem(txt)


class AnchorsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: DatasetWorker | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._gen_btn = QPushButton("▶  Generate anchors")
        self._gen_btn.setProperty("role", "primary")
        self._gen_btn.setFixedHeight(30)
        self._gen_btn.clicked.connect(self._start)

        root.addWidget(PageHeader(
            crumbs=["Data & Training", "Phase Anchors"],
            title="Generate phase anchors",
            desc="Compute latent anchors per phase from an existing model. "
                 "The Streaming GUI uses these as the four corners of the phase XY pad.",
            actions=[self._gen_btn],
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(16)

        cols = QHBoxLayout()
        cols.setSpacing(20)
        cols.addWidget(self._build_params(), 1)

        scatter_panel = Panel()
        scatter_panel.setFixedWidth(380)
        self._scatter = _ScatterPanel()
        scatter_panel._root.addWidget(self._scatter)
        cols.addWidget(scatter_panel)

        cl.addLayout(cols)

        self._prog = ProgressPanel()
        cl.addWidget(self._prog)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_params(self) -> QWidget:
        panel = Panel()
        panel.add_header(section_title("Parameters"))
        body = panel.body_layout()

        self._model = FileInput(placeholder="~/exports/model_streaming.ts", directory=False)
        body.addWidget(Field("RAVE model", inline=True).add(self._model))

        self._base = FileInput(placeholder="~/datasets/song-phases/")
        self._base.path_changed.connect(self._update_phase_order)
        body.addWidget(Field("Base folder", inline=True).add(self._base))

        self._order_edit = QLineEdit("—")
        self._order_edit.setStyleSheet(
            f"color:{FG0}; font-size:11px; {MONO} background:{BG1};"
            f"border:1px solid {LINE1}; border-radius:3px; padding:6px 10px;"
        )
        self._order_edit.setPlaceholderText("intro,verse,chorus,bridge")
        self._order_edit.setToolTip("Auto-filled from folder names. Edit to reorder (comma-separated).")
        body.addWidget(Field("Phase order", inline=True).add(self._order_edit))

        self._out = FileInput(placeholder="~/anchors/model_phases.json", directory=False, save=True)
        body.addWidget(Field("Output", inline=True).add(self._out))

        body.addStretch()
        return panel

    def _update_phase_order(self, base: str):
        import os
        if not os.path.isdir(base):
            return
        phases = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        self._order_edit.setText(",".join(phases) if phases else "")
        self._phase_names = phases

    def _start(self):
        if self._worker and self._worker.isRunning():
            return
        model = self._model.path
        base  = self._base.path
        if not model or not base:
            self._prog.start("Missing inputs")
            self._prog.finish(False, "Select a RAVE model and base folder first.")
            return

        self._gen_btn.setEnabled(False)
        self._prog.start("Generating phase anchors…")

        from src.phase_workflow import generate_anchors_only
        self._worker = DatasetWorker(generate_anchors_only, model, base)
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    @Slot(bool, str)
    def _on_done(self, ok: bool, msg: str):
        self._prog.finish(ok, msg)
        self._gen_btn.setEnabled(True)
        self._worker = None
