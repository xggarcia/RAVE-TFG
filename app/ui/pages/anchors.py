"""Phase anchors page — generate latent anchors + PCA scatter preview."""
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

_ANCHOR_STYLES = [
    ("#a8e63d", "intro"),
    ("#40d0d8", "verse"),
    ("#d840b8", "chorus"),
    ("#e8c040", "bridge"),
    ("#e0406a", "outro"),
]



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

            self._empty_txt = pg.TextItem(
                text="Generate anchors to see the latent space map",
                color=FG3, anchor=(0.5, 0.5),
            )
            self._empty_txt.setFont(pg.QtGui.QFont("JetBrains Mono", 9))
            self._empty_txt.setPos(0, 0)
            self._plot.addItem(self._empty_txt)

            layout.addWidget(self._plot)
        else:
            self._empty_txt = None
            fb = _lbl("pyqtgraph not installed — install it to see the scatter plot.",
                       size=11, color=FG3)
            fb.setAlignment(Qt.AlignCenter)
            fb.setContentsMargins(14, 60, 14, 60)
            layout.addWidget(fb)

        caption = _lbl("PCA projection · latent dim → 2D", size=10, color=FG3, mono=True)
        caption.setAlignment(Qt.AlignCenter)
        caption.setContentsMargins(0, 6, 0, 10)
        layout.addWidget(caption)

    def update_from_pca(self, pca_data: list[dict]):
        """Refresh plot with real PCA-projected latent data.

        pca_data: list of {"label": str, "points": [[x,y],...], "anchor_xy": [x,y]}
        """
        if not _HAS_PG or not pca_data:
            return
        self._plot.clear()
        self._empty_txt = None
        colors = [s[0] for s in _ANCHOR_STYLES]
        for i, phase in enumerate(pca_data):
            color = colors[i % len(colors)]
            pts = phase.get("points", [])
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                sc = pg.ScatterPlotItem(
                    x=xs, y=ys, size=3,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(color + "55"),
                )
                self._plot.addItem(sc)
            ax, ay = phase.get("anchor_xy", [0.0, 0.0])
            dot = pg.ScatterPlotItem(
                x=[ax], y=[ay], size=13,
                pen=pg.mkPen(color, width=1.5),
                brush=pg.mkBrush(color),
            )
            self._plot.addItem(dot)
            txt = pg.TextItem(text=phase.get("label", f"phase{i}"),
                              color=color, anchor=(0, 1))
            txt.setFont(pg.QtGui.QFont("JetBrains Mono", 8))
            txt.setPos(ax + 0.02, ay + 0.02)
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

        self._out = FileInput(placeholder="~/anchors/", directory=True)
        body.addWidget(Field("Output folder", hint="Files named after the model automatically.", inline=True).add(self._out))

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
        out_dir = self._out.path or None
        order_text = self._order_edit.text().strip()
        phase_labels = [p.strip() for p in order_text.split(",") if p.strip()] or None
        self._worker = DatasetWorker(generate_anchors_only, model, base,
                                     phase_labels=phase_labels, output_dir=out_dir)
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    @Slot(bool, str)
    def _on_done(self, ok: bool, msg: str):
        self._prog.finish(ok, msg)
        self._gen_btn.setEnabled(True)
        self._worker = None
        if ok:
            import json, os
            model_stem = os.path.splitext(os.path.basename(self._model.path))[0]
            out_dir = self._out.path or os.path.dirname(self._model.path)
            bundle_path = os.path.join(out_dir, f"{model_stem}_phases.json")
            legacy_pca_path = os.path.join(out_dir, f"{model_stem}_phases_pca.json")
            pca_data = []

            try:
                if os.path.exists(bundle_path):
                    with open(bundle_path, encoding="utf-8") as f:
                        bundle = json.load(f)
                    if isinstance(bundle, dict):
                        pca_data = bundle.get("phase_pca", [])

                # Backward compatibility: old split files
                if not pca_data and os.path.exists(legacy_pca_path):
                    with open(legacy_pca_path, encoding="utf-8") as f:
                        pca_data = json.load(f)

                self._scatter.update_from_pca(pca_data)
            except Exception as exc:
                self._prog.append_log(f"[WARN] Could not load phase map preview: {exc}")
