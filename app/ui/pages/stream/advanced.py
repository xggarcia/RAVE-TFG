from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from app.ui.widgets.form import ACID, BG0, FG2, FG3, LINE0, Panel, _lbl, section_title
from app.ui.widgets.latent_radar import LatentRadarWidget

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

_BTN_ON  = (
    f"color:#000; background:{ACID}; border:1px solid {ACID}; "
    f"border-radius:4px; font-size:10px; padding:2px 7px;"
)
_BTN_OFF = (
    f"color:{FG3}; background:transparent; border:1px solid {LINE0}; "
    f"border-radius:4px; font-size:10px; padding:2px 7px;"
)
_PHASE_COLORS = ["#a8e63d", "#40d0d8", "#d840b8", "#e8c040", "#e0406a"]


class _LatentMapWidget(QWidget):
    """2D PCA scatter map of the latent space. Click to steer generation."""

    pointClicked = Signal(float, float)  # PCA-space x, y

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pca_data: list = []
        self._cursor_item = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        if _HAS_PG:
            self._plot = pg.PlotWidget(background=BG0)
            self._plot.setFixedHeight(200)
            self._plot.hideAxis("bottom")
            self._plot.hideAxis("left")
            self._plot.setMouseEnabled(x=False, y=False)
            self._plot.setMenuEnabled(False)
            self._plot.scene().sigMouseClicked.connect(self._on_scene_click)
            lay.addWidget(self._plot)
        else:
            fb = _lbl("Install pyqtgraph to see the latent map", 10, FG3)
            fb.setAlignment(Qt.AlignCenter)
            lay.addWidget(fb)

    def load(self, pca_data: list):
        if not _HAS_PG:
            return
        self._pca_data = pca_data
        self._cursor_item = None
        self._plot.clear()
        for i, phase in enumerate(pca_data):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            pts = phase.get("points", [])
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                sc = pg.ScatterPlotItem(x=xs, y=ys, size=3,
                    pen=pg.mkPen(None), brush=pg.mkBrush(color + "55"))
                self._plot.addItem(sc)
            ax, ay = phase.get("anchor_xy", [0.0, 0.0])
            dot = pg.ScatterPlotItem(x=[ax], y=[ay], size=12,
                pen=pg.mkPen(color, width=1.5), brush=pg.mkBrush(color))
            self._plot.addItem(dot)
            txt = pg.TextItem(text=phase.get("label", f"p{i}"), color=color, anchor=(0, 1))
            txt.setFont(pg.QtGui.QFont("JetBrains Mono", 8))
            txt.setPos(ax + 0.01, ay + 0.01)
            self._plot.addItem(txt)

    def clear(self):
        self._pca_data = []
        self._cursor_item = None
        if _HAS_PG:
            self._plot.clear()

    def _on_scene_click(self, event):
        if not self._pca_data:
            return
        pos = event.scenePos()
        vb = self._plot.plotItem.vb
        pt = vb.mapSceneToView(pos)
        x, y = float(pt.x()), float(pt.y())
        if self._cursor_item is not None:
            self._plot.removeItem(self._cursor_item)
        self._cursor_item = pg.ScatterPlotItem(
            x=[x], y=[y], size=14,
            pen=pg.mkPen("#ffffff", width=2),
            brush=pg.mkBrush("#ffffff44"),
        )
        self._plot.addItem(self._cursor_item)
        self.pointClicked.emit(x, y)


class AdvancedSlotPanel(Panel):
    """Per-slot advanced controls: latent map and latent per-dim scale."""

    latentDimChanged = Signal(int, int, float)         # slot_idx, dim, scale
    phaseMapClicked  = Signal(int, dict)               # slot_idx, {mean_z, std_z}
    _MAX_POINTS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slot_idx  = -1
        self._pca_data:     list = []
        self._anchors_data: list = []

        self._header_lbl = section_title("Advanced")
        self.add_header(self._header_lbl)

        self._body_w = self.add_body()
        body = self._body_w.layout()
        body.setContentsMargins(14, 10, 14, 14)
        body.setSpacing(10)
        body.setAlignment(Qt.AlignHCenter)

        # ── placeholder ──────────────────────────────────────────────────────
        self._placeholder = _lbl("Select a slot to edit", 11, FG3)
        self._placeholder.setAlignment(Qt.AlignHCenter)
        body.addWidget(self._placeholder, alignment=Qt.AlignHCenter)

        # ── latent section (shown when streaming + model loaded) ─────────────
        self._latent_w = QWidget()
        self._latent_w.setStyleSheet("background:transparent;")
        ll = QVBoxLayout(self._latent_w)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        ll.setAlignment(Qt.AlignHCenter)

        self._dim_lbl = _lbl("Dim 0 — scale 1.00", 10, FG2, mono=True)
        ll.addWidget(self._dim_lbl)

        self._radar = LatentRadarWidget()
        self._radar.dimSelected.connect(self._on_dim_selected)
        self._radar.scaleChanged.connect(self._on_scale_changed)
        ll.addWidget(self._radar)

        body.addWidget(self._latent_w, alignment=Qt.AlignHCenter)
        self._latent_w.setVisible(False)

        # ── latent map section (shown when anchor file is loaded) ────────────
        self._map_section = QWidget()
        self._map_section.setStyleSheet("background:transparent;")
        ml = QVBoxLayout(self._map_section)
        ml.setContentsMargins(0, 6, 0, 0)
        ml.setSpacing(4)
        ml.setAlignment(Qt.AlignHCenter)
        ml.addWidget(section_title("Latent map · click to steer"))
        self._map_widget = _LatentMapWidget()
        self._map_widget.pointClicked.connect(self._on_map_point)
        ml.addWidget(self._map_widget)

        body.addWidget(self._map_section, alignment=Qt.AlignHCenter)
        self._map_section.setVisible(False)

        # Start disabled until a slot is selected
        self._body_w.setEnabled(False)

    # ── public API ────────────────────────────────────────────────────────────

    def set_slot(self, slot_idx: int, name: str):
        self._slot_idx = slot_idx
        self._header_lbl.setText(f"Slot {name} — Advanced")
        self._body_w.setEnabled(True)
        self._placeholder.setVisible(False)
        if len(self._radar.get_scales()) <= 1:
            self._radar.set_dims(8)
        self._latent_w.setVisible(True)
        self._update_dim_lbl()

    def set_slot_unloaded(self, slot_idx: int, name: str):
        self._slot_idx = slot_idx
        self._header_lbl.setText(f"Slot {name} — Advanced")
        self._body_w.setEnabled(False)
        self._placeholder.setText("Load and start a model to enable advanced controls")
        self._placeholder.setVisible(True)
        self._latent_w.setVisible(False)

    def set_latent_size(self, n_dims: int):
        if n_dims < 1:
            return
        n_dims = min(self._MAX_POINTS, n_dims)
        old_scales = self._radar.get_scales()
        old_active = self._radar.active_dim

        scales = [old_scales[i] if i < len(old_scales) else 1.0 for i in range(n_dims)]
        self._radar.set_scales(scales, active_dim=min(old_active, n_dims - 1))
        self._placeholder.setVisible(False)
        self._latent_w.setVisible(True)
        self._update_dim_lbl()

    def load_state(self, state: dict):
        scales = [float(v) for v in state.get("scales", [1.0] * 8)]
        active_dim = int(state.get("active_dim", 0))

        n_dims = max(1, min(self._MAX_POINTS, len(scales)))
        self._radar.set_scales(scales[:n_dims], active_dim=active_dim)

        self._placeholder.setVisible(False)
        self._latent_w.setVisible(True)
        self._update_dim_lbl()

    def dump_state(self) -> dict:
        return {
            "scales": self._radar.get_scales(),
            "active_dim": int(self._radar.active_dim),
        }

    def load_phase_map(self, pca_data: list, anchors_data: list):
        """Show the 2D latent map; pca_data from phase bundle or legacy split files."""
        self._pca_data = pca_data
        # Align anchors_data to pca_data order by label
        label_to_anchor = {a["label"]: a for a in anchors_data}
        self._anchors_data = [label_to_anchor.get(p["label"], {}) for p in pca_data]
        self._map_widget.load(pca_data)
        self._map_section.setVisible(True)

    def clear_phase_map(self):
        self._pca_data = []
        self._anchors_data = []
        self._map_widget.clear()
        self._map_section.setVisible(False)

    def clear(self):
        self._slot_idx = -1
        self._header_lbl.setText("Advanced")
        self._body_w.setEnabled(False)
        self._placeholder.setText("Select a slot to edit")
        self._placeholder.setVisible(True)
        self._latent_w.setVisible(False)

    # ── internal slots ────────────────────────────────────────────────────────

    def _on_dim_selected(self, dim: int):
        self._update_dim_lbl()

    def _on_scale_changed(self, dim: int, scale: float):
        self._update_dim_lbl()
        if self._slot_idx >= 0:
            self.latentDimChanged.emit(self._slot_idx, dim, scale)

    def _update_dim_lbl(self):
        dim    = self._radar.active_dim
        scales = self._radar.get_scales()
        scale  = scales[dim] if dim < len(scales) else 1.0
        self._dim_lbl.setText(f"Dim {dim} — scale {scale:.2f}")

    def _on_map_point(self, x: float, y: float):
        if self._slot_idx < 0 or not self._anchors_data or not self._pca_data:
            return
        anchor_xys = [p.get("anchor_xy", [0.0, 0.0]) for p in self._pca_data]
        # Inverse-square weighting so position dominates the blend.
        # 1/d is too flat with 2–4 anchors: distance ratios stay close to 1
        # and the blend collapses toward the centroid regardless of click.
        sq_dists = [(x - ax) ** 2 + (y - ay) ** 2 + 1e-6 for ax, ay in anchor_xys]
        weights = [1.0 / d2 for d2 in sq_dists]
        total = sum(weights)
        weights = [w / total for w in weights]

        # Determine dimensionality from first valid anchor
        valid = [a for a in self._anchors_data if a.get("mean_z")]
        if not valid:
            return
        n_dims = len(valid[0]["mean_z"])
        has_std = bool(valid[0].get("std_z"))
        blended_mean = [0.0] * n_dims
        # Start std at 0 so weights sum to anchor std exactly.
        # The previous 1.0 init was an additive offset on top of the blend,
        # which dwarfed inter-anchor std differences and flattened the response.
        blended_std  = [0.0] * n_dims if has_std else [1.0] * n_dims
        for w, anchor in zip(weights, self._anchors_data):
            mean_z = anchor.get("mean_z", [])
            std_z  = anchor.get("std_z",  [])
            for j in range(min(n_dims, len(mean_z))):
                blended_mean[j] += w * mean_z[j]
            if has_std:
                for j in range(min(n_dims, len(std_z))):
                    blended_std[j] += w * std_z[j]

        self.phaseMapClicked.emit(self._slot_idx, {
            "mean_z": blended_mean,
            "std_z":  blended_std,
        })
