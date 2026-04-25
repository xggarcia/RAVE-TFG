from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from app.ui.widgets.form import ACID, FG2, FG3, LINE0, Panel, _lbl, section_title
from app.ui.widgets.latent_radar import LatentRadarWidget

_BTN_ON  = (
    f"color:#000; background:{ACID}; border:1px solid {ACID}; "
    f"border-radius:4px; font-size:10px; padding:2px 7px;"
)
_BTN_OFF = (
    f"color:{FG3}; background:transparent; border:1px solid {LINE0}; "
    f"border-radius:4px; font-size:10px; padding:2px 7px;"
)


class AdvancedSlotPanel(Panel):
    """Per-slot advanced controls: use-prior toggle and latent per-dim scale."""

    latentDimChanged = Signal(int, int, float)         # slot_idx, dim, scale
    usePriorChanged  = Signal(int, bool)               # slot_idx, enabled
    _MAX_POINTS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slot_idx  = -1
        self._prior_on  = False

        self._header_lbl = section_title("Advanced")
        self.add_header(self._header_lbl)

        self._body_w = self.add_body()
        body = self._body_w.layout()
        body.setContentsMargins(14, 10, 14, 14)
        body.setSpacing(10)

        # ── global controls row ──────────────────────────────────────────────
        glob = QHBoxLayout()
        self._prior_btn = QPushButton("PRIOR OFF")
        self._prior_btn.setFixedHeight(26)
        self._prior_btn.setStyleSheet(_BTN_OFF)
        self._prior_btn.clicked.connect(self._toggle_prior)
        glob.addWidget(self._prior_btn)
        glob.addStretch()
        body.addLayout(glob)

        # ── placeholder ──────────────────────────────────────────────────────
        self._placeholder = _lbl("Select a slot to edit", 11, FG3)
        body.addWidget(self._placeholder)

        # ── latent section (shown when streaming + model loaded) ─────────────
        self._latent_w = QWidget()
        self._latent_w.setStyleSheet("background:transparent;")
        ll = QVBoxLayout(self._latent_w)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        self._dim_lbl = _lbl("Dim 0 — scale 1.00", 10, FG2, mono=True)
        ll.addWidget(self._dim_lbl)

        self._radar = LatentRadarWidget()
        self._radar.dimSelected.connect(self._on_dim_selected)
        self._radar.scaleChanged.connect(self._on_scale_changed)
        ll.addWidget(self._radar)

        body.addWidget(self._latent_w)
        self._latent_w.setVisible(False)

        # Start disabled until a slot is selected
        self._body_w.setEnabled(False)

    # ── public API ────────────────────────────────────────────────────────────

    def set_slot(self, slot_idx: int, name: str):
        self._slot_idx = slot_idx
        self._header_lbl.setText(f"Slot {name} — Advanced")
        self._body_w.setEnabled(True)
        self._placeholder.setVisible(False)
        # Show radar immediately with a default dim count; dims update on model load
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
        prior_on = bool(state.get("prior_on", False))
        scales = [float(v) for v in state.get("scales", [1.0] * 8)]
        active_dim = int(state.get("active_dim", 0))

        n_dims = max(1, min(self._MAX_POINTS, len(scales)))
        self._radar.set_scales(scales[:n_dims], active_dim=active_dim)

        self._prior_on = prior_on
        self._prior_btn.setText("PRIOR ON" if self._prior_on else "PRIOR OFF")
        self._prior_btn.setStyleSheet(_BTN_ON if self._prior_on else _BTN_OFF)

        self._placeholder.setVisible(False)
        self._latent_w.setVisible(True)
        self._update_dim_lbl()

    def dump_state(self) -> dict:
        return {
            "prior_on": bool(self._prior_on),
            "scales": self._radar.get_scales(),
            "active_dim": int(self._radar.active_dim),
        }

    def clear(self):
        self._slot_idx = -1
        self._header_lbl.setText("Advanced")
        self._body_w.setEnabled(False)
        self._placeholder.setText("Select a slot to edit")
        self._placeholder.setVisible(True)
        self._latent_w.setVisible(False)

    # ── internal slots ────────────────────────────────────────────────────────

    def _toggle_prior(self):
        self._prior_on = not self._prior_on
        self._prior_btn.setText("PRIOR ON" if self._prior_on else "PRIOR OFF")
        self._prior_btn.setStyleSheet(_BTN_ON if self._prior_on else _BTN_OFF)
        if self._slot_idx >= 0:
            self.usePriorChanged.emit(self._slot_idx, self._prior_on)

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
