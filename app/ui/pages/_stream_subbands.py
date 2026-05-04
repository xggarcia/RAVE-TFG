"""Subband control widget — envelope painter.

Bands have heights proportional to their sonic weight (top = strongest).
Painted values modulate the latent column-by-column. There is no playhead —
the engine just reads the column under its own internal cursor.
"""
from PySide6.QtCore import Qt, QRect, QRectF, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QLinearGradient
from PySide6.QtWidgets import QWidget

import numpy as np
import torch

from app.ui.widgets.form import ACID, AMBER, BG0, BG1, LINE0


# Per-band visual weights — match the row_weights used in the engine
# projector so the UI hierarchy reflects actual sonic impact.
_BAND_WEIGHTS = (1.0, 0.78, 0.58, 0.42, 0.30)
_BAND_COLORS = ("#e0406a", "#d840b8", ACID, "#40d0d8", AMBER)
_LABEL_W = 56  # px reserved on the left for band labels


class SubbandDrawWidget(QWidget):
    """Timeline envelope painter. ``num_subbands`` × ``n_timesteps``.

    Signals:
        subbandPatternChanged(torch.Tensor): emits shape (num_subbands, n_timesteps),
            values are continuous floats in [0.0, 1.0].
    """

    subbandPatternChanged = Signal(torch.Tensor)

    def __init__(self, num_subbands: int = 5, n_timesteps: int = 96, parent=None):
        super().__init__(parent)
        self.num_subbands = num_subbands
        self.n_timesteps = n_timesteps
        self.padding = 8

        self.grid = np.zeros((num_subbands, n_timesteps), dtype=np.float32)

        self.colors = list(_BAND_COLORS[:num_subbands])
        self.band_weights = list(_BAND_WEIGHTS[:num_subbands])

        self.setMinimumSize(_LABEL_W + 8 * self.n_timesteps + self.padding * 2, 160)
        self.setMouseTracking(True)

        self._mouse_pressed = False
        self._stroke_value = 1.0
        self._last_cell: tuple[int, int] | None = None
        self._playhead_col = 0

    # ---------- Public API ----------

    def set_pattern(self, pattern: torch.Tensor | np.ndarray):
        if isinstance(pattern, torch.Tensor):
            pattern = pattern.cpu().numpy()
        if pattern.shape == (self.num_subbands, self.n_timesteps):
            self.grid = np.asarray(pattern, dtype=np.float32).clip(0.0, 1.0)
            self.update()

    def clear(self):
        self.grid.fill(0.0)
        self.update()
        self._emit_pattern()

    def invert(self):
        self.grid = 1.0 - self.grid
        self.update()
        self._emit_pattern()

    # ---------- Mouse / painting ----------

    def mousePressEvent(self, event):
        self._mouse_pressed = True
        self._stroke_value = 0.0 if (event.buttons() & Qt.RightButton) else 1.0
        self._last_cell = None
        self._paint_at(event.position(), self._stroke_value)

    def mouseMoveEvent(self, event):
        if self._mouse_pressed:
            self._paint_at(event.position(), self._stroke_value)

    def mouseReleaseEvent(self, event):
        self._mouse_pressed = False
        self._last_cell = None

    def _paint_at(self, pos, value: float):
        col, row = self._pos_to_grid(pos)
        if 0 <= row < self.num_subbands and 0 <= col < self.n_timesteps:
            if self._last_cell is None:
                self._set_cell(row, col, value)
            else:
                self._paint_line(self._last_cell, (row, col), value)
            self._last_cell = (row, col)
            self.update()
            self._emit_pattern()

    def _set_cell(self, row: int, col: int, value: float):
        self.grid[row, col] = max(0.0, min(1.0, float(value)))

    def _paint_line(self, start: tuple[int, int], end: tuple[int, int], value: float):
        sr, sc = start
        er, ec = end
        dr = er - sr
        dc = ec - sc
        steps = max(abs(dr), abs(dc))
        if steps <= 0:
            self._set_cell(er, ec, value)
            return
        for i in range(steps + 1):
            t = i / float(steps)
            row = int(round(sr + dr * t))
            col = int(round(sc + dc * t))
            if 0 <= row < self.num_subbands and 0 <= col < self.n_timesteps:
                self._set_cell(row, col, value)

    def _pos_to_grid(self, pos) -> tuple[int, int]:
        x = pos.x() - self.padding - _LABEL_W
        y = pos.y() - self.padding
        grid_w = max(1, self.width() - self.padding * 2 - _LABEL_W)
        col_w = grid_w / max(1, self.n_timesteps)
        col = int(x / col_w) if col_w > 0 else -1

        row = self._row_from_y(y)
        return col, row

    def _row_from_y(self, y: float) -> int:
        avail_h = max(1, self.height() - self.padding * 2)
        weight_sum = sum(self.band_weights) or 1.0
        cursor = 0.0
        for i, w in enumerate(self.band_weights):
            bh = avail_h * w / weight_sum
            if y < cursor + bh:
                return i
            cursor += bh
        return self.num_subbands - 1

    def _emit_pattern(self):
        self.subbandPatternChanged.emit(torch.from_numpy(self.grid.copy()).float())

    def set_playhead_column(self, column: int):
        if self.n_timesteps <= 0:
            self._playhead_col = 0
        else:
            self._playhead_col = int(column) % self.n_timesteps
        self.update()

    # ---------- Render ----------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        full = self.rect()
        p.fillRect(full, QColor(BG0))

        avail_h = max(1, full.height() - self.padding * 2)
        weight_sum = sum(self.band_weights) or 1.0
        # Pixel band heights (last absorbs rounding).
        band_heights: list[int] = []
        acc = 0
        for i, w in enumerate(self.band_weights):
            if i == self.num_subbands - 1:
                band_heights.append(avail_h - acc)
            else:
                bh = int(round(avail_h * w / weight_sum))
                band_heights.append(bh)
                acc += bh

        grid_x0 = self.padding + _LABEL_W
        grid_w = max(1, full.width() - self.padding * 2 - _LABEL_W)
        col_w = grid_w / max(1, self.n_timesteps)

        # ----- Bands -----
        y_cursor = self.padding
        for row in range(self.num_subbands):
            bh = band_heights[row]
            band_top = y_cursor
            band_bot = y_cursor + bh

            # Band background (slightly darker for lower bands to reinforce hierarchy).
            shade_t = row / max(1, self.num_subbands - 1)
            band_bg = QColor(BG1)
            band_bg.setAlphaF(1.0 - 0.18 * shade_t)
            p.fillRect(QRect(grid_x0, band_top, grid_w, bh), band_bg)

            base_color = QColor(self.colors[row])

            # Clip per-column drawing to the band's grid rect so the last
            # column's right edge can't bleed past grid_x0 + grid_w.
            p.save()
            p.setClipRect(QRectF(grid_x0, band_top, grid_w, bh))

            # Per-column fill — height proportional to painted value.
            for col in range(self.n_timesteps):
                v = float(self.grid[row, col])
                if v <= 0.001:
                    continue

                x = grid_x0 + col * col_w
                xw = max(1.0, col_w + 0.5)
                fill_h = max(1.0, v * (bh - 2))
                fill_y = band_bot - fill_h

                grad = QLinearGradient(QPointF(0, fill_y), QPointF(0, band_bot))
                top_c = QColor(base_color).lighter(135)
                top_c.setAlphaF(0.95)
                bot_c = QColor(base_color)
                bot_c.setAlphaF(0.55)
                grad.setColorAt(0.0, top_c)
                grad.setColorAt(1.0, bot_c)
                p.fillRect(QRectF(x, fill_y, xw, fill_h), grad)

                edge = QColor(base_color).lighter(150)
                p.setPen(QPen(edge, 1.2))
                p.drawLine(QPointF(x, fill_y), QPointF(x + xw, fill_y))

            p.restore()

            # Band separator
            p.setPen(QPen(QColor(LINE0), 0.5))
            p.drawLine(grid_x0, band_bot, grid_x0 + grid_w, band_bot)

            # Label
            label_color = QColor(self.colors[row])
            label_color.setAlphaF(0.9)
            p.setPen(label_color)
            p.drawText(
                QRect(self.padding, band_top, _LABEL_W - 6, bh),
                Qt.AlignVCenter | Qt.AlignRight,
                f"BAND {row + 1}",
            )

            y_cursor = band_bot

        if self.n_timesteps > 0:
            play_x = grid_x0 + self._playhead_col * col_w
            p.setPen(QPen(QColor("#ffffff"), 1.8))
            p.drawLine(QPointF(play_x, self.padding), QPointF(play_x, self.padding + avail_h))

        # ----- Outer frame -----
        frame_pen = QPen(QColor(LINE0), 1)
        p.setPen(frame_pen)
        p.drawLine(grid_x0, self.padding, grid_x0, self.padding + avail_h)
        p.drawLine(int(grid_x0 + grid_w) - 1, self.padding,
                   int(grid_x0 + grid_w) - 1, self.padding + avail_h)

        p.end()
