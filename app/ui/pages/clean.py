from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame, QPushButton,
    QLabel, QCheckBox, QLineEdit, QDialog, QDialogButtonBox,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    PageHeader, Panel, section_title, _lbl,
    BG0, BG1, BG2, BG3, FG0, FG1, FG2, FG3, LINE0, LINE1, MAG, ACID, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.clean_worker import CleanWorker, scan_buckets, fmt_size

MAG_BG  = "#3d1828"
MAG_DIM = "#8c2040"


class CleanPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: CleanWorker | None = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._build()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_buckets()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background:{BG0};")

        root.addWidget(PageHeader(
            crumbs=["Maintenance", "Clean User Data"],
            title="Clean user data",
            desc="Delete workspace data. Does not touch the demo models bundled with RAVE-TFG.",
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        content.setStyleSheet(f"background:{BG0};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(16)

        self._panel = Panel()
        self._panel_header_lbl = section_title("What will be removed")
        self._total_lbl = _lbl("0 B", size=10, color=MAG, mono=True)

        self._panel.add_header(self._panel_header_lbl, self._total_lbl)

        self._buckets_area = QWidget()
        self._buckets_area.setStyleSheet("background:transparent;")
        self._buckets_layout = QVBoxLayout(self._buckets_area)
        self._buckets_layout.setContentsMargins(0, 0, 0, 0)
        self._buckets_layout.setSpacing(0)
        self._panel._root.addWidget(self._buckets_area)

        # Footer with warning + button
        footer = QWidget()
        footer.setStyleSheet(f"background:{BG1}; border-top:1px solid {LINE0};")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(16, 12, 16, 12)
        fl.setSpacing(10)
        warn_lbl = QLabel("⚠  This operation is irreversible. Checkpoints cannot be recovered once deleted.")
        warn_lbl.setStyleSheet(f"color:{FG1}; font-size:12px; background:transparent;")
        warn_lbl.setWordWrap(True)
        fl.addWidget(warn_lbl, 1)
        self._clean_btn = QPushButton("🧹  Clean selected")
        self._clean_btn.setProperty("role", "danger")
        self._clean_btn.setFixedHeight(30)
        self._clean_btn.clicked.connect(self._confirm_clean)
        fl.addWidget(self._clean_btn)
        self._panel._root.addWidget(footer)

        cl.addWidget(self._panel)

        self._progress = ProgressPanel()
        cl.addWidget(self._progress)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _refresh_buckets(self):
        # Clear old rows
        while self._buckets_layout.count():
            item = self._buckets_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checkboxes.clear()

        buckets = scan_buckets()
        total = sum(b["size"] for b in buckets)
        self._total_lbl.setText(fmt_size(total))

        if not buckets:
            empty = _lbl("Nothing to clean.", size=12, color=FG3)
            empty.setContentsMargins(18, 14, 18, 14)
            self._buckets_layout.addWidget(empty)
            self._clean_btn.setEnabled(False)
            return

        self._clean_btn.setEnabled(True)
        for i, b in enumerate(buckets):
            row = _BucketRow(b, last=(i == len(buckets) - 1))
            self._checkboxes[b["path"]] = row.checkbox
            self._buckets_layout.addWidget(row)

    def _confirm_clean(self):
        selected = [path for path, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            return
        dlg = _ConfirmDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self._run_clean(selected)

    def _run_clean(self, paths: list[str]):
        if self._worker and self._worker.isRunning():
            return
        self._clean_btn.setEnabled(False)
        self._progress.start("Deleting selected data…")
        self._worker = CleanWorker(paths)
        self._worker.log.connect(self._progress.append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str):
        self._progress.finish(success, message)
        self._clean_btn.setEnabled(True)
        self._refresh_buckets()


class _BucketRow(QWidget):
    def __init__(self, bucket: dict, last: bool, parent=None):
        super().__init__(parent)
        self._last = last
        hl = QHBoxLayout(self)
        hl.setContentsMargins(18, 12, 18, 12)
        hl.setSpacing(14)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        self.checkbox.setStyleSheet(
            f"QCheckBox::indicator {{ width:14px; height:14px; border:1px solid #3f4b42; border-radius:2px; background:#232926; }}"
            f"QCheckBox::indicator:checked {{ background:#a8e63d; border-color:#a8e63d; }}"
        )
        hl.addWidget(self.checkbox)

        info = QVBoxLayout()
        info.setSpacing(2)
        info.addWidget(_lbl(bucket["label"], size=13, color=FG0, bold=True))
        info.addWidget(_lbl(f"{bucket['path']} · {bucket['count']} items", size=10, color=FG3, mono=True))
        hl.addLayout(info, 1)

        hl.addWidget(_lbl(fmt_size(bucket["size"]), size=12, color=FG0, mono=True))

    def paintEvent(self, event):
        p = QPainter(self)
        if not self._last:
            p.setPen(QPen(QColor(LINE0), 1))
            p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


class _ConfirmDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm deletion")
        self.setFixedWidth(440)
        self.setStyleSheet(
            f"QDialog {{ background:{BG2}; }}"
            f"QLabel {{ background:transparent; color:{FG0}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QWidget()
        header.setFixedHeight(60)
        header.setStyleSheet(f"background:{BG2}; border-bottom:1px solid {LINE0};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 14, 20, 14)
        hl.setSpacing(12)
        icon_lbl = QLabel("⚠")
        icon_lbl.setFixedSize(32, 32)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet(
            f"background:#3d1828; color:{MAG}; font-size:14px; border-radius:16px;"
        )
        hl.addWidget(icon_lbl)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_col.addWidget(_lbl("Delete workspace data?", size=14, color=FG0, bold=True))
        title_col.addWidget(_lbl("This cannot be undone.", size=11, color=FG2))
        hl.addLayout(title_col)
        layout.addWidget(header)

        # Body
        body = QWidget()
        body.setStyleSheet(f"background:{BG2};")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(20, 16, 20, 16)
        bl.setSpacing(12)
        bl.addWidget(_lbl(
            'Type DELETE below to confirm. Active training runs will be stopped first.',
            size=12, color=FG1, wrap=True,
        ))
        self._confirm_edit = QLineEdit()
        self._confirm_edit.setPlaceholderText("DELETE")
        self._confirm_edit.setStyleSheet(
            f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
            f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
        )
        bl.addWidget(self._confirm_edit)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self._delete_btn = QPushButton("🧹  Delete everything")
        self._delete_btn.setStyleSheet(
            f"background:{MAG}; color:{FG0}; border:1px solid {MAG}; border-radius:4px; padding:6px 12px; font-weight:600;"
        )
        self._delete_btn.clicked.connect(self._try_accept)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._delete_btn)
        bl.addLayout(btn_row)
        layout.addWidget(body)

    def _try_accept(self):
        if self._confirm_edit.text() == "DELETE":
            self.accept()
        else:
            self._confirm_edit.setStyleSheet(
                f"background:{BG1}; color:{MAG}; {MONO} font-size:12px;"
                f"border:1px solid {MAG}; border-radius:4px; padding:6px 10px;"
            )
