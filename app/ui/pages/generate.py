import os
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.form import (
    ACID,
    BG0,
    BG1,
    FG0,
    FG2,
    FG3,
    LINE0,
    LINE1,
    MONO,
    Field,
    FileInput,
    PageHeader,
    Panel,
    _lbl,
    section_title,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.ui.widgets.waveform import MiniWaveform
from app.workers.generate_worker import GenerateWorker


class GeneratePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: GenerateWorker | None = None
        self._output_path: str = ""
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background:{BG0};")

        self._render_btn = QPushButton("▶  Render")
        self._render_btn.setProperty("role", "primary")
        self._render_btn.setFixedHeight(30)
        self._render_btn.clicked.connect(self._start)

        root.addWidget(
            PageHeader(
                crumbs=["Generate & Stream", "Generate Audio"],
                title="Generate audio",
                desc="Render audio from a trained model and preview the output.",
                actions=[self._render_btn],
            )
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(20)

        cl.addWidget(self._build_params())

        self._progress = ProgressPanel()
        cl.addWidget(self._progress)

        self._success = self._build_success()
        self._success.hide()
        cl.addWidget(self._success)

        cl.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_params(self) -> QWidget:
        panel = Panel()
        panel.add_header(section_title("Render parameters"))
        body = panel.body_layout()

        self._model = FileInput(
            value="models/demo_model/demo_model.ts",
            placeholder="models/demo_model/demo_model.ts",
            directory=False,
        )
        self._audio = FileInput(
            value="input_data/demo_data/audio1.wav",
            placeholder="input_data/demo_data/audio1.wav",
            directory=False,
        )
        self._name = QLineEdit("generated_audio")
        self._name.setStyleSheet(
            f"background:{BG1}; color:{FG0}; {MONO} font-size:12px; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
        )
        self._duration = QLineEdit("30")
        self._duration.setStyleSheet(
            f"background:{BG1}; color:{FG0}; {MONO} font-size:12px; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
        )

        body.addWidget(Field("Model (.ts)", inline=True).add(self._model))
        body.addWidget(Field("Reference audio", inline=True).add(self._audio))
        body.addWidget(Field("Output name", inline=True).add(self._name))
        body.addWidget(Field("Duration (seconds)", inline=True).add(self._duration))
        body.addStretch()
        return panel

    def _build_success(self) -> QWidget:
        panel = Panel()
        self._success_msg = _lbl("", size=12, color=FG0)
        panel.add_header(section_title("Render complete"), self._success_msg)
        body = panel.body_layout()

        self._out_path_lbl = _lbl("", size=10, color=FG3, mono=True)
        body.addWidget(self._out_path_lbl)

        self._wave = MiniWaveform(seed=21, width=760, height=54)
        body.addWidget(self._wave)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._reveal_btn = QPushButton("Reveal")
        self._reveal_btn.clicked.connect(self._reveal)
        self._again_btn = QPushButton("Render another")
        self._again_btn.setProperty("role", "primary")
        self._again_btn.clicked.connect(lambda: self._success.hide())
        actions.addWidget(self._reveal_btn)
        actions.addWidget(self._again_btn)
        actions.addStretch()
        body.addLayout(actions)

        return panel

    def _start(self):
        if self._worker and self._worker.isRunning():
            return

        model = self._model.path
        audio = self._audio.path
        if not model or not audio:
            self._progress.start("Missing inputs")
            self._progress.finish(False, "Select a model and reference audio file.")
            return

        try:
            duration = int(self._duration.text())
        except ValueError:
            self._progress.start("Invalid duration")
            self._progress.finish(False, "Duration must be an integer number of seconds.")
            return

        self._render_btn.setEnabled(False)
        self._success.hide()
        self._progress.start("Rendering audio…")

        self._worker = GenerateWorker(
            model_path=model,
            audio_path=audio,
            output_name=self._name.text().strip() or "generated_audio",
            duration=duration,
            random_mode=True,
        )
        self._worker.log.connect(self._progress.append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot(bool, str, str)
    def _on_finished(self, success: bool, message: str, output_path: str):
        self._progress.finish(success, message)
        self._render_btn.setEnabled(True)
        self._worker = None
        if not success:
            return

        self._output_path = output_path
        self._success_msg.setText(Path(output_path).name)
        self._out_path_lbl.setText(output_path)
        if os.path.exists(output_path):
            self._wave.set_path(output_path)
        self._success.show()

    def _reveal(self):
        if not self._output_path:
            return
        folder = str(Path(self._output_path).resolve().parent)
        os.startfile(folder)
