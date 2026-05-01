"""Train model page — pipeline step sidebar + stacked sub-pages."""
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from app.ui.pages._train_form_widgets import _TrainStepList
from app.ui.pages._train_only_page import _TrainOnlyPage
from app.ui.pages.export import ExportPage
from app.ui.pages.preprocess import PreprocessPage
from app.ui.pages.workflow import WorkflowPage


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._step_list = _TrainStepList()
        self._step_list.navigate.connect(self._navigate)
        body.addWidget(self._step_list)

        self._stack = QStackedWidget()
        self._detail_widgets: dict[str, QWidget] = {
            "do_all":     WorkflowPage(),
            "preprocess": PreprocessPage(),
            "train_only": _TrainOnlyPage(),
            "export":     ExportPage(),
        }
        for widget in self._detail_widgets.values():
            self._stack.addWidget(widget)

        body.addWidget(self._stack, 1)

        root_widget = QWidget()
        root_widget.setLayout(body)
        root.addWidget(root_widget, 1)

        self._navigate("do_all")

    def _navigate(self, step_id: str):
        if step_id not in self._detail_widgets:
            return
        self._stack.setCurrentWidget(self._detail_widgets[step_id])
        self._step_list.set_active(step_id)
