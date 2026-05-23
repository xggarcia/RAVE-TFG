from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget
from PySide6.QtCore import Qt

from app.ui.shell.titlebar import TitleBar
from app.ui.shell.sidebar import Sidebar
from app.ui.shell.status_rail import StatusRail
from app.ui.pages.home import HomePage
from app.ui.pages.generate import GeneratePage
from app.ui.pages.stream import StreamPage
from app.ui.pages.preprocess import PreprocessPage
from app.ui.pages.export import ExportPage
from app.ui.pages.clean import CleanPage
from app.ui.pages.train import TrainPage
from app.ui.pages.dataset import DatasetPage
from app.ui.pages.workflow import WorkflowPage
from app.ui.pages.anchors import AnchorsPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAVE-TFG")
        self.setMinimumSize(1240, 760)
        self.resize(1520, 900)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self._pages: dict[str, QWidget] = {}
        self._build()

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title bar
        self._titlebar = TitleBar()
        self._titlebar.homeRequested.connect(lambda: self._navigate("home"))
        outer.addWidget(self._titlebar)

        # Middle: sidebar + content
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.navigate.connect(self._navigate)
        middle.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        middle.addWidget(self._stack, 1)

        outer.addLayout(middle, 1)

        # Status rail
        self._status_rail = StatusRail()
        outer.addWidget(self._status_rail)

        # Register pages
        home = HomePage()
        home.navigate.connect(self._navigate)
        self._register_page("home", home)

        self._register_page("preprocess", PreprocessPage())
        self._register_page("generate", GeneratePage())
        stream_page = StreamPage()
        stream_page.statusChanged.connect(self._status_rail.set_status)
        self._register_page("stream", stream_page)
        self._register_page("export", ExportPage())
        self._register_page("clean", CleanPage())
        self._register_page("train", TrainPage())
        self._register_page("dataset", DatasetPage())
        self._register_page("workflow", WorkflowPage())
        self._register_page("anchors", AnchorsPage())

        self._navigate("home")

    def _register_page(self, page_id: str, widget: QWidget):
        self._pages[page_id] = widget
        self._stack.addWidget(widget)

    def _navigate(self, page_id: str):
        if page_id not in self._pages:
            return
        self._stack.setCurrentWidget(self._pages[page_id])
        self._sidebar.set_active(page_id)
