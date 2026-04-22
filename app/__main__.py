import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont

from app.ui.main_window import MainWindow


def _load_stylesheet(app: QApplication) -> None:
    qss_path = Path(__file__).parent / "ui" / "tokens.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))


def _load_fonts() -> None:
    # Try to load bundled fonts; fall back to system fonts gracefully.
    fonts_dir = Path(__file__).parent / "ui" / "fonts"
    if fonts_dir.exists():
        for f in fonts_dir.glob("*.ttf"):
            QFontDatabase.addApplicationFont(str(f))


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("RAVE-TFG")
    app.setApplicationVersion("0.4.2")

    _load_fonts()

    # Set default fonts
    sans = QFont("Inter")
    sans.setStyleHint(QFont.SansSerif)
    sans.setPointSizeF(9.5)
    app.setFont(sans)

    _load_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
