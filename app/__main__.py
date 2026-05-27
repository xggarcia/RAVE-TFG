import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont, QIcon

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


def _load_app_icon() -> QIcon | None:
    base = Path(__file__).parent
    candidates = [
        base / "ui" / "assets" / "app_icon.ico",
        base / "ui" / "assets" / "app_icon.png",
        base / "ui" / "assets" / "icon.ico",
        base / "ui" / "assets" / "icon.png",
        base / "ui" / "icon.ico",
        base / "ui" / "icon.png",
        base.parent / "app_icon.ico",   # install root fallback
        base.parent / "app_icon.png",
    ]
    for icon_path in candidates:
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon
    return None


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # Ensures taskbar grouping/icon uses this app identity instead of python.exe.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RAVE.TFG.Desktop")
    except Exception:
        pass


def main() -> None:
    _set_windows_app_id()
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
    app_icon = _load_app_icon()
    if app_icon is not None:
        app.setWindowIcon(app_icon)
        window.setWindowIcon(app_icon)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
