import sys
import os

# ── ensure the project root is on sys.path so sub-packages resolve ────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore    import Qt

import config
from ui import MainWindow


def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_SHORT_NAME)
    app.setOrganizationName(config.ORG_NAME)

    window = MainWindow()
    window.show()

    # ── optional command-line argument ───────────────────────────────────
    args = sys.argv[1:]
    if args:
        if os.path.isfile(args[0]):
            from PyQt5.QtCore import QTimer
            # Short delay so the window is fully shown before loading
            QTimer.singleShot(200, lambda: window._start_video(args[0]))
        else:
            print(f"[EndoSim] Warning: '{args[0]}' is not a valid file. "
                  "Pass a valid video file path.")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
