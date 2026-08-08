from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QWizard

from app.config import CONFIG_PATH, load_config, save_config
from app.ui.main_window import MainWindow
from app.ui.setup_wizard import SetupWizard


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Friend")

    is_first_run = not CONFIG_PATH.exists()
    config = load_config()

    if is_first_run:
        wizard = SetupWizard(config)
        if wizard.exec() != QWizard.Accepted:
            return 0
        save_config(config)

    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
