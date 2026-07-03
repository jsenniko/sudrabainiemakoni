import sys
from guihelpers.base_window import BaseCloudImageWindow, excepthook

try:
    from .star_reference_ui import Ui_MainWindow
except ImportError:
    from star_reference_ui import Ui_MainWindow


class StarReferenceWindow(BaseCloudImageWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()

    def main(self):
        pass


def main():
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("ERROR: PyQt5 is not installed.")
        print("Please install GUI dependencies with: pip install sudrabainiemakoni[gui]")
        sys.exit(1)

    sys.excepthook = excepthook

    try:
        app = QApplication(sys.argv)
        myapp = StarReferenceWindow()
        myapp.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        import os
        error_log = os.path.join(os.path.dirname(__file__), 'star_reference_error.log')
        with open(error_log, 'w') as f:
            f.write("ERROR starting GUI application:\n")
            f.write(traceback.format_exc())
        print("ERROR starting GUI application:")
        print(traceback.format_exc())
        print(f"Error logged to: {error_log}")
        sys.exit(1)


if __name__ == '__main__':
    main()
