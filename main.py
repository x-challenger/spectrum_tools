from PyQt5.QtWidgets import QApplication
import sys
from ui.gui import MainWindow

def main():

    app = QApplication(sys.argv)

    w = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':

    main()