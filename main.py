from PyQt5.QtWidgets import QApplication
import sys
from ui.gui import MainWindow
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    logger.info('日志记录开始')

    app = QApplication(sys.argv)
    w = MainWindow()

    res = app.exec_()
    logger.info('日志记录结束')
    sys.exit(res)



if __name__ == '__main__':

    main()