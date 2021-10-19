from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication
import sys
from ui.gui import MainWindow
import logging
import resources
import os

LOG_FOLDER = 'log'
LOG_FILE = 'log'
LOG_FORMAT = '%(asctime)-15s %(message)s'

if not os.path.exists(LOG_FOLDER):
    os.mkdir(LOG_FOLDER)

# 重定向错误输出文件
ERROR_FILE = 'error'

if '--debug' in sys.argv:
    # 控制台输出日志
    log_level = logging.DEBUG
    stderr_fp = open(os.path.join(LOG_FOLDER, ERROR_FILE), 'w')
    sys.stderr = stderr_fp

else:
    log_level = logging.WARNING
    stderr_fp = None

logging.basicConfig(level=log_level, filename=os.path.join(LOG_FOLDER, LOG_FILE), format=LOG_FORMAT, encoding='utf-8')

logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

try: # 解决任务栏图标不能显示的问题(修改appid)
    from PyQt5.QtWinExtras import QtWin
    myappid = 'com.x.spectrum.1.0'
    QtWin.setCurrentProcessExplicitAppUserModelID(myappid) 
except ImportError:
    pass

def main():

    logger.info('日志记录开始')
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(':/icons/application_icon.ico'))
    w = MainWindow()
    res = app.exec_()

    logger.info('日志记录结束')
    if not stderr_fp is None:
        stderr_fp.close() # 关闭错误日志文件指针
    sys.exit(res)

if __name__ == '__main__':

    main()
    