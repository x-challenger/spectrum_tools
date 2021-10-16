from enum import Flag
import re
import typing
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QAction, QApplication, QDesktopWidget, QDoubleSpinBox,
                             QFileDialog, QFontDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget, qApp, QComboBox)
from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtSvg import QSvgWidget
import os
import sys
from pathlib import Path
import matplotlib
# from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.pyplot import text
from modules.ifft_spectrum import *
import threading
import time
from threading import Timer
import logging
from matplotlib.backend_bases import Event

logger = logging.getLogger(__name__)

matplotlib.use('Qt5Agg')


class OverWriteError(Exception):
    """raised when singleton's attribute which already exists has been overwrite.

    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Singleton:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class DataBase(Singleton):

    def __init__(self):
        Singleton.__init__(self)

    def __getitem__(self, name):

        return self.__getattribute__(name)

    def init(self, filepath, Figure):
        """初始化

        Parameters
        ----------
        filepath : str
            光谱数据路径
        """

        start_time = perf_counter()
        self.spectrum = Spectrum(filepath=filepath, Figure=Figure)
        logger.debug(f'ifft_spectrum初始化用时:{perf_counter() - start_time}')
        start_time = perf_counter()
        self.canvas.figure.tight_layout()
        self.canvas.draw()  # 只有执行此函数才能重新显示图形
        logger.debug(f'self.canvas.draw用时:{perf_counter() - start_time}')
        logger.debug('初始图形已绘制')

        self.draggable_lines = []
        for line in self.canvas.fig.axes[0].lines:
            if line.get_label() == 'raw':  # 这个名字可能会变动
                continue
            dl = DraggableStraightLine(line)
            dl.connect()

            self.draggable_lines.append(dl)

    def set_attribute(self, name: str, value):
        """为数据库增加属性

        Parameters
        ----------
        name : str
            属性名称
        value :
            属性值

        Raises
        ------
        OverWriteError
            当属性已存在时报错
        """
        if hasattr(self, name):
            raise OverWriteError('attribute %s already exists.' % name)
        self.__setattr__(name, value)


class Canvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, figure=None):
        self.fig = Figure()
        super().__init__(figure=self.fig)

    def update_legend(self, a0):

        for ax in self.figure.axes:
            ax.legend()

        self.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        self.figure.tight_layout()
        self.draw()

        
class LineEditor(QWidget):

    def __init__(self, name, min_, max_, step, prec=0) -> None:
        super().__init__()

        self.name = name
        self.setObjectName(name)

        kwargs = locals().copy()
        kwargs.pop('self')
        kwargs.pop('__class__')
        self.initUI(**kwargs)

    def initUI(self, name: str, min_, max_, step, prec=0, **kwargs):

        self.hlayout = QHBoxLayout()
        if os.path.exists(svg_path := '../resources/label/%s.svg' % name.replace(' ', '_')):
            self.lamda_max_svg = QSvgWidget(svg_path)
            self.lamda_max_svg.setFixedSize(36, 14)
            self.hlayout.addWidget(
                self.lamda_max_svg, alignment=Qt.AlignCenter)
        else:
            self.label = QLabel(f'{name}:')

            self.hlayout.addWidget(self.label, alignment=Qt.AlignCenter)

        if 'lambda' in name:
            unit = 'nm'
        if 'threshold' in name:
            unit = ''

        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setFixedWidth(100)
        self.spin_box.setRange(min_, max_)
        self.spin_box.setSingleStep(step)
        self.spin_box.setSuffix(unit)
        self.spin_box.setDecimals(prec)
        self.spin_box.setObjectName(name)
        self.spin_box.valueChanged.connect(self.on_value_changed)

        # 如果要将信号连接到数据库的方法, 必须将在此处初始化的实例存储起来,
        # 否则信号最终可能无法连接到方法. 比如使用db = DataBase(),
        # 则db会被垃圾回收, 最终导致信号无法得到处理.
        # self.spin_box.editingFinished.connect(lambda: self.update_spectrum())

        self.hlayout.addWidget(self.spin_box, alignment=Qt.AlignRight)

        self.setLayout(self.hlayout)

        self.db = DataBase()

    def reset_spinbox(self):
        try:
            self.spin_box.setValue(
                self.db['init_%s' % self.objectName().replace(' ', '_')])
        except AttributeError:
            pass

    def update_spectrum(self):
        """更新光谱,重新绘图

        Parameters
        ----------
        name : str
            被修改的参数名称
        value_fun : 
            调用此函数获得值: value=value_fun()
        """
        def get_value():
            """查询当前spinbox中的lambda min, lambda max, threshold
                返回当前值对应的omega_max, omega_min, threshold


            """
            clnbox = self.parent().parent()

            value = {}
            for spin_box in clnbox.findChildren(QDoubleSpinBox):
                if spin_box.objectName() == 'lambda max':
                    value['omega_min'] = self.db.spectrum.lambda_omega_converter(
                        spin_box.value())
                elif spin_box.objectName() == 'lambda min':
                    value['omega_max'] = self.db.spectrum.lambda_omega_converter(
                        spin_box.value())

                else:
                    value[spin_box.objectName()] = spin_box.value()

            return value

        def isequal_last_value(value: dict):
            """判断本次所需要更新的值是否与上次相同

            Parameters
            ----------
            value : dict
                所需要更新的值
            """
            last_value = {
                'omega_max': self.db.spectrum.omega_max,
                'omega_min': self.db.spectrum.omega_min,
                'threshold': self.db.spectrum.clear_noise_final_threshold,
            }

            equal = False
            for key in value:
                if value[key] != last_value[key]:
                    continue
                equal = True

        value = get_value()

        if hasattr(self.db, 'spectrum') and not isequal_last_value(value):  # 只有在已经打开光谱的情况下才会更新光谱
            self.db.spectrum.update(mode='manual', **value)
            self.db.canvas.draw()

    def on_value_changed(self, a0):
        """
        当输入框数值改变时作出反应
        """

        # 当lambda的某个窗口的值改变时, 应该调整另一个窗口的取值范围
        if self.objectName() == 'lambda max':  # lambda max的值改变
            lambda_min_sbox = self.parentWidget().findChild(QDoubleSpinBox, 'lambda min')
            lambda_min_sbox.setMaximum(self.spin_box.value())

        if self.objectName() == 'lambda min':  # 聚焦于lambda min输入框
            lambda_max_sbox = self.parentWidget().findChild(QDoubleSpinBox, 'lambda max')
            lambda_max_sbox.setMinimum(self.spin_box.value())

        if ClearNoiseBox.lock: # 如果被锁, 则不更新光谱
            return

        self.update_spectrum()


class GroupBox(QGroupBox):

    def __init__(self, type_):

        if type_ == 'omega':

            super().__init__('1')

        elif type_ == 'threshold':

            super().__init__('2')

        self.initUI(type_)

    def initUI(self, type_):

        vbox = QVBoxLayout()

        if type_ == 'omega':
            self.omega_max_line_editor = LineEditor('lambda max', 1, 2000, 1)
            self.omega_min_line_editor = LineEditor('lambda min', 1, 2000, 1)

            vbox.addWidget(self.omega_max_line_editor,
                           alignment=Qt.AlignCenter)
            vbox.addWidget(self.omega_min_line_editor,
                           alignment=Qt.AlignCenter)

        elif type_ == 'threshold':

            self.threshold_line_editor = LineEditor(
                'threshold', 0, 1, 0.001, 3)

            vbox.addWidget(self.threshold_line_editor)

        self.setLayout(vbox)


class ClearNoiseBox(QGroupBox):

    lock = False  # 用来控制是否进行更新光谱操作的全局锁

    def __init__(self):

        super().__init__('clear noise')
        self.initUI()

    def initUI(self):

        vbox = QVBoxLayout()

        # 添加恢复到默认参数选项
        hbox = QHBoxLayout()
        vbox.addLayout(hbox)
        self.reset_btn = QPushButton('reset')
        hbox.addWidget(self.reset_btn, alignment=Qt.AlignCenter)
        self.reset_btn.setToolTip(
            'click to reset the parameters to initial value.')
        self.reset_btn.clicked.connect(self.on_reset_btn_clicked)

        vbox.addWidget(GroupBox('omega'), alignment=Qt.AlignCenter)
        vbox.addWidget(QLabel('and', self), alignment=Qt.AlignCenter)
        vbox.addWidget(GroupBox('threshold'), alignment=Qt.AlignCenter)

        self.setLayout(vbox)

    def on_reset_btn_clicked(self):
        for editor in self.findChildren(LineEditor):
            editor.reset_spinbox()


class DraggableStraightLine:
    """可以在图中拖动的直线
    """
    lock = None  # 控制一次只能移动一条线

    db = None # 用来修改spin_box中的值

    def __init__(self, line: Line2D) -> None:
        self.line = line
        self.xy_data = None
        self.xy_mouse = None

        self.background = None
        self.new_value = None

        if DraggableStraightLine.db is None:  # 若db为空, 则进行初始化

            DraggableStraightLine.db = DataBase()

    def connect(self):

        self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)

    def on_press(self, event: Event):
        print('mouse pressed')
        # 若鼠标不在该axes上或者锁不为空, 则返回
        if event.inaxes != self.line.axes or DraggableStraightLine.lock != None:
            return

        if not self.line.contains(event)[0]:  # 查看鼠标是否在该条线上
            return

        DraggableStraightLine.lock = self
        self.mouse_press_xy = event.xdata, event.ydata
        self.line_xy_0 = self.line.get_xydata()

        canvas = self.line.figure.canvas
        axes = self.line.axes

        # animated=True意味着告诉matplotlib只有在显式请求时才重新绘制该对象
        self.line.set_animated(True)
        canvas.draw()  # 此时绘图结果只包含除上述线段外的其他内容

        # 保存背景, 该背景在以后绘制中保持不变
        self.background = canvas.copy_from_bbox(self.line.axes.bbox)

        # 重新绘制此条线, 因为canvas.draw()并未绘制该条线
        axes.draw_artist(self.line)  # 显式请求绘制animated=True的artist, 使用缓存渲染器

        canvas.blit(axes.bbox)  # 将已经更新的RGBA缓冲显示在GUI上


    def on_motion(self, event):
        """
        移动线段
        """

        # 若鼠标移动时间不在线段所在的区域, 或者被选中的对象不是该线段, 则返回.
        # 由此可以看出matplotlib的事件处理是多线程的, 所有DraggableStraightLine对象均接受到此信号,
        # 但只有self.lock锁定的对象(被锁定意味着在press时间中被选中)才能继续移动
        if event.inaxes != self.line.axes or DraggableStraightLine.lock != self:
            return

        p0, p1 = self.line_xy_0

        x_press, y_press = self.mouse_press_xy

        x_mouse, y_mouse = event.xdata, event.ydata

        if (p0 == p1).all():  # 如果两个端点的值直接相等, 没有什么意义, 不做处理
            return

        if p0[0] == p1[0]:  # 如果两个端点的x值相等, 那么说明为竖线, 应该水平移动

            delta_x = x_mouse - x_press

            self.line.set_xdata([p0[0] + delta_x] * 2)

            self.new_value = p0[0] + delta_x

        elif p0[1] == p1[1]:  # 若两端点y值相等, 则说明为水平线段, 应该竖直移动

            delta_y = y_mouse - y_press

            self.line.set_ydata([p0[1] + delta_y] * 2)

            self.new_value = p0[1] + delta_y

        else:  # 如果既不水平也不垂直则不做处理
            return

        canvas = self.line.figure.canvas
        axes = self.line.axes

        canvas.restore_region(self.background)  # 重新恢复背景(不包含要移动的线段)

        axes.draw_artist(self.line)  # 重新绘制线段

        canvas.blit(axes.bbox)  # 显示图形

        ClearNoiseBox.lock = True # 上锁, 禁用绘图更新
        # 更新spinbox中的值
        self.update_value(update_fig=False)
        
    def on_release(self, event):
        """
        鼠标释放时, 绘制最终的图形, 释放锁

        Args:
            event (Event): 鼠标释放事件
        """

        if DraggableStraightLine.lock is not self:
            return

        if self.new_value is None: # 没有这个值则说明没有移动
            return

        DraggableStraightLine.lock = None
        self.mouse_press_xy = None
        self.line_xy_0 = None

        self.line.set_animated(False) # 关闭animated
        self.background = None

        self.line.figure.canvas.draw()  # 重新绘制整副图形(包含被移动的线段(animated已关闭))
        self.update_value(update_fig = True)
        self.new_value = None

    def update_value(self, update_fig:bool=False):
        """
        更新spin_box中的值
        """

        ClearNoiseBox.lock = (not update_fig)
        spec = DraggableStraightLine.db.spectrum

        for line in [spec.lambda_min_line, spec.lambda_max_line, spec.threshold_line]:

            if line.line is self.line:
                sbox = DraggableStraightLine.db.main_window.findChild(
                    QDoubleSpinBox, line.name)

                # 鼠标释放时在移动事件中已经将spin_box的值更新好, 因此为spin_box设置同一值不会起到任何作用, 这是只需要触发一个valueChanged信号即可
                sbox.setValue(self.new_value)
                sbox.valueChanged.emit(self.new_value)

        ClearNoiseBox.lock = False


class NavigationToolbar(NavigationToolbar2QT):

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates=coordinates)

        for action in self.findChildren(QAction):
            if action.text() == 'Customize':
                action.triggered.connect(canvas.update_legend)

class MainWidget(QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.initUI()

    def initUI(self):

        # 设置layout
        self.hlayout = QHBoxLayout()
        self.setLayout(self.hlayout)

        # 添加画布及其工具条
        vbox = QVBoxLayout()  # 竖直放置toolbar和canvas
        self.hlayout.addLayout(vbox, 1)

        self.canvas = Canvas(self)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self)
        self.canvas_toolbar.findChildren(QAction)[9].triggered.connect(self.canvas.update_legend)
        vbox.addWidget(self.canvas_toolbar)
        vbox.addWidget(self.canvas)

        # 将这块画布添加进数据库
        DataBase().set_attribute('canvas', self.canvas)

        # 设置纵向layout用于放置各类交互框
        self.vlayout = QVBoxLayout()
        self.hlayout.addLayout(self.vlayout, 0)

        self.vlayout.addStretch()  # 先增加一个伸缩空间, 用于居中输入栏

        self.vlayout.addWidget(ClearNoiseBox())
        self.info_label = QLabel()
        self.vlayout.addWidget(self.info_label, alignment=Qt.AlignCenter)
        self.info_label.setObjectName('info label')
        self.info_label.setWordWrap(True)

        self.vlayout.addStretch()

        self.set_font()

    def set_font(self):

        font = QFont('Microsoft YaHei UI')
        font.setPixelSize(12)
        self.setFont(font)


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.init_UI()

        self.show()

    def init_UI(self):

        self.resize(950, 600)
        self.setWindowTitle('spectrum tools')

        self.center()

        self.setCentralWidget(MainWidget())

        self.add_file_menu()

    def add_file_menu(self):

        def get_act(text: str, triggered_fun, shortcut: str = '', statusTip: str = ''):
            act = QAction(text, self)
            act.setShortcut(shortcut)
            act.setStatusTip(statusTip)
            act.triggered.connect(triggered_fun)

            return act

        def open_file(e):
            store_path_file = 'data/recent_open'
            if os.path.exists(store_path_file):
                with open(store_path_file, 'rb') as fp:
                    init_dir = fp.read().decode('utf-8')
            else:
                init_dir = str(Path.home())
            # fname = QFileDialog.getOpenFileName(
            #     self, caption='open file', directory=init_dir)
            fname = ['data/spec.txt']

            if fname[0]:

                # 保存这次打开的文件夹
                with open(store_path_file, 'wb+') as fp:
                    fp.write(os.path.split(fname[0])[0].encode('utf-8'))

                self.db = DataBase()
                # db.canvas.axes.clear()
                # 初始化spectrum,并在figure上进行绘图
                self.db.init(fname[0], self.db.canvas.figure)

                self.db.set_attribute('main_window', self)

                # 同步double spinbox中lambda max, lambda min和threshold的值
                spec = self.db.spectrum
                lambda_max = spec.lambda_omega_converter(spec.omega_min)
                lambda_min = spec.lambda_omega_converter(spec.omega_max)
                threshold = spec.clear_noise_final_threshold
                # 将值填写到spinbox中
                # 现在怀疑由于以下同步值的同时图形会被重绘, 因此会托慢启动速度, 需要先禁用更新
                ClearNoiseBox.lock = True
                self.findChild(
                    QDoubleSpinBox, 'lambda max').setValue(lambda_max)
                self.findChild(
                    QDoubleSpinBox, 'lambda min').setValue(lambda_min)
                self.findChild(QDoubleSpinBox, 'threshold').setValue(threshold)

                ClearNoiseBox.lock = None  # 锁置空, 启用更新

                # 将初始值保存到数据库中
                self.db.set_attribute('init_lambda_max', lambda_max)
                self.db.set_attribute('init_lambda_min', lambda_min)
                self.db.set_attribute('init_threshold', threshold)



        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')

        open_act = get_act(text='&Open', triggered_fun=open_file,
                           shortcut='ctrl+o', statusTip='open a new file')
        exit_act = get_act(text='&Exit', triggered_fun=qApp.exit,
                           shortcut='ctrl+Q', statusTip='Exit application')
        file_menu.addActions([open_act, exit_act])

    def center(self):

        qr = self.frameGeometry()

        qr.moveCenter(QDesktopWidget().availableGeometry().center())

        self.move(qr.topLeft())

    def resizeEvent(self, a0) -> None:

        logger.debug(f'窗口大小改变:{a0.size()}')
        return super().resizeEvent(a0)


def main():

    app = QApplication(sys.argv)

    w = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':

    main()
