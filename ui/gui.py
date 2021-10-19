from enum import Flag
import re
import typing
from PyQt5 import QtCore
from PyQt5.QtGui import QFont, QCloseEvent
from PyQt5.QtWidgets import (QAction, QApplication, QDesktopWidget, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFileDialog, QFontDialog, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget, qApp, QComboBox, QProgressDialog)
from PyQt5.QtCore import Q_FLAG, QSaveFile, Qt, QTimer
from PyQt5 import QtGui
# from PyQt5.QtSvg import QSvgWidget
import os
from os import path
import sys
from pathlib import Path
import matplotlib
# from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.pyplot import text
from numpy.lib.npyio import save
from modules.ifft_spectrum import *
import threading
import time
from threading import Timer
import logging
from matplotlib.backend_bases import Event
import json
import pickle
import time
CONFIG_FILE_PATH = './config/config.json'



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
        # 将两者绑定在一起, 改变任一者, 另一者便会跟着改变
        self.__dict__ = self._shared_state

    def clear(self, exclude=['canvas', 'config']):
        """
        清空数据库

        Args:
            exclude (list, optional): 该列表中的属性不会被删除. Defaults to ['canvas', 'config'].
        """
        # Singleton._shared_state与所有实例的__dict__相连, 而用popitem()或者pop()方法
        # 删除元素后不会切断这种联系, 因此两者皆会变化.
        #  pop()会报RuntimeError(在删除元素过程中改变了字典大小)
        keys = list(Singleton._shared_state.keys())

        for key in keys:

            if key in exclude:
                continue

            Singleton._shared_state.pop(key)


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
        pass

        start_time = perf_counter()
        self.spectrum = Spectrum(filepath=filepath, Figure=Figure)
        logger.debug(f'ifft_spectrum初始化用时:{perf_counter() - start_time}')
        start_time = perf_counter()
        self.canvas.figure.tight_layout()
        logger.debug(f'self.canvas.draw用时:{perf_counter() - start_time}')
        logger.debug('初始图形已绘制')

        self.draggable_lines = []
        for line in self.canvas.fig.axes[0].lines:
            if line.get_label() == 'raw':  # 这个名字可能会变动
                continue
            dl = DraggableStraightLine(line)
            dl.connect()

            self.draggable_lines.append(dl)

        self.pickable_legends = PickableLegend(self.canvas.figure)

        # # 只有执行此函数才能重新显示图形, 必须放在初始化pickable_legends的后面,
        # # 否则初始绘制的图形的legend无法被选定
        # self.canvas.draw()

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

    def load_config(self):
        """
        加载程序配置
        """

        if os.path.exists(CONFIG_FILE_PATH):

            with open(CONFIG_FILE_PATH, 'r') as inp:
                self.config = json.load(inp)
        else:

            self.config = {}

    def save_config(self):

        if not hasattr(self, 'config'):
            return
        logger.info('保存配置')

        folder = path.split(CONFIG_FILE_PATH)[0]
        if not path.exists(folder):
            
            os.mkdir(folder)

        with open(CONFIG_FILE_PATH, 'w') as outp:
            json.dump(self.config, outp)

        logger.info('配置保存完成')


    def update_spectrum(self, **kwargs):
        """更新光谱,重新绘图
        """

        self.spectrum.update(**kwargs)

        self.canvas.draw()


class Config():
    """
    程序配置对象
    """

    def __init__(self):
        pass

    def __getitem__(self, name):

        return self.__getattribute__(name)


class Canvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, figure=None):
        self.fig = Figure()
        super().__init__(figure=self.fig)
        # self.mpl_connect('draw_event', self.on_draw)

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
            # self.lamda_max_svg = QSvgWidget(svg_path)
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

        if not hasattr(self.db, 'spectrum'):
            return
        value = get_value()

        if not isequal_last_value(value):  # 只有在已经打开光谱的情况下才会更新光谱
            self.db.update_spectrum(mode='manual', **value)

    def on_value_changed(self, a0):
        """
        当输入框数值改变时作出反应
        """

        # 当lambda的某个窗口的值改变时, 应该调整另一个窗口的取值范围
        if self.objectName() == 'lambda max':  # lambda max的值改变
            lambda_min_sbox = self.parentWidget().findChild(QDoubleSpinBox, 'lambda min')
            lambda_min_sbox.setMaximum(self.spin_box.value() - 1)

        if self.objectName() == 'lambda min':  # 聚焦于lambda min输入框
            lambda_max_sbox = self.parentWidget().findChild(QDoubleSpinBox, 'lambda max')
            lambda_max_sbox.setMinimum(self.spin_box.value() + 1)

        if ClearNoiseBox.lock:  # 如果被锁, 则不更新光谱
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
        self.db = DataBase()

        vbox = QVBoxLayout()

        if type_ == 'omega':
            # 将精度设置为0.01是考虑到左右移动线段时如果精度太低,
            # 线条不会定在指定位置(波长不为整数的位置), 这样不利于对某些噪声点进行处理
            default_lambda_range = {
                'lambda_min_range': (1, 4000),
                'lambda_max_range': (1, 4000)}
            lambda_range = self.db.config.get('lambda_range', default_lambda_range)
            lambda_min_range = lambda_range['lambda_min_range']
            lambda_max_range = lambda_range['lambda_max_range']

            self.db.config['lambda_range'] = lambda_range # 将lambda的取值范围保存

            self.lambda_max_line_editor = LineEditor(
                'lambda max', lambda_max_range[0], lambda_max_range[1], 1, prec=2)
            self.lambda_min_line_editor = LineEditor(
                'lambda min', lambda_min_range[0], lambda_min_range[1], 1, prec=2)

            vbox.addWidget(self.lambda_max_line_editor,
                           alignment=Qt.AlignCenter)
            vbox.addWidget(self.lambda_min_line_editor,
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

    db = None  # 用来修改spin_box中的值

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

        ClearNoiseBox.lock = True  # 上锁, 禁用绘图更新
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

        if self.new_value is None:  # 没有这个值则说明没有移动
            return

        DraggableStraightLine.lock = None
        self.mouse_press_xy = None
        self.line_xy_0 = None

        self.line.set_animated(False)  # 关闭animated
        self.background = None

        self.line.figure.canvas.draw()  # 重新绘制整副图形(包含被移动的线段(animated已关闭))
        self.update_value(update_fig=True)
        self.new_value = None

    def update_value(self, update_fig: bool = False):
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


class PickableLegend:
    """
    该对象会传入的fig中的所有legend变成pickable
    在初始化过程中会刷新图像
    """
    def __init__(self, fig):
        self.legend = {}
        self.legend2origin = {}

        for axes in fig.get_axes():
            self.legend[axes] = axes.get_legend()

            for legend_line, origin_line in zip(self.legend[axes].get_lines(),
                                                axes.get_lines()):

                legend_line.set_picker(5)  # 5 pts tolerance

                self.legend2origin[legend_line] = origin_line

                # 根据fig中各线条是否可见来设置legend_line的透明度
                legend_line.set_visible(True) # 被设置为可选取的legend_line必须可见
                legend_line.set_alpha(1.0 if origin_line.get_visible() else 0.2)

        fig.canvas.mpl_connect('pick_event', self.on_pick)

        fig.canvas.draw() # 刷新图像

    def on_pick(self, event):

        logger.debug('pick event')

        picked_legend_line = event.artist

        origin_line = self.legend2origin[picked_legend_line]

        visible = not origin_line.get_visible()

        origin_line.set_visible(visible)

        picked_legend_line.set_alpha(1.0 if visible else 0.2)

        logger.debug(f'picked_legend_line\'s alpha:{picked_legend_line.get_alpha()}')
        logger.debug(
            f'picked_legend_line\'s color:{picked_legend_line.get_color()}')

        logger.debug(
            f'picked_legend_line\'s visible:{picked_legend_line.get_visible()}')



        picked_legend_line.figure.canvas.draw()


class NavigationToolbar(NavigationToolbar2QT):

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates=coordinates)

        for action in self.findChildren(QAction):
            if action.text() == 'Customize':
                action.triggered.connect(canvas.update_legend)

class MaskWidget(QWidget):

    def __init__(self, parent: typing.Optional['QWidget'] = ...) -> None:
        super().__init__(parent=parent)

        self.background_style_sheet = 'background:rgba(0, 0, 0, 50)'
        self.setWindowFlag(Qt.FramelessWindowHint, True)

        self.setAttribute(Qt.WA_StyledBackground)

        self.setStyleSheet(self.background_style_sheet)

        self.hide()

        vbox = QVBoxLayout(self)
        self.label = QLabel(self)
        vbox.addWidget(self.label, alignment=Qt.AlignCenter)
        vbox.addStretch(1)
        font = QFont('Nirmala UI Semilight')
        font.setPixelSize(30)
        self.label.setFont(font)

        self.label.setStyleSheet('background:rgba(0, 0, 0, 0)')
        self.label.setWindowOpacity(0)
        self.setLayout(vbox)

    def show(self, text):

        if self.parent() is None:
            return

        self.label.setText(text)
        parent_rect = self.parentWidget().geometry()

        self.setGeometry(0, 0, parent_rect.width(), parent_rect.height())

        return super().show()

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
        self.canvas_toolbar.findChildren(
            QAction)[9].triggered.connect(self.canvas.update_legend)
        vbox.addWidget(self.canvas_toolbar)
        vbox.addWidget(self.canvas)

        # 将这块画布添加进数据库
        self.db = DataBase()
        self.db.set_attribute('canvas', self.canvas)

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

        self.mask_widget = MaskWidget(self)

    def set_font(self):

        font = QFont('Microsoft YaHei UI')
        font.setPixelSize(12)
        self.setFont(font)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:

        new_size = a0.size()
        self.mask_widget.resize(new_size.width(), new_size.height())
        return super().resizeEvent(a0)

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.init_UI()

        self.show()

    def init_UI(self):

        self.db = DataBase()
        self.db.load_config()  # 加载程序配置

        size = self.db.config.get('default_main_window_size', (1120, 700))
        self.resize(*size)
        self.db.config['default_main_window_size'] = size # 保存这个size

        self.setWindowTitle('spectrum tools')

        self.center()

        self.setCentralWidget(MainWidget())

        self.add_file_menu()

        self.setAcceptDrops(True)



    def add_file_menu(self):

        def get_act(text: str, triggered_fun, shortcut: str = '', statusTip: str = ''):
            act = QAction(text, self)
            act.setShortcut(shortcut)
            act.setStatusTip(statusTip)
            act.triggered.connect(triggered_fun)

            return act

        def open_file(e):

            init_dir = dir_ if (dir_ := self.db.config.get(
                'open_file_init_dir', None)) else str(Path.home())

            fname = QFileDialog.getOpenFileName(
                self, caption='open file', directory=init_dir, filter='data file (*.txt *.dat *.csv)')
            # fname = ['tests/data/spec.txt']

            if fname[0]:
                # 保存这次打开的文件夹
                self.db.config['open_file_init_dir'] = os.path.split(fname[0])[
                    0]
                
                self.init_system(fname[0]) # 初始化系统

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')

        open_act = get_act(text='&Open', triggered_fun=open_file,
                           shortcut='ctrl+o', statusTip='open a new file')
        save_act = get_act(text='&Save', triggered_fun=self.on_save,
                           shortcut='ctrl+s', statusTip='Save result')
        exit_act = get_act(text='&Exit', triggered_fun=qApp.exit,
                           shortcut='ctrl+Q', statusTip='Exit application')
        file_menu.addActions([open_act, save_act, exit_act])

    def center(self):

        qr = self.frameGeometry()

        qr.moveCenter(QDesktopWidget().availableGeometry().center())

        self.move(qr.topLeft())

    def resizeEvent(self, a0) -> None:

        logger.debug(f'窗口大小改变:{a0.size()}')


        return super().resizeEvent(a0)

    def on_save(self):

        if not hasattr(self.db, 'spectrum'):  # 判断当前是否已经加载光谱, 若没有, 则不能保存

            QMessageBox.critical(
                self, self.windowTitle(), 'No results can be saved.')

            return

        def get_save_path():
            """
            从用户端获取保存路径
            """
            init_save_dir = dir_ if (dir_ := self.db.config.get(
                'save_file_init_dir', None)) else path.join(str(Path.home()), 'Desktop')

            folder = QFileDialog.getExistingDirectory(
                self, caption='select a folder', directory=init_save_dir)


            if folder == '':  # 用户选择取消则直接返回
                return
            
            self.db.config['save_file_init_dir'] = folder # 保存用户该次选择的文件夹

            default_name = time.strftime('%Y_%m_%d_%H_%M', time.localtime())

            # ----获取用户自定义文件夹名称----
            while True:
                name, ok = QInputDialog.getText(
                    self, self.windowTitle(), 'Enter a name for your result folder:', QLineEdit.Normal, default_name)

                if not ok:
                    return

                save_parent_folder = path.join(folder, name)

                if path.exists(save_parent_folder):  # 若用户指定的文件夹已存在, 则报错

                    warning_text = 'The following directory already exist:\n%s\n' % save_parent_folder + \
                        'Do you want to overwrite it?'

                    res = QMessageBox.warning(self, self.windowTitle(),
                                              warning_text,
                                              QMessageBox.Yes | QMessageBox.No,
                                              )

                    if res == QMessageBox.Yes:  # 用户选择继续保存
                        break
                    elif res == QMessageBox.No:  # 用户选择取消 则返回继续询问文件夹名称
                        continue
                else:  # 文件夹不存在 直接退出循环
                    break

            save_data_folder = path.join(save_parent_folder, 'data')
            save_figure_folder = path.join(save_parent_folder, 'figure')

            return save_data_folder, save_figure_folder, name
        def save_spectrum_data():
            """
            保存逆傅立叶变换实际使用的光谱
            """

            spec = self.db.spectrum
            # 只存频率大于零的部分
            ifft_spec = spec.ifft_spectrum[spec.ifft_spectrum[:, 0] > 0]
            lambda_ = spec.lambda_omega_converter(ifft_spec[:, 0])
            np.savetxt(path.join(save_data_folder, '%s_cleaned_spectrum(nm).txt' % name),
                       np.column_stack((lambda_, ifft_spec[:, 1])))

        def save_pulse_data():
            """保存脉冲时序数据
            """
            pulse = self.db.spectrum.pulse
            # 只存频率大于零的部分
            np.savetxt(path.join(save_data_folder, '%s_time_domain(fs).txt' % name),
                       np.column_stack((pulse.t, pulse.intensity)))


        def save_sub_figure(ax: plt.Axes, pad, save_path):
            """
            保存子图

            Args:
                ax (plt.Axes): 子图ax
                save_path (str): 保存路径
            """
            fig, ax2 = plt.subplots(1, 1)
            logger.debug('正在保存子图')
            for line in ax.get_lines():

                if not line.get_visible():  # 如果这条线已经不可见, 则不进行绘制
                    continue

                line_prop = line.properties()

                # 绘制辅助线
                if len(xdata := line_prop['xdata']) == 2:  # 说明为竖线或横线
                    if xdata[0] == xdata[1]:  # 说明为竖线
                        ax2.axvline(x=xdata[0],
                                    label=line_prop['label'],
                                    color=line_prop['color'],
                                    linestyle=line_prop['linestyle'])
                    else:  # 说明为横线
                        ax2.axhline(y=line_prop['ydata'][0],
                                    label=line_prop['label'],
                                    color=line_prop['color'],
                                    linestyle=line_prop['linestyle'])

                    continue

                # 绘制数据线
                ax2.plot(*np.array(line.get_xydata()).T,
                         color=line.get_c(),
                         linestyle=line.get_linestyle(),
                         label=line.get_label())

                # 添加文字和annotate
                for text in ax.texts:
                    if hasattr(text, 'xyann'):  # 若有 xyann则说明为annotate
                        ax2.annotate(text.get_text(),
                                     xy=text.xyann,
                                     xycoords='data',
                                     xytext=text.xy,
                                     textcoords='data',
                                     arrowprops=text.arrowprops
                                     )
                    else:
                        text_prop = text.properties()
                        ax2.text(x=text_prop['position'][0],
                                 y=text_prop['position'][1],
                                 s=text.get_text(),
                                 verticalalignment=text.get_verticalalignment(),
                                 horizontalalignment=text.get_horizontalalignment()
                                 )

            # 添加标题
            ax2.set_title(ax.get_title())

            ax2.set_xlim(ax.get_xlim())
            ax2.set_ylim(ax.get_ylim())
            leg = ax2.legend()

            fig.canvas.draw()
            fig.savefig(save_path, dpi=600)

            logger.debug('子图保存完成')

        def save_both_fig(fig, path):
            """
            保存整体图像
            """


            pickable_legend2origin = self.db.pickable_legends.legend2origin

            # 保存修改过的origin_line
            modified_origin_line = []
            for legend_line in pickable_legend2origin:
                origin_line = pickable_legend2origin[legend_line]

                # 如果该条线不可见, 应该在label前加_, 从而去掉该条线
                if not origin_line.get_visible(): 
                    modified_origin_line.append(origin_line)
                    origin_line.set_label('_%s' % origin_line.get_label())


            axes = fig.get_axes()

            for ax in axes:
                # 重新设置legend, 使前面的设置生效, 此时legend不再pickable, 
                # 但不可见的线对应的legend消失
                ax.legend()

            fig.canvas.draw()

            fig.savefig(path, dpi=600)

            logger.debug('整图绘制完成')
            for line in modified_origin_line:
                line.set_label(line.get_label()[1:]) # 去掉前面的_

            for ax in axes:
                ax.legend()

            self.db.pickable_legends = PickableLegend(fig) # 重新设置pickable legend

        if (res:=get_save_path()) is None:
            return

        save_data_folder, save_figure_folder, name = res

        # -----测试用--------
        # dir_ = 'tests/result'
        # save_data_folder, save_figure_folder, name = (
        #     path.join(dir_, 'data'), path.join(dir_, 'figure'), 's')
        # ------------------

        for save_path in [save_data_folder, save_figure_folder]:

            os.makedirs(save_path, exist_ok=True)  # 创建文件夹

        pd = QProgressDialog()

        # 如果为modal则进度条显示时, 用户无法与应用其他部分交互,
        # 此处设为这样是避免存储过程中意外修改数据, 造成不必要的错误
        pd.setWindowModality(Qt.WindowModal)
        pd.setRange(0, 4)
        step = 0

        save_spectrum_data()
        save_pulse_data()

        step += 1
        pd.setValue(step)  # 第一阶段: 数据保存完成

        fig = self.db.canvas.figure

        for i, ax in enumerate(fig.axes):
            save_path = f"{name}_{ax.title.get_text().replace(' ', '_')}.png"
            save_sub_figure(ax=ax, pad=1.0, save_path=path.join(
                save_figure_folder, save_path))
            step += 1
            pd.setValue(step)  # 每副子图绘制完成

        save_both_fig(fig, path.join(
            save_figure_folder, '%s_both_domain.png' % name))

        step += 1
        pd.setValue(step)  # 整图绘制完成

    def init_system(self, file_path):

        """
        当打开文件时, 执行初始化操作
        """

        # 如果这次已经加载过光谱, 需要先清除数据
        if hasattr(self.db, 'spectrum'):
            self.db.clear()  # 清除数据库
            self.db.canvas.figure.clear()  # 清楚图上的数据

        # 初始化spectrum,并在figure上进行绘图
        logger.debug('初始化')
        self.db.init(file_path, self.db.canvas.figure)

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
    
    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        logger.debug('拖拽进入应用主界面')
        logger.debug(f'a0.mimeData().hasUrls():{a0.mimeData().hasUrls()}')

        if a0.mimeData().hasUrls():
            logger.debug(f'a0.mimeData().urls():{a0.mimeData().urls()}')
            file_path = a0.mimeData().urls()[0].toLocalFile()
            
            if not file_path.split('.')[-1] in ['txt', 'dat', 'csv']:
                logger.debug(f"因该文件类型{file_path.split('.')[-1]}不受支持, 拖拽被拒绝.")
                return # 如果拖拽文件不为指定后缀则拒绝拖拽

            logger.debug('拖拽被接受')
            a0.accept() 
            self.centralWidget().mask_widget.show('release to open this file.')
        return super().dragEnterEvent(a0)

    def dragLeaveEvent(self, a0: QtGui.QDragLeaveEvent) -> None:

        self.centralWidget().mask_widget.close() # 关闭mask

        return super().dragLeaveEvent(a0)
    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        self.centralWidget().mask_widget.close()
        if a0.mimeData().hasUrls():
            file_path = a0.mimeData().urls()[0].toLocalFile()
            logger.debug(f"文件路径已获取:{file_path}")  

            for suffix in ['.txt', '.dat', '.csv']:
                if file_path.endswith(suffix):
                    logger.debug(f"正在打开:{file_path}")  
                    self.init_system(file_path)
        return super().dropEvent(a0)
    def closeEvent(self, a0: QCloseEvent) -> None:
        self.db.save_config()
        return super().closeEvent(a0)

def main():

    app = QApplication(sys.argv)

    w = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':

    main()
