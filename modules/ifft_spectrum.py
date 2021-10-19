import logging
import os
from typing import List
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import numpy as np
from numpy import cos, fft, ndarray, ndindex, pi
from numpy.lib.type_check import nan_to_num
from scipy import interpolate, fft
from scipy.interpolate import interp1d
from pathos.multiprocessing import ProcessPool as Pool
import matplotlib.transforms as transforms
from time import perf_counter
from line_profiler import LineProfiler
import cProfile
import time
import logging
logger = logging.getLogger(__name__)


class Line:
    """
    用来存储一些制定线段
    """

    def __init__(self, line, name) -> None:

        self.name = name
        self.line = line


class Pulse:

    def __init__(self, t, intensity) -> None:

        self.t, self.intensity = t, intensity

        self.intensity /= max(self.intensity)

        self.time_scope = max(self.t) - min(self.t)

        self.HMP_l, self.HMP_r, self.FWHM = self.cal_FWHM(
            self.t, self.intensity)

    def cal_FWHM(self, x: ndarray, y: ndarray):
        """计算y的FWHM(半高全宽)

        Parameters
        ----------
        x : ndarray
            自变量
        y : ndarray
            因变量

        Returns
        -------
        tuple
            (HMP_l, HMP_r, FWHM)
            HMP_l:半高处左边的点
            HMP_r:半高处右边的点
            FWHM:半高全宽值
        """

        ty = np.column_stack([x, np.nan_to_num(y)])

        ty_upper_half = ty[ty[:, 1] >= .5]

        if len(ty_upper_half) != 0:
            t_upper_half, y_upper_half = ty_upper_half[:,
                                                       0], ty_upper_half[:, 1]

            t_left_half, t_right_half = t_upper_half[0], t_upper_half[-1]
            y_left_half, y_right_half = y_upper_half[0], y_upper_half[-1]

            HMP_l = (t_left_half, y_left_half)
            HMP_r = (t_right_half, y_right_half)
        else:
            HMP_l = (0, 0)
            HMP_r = (np.inf, 0)

        return (HMP_l, HMP_r, abs(HMP_r[0] - HMP_l[0]))

    def t_FWHM_window(self, n: int):
        """根据函数的半高全宽, 用n*(self.FWHM)作为时间窗口, 对数据进行截取

        Parameters
        ----------
        n : int

        """

        width = n * self.FWHM

        t_max = width / 2
        t_min = - width / 2
        ty = np.column_stack([self.t, self.intensity])

        ty = ty[(t_min < ty[:, 0]) * (ty[:, 0] < t_max)]

        return np.hsplit(ty, 2)

    def draw(self, n, ax: plt.Axes, cl: str = '', label: str = 'pulse'):
        """对脉冲进行绘图展示

        Parameters
        ----------
        n : int
            控制时间窗口的大小,n代表FWHM的倍数


        ax : plt.Axes

        cl : str, optional
            颜色和线条类型, by default ''
        label : str, optional
            线条名称, by default 'pulse'
        """
        # 截取n个半高宽时间窗口以内的数据进行展示
        if ax.lines == []:
            ax.plot(*self.t_FWHM_window(n), cl, label=label)
            ax.legend()
        else:
            for line in ax.get_lines():
                if line._label == 'pulse':
                    line.set_data(*self.t_FWHM_window(n))


class Spectrum:

    def __init__(self, Figure: Figure = None, filepath=None, lamda=None, power=None, omega=None, delimiter='\t', method: str = 'ifft') -> None:
        """     

        Parameters
        ----------
        filepath : string
            光谱数据所在的文件, 若给定, 则lamda, oemga, power不再生效
        Figure: Figure
            绘图对象
        lamda : array like
            光谱数据中的波长分量, 单位为nm( 10^(-9)m ), 若给定, 则omega不再有效
        power : array like
            光谱数据中的强度分量
        omega : array like, optional
            光谱数据中的频率分量, 单位为10^(17)Hz default None
        delimiter : string, optional
            光谱文件分隔符 default '\t'
        method : str
            决定使用ift还是ifft
            ift : 逆傅里叶变换
            ifft : 快速傅里叶逆变换

        注:
        1. filepath, lamda, omega至少给定一个, 否则报错.
        2. 考虑到计算机存储数的精度,因此波长单位采用nm, 频率单位采用10^17Hz.
        3. self.spectrum中存储的是功率谱.
        """

        self.fig = Figure
        self.method = method
        if filepath:
            # 　从文件加载数据并将其存储在self.spectrum中
            self.load_spectrum(filepath, delimiter)

        elif lamda:
            self.spectrum = np.column_stack([lamda, power])

        elif omega:
            self.spectrum = np.column_stack([omega, power])

        else:
            raise ValueError('filepath, lamda, omega must be given one.')

        if filepath or lamda:

            # 将波长转化为频率，以便后续处理
            self.spectrum[:, 0] = self.lambda_omega_converter(
                self.spectrum[:, 0])

        self.spectrum[:, 1] /= max(self.spectrum[:, 1])  # 光谱功率归一化

        self.origin_spectrum = self.spectrum.copy()  # 保存原始光谱数据

        self.update(mode='auto')

    def lambda_omega_converter(self, x):
        """将x对应的波长和频率进行相互转化
        注: omega的数量级为10^17

        Parameters:
        -------
        x : ndarray or scalar
            波长或频率组成的数组或者标量

        Returns
        -------
        array like
            若x为频率,则返回波长, 反之亦然
        """

        return 6 * np.pi / x

    def load_spectrum(self, filepath, delimiter):
        """从给定文件中加载光谱数据

        Parameters
        ----------
        filepath : string
            文件路径
        delimiter : string
            数据分隔符
        """
        for delimiter_ in [delimiter, '\t', ' ', ',']: # 一个一个文件分隔符进行尝试, 如果加载错误, 则说明文件分隔符不对
            try:
                self.spectrum = np.loadtxt(fname=filepath, delimiter=delimiter_)
            except:
                continue

    def clear_noise(self, *, mode: str, spectrum: ndarray, omega_min: float = None, omega_max: float = None, omega_window_factor: float = 5, threshold: float = 0.):
        """
        清除光谱数据中的噪声数据
        本函数会将低于阈值的数据归为0, 将在omega窗口外的数据归为0.

        Parameters
        ----------
        mode : str
            清除噪声的模式, 分为'auto'和'manual'两种:
            'auto':从初始阈值开始,自动检测最合适阈值.
            'manual':此模式下,应指定omega_min, omega_max和threshold否则函数不对光谱进行处理
        spectrum
            光谱数据
        omega_min : float, optional
            频率下限, by default None
        omega_max : float, optional
            频率上限, by default None
        omega_window_factor : float
            频率窗口因子 频率窗口大小=FWHM * omega_window_factor
        threshold : float, optional
            功率值(已归一化)在threshold以下的数据被清除, by default None
        """

        def omega_window(spectrum: ndarray, omega_min, omega_max):
            """将spectrum在频率窗口之外的成分归零
                如果omega_min, 或者omega_max为None则不对相应部分进行归零处理

            Parameters
            ----------
            spectrum : ndarray
                光谱
            omega_min : float
                频率最小值
            omega_max : float
                频率最大值

            Returns
            -------
            spectrum
                清零后的光谱数据
            """
            spectrum_ = spectrum.copy()

            if omega_min != None:
                spectrum_[:, 1] = np.where(
                    spectrum_[:, 0] >= omega_min, spectrum_[:, 1], 0)

            if omega_max != None:
                spectrum_[:, 1] = np.where(
                    spectrum_[:, 0] <= omega_max, spectrum_[:, 1], 0)

            return spectrum_

        def zero_by_threshold(spectrum, threshold):
            """将spectrum[:, 1]中小于threshold的数据归零

            Parameters
            ----------
            spectrum : 
                光谱数据
            threshold : 
                阈值

            Returns
            -------
                spectrum
            """

            spectrum_ = spectrum.copy()

            spectrum_[:, 1] = np.where(
                spectrum_[:, 1] >= threshold, spectrum[:, 1], 0)

            return spectrum_

        spectrum_backup = spectrum.copy()

        if mode == 'auto':
            # 自动调节阈值清除噪声
            threshold = 0.005  # 起始阈值
            threshold_increase_step = 0.001

            # 去除噪音的控制参数 阈值增加所被删除的数据个数除以步长应该大于该阈值, 否则停止阈值增加(去除噪音完成)
            minimum_data_deleted_per_step = len(spectrum[:, 0])

            init_len = np.count_nonzero(spectrum)
            spectrum = zero_by_threshold(spectrum, threshold)  # 在初始时先去噪一次

            last_len = np.count_nonzero(spectrum)

            while True:

                threshold += threshold_increase_step

                new_spectrum = zero_by_threshold(spectrum, threshold)

                next_len = np.count_nonzero(new_spectrum)

                if (last_len - next_len) / (threshold_increase_step * init_len) >= minimum_data_deleted_per_step:

                    last_len = next_len

                    spectrum = new_spectrum

                    continue
                else:
                    threshold -= threshold_increase_step
                    break

            # 将光谱限制在一定范围内
            pl, pr, fwhm = self.cal_FWHM(*np.hsplit(self.spectrum, 2))
            omega_0 = (pl[0] + pr[0]) / 2
            omega_min = omega_0 - fwhm * omega_window_factor
            omega_min = omega_min if omega_min >=0 else min(self.spectrum[:, 0])
            omega_max = omega_0 + fwhm * omega_window_factor
            omega_max = omega_max if (spec_omega_max:=max(self.spectrum[:, 0])) >= omega_max else spec_omega_max 

            spectrum = omega_window(spectrum, omega_min, omega_max)
            self.omega_min, self.omega_max = omega_min, omega_max

        if mode == 'manual':

            spectrum = omega_window(spectrum, omega_min, omega_max)

            spectrum = zero_by_threshold(spectrum, threshold)

            self.omega_min, self.omega_max = omega_min, omega_max

        return spectrum, threshold

    def interpolation(self):
        """对self.spectrum进行插值
        在生成的插值数组中, 频率间隔相等, 
        在最初情况下频率的最小间隔等于self.spectrum的最小频率间隔
        在后续更新的情况下, 最小频率间隔由上一个pulse的时间范围决定(处于性能考虑)
        Return
        ---------
        spectrum
        """
        def find_min_delta_omega(oemga: ndarray):
            """计算频率间隔的最小值
            Parameters
            ----------
            oemga : ndarray
                频率数组

            Returns
            -------
            float  
                min_delta_omega
            """

            min_delta_omega = abs(omega[0] - omega[1])
            for index in range(0, len(omega) - 1):
                delta_omega = abs(omega[index] - omega[index+1])
                if delta_omega < min_delta_omega:
                    min_delta_omega = delta_omega

            return min_delta_omega

        def generate_equal_space_array(start: float, step: float, N: int):
            """产生一个长度为N的数组

            Parameters
            ----------
            start : float
                数组起始值
            step : float
                数组间距
            N : int
                数组长度
            """
            return np.arange(0, N, 1) * step

        def flatten(array): return array.flatten()  # 将数组变成一维

        omega, power = map(flatten, np.hsplit(self.spectrum, 2))
        interp_fun = interp1d(omega, power, kind='cubic',
                              bounds_error=False, fill_value=(0, 0))

        min_delta_omega = find_min_delta_omega(omega)

        # 下式中乘100是因为函数接口中给出的self.delta_t的单位为fs,
        # 数量级为10^(-15), 在傅里叶变换过程中,时间的数量级始终为10^(-17)
        self.N = int(2 * pi / (min_delta_omega * self.delta_t * 100))

        # 计算出N之后计算实际的\delta_t
        self.real_delta_t = 2 * pi / (min_delta_omega * self.N) * 0.01

        # 防止设置的self.delta_t过大,导致频谱成分丢失.
        if (omega_max := max(omega)) > (pi / self.delta_t):
            raise Warning('self.delta_t过大, 光谱的最大频率已大于奈奎斯特频率')

        # 产生新的等间隔omega数组,
        # 注意:此时omega_new中频率的最大值应该略微小于原来数组omega中的最大值max(omega)
        omega_new = np.arange(start=0, stop=omega_max, step=min_delta_omega)

        power_new = interp_fun(omega_new)

        if min(power_new) < 0:  # 插值出现负数, 将负数全变成0
            power_new = np.where(power_new >= 0, power_new, 0)

        return np.column_stack([omega_new, power_new])

        # plt.plot(omega, power, 'r',
        #          #  omega_new, power_new, 'b+'
        #          )

        # plt.show()

    def ifft(self, spectrum):
        """对spectrum进行逆快速傅里叶变换

        Returns
        -------
        tuple
            (t, intensity)
            t: 时间坐标
            intensity: 时间t处的功率
        """

        self.amp_spectrum = spectrum.copy()
        self.amp_spectrum[:, 1] = np.sqrt(self.amp_spectrum[:, 1])

        amp = fft.ifft(self.amp_spectrum[:, 1], n=self.N)

        delta_omega = abs(
            self.interpolated_spectrum[:, 0][1] - self.interpolated_spectrum[:, 0][0])

        delta_t = self.real_delta_t

        # self.amp_spectrum和self.origin_spectrum的最大频率值严重不符

        intensity = np.abs(amp) ** 2

        # 将脉冲的后半部分挪到负时间轴
        min_index = intensity.argmin()
        intensity = np.concatenate(
            [intensity[min_index:], intensity[0:min_index]])

        t = np.arange(min_index - self.N, min_index, 1) * delta_t
        return (t, intensity)

    def ift(self, spectrum) -> np.ndarray:
        """对self.spectrum进行傅里叶变换

        Parameters:
        delta_t : float
            时间间隔

        Returns
        -------
        np.ndarray
            (t, y)
        """

        def f(t: np.ndarray):
            """光功率的函数

            Parameters
            ----------
            t : np.ndarray
                时间
            """

            def single_process_f(t: np.ndarray):

                y = np.zeros((len(t), ), dtype=np.complex128)

                omega_array = spectrum[:, 0]

                for omega, amp in zip(omega_array, np.sqrt(spectrum[:, 1])):

                    # 波长的单位应为nm，时间的单位应为fs
                    y += amp * np.e**(1j * omega * t * 100)

                return abs(y) ** 2

            with Pool(nodes=os.cpu_count()) as p:
                res = p.map(single_process_f,
                            np.array_split(t, os.cpu_count()))

            return np.concatenate(res)  # 输出值是光功率

        def update_t_y(t_min, t_max, delta_t, t, y: np.ndarray):

            t_c_min, t_c_max = t[0], t[-1]

            if t_max < t_c_max:
                raise ValueError(
                    'current maximum value of t larger than expected maximum value of t')

            if t_min > t_c_min:
                raise ValueError(
                    'current minimum value of t smaller than expected minimum value of t')

            t_lp = np.linspace(t_min, t_c_min, int(
                (t_c_min - t_min) / delta_t))
            y_lp = f(t_lp)

            t_rp = np.linspace(t_c_max, t_max, int(
                (t_max - t_c_max) / delta_t))
            y_rp = f(t_rp)

            return (np.concatenate((t_lp, t, t_rp)), np.concatenate((y_lp, y, y_rp)))

        def extend_y_to_half_maximum(t, y, delta_t):

            while True:

                t_min, t_max = t[0], t[-1]
                # 扩展右边界
                if (current_ratio := y[-1] / max(y)) >= 0.3:
                    t_max = t[-1] / (1 - current_ratio)

                # 扩展左边界
                if (current_ratio := y[0] / max(y)) >= 0.3:
                    t_min = t[0] / (1 - current_ratio)

                if max(y[0] / max(y), y[-1] / max(y)) <= 0.3:
                    break

                t, y = update_t_y(
                    t_min=t_min,
                    t_max=t_max,
                    delta_t=delta_t,
                    t=t,
                    y=y
                )
            return (t, y)

        t_min = -10
        t_max = 10

        delta_t = self.delta_t

        t = np.linspace(t_min, t_max, int((t_max - t_min) / delta_t))

        y = f(t)

        t, y = extend_y_to_half_maximum(t, y, delta_t)

        y /= max(y)

        return (t, y)

    def cal_FWHM(self, x: ndarray, y: ndarray):
        """计算y的FWHM(半高全宽)

        Parameters
        ----------
        x : ndarray
            自变量
        y : ndarray
            因变量

        Returns
        -------
        tuple
            (HMP_l, HMP_r, FWHM)
            HMP_l:半高处左边的点
            HMP_r:半高处右边的点
            FWHM:半高全宽值
        """
        ty = np.column_stack([x, y])

        ty_upper_half = ty[ty[:, 1] >= .5]

        t_upper_half, y_upper_half = ty_upper_half[:, 0], ty_upper_half[:, 1]

        t_left_half, t_right_half = t_upper_half[0], t_upper_half[-1]
        y_left_half, y_right_half = y_upper_half[0], y_upper_half[-1]

        HMP_l = (t_left_half, y_left_half)
        HMP_r = (t_right_half, y_right_half)

        return (HMP_l, HMP_r, abs(t_right_half - t_left_half))

    def draw(self, show_interpolated_spec: bool = False, show_cleaned_spec: bool = False):

        def plot_spectrum(ax: plt.Axes, spectrum, cl, label, aux_line: bool = False, **kwargs):
            """绘制光谱数据

            Parameters
            ----------
            ax : plt.Axes

            spectrum : array like
                光谱
            cl : string
                线条颜色和样式控制
            label : string
                线条标签
            """
            lamda_power = spectrum.copy()

            lamda_power[:, 0] = self.lambda_omega_converter(lamda_power[:, 0])

            if ax.lines == []:  # 光谱原图只绘制一次
                ax.plot(lamda_power[:, 0], lamda_power[:, 1],
                        cl, label=label, **kwargs)

            if aux_line:
                pl, pr, FWHM = self.cal_FWHM(
                    lamda_power[:, 0], lamda_power[:, 1])
                # TODO 绘制辅助线
                draw_auxiliary_line(ax, pl, pr, unit='nm', footnote='\lambda')

        def draw_auxiliary_line(ax: plt.Axes, pl, pr, unit: str, footnote: str):
            """绘制半高宽辅助线

            Parameters
            ----------
            ax : plt.Axes
                绘图对象
            pl : tuple
                半高处左边点
            pr : tuple
                半高处右边点
            unit : str
                半高宽的单位
            footnote : str
                半高宽的脚标
            """
            def get_annot(ax):

                return [child for child in ax.get_children()
                        if isinstance(child, matplotlib.text.Annotation)]

            def get_text(ax, part_string):

                return [child for child in ax.get_children()
                        if isinstance(child, matplotlib.text.Text) and part_string in child.get_text()]

            if (annot := get_annot(ax)) != []:
                annot[0].xy = pr
                annot[0].xyann = pl
            else:
                ax.annotate('',
                            xy=pr, xycoords='data',
                            xytext=pl, textcoords='data',
                            arrowprops=dict(arrowstyle='<|-|>',
                                            color='red')
                            )

            # 判断半高宽是否为inf
            if abs(pr[0] - pl[0]) == np.inf:
                s = '$\Delta_{%s} = \infty %s$' % (
                    footnote, unit)
            else:
                s = '$\Delta_{%s} = %.2f %s$' % (
                    footnote, abs(pr[0] - pl[0]), unit)

            # 判断是否已经存在文本
            if (t := get_text(ax, 'Delta')) != []:
                t[0].set_text(s)

            else:
                ax.text(x=(pr[0] + pl[0]) / 2,
                        y=(pr[1] + pr[1]) / 2,
                        s=s,
                        verticalalignment='bottom', horizontalalignment='center'
                        )

            # ylim = ax.get_ylim()

            # for p in [pl, pr]:
            #     ax.vlines(p[0], ymin=ylim[0], ymax=p[1],
            #               linestyle='--', color='red')
            #     # 当绘制了辅助线之后, y轴范围会改变, 必须重新将ylim设置回原来的值
            #     ax.set_ylim(ylim)

        start_time = perf_counter()

        if self.fig is None:
            self.fig = plt.figure()
        if not hasattr(self, 'grid_spec'):
            self.grid_spec = self.fig.add_gridspec(2, 2)
        logger.debug(f'创建grid_spec耗时:{perf_counter()- start_time}')
        end_time = perf_counter()

        # 绘制原始光谱数据
        if (axes := self.fig.get_axes()) == []:
            ax1 = self.fig.add_subplot(self.grid_spec[0, :])
            ax2 = self.fig.add_subplot(self.grid_spec[1, :])

        else:
            for ax in axes:
                if ax.get_title() == 'frequency domain':
                    ax1 = ax
                if ax.get_title() == 'time domain':
                    ax2 = ax

        ax1.set_title('frequency domain')

        if ax1.lines == []:
            plot_spectrum(ax1, self.origin_spectrum, 'b',
                          label='raw', aux_line=True)

            if show_interpolated_spec:
                # 绘制插值后光谱数据
                plot_spectrum(ax1, self.interpolated_spectrum,
                              cl='C0^', label='interpolated_spectrum')

            if show_cleaned_spec:
                # 绘制去噪后光谱数据
                if hasattr(self, 'ifft_spectrum'):
                    cleaned_spectrum = self.ifft_spectrum
                elif hasattr(self, 'ift_spectrum'):
                    cleaned_spectrum = self.ift_spectrum
                plot_spectrum(ax1, cleaned_spectrum, 'gv',
                              'used spectrum(after clear noise)', aux_line=True)

        # 绘制阈值线
        lambda_min = self.lambda_omega_converter(self.omega_max)

        # 图上没有预知线, 需要初始化
        if len(ax1.lines) == 1:
            self.threshold_line = Line(ax1.axhline(y=self.clear_noise_final_threshold,
                                                   color='red', label='threshold'), 'threshold')
            # 绘制omega窗口
            if self.omega_min != None:
                lambda_max = self.lambda_omega_converter(self.omega_min)
                self.lambda_max_line = Line(ax1.axvline(lambda_max,
                                                        color='r', label='$\lambda_{max}$'), 'lambda max')

            if self.omega_max != None:
                lambda_min = self.lambda_omega_converter(self.omega_max)
                self.lambda_min_line = Line(ax1.axvline(lambda_min,
                                                        color='g', label='$\lambda_{min}$'), 'lambda min')

            try:  # 尝试根据现有波长窗口设置x边界, 如果有任意一者为None, 则跳过
                ax1.set_xlim([lambda_min - 50, lambda_max + 50])
                pass
            except:
                pass

            ax1.legend()
        
        # 刷新阈值线
        else:
            for line in ax1.lines:
                if line._label == 'threshold':
                    line.set_ydata([self.clear_noise_final_threshold] * 2)

                if line._label == '$\\lambda_{max}$':
                    line.set_xdata(
                        [self.lambda_omega_converter(self.omega_min)] * 2)

                if line._label == '$\\lambda_{min}$':
                    line.set_xdata(
                        [self.lambda_omega_converter(self.omega_max)] * 2)

        logger.debug(f'绘制原始光谱数据耗时:{perf_counter()- end_time}')
        end_time = perf_counter()

        # 绘制脉冲及其辅助线
        ax2.set_title('time domain')
        # 绘制4FWHM内的数据
        self.pulse.draw(n=4, ax=ax2, cl='r')
        draw_auxiliary_line(ax2, self.pulse.HMP_l, self.pulse.HMP_r, 'fs', 't')

        logger.debug(f'绘制脉冲数据耗时:{perf_counter()- end_time}')
        end_time = perf_counter()


    def update(self, *, mode: str, omega_min: float = None, omega_max: float = None, threshold: float = 0, delta_t: float = .1):

        # 找到光谱的最小频率和最大频率
        logger.debug('正在刷新')
        logger.debug(
            f'omega_min:{omega_min}\t omega_max:{omega_max}, threshold:{threshold}')
        start = perf_counter()
        self.delta_t = delta_t

        if omega_max is None and hasattr(self, 'omega_max'):
            omega_max = self.omega_max
        if omega_min is None and hasattr(self, 'omega_min'):
            omega_min = self.omega_min
        if self.method == 'ifft':
            # 对self.spectrum进行插值, 生成self.interpolated_spectrum
            self.interpolated_spectrum = self.interpolation()
            # 对self.interpolate_spectrum进行去噪处理, 结果存储于self.spectrum中
            # 将最终去噪阈值存储于self.clear_noise_final_threshold中
            self.ifft_spectrum, self.clear_noise_final_threshold = \
                self.clear_noise(mode=mode, spectrum=self.interpolated_spectrum,
                                 omega_min=omega_min, omega_max=omega_max, threshold=threshold)

            # 对self.spectrum进行逆快速傅里叶变换
            self.pulse = Pulse(*self.ifft(spectrum=self.ifft_spectrum))
        elif self.method == 'ift':

            self.ift_spectrum, self.clear_noise_final_threshold = \
                self.clear_noise(mode=mode, spectrum=self.spectrum,
                                 omega_min=omega_min, omega_max=omega_max, threshold=threshold)

            self.pulse = Pulse(*self.ift(self.ift_spectrum))

        # 使用line_profiler分析代码执行时间
        # ifft_end = perf_counter()
        # logger.debug(f'ifft 用时:{ifft_end-start}')
        # lp_profiler = LineProfiler()
        # lp_fun = lp_profiler(self.draw)
        # lp_fun()
        # lp_profiler.print_stats()

        self.draw()

        # with cProfile.Profile() as pf:
        #     self.draw()
        #     pf.print_stats(sort='tottime')

        logger.debug(f'绘制用时:{perf_counter()-start}')


if __name__ == '__main__':
    filepath = 'data/spec.txt'

    start = perf_counter()
    spectrum = Spectrum(filepath=filepath, method='ifft')
    print(f'used time:{perf_counter() - start}s')

    plt.show()


