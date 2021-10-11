import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from numpy import cos, fft, ndarray, ndindex, pi
from scipy import interpolate, fft
from scipy.interpolate import interp1d
from pathos.multiprocessing import ProcessPool as Pool
import matplotlib.transforms as transforms


class Pulse:

    def __init__(self, t, intensity) -> None:

        self.t, self.intensity = t, intensity

        self.intensity /= max(self.intensity)

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
        ty = np.column_stack([x, y])

        ty_upper_half = ty[ty[:, 1] >= .5]

        t_upper_half, y_upper_half = ty_upper_half[:, 0], ty_upper_half[:, 1]

        t_left_half, t_right_half = t_upper_half[0], t_upper_half[-1]
        y_left_half, y_right_half = y_upper_half[0], y_upper_half[-1]

        HMP_l = (t_left_half, y_left_half)
        HMP_r = (t_right_half, y_right_half)

        return (HMP_l, HMP_r, abs(t_right_half - t_left_half))

    def t_FWHM_window(self, n:int):
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
        ax.plot(*self.t_FWHM_window(n), cl, label=label)


class Spectrum:

    def __init__(self, filepath=None, lamda=None, power=None, omega=None, delimiter='\t') -> None:
        """     

        Parameters
        ----------
        filepath : string
            光谱数据所在的文件, 若给定, 则lamda, oemga, power不再生效
        lamda : array like
            光谱数据中的波长分量, 单位为nm( 10^(-9)m ), 若给定, 则omega不再有效
        power : array like
            光谱数据中的强度分量
        omega : array like, optional
            光谱数据中的频率分量, 单位为10^(17)Hz default None
        delimiter : string, optional
            光谱文件分隔符 default '\t'

        注:
        1. filepath, lamda, omega至少给定一个, 否则报错.
        2. 考虑到计算机存储数的精度,因此波长单位采用nm, 频率单位采用10^17Hz.
        3. self.spectrum中存储的是功率谱.
        """

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
        """从给定文件中加载光谱数据, 该函数会保存一份数据备份在self.origin_data中

        Parameters
        ----------
        filepath : string
            文件路径
        delimiter : string
            数据分隔符
        """

        self.spectrum = np.loadtxt(fname=filepath, delimiter=delimiter)

    def clear_noise(self, *, mode: str, omega_min: float = None, omega_max: float = None, threshold: float = 0):
        """
        清除光谱数据中的噪声数据
        该函数会将去噪结果存储在self.spectrum中, 将最终阈值存储在self.clear_noise_final_hreshold中

        Parameters
        ----------
        mode : str
            清除噪声的模式, 分为'auto'和'manual'两种:
            'auto':从初始阈值开始,自动检测最合适阈值.
            'manual':此模式下,应指定omega_min, omega_max和threshold否则函数不对光谱进行处理
        omega_min : float, optional
            频率下限, by default None
        omega_max : float, optional
            频率上限, by default None
        threshold : float, optional
            功率值(已归一化)在threshold以下的数据被清除, by default None
        """

        spectrum = self.origin_spectrum.copy()  # 每次清除噪声总是从原始数据开始

        if mode == 'auto':
            # 清除噪声

            threshold = 0.005  # 起始阈值
            threshold_increase_step = 0.001

            # 去除噪音的控制参数 阈值增加所被删除的数据个数除以步长应该大于该阈值, 否则停止阈值增加(去除噪音完成)
            minimum_data_deleted_per_step = 478000

            init_len = len(spectrum)
            spectrum = spectrum[spectrum[:, 1] >= threshold]
            last_len = len(spectrum)

            while True:

                threshold += threshold_increase_step

                new_spectrum = spectrum[spectrum[:, 1] >= threshold]
                next_len = len(new_spectrum)

                if (last_len - next_len) / (threshold_increase_step * init_len) >= minimum_data_deleted_per_step:

                    spectrum = new_spectrum

                    last_len = next_len

                    continue
                else:
                    break

        if mode == 'manual':

            # 在以下去除噪声的过程中, 如果某个值未给定,即为None,
            # 则无法进行比较操作,报TypeError,这时只需跳过即可
            try:
                spectrum = spectrum[spectrum[:, 0] >= omega_min]
            except TypeError:
                pass

            try:
                spectrum = spectrum[spectrum[:, 0] <= omega_max]
            except TypeError:
                pass

        self.spectrum = spectrum
        self.clear_noise_final_threshold = threshold

    def find_omega_boundary(self):
        """找到光谱频率的最小值和最大值,并将其存储在self.omega_min和omega_max中.
        """

        self.omega_min, self.omega_max = [
            fun(self.spectrum[:, 0]) for fun in [min, max]]

    def interpolation(self):
        """对self.spectrum进行插值, 生成self.interpolated_spectrum
        在生成的插值数组中, 频率间隔相等, 等于self.spectrum的最小频率间隔
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

        # 找到\delta_\omega
        min_delta_omega = find_min_delta_omega(omega)

        # 下式中乘100是因为函数接口中给出的self.delta_t的单位为fs,
        # 数量级为10^(-15), 在傅里叶变换过程中,时间的数量级始终为10^(-17)
        self.N = int(2 * pi / (min_delta_omega * self.delta_t * 100))

        # 计算出N之后计算实际的\delta_t
        self.real_delta_t = 2 * pi / (min_delta_omega * self.N) * 0.01

        # 防止设置的self.delta_t过大,导致频谱成分丢失.
        if (omega_max:= max(omega)) > (pi / self.delta_t):
            raise Warning('self.delta_t过大, 光谱的最大频率已大于奈奎斯特频率')

        # 产生新的等间隔omega数组, 
        # 注意:此时omega_new中频率的最大值应该略微小于原来数组omega中的最大值max(omega)
        omega_new = np.arange(start=0, stop=omega_max, step=min_delta_omega)

        power_new = interp_fun(omega_new)

        self.interpolated_spectrum = np.column_stack([omega_new, power_new])

        # plt.plot(omega, power, 'r', omega_new, power_new, 'b+')

        # plt.show()

    def ifft(self):
        """对self.interpolated_spectrum进行逆傅里叶变换

        Returns
        -------
        tuple
            (t, intensity)
            t: 时间坐标
            intensity: 时间t处的功率
        """

        self.amp_spectrum = self.interpolated_spectrum.copy()
        self.amp_spectrum[:, 1] = np.sqrt(self.amp_spectrum[:, 1])

        # 根据离散傅里叶变换的定义, 0和奈奎斯特采样频率处的振幅对原函数的贡献为1/NF[n], 中间频率分量的贡献为2/NF[n],
        # 但光栅光谱仪测出来的光谱值, 应指的是实际脉冲中频率分量的贡献F[n], 若要使这两者相同, 则应对光栅光谱仪测出的数据除以2.
        # if len(self.amp_spectrum) // 2 == 0:  # n even
        #     self.amp_spectrum[1:-1] /= 2
        # else:
        #     self.amp_spectrum[1:-1] /= 2

        amp = fft.ifft(self.amp_spectrum[:, 1], n=self.N)

        delta_omega = abs(
            self.interpolated_spectrum[:, 0][1] - self.interpolated_spectrum[:, 0][0])


        delta_t = self.real_delta_t

        # self.amp_spectrum和self.origin_spectrum的最大频率值严重不符

        intensity = np.abs(amp) ** 2

        # 将脉冲的后半部分挪到负时间轴
        min_index = intensity.argmin()
        intensity = np.concatenate([intensity[min_index:], intensity[0:min_index]])

        t = np.arange(min_index - self.N, min_index, 1) * delta_t
        return (t, intensity)

    def ift(self, delta_t, spectrum:ndarray=None) -> np.ndarray:
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

        if spectrum is None:
            spectrum = self.spectrum

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

    def draw(self):

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

            # lamda_power[:, 0] = self.lambda_omega_converter(lamda_power[:, 0])
            ax.plot(lamda_power[:, 0], lamda_power[:, 1], cl, label=label, **kwargs)

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

            ax.annotate('',
                        xy=pr, xycoords='data',
                        xytext=pl, textcoords='data',
                        arrowprops=dict(arrowstyle='<|-|>',
                                        color='red')
                        )

            ax.text(x=(pr[0] + pl[0]) / 2,
                    y=(pr[1] + pr[1]) / 2,
                    s='$\Delta_{%s} = %.2f %s$' % (
                        footnote, abs(pr[0] - pl[0]), unit),
                    verticalalignment='bottom', horizontalalignment='center'
                    )

            ylim = ax.get_ylim()

            def on_draw(e, ax=ax):
                a = 1
                print(a)
                pass

            self.spectrum_aux_lines = []

            for p in [pl, pr]:
                    ax.vlines(p[0], ymin=ylim[0], ymax=p[1],
                              linestyle='--', color='red')
                    # 当绘制了辅助线之后, y轴范围会改变, 必须重新将ylim设置回原来的值
                    ax.set_ylim(ylim) 

        self.fig = plt.figure()
        self.grid_spec = self.fig.add_gridspec(2, 2)

        # 将原始光谱数据绘制在图上, 以便手动设置去噪参数
        # plot_spectrum(self.axes[0], self.origin_spectrum, 'b', label='raw')

        # 去噪后
        ax1 = self.fig.add_subplot(self.grid_spec[0, :])
        plot_spectrum(ax1, self.spectrum, 'b+',
                      'after clear noise', aux_line=True)

        plot_spectrum(ax1, self.interpolated_spectrum, 'r', label='interpolated')
        
        # 绘制脉冲及其辅助线
        ax2 = self.fig.add_subplot(self.grid_spec[1, :])
        self.pulse.draw(n=4, ax=ax2, cl='r')
        draw_auxiliary_line(ax2, self.pulse.HMP_l, self.pulse.HMP_r, 'fs', 't')

        # # 绘制快速傅里叶变换得到的光谱
        # spec = fft.fft(np.sqrt(self.pulse.intensity))
        # freq = fft.fftfreq(n=len(self.pulse.intensity), d=self.delta_t)
        # ax3 = self.fig.add_subplot(self.grid_spec[2, :-1])
        # ax3.plot(freq, abs(spec)**2, 'r+')
        # ax3.set_title('power spectrum of fft')

        # ax4 = self.fig.add_subplot(self.grid_spec[2, -1])
        # ax4.plot( fft.ifft(
        #    abs(spec) ) )
        # ax4.set_title('ifft of amplitude spectrum')

        # ax5 = self.fig.add_subplot(self.grid_spec[3, 0])
        # ax5.plot(
        #     np.concatenate((np.zeros(30), abs(spec[:170])))
        # )
        # ax6 = self.fig.add_subplot(self.grid_spec[3, 1])
        # ax6.plot( fft.ifft(
        #     np.concatenate((np.zeros(30), abs(spec[:170])))
        #     ))

        plt.legend()
        plt.tight_layout()

        plt.show()

        return self.fig

    def update(self, *, mode: str, omega_min: float = None, omega_max: float = None, threshold: float = None, delta_t: float = .1):

        # 对光谱进行去噪处理, 结果存储于self.spectrum中
        # 将最终去噪阈值存储于self.clear_noise_final_threshold中
        self.clear_noise(mode='manual', omega_min=omega_min,
                         omega_max=omega_max, threshold=threshold)

        # 找到光谱的最小频率和最大频率
        self.find_omega_boundary()

        self.delta_t = delta_t
        # 对self.spectrum进行插值, 生成self.interpolated_spectrum
        self.interpolation()

        # self.pulse = Pulse(*self.ifft())
        
        # 对self.spectrum进行逆傅里叶变换
        self.pulse = Pulse(*self.ifft())

        self.draw()

if __name__ == '__main__':
    filepath = './data/gussian_spectrum_3fs_800nm.txt'

    spectrum = Spectrum(filepath)
