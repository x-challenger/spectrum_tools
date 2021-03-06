from typing import List
from matplotlib import pyplot as plt
import numpy as np
from numpy import cos, fft, ndarray, ndindex, pi
from scipy import interpolate, fft
from scipy.interpolate import interp1d

class Pulse:
    
    def __init__(self, t, intensity) -> None:

        self.t, self.intensity = t, intensity

        self.intensity /= max(self.intensity)

        self.FWHM = self.cal_FWHM()

    def cal_FWHM(self):

        ty = np.column_stack([self.t, self.intensity])

        t_upper_half = ty[ty[:, 1] >= .5][:, 0]

        self.t_left_half, self.t_right_half = t_upper_half[0], t_upper_half[-1]

        return abs(self.t_right_half - self.t_left_half)

    def draw(self, ax:plt.Axes, cl:str='', label:str='pulse'):

        ax.plot(self.t, self.intensity, cl, label=label)

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
            光谱数据中的频率分量, 单位为PHz( 10^(15)Hz ) default None
        delimiter : string, optional
            光谱文件分隔符 default '\t'

        注:
        1. filepath, lamda, omega至少给定一个, 否则报错.
        2. 考虑到计算机存储数的精度,因此波长单位采用nm, 频率单位采用PHz.
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
            self.spectrum[:, 0] = self.lambda_omega_converter(self.spectrum[:, 0])

        self.spectrum[:, 1] /= max(self.spectrum[:, 1])  # 光谱功率归一化

        self.origin_spectrum = self.spectrum.copy()  # 保存原始光谱数据

        self.update(mode='auto')

    def lambda_omega_converter(self, x):
        """将x对应的波长和频率进行相互转化

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

    def omega2lambda(self):
        pass

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

    def clear_noise(self, *, mode: str, omega_min: float = None, omega_max: float = None, threshold: float = None):
        """清除光谱数据中的噪声数据

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
        该函数会将去噪结果存储在self.spectrum中, 将最终阈值存储在self.clear_noise_final_hreshold中
        """

        spectrum = self.origin_spectrum  # 每次清除噪声总是从原始数据开始

        if mode == 'auto':
            # 清除噪声

            threshold = 0.005  # 起始阈值
            threshold_increase_step = 0.001

            # 去除噪音的控制参数 阈值增加所被删除的数据个数除以步长应该大于该阈值, 否则停止阈值增加(去除噪音完成)
            minimum_data_deleted_per_step = 478000

            spectrum = spectrum[spectrum[:, 1] >= threshold]
            last_len = len(spectrum)

            while True:

                threshold += threshold_increase_step

                new_spectrum = spectrum[spectrum[:, 1] >= threshold]
                next_len = len(new_spectrum)

                if (last_len - next_len) / threshold_increase_step >= minimum_data_deleted_per_step:

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

            try:
                spectrum = spectrum[spectrum[:, 1] >= threshold]
            except TypeError:
                pass

        self.spectrum = spectrum
        self.clear_noise_final_threshold = threshold

    def find_omega_boundary(self):
        """找到光谱频率的最小值和最大值,并将其存储在self.omega_min和omega_max中.
        """

        self.omega_min, self.omega_max = [fun(self.spectrum[:, 0]) for fun in [min, max]]

    def interpolation(self):
        """对self.spectrum进行插值, 生成self.interpolated_spectrum
        在生成的插值数组中, 频率间隔相等, 等于self.spectrum的最小频率间隔
        """
        def find_min_delta_omega(oemga:ndarray):
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

        def generate_equal_space_array(start:float, step:float, N: int) :
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

        flatten = lambda array: array.flatten() # 将数组变成一维

        omega, power = map(flatten, np.hsplit(self.spectrum, 2))
        interp_fun = interp1d(omega, power, kind='linear', bounds_error=False, fill_value=(0, 0))

        min_delta_omega = find_min_delta_omega(omega)

        N = int(self.omega_max / min_delta_omega)

        omega_new = generate_equal_space_array(0, min_delta_omega, N + 1)

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

        self.amp_spectrum = np.sqrt(abs(self.interpolated_spectrum))

        # 根据离散傅里叶变换的定义, 0和奈奎斯特采样频率处的振幅对原函数的贡献为1/NF[n], 中间频率分量的贡献为2/NF[n],
        # 但光栅光谱仪测出来的光谱值, 应指的是实际脉冲中频率分量的贡献F[n], 若要使这两者相同, 则应对光栅光谱仪测出的数据除以2.
        if len(self.amp_spectrum) // 2 == 0: # n even
            self.amp_spectrum[1:-1] /= 2
        else:
            self.amp_spectrum[1:-1] /= 2

        amp = fft.irfft(self.amp_spectrum[:, 1])
   
        delta_omega = abs(self.interpolated_spectrum[:, 0][1] - self.interpolated_spectrum[:, 0][0])
        delta_t = 2 * pi / (max(self.interpolated_spectrum[:, 0]) + delta_omega)

        # self.amp_spectrum和self.origin_spectrum的最大频率值严重不符
        t = np.arange(0, len(amp), 1) * delta_t

        return (t, abs(amp)**2)
    
    def draw(self):

        def plot_spectrum(ax:plt.Axes, spectrum, cl, label):
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
            ax.plot(self.lambda_omega_converter(spectrum[:, 0]), spectrum[:, 1], cl, label = label)


        self.fig, self.axes= plt.subplots(2, 1)

        # 将原始光谱数据绘制在图上, 以便手动设置去噪参数
        plot_spectrum(self.axes[0], self.origin_spectrum, 'b', label='raw')

        # 去噪后
        plot_spectrum(self.axes[0], self.spectrum, 'r', 'after clear noise')

        # 插值后 调试用 实际不必展示
        plot_spectrum(self.axes[0], self.interpolated_spectrum, 'y*', 'interpolated')

        self.pulse.draw(self.axes[1], cl='r+')

        plt.legend()

        self.fig.show()        

        return self.fig
        

    def update(self, *, mode: str, omega_min: float = None, omega_max: float = None, threshold: float = None):

        self.clear_noise(mode=mode, omega_min=omega_min,
                         omega_max=omega_max, threshold=threshold)

        # 找到光谱的最小频率和最大频率
        self.find_omega_boundary()

        self.interpolation()

        self.pulse = Pulse(*self.ifft())

        self.draw()


filepath = 'D:\space\work\spectrum_tools\data\gussian_spectrum_3fs_800nm.txt'


spectrum = Spectrum(filepath)
# %%
