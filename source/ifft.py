from math import e
from numpy import matrixlib
from pathos.multiprocessing import ProcessPool as Pool
import os
import time
import numpy as np
import matplotlib.pyplot as plt


class Pulse:

    def __init__(self, pulse_width, t_min, t_max):

        self.pulse_width = pulse_width

        self.t_min = t_min

        self.t_max = t_max


class Spectrum_FFT:

    def __init__(self, file_path: str = None, line_sep='\t', spectrum: np.ndarray = None, show:bool = False):

        # 输入的光谱必须为功率谱

        self.threshold = 0.005  # 这个值会在后续计算中被改变

        self.threshold_increase_step = 0.0001

        self.data_deleted_per_step_threshold = 0.001

        self.delta_t = 0.01

        self.show = show

        if not file_path is None:

            self.file_path = file_path
            self.spectrum = self.get_spectrum(line_sep)

        elif not spectrum is None:

            self.spectrum = spectrum.copy()

            self.power_spectrum = self.spectrum.copy() # 先将功率谱保存，绘图时用功率光谱

            self.spectrum[:, 1] = np.sqrt(self.spectrum[:, 1]) # 先将功率谱转化为振幅谱

            self.raw_spectrum = self.spectrum.copy()


        else:
            raise ValueError('file_path and spectrum mush given one.')

        self.t, self.y = self.ifft()

        self.ty = np.array([self.t, self.y]).transpose()

        self.pulse = Pulse(*self.cal_FWHM())

        self.power_spectrum_pl, self.power_spectrum_pr, self.power_spectrum_width = self.cal_power_spectrum_FWHM()

    def get_spectrum(self, line_sep):

        def clear_noise(spectrum: np.ndarray):

            init_len = len(spectrum)
            spectrum = spectrum[spectrum[:, 1] >= self.threshold]
            last_len = len(spectrum)

            while True:

                self.threshold += self.threshold_increase_step

                next_len = len(spectrum[spectrum[:, 1] >= self.threshold])

                if (last_len - next_len) / (self.threshold_increase_step * init_len) >= self.data_deleted_per_step_threshold:

                    spectrum = spectrum[spectrum[:, 1] >= self.threshold]

                    last_len = next_len

                    continue

                else:

                    break

            return spectrum

        spectrum = np.loadtxt(self.file_path, delimiter=line_sep)

        # 归一化功率
        power = spectrum[:, 1]
        spectrum[:, 1] = power / np.max(power)

        self.raw_spectrum = spectrum.copy()  # 保存原始光谱数据

        self.power_spectrum = spectrum.copy() # 保存功率光谱(这个命名不准确，因为功率光谱应没有负值，但测量结果因噪声处理而产生了负值)，供绘图时使用

        # 去掉小于0的光谱数据
        spectrum = spectrum[spectrum[:, 1] >= 0]

        spectrum = clear_noise(spectrum.copy())

        spectrum[:, 1] = np.sqrt(spectrum[:, 1]) # 必须先转化成振幅谱

        return spectrum

    def fun_y(self, t: float):

        return self.ty[self.ty[:, 0] == t][0][1]

    def ifft(self) -> np.ndarray:

        def f(t: np.ndarray):

            def single_process_f(t: np.ndarray):


                y = np.zeros((len(t), ), dtype=np.complex128)

                omega_array = 2 * np.pi * 3 / self.spectrum[:, 0]
                
                # TODO 应尝试将并行用在该for循环上, 而非拆分numpy数组.
                for omega, power in zip(omega_array, self.spectrum[:, 1]):

                    # 波长的单位应为nm，时间的单位应为fs
                    y += power * np.e**(1j * omega * t * 100)

                return abs(y) ** 2
            
            with Pool(nodes = os.cpu_count()) as p:
                res = p.map(single_process_f, np.array_split(t, os.cpu_count()))

            return  np.concatenate(res)# 输出值是光功率


        def update_t_y(t_min, t_max, delta_t, t, y: np.ndarray):

            t_c_min, t_c_max = t[0], t[-1]

            if t_max < t_c_max:
                raise ValueError(
                    'current maximum value of t larger than expected maximum value of t')

            if t_min > t_c_min:
                raise ValueError(
                    'current minimum value of t smaller than expected minimum value of t')

            t_lp = np.linspace(t_min, t_c_min, int(


                #     y += power * np.e**(1j * omega * t * 100)
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

        delta_t = self.delta_t
        t_min = -10
        t_max = 10

        t = np.linspace(t_min, t_max, int((t_max - t_min) / delta_t))

        y = f(t)

        t, y = extend_y_to_half_maximum(t, y, delta_t)

        y /= max(y)

        return (t, y)

    def cal_FWHM(self):

        ty = np.array([self.t, self.y]).transpose()

        ty_l = ty[(ty[:, 0] <= 0) * (ty[:, 1] <= 0.5)]
        t_l = ty_l[ty_l[:, 1].argmax()][0]

        ty_r = ty[(ty[:, 0] >= 0) * (ty[:, 1] <= 0.5)]

        t_r = ty_r[ty_r[:, 1].argmax()][0]
        return (abs(t_r - t_l), t_l, t_r)

    def cal_power_spectrum_FWHM(self):

        pl = self.power_spectrum[self.power_spectrum[:, 1] >= 0.5][0]
        pr = self.power_spectrum[self.power_spectrum[:, 1] >= 0.5][-1]

        delta_lamda = pr[0] - pl[0]

        return (pl, pr, delta_lamda)


    def draw(self):

        def draw_auxiliary_line(ax: plt.Axes, pl, pr, unit:str, footnote:str):

            ax.annotate( '',
                        xy=pr, xycoords='data',
                        xytext=pl, textcoords='data',
                        arrowprops=dict(arrowstyle='<|-|>',
                                        color='red')
                        )

            ax.text(x=(pr[0] + pl[0]) / 2,
                    y=(pr[1] + pr[1]) / 2,
                    s='$\Delta_{%s} = %.2f %s$' % (footnote, abs(pr[0] - pl[0]), unit),
                    verticalalignment='bottom', horizontalalignment='center'
                    )

            ylim = ax.get_ylim()

            for p in [pl, pr]:
                ax.axvline(p[0], ymax=(pl[1] - ylim[0]) /
                           (ylim[1] - ylim[0]), linestyle='--', color='red')

        def plot_yt(ax: plt.Axes):

            ax.plot(self.t, self.y)

            ax.set_title('pulse')
            ax.set_xlabel('$t(fs)$')
            ax.set_ylabel('$intensity$')

        def plot_power_spectrum(ax: plt.Axes):

            ax.plot(self.power_spectrum[:, 0],
                    self.power_spectrum[:, 1], label='power spectrum')

            draw_auxiliary_line(ax, self.power_spectrum_pl, self.power_spectrum_pr, 'nm', '\lambda')

            ax.axhline(self.threshold, color='red', label='threshold')

            ax.set_title('raw spectrum')

            ax.set_xlabel('$\lambda(nm)$')
 
            ax.set_ylabel('$power$')

            ax.legend()

        fig, axs = plt.subplots(2, 1)

        plot_power_spectrum(axs[0])

        plot_yt(axs[1])

        t = np.array([self.pulse.t_min, self.pulse.t_max])

        pl = (t[0], self.fun_y(t[0]))
        pr = (t[1], self.fun_y(t[1]))

        draw_auxiliary_line(axs[1], pl, pr, 'fs', 't')

        plt.subplots_adjust(hspace=0.5)

        plt.grid(True)

        if self.show:
            plt.show()


def generate_gussian_spectrum(tau_p, lamda_0):
    '''
        tau_p: 时域脉冲功率函数的FWHM
        lamda_0: 中心波长
    '''

    def gussian_spectrum(omega, omega_0, tau_p):

        pow_top = - (omega - omega_0) ** 2 * tau_p**2 * 10**4

        pow_bottom = 8 * np.log(2)

        amplitude = np.e ** (pow_top / pow_bottom)

        amplitude = amplitude / max(amplitude)

        return np.array([2 * np.pi * 3 / omega, amplitude ** 2]).transpose() # 平方输出功率光谱


    omega_0 = 2 * np.pi * 3 / lamda_0

    delta_omega = 4 * np.log(2) * 0.01 / tau_p

    omega_max = omega_0 + delta_omega * 2

    omega_min = omega_0 - delta_omega * 2

    lamda_max = 2 * np.pi * 3 / (omega_0 - delta_omega)
    lamda_min = 2 * np.pi * 3 / (omega_0 + delta_omega)

    delta_lamda = lamda_max - lamda_min

    d_omega = 0.00001

    return gussian_spectrum(np.linspace(
        omega_min, omega_max, 
        int((omega_max - omega_min) / d_omega)), omega_0 = omega_0, tau_p=tau_p)

start_time = time.perf_counter()

# spectrum = Spectrum_FFT('./spec.txt')

# spectrum.draw()

spectrum_gussian = Spectrum_FFT(spectrum=generate_gussian_spectrum(3, 800), show=True)

end_time = time.perf_counter()
print('time used: %.3fs' % (end_time - start_time))

spectrum_gussian.draw()

