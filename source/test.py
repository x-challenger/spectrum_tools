import numpy as np
from scipy.fft import *
N = 20
freq = fftfreq(N, 0.1) * 6.28
freq[N//2]

a = np.array([True, False])
b = np.array([True, True])

a * b

