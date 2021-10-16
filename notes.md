## 对于离散傅里叶变换中频率的思考
对于连续傅里叶变换,如果输入的原脉冲 $f(t)$ 为实函数,则对应的傅里叶变换光谱存在关系:
$$F(-\omega)=F^*(\omega)$$
从连续傅里叶变换到离散傅里叶变换意味着:
$$
f(t) \rightarrow f[k],  (t_k = kT)
$$

$$
F(\omega) \rightarrow F[n], (\omega_n = \frac{2 \pi}{NT}n)
$$
上式中, $T$为采样间隔, $N$为采样数.$n, k = 0, 1, 2, ...N-1$.
根据奈奎斯特采样定理, 对于一个时域采样间隔为T的函数,其频谱应带限于$(0, \frac{1}{2T})$,即奈奎斯特采样频率:
$$ \omega_{nquist}= \frac{2 \pi }{2 T}=\frac{2 \pi }{NT}\frac{N}{2}=\omega_{\frac{N}{2}}$$

对于$n' > \frac{N}{2}$的$\omega_{n'}$来说, 存在$n< \frac{N}{2}$,使得$n'=N-n$,
则有:
$$
e^{-j\omega_n t_k}=e^{-j \frac{2 \pi}{NT}n k T}=e^{-j \frac{2 \pi}{N}n k }
$$


$$e^{-j \omega_{N-n}t_k}=e^{-j \frac{2 \pi}{N}(N-n)k}=e^{-j2\pi k}e^{j \frac{2\pi}{N}nk}=e^{-j \frac{2\pi}{N}(-n)k} \tag{1}$$
即$\omega_{n'}= \frac{2\pi}{N}(-n)=-\omega_n$.

在`numpy, scipy`中,将$n'>\frac{N}{2}$对应的频率$\omega_{n'}$称为负频率,即$\omega_{n'}=-\omega_n$.

对于实函数$f(t)$, 其离散傅里叶变换为:
$$F[n]=\sum_{k=0}^{N-1}f[k]e^{-j\frac{2\pi}{N}nk}$$
而
$$F[N-n]=\sum_{k=0}^{N-1}f[k]e^{-j\frac{2\pi}{N}(N-n)k}$$
根据式$(1)$

$$
\begin{aligned}
    F[N-n]&=\sum_{k=0}^{N-1}f[k]e^{-j\frac{2\pi}{N}(-n)k} \\
        &=\sum_{k=0}^{N-1}f[k]e^{j\frac{2\pi}{N}nk}\\
        &=\sum_{k=0}^{N-1}f[k](e^{-j\frac{2\pi}{N}nk})^*\\
        &=(\sum_{k=0}^{N-1}f[k]e^{-j\frac{2\pi}{N}nk})^*\\
        &=F^*[n]
\end{aligned}

$$

这说明**对于实函数**,当$n'>\frac{N}{2}$时,频率$\omega_{n'}$并没有给出任何新的信息(因为两者模值相等).但若f(t)为复函数,则并无此结论.

### 为什么实函数光谱中会存在负频率项且和正频率项的模值相等呢?
    
一个实函数可以表示为:
$$
f(t) = \frac{a_0}{2} + \sum_n a_n \cos{\omega_n t} + b_n \sin{\omega_n t}

$$
其中的正弦分量和余弦分量可分别写为:
$$
\begin{aligned}
    \cos{\omega t} &= \frac{e^{i \omega t} + e^{ - i \omega t}}{2} \\
    \sin{\omega t} &= \frac{e^{i \omega t} - e^{ - i \omega t}}{2}
\end{aligned}
$$
这便是$-\omega$的来源.

## 对光谱进行逆离散傅里叶变换时应该注意的问题
1. `numpy,scipy`中,在进行离散傅里叶变换和逆变换时,均采用$f$而非$\omega$来表示频率,但在理论处理过程中,却始终采用了$\omega$,这点需要注意.
2. 因$\Delta_{\omega}=\frac{ 2 \pi }{N T}$, 在处理过程中, 频率间隔$\Delta_{\omega}$由光谱数据的最小频率间隔决定, 而T则为所需要的光谱分辨率, 那么N将有这两者决定,即$N = \frac{2 \pi}{\Delta_{\omega} T}$, 考虑到N为整数, 即$N = [\frac{2 \pi}{\Delta_{\omega} T}]$, 故真实时间分辨率$T'=\frac{2 \pi}{N \Delta_\omega}$
3. 奈奎斯特采样频率$\omega_{Nq}=\frac{\pi}{T}$由时间分辨率单独决定, 与 N 无关, 与$\Delta_{\omega}$无关.在光谱领域内,若要求时间分辨率$T=0.01fs$,则$\omega_{Nq}\approx314\times10^{15} = 3.14 \times 10^{17}$, 对于一般的实际光谱, $800nm$对应的圆频率为$\omega=\frac{2\pi c}{800}=0.02 \times 10 ^ {17}$, 相应的, $200nm$对应的圆频率为$\omega=0.08\times 10 ^ {17} \llless \omega_{Nq}$, 因此, 实际光谱在进行离散傅里叶变换时**永远在奈奎斯特频率的左半部分**, 对于大于实际光谱频率的频率成分,全部补零.为防止实际设置的$T$过大,导致实际频谱成分丢失($\omega > \frac{2\pi}{T}$的频谱将会被舍去.)
4.  逆傅里叶变换后函数周期$P=NT=\frac{2\pi}{\Delta_\omega T} T = \frac{2\pi}{\Delta_\omega}$,从此式可以看出,函数周期仅由$\Delta_\omega$决定.
5.  考虑到:
    $$f(t)=\int_{-\infty}^{+\infty}F(\omega)e^{j\omega  t}\ \textrm{d}\omega$$

    在实际计算过程中,考虑到计算机存储的精度有限,用到的频率和时间的值为$\omega_u$和$t_u$, 且令
    $$\omega = \omega_u \times 10 ^{17}$$   
    为保持连续傅里叶变换和离散傅立叶变换形式不变,即须使$\omega_u t_u=\omega t$, 则有:
    $$t = \frac{\omega_u}{\omega}t_u=t_u \times 10^{-17}$$
    因此作离散傅立叶逆变换之后, 时间的数量级为$10^{-17}$.**在处理过程中需要注意**:
    
    - 计算N时, $N = [\frac{2 \pi}{\Delta_{\omega} T}] = [\frac{2 \pi}{\Delta_{\omega_u} T_u}]=[\frac{2 \pi}{\Delta_{\omega_u} T_{fs} \times 100}]$,其中$T_{fs}$为函数接口指定 $T$ 时采用的量, 单位为$fs$, 数量级为$10^{-15}$(因为这样更方便).
    - 若采用上述公式, 此时N对应的真实时间分辨率$T'=\frac{2 \pi}{N \Delta_\omega}$的数量级为$10^{-17}$, 但在后续处理过程中仍希望采用$fs$为时间单位, 因此有:$T'_{fs} = T' * 0.01$, 此时$T'_{fs}$的单位为$fs$, 数量级为$10^{-15}$.
6. 考虑到离散傅里叶变换结束后, 得到的实际上是一个周期为$P=NT$的函数, 变换后得到的函数$f[k]$的时间范围$t\in [0, (N-1)T]$, 若要计算脉宽, 需要$t<0$时的函数, 此时只需将函数$f[k]$复制到$t\in[-NT, -T)$即可.