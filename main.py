import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)

# 信号参数
amplitude = 1
width = 0.1  # 脉冲宽度
period = 1  # 信号周期

# 设置新的采样率
new_sampling_rate = 120 
# 采样率太低会出现混叠效应，因此设置在120
# 尽管奈奎斯特采样定理要求2倍，但实测100倍以下可能出现混叠
num_samples = int(new_sampling_rate * period)

# 生成新的时间轴和信号
t = np.linspace(0, period, num_samples, endpoint=False)
signal = amplitude * (t % period < width)

def pad_to_power_of_two(x):
    N = len(x)
    if np.log2(N) % 1 > 0:  # 检查N是否为2的幂
        next_power_of_two = 2**int(np.ceil(np.log2(N)))
        padded_x = np.zeros(next_power_of_two)
        padded_x[:N] = x
        return padded_x
    else:
        return x

def fft(x):
    x = pad_to_power_of_two(x)
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    # W_nk = exp(-2j * pi * k / N)
    # T是FFT的奇数项
    T = [ np.exp(-2j * np.pi * k / N)  * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 绘制时域波形
axs[0, 0].plot(t, signal)
axs[0, 0].set_title('时域波形', fontproperties=my_font)
axs[0, 0].set_xlabel('时间', fontproperties=my_font)
axs[0, 0].set_ylabel('幅度', fontproperties=my_font)
axs[0, 0].grid(True)

# 计算并绘制频谱图
N = 1024  # FFT长度

zero_padded_signal = np.pad(signal, (0, N - len(signal)), 'constant')

fft_result = fft(zero_padded_signal)

# FFT结果的幅度谱。对FFT结果取绝对值，并除以FFT长度N，以获得每个频率分量的幅度值。
fft_magnitude = np.abs(fft_result) / N
# 保留正一半的FFT结果，因为实际分量大小是正负的和，因此需要乘以2
fft_magnitude = fft_magnitude[:N // 2] * 2  
# 计算频率轴
freqs = np.fft.fftfreq(N, d=t[1] - t[0])[:N // 2]
freqs = freqs[:N//2]
fft_magnitude = fft_magnitude[:N//2]

# 截取
if len(freqs) > len(fft_magnitude):
    freqs = freqs[:len(fft_magnitude)]
axs[0, 1].stem(freqs, fft_magnitude)
axs[0, 1].set_title('频谱图', fontproperties=my_font)
axs[0, 1].set_xlabel('频率', fontproperties=my_font)
axs[0, 1].set_ylabel('幅度', fontproperties=my_font)
# axs[0, 1].set_xlim(0, 100) 
axs[0, 1].grid(True)


# 绘制不同FFT长度的频谱图
# N越大，分辨率越强，栅栏效应越弱，能看到更多的谱线
# 分辨率越强，每个频率分量之间的间隔越小，信号越窄、越集中
for idx, N in enumerate([256, 512, 1024, 2048]):
    fft_result = fft(signal)
    fft_magnitude = np.abs(fft_result) / N
    fft_magnitude = fft_magnitude[:N // 2] * 2
    freqs = np.fft.fftfreq(N, d=t[1] - t[0])[:N // 2]
    print(len(freqs), len(fft_magnitude))
    # 截取
    if len(freqs) > len(fft_magnitude):
        freqs = freqs[:len(fft_magnitude)]
    axs[1, 0].stem(freqs, fft_magnitude, label=f'FFT Length = {N}', linefmt=f'C{idx}', markerfmt=f'C{idx}o')

axs[1, 0].set_title('频谱图 - 不同FFT长度对比', fontproperties=my_font)
axs[1, 0].set_xlabel('频率', fontproperties=my_font)
axs[1, 0].set_ylabel('幅度', fontproperties=my_font)
axs[1, 0].legend()
axs[1, 0].grid(True)

# 优化后的频谱图
# 加窗，汉明窗，非矩形窗，减少频谱泄露
# 其他也可，如汉宁窗、布莱克曼窗等
windowed_signal = signal * np.hamming(len(signal))
# 末端补零，保证最小记录点数N达到1024
# 实际上，N应满足T1/T
# 补零可以在保持原频谱形状不变的情况下，使谱线变密
zero_padded_signal = np.pad(windowed_signal, (0, N - len(windowed_signal)), 'constant')
fft_result = fft(zero_padded_signal)
fft_magnitude = np.abs(fft_result) / len(zero_padded_signal)
fft_magnitude = fft_magnitude[:len(zero_padded_signal) // 2] * 2
freqs = np.fft.fftfreq(len(zero_padded_signal), d=t[1] - t[0])[:len(zero_padded_signal) // 2]
axs[1, 1].stem(freqs, fft_magnitude)
axs[1, 1].set_title('优化后的频谱图', fontproperties=my_font)
axs[1, 1].set_xlabel('频率', fontproperties=my_font)
axs[1, 1].set_ylabel('幅度', fontproperties=my_font)
axs[1, 1].grid(True)

# 调整布局
plt.tight_layout()
plt.suptitle(f'fm={1/period}Hz,fs={new_sampling_rate}Hz', fontproperties=my_font)
plt.show()
