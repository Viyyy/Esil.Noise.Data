import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def replace_audio_segment(input_path, output_path, start_time, end_time):
    # 加载音频文件
    y, sr = librosa.load(input_path, sr=None)

    # 将时间转换为样本数
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # 提取指定时间段的音频片段
    segment = y[start_sample:end_sample]

    # 创建一个新的音频数组
    new_audio = np.zeros_like(y)

    # 将音频的其他部分替换为提取的片段
    segment_length = len(segment)
    for i in range(0, len(new_audio), segment_length):
        new_audio[i:i+segment_length] = segment[:min(segment_length, len(new_audio) - i)]

    # 保存新的音频文件
    sf.write(output_path, new_audio, sr)
    
def find_intervals(arr, k):
    if list(arr)==0:
        return 0, []

    intervals = []
    current_interval = [arr[0]]

    for i in range(1, len(arr)):
        if abs(arr[i] - arr[i - 1]) < k:
            current_interval.append(arr[i])
        else:
            intervals.append(current_interval)
            current_interval = [arr[i]]

    intervals.append(current_interval)

    return len(intervals), intervals

def get_max_length_interval(intervals):
    max_length = 0
    start = 0 
    end = 0
    for interval in intervals:
        if interval[-1] - interval[0] > max_length:
            max_length = interval[-1] - interval[0]
            start = interval[0]
            end = interval[-1]
    return start, end

def plot_intervals(intervals, ax, start=0, end=None):
    # fig, ax = plt.subplots()

    for i, interval in enumerate(intervals):
        y = [i] * len(interval)
        ax.plot(interval, y, marker='o', linestyle='-', label=f'Event {i+1}')

    ax.set_yticks(range(len(intervals)))
    ax.set_yticklabels([f'E{i+1}' for i in range(len(intervals))])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Events')
    # 设置x轴的范围
    ax.set_xlim(start, end)
    ax.title.set_text('Noise Events')
    # 设置y轴的范围
    ax.legend()
    # plt.show()
    
# 加载音频文件
filename = r"警铃_0.wav"
y, sr = librosa.load(filename)

fig,axes = plt.subplots(3,1, figsize=(6, 8))
min_ = np.min(y)
max_ = np.max(y)
# 计算短时能量
frame_length = 1024
hop_length = 512
energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y), hop_length)
])

# 转换为分贝
ref_energy = np.max(energy)
# ref_energy = np.min(energy)
ref_energy = 2 * 10**(-5) * ref_energy
energy_db = 10 * np.log10(energy / ref_energy)

# 时间轴
times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)

# 绘制图像
# plt.figure(figsize=(14, 5))
axes[0].plot(times, energy_db, label='Calculated dB from audio')

# 设置label
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Energy (dB)')

# 设置axis的标题
axes[0].title.set_text('Instantaneous Energy vs Time')
axes[0].set_xlim(0, len(y)/sr)
axes[0].legend()
# plt.show()

# 设定能量阈值
energy_threshold = np.mean(energy) * 0.5

# 找到静音部分
silent_frames = np.where(energy < energy_threshold)[0]
silent_times = librosa.frames_to_time(silent_frames, sr=sr, hop_length=hop_length)

# 找到声音集中的部分
sound_frames = np.where(energy >= energy_threshold)[0]
sound_times = librosa.frames_to_time(sound_frames, sr=sr, hop_length=hop_length)

# 可视化结果
# plt.figure(figsize=(14, 5))
axes[1].plot(librosa.times_like(energy, sr=sr, hop_length=hop_length), energy, label='Energy')
axes[1].axhline(y=energy_threshold, color='r', linestyle='--', label='Energy Threshold')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Energy')
axes[1].title.set_text('Energy of the audio signal')
axes[1].set_xlim(0, len(y)/sr)
# axes[1].title('Energy of the audio signal')
axes[1].legend()
# plt.show()

# # 打印结果
# print("静音部分时间点（秒）:", silent_times)
# print("声音集中的部分时间点（秒）:", sound_times)

time_delta = 0.5 # 定义区间长度（秒）

# 找到区间并绘图
num_intervals, intervals = find_intervals(sound_times, time_delta)
start, end = get_max_length_interval(intervals)
print(f'Number of intervals: {num_intervals}')
print('Intervals:', intervals)
replace_audio_segment(filename, output_path='output_audio.wav', start_time=start, end_time=end)

plot_intervals(intervals, axes[2], end=len(y)/sr)

fig.tight_layout()
plt.show()



a = 1
