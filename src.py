import soundfile as sf
import time
import noisereduce as nr
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def _main():
    x, fs   = sf.read("noise_audio.wav")
    cx, cfs = sf.read("clean.wav")

    start_time = time.time()

    # パワーで検出の基礎となる発話区間を推定
    x_nr = nr.reduce_noise(x, fs)
    NYQ = 0.5 * fs
    b, a = signal.butter(4, [71 / NYQ, 800 / NYQ], btype='band')
    x_nr_bpf = signal.filtfilt(b, a, x_nr)
    x_nr_bpf /= max(abs(x_nr_bpf))
    x_nr_bpf_power_dB = 10 * np.log10((x_nr_bpf ** 2) + 1e-12)
    trimming_threshold_dB = -20  # マジックナンバー
    start = 0
    for i in range(len(x_nr_bpf_power_dB)):
        if trimming_threshold_dB < x_nr_bpf_power_dB[i]:
            start = i
            break
    end = len(x_nr_bpf_power_dB) - 1
    for i in range(len(x_nr_bpf_power_dB)-1, -1, -1):
        if trimming_threshold_dB < x_nr_bpf_power_dB[i]:
            end = i
            break

    # ゼロクロスのカウント
    ZEROCROSS_WINDOW_SIZE = int(0.005 * fs)  # マジックナンバー
    b, a = signal.butter(4, 800 / NYQ, btype='high')
    x_nr_hpf = signal.filtfilt(b, a, x_nr)
    ZC_MARGIN = int(0.1 * fs)  # マジックナンバー
    zc = np.zeros(len(x))
    for i in range(len(x)):
        zc_win = ZEROCROSS_WINDOW_SIZE / 2
        zc_start    = int(max(0, i - zc_win))
        zc_end      = int(min(i + zc_win, len(x)))
        target = x_nr_hpf[zc_start: zc_end]
        zc[i] = _count_zero_cross(target)
    zc_temp = zc[ZEROCROSS_WINDOW_SIZE: len(zc) - ZEROCROSS_WINDOW_SIZE]
    THRESHOLD = (max(zc_temp) + min(zc_temp)) / 2
    # start 側をスライド
    start2 = 0
    for i in range(start, -1, -1):
        if 0 < i and THRESHOLD <= zc[i]:
            s = zc[i-ZC_MARGIN: i]
            j = [j for j, z in enumerate(s) if z <= THRESHOLD]
            if j:  # is not Empty
                i -= s[max(j)]
            else:
                start2 = i
                break
    # end 側をスライド
    end2 = len(x) - 1
    for i in range(end, len(x), 1):
        if i < len(x) and THRESHOLD <= zc[i]:
            s = zc[i: i+ZC_MARGIN]
            j = [j for j, z in enumerate(s) if z <= THRESHOLD]
            if j:  # is not Empty
                i += s[min(j)]
            else:
                end2 = i
                break

    end_time = time.time()
    print(f"RTF: {(end_time - start_time) / (len(x) / fs)}")

    # time -> array
    speech_section = np.zeros(len(x))
    speech_section[start: end] = 1
    speech_section2 = np.zeros(len(x))
    speech_section2[start2: end2] = 1

    # drow graph
    x_t = np.linspace(0, len(x) / fs, len(x))
    cx_t = np.linspace(0, len(cx) / cfs, len(cx))
    plt.fill_between(x_t, -1, 1, where=speech_section,  color='gray', alpha=0.7)
    plt.fill_between(x_t, -1, 1, where=speech_section2, color='gray', alpha=0.3)
    plt.plot(x_t, x, color="black")
    plt.plot(cx_t, cx)
    plt.plot(x_t, zc * max(abs(x)) / max(abs(zc)))
    plt.plot(x_t, np.ones(len(x)) * THRESHOLD * max(abs(x)) / max(abs(zc)))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('pySATEN')  # StArT-and-End-detectioN
    plt.ylim(-max(abs(x))-0.01, max(abs(x))+0.01)
    plt.show()


def _count_zero_cross(a: np.ndarray) -> int:
    differences = np.diff(a)
    sign_changes = np.sign(differences)
    return np.sum(np.abs(np.diff(sign_changes)) == 2) // 2


if __name__ == "__main__":
    _main()
