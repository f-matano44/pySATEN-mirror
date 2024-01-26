import soundfile as sf
import time
import noisereduce as nr
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def sed(data: np.ndarray, samplerate: float) -> tuple[float, float]:
    _, _, start_s, end_s, _, _, _ = _sed_debug(data, samplerate)
    return start_s, end_s


def _sed_debug(data: np.ndarray, samplerate: float,
        pw_threshold_dB: int = -20,
        zc_window_length_s: float = 0.025,
        zc_hop_length_s: float = 0.01,
        zc_margin_s: float = 0.1
    ):
    # 定数
    nyq = 0.5 * samplerate
    f0_floor = 71  # ref: WORLD (by Masanori MORISE)
    f0_ceil = 800  # ref: WORLD (by Masanori MORISE)
    band_b, band_a = signal.butter(
        4, [f0_floor / nyq, f0_ceil / nyq], btype='band')
    zc_window_length = int(zc_window_length_s * samplerate)
    zc_hop_length = int(zc_hop_length_s * samplerate)
    zc_margin = int(zc_margin_s * samplerate) 
    high_pass_filter = signal.firwin(
        31, f0_ceil / nyq, pass_zero=False)

    # パワーで検出の基礎となる発話区間を推定
    x_nr = nr.reduce_noise(data, samplerate)
    x_nr_bpf = signal.filtfilt(band_b, band_a, x_nr)
    x_nr_bpf_normalized = x_nr_bpf / max(abs(x_nr_bpf))
    x_power_dB = 10 * np.log10((x_nr_bpf_normalized ** 2) + 1e-12)
    power_cond = lambda x: pw_threshold_dB < x
    start1 = np.where(power_cond(x_power_dB))[0][0] \
        if np.any(power_cond(x_power_dB)) else 0
    end1 = np.where(power_cond(x_power_dB))[0][-1] \
        if np.any(power_cond(x_power_dB)) else len(x) - 1

    # ゼロクロスのカウント
    x_nr_hpf = signal.filtfilt(high_pass_filter, [1.0], x_nr)
    zc = np.zeros(len(data))
    for i in range(len(data)):
        zc_start    = int(max(0, i - zc_window_length))
        zc_end      = int(min(i + zc_window_length, len(data)))
        target = x_nr_hpf[zc_start: zc_end]
        zc[i] = _count_zero_cross(target)
    zc_temp = zc[zc_window_length: len(zc) - zc_window_length]
    zc_threshold = (max(zc_temp) + min(zc_temp)) / 2
    # start 側をスライド
    start2 = 0
    for i in range(start1, -1, -1):
        if 0 < i and zc_threshold <= zc[i]:
            s = zc[i-zc_margin: i]
            j = [j for j, z in enumerate(s) if z <= zc_threshold]
            if j:  # is not Empty
                i -= s[max(j)]
            else:
                start2 = i
                break
    # end 側をスライド
    end2 = len(data) - 1
    for i in range(end1, len(data), 1):
        if i < len(data) and zc_threshold <= zc[i]:
            s = zc[i: i+zc_margin]
            j = [j for j, z in enumerate(s) if z <= zc_threshold]
            if j:  # is not Empty
                i += s[min(j)]
            else:
                end2 = i
                break

    start1_s = start1 / samplerate
    end1_s = end1 / samplerate
    start2_s = start2 / samplerate
    end2_s = end2 / samplerate

    return start1_s, end1_s, start2_s, end2_s, x_power_dB, zc, zc_threshold


def _count_zero_cross(a: np.ndarray) -> int:
    differences = np.diff(a)
    sign_changes = np.sign(differences)
    return np.sum(np.abs(np.diff(sign_changes)) == 2) // 2
