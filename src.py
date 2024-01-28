import soundfile as sf
import time
import noisereduce as nr
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def sed(data: np.ndarray, samplerate: float) -> tuple[float, float]:
    _, _, start_s, end_s, _, _, _ = _sed_debug(data, samplerate)
    return start_s, end_s


def _sed_debug(
        data: np.ndarray, samplerate: float,
        win_length_s: float = 0.04,
        hop_length_s: float = 0.01,
        pw_threshold_dB: int = -20,
        zc_margin_s: float = 0.1
    ):
    # 定数
    nyq: int = int(0.5 * samplerate)
    f0_floor: int = 71  # ref: WORLD (by Masanori MORISE)
    f0_ceil: int = 800  # ref: WORLD (by Masanori MORISE)
    win_length: int = int(win_length_s * samplerate)
    hop_length: int = int(hop_length_s * samplerate)
    zc_margin:  int = int(zc_margin_s * samplerate) 
    band_b, band_a = signal.butter(
        4, [f0_floor / nyq, f0_ceil / nyq], btype='band')
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

    start1_s = start1 / samplerate
    end1_s = end1 / samplerate

    # ゼロクロスのカウント
    x_nr_hpf = signal.filtfilt(high_pass_filter, [1.0], x_nr)
    zc = np.zeros(len(data))
    for i in range(len(data)):
        zc_start    = int(max(0, i - (win_length / 2)))
        zc_end      = int(min(i + (win_length / 2), len(data) - 1))
        target = x_nr_hpf[zc_start: zc_end]
        zc[i] = _count_zero_cross(target) / len(target)
    zc_threshold = (max(zc) + min(zc)) / 2
    # start 側をスライド
    start2 = 0
    for i in range(start1, -1, -hop_length):
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
    for i in range(end1, len(data), hop_length):
        if i < len(data) and zc_threshold <= zc[i]:
            s = zc[i: i+zc_margin]
            j = [j for j, z in enumerate(s) if z <= zc_threshold]
            if j:  # is not Empty
                i += s[min(j)]
            else:
                end2 = i
                break

    start2_s = start2 / samplerate
    end2_s = end2 / samplerate

    return start1_s, end1_s, start2_s, end2_s, x_power_dB, zc, zc_threshold


def _count_zero_cross(a: np.ndarray) -> int:
    differences = np.diff(a)
    sign_changes = np.sign(differences)
    return np.sum(np.abs(np.diff(sign_changes)) == 2) // 2
