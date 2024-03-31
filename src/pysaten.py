import soundfile as sf
import time
import noisereduce as nr
from scipy import signal
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import statistics as stat


def vsed(data: np.ndarray, samplerate: float) -> tuple[float, float]:
    _, _, start_s, end_s, _, _, _, _\
        = _vsed_debug(data=data, samplerate=samplerate)
    return start_s, end_s


def _vsed_debug(
        data: np.ndarray, samplerate: float,
        win_length_s:    float = None,
        hop_length_s:    float = 0.01,
        pw_threshold:    float = 0.01,
        zc_threshold:    float = 0.5,
        zc_margin_s:     float = None,
        noise_seed:        int = 0
    ):
    win_length_s = win_length_s if win_length_s is not None else hop_length_s * 4
    zc_margin_s  = zc_margin_s  if zc_margin_s  is not None else win_length_s

    # constants
    nyq:        int = int(0.5 * samplerate)
    f0_floor:   int = 71   # ref: WORLD (by Masanori MORISE)
    f0_ceil:    int = 800  # ref: WORLD (by Masanori MORISE)
    win_length: int = int(win_length_s * samplerate)
    zc_margin:  int = int(zc_margin_s * samplerate) 
    hop_length: int = int(hop_length_s * samplerate)
    band_b, band_a  = signal.butter(4, [f0_floor / nyq, f0_ceil / nyq], btype='band')
    high_b          = signal.firwin(31, f0_floor / nyq, pass_zero=False)
    normalize       = lambda a: (a - a.min()) / (a.max() - a.min())

    # Add white noise
    np_random = default_rng(noise_seed)
    white_noise = np_random.normal(0, 1, len(data))
    data = _add_noise_to_signal(data, white_noise, 20)
    data = data if np.all(np.abs(data) <= 1) else data / max(abs(data))

    # POWER
    x_nr     = nr.reduce_noise(data, samplerate)
    x_nr_bpf = signal.filtfilt(band_b, band_a, x_nr)
    x_power  = np.zeros(int(np.ceil(len(x_nr_bpf) / hop_length)))
    for i in range(len(x_power)):
        idx        = i * hop_length
        pw_start   = int(max(0, idx - (win_length / 2)))
        pw_end     = int(min(idx + (win_length / 2), len(data) - 1))
        target     = x_nr_bpf[pw_start: pw_end]
        x_power[i] = sum(target ** 2) / len(target)
    x_power      = normalize(x_power)
    x_power_mean = stat.mean(x_power)
    power_check  = lambda x: pw_threshold < x
    start1:  int = np.where(power_check(x_power))[0][0] \
        if np.any(power_check(x_power)) else 0
    end1:    int = np.where(power_check(x_power))[0][-1] \
        if np.any(power_check(x_power)) else len(x_power) - 1

    start1_s = max(0,                   start1 * hop_length_s)
    end1_s   = min(end1 * hop_length_s, len(data) / samplerate)

    # ZERO-CROSS
    x_nr_hpf = signal.filtfilt(high_b, [1.0], x_nr)
    zc       = np.zeros(int(np.ceil(len(data) / hop_length)))
    for i in range(len(zc)):
        idx      = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end   = int(min(idx + (win_length / 2), len(data) - 1))
        target   = x_nr_hpf[zc_start: zc_end]
        zc[i]    = _count_zero_cross(target) / len(target)
    zc           = normalize(zc)
    ## start 側をスライド
    start2 = 0
    for i in range(_my_round(start1_s / hop_length_s), -1, -1):
        if 0 < i and zc_threshold <= zc[i]:
            zc_temp_start = max(0, i-int(zc_margin/hop_length))
            zc_temp = zc[zc_temp_start: i]
            zc_idx = [j for j, z in enumerate(zc_temp) if z <= zc_threshold]
            if 0 < len(zc_idx):
                i -= len(zc_temp) - max(zc_idx)
            else:
                start2 = i
                break
    ## end 側をスライド
    end2 = len(zc)
    for i in range(_my_round(end1_s / hop_length_s), len(zc), 1):
        if i < len(data) and zc_threshold <= zc[i]:
            zc_temp = zc[i: i+int(zc_margin/hop_length)]
            j = [j for j, z in enumerate(zc_temp) if z <= zc_threshold]
            if j:  # is not Empty
                i += min(j)
            else:
                end2 = i
                break

    start2_s = max(0                  , start2 * hop_length_s)
    end2_s   = min(end2 * hop_length_s, len(data) / samplerate)

    temporal_positions = np.linspace(0, len(zc) * hop_length_s, len(zc))
    return start1_s, end1_s, start2_s, end2_s, \
        temporal_positions, x_power, zc, zc_threshold


def _add_noise_to_signal(signal, noise, desired_snr_db):
    """ 指定されたSNRでノイズを信号に追加する """
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(signal_power / (desired_snr_linear * noise_power))
    noise_adjusted = noise * scaling_factor
    return signal + noise_adjusted


def _count_zero_cross(a: np.ndarray) -> int:
    sign = np.sign(a)[a != 0 & ~np.isnan(a)]
    return np.sum(np.abs(np.diff(sign)) != 0)


def _my_round(a: float) -> int:
    return int(np.floor(a + 0.5))
