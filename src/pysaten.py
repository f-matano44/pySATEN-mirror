import soundfile as sf
import time
import noisereduce as nr
from scipy import signal
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from librosa.feature import zero_crossing_rate, rms
import statistics as stat


def vsed(data: np.ndarray, samplerate: float) -> tuple[float, float]:
    _, _, start_s, end_s, _, _, _, _, _\
        = vsed_debug(
            data=data.copy(),
            samplerate=samplerate,
            rms_threshold=None,
            zcs_threshold=None
        )
    return start_s, end_s


def vsed_debug(
        data: np.ndarray, samplerate: float,
        win_length_s:  float = None,
        hop_length_s:  float = 0.01,
        rms_threshold: float = None,
        zcs_threshold: float = None,
        zcs_margin_s:  float = None,
        noise_seed:      int = 0
    ):
    win_length_s = win_length_s if win_length_s is not None else hop_length_s * 4
    zcs_margin_s = zcs_margin_s if zcs_margin_s is not None else win_length_s

    # constants
    nyq:        int = int(0.5 * samplerate)
    f0_floor:   int = 71   # ref: WORLD (by Masanori MORISE)
    f0_ceil:    int = 800  # ref: WORLD (by Masanori MORISE)
    win_length: int = int(win_length_s * samplerate)
    hop_length: int = int(hop_length_s * samplerate)
    zcs_margin: int = int(zcs_margin_s * samplerate) 
    band_b, band_a  = signal.butter(4, [f0_floor / nyq, f0_ceil / nyq], btype='band')
    high_b          = signal.firwin(31, f0_floor / nyq, pass_zero=False)
    normalize       = lambda a: (a - a.min()) / (a.max() - a.min())

    # preprocess
    ## add white noise
    np_random = default_rng(noise_seed)
    white_noise = np_random.normal(0, 1, len(data))
    data = _add_noise_to_signal(data, white_noise, 20)
    data = data if max(abs(data)) <= 1 else data / max(abs(data))
    ## reduce start and end
    oneside_win_length = int(win_length / 2)
    data[: oneside_win_length] = \
        data[:             oneside_win_length] * np.linspace(0, 1, oneside_win_length)
    data[len(data) - oneside_win_length: ] = \
        data[len(data) - oneside_win_length: ] * np.linspace(1, 0, oneside_win_length)

    # ROOT-MEAN-SQUARE
    x_nr     = nr.reduce_noise(data, samplerate)
    x_nr_bpf = signal.filtfilt(band_b, band_a, x_nr)
    x_rms    = normalize(rms(
        y=x_nr_bpf, frame_length=win_length, hop_length=hop_length)[0])
    rms_threshold = stat.mean(x_rms) if rms_threshold is None else rms_threshold
    rms_check = lambda x: rms_threshold < x
    start1: int = np.where(rms_check(x_rms))[0][0] \
        if np.any(rms_check(x_rms)) else 0
    end1:   int = np.where(rms_check(x_rms))[0][-1] \
        if np.any(rms_check(x_rms)) else len(x_rms) - 1

    start1_s = max(0,                   start1 * hop_length_s)
    end1_s   = min(end1 * hop_length_s, len(data) / samplerate)

    # ZERO-CROSS
    x_nr_hpf = signal.filtfilt(high_b, [1.0], x_nr)
    x_zcs = normalize(zero_crossing_rate(
        x_nr_hpf, frame_length=win_length, hop_length=hop_length)[0])
    zcs_threshold = stat.mean(x_zcs) if zcs_threshold is None else zcs_threshold
    ## start 側をスライド
    start2 = 0
    for i in range(_my_round(start1_s / hop_length_s), 0, -1):
        if 0 < i and zcs_threshold <= x_zcs[i]:
            zcs_temp_start = max(0, i-int(zcs_margin/hop_length))
            zcs_temp = x_zcs[zcs_temp_start: i]
            zcs_idx = [j for j, z in enumerate(zcs_temp) if z <= zcs_threshold]
            if 0 < len(zcs_idx):
                i -= len(zcs_temp) - max(zcs_idx)
            else:
                start2 = i
                break
    ## end 側をスライド
    end2 = len(x_zcs)
    for i in range(_my_round(end1_s / hop_length_s), len(x_zcs), 1):
        if i < len(data) and zcs_threshold <= x_zcs[i]:
            zcs_temp = x_zcs[i: i+int(zcs_margin/hop_length)]
            j = [j for j, z in enumerate(zcs_temp) if z <= zcs_threshold]
            if j:  # is not Empty
                i += min(j)
            else:
                end2 = i
                break

    start2_s = max(0                  , start2 * hop_length_s)
    end2_s   = min(end2 * hop_length_s, len(data) / samplerate)

    feats_timestamp = np.linspace(0, len(x_zcs) * hop_length_s, len(x_zcs))
    return start1_s, end1_s, start2_s, end2_s, \
        feats_timestamp, x_rms, rms_threshold, x_zcs, zcs_threshold


def _add_noise_to_signal(signal, noise, desired_snr_db):
    """ 指定されたSNRでノイズを信号に追加する """
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(signal_power / (desired_snr_linear * noise_power))
    noise_adjusted = noise * scaling_factor
    return signal + noise_adjusted


def _my_round(a: float) -> int:
    return int(np.floor(a + 0.5))
