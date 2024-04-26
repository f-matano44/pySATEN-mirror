import statistics as stat
from typing import Optional

import noisereduce as nr
import numpy as np
from librosa import resample
from librosa.feature import rms, zero_crossing_rate
from numpy.random import default_rng
from scipy.signal import filtfilt, firwin


def vsed(data: np.ndarray, samplerate: int) -> tuple[float, float]:
    _, _, start_s, end_s, _, _, _, _, _ = vsed_debug(
        data=data, samplerate=samplerate, rms_threshold=0.05, zcs_threshold=0.5
    )
    return start_s, end_s


def vsed_debug(
    data: np.ndarray,
    samplerate: int,
    win_length_s: Optional[float] = None,
    hop_length_s: float = 0.01,
    rms_threshold: Optional[float] = None,
    zcs_threshold: Optional[float] = None,
    margin_s: Optional[float] = None,
    noise_seed: int = 0,
):
    data = data.copy()
    win_length_s = win_length_s if win_length_s is not None else hop_length_s * 4
    margin_s = margin_s if margin_s is not None else win_length_s * 2

    # resample
    SATEN_FS = 96000
    if samplerate != SATEN_FS:
        data = resample(
            y=data, orig_sr=samplerate, target_sr=SATEN_FS, res_type="soxr_lq"
        )
        samplerate = SATEN_FS

    # constants
    f0_floor: int = 71  # from WORLD library
    f0_ceil: int = 800  # from WORLD library
    win_length: int = int(win_length_s * samplerate)
    hop_length: int = int(hop_length_s * samplerate)
    margin: int = int(margin_s * samplerate)
    band_b = firwin(31, [f0_floor, f0_ceil], pass_zero=False, fs=samplerate)
    high_b = firwin(31, f0_ceil, pass_zero=False, fs=samplerate)
    rand = default_rng(noise_seed)

    # preprocess
    # add blue noise
    data_rms = np.sort(rms(y=data)[0])
    noise = _gen_blue_noise(len(data), samplerate, rand)
    noise_snr = min(20 * np.log10(data_rms[-2] / data_rms[1]), 10)
    data = _add_noise_to_signal(data, noise, noise_snr)
    data = data if max(abs(data)) <= 1 else data / max(abs(data))
    # reduce start and end
    HALF_OF_WINDOW = int(win_length / 2)
    data[:HALF_OF_WINDOW] = data[:HALF_OF_WINDOW] * np.linspace(0, 1, HALF_OF_WINDOW)
    data[len(data) - HALF_OF_WINDOW :] = data[len(data) - HALF_OF_WINDOW :] * np.linspace(
        1, 0, HALF_OF_WINDOW
    )

    # ROOT-MEAN-SQUARE
    x_nr = nr.reduce_noise(data, samplerate)
    x_nr_bpf = filtfilt(band_b, 1.0, x_nr)
    x_rms = _normalize(rms(y=x_nr_bpf, frame_length=win_length, hop_length=hop_length)[0])
    rms_threshold = stat.mean(x_rms) if rms_threshold is None else rms_threshold
    # start 側をスライド
    start1 = 0
    for i in range(len(x_rms)):
        if 0 < i and rms_threshold <= x_rms[i]:
            rms_temp_end = min(i + int(margin / hop_length), len(x_rms))
            rms_temp = x_rms[i:rms_temp_end]  # ->
            rms_idx = [j for j, r in enumerate(rms_temp) if r < rms_threshold]
            if rms_idx:  # is not empty
                i = max(rms_idx)
            else:
                start1 = i
                break
    # end 側をスライド
    end1 = len(x_rms)
    for i in range(len(x_rms) - 1, 0, -1):
        if rms_threshold <= x_rms[i]:
            rms_temp_end = max(0, i - int(margin / hop_length))
            rms_temp = x_rms[rms_temp_end:i]  # <-
            rms_idx = [j for j, r in enumerate(rms_temp) if r < rms_threshold]
            if rms_idx:  # is not Empty
                i = min(rms_idx)
            else:
                end1 = i
                break

    start1_s = max(0, start1 * hop_length_s)
    end1_s = min(end1 * hop_length_s, len(data) / samplerate)

    # ZERO-CROSS
    x_nr_hpf = filtfilt(high_b, 1.0, x_nr)
    x_zcs = _normalize(
        zero_crossing_rate(x_nr_hpf, frame_length=win_length, hop_length=hop_length)[0]
    )
    zcs_threshold = stat.mean(x_zcs) if zcs_threshold is None else zcs_threshold
    # start 側をスライド
    start2 = 0
    for i in range(_my_round(start1_s / hop_length_s), 0, -1):
        if zcs_threshold <= x_zcs[i]:
            zcs_temp_end = max(0, i - int(margin / hop_length))
            zcs_temp = x_zcs[zcs_temp_end:i]
            zcs_idx = [j for j, z in enumerate(zcs_temp) if z < zcs_threshold]
            if zcs_idx:  # is not empty
                i = min(zcs_idx)
            else:
                start2 = i
                break
    # end 側をスライド
    end2 = len(x_zcs)
    for i in range(_my_round(end1_s / hop_length_s), len(x_zcs), 1):
        if i < len(data) and zcs_threshold <= x_zcs[i]:
            zcs_temp_end = min(i + int(margin / hop_length), len(x_zcs))
            zcs_temp = x_zcs[i:zcs_temp_end]
            zcs_idx = [j for j, z in enumerate(zcs_temp) if z <= zcs_threshold]
            if zcs_idx:  # is not Empty
                i = max(zcs_idx)
            else:
                end2 = i
                break

    start2_s = max(0, start2 * hop_length_s)
    end2_s = min(end2 * hop_length_s, len(data) / samplerate)

    feats_timestamp = np.linspace(0, len(x_zcs) * hop_length_s, len(x_zcs))
    return (
        start1_s,
        end1_s,
        start2_s,
        end2_s,
        feats_timestamp,
        x_rms,
        rms_threshold,
        x_zcs,
        zcs_threshold,
    )


def _normalize(a: np.ndarray) -> np.ndarray:
    return (a - a.min()) / (a.max() - a.min())


def _gen_blue_noise(length: int, fs: int, rand: np.random.Generator) -> np.ndarray:
    length2 = length + 1000
    # white noise
    wh = rand.uniform(low=-1.0, high=1.0, size=length2)
    # fft
    WH = np.fft.rfft(wh)
    WH_f = np.fft.rfftfreq(len(wh), 1 / fs)
    # white -> blue
    BL = WH * np.sqrt(WH_f)
    # irfft
    bl = np.fft.irfft(BL)
    # normalize
    bl /= np.max(np.abs(bl))

    return bl[:length]


def _add_noise_to_signal(
    signal: np.ndarray, noise: np.ndarray, desired_snr_db: np.ndarray
):
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    scaling_factor = np.sqrt(signal_power / (desired_snr_linear * noise_power))
    noise_adjusted = noise * scaling_factor
    return signal + noise_adjusted


def _my_round(a: float) -> int:
    return int(np.floor(a + 0.5))
