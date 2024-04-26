import statistics as stat
from typing import Optional

import noisereduce as nr
import numpy as np
from librosa import resample
from librosa.feature import rms
from librosa.feature import zero_crossing_rate as zcr
from numpy.random import default_rng
from scipy.signal import filtfilt, firwin


def vsed(data: np.ndarray, samplerate: int) -> tuple[float, float]:
    _, _, start_s, end_s, _, _, _, _, _ = vsed_debug(
        data=data, samplerate=samplerate, rms_threshold=0.05, zcr_threshold=0.5
    )
    return start_s, end_s


def vsed_debug(
    data: np.ndarray,
    samplerate: int,
    win_length_s: Optional[float] = None,
    hop_length_s: float = 0.01,
    rms_threshold: Optional[float] = None,
    zcr_threshold: Optional[float] = None,
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
    margin: int = int(margin_s / hop_length_s)
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
    start1 = _slide_index(
        goto_min=False, a=x_rms, start_idx=0, threshold=rms_threshold, margin=margin
    )
    # end 側をスライド
    end1 = _slide_index(
        goto_min=True,
        a=x_rms,
        start_idx=len(x_rms) - 1,
        threshold=rms_threshold,
        margin=margin,
    )

    # ZERO-CROSS
    x_nr_hpf = filtfilt(high_b, 1.0, x_nr)
    x_zcr = _normalize(zcr(x_nr_hpf, frame_length=win_length, hop_length=hop_length)[0])
    zcr_threshold = stat.mean(x_zcr) if zcr_threshold is None else zcr_threshold
    # start 側をスライド
    start2 = _slide_index(
        goto_min=True,
        a=x_zcr,
        start_idx=start1,
        threshold=zcr_threshold,
        margin=margin,
    )
    # end 側をスライド
    end2 = _slide_index(
        goto_min=False, a=x_zcr, start_idx=end1, threshold=zcr_threshold, margin=margin
    )

    # RMS
    start1_s = max(0, start1 * hop_length_s)
    end1_s = min(end1 * hop_length_s, len(data) / samplerate)
    # ZCR
    start2_s = max(0, start2 * hop_length_s)
    end2_s = min(end2 * hop_length_s, len(data) / samplerate)

    feats_timestamp = np.linspace(0, len(x_zcr) * hop_length_s, len(x_zcr))

    return (
        start1_s,
        end1_s,
        start2_s,
        end2_s,
        feats_timestamp,
        x_rms,
        rms_threshold,
        x_zcr,
        zcr_threshold,
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


def _slide_index(
    goto_min: bool, a: np.ndarray, start_idx: int, threshold: float, margin: int
) -> int:

    stop_idx: int = -1 if goto_min else len(a)
    step: int = -1 if goto_min else 1

    for i in range(start_idx, stop_idx, step):
        if threshold <= a[i]:
            a_check_end = max(0, i - margin) if goto_min else min(i + margin, len(a))
            a_check = a[a_check_end:i] if goto_min else a[i:a_check_end]
            indices_below_threshold = [j for j, b in enumerate(a_check) if b < threshold]
            if indices_below_threshold:  # is not empty
                i = (
                    min(indices_below_threshold)
                    if goto_min
                    else max(indices_below_threshold)
                )
            else:  # indices_below_threshold is empty -> finish!!!
                return i
    return 0
