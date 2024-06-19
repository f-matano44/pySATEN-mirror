import statistics as stat
from typing import Optional

import noisereduce as nr
import numpy as np
from librosa import resample
from librosa.feature import rms
from librosa.feature import zero_crossing_rate as zcr
from numpy.random import default_rng
from scipy.signal import cheby1, sosfilt


def vsed(data: np.ndarray, samplerate: int) -> tuple[float, float]:
    _, _, start_s, end_s, _, _, _, _, _ = vsed_debug(
        data=data,
        samplerate=samplerate,
        rms_threshold=0.05,
        zcr_threshold=0.68,
        offset_s=0.03,
    )
    return start_s, end_s


def vsed_debug(
    data: np.ndarray,
    samplerate: int,
    win_length_s: Optional[float] = None,
    hop_length_s: float = 0.01,
    rms_threshold: Optional[float] = None,
    zcr_threshold: Optional[float] = None,
    zcr_margin_s: float = 0.1,
    offset_s: float = 0,
    noise_seed: int = 0,
):
    data = data.copy()
    win_length_s = win_length_s if win_length_s is not None else hop_length_s * 4

    # resample
    data, samplerate = _resample_to_96k(data, samplerate)

    # constants
    f0_floor: int = 71  # from WORLD library
    f0_ceil: int = 800  # from WORLD library
    win_length: int = int(win_length_s * samplerate)
    hop_length: int = int(hop_length_s * samplerate)
    zcr_margin: int = int(zcr_margin_s / hop_length_s)

    # filter design
    nyquist: int = int(samplerate / 2)
    apass: int = 1
    order: int = 12

    # set random seed
    rand = default_rng(noise_seed)

    # preprocess: add blue noise && remove background noise
    data_rms = np.sort(rms(y=data)[0])
    noise = _gen_blue_noise(len(data), samplerate, rand)
    signal_amp = data_rms[-2]
    noise_amp = max(data_rms[1], 1e-10)
    snr = min(20 * np.log10(signal_amp / noise_amp), 10)
    data = data + noise * (signal_amp / 10 ** (snr / 20))
    data = data if max(abs(data)) <= 1 else data / max(abs(data))
    x_nr = nr.reduce_noise(data, samplerate)

    # ROOT-MEAN-SQUARE
    wp = [f0_floor / nyquist, f0_ceil / nyquist]
    band_sos = cheby1(order, rp=apass, Wn=wp, btype="bandpass", output="sos")
    x_nr_bpf = sosfilt(band_sos, x_nr)
    x_rms = _normalize(rms(y=x_nr_bpf, frame_length=win_length, hop_length=hop_length)[0])
    rms_threshold = stat.mean(x_rms) if rms_threshold is None else rms_threshold
    start1: int = (
        np.where(rms_threshold < x_rms)[0][0] if np.any(rms_threshold < x_rms) else 0
    )
    end1: int = (
        np.where(rms_threshold < x_rms)[0][-1]
        if np.any(rms_threshold < x_rms)
        else len(x_rms) - 1
    )

    # ZERO-CROSS
    wp = f0_ceil / nyquist
    high_sos = cheby1(order, rp=apass, Wn=wp, btype="low", output="sos")
    x_nr_hpf = sosfilt(high_sos, x_nr)
    x_zcr = _normalize(zcr(x_nr_hpf, frame_length=win_length, hop_length=hop_length)[0])
    zcr_threshold = stat.mean(x_zcr) if zcr_threshold is None else zcr_threshold
    # slide start index
    start2 = _slide_index(
        goto_min=True,
        a=x_zcr,
        start_idx=start1,
        threshold=zcr_threshold,
        margin=zcr_margin,
    )
    # slide end index
    end2 = _slide_index(
        goto_min=False,
        a=x_zcr,
        start_idx=end1,
        threshold=zcr_threshold,
        margin=zcr_margin,
    )

    # RMS
    start1_s = max(0, start1 * hop_length_s)
    end1_s = min(end1 * hop_length_s, len(data) / samplerate)
    # ZCR
    start2_s = max(0, start2 * hop_length_s)
    end2_s = min(end2 * hop_length_s, len(data) / samplerate)

    start2_s -= offset_s
    end2_s += offset_s

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


def _resample_to_96k(data: np.ndarray, src_fs: int) -> tuple[np.ndarray, int]:
    target_fs = 96000
    data = resample(y=data, orig_sr=src_fs, target_sr=target_fs, res_type="soxr_lq")
    return data, target_fs


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
