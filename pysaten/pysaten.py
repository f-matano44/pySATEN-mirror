# SPDX-License-Identifier:GPL-3.0-or-later

from typing import Optional

import noisereduce as nr
import numpy as np
from librosa import resample
from librosa.feature import rms
from librosa.feature import zero_crossing_rate as zcr
from numpy.random import default_rng
from scipy.signal import cheby1, firwin, lfilter, sosfilt

_F0_FLOOR: int = 71  # from WORLD library
_F0_CEIL: int = 800  # from WORLD library


def vsed(y: np.ndarray, sr: int) -> tuple[float, float]:
    _, _, _, _, start_s, end_s, _, _, _ = vsed_debug(y, sr)
    return start_s, end_s


def vsed_debug(
    y: np.ndarray,
    orig_sr: int,
    # -------------------------------------------
    win_length_s: Optional[float] = None,
    hop_length_s: float = 0.01,
    # -------------------------------------------
    rms_threshold: float = 0.03,
    zcr_threshold: float = 0.67,
    zcr_margin_s: float = 0.1,
    offset_s: float = 0.03,
    # -------------------------------------------
    noise_seed: int = 0,
):
    # resample
    sr: int = 96000
    y_rsp: np.ndarray = resample(
        y=y, orig_sr=orig_sr, target_sr=sr, res_type="soxr_lq"
    )

    # constants
    win_length_s = (
        win_length_s if win_length_s is not None else hop_length_s * 4
    )
    win_length: int = int(win_length_s * sr)
    hop_length: int = int(hop_length_s * sr)
    zcr_margin: int = int(zcr_margin_s / hop_length_s)

    # preprocess: add blue noise && remove background noise
    y_nr = _00_preprocess(y_rsp, sr, noise_seed)

    # step1: Root mean square
    start1, end1, y_rms = _01_rms(
        y_nr, sr, rms_threshold, win_length, hop_length
    )

    # step2: Zero cross
    start2, end2, y_zcr = _02_zcr(
        y_nr,
        sr,
        start1,
        end1,
        zcr_threshold,
        zcr_margin,
        win_length,
        hop_length,
    )

    # index -> second: rms
    start1_s: float = max(0, start1 * hop_length_s)
    end1_s: float = min(end1 * hop_length_s, len(y_rsp) / sr)

    # index -> second: zrs
    start2_s: float = max(0, start2 * hop_length_s)
    end2_s: float = min(end2 * hop_length_s, len(y_rsp) / sr)

    feats_timestamp = np.linspace(0, len(y_zcr) * hop_length_s, len(y_zcr))

    return (
        start1_s,
        end1_s,
        start2_s,
        end2_s,
        start2_s - offset_s,
        end2_s + offset_s,
        feats_timestamp,
        y_rms,
        y_zcr,
    )


def _00_preprocess(y: np.ndarray, sr: int, noise_seed: int) -> np.ndarray:
    data_rms = np.sort(rms(y=y)[0])
    signal_amp = data_rms[-2]
    noise_amp = max(data_rms[1], 1e-10)
    snr = min(20 * np.log10(signal_amp / noise_amp), 10)
    noise = _gen_blue_noise(len(y), sr, noise_seed)
    y_blue = y + noise * (signal_amp / 10 ** (snr / 20))
    y_blue = y_blue if max(abs(y_blue)) <= 1 else y_blue / max(abs(y_blue))
    return nr.reduce_noise(y_blue, sr)


def _01_rms(
    y, sr, threshold, win_length, hop_length
) -> tuple[int, int, np.ndarray]:
    nyq: int = int(sr / 2)
    wp = [_F0_FLOOR / nyq, _F0_CEIL / nyq]
    band_sos = cheby1(N=12, rp=1, Wn=wp, btype="bandpass", output="sos")
    y_bpf = sosfilt(band_sos, y)
    y_rms = _normalize(
        rms(y=y_bpf, frame_length=win_length, hop_length=hop_length)[0]
    )
    start1: int = (
        np.where(threshold < y_rms)[0][0] if np.any(threshold < y_rms) else 0
    )
    end1: int = (
        np.where(threshold < y_rms)[0][-1]
        if np.any(threshold < y_rms)
        else len(y_rms) - 1
    )
    return start1, end1, y_rms


def _02_zcr(y, sr, start1, end1, threshold, margin, win_length, hop_length):
    high_b = firwin(101, _F0_CEIL, pass_zero=False, fs=sr)
    y_hpf = lfilter(high_b, 1.0, y)
    y_zcr = _normalize(
        zcr(y_hpf, frame_length=win_length, hop_length=hop_length)[0]
    )
    # slide start index
    start2 = _slide_index(
        goto_min=True,
        y=y_zcr,
        start_idx=start1,
        threshold=threshold,
        margin=margin,
    )
    # slide end index
    end2 = _slide_index(
        goto_min=False,
        y=y_zcr,
        start_idx=end1,
        threshold=threshold,
        margin=margin,
    )
    return start2, end2, y_zcr


def _normalize(y: np.ndarray) -> np.ndarray:
    return (y - y.min()) / (y.max() - y.min())


def _gen_blue_noise(length: int, sr: int, noise_seed: int) -> np.ndarray:
    rand: np.random.Generator = default_rng(noise_seed)
    # white noise
    wh = rand.uniform(low=-1.0, high=1.0, size=length + 1000)
    # fft
    WH = np.fft.rfft(wh)
    WH_f = np.fft.rfftfreq(len(wh), 1 / sr)
    # white -> blue
    BL = WH * np.sqrt(WH_f)
    # irfft
    bl = np.fft.irfft(BL)
    # normalize
    bl /= np.max(np.abs(bl))

    return bl[:length]


def _slide_index(
    goto_min: bool,
    y: np.ndarray,
    start_idx: int,
    threshold: float,
    margin: int,
) -> int:

    stop_idx: int = -1 if goto_min else len(y)
    step: int = -1 if goto_min else 1

    for i in range(start_idx, stop_idx, step):
        if threshold <= y[i]:
            a_check_end = (
                max(0, i - margin) if goto_min else min(i + margin, len(y))
            )
            a_check = y[a_check_end:i] if goto_min else y[i:a_check_end]
            indices_below_threshold = [
                j for j, b in enumerate(a_check) if b < threshold
            ]
            if indices_below_threshold:  # is not empty
                i = (
                    min(indices_below_threshold)
                    if goto_min
                    else max(indices_below_threshold)
                )
            else:  # indices_below_threshold is empty -> finish!!!
                return i
    return 0
