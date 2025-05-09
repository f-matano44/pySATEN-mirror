# SPDX-License-Identifier:GPL-3.0-or-later

from typing import Optional

import noisereduce as nr
import numpy as np
from librosa import resample
from scipy.signal import cheby1, firwin, lfilter, sosfilt

from ..utility import color_noise
from ..utility.constants import F0_CEIL, F0_FLOOR, NYQ, SR
from ..utility.others import slide_index
from ..utility.signal import normalize, rms, zcr


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
    y_rsp: np.ndarray = resample(
        y=y, orig_sr=orig_sr, target_sr=SR, res_type="soxr_lq"
    )

    # constants
    win_length_s = (
        win_length_s if win_length_s is not None else hop_length_s * 4
    )
    win_length: int = int(win_length_s * SR)
    hop_length: int = int(hop_length_s * SR)
    zcr_margin: int = int(zcr_margin_s / hop_length_s)

    # preprocess: add blue noise && remove background noise
    y_nr = _00_preprocess(y_rsp, SR, noise_seed)

    # step1: Root mean square
    start1, end1, y_rms = _01_rms(
        y_nr, SR, rms_threshold, win_length, hop_length
    )

    # step2: Zero cross
    start2, end2, y_zcr = _02_zcr(
        y_nr,
        SR,
        start1,
        end1,
        zcr_threshold,
        zcr_margin,
        win_length,
        hop_length,
    )

    # index -> second: rms
    start1_s: float = max(0, start1 * hop_length_s)
    end1_s: float = min(end1 * hop_length_s, len(y_rsp) / SR)

    # index -> second: zrs
    start2_s: float = max(0, start2 * hop_length_s)
    end2_s: float = min(end2 * hop_length_s, len(y_rsp) / SR)

    # add offset
    start3_s: float = max(0, start2_s - offset_s)
    end3_s: float = min(end2_s + offset_s, len(y_rsp) / SR)

    feats_timestamp = np.linspace(0, len(y_zcr) * hop_length_s, len(y_zcr))

    return (
        start1_s,
        end1_s,
        start2_s,
        end2_s,
        start3_s,
        end3_s,
        feats_timestamp,
        y_rms,
        y_zcr,
    )


def _00_preprocess(y: np.ndarray, sr: int, noise_seed: int) -> np.ndarray:
    data_rms = np.sort(rms(y, 2048, 512))  # <- default of librosa
    signal_amp = data_rms[-2]
    noise_amp = max(data_rms[1], 1e-10)
    snr = min(20 * np.log10(signal_amp / noise_amp), 10)
    noise = color_noise.blue(len(y), sr, noise_seed).cpu().numpy()
    y_blue = y + noise * (signal_amp / 10 ** (snr / 20))
    y_blue = y_blue if max(abs(y_blue)) <= 1 else y_blue / max(abs(y_blue))
    return nr.reduce_noise(y_blue, sr)


def _01_rms(
    y, sr, threshold, win_length, hop_length
) -> tuple[int, int, np.ndarray]:
    wp = [F0_FLOOR / NYQ, F0_CEIL / NYQ]
    band_sos = cheby1(N=12, rp=1, Wn=wp, btype="bandpass", output="sos")
    y_bpf = sosfilt(band_sos, y)
    y_rms = normalize(rms(y_bpf, win_length, hop_length))
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
    high_b = firwin(101, F0_CEIL, pass_zero=False, fs=sr)
    y_hpf = lfilter(high_b, 1.0, y)
    y_zcr = normalize(zcr(y_hpf, win_length, hop_length))
    # slide start index
    start2 = slide_index(
        goto_min=True,
        y=y_zcr,
        start_idx=start1,
        threshold=threshold,
        margin=margin,
    )
    # slide end index
    end2 = slide_index(
        goto_min=False,
        y=y_zcr,
        start_idx=end1,
        threshold=threshold,
        margin=margin,
    )
    return start2, end2, y_zcr
