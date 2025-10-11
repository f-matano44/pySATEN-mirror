# SPDX-License-Identifier:GPL-3.0-or-later

from typing import Final, Optional

import noisereduce as nr
import numpy as np
import numpy.typing as npt
from librosa import resample
from scipy.signal import cheby1, filtfilt, firwin, sosfilt

from ..utility.color_noise import blue as blue_noise
from ..utility.constants import F0_CEIL, F0_FLOOR, SR
from ..utility.signal import normalize
from ..utility.signal import root_mean_square as rms
from ..utility.signal import slide_index
from ..utility.signal import zero_crossing_rate as zcr


class vsed_debug_v2:
    def __init__(
        self,
        y: npt.NDArray,
        orig_sr: int | float,
        # -------------------------------------------
        win_length_s: Optional[float] = None,
        hop_length_s: float = 0.005,
        zcr_margin_s: float = 0.08,
        # -------------------------------------------
        rms_threshold: float = 0.03,
        zcr_threshold: float = 0.67,
        offset_s: float = 0.03,
        # -------------------------------------------
        noise_seed: int = 0,
    ):
        # For debug code -------------------------------------------
        self.rms_threshold: Final[float] = rms_threshold
        self.zcr_threshold: Final[float] = zcr_threshold

        # Resample -------------------------------------------
        y_rsp: Final[npt.NDArray] = resample(y=y, orig_sr=orig_sr, target_sr=SR)
        y_length_s: Final[float] = len(y_rsp) / SR

        # constants -------------------------------------------
        win_length_s = win_length_s if win_length_s is not None else hop_length_s * 4
        win_length: Final[int] = int(win_length_s * SR)
        hop_length: Final[int] = int(hop_length_s * SR)
        zcr_margin: Final[int] = int(zcr_margin_s / hop_length_s)

        # Preprocess -------------------------------------------
        y_nr, yf_nr = _00_preprocess(y_rsp, SR, noise_seed)
        # get root-mean-square
        y_rms: Final[npt.NDArray] = _01_rms(y_nr, yf_nr, SR, win_length, hop_length)
        y_rms.flags.writeable = False
        # get zero-crossing-rate
        self.y_zcr: Final[npt.NDArray] = _02_zcr(y_nr, SR, win_length, hop_length)
        self.y_zcr.flags.writeable = False
        # get rms weight
        self.bell = _gaussian_curve(self.y_zcr)
        self.bell.flags.writeable = False
        rms_weight: Final[npt.NDArray] = (1 - self.y_zcr) * self.bell
        rms_weight.flags.writeable = False

        # step1: Root mean square -------------------------------------------
        self.y_rms: Final[npt.NDArray] = normalize(y_rms * rms_weight)
        self.y_rms.flags.writeable = False
        start1: Final[int] = (
            np.where(rms_threshold < self.y_rms)[0][0]
            if np.any(rms_threshold < self.y_rms)
            else 0
        )
        end1: Final[int] = (
            np.where(rms_threshold < self.y_rms)[0][-1]
            if np.any(rms_threshold < self.y_rms)
            else len(self.y_rms) - 1
        )

        # step2: Zero cross -------------------------------------------
        start2: Final[int] = slide_index(
            goto_min=True,
            y=self.y_zcr,
            start_idx=start1,
            threshold=zcr_threshold,
            margin=zcr_margin,
        )
        end2: Final[int] = slide_index(
            goto_min=False,
            y=self.y_zcr,
            start_idx=end1,
            threshold=zcr_threshold,
            margin=zcr_margin,
        )

        # index -> second: rms -------------------------------------------
        self.start1_s: Final[float] = max(0, start1 * hop_length_s)
        self.end1_s: Final[float] = min(end1 * hop_length_s, y_length_s)

        # index -> second: zrs -------------------------------------------
        self.start2_s: Final[float] = max(0, start2 * hop_length_s)
        self.end2_s: Final[float] = min(end2 * hop_length_s, y_length_s)

        # add offset -------------------------------------------
        self.start3_s: Final[float] = max(0, self.start2_s - offset_s)
        self.end3_s: Final[float] = min(self.end2_s + offset_s, y_length_s)

        # get timestamp of features -------------------------------------------
        self.feats_timestamp: Final[npt.NDArray] = np.linspace(
            0, len(self.y_zcr) * hop_length_s, len(self.y_zcr)
        )
        self.feats_timestamp.flags.writeable = False

    def get_result(self) -> tuple[float, float]:
        return self.start3_s, self.end3_s


def _00_preprocess(y: np.ndarray, sr: int, noise_seed: int) -> tuple:
    data_rms = np.sort(rms(y, 2048, 512))  # <- default of librosa
    signal_amp = data_rms[-2]
    noise_amp = max(data_rms[1], 1e-10)
    snr = min(20 * np.log10(signal_amp / noise_amp), 10)
    noise = blue_noise(len(y), sr, noise_seed)
    y_blue = y + noise * (signal_amp / 10 ** (snr / 20))
    y_nr = nr.reduce_noise(y_blue, sr)
    yf_nr = nr.reduce_noise(y_blue[::-1], sr)
    return (
        y_nr / np.abs(y_nr).max(),
        yf_nr / np.abs(y_nr).max(),
    )  # return normalized value


def _01_rms(y, yf, sr, win_length, hop_length) -> np.ndarray:
    wp = (F0_FLOOR, F0_CEIL)
    band_sos = cheby1(N=12, rp=1, Wn=wp, btype="bandpass", output="sos", fs=sr)
    # normal
    y_bpf = sosfilt(band_sos, y)
    y_rms = normalize(rms(y_bpf, win_length, hop_length))
    # flip
    yf_bpf = sosfilt(band_sos, yf)
    yf_rms = normalize(rms(yf_bpf, win_length, hop_length))
    # mix
    idx = np.argmax(y_rms)
    y_rms[:idx] = yf_rms[len(yf_rms) - idx :][::-1]
    return y_rms


def _02_zcr(y, sr, win_length, hop_length):
    high_b = firwin(101, F0_CEIL, pass_zero=False, fs=sr)
    y_hpf = filtfilt(high_b, 1.0, y)
    y_zcr = normalize(zcr(y_hpf, win_length, hop_length))
    y_zcr = normalize(np.clip(y_zcr, 0, 0.5)) ** 2
    return y_zcr


def _gaussian_curve(zcr: npt.NDArray) -> np.ndarray:
    idx = np.linspace(0.0, 1.0, zcr.size)
    mu = np.average(np.arange(zcr.size), weights=(1.0 - zcr)) / zcr.size
    sigma = min(mu, 1.0 - mu) / 2.0
    coef = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((idx - mu) / sigma) ** 2
    pdf = coef * np.exp(exponent)
    return normalize(pdf)
