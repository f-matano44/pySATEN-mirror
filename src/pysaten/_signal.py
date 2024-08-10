import math

import numpy as np

from . import _utility as util


def rms(y, win_length, hop_length):
    rms = np.zeros(math.ceil(float(len(y)) / hop_length))
    for i in range(len(rms)):
        # get target array
        idx = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zc_start:zc_end]
        # calc rms
        rms[i] = util.sqrt(np.mean(util.pow(target, 2)))
    return rms


def zcr(y, win_length, hop_length):
    zcr = np.zeros(math.ceil(float(len(y)) / hop_length))
    for i in range(len(zcr)):
        # get target array
        idx = i * hop_length
        zcr_start = int(max(0, idx - (win_length / 2)))
        zcr_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zcr_start:zcr_end]
        # calc zcr
        sign_arr = np.sign(target)[target != 0 & ~np.isnan(target)]
        zcr[i] = np.sum(np.abs(np.diff(sign_arr)) != 0) / hop_length
    return zcr


def blue_noise(length: int, sr: int, noise_seed: int) -> np.ndarray:
    rand: np.random.Generator = np.random.default_rng(noise_seed)
    length2 = length + 1000
    # white noise
    wh = rand.uniform(low=-1.0, high=1.0, size=length2)
    # fft
    WH = np.fft.rfft(wh)
    WH_f = np.fft.rfftfreq(len(wh), 1 / sr)
    # white -> blue
    BL = WH * util.sqrt(WH_f)
    # irfft
    bl = np.fft.irfft(BL)
    # normalize
    bl /= np.max(np.abs(bl))

    return bl[:length]
