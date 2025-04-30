import math

import numpy as np


def rms(y, win_length, hop_length):
    rms = np.zeros(math.ceil(float(len(y)) / hop_length))
    for i in range(len(rms)):
        # get target array
        idx = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zc_start:zc_end]
        # calc rms
        rms[i] = _sqrt(np.mean(_pow(target, 2)))
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


def normalize(y: np.ndarray) -> np.ndarray:
    return (y - y.min()) / (y.max() - y.min())


def _pow(a, b):
    return a**b


def _sqrt(a):
    return a**0.5
