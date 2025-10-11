import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm


def white(length: int, seed: int) -> NDArray:
    return truncnorm.rvs(-3, 3, loc=0, scale=1, size=length, random_state=seed)


def blue(length: int, sr: float, seed: int) -> NDArray:
    offset = int(length / 2)
    # white noise
    wh = white(length + (offset * 2), seed)
    # fft
    WH_f = np.fft.rfft(wh)
    freqs = np.fft.rfftfreq(len(wh), 1 / sr)
    # white -> blue
    BL_f = WH_f * np.sqrt(freqs)
    # irfft
    bl = np.fft.irfft(BL_f)
    # normalize
    bl /= np.abs(bl).max()

    return bl[offset : length + offset]


def pink(length: int, sr: float, seed: int) -> NDArray:
    offset = int(length / 2)
    # white noise
    wh = white(length + (offset * 2), seed)
    # fft
    WH_f = np.fft.rfft(wh)
    freqs = np.fft.rfftfreq(len(wh), 1 / sr)
    # white -> pink
    mask = freqs > 20.0
    PK_f = np.zeros_like(WH_f)
    PK_f[mask] = WH_f[mask] / np.sqrt(freqs[mask])
    # irfft
    pk = np.fft.irfft(PK_f)
    # normalize
    pk /= np.abs(pk).max()

    return pk[offset : length + offset]
