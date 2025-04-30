import numpy as np
from numpy.fft import irfft, rfft, rfftfreq


def white(length: int, noise_seed: int) -> np.ndarray:
    rand: np.random.Generator = np.random.default_rng(noise_seed)
    return rand.uniform(low=-1.0, high=1.0, size=length)


def blue(length: int, sr: int, noise_seed: int) -> np.ndarray:
    offset = int(length * 0.10)  # TODO: temporary placeholder
    # white noise
    wh = white(length + (offset * 2), noise_seed)
    # fft
    WH = rfft(wh)
    WH_f = rfftfreq(len(wh), 1 / sr)
    # white -> blue
    BL = WH * np.sqrt(WH_f)
    # irfft
    bl = irfft(BL)
    # normalize
    bl /= np.max(np.abs(bl))

    return bl[offset : length + offset]


def pink(length: int, sr: int, noise_seed: int) -> np.ndarray:
    offset = int(length * 0.10)  # TODO: temporary placeholder
    # white noise
    wh = white(length + (offset * 2), noise_seed)
    # fft
    WH = rfft(wh)
    WH_f = rfftfreq(len(wh), 1 / sr)
    # white -> pink
    PK = WH.copy()
    for i in range(len(WH)):
        PK[i] = WH[i] / np.sqrt(WH_f[i]) if 20 < WH_f[i] else 0
    # irfft
    pk = np.fft.irfft(PK)
    # normalize
    pk /= np.max(np.abs(pk))

    return pk[offset : length + offset]
