import numpy as np


def rms(y, win_length, hop_length):
    rms = np.zeros(int(np.ceil(len(y) / hop_length)))
    for i in range(len(rms)):
        # get target array
        idx = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zc_start:zc_end]
        # calc rms
        rms[i] = np.sqrt(np.mean(pow(target, 2)))
    return rms


def zcr(y, win_length, hop_length):
    zc = np.zeros(int(np.ceil(len(y) / hop_length)))
    for i in range(len(zc)):
        # get target array
        idx = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zc_start:zc_end]
        # calc zcr
        sign_arr = np.sign(target)[target != 0 & ~np.isnan(target)]
        zc[i] = np.sum(np.abs(np.diff(sign_arr)) != 0) / hop_length
    return zc


def blue_noise(length: int, sr: int, noise_seed: int) -> np.ndarray:
    rand: np.random.Generator = np.random.default_rng(noise_seed)
    length2 = int(pow(2, np.ceil(np.log2(length)) + 1))
    # white noise
    wh = rand.uniform(low=-1.0, high=1.0, size=length2)
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
