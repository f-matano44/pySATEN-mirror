import numpy as np
import numpy.typing as npt


def rms(y: npt.NDArray, win_length: int, hop_length: int) -> npt.NDArray:
    rms = np.zeros(int(np.ceil(float(len(y)) / hop_length)))
    for i in range(len(rms)):
        # get target array
        idx = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zc_start:zc_end]
        # calc rms
        rms[i] = _sqrt(np.mean(_pow(target, 2)))
    return rms


def zcr(y: npt.NDArray, win_length: int, hop_length: int) -> npt.NDArray:
    zcr = np.zeros(int(np.ceil(float(len(y)) / hop_length)))
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


def normalize(y: npt.NDArray) -> npt.NDArray:
    return (y - y.min()) / (y.max() - y.min())


def _pow(a: npt.NDArray, b: float) -> npt.NDArray:
    return a**b


def _sqrt(a: float) -> float:
    return a**0.5
