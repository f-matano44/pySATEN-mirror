import numpy as np
import numpy.typing as npt


def root_mean_square(y: npt.NDArray, win_length: int, hop_length: int) -> npt.NDArray:
    rms = np.zeros(int(np.ceil(float(len(y)) / hop_length)))
    for i in range(len(rms)):
        # get target array
        idx = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zc_start:zc_end]
        # calc rms
        rms[i] = np.sqrt(np.mean(np.power(target, 2)))
    return rms


def zero_crossing_rate(y: npt.NDArray, win_length: int, hop_length: int) -> npt.NDArray:
    zcr = np.zeros(int(np.ceil(float(len(y)) / hop_length)))
    for i in range(len(zcr)):
        # get target array
        idx = i * hop_length
        zcr_start = int(max(0, idx - (win_length / 2)))
        zcr_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zcr_start:zcr_end]
        # calc zcr
        sign_arr = np.sign(target)[target != 0 & ~np.isnan(target)]
        zcr[i] = np.sum(np.abs(np.diff(sign_arr)) != 0) / len(target)
    return zcr


def normalize(y: npt.NDArray) -> npt.NDArray:
    return (y - y.min()) / (y.max() - y.min())


def slide_index(
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
            a_check_end = max(0, i - margin) if goto_min else min(i + margin, len(y))
            a_check = y[a_check_end:i] if goto_min else y[i:a_check_end]
            indices_below_threshold = [j for j, b in enumerate(a_check) if b < threshold]
            if indices_below_threshold:  # is not empty
                i = (
                    min(indices_below_threshold)
                    if goto_min
                    else max(indices_below_threshold)
                )
            else:  # indices_below_threshold is empty -> finish!!!
                return i
    return 0 if goto_min else len(y)
