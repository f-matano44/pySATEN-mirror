import argparse
import time
from typing import Tuple

import numpy as np
import numpy.typing as npt
import soundfile

from .lv1.pysaten import vsed_debug


def cli_runner() -> None:
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    # trimming
    y, sr = soundfile.read(args.input)
    y_trimmed: npt.NDArray = trim(y, sr)
    soundfile.write(args.output, y_trimmed, sr)


def trim(y: npt.NDArray[np.floating], sr: int) -> npt.NDArray[np.floating]:
    s_sec, e_sec = vsed(y, sr)
    return y[int(s_sec * sr) : int(e_sec * sr)]


def vsed(y: npt.NDArray[np.floating], sr: int) -> Tuple[float, float]:
    seed = int(time.time())
    # shape check (monaural only)
    if y.ndim != 1:
        raise ValueError("PySaten only supports mono audio.")
    # trim
    _, _, _, _, start_s, end_s, _, _, _ = vsed_debug(y, sr, noise_seed=seed)
    return start_s, end_s
