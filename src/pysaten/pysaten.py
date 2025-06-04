import argparse
import time
from typing import Optional

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from .v1 import vsed_debug_v1


def cli_runner() -> None:
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    # trimming
    y, sr = librosa.load(args.input, sr=48000)
    y_trimmed: npt.NDArray[np.floating] = trim(y, sr)
    sf.write(args.output, y_trimmed, sr, format="WAV", subtype="PCM_24")


def trim(y: npt.NDArray[np.floating], sr: float) -> npt.NDArray[np.floating]:
    s_sec, e_sec = vsed(y, sr)
    return y[int(s_sec * sr) : int(e_sec * sr)]


def vsed(
    y: npt.NDArray[np.floating], sr: float, seed: Optional[int] = None
) -> tuple[float, float]:
    seed = time.time_ns() if seed is None else seed
    # shape check (monaural only)
    if y.ndim != 1:
        raise ValueError("PySaten only supports monaural audio.")
    # trim
    _, _, _, _, start_s, end_s, _, _, _ = vsed_debug_v1(y, sr, noise_seed=seed)
    return start_s, end_s
