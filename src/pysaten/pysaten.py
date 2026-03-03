import argparse
from time import time_ns

import librosa
import soundfile as sf
from numpy import floating
from numpy.typing import NDArray

from .v2 import vsed_debug_v2


def cli_runner() -> None:
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    # trimming
    y, sr = librosa.load(args.input, sr=48000)
    y_trimmed: NDArray[floating] = trim(y, sr)
    sf.write(args.output, y_trimmed, sr, format="WAV", subtype="PCM_24")


def trim(y: NDArray[floating], sr: float) -> NDArray[floating]:
    s_sec, e_sec = vsed(y, sr)
    return y[int(s_sec * sr) : int(e_sec * sr)]


def vsed(
    y: NDArray[floating], sr: float, seed: int = time_ns() % 2**32
) -> tuple[float, float]:
    # shape check (monaural only)
    if y.ndim != 1:
        raise ValueError("PySaten only supports monaural audio.")
    # trim
    return vsed_debug_v2(y, sr, noise_seed=seed).get_result()
