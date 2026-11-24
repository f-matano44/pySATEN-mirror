import argparse
from time import time_ns

import librosa
import soundfile as sf
from numpy import floating
from numpy.typing import NDArray

from .v2 import vsed_debug_v2


def _cli_runner() -> None:
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    # trimming
    y, sr = librosa.load(args.input, sr=48000)
    y_trimmed: NDArray[floating] = trim(y, sr)[0]
    sf.write(args.output, y_trimmed, sr, format="WAV", subtype="PCM_24")


def trim(y: NDArray[floating], sr: float) -> tuple[NDArray[floating], tuple[int, int]]:
    s_sec, e_sec = _vsed(y, sr)
    s_idx = int(s_sec * sr)
    e_idx = int(e_sec * sr)
    y_trimmed = y[s_idx:e_idx]
    return y_trimmed, (s_idx, e_idx)


def _vsed(
    y: NDArray[floating], sr: float, seed: int = time_ns() % 2**32
) -> tuple[float, float]:
    # shape check (monaural only)
    if y.ndim != 1:
        raise ValueError("PySaten only supports monaural audio.")
    # trim
    return vsed_debug_v2(y, sr, noise_seed=seed).get_result()
