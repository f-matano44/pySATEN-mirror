# SPDX-License-Identifier:GPL-3.0-or-later

import argparse

import numpy as np
import soundfile

from .lv1.pysaten import vsed_debug


def cli_runner():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    # trimming
    y, sr = soundfile.read(args.input)
    y_trimmed = trim(y, sr)
    soundfile.write(args.output, y_trimmed, sr)


def trim(y: np.ndarray, sr: int) -> np.ndarray:
    s_sec, e_sec = vsed(y, sr)
    return y[int(s_sec * sr) : int(e_sec * sr)]


def vsed(y: np.ndarray, sr: int) -> tuple[float, float]:
    # shape check (monaural only)
    if len(y.shape) != 1:
        raise ValueError(
            "Error: The audio file is not mono. It has more than one channel."
        )
    # trim
    _, _, _, _, start_s, end_s, _, _, _ = vsed_debug(y, sr)
    return start_s, end_s
