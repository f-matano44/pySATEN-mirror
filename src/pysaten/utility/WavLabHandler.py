from pathlib import Path
from typing import List, Tuple, TypedDict

import librosa
import numpy.typing as npt


class _TimeAlignment(TypedDict):
    start: float
    end: float
    phoneme: str


class WavLabHandler:
    __x_orig: npt.NDArray
    __x: npt.NDArray
    __sr: float
    __monophone_label: List[_TimeAlignment]

    def __init__(self, wav_path: Path, lab_path: Path):
        # load audio
        self.__x_orig, self.__sr = librosa.load(wav_path, sr=None)
        self.__x = self.__x_orig.copy()

        # load label
        with lab_path.open() as f:
            if sum(1 for _ in f) < 3:
                raise ValueError("Invalid label format")

        with lab_path.open() as f:
            self.__monophone_label = []
            for line in f:
                sp = line.split()
                align: _TimeAlignment = {
                    "start": float(sp[0]) / 1e7,
                    "end": float(sp[1]) / 1e7,
                    "phoneme": sp[2],
                }
                self.__monophone_label.append(align)

    def get_wave(self) -> Tuple[npt.NDArray, float]:
        return self.__x, self.__sr

    def get_answer(self) -> Tuple[float, float]:
        return (
            self.__monophone_label[1]["start"],
            self.__monophone_label[-1]["start"],
        )
