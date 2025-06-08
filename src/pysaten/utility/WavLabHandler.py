from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Final

import librosa
import numpy as np
import numpy.random as rand
import numpy.typing as npt

from .color_noise import pink as pk_noise
from .color_noise import white as wh_noise


@dataclass
class _TimeAlignment:
    def __init__(self, start, end, phoneme) -> None:
        self.start: Final[float] = start
        self.end: Final[float] = end
        self.phoneme: Final[str] = phoneme


class WavLabHandler:
    __x: Final[npt.NDArray[np.floating]]
    __sr: Final[float]
    __monophone_label: Final[list[_TimeAlignment]]

    def __init__(self, wav_path: Path, lab_path: Path) -> None:
        # load audio
        self.__x, self.__sr = librosa.load(wav_path, sr=None)
        self.__x.flags.writeable = False  # x is immutable

        # load label
        with lab_path.open() as f:
            self.__monophone_label = []
            for line in f:
                sp: list[str] = line.split()
                align = _TimeAlignment(
                    start=int(sp[0]) / 1e7,
                    end=int(sp[1]) / 1e7,
                    phoneme=sp[2],
                )
                self.__monophone_label.append(align)
        if len(self.__monophone_label) < 3:
            raise ValueError("Invalid label format")

    def get_answer(self) -> tuple[float, float]:
        return (
            self.__monophone_label[1].start,
            self.__monophone_label[-1].start,
        )

    def get_signal(self) -> tuple[npt.NDArray[np.floating], float]:
        return self.__x, self.__sr

    def get_noise_signal(
        self, snr: float, is_white: bool, with_pulse: bool, noise_seed: int
    ) -> tuple[npt.NDArray, int]:
        x: npt.NDArray[np.floating] = self.__x.copy()
        sr: int = int(self.__sr)
        ans_s_sec, ans_e_sec = self.get_answer()
        speech_start_idx: int = int(ans_s_sec * sr)
        speech_end_idx: int = int(ans_e_sec * sr)

        # generate noise (white or pink)
        noise: npt.NDArray[np.floating] = (
            (
                wh_noise(len(x), noise_seed)
                if is_white
                else pk_noise(len(x), sr, noise_seed)
            )
            .cpu()
            .numpy()
        )

        # mix stationary noise and signal (in specified snr)
        if snr == inf:
            noised_x = x
        elif snr == -inf:
            noised_x = noise
        else:
            noise_scale = WavLabHandler.__determine_noise_scale(
                x[speech_start_idx:speech_end_idx], noise, int(snr)
            )
            noised_x = x + noise * noise_scale

        # add pulse noise
        if with_pulse:
            rand.seed(noise_seed)
            pulse: npt.NDArray[np.floating] = rand.random(2) - 0.5 * 2
            # determine index adding pulse noise
            start_pulse_index: int = np.random.randint(0, speech_start_idx)
            end_pulse_index: int = np.random.randint(speech_end_idx, len(x) - 1)
            # add pulse noise
            noised_x[start_pulse_index] = pulse[0]
            noised_x[end_pulse_index] = pulse[1]
        return noised_x, sr

    @staticmethod
    def __determine_noise_scale(
        signal: npt.NDArray[np.floating],
        noise: npt.NDArray[np.floating],
        desired_snr_db: int,
    ) -> float:
        desired_snr_linear: float = 10 ** (desired_snr_db / 10)
        signal_power: float = np.mean(signal**2)
        noise_power: float = np.mean(noise**2)
        scaling_factor: float = np.sqrt(signal_power / (desired_snr_linear * noise_power))
        return scaling_factor
