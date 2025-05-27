from math import inf
from pathlib import Path
from typing import List, Tuple, TypedDict

import librosa
import numpy as np
import numpy.random as rand
import numpy.typing as npt

from .color_noise import pink as pk_noise
from .color_noise import white as wh_noise


class _TimeAlignment(TypedDict):
    start: float
    end: float
    phoneme: str


class WavLabHandler:
    __x: npt.NDArray
    __sr: float
    __monophone_label: List[_TimeAlignment]

    def __init__(self, wav_path: Path, lab_path: Path):
        # load audio
        self.__x, self.__sr = librosa.load(wav_path, sr=None)

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

    def get_answer(self) -> Tuple[float, float]:
        return (
            self.__monophone_label[1]["start"],
            self.__monophone_label[-1]["start"],
        )

    def get_signal(self) -> Tuple[npt.NDArray, float]:
        return self.__x, self.__sr

    def get_noise_signal(
        self, snr: float, is_white: bool, with_pulse: bool, noise_seed: int
    ) -> Tuple[npt.NDArray, int]:
        x = self.__x.copy()
        sr = int(self.__sr)
        ans_s_sec, ans_e_sec = self.get_answer()
        speech_start_idx: int = int(ans_s_sec * sr)
        speech_end_idx: int = int(ans_e_sec * sr)

        # generate noise (white or pink)
        noise = (
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
        rand.seed(noise_seed)
        pulse = rand.random(2) - 0.5 * 2
        # determine index adding pulse noise
        start_pulse_index = np.random.randint(0, speech_start_idx)
        end_pulse = np.random.randint(speech_end_idx, len(x) - 1)
        # add pulse noise
        noised_x[start_pulse_index] = pulse[0]
        noised_x[end_pulse] = pulse[1]
        return noised_x, sr

    @staticmethod
    def __determine_noise_scale(
        signal: npt.NDArray, noise: npt.NDArray, desired_snr_db: int
    ) -> float:
        desired_snr_linear = 10 ** (desired_snr_db / 10)
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        scaling_factor = np.sqrt(signal_power / (desired_snr_linear * noise_power))
        return float(scaling_factor)
