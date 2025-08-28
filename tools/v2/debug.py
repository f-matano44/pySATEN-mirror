from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from librosa.feature import rms
from tqdm import tqdm

import pysaten
from pysaten.utility.WavLabHandler import WavLabHandler


def _main() -> None:
    fs = 48000
    rand = np.random.default_rng(0)

    base_dir = "wav_and_lab/tohoku_itako/ITA_emotion_normal"
    for i in tqdm(range(7, 8)):
        handler = WavLabHandler(
            wav_path=Path(f"{base_dir}_synchronized_wav/emoNormal{i:03d}.wav"),
            lab_path=Path(f"{base_dir}_label/emoNormal{i:03d}.lab"),
        )
        x, sr = handler.get_noise_signal(
            snr=-5, is_white=False, with_pulse=False, noise_seed=0
        )
        cx, _ = handler.get_signal()

        result = pysaten.v2.vsed_debug_v2(
            x, fs, noise_seed=int(rand.integers(0, 20250608))
        )

        # time -> array
        section1 = np.zeros(len(x)).astype(bool)
        section1[int(result.start1_s * fs) : int(result.end1_s * fs)] = True
        section2 = np.zeros(len(x)).astype(bool)
        section2[int(result.start2_s * fs) : int(result.end2_s * fs)] = True
        section3 = np.zeros(len(x)).astype(bool)
        section3[int(result.start3_s * fs) : int(result.end3_s * fs)] = True

        # draw graph
        plt.clf()
        plt.rcParams["font.family"] = "Liberation Serif"
        plt.rcParams["font.size"] = 16
        x_t = np.linspace(0, len(x) / fs, len(x))
        # plt.fill_between(x_t, -1, 1, where=section1.tolist(), color="gray", alpha=0.9)
        # plt.fill_between(x_t, -1, 1, where=section2.tolist(), color="gray", alpha=0.6)
        plt.fill_between(x_t, -1, 1, where=section3.tolist(), color="gray", alpha=0.9)
        plt.plot(x_t, x, color="black")
        plt.plot(x_t, cx, color="white")
        # plt.plot(result.feats_timestamp, result.y_rms, color="gray")
        # plt.plot(result.feats_timestamp, result.y_zcr, color="gray")
        # plt.plot(x_t, np.ones(len(x)) * result.zcr_threshold)
        # plt.plot(x_t, np.ones(len(x)) * result.rms_threshold)
        # plt.plot(result.feats_timestamp, result.bell, color="gray")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude & Feature")
        plt.ylim(-0.5, 0.5)
        plt.xlim(0, 11.5)
        # plt.savefig(f"graph/{i:03d}.pdf")
        plt.show()


def add_noise_to_signal(signal, noise, desired_snr_db):
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    signal_power = (np.sort(rms(y=signal)[0]) ** 2)[-2]
    noise_power = np.mean(noise**2)
    scaling_factor = np.sqrt(signal_power / (desired_snr_linear * noise_power))
    noise_adjusted = noise * scaling_factor
    return signal + noise_adjusted


def calculate_snr_db(signal, noise):
    signal_power = (np.sort(rms(y=signal)[0]) ** 2)[-2]
    noise_power = (np.sort(rms(y=noise)[0]) ** 2)[1]
    return 10 * np.log10(signal_power / noise_power)


if __name__ == "__main__":
    _main()
