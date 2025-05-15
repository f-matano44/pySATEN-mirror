import statistics as stat
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm
from utils import gen_noise_signal

import pysaten
from pysaten.utility.WavLabHandler import WavLabHandler


def _main():
    rand = default_rng(0)
    result = []
    for zcr_thres in np.linspace(0.1, 1.0, 10):
        for rms_thres in np.linspace(0.05, 0.5, 10):
            this_param = []
            not_abs_error = []
            for i in tqdm(range(1, 101)):
                for character in ["zundamon", "tohoku_itako"]:
                    wav = f"wav_and_lab/{character}/ITA_emotion_normal_synchronized_wav"
                    wav_path = Path(f"{wav}/emoNormal{i:03}.wav")
                    lab = f"wav_and_lab/{character}/ITA_emotion_normal_label"
                    lab_path = Path(f"{lab}/emoNormal{i:03}.lab")

                    handler = WavLabHandler(wav_path, lab_path)
                    x, fs = librosa.load(wav_path, sr=None)
                    ans_s, ans_e = handler.get_answer()

                    x = gen_noise_signal(x, fs, 25, False, rand, ans_s, ans_e)
                    _, _, _, _, S, E, _, _, _ = pysaten.vsed_debug(
                        x, fs, rms_threshold=rms_thres, zcr_threshold=zcr_thres
                    )

                    this_param.append(abs(S - ans_s))
                    this_param.append(abs(E - ans_e))
                    not_abs_error.append(S - ans_s)
                    not_abs_error.append(ans_e - E)
            result.append(
                [
                    rms_thres,
                    zcr_thres,
                    stat.mean(this_param),
                    stat.mean(not_abs_error),
                ]
            )
            print(f"{rms_thres:.3f}, {zcr_thres:.2f} -> {stat.mean(this_param):.3f}\n")

    pw_csv = "tuning_result1.csv"
    df = pd.DataFrame(
        result, columns=["rme_threshold", "zcr_threshold", "error_s", "not_abs_error_s"]
    )
    df.to_csv(pw_csv, index=False)

    df = pd.read_csv(pw_csv)
    min_row = df[df["error_s"] == df["error_s"].min()]
    print(f"{min_row}")


if __name__ == "__main__":
    _main()
