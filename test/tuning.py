from pysaten import pysaten
import numpy as np
from numpy.random import default_rng
from utils import load_answer, gen_noise_signal
from tqdm import tqdm
import soundfile as sf
import statistics as stat
import pandas as pd


def _main():
    rand = default_rng(0)
    result = []
    for zcs_threshold in np.linspace(0.01, 1, 100):
        for rms_threshold in np.linspace(0.005, 0.5, 100):
            this_param = []
            for i in tqdm(range(1, 101)):
                for chara in ["zundamon", "tohoku_itako"]:
                    ans_s, ans_e = load_answer(
                        f"wav_and_lab/{chara}/ITA_emotion_normal_label/emoNormal{i:03}.lab")
                    x, fs = sf.read(
                        f"wav_and_lab/{chara}/ITA_emotion_normal_synchronized_wav/emoNormal{i:03}.wav")
                    x = gen_noise_signal(x, fs, 25, False, rand, ans_s, ans_e)
                    _, _, S, E, _, _, _, _, _ = pysaten.vsed_debug(
                        x, fs,
                        rms_threshold=rms_threshold,
                        zcs_threshold=zcs_threshold)
                    this_param.append(abs(S - ans_s))
                    this_param.append(abs(E - ans_e))
            result.append([rms_threshold, zcs_threshold, stat.mean(this_param)])
            print(f"{rms_threshold:.3f}, {zcs_threshold:.2f} -> {stat.mean(this_param):.3f}\n")
    
    pw_csv = "tuning_result.csv"
    df = pd.DataFrame(result, columns=["rme_threshold", "zcs_threshold", "error_s"])
    df.to_csv(pw_csv, index=False)

    df = pd.read_csv(pw_csv)
    min_row = df[df["error_s"] == df["error_s"].min()]
    print(f"{min_row}")


if __name__ == "__main__":
    _main()