import sys

import numpy as np
import pandas as pd
from inaSpeechSegmenter import Segmenter
from rvad import rvad_fast
from soundfile import read, write
from tqdm import tqdm
from utils import gen_noise_signal, load_answer

sys.path.append("../src")
import pysaten


def _main():
    rand = np.random.default_rng(0)
    inaSegmenter = Segmenter(detect_gender=False)
    result = []
    for snr in [None, 20, 15, 10, 5, 0, -5]:
        saten = []
        rvad = []
        ina = []
        print(f"SNR: {snr}", file=sys.stderr)
        for i in tqdm(range(1, 101)):  # 324+1)):
            ans_s, ans_e = load_answer(
                f"wav_and_lab/usagi/ITA_recitation_normal_label/rct{i:03}.lab"
            )
            wavfile = f"wav_and_lab/usagi/ITA_recitation_normal_synchronized_wav/normal_recitation_{i:03}.wav"
            x, fs = read(wavfile)
            if snr is not None:
                x = gen_noise_signal(x, fs, snr, True, rand, ans_s, ans_e)
            # SATEN
            S, E = pysaten.vsed(x, fs)
            E = 0 if S <= 0.01 and 0.01 >= abs(E - len(x) / fs) else E
            saten.append(abs(S - ans_s))
            saten.append(abs(E - ans_e))

            # rVAD
            S, E = rvad_fast.vad(x, fs)
            rvad.append(abs(S - ans_s))
            rvad.append(abs(E - ans_e))

            # ina
            tempwav = "temp.wav"
            write(tempwav, x, fs)
            segments = inaSegmenter(tempwav)
            ina_temp = []
            for segment in segments:
                label, s, e = segment
                if label == "speech":
                    ina_temp.append(s)
                    ina_temp.append(e)
            try:
                start = abs(ina_temp[0] - ans_s)
                end = abs(ina_temp[-1] - ans_e)
            except IndexError:
                start = abs(0 - ans_s)
                end = abs(0 - ans_s)
            ina.append(start)
            ina.append(end)

        result.append(
            [
                snr,
                f"{np.mean(saten):.3f} ({np.std(saten):.3f})",
                f"{np.mean(rvad):.3f} ({np.std(rvad):.3f})",
                f"{np.mean(ina):.3f} ({np.std(ina):.3f})",
            ]
        )

    df = pd.DataFrame(
        result,
        columns=[
            "snr",
            "saten_mean (std)",
            "rvad_mean (std)",
            "ina_mean (std)",
        ],
    )
    df.to_csv("test_result.csv", index=False)


if __name__ == "__main__":
    _main()
