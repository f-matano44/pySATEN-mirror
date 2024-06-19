import sys

import numpy as np
import pandas as pd
from inaSpeechSegmenter import Segmenter
from marblenet import marblenet
from rvad import rvad_fast
from soundfile import read, write
from tqdm import tqdm
from utils import gen_noise_signal, load_answer

import pysaten


def _main():
    rand = np.random.default_rng(0)
    result = []

    wavfiles = [
        "wav_and_lab/metan/ITA_recitation_normal_synchronized_wav/recitation",
        "wav_and_lab/sora/ITA_recitation_normal_synchronized_wav/recitation",
        "wav_and_lab/usagi/ITA_recitation_normal_synchronized_wav/normal_recitation_",
    ]

    labfiles = [
        "wav_and_lab/metan/ITA_recitation_normal_label/recitation",
        "wav_and_lab/sora/ITA_recitation_normal_label/rct",
        "wav_and_lab/usagi/ITA_recitation_normal_label/rct",
    ]

    marble = marblenet.MarbleNet()
    for snr in [None, 20, 15, 10, 5, 0, -5]:
        saten = []
        rvad = []
        ina = []
        nemo = []
        print(f"SNR: {snr}", file=sys.stderr)
        for i in tqdm(range(1, 324 + 1)):
            for speaker in [0, 1, 2]:
                ans_s, ans_e = load_answer(f"{labfiles[speaker]}{i:03}.lab")
                wavfile = f"{wavfiles[speaker]}{i:03}.wav"
                x, fs = read(wavfile)
                if snr is not None:
                    x = gen_noise_signal(x, fs, snr, True, rand, ans_s, ans_e)
                # SATEN
                S, E = pysaten.vsed(x, fs)
                E = 0 if S == 0 and abs(E - len(x) / fs) <= 0.01 else E
                saten.append(abs(S - ans_s))
                saten.append(abs(E - ans_e))

                # rVAD
                S, E = rvad_fast.vad(x, fs)
                rvad.append(abs(S - ans_s))
                rvad.append(abs(E - ans_e))

                # ina
                S, E = _ina_speech_segmenter(x, fs)
                ina.append(abs(S - ans_s))
                ina.append(abs(E - ans_e))

                # nemo
                S, E = marble.vad_test(x, fs)
                nemo.append(abs(S - ans_s))
                nemo.append(abs(E - ans_e))

        result.append(
            [
                snr if snr is not None else "Inf",
                f"{np.mean(saten):.3f} ({(1.96*np.std(saten)/np.sqrt(len(saten))):.3f})",
                f"{np.mean(rvad):.3f} ({(1.96*np.std(rvad)/np.sqrt(len(rvad))):.3f})",
                f"{np.mean(ina):.3f} ({(1.96*np.std(ina)/np.sqrt(len(ina))):.3f})",
                f"{np.mean(nemo):.3f} ({(1.96*np.std(nemo)/np.sqrt(len(nemo))):.3f})",
            ]
        )

    df = pd.DataFrame(
        result,
        columns=[
            "snr",
            "saten_mean (95%CI)",
            "rvad_mean (95%CI)",
            "ina_mean (95%CI)",
            "nemo_mean (95%CI)",
        ],
    )
    df.to_csv("test_white_result.csv", index=False)


def _ina_speech_segmenter(x, fs):
    inaSegmenter = Segmenter(detect_gender=False)
    write("temp.wav", x, fs)
    segments = inaSegmenter("temp.wav")
    ina_temp = []
    for segment in segments:
        label, s, e = segment
        if label == "speech":
            ina_temp.append(s)
            ina_temp.append(e)
    S = ina_temp[0] if len(ina_temp) != 0 else 0
    E = ina_temp[-1] if len(ina_temp) != 0 else 0
    return S, E


if __name__ == "__main__":
    _main()
