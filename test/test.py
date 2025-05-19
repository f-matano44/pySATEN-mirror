import sys
from math import inf
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import numpy as np
import pandas as pd
from inaSpeechSegmenter import Segmenter
from marblenet import marblenet
from rvad import rvad_fast
from soundfile import write
from tqdm import tqdm

import pysaten
from pysaten.utility.WavLabHandler import WavLabHandler

inaSegmenter = Segmenter(detect_gender=False)
marble = marblenet.MarbleNet()


def _main():
    rand = np.random.default_rng(0)

    wav_files = [
        "wav_and_lab/metan/ITA_recitation_normal_synchronized_wav/recitation",
        "wav_and_lab/sora/ITA_recitation_normal_synchronized_wav/recitation",
        "wav_and_lab/usagi/ITA_recitation_normal_synchronized_wav/normal_recitation_",
    ]

    lab_files = [
        "wav_and_lab/metan/ITA_recitation_normal_label/recitation",
        "wav_and_lab/sora/ITA_recitation_normal_label/rct",
        "wav_and_lab/usagi/ITA_recitation_normal_label/rct",
    ]

    with TemporaryDirectory() as temp_dir:
        for snr in [inf, 20, 15, 10, 5, 0, -5, -inf]:
            saten = []
            rvad = []
            ina = []
            nemo = []
            print(f"SNR: {snr}", file=sys.stderr)
            for i in tqdm(range(1, 324 + 1)):
                for speaker in [0, 1, 2]:
                    # load wav and label
                    wav_path = Path(f"{wav_files[speaker]}{i:03}.wav")
                    lab_path = Path(f"{lab_files[speaker]}{i:03}.lab")
                    handler = WavLabHandler(wav_path, lab_path)

                    # create noised signal
                    x, fs = handler.get_noise_signal(
                        snr, True, True, int(rand.integers(0, 20250515))
                    )

                    # save noise signal
                    temp_wav = f"{temp_dir}/temp.wav"
                    write(temp_wav, x, fs)
                    temp_wav_16k = f"{temp_dir}/temp_16k.wav"
                    x_16k = librosa.resample(x, orig_sr=fs, target_sr=16000)
                    write(temp_wav_16k, x_16k, 16000)

                    # get answer label
                    ans_s, ans_e = handler.get_answer()

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
                    S, E = _ina_speech_segmenter(temp_wav)
                    ina.append(abs(S - ans_s))
                    ina.append(abs(E - ans_e))

                    # nemo
                    S, E = marble.vad_test(temp_wav_16k)
                    nemo.append(abs(S - ans_s))
                    nemo.append(abs(E - ans_e))

            pd.DataFrame(
                {
                    "pySATEN": saten,
                    "rVAD": rvad,
                    "inaSpeechSegmenter": ina,
                    "MarbleNet": nemo,
                }
            ).to_csv(f"test_{str(snr)}.csv", index=False)


def _ina_speech_segmenter(temp_wav: Path):
    segments = inaSegmenter(temp_wav)
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
