from pathlib import Path
from time import time

from tqdm import tqdm

import pysaten
from pysaten.utility.WavLabHandler import WavLabHandler

"""
$ python speed_test.py
python speed_test.py
wav_lab: 0.9626076221466064
noise: 417.0023241043091
saten: 15.571624040603638
"""


def _main():
    before = time()
    for i in tqdm(range(1, 101)):
        for character in ["zundamon", "tohoku_itako"]:
            wav = f"wav_and_lab/{character}/ITA_emotion_normal_synchronized_wav"
            wav_path = Path(f"{wav}/emoNormal{i:03}.wav")
            lab = f"wav_and_lab/{character}/ITA_emotion_normal_label"
            lab_path = Path(f"{lab}/emoNormal{i:03}.lab")

            handler = WavLabHandler(wav_path, lab_path)
    after = time()
    print(f"wav_lab: {after - before}")

    before = time()
    for i in tqdm(range(1, 101)):
        for character in ["zundamon", "tohoku_itako"]:
            x, fs = handler.get_noise_signal(25, False, False, 0)
    after = time()
    print(f"noise: {after - before}")

    before = time()
    for i in tqdm(range(1, 101)):
        _, _ = pysaten.vsed(x, fs)
        _, _ = pysaten.vsed(x, fs)
    after = time()
    print(f"saten: {after - before}")


if __name__ == "__main__":
    _main()
