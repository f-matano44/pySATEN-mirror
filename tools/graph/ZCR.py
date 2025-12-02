from configparser import ConfigParser

import librosa
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pysaten.utility.signal import zero_crossing_rate
from pysaten.v2 import vsed_debug_v2

config = ConfigParser()
config.read("config.ini")

linewidth = float(config["line"]["width"])
plt.rcParams["font.size"] = int(config["font"]["size"])
plt.rcParams["font.family"] = config["font"]["family"]


def _main() -> None:
    wav = "wav_and_lab/zundamon/ITA_emotion_normal_synchronized_wav/emoNormal001.wav"
    lab = "wav_and_lab/zundamon/ITA_emotion_normal_label/emoNormal001.lab"

    # データ読み込み: wav
    y, fs = librosa.load(wav)
    t = np.linspace(0, len(y) / fs, len(y))
    # データ読み込み: lab
    with open(lab) as f:
        ans = [line.rstrip("\n").split(" ") for line in f]
    start = float(ans[0][1]) / 1e7
    end = float(ans[-1][0]) / 1e7

    hop_length: int = int(0.005 * fs)
    win_length: int = hop_length * 4

    zcr_1 = zero_crossing_rate(y, win_length, hop_length)
    zcr_t = np.linspace(0, len(y) / fs, len(zcr_1))

    v2 = vsed_debug_v2(y, fs)

    # データ形式の定義
    data_list: list[tuple[NDArray, NDArray, str]] = [
        (t, y, "(c) Clean Speech Waveform"),
        (zcr_t, zcr_1, "(d) Conventional ZCR"),
        (v2.feats_timestamp, v2.y_zcr, "(e) Fixed ZCR"),
    ]

    # 最初のx範囲を基準とする
    x_ref = data_list[0][0]
    x_min, x_max = x_ref.min(), x_ref.max()

    # 描画
    fig, axes = plt.subplots(len(data_list), ncols=1, figsize=(8, 6))
    for ax, (x, y, label) in zip(axes, data_list):
        ax.axvspan(start, end, color="lightgray")
        ax.plot(x, y, color="black", linewidth=linewidth)
        ax.set_title(label, fontsize=int(config["font"]["size"]) * 1.1)
        ax.tick_params(labelleft=False, axis="y", length=0)
        ax.set_xlim(x_min, x_max)
        # Wave の中心を０に
        if "Wave" in label:
            amp = np.max(np.abs(y))
            ax.set_ylim(-amp * 1.15, amp * 1.15)
            ax.set_ylabel("Amp.")
        else:
            ax.set_ylabel("ZCR")
        # 枠線の太さを調整
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth * 0.7)

    # x軸ラベルは最下段のみ
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time [s]", color="black")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.show()


if __name__ == "__main__":
    _main()
