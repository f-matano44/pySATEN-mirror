from configparser import ConfigParser

import librosa
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

config = ConfigParser()
config.read("config.ini")

linewidth = float(config["line"]["width"])
plt.rcParams["font.size"] = int(config["font"]["size"])
plt.rcParams["font.family"] = config["font"]["family"]


def _main() -> None:
    wav = "wav_and_lab/tohoku_itako/ITA_emotion_normal_synchronized_wav/emoNormal011.wav"
    lab = "wav_and_lab/tohoku_itako/ITA_emotion_normal_label/emoNormal011.lab"

    # データ読み込み: wav
    y1, fs = librosa.load(wav)
    t1 = np.linspace(0, len(y1) / fs, len(y1))
    # データ読み込み: lab
    with open(lab) as f:
        ans = [line.rstrip("\n").split(" ") for line in f]
    start = float(ans[0][1]) / 1e7
    end = float(ans[-1][0]) / 1e7

    # データ形式の定義
    data_list: list[tuple[NDArray, NDArray, str]] = [
        (t1, y1, "(a) VAD"),
        (t1, y1, "(b) VSED"),
    ]

    # 最初のx範囲を基準とする
    x_ref = data_list[0][0]
    x_min, x_max = x_ref.min(), x_ref.max()

    # 描画
    fig, axes = plt.subplots(len(data_list), ncols=1, figsize=(8, 6))
    for ax, (x, y, label) in zip(axes, data_list):
        if "VAD" in label:
            for line in ans:
                if not (line[2] == "sil" or line[2] == "pau"):
                    ax.axvspan(
                        int(line[0]) / 1e7,
                        int(line[1]) / 1e7,
                        color="lightgray",
                    )
        elif "VSED" in label:
            ax.axvspan(start, end, color="lightgray")
        ax.plot(x, y, color="black", linewidth=linewidth)
        ax.set_title(label, fontsize=int(config["font"]["size"]))
        ax.set_ylabel("Amp.")
        ax.tick_params(labelleft=False, axis="y", length=0)
        ax.set_xlim(x_min, x_max)
        amp = np.max(np.abs(y))
        ax.set_ylim(-amp * 1.15, amp * 1.15)
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
