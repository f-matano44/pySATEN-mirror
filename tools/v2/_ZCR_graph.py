import librosa
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pysaten.utility.signal import normalize, zero_crossing_rate
from pysaten.v2 import _00_preprocess, _02_zcr

linewidth = 2.27
plt.rcParams["font.size"] = 32
plt.rcParams["font.family"] = "serif"


def _main() -> None:
    wav = "wav_and_lab/tohoku_itako/ITA_emotion_normal_synchronized_wav/emoNormal001.wav"

    # データ読み込み
    y1, fs = librosa.load(wav)
    t1 = np.linspace(0, len(y1) / fs, len(y1))

    hop_length: int = int(0.005 * fs)
    win_length: int = hop_length * 4

    zcr_1 = zero_crossing_rate(y1, win_length, hop_length)
    zcr_t = np.linspace(0, len(y1) / fs, len(zcr_1))

    y2, _ = _00_preprocess(y1, int(fs), 0)
    zcr_2 = 1 - normalize(_02_zcr(y2, fs, win_length, hop_length))

    # データ形式の定義
    data_list: list[tuple[NDArray, NDArray, str]] = [
        (t1, y1, "Wave"),
        (zcr_t, zcr_1, "ZCR"),
        (zcr_t, zcr_2, "Fixed ZCR"),
    ]

    # 最初のx範囲を基準とする
    x_ref = data_list[0][0]
    x_min, x_max = x_ref.min(), x_ref.max()

    # 描画
    fig, axes = plt.subplots(len(data_list), ncols=1, figsize=(8, 6))
    for ax, (x, y, label) in zip(axes, data_list):
        ax.plot(x, y, color="black", linewidth=linewidth)
        ax.set_ylabel(label)
        ax.tick_params(labelleft=False)
        ax.set_xlim(x_min, x_max)
        # Wave の中心を０に
        if label == "Wave":
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
