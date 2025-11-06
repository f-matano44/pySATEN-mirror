from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pysaten.utility.WavLabHandler import WavLabHandler
from pysaten.v2 import vsed_debug_v2

linewidth = 1.15
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "serif"


def _main() -> None:
    # データ形式の定義
    data_list: list[tuple[WavLabHandler, int]] = []

    base_dir = "wav_and_lab/zundamon"
    handler = WavLabHandler(
        Path(f"{base_dir}/ITA_emotion_normal_synchronized_wav/emoNormal001.wav"),
        Path(f"{base_dir}/ITA_emotion_normal_label/emoNormal001.lab"),
    )

    for snr in [20, 10, 0]:
        data_list.append((handler, snr))

    # 描画
    fig, axes = plt.subplots(len(data_list), ncols=1, figsize=(8, 6))
    for ax, (handler, snr) in zip(axes, data_list):
        y, fs = handler.get_noise_signal2(snr, "pink", 0)
        v2 = vsed_debug_v2(y, fs)
        x = np.linspace(0, len(y) / fs, len(y))
        start, end = handler.get_answer()
        ax.axvspan(start, end, color="lightgray")
        ax.plot(x, y, color="black", linewidth=linewidth)
        ax.axvline(
            x=start + ((end - start) / 2),
            color="black",
            linewidth=linewidth,
        )
        ax.axvline(
            x=_estimate_center(v2.y_zcr, len(y) / fs),
            color="black",
            linestyle="--",
            linewidth=linewidth,
        )
        ax.set_title(f"SNR {snr} [dB]")
        ax.tick_params(labelleft=False, axis="y", length=0)
        ax.set_xlim(x.min(), x.max())
        # Wave の中心を０に
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


def _estimate_center(zcr: NDArray, length: float) -> float:
    return (np.average(np.arange(zcr.size), weights=(1.0 - zcr)) / zcr.size) * length


if __name__ == "__main__":
    _main()
