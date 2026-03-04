from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from numpy.typing import NDArray

config = ConfigParser()
config.read(Path(__file__).with_name("config.ini"))

linewidth = float(config["line"]["width"])
plt.rcParams["font.size"] = int(config["font"]["size"])
plt.rcParams["font.family"] = config["font"]["family"]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO = (BASE_DIR / "../v2/fan_noise.flac").resolve()


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="長時間スペクトルを表示する")
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help="入力音声ファイル",
    )
    parser.add_argument(
        "--duration-sec",
        type=int,
        default=5,
        help="スペクトル計算に使う秒数",
    )
    return parser.parse_args()


def _compute_long_term_spectrum(
    y: NDArray[np.float64], fs: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    window = np.hanning(y.shape[0]).astype(np.float64)
    y_windowed = y * window
    spectrum = np.fft.rfft(y_windowed)
    power = np.abs(spectrum) ** 2
    eps = np.finfo(np.float64).eps
    power_db = 10.0 * np.log10(np.maximum(power, eps))
    power_db -= np.max(power_db)
    freq = np.fft.rfftfreq(y.shape[0], d=1.0 / fs)
    return freq.astype(np.float64), power_db.astype(np.float64)


def _extract_first_duration(
    y: NDArray[np.float64], fs: int, duration_sec: int
) -> NDArray[np.float64]:
    if duration_sec <= 0:
        msg = "--duration-sec は1以上にしてください"
        raise ValueError(msg)

    sample_count = min(y.shape[0], fs * duration_sec)
    if sample_count < 2:
        msg = "音声長が短すぎるためスペクトルを計算できない"
        raise ValueError(msg)
    return y[:sample_count]


def _main() -> None:
    args = _parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        msg = f"音声ファイルが見つからない: {audio_path}"
        raise FileNotFoundError(msg)

    y_raw, fs = librosa.load(audio_path, sr=None, mono=True)
    y = y_raw.astype(np.float64)
    y_clip = _extract_first_duration(y=y, fs=int(fs), duration_sec=args.duration_sec)
    freq, mean_power_db = _compute_long_term_spectrum(y=y_clip, fs=int(fs))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(freq[1:], mean_power_db[1:], color="black", linewidth=linewidth)
    ax.set_xlim(20, fs / 2)
    ax.set_ylim(-100, 3)
    xticks = [tick for tick in (100.0, 1000.0, 10000.0) if tick <= fs / 2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(tick)}" for tick in xticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative Power [dB]")

    for spine in ax.spines.values():
        spine.set_linewidth(linewidth * 0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _main()
