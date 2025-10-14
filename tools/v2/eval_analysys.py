from math import inf

import numpy as np
import pandas as pd

color = "white"

snr_list = [inf, 20, 15, 10, 5, 0, -5, -inf]
vad = [
    "pySATEN2",
    "rVAD",
    "inaSpeechSegmenter",
    "Silero_vad",
    "SpeechBrain",
    "WebRTC",
    "WhisperX",
]
result: dict[str, list] = {
    "label": ["Inf", "20", "15", "10", "5", "0", "-5", "-Inf"],
    vad[0]: [],
    vad[1]: [],
    vad[2]: [],
    vad[3]: [],
    vad[4]: [],
    vad[5]: [],
    vad[6]: [],
}

with open(f"{color}_result.md", "w") as f:
    f.write(f"|SNR|{vad[0]}|{vad[1]}|{vad[2]}|{vad[3]}|{vad[4]}|{vad[5]}|{vad[6]}|\n")
    f.write("|:---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for snr in snr_list:
        f.write(f"|{snr}|")
        file_path = f"results/{color}_{str(snr)}.csv"
        df = pd.read_csv(file_path)
        for method in vad:
            method_np = df[method].to_numpy(dtype=float)
            answer_np = df["answer"].to_numpy(dtype=float)
            mask = ~np.isnan(method_np)
            masked_diff = answer_np[mask] - method_np[mask]
            diff_median = (
                "{:.3f}".format(np.median(np.abs(masked_diff)))
                if masked_diff.size > 0
                else "None"
            )
            f.write(f"{diff_median} ({masked_diff.size / df[method].size:.2f})|")
        f.write("\n")
