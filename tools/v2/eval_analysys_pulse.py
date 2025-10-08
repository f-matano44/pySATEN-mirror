import numpy as np
import pandas as pd

color = "pulse"

snr_list = ["without_pulse", "with_pulse"]
vad = [
    "pySATEN2",
    "rVAD",
    "inaSpeechSegmenter",
    "Silero_vad",
    "WhisperX",
]
result: dict[str, list] = {
    vad[0]: [],
    vad[1]: [],
    vad[2]: [],
    vad[3]: [],
    vad[4]: [],
}

with open(f"{color}_result.md", "w") as f:
    f.write(f"|SNR|{vad[0]}|{vad[1]}|{vad[2]}|{vad[3]}|{vad[4]}|\n")
    f.write("|:---:|---:|---:|---:|---:|---:|\n")
    for snr in snr_list:
        f.write(f"|{snr}|")
        file_path = f"results/{str(snr)}.csv"
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
