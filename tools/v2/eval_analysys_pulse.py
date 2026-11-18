import numpy as np
import pandas as pd

color = "pulse"

snr_list = ["without_pulse", "with_pulse"]
vad = [
    "pySATEN2",
    "rVAD",
    "inaSpeechSegmenter",
    "Silero_vad",
    "SpeechBrain",
    "WebRTC",
    "WhisperX",
    "SE + RMS",
]
result: dict[str, list[float | str]] = {
    vad[0]: [],
    vad[1]: [],
    vad[2]: [],
    vad[3]: [],
    vad[4]: [],
    vad[5]: [],
    vad[6]: [],
}

table_header = (
    f"|SNR|{vad[0]}|{vad[1]}|{vad[2]}|{vad[3]}|{vad[4]}|{vad[5]}|"
    f"{vad[6]}|{vad[7]}|\n"
)
table_alignment = "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"

with (
    open(f"{color}_result.md", "w") as result_file,
    open(f"{color}_rejection_rate_result.md", "w") as rejection_rate_file,
):
    for output_file in (result_file, rejection_rate_file):
        output_file.write(table_header)
        output_file.write(table_alignment)

    for snr in snr_list:
        result_file.write(f"|{snr}|")
        rejection_rate_file.write(f"|{snr}|")
        file_path = f"results/{snr}.csv"
        df = pd.read_csv(file_path)
        if len(df) == 0 or len(df) % 2 != 0:
            raise ValueError(
                f"The number of result rows must be positive and even: {file_path}"
            )

        answer_np = df["answer"].to_numpy(dtype=float)
        answer_start = answer_np[0::2]
        answer_end = answer_np[1::2]
        speech_durations = answer_end - answer_start
        if np.any(speech_durations <= 0):
            raise ValueError(f"Speech durations must be positive: {file_path}")

        for method in vad:
            method_np = df[method].to_numpy(dtype=float)
            mask = ~np.isnan(method_np)
            masked_diff = answer_np[mask] - method_np[mask]
            diff_median = (
                f"{np.median(np.abs(masked_diff)):.3f}"
                if masked_diff.size > 0
                else "None"
            )
            recognition_rate = masked_diff.size / method_np.size
            result_file.write(f"{diff_median} ({recognition_rate:.2f})|")

            estimated_start = method_np[0::2]
            estimated_end = method_np[1::2]
            recognized = ~np.isnan(estimated_start) & ~np.isnan(estimated_end)
            overlap_durations = np.maximum(
                0.0,
                np.minimum(answer_end[recognized], estimated_end[recognized])
                - np.maximum(answer_start[recognized], estimated_start[recognized]),
            )
            rejection_rates = np.ones_like(speech_durations)
            rejection_rates[recognized] = (
                1.0 - overlap_durations / speech_durations[recognized]
            )
            rejection_rate_mean = np.mean(rejection_rates)
            rejection_rate_file.write(f"{rejection_rate_mean:.3f}|")
        result_file.write("\n")
        rejection_rate_file.write("\n")
