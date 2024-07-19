import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

snrlist = [None, 20, 15, 10, 5, 0, -5, -999]
vad = ["pySATEN", "rVAD", "inaSpeechSegmenter", "MarbleNet"]
result: dict = {
    "label": ["Inf", "20", "15", "10", "5", "0", "-5", "-Inf"],
    vad[0]: {"low": [], "med": [], "high": [], "width": []},
    vad[1]: {"low": [], "med": [], "high": [], "width": []},
    vad[2]: {"low": [], "med": [], "high": [], "width": []},
    vad[3]: {"low": [], "med": [], "high": [], "width": []},
}

print(f"SNR, {vad[0]}, {vad[1]}, {vad[2]}, {vad[3]}")
for snr in snrlist:
    trueSNR = (
        (snr if str(snr) != str(None) else "Inf") if str(snr) != str(-999) else "-Inf"
    )
    print(
        f"{trueSNR}, ",
        end="",
    )
    file_path = f"test_{str(snr)}.csv"
    df = pd.read_csv(file_path)
    for method in vad:
        result[method]["low"].append(df[method].quantile(0.25))
        result[method]["med"].append(df[method].median())
        result[method]["high"].append(df[method].quantile(0.75))
        result[method]["width"].append(
            df[method].quantile(0.75) - df[method].quantile(0.25)
        )
        print(f"{float(result[method]['med'][-1]):.3f} ", end="")
        print(
            f"({float(result[method]['low'][-1]):.3f}"
            + f":{float(result[method]['high'][-1]):.3f}"
            + f": {result[method]['width'][-1]:.3f}), ",
            end="",
        )
    print("")

# color: https://contents-open.hatenablog.com/entry/2021/08/19/231157
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
x = np.arange(len(result["label"]))
offset = 0.11
markersize = 7
capsize = 0
linewidth = 2
plt.errorbar(
    x + 1.5 * offset,
    result[vad[3]]["med"],
    yerr=[result[vad[3]]["low"], result[vad[3]]["high"]],
    fmt="*:",
    label=str(vad[3]),
    capsize=capsize,
    markersize=markersize * 1.5,
    linewidth=linewidth,
    color="#FF4B00",
)
plt.errorbar(
    x + 0.5 * offset,
    result[vad[2]]["med"],
    yerr=[result[vad[2]]["low"], result[vad[2]]["high"]],
    fmt="D-.",
    label=str(vad[2]),
    capsize=capsize,
    markersize=markersize,
    linewidth=linewidth,
    color="#005AFF",
)
plt.errorbar(
    x - 0.5 * offset,
    result[vad[1]]["med"],
    yerr=[result[vad[1]]["low"], result[vad[1]]["high"]],
    fmt="s--",
    label=str(vad[1]),
    capsize=capsize,
    markersize=markersize,
    linewidth=linewidth,
    color="#03AF7A",
)
plt.errorbar(
    x - 1.5 * offset,
    result[vad[0]]["med"],
    yerr=[result[vad[0]]["low"], result[vad[0]]["high"]],
    fmt="o-",
    label="SATEN Lv.1",
    capsize=capsize,
    markersize=markersize,
    linewidth=linewidth,
    color="#4DC4FF",
)
plt.xticks(x, result["label"])
plt.legend(handlelength=4, fontsize=12, loc="upper left")
# plt.ylim(-0.5, 3.5)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid()
plt.yscale("log")
plt.ylabel("Error [s]", fontsize=20)
plt.xlabel("Signal-to-Noise Ratio [dB]", fontsize=20)
plt.show()
