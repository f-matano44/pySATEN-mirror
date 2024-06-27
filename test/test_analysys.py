import pandas as pd

print("SNR, pySATEN, rVAD, inaSpeechSegmenter, MarbleNet")
for snr in [None, 20, 15, 10, 5, 0, -5, -999]:
    print(
        f"{(snr if str(snr) != str(None) else 'Inf') if str(snr) != str(-999) else '-Inf'}, ",
        end="",
    )
    file_path = f"test_{str(snr)}.csv"
    df = pd.read_csv(file_path)
    for method in ["pySATEN", "rVAD", "inaSpeechSegmenter", "MarbleNet"]:
        print(f"{df[method].median():.3f} ", end="")
        print(
            f"({df[method].quantile(0.25):.3f}:{df[method].quantile(0.75):.3f}), ",
            end="",
        )
    print("")
