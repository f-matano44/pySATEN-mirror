import statistics as stat
from pathlib import Path


def _main() -> None:
    length_list = []
    for i in range(1, 101):
        for character in ["zundamon", "tohoku_itako"]:
            lab = f"wav_and_lab/{character}/ITA_emotion_normal_label"
            lab_path = Path(f"{lab}/emoNormal{i:03}.lab")
            with open(lab_path) as f:
                for line in f:
                    start, end, phoneme = line.strip().split(" ")
                    if not (phoneme == "sil" or phoneme == "pau"):
                        length_list.append((float(end) - float(start)) / 1e7)
    print(stat.mean(length_list))
    # 0.078... -> 0.08


if __name__ == "__main__":
    _main()
