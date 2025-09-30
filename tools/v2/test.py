import sys
from math import inf
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import whisperx
from inaSpeechSegmenter import Segmenter
from rVADfast import rVADfast
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from soundfile import write
from tqdm import tqdm

from pysaten.utility.WavLabHandler import WavLabHandler
from pysaten.v2 import vsed_debug_v2

rvad = rVADfast()
inaSegmenter = Segmenter(detect_gender=False)
silero = load_silero_vad()
whisper_model = whisperx.load_model("large-v3", "cpu", compute_type="int8", language="ja")

noise_type = "pink"


def _main():
    rand = np.random.default_rng(0)

    wav_files = [
        "wav_and_lab/metan/ITA_recitation_normal_synchronized_wav/recitation",
        "wav_and_lab/sora/ITA_recitation_normal_synchronized_wav/recitation",
        "wav_and_lab/usagi/ITA_recitation_normal_synchronized_wav/normal_recitation_",
    ]

    lab_files = [
        "wav_and_lab/metan/ITA_recitation_normal_label/recitation",
        "wav_and_lab/sora/ITA_recitation_normal_label/rct",
        "wav_and_lab/usagi/ITA_recitation_normal_label/rct",
    ]

    with TemporaryDirectory() as temp_dir:
        for snr in [inf, 20, 15, 10, 5, 0, -5, -inf]:
            ans_list = []
            saten2_list = []
            rvad_list = []
            ina_list = []
            silero_list = []
            whisper_list = []

            print(f"SNR: {snr}", file=sys.stderr)
            for i in tqdm(range(1, 324 + 1)):
                for speaker in [0, 1, 2]:
                    # load wav and label
                    wav_path = Path(f"{wav_files[speaker]}{i:03}.wav")
                    lab_path = Path(f"{lab_files[speaker]}{i:03}.lab")
                    handler = WavLabHandler(wav_path, lab_path)

                    # create noised signal
                    x, fs = handler.get_noise_signal2(
                        snr, noise_type, int(rand.integers(0, 20250922))
                    )

                    # save noise signal
                    temp_wav = f"{temp_dir}/temp.wav"
                    write(temp_wav, x, fs)

                    # get answer label
                    ans_list.extend(handler.get_answer())

                    # SATEN2
                    saten2_list.extend(vsed_debug_v2(x, fs, noise_seed=i).get_result())

                    # rVAD
                    rvad_list.extend(_rvad_fast(x, fs))

                    # ina
                    ina_list.extend(_ina_speech_segmenter(temp_wav))

                    # silero vad
                    silero_list.extend(_silero_vad(temp_wav))

                    # whisperx
                    whisper_list.extend(_whisper(temp_wav))

            pd.DataFrame(
                {
                    "answer": ans_list,
                    "pySATEN2": saten2_list,
                    "rVAD": rvad_list,
                    "inaSpeechSegmenter": ina_list,
                    "Silero_vad": silero_list,
                    "WhisperX": whisper_list,
                }
            ).to_csv(f"test_result/{noise_type}_{str(snr)}.csv", index=False)


def _rvad_fast(x, fs):
    label, timestamp = rvad(x, fs)
    if not any(label):
        return None, None
    else:
        valid = label * timestamp
        valid_only = valid[0 < valid]
        return valid_only[0], valid_only[-1]


def _ina_speech_segmenter(audio_file: Path):
    segments = inaSegmenter(audio_file)
    ina_temp = []
    for segment in segments:
        label, s, e = segment
        if label == "speech":
            ina_temp.append(s)
            ina_temp.append(e)
    S = ina_temp[0] if len(ina_temp) != 0 else None
    E = ina_temp[-1] if len(ina_temp) != 0 else None
    return S, E


def _silero_vad(audio_file):
    wav = read_audio(audio_file)
    speech_timestamps = get_speech_timestamps(wav, silero, return_seconds=True)

    seg = []
    for segment in speech_timestamps:
        # print(segment)
        seg.append(segment["start"])
        seg.append(segment["end"])

    S = seg[0] if len(seg) != 0 else None
    E = seg[-1] if len(seg) != 0 else None
    return S, E


def _whisper(audio_file):
    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, batch_size=16)

    seg = []
    for segment in result["segments"]:
        seg.append(segment["start"])
        seg.append(segment["end"])

    S = seg[0] if len(seg) != 0 else None
    E = seg[-1] if len(seg) != 0 else None
    return S, E


if __name__ == "__main__":
    _main()
