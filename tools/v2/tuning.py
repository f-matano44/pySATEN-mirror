from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import optuna
from tqdm import tqdm

from pysaten.utility.WavLabHandler import WavLabHandler
from pysaten.v2 import vsed_debug_v2

SEED: Final[int] = 20250923


@lru_cache(maxsize=None)
def _load_item(i: int, character: str):
    wav = f"wav_and_lab/{character}/ITA_emotion_normal_synchronized_wav"
    wav_path = Path(f"{wav}/emoNormal{i:03}.wav")
    lab = f"wav_and_lab/{character}/ITA_emotion_normal_label"
    lab_path = Path(f"{lab}/emoNormal{i:03}.lab")

    handler = WavLabHandler(wav_path, lab_path)
    x, fs = handler.get_noise_signal2(20, "pink", SEED)
    ans_s, ans_e = handler.get_answer()
    return x, fs, ans_s, ans_e


def function(rms_thres: float, zcr_thres: float) -> float:
    error: list[float] = []

    for i in tqdm(range(1, 101)):
        for character in ["zundamon", "tohoku_itako"]:
            x, fs, ans_s, ans_e = _load_item(i, character)

            S, E = vsed_debug_v2(
                x,
                fs,
                rms_threshold=rms_thres,
                zcr_threshold=zcr_thres,
                offset_s=0,
                noise_seed=SEED,
            ).get_result()

            error.append(S - ans_s)
            error.append(E - ans_e)
    err = np.asarray(error, dtype=np.float64)
    return np.sqrt(np.mean(err**2))


def objective(trial: optuna.Trial) -> float:
    rms_thres = trial.suggest_float("rms_thres", 0.0, 0.5, step=0.001)
    zcr_thres = trial.suggest_float("zcr_thres", 0.0, 1.0, step=0.01)
    value = function(rms_thres, zcr_thres)
    trial.report(value, step=0)
    return value


def _main():
    study = optuna.create_study(
        study_name="Tuning for SATEN2",
        direction="minimize",
        storage="sqlite:///study.db",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    print(study.best_value, study.best_params)


if __name__ == "__main__":
    _main()
