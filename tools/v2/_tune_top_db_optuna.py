from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import optuna

from pysaten.utility.WavLabHandler import WavLabHandler


def collect_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for speaker in ("tohoku_itako", "zundamon"):
        wav_dir = Path("wav_and_lab") / speaker / "ITA_emotion_normal_synchronized_wav"
        lab_dir = Path("wav_and_lab") / speaker / "ITA_emotion_normal_label"
        for i in range(1, 101):
            wav_file = wav_dir / f"emoNormal{i:03d}.wav"
            lab_file = lab_dir / f"emoNormal{i:03d}.lab"
            if wav_file.is_file() and lab_file.is_file():
                pairs.append((wav_file, lab_file))
    if len(pairs) == 0:
        raise FileNotFoundError()
    return pairs


def detect_boundary_sec(
    x: npt.NDArray[np.floating], sr: int, top_db: float
) -> tuple[float, float]:
    _, index = librosa.effects.trim(x, top_db=top_db)
    return index[0] / sr, index[1] / sr


def evaluate_top_db(top_db: float, pairs: list[tuple[Path, Path]]) -> float:
    squared_errors: list[float] = []
    for wav_path, lab_path in pairs:
        handler = WavLabHandler(wav_path=wav_path, lab_path=lab_path)
        x, sr = handler.get_signal()
        ans_s, ans_e = handler.get_answer()
        pred_s, pred_e = detect_boundary_sec(x.astype(float), int(sr), top_db)
        squared_errors.append((pred_s - ans_s) ** 2)
        squared_errors.append((pred_e - ans_e) ** 2)
    return float(np.sqrt(np.mean(squared_errors)))


def save_optimization_plots(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_trials: list[optuna.trial.FrozenTrial] = [
        trial for trial in study.trials if trial.value is not None
    ]
    if len(valid_trials) == 0:
        return

    trial_numbers = np.asarray([trial.number for trial in valid_trials], dtype=float)
    values = np.asarray([float(trial.value) for trial in valid_trials], dtype=float)
    top_dbs = np.asarray(
        [float(trial.params["top_db"]) for trial in valid_trials], dtype=float
    )
    best_so_far = np.minimum.accumulate(values)

    plt.figure(figsize=(9, 4))
    plt.plot(trial_numbers, values, label="RMSE")
    plt.plot(trial_numbers, best_so_far, label="Best so far")
    plt.xlabel("Trial")
    plt.ylabel("RMSE [s]")
    plt.title("Optimization history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "optuna_history.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(top_dbs, values, s=22, alpha=0.8)
    plt.xlabel("top_db")
    plt.ylabel("RMSE [s]")
    plt.title("top_db vs RMSE")
    plt.tight_layout()
    plt.savefig(output_dir / "optuna_top_db_vs_rmse.png", dpi=150)
    plt.close()


def main() -> None:
    pairs = collect_pairs()
    sampler = optuna.samplers.TPESampler(seed=20260302)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        top_db = trial.suggest_float("top_db", 0, 100)
        return evaluate_top_db(top_db=top_db, pairs=pairs)

    study.optimize(objective, n_trials=100)
    save_optimization_plots(study, Path("results"))

    print(f"files={len(pairs)}")
    print(f"best_top_db={study.best_params['top_db']:.6f}")
    print(f"best_rmse_sec={study.best_value:.6f}")
    print("plots=results/optuna_history.png, results/optuna_top_db_vs_rmse.png")


if __name__ == "__main__":
    main()
