"""Evaluate SATEN sensitivity to symmetric waveform margin extension.

Run from any working directory, for example::

    python tools/v2/eval_margin_sensitivity.py
"""

import argparse
import csv
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from pysaten.utility.WavLabHandler import WavLabHandler
from pysaten.v2 import vsed_debug_v2

SCRIPT_DIR: Final = Path(__file__).resolve().parent
DEFAULT_MARGINS: Final = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
UTTERANCES_PER_SPEAKER: Final = 324
SPEAKER_FILES: Final = (
    (
        "metan",
        "metan/ITA_recitation_normal_synchronized_wav/recitation",
        "metan/ITA_recitation_normal_label/recitation",
    ),
    (
        "sora",
        "sora/ITA_recitation_normal_synchronized_wav/recitation",
        "sora/ITA_recitation_normal_label/rct",
    ),
    (
        "usagi",
        "usagi/ITA_recitation_normal_synchronized_wav/normal_recitation_",
        "usagi/ITA_recitation_normal_label/rct",
    ),
)


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    wav_path: Path
    lab_path: Path
    saten_seed: int


@dataclass(frozen=True)
class Condition:
    noise_condition: str
    snr_db: int | None


@dataclass(frozen=True)
class UtteranceResult:
    utterance_id: str
    margin_sec: float
    noise_condition: str
    snr_db: int | None
    onset_error: float
    offset_error: float
    rejection_rate: float
    answer_onset: float
    answer_offset: float
    estimated_onset: float
    estimated_offset: float


CONDITIONS: Final = (
    Condition("clean", None),
    Condition("fan", 10),
    Condition("fan", 0),
)


def _median_absolute_error(
    answer: npt.NDArray[np.float64], estimated: npt.NDArray[np.float64]
) -> float:
    mask = ~np.isnan(estimated)
    masked_diff = answer[mask] - estimated[mask]
    if masked_diff.size == 0:
        raise ValueError("No recognized endpoints are available for MdAE")
    return float(np.median(np.abs(masked_diff)))


def _speech_period_rejection_rates(
    answer: npt.NDArray[np.float64], estimated: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    answer_start = answer[0::2]
    answer_end = answer[1::2]
    speech_durations = answer_end - answer_start
    if np.any(speech_durations <= 0):
        raise ValueError("Speech durations must be positive")

    estimated_start = estimated[0::2]
    estimated_end = estimated[1::2]
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
    return rejection_rates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=SCRIPT_DIR / "wav_and_lab"
    )
    parser.add_argument(
        "--fan-noise", type=Path, default=SCRIPT_DIR / "fan_noise.flac"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "margin_sensitivity_results"
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=list(DEFAULT_MARGINS),
        metavar="SECONDS",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        help="Limit the corpus for a smoke test; omit for the paper experiment.",
    )
    return parser.parse_args()


def _utterances(data_dir: Path, limit: int | None = None) -> list[Utterance]:
    utterances = [
        Utterance(
            utterance_id=f"{speaker}_{index:03}",
            wav_path=data_dir / f"{wav_prefix}{index:03}.wav",
            lab_path=data_dir / f"{lab_prefix}{index:03}.lab",
            saten_seed=index,
        )
        for index in range(1, UTTERANCES_PER_SPEAKER + 1)
        for speaker, wav_prefix, lab_prefix in SPEAKER_FILES
    ]
    if limit is not None:
        utterances = utterances[:limit]
    missing = [
        path
        for utterance in utterances
        for path in (utterance.wav_path, utterance.lab_path)
        if not path.is_file()
    ]
    if missing:
        examples = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} ITA Corpus input files are missing. First entries:\n{examples}"
        )
    return utterances


def _legacy_fan_seeds(snr_db: int, utterance_count: int) -> list[int]:
    """Reproduce the corresponding fan-noise seeds from the existing eval loop."""
    legacy_snr_order = ("inf", "20", "15", "10", "5", "0", "-5", "-inf")
    target = str(snr_db)
    block_index = legacy_snr_order.index(target)
    full_corpus_count = len(SPEAKER_FILES) * UTTERANCES_PER_SPEAKER
    rng = np.random.default_rng(0)
    seeds = rng.integers(
        0, 20260223, size=(block_index + 1) * full_corpus_count
    )
    start = block_index * full_corpus_count
    return [int(seed) for seed in seeds[start : start + utterance_count]]


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"ITA Corpus directory does not exist: {args.data_dir}")
    if not args.fan_noise.is_file():
        raise FileNotFoundError(f"Fan-noise file does not exist: {args.fan_noise}")
    if args.max_utterances is not None and args.max_utterances <= 0:
        raise ValueError("--max-utterances must be positive")
    if not args.margins or any(
        not np.isfinite(margin) or margin < 0 for margin in args.margins
    ):
        raise ValueError("All --margins values must be finite and non-negative")
    if len(set(args.margins)) != len(args.margins):
        raise ValueError("--margins must not contain duplicates")


def _margin_rows(
    utterances: Sequence[Utterance],
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for utterance in tqdm(utterances, desc="Original margin statistics"):
        handler = WavLabHandler(utterance.wav_path, utterance.lab_path)
        waveform, sample_rate = handler.get_signal()
        onset, offset = handler.get_answer()
        duration = len(waveform) / sample_rate
        leading_margin = onset
        trailing_margin = duration - offset
        if not np.isfinite([onset, offset, duration]).all():
            raise ValueError(f"Non-finite timing value: {utterance.utterance_id}")
        if onset < 0 or offset <= onset or trailing_margin < -1e-9:
            raise ValueError(
                f"Ground truth is outside the waveform: {utterance.utterance_id} "
                f"(duration={duration}, onset={onset}, offset={offset})"
            )
        rows.append(
            {
                "utterance_id": utterance.utterance_id,
                "leading_margin_sec": leading_margin,
                "trailing_margin_sec": max(0.0, trailing_margin),
                "duration_sec": duration,
            }
        )
    return rows


def _margin_summary_rows(
    rows: Sequence[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    summaries: list[dict[str, str | float]] = []
    for margin_type, key in (
        ("leading", "leading_margin_sec"),
        ("trailing", "trailing_margin_sec"),
    ):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summaries.append(
            {
                "margin_type": margin_type,
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "percentile_95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
        )
    return summaries


def _print_margin_statistics(rows: Sequence[dict[str, str | float]]) -> None:
    print("margin_type,min,median,mean,percentile_95,max")
    for row in rows:
        print(
            f"{row['margin_type']},{float(row['min']):.9f},"
            f"{float(row['median']):.9f},{float(row['mean']):.9f},"
            f"{float(row['percentile_95']):.9f},{float(row['max']):.9f}"
        )


def _evaluate(
    utterances: Sequence[Utterance],
    margins: Sequence[float],
    fan_noise_path: Path,
) -> list[UtteranceResult]:
    fan_seeds = {
        condition.snr_db: _legacy_fan_seeds(condition.snr_db, len(utterances))
        for condition in CONDITIONS
        if condition.snr_db is not None
    }
    results: list[UtteranceResult] = []
    total = len(utterances) * len(margins) * len(CONDITIONS)
    progress = tqdm(total=total, desc="SATEN margin evaluation")
    for utterance_index, utterance in enumerate(utterances):
        handler = WavLabHandler(
            utterance.wav_path, utterance.lab_path, fan_noise_path
        )
        original_waveform, sample_rate = handler.get_signal()
        original_onset, original_offset = handler.get_answer()
        for margin_sec in margins:
            expected_padding_samples = round(margin_sec * sample_rate)
            shifted_onset, shifted_offset = handler.get_answer(margin_sec)
            expected_shift = expected_padding_samples / sample_rate
            if not np.allclose(
                [shifted_onset, shifted_offset],
                [original_onset + expected_shift, original_offset + expected_shift],
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError(
                    f"Incorrect ground-truth shift: {utterance.utterance_id}, "
                    f"margin={margin_sec}"
                )

            for condition in CONDITIONS:
                if condition.snr_db is None:
                    waveform, returned_sample_rate = handler.get_signal(margin_sec)
                else:
                    waveform, returned_sample_rate = handler.get_fan_noise_signal(
                        condition.snr_db,
                        fan_seeds[condition.snr_db][utterance_index],
                        margin_sec,
                    )
                expected_length = len(original_waveform) + 2 * expected_padding_samples
                if len(waveform) != expected_length:
                    raise AssertionError(
                        f"Incorrect padded length: {utterance.utterance_id}, "
                        f"margin={margin_sec}, got={len(waveform)}, "
                        f"expected={expected_length}"
                    )
                if returned_sample_rate != sample_rate:
                    raise AssertionError("Sample rate changed during margin extension")
                estimated_onset, estimated_offset = vsed_debug_v2(
                    waveform, sample_rate, noise_seed=utterance.saten_seed
                ).get_result()
                endpoints = np.asarray(
                    [
                        shifted_onset,
                        shifted_offset,
                        estimated_onset,
                        estimated_offset,
                    ]
                )
                if not np.all(np.isfinite(endpoints)):
                    raise ValueError(
                        f"SATEN produced a non-finite endpoint: "
                        f"{utterance.utterance_id}, margin={margin_sec}, "
                        f"condition={condition}"
                    )
                answer = np.asarray(
                    [shifted_onset, shifted_offset], dtype=np.float64
                )
                estimated = np.asarray(
                    [estimated_onset, estimated_offset], dtype=np.float64
                )
                rejection_rate = float(
                    _speech_period_rejection_rates(answer, estimated)[0]
                )
                results.append(
                    UtteranceResult(
                        utterance_id=utterance.utterance_id,
                        margin_sec=margin_sec,
                        noise_condition=condition.noise_condition,
                        snr_db=condition.snr_db,
                        onset_error=estimated_onset - shifted_onset,
                        offset_error=estimated_offset - shifted_offset,
                        rejection_rate=rejection_rate,
                        answer_onset=shifted_onset,
                        answer_offset=shifted_offset,
                        estimated_onset=estimated_onset,
                        estimated_offset=estimated_offset,
                    )
                )
                progress.update()
    progress.close()
    return results


def _summary_rows(
    results: Sequence[UtteranceResult], margins: Sequence[float]
) -> Iterator[dict[str, str | int | float | None]]:
    for condition in CONDITIONS:
        for margin_sec in margins:
            selected = [
                result
                for result in results
                if result.margin_sec == margin_sec
                and result.noise_condition == condition.noise_condition
                and result.snr_db == condition.snr_db
            ]
            answers = np.asarray(
                [
                    endpoint
                    for row in selected
                    for endpoint in (row.answer_onset, row.answer_offset)
                ],
                dtype=np.float64,
            )
            estimates = np.asarray(
                [
                    endpoint
                    for row in selected
                    for endpoint in (row.estimated_onset, row.estimated_offset)
                ],
                dtype=np.float64,
            )
            mdae = _median_absolute_error(answers, estimates)
            yield {
                "margin_sec": margin_sec,
                "noise_condition": condition.noise_condition,
                "snr_db": condition.snr_db,
                "mdae": mdae,
                "rejection_rate": float(
                    np.mean([row.rejection_rate for row in selected])
                ),
            }


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _check_clean_baseline(
    results: Sequence[UtteranceResult], utterance_count: int
) -> None:
    baseline_path = SCRIPT_DIR / "results" / "fan_inf.csv"
    if utterance_count != len(SPEAKER_FILES) * UTTERANCES_PER_SPEAKER:
        print("Clean baseline check skipped for a limited corpus run.", file=sys.stderr)
        return
    if not baseline_path.is_file():
        print(f"WARNING: Clean baseline CSV not found: {baseline_path}", file=sys.stderr)
        return
    actual = [
        endpoint
        for row in results
        if row.margin_sec == 0.0 and row.noise_condition == "clean"
        for endpoint in (row.estimated_onset, row.estimated_offset)
    ]
    if not actual:
        print("WARNING: Clean margin=0 condition was not requested.", file=sys.stderr)
        return
    with baseline_path.open(encoding="utf-8", newline="") as file:
        expected = [float(row["pySATEN2"]) for row in csv.DictReader(file)]
    if len(actual) != len(expected):
        raise AssertionError(
            f"Clean baseline endpoint count differs: {len(actual)} != {len(expected)}"
        )
    difference = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
    if difference > 1e-12:
        raise AssertionError(
            f"Clean margin=0 does not match {baseline_path}; "
            f"maximum absolute difference={difference}"
        )
    print(
        f"Clean margin=0 baseline matched (max absolute difference={difference:.3g}).",
        file=sys.stderr,
    )


def _main() -> None:
    args = _parse_args()
    _validate_paths(args)
    utterances = _utterances(args.data_dir, args.max_utterances)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    margin_rows = _margin_rows(utterances)
    margin_summary_rows = _margin_summary_rows(margin_rows)
    _print_margin_statistics(margin_summary_rows)
    _write_csv(
        args.output_dir / "original_margin_by_utterance.csv",
        ("utterance_id", "leading_margin_sec", "trailing_margin_sec", "duration_sec"),
        margin_rows,
    )
    _write_csv(
        args.output_dir / "original_margin_summary.csv",
        ("margin_type", "min", "median", "mean", "percentile_95", "max"),
        margin_summary_rows,
    )

    results = _evaluate(utterances, args.margins, args.fan_noise)
    result_rows = [
        {
            "utterance_id": row.utterance_id,
            "margin_sec": row.margin_sec,
            "noise_condition": row.noise_condition,
            "snr_db": row.snr_db,
            "onset_error": row.onset_error,
            "offset_error": row.offset_error,
            "rejection_rate": row.rejection_rate,
        }
        for row in results
    ]
    _write_csv(
        args.output_dir / "margin_sensitivity_by_utterance.csv",
        (
            "utterance_id",
            "margin_sec",
            "noise_condition",
            "snr_db",
            "onset_error",
            "offset_error",
            "rejection_rate",
        ),
        result_rows,
    )
    _write_csv(
        args.output_dir / "margin_sensitivity_summary.csv",
        ("margin_sec", "noise_condition", "snr_db", "mdae", "rejection_rate"),
        _summary_rows(results, args.margins),
    )
    _check_clean_baseline(results, len(utterances))
    print(f"Results written to: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    _main()
