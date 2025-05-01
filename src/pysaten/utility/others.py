import numpy as np


def slide_index(
    goto_min: bool,
    y: np.ndarray,
    start_idx: int,
    threshold: float,
    margin: int,
) -> int:

    stop_idx: int = -1 if goto_min else len(y)
    step: int = -1 if goto_min else 1

    for i in range(start_idx, stop_idx, step):
        if threshold <= y[i]:
            a_check_end = (
                max(0, i - margin) if goto_min else min(i + margin, len(y))
            )
            a_check = y[a_check_end:i] if goto_min else y[i:a_check_end]
            indices_below_threshold = [
                j for j, b in enumerate(a_check) if b < threshold
            ]
            if indices_below_threshold:  # is not empty
                i = (
                    min(indices_below_threshold)
                    if goto_min
                    else max(indices_below_threshold)
                )
            else:  # indices_below_threshold is empty -> finish!!!
                return i
    return 0 if goto_min else len(y)
