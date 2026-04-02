import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCORE_PATTERN = re.compile(r"^(None|-?\d+(?:\.\d+)?) \((\d+(?:\.\d+)?)\)$")
HEADER_LABELS = {
    "SNR": "SNR~[dB]",
    "pySATEN2": "SATEN",
    "Silero_vad": "Silero VAD",
}


@dataclass(frozen=True)
class ScoreCell:
    value_text: str | None
    value: float | None
    coverage_text: str


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Markdown の表を LaTeX の tabular 環境へ変換します。"
    )
    parser.add_argument("path", help="入力ファイルへの相対パス")
    args = parser.parse_args()

    input_path = Path(args.path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError as error:
        print(error, file=sys.stderr)
        return 1

    print(_convert_markdown_tables(text), end="")
    return 0


def _convert_markdown_tables(text: str) -> str:
    lines = text.splitlines()
    converted_lines: list[str] = []
    index = 0

    while index < len(lines):
        table_lines = _read_table(lines, index)
        if table_lines is None:
            converted_lines.append(lines[index])
            index += 1
            continue

        converted_lines.extend(_convert_table(table_lines))
        index += len(table_lines)

    output = "\n".join(converted_lines)
    if text.endswith("\n"):
        output += "\n"
    return output


def _read_table(lines: list[str], start: int) -> list[str] | None:
    if start + 1 >= len(lines):
        return None

    header_cells = _split_markdown_row(lines[start])
    separator_cells = _split_markdown_row(lines[start + 1])
    if header_cells is None or separator_cells is None:
        return None
    if len(header_cells) == 0 or len(header_cells) != len(separator_cells):
        return None
    if not all(_is_alignment_cell(cell) for cell in separator_cells):
        return None

    table_lines = [lines[start], lines[start + 1]]
    index = start + 2
    while index < len(lines):
        cells = _split_markdown_row(lines[index])
        if cells is None or len(cells) != len(header_cells):
            break
        table_lines.append(lines[index])
        index += 1

    return table_lines


def _split_markdown_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped or "|" not in stripped:
        return None

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    starts_with_pipe = stripped.startswith("|")
    ends_with_pipe = stripped.endswith("|")

    for char in stripped:
        if escaped:
            if char == "|":
                current.append("|")
            else:
                current.append("\\")
                current.append(char)
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())

    if starts_with_pipe and cells and cells[0] == "":
        cells = cells[1:]
    if ends_with_pipe and cells and cells[-1] == "":
        cells = cells[:-1]

    return cells


def _is_alignment_cell(cell: str) -> bool:
    stripped = cell.strip()
    if not stripped:
        return False

    if stripped.startswith(":"):
        stripped = stripped[1:]
    if stripped.endswith(":"):
        stripped = stripped[:-1]
    return stripped != "" and all(char == "-" for char in stripped)


def _convert_table(table_lines: list[str]) -> list[str]:
    header_cells = _split_markdown_row(table_lines[0])
    alignment_cells = _split_markdown_row(table_lines[1])
    if header_cells is None or alignment_cells is None:
        raise ValueError("table_lines is not a markdown table")

    row_cells = [_split_markdown_row(line) for line in table_lines[2:]]
    if any(cells is None for cells in row_cells):
        raise ValueError("table row is invalid")

    rows = [cells for cells in row_cells if cells is not None]
    if _is_pulse_table(rows):
        return _convert_pulse_table(header_cells, rows)
    if _is_evaluation_table(header_cells, rows):
        return _convert_evaluation_table(header_cells, rows)

    return _convert_generic_table(header_cells, alignment_cells, rows)


def _is_pulse_table(rows: list[list[str]]) -> bool:
    labels = [row[0] for row in rows]
    return labels == ["without_pulse", "with_pulse"]


def _is_evaluation_table(header_cells: list[str], rows: list[list[str]]) -> bool:
    if not rows or header_cells[0] != "SNR":
        return False
    return all(_parse_score_cell(cell) is not None for row in rows for cell in row[1:])


def _convert_evaluation_table(header_cells: list[str], rows: list[list[str]]) -> list[str]:
    column_spec = f"r|{'r' * (len(header_cells) - 1)}"
    tabular_lines = [f"\\begin{{tabular}}{{{column_spec}}}\\hline", ""]
    tabular_lines.extend(_format_evaluation_header(header_cells))
    tabular_lines.append("")

    for row in rows:
        tabular_lines.extend(_format_evaluation_row(row))
        tabular_lines.append("")

    tabular_lines[-1] = r"\hline"
    tabular_lines.append(r"\end{tabular}%")
    return tabular_lines


def _convert_pulse_table(header_cells: list[str], rows: list[list[str]]) -> list[str]:
    column_spec = f"c|{'r' * (len(header_cells) - 1)}"
    tabular_lines = [f"\\begin{{tabular}}{{{column_spec}}}\\hline", ""]
    tabular_lines.extend(_format_pulse_header(header_cells))
    tabular_lines.append("")

    for row in rows:
        tabular_lines.extend(_format_pulse_row(row))
        tabular_lines.append("")

    tabular_lines[-1] = r"\hline"
    tabular_lines.append(r"\end{tabular}%")
    return tabular_lines


def _format_pulse_header(header_cells: list[str]) -> list[str]:
    formatted_lines = [r"\multicolumn{1}{c|}{}"]
    for cell in header_cells[1:]:
        label = HEADER_LABELS.get(cell, cell.replace("_", " "))
        formatted_lines.append(rf" & \multicolumn{{1}}{{c}}{{{_escape_latex(label)}}}")

    formatted_lines[-1] += r" \\ \hline"
    return formatted_lines


def _format_evaluation_header(header_cells: list[str]) -> list[str]:
    formatted_lines: list[str] = []
    for index, cell in enumerate(header_cells):
        label = HEADER_LABELS.get(cell, cell.replace("_", " "))
        prefix = "" if index == 0 else " & "
        if index == 0:
            formatted = rf"\multicolumn{{1}}{{c|}}{{{label}}}"
        else:
            formatted = rf"\multicolumn{{1}}{{c}}{{{_escape_latex(label)}}}"
        formatted_lines.append(prefix + formatted)

    formatted_lines[-1] += r" \\ \hline"
    return formatted_lines


def _format_pulse_row(row: list[str]) -> list[str]:
    label = _format_pulse_label(row[0])
    scores = [_parse_score_cell(cell) for cell in row[1:]]
    if any(score is None for score in scores):
        raise ValueError("pulse row contains invalid score")

    parsed_scores = [score for score in scores if score is not None]
    best_value = _find_best_value(parsed_scores)

    formatted_lines = [label]
    for score in parsed_scores:
        formatted_lines.append(f" & {_format_score_cell(score, best_value, False)}")

    formatted_lines[-1] += r" \\"
    return formatted_lines


def _format_evaluation_row(row: list[str]) -> list[str]:
    label = row[0]
    scores = [_parse_score_cell(cell) for cell in row[1:]]
    if any(score is None for score in scores):
        raise ValueError("evaluation row contains invalid score")

    parsed_scores = [score for score in scores if score is not None]
    highlight_missing = label == "-inf"
    best_value = _find_best_value(parsed_scores) if not highlight_missing else None

    formatted_lines = [_format_snr_label(label)]
    for score in parsed_scores:
        prefix = " & "
        formatted_lines.append(
            prefix + _format_score_cell(score, best_value, highlight_missing)
        )

    formatted_lines[-1] += r" \\"
    return formatted_lines


def _find_best_value(scores: list[ScoreCell]) -> float | None:
    values = [score.value for score in scores if score.value is not None]
    if not values:
        return None
    return min(values)


def _format_pulse_label(label: str) -> str:
    if label == "without_pulse":
        return "Clean"
    if label == "with_pulse":
        return "w/ Pulse"
    return _escape_latex(label.replace("_", " "))


def _format_snr_label(label: str) -> str:
    if label == "inf":
        return r"$\infty$ (Clean)"
    if label == "-inf":
        return r"$-\infty$ (Noise only)"
    if _is_negative_number(label):
        return f"${label}$"
    return _escape_latex(label.replace("_", " "))


def _is_negative_number(text: str) -> bool:
    if not text.startswith("-"):
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _format_score_cell(
    score: ScoreCell, best_value: float | None, highlight_missing: bool
) -> str:
    coverage = score.coverage_text
    if score.value is None:
        if highlight_missing:
            coverage = rf"\textbf{{{coverage}}}"
        return f"--- ({coverage})"

    value_text = score.value_text
    if value_text is None:
        raise ValueError("score.value_text must not be None")
    if best_value is not None and math.isclose(score.value, best_value):
        value_text = rf"\textbf{{{value_text}}}"
    return f"{value_text} ({coverage})"


def _parse_score_cell(cell: str) -> ScoreCell | None:
    matched = SCORE_PATTERN.fullmatch(cell)
    if matched is None:
        return None

    value_text = matched.group(1)
    if value_text == "None":
        return ScoreCell(value_text=None, value=None, coverage_text=matched.group(2))

    return ScoreCell(
        value_text=value_text,
        value=float(value_text),
        coverage_text=matched.group(2),
    )


def _convert_generic_table(
    header_cells: list[str], alignment_cells: list[str], rows: list[list[str]]
) -> list[str]:
    tabular_lines = [f"\\begin{{tabular}}{{{_build_column_spec(alignment_cells)}}}", "\\hline"]
    tabular_lines.append(_format_generic_row(header_cells))
    tabular_lines.append("\\hline")

    for row_cells in rows:
        tabular_lines.append(_format_generic_row(row_cells))

    tabular_lines.append("\\hline")
    tabular_lines.append("\\end{tabular}")
    return tabular_lines


def _build_column_spec(alignment_cells: list[str]) -> str:
    column_spec: list[str] = []
    for cell in alignment_cells:
        stripped = cell.strip()
        if stripped.startswith(":") and stripped.endswith(":"):
            column_spec.append("c")
        elif stripped.endswith(":"):
            column_spec.append("r")
        else:
            column_spec.append("l")
    return "".join(column_spec)


def _format_generic_row(cells: list[str]) -> str:
    escaped_cells = [_escape_latex(cell) for cell in cells]
    return " & ".join(escaped_cells) + r" \\"


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


if __name__ == "__main__":
    raise SystemExit(_main())
