# pySATEN

[![PyPI - Version](https://img.shields.io/pypi/v/pysaten)](https://pypi.org/project/pysaten/)
[![Downloads](https://static.pepy.tech/badge/pysaten)](https://pepy.tech/project/pysaten)

* Main: <https://gitlab.com/f-matano44/pysaten>
* Mirror: <https://github.com/f-matano44/pySATEN-mirror>

## About

This library detects silence segment from speech signal.

![\(alt: Image of Voice Start and End Detection\)](https://gitlab.com/f-matano44/pysaten/-/raw/main/image/signal_graph.svg)

## Installation

```sh
pip install pysaten
```

## Usage

### Command line

```sh
pysaten_trim input.wav trimmed.wav
```

### Python

```python
import pysaten

# y: Target signal(numpy.ndarray), obtained using libraries such as librosa or soundfile.
# sr: Sampling rate.

# Get trimmed signal for the speech segment only.
y_trimmed: numpy.ndarray = pysaten.trim(y, sr)

# If you trim manually or want to get start/end time...
start_sec, end_sec = pysaten.vsed(y, sr)
y_trimmed = y[int(start_sec * sr) : int(end_sec * sr)]
```

### For development

```sh
git clone https://gitlab.com/f-matano44/pysaten.git
uv sync
```

## License

Copyright 2024 Fumiyoshi MATANO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Acknowledgements 🙏

* tools/v1/marblenet: Apache License Version 2.0
  * <https://github.com/NVIDIA/NeMo>
* tools/v1/rvad: MIT License
  * <https://github.com/zhenghuatan/rVAD>
* inaSpeechSegmenter
  * <https://github.com/ina-foss/inaSpeechSegmenter>
* Silero VAD
  * <https://github.com/snakers4/silero-vad>
* SpeechBrain VAD
  * <https://huggingface.co/speechbrain/vad-crdnn-libriparty>
* WebRTC VAD
  * <https://github.com/wiseman/py-webrtcvad>
* WhisperX
  * <https://github.com/m-bain/whisperX>

## Citation

> F. Matano, M. Morise, ``SATEN, A Speech Trimming Method for Speech Corpus Construction in Text-to-Speech," IEICE Transactions on Information and Systems, Advance Publication, 2026.
> [【DOI】](https://doi.org/10.1587/transinf.2026EDP7101)
