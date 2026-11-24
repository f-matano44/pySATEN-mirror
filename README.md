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

Output is saved as 48 kHz, 24-bit PCM.

```sh
pysaten_trim input.wav trimmed.wav
```

### Python

The Python API may change incompatibly between major versions.
Its design is inspired by `librosa.effects.trim`.

```python
import pysaten

y_trimmed, index = pysaten.trim(y, sr)
```

**Parameters:**

* `y` : `numpy.ndarray` - Mono audio signal.
* `sr` : `float` - Sampling rate in Hz.

**Returns:**

* `y_trimmed` : `numpy.ndarray` - Trimmed audio signal.
* `index` : `tuple[int, int]` - Start and end sample indices in `y`.

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

## Citation (version 2.0 or later)

> F. Matano, M. Morise, ``SATEN, A Speech Trimming Method for Speech Corpus Construction in Text-to-Speech," IEICE Transactions on Information and Systems, Advance online publication, Article 2026EDP7101, Nov. 2026.
> [【DOI】](https://doi.org/10.1587/transinf.2026EDP7101)
