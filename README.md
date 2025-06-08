# pySATEN
[![PyPI - Version](https://img.shields.io/pypi/v/pysaten)](https://pypi.org/project/pysaten/)
[![Downloads](https://static.pepy.tech/badge/pysaten)](https://pepy.tech/project/pysaten)

* Main: https://gitlab.com/f-matano44/pysaten
* Mirror: https://github.com/f-matano44/pySATEN-mirror

## About
This library detects silence segment from speech signal.

![\(alt: Image of voice segment detection\)](https://gitlab.com/f-matano44/pysaten/-/raw/main/image/signal_graph.svg)


## Installation
```
$ pip install pysaten
```


## Usage
### Command line
The audio file is exported as 24-bit PCM at 48 kHz, mono.
```
$ pysaten_trim input.wav trimmed.wav
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

### For development (Linux only)
```
$ git clone https://gitlab.com/f-matano44/pysaten.git
$ poetry install
```


## License
Copyright 2024 Fumiyoshi MATANO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.



## Acknowledgements
The following programs were used to [\[evaluate the performance of pysaten\]](tools/v1/).
We would like to take this opportunity to express our gratitude.

* test/marblenet: Apache License Version 2.0
    * https://github.com/NVIDIA/NeMo
* test/rvad: MIT License
    * https://github.com/zhenghuatan/rVAD
* inaSpeechSegmenter
    * https://github.com/ina-foss/inaSpeechSegmenter


## Cite this
### Library version 1.X (Non-peer-reviewed)
#### Japanese
俣野 文義，小口 純矢，森勢 将雅，``音声コーパス構築のための仮定を追加した発話区間検出法の提案と基礎評価,'' 日本音響学会第 152 回 (2024 年秋季) 研究発表会, pp.1161--1162 (2024.09).

#### English
F. Matano, J. Koguchi, M. Morise, ``Proposal and basic evaluation of a voice activity detection with additional assumptions for speech corpus construction,'' Proceedings of the 2024 Autumn meeting of the Acoustical Society of Japan, pp.1161--1162 (2024.09) (in Japanese).
