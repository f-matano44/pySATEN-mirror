from typing import Optional

import numpy as np
import numpy.typing as npt
import webrtcvad


class WebRTC_VAD:
    def __init__(self) -> None:
        self._vad = webrtcvad.Vad()

    def set_mode(self, mode: int) -> None:
        self._vad.set_mode(mode)

    @staticmethod
    def _frame_bytes(sig_pcm16: bytes, frame_samples: int) -> list[bytes]:
        step = frame_samples * 2
        n = len(sig_pcm16)
        return [sig_pcm16[i : i + step] for i in range(0, n - (n % step), step)]

    def detect_segments(
        self, x: npt.NDArray[np.floating], sr: int
    ) -> tuple[Optional[float], Optional[float]]:
        assert x.ndim == 1
        assert sr in (8000, 16000, 32000, 48000)

        frame_ms: int = 10
        frame_sec = frame_ms / 1000.0

        pcm = (x * 32767.0).astype(np.int16).tobytes()
        frame_samples = int(sr * frame_ms / 1000)
        frames = self._frame_bytes(pcm, frame_samples)

        segments: list[tuple[float, float]] = []
        in_speech = False
        start_f = 0

        for i, fb in enumerate(frames):
            sp = self._vad.is_speech(fb, sample_rate=sr)
            if sp and not in_speech:
                in_speech = True
                start_f = i
            elif not sp and in_speech:
                in_speech = False
                end_f = i - 1
                segments.append((start_f * frame_sec, (end_f + 1) * frame_sec))

        if in_speech:
            end_f = len(frames) - 1
            segments.append((start_f * frame_sec, (end_f + 1) * frame_sec))

        seg = []
        for segment in segments:
            # print(segment)
            seg.append(segment[0])
            seg.append(segment[1])

        S = seg[0] if len(seg) != 0 else None
        E = seg[-1] if len(seg) != 0 else None
        return S, E
