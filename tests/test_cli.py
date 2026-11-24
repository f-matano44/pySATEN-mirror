import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


class CommandLineTest(unittest.TestCase):
    def test_output_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.wav"
            output_path = Path(directory) / "output.wav"
            sample_rate = 16000
            time = np.arange(sample_rate * 2) / sample_rate
            signal = 0.2 * np.sin(2 * np.pi * 440 * time)
            sf.write(input_path, signal, sample_rate)

            command = shutil.which("pysaten_trim")
            self.assertIsNotNone(command)
            assert command is not None
            subprocess.run(
                [
                    command,
                    str(input_path),
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            info = sf.info(output_path)
            self.assertEqual(info.samplerate, 48000)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.format, "WAV")
            self.assertEqual(info.subtype, "PCM_24")
            self.assertGreater(info.frames, 0)


if __name__ == "__main__":
    unittest.main()
