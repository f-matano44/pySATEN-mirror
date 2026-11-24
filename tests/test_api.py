import unittest
from unittest.mock import patch

import numpy as np

import pysaten


class TrimTest(unittest.TestCase):
    def test_trim_returns_signal_and_sample_indices(self) -> None:
        y = np.arange(100, dtype=np.float64)
        with patch("pysaten.pysaten._vsed", return_value=(0.2, 0.7)):
            y_trimmed, index = pysaten.trim(y, 100)

        self.assertEqual(index, (20, 70))
        np.testing.assert_array_equal(y_trimmed, y[20:70])

    def test_rejects_multichannel_input(self) -> None:
        y = np.zeros((2, 100), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "monaural"):
            pysaten.trim(y, 100)


if __name__ == "__main__":
    unittest.main()
