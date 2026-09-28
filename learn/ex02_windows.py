"""Exercise 2: 2. Split overlapping windows — beginner, 20 minutes

Read `chunk_text` in `src/chunking/splitters.py`.
Implement `split_windows(text, size)` returning `(start, end, text)` tuples.
Match the project's overlap: `min(200, size // 5)`. Stop as soon as a window
reaches the end. Empty text returns `[]`. Validate size even for empty text:
reject booleans, non-integers, and values outside 1–2,000 with `ValueError`.
Example: `'abcdefghijk'`, size 5 gives `(0,5,'abcde')`, `(4,9,'efghi')`,
and `(8,11,'ijk')`. Explain why size 1 still makes progress.


Implement split_windows below, then run from the repository root:
    .venv/bin/python "him learn/ex02_windows.py"

Tests intentionally fail until you implement the function.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def split_windows(text, size):
    lis = []
    last = 0
    i = 0
    while i <= len(text):
        lis.append((last, i + size, text[last : i + size]))
        i += size
        last = i - 1
    return lis


def span(start, end, path="a.txt"):
    return MinimalSource(
        file_path=path, first_character_index=start, last_character_index=end
    )


class TestExercise(unittest.TestCase):
    def test_02_windows(self):
        self.assertEqual(
            split_windows("abcdefghijk", 5),
            [(0, 5, "abcde"), (4, 9, "efghi"), (8, 11, "ijk")],
        )
        self.assertEqual(split_windows("", 5), [])
        self.assertEqual(split_windows("éa", 1), [(0, 1, "é"), (1, 2, "a")])
        self.assertEqual(split_windows("abcde", 5), [(0, 5, "abcde")])
        windows = split_windows("x" * 2100, 1500)
        self.assertEqual([(s, e) for s, e, _ in windows], [(0, 1500), (1300, 2100)])
        for size in [True, False, 0, -1, 2001, 2.5, "5"]:
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    split_windows("", size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
