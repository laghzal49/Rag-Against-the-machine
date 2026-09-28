"""Exercise 1: 1. Recover a source passage — beginner, 10 minutes

Read `MinimalSource` in `src/models/schemas.py` and
`Answer.context_from_sources` in `src/generation/answer.py`.
Implement `extract_span(text, start, end)`. Use character offsets with an
included start and excluded end. Require `0 <= start < end <= len(text)`
and a span of at most 2,000 characters; otherwise raise `ValueError`.
Inputs for start and end are integers. Example: `('café!', 3, 4)` gives `'é'`.
Explain why UTF-8 byte offsets cannot always be used to slice a Python string.


Implement extract_span below, then run from the repository root:
    .venv/bin/python "him learn/ex01_spans.py"

Tests intentionally fail until you implement the function.
"""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def extract_span(text, start, end):
    raise NotImplementedError("Task 1")

def span(start, end, path="a.txt"):
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


class TestExercise(unittest.TestCase):
    def test_01_spans(self):
        self.assertEqual(extract_span("café!", 3, 4), "é")
        self.assertEqual(extract_span("abc", 0, 3), "abc")
        self.assertEqual(len(extract_span("x" * 2000, 0, 2000)), 2000)
        for text, start, end in [("abc", -1, 2), ("abc", 1, 1),
                                 ("abc", 2, 1), ("abc", 0, 4),
                                 ("x" * 2001, 0, 2001)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    extract_span(text, start, end)


if __name__ == "__main__":
    unittest.main(verbosity=2)
