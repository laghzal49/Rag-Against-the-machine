"""Exercise 6: 6. Compare source spans — intermediate, 15 minutes

Read `calcultion` (the actual function spelling) in `src/evaluation/metrics.py`.
Implement `span_iou(first, second)` for two valid `MinimalSource` objects.
Different file paths give zero. Otherwise return intersection length divided
by union length. Touching spans have zero overlap. `[0,10)` and `[5,15)`
in the same file give `1/3`.


Implement span_iou below, then run from the repository root:
    .venv/bin/python "him learn/ex06_iou.py"

Tests intentionally fail until you implement the function.
"""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def span_iou(first, second):
    raise NotImplementedError("Task 6")

def span(start, end, path="a.txt"):
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


class TestExercise(unittest.TestCase):
    def test_06_iou(self):
        self.assertAlmostEqual(span_iou(span(0, 10), span(5, 15)), 1 / 3)
        self.assertEqual(span_iou(span(0, 10), span(0, 10)), 1)
        self.assertEqual(span_iou(span(0, 10), span(10, 20)), 0)
        self.assertEqual(span_iou(span(0, 10), span(0, 10, "b.txt")), 0)
        self.assertAlmostEqual(span_iou(span(0, 20), span(5, 10)), 0.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
