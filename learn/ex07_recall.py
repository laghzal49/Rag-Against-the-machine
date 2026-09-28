"""Exercise 7: 7. Measure source recall — intermediate, 15 minutes

Read `evaluation` in `src/evaluation/metrics.py`. Implement
`source_recall(references, predictions)`. A reference is matched if any
prediction has IoU **strictly greater than 0.05**. Count each reference once,
even if multiple predictions match. Divide by the number of references;
return zero if either list is empty. Predictions need not be uniquely assigned.
Explain why retrieving the same matching span twice does not improve recall.

You may import calcultion from src.eval to calculate IoU, or implement
a local helper. This exercise does not depend on completing exercise 6.


Implement source_recall below, then run from the repository root:
    .venv/bin/python "him learn/ex07_recall.py"

Tests intentionally fail until you implement the function.
"""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def source_recall(references, predictions):
    raise NotImplementedError("Task 7")

def span(start, end, path="a.txt"):
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


class TestExercise(unittest.TestCase):
    def test_07_recall(self):
        refs = [span(0, 10), span(20, 30)]
        self.assertEqual(source_recall(refs, [span(0, 10)] * 2), 0.5)
        self.assertEqual(source_recall(refs, refs), 1)
        self.assertEqual(source_recall([], refs), 0)
        self.assertEqual(source_recall(refs, []), 0)
        self.assertEqual(source_recall([span(0, 20)], [span(19, 20)]), 0)
        self.assertEqual(source_recall([span(0, 20)], [span(18, 20)]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
