"""Exercise 4: 4. Build postings — intermediate, 15 minutes

Read `Bm25.build_inverted_index`. Implement `build_postings(documents)` where
documents is a list of token lists and the result maps term to `{doc_id: count}`.
Document IDs start at zero. For `[['cat','cat'],['dog','cat']]`, cat has postings
`{0: 2, 1: 1}`. Explain the difference between term frequency and document frequency.


Implement build_postings below, then run from the repository root:
    .venv/bin/python "him learn/ex04_postings.py"

Tests intentionally fail until you implement the function.
"""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def build_postings(documents):
    raise NotImplementedError("Task 4")

def span(start, end, path="a.txt"):
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


class TestExercise(unittest.TestCase):
    def test_04_postings(self):
        self.assertEqual(build_postings([["cat", "cat"], [], ["dog", "cat"]]),
                         {"cat": {0: 2, 2: 1}, "dog": {2: 1}})
        self.assertEqual(build_postings([]), {})
        self.assertEqual(build_postings([[]]), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
