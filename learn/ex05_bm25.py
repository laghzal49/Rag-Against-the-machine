"""Exercise 5: 5. Calculate one BM25 contribution — intermediate, 25 minutes

Read `Bm25.idf` and `Bm25._score_terms`. Implement
`bm25_term_score(n, df, tf, dl, avgdl, k=1.5, b=0.5)`:

```text
idf = log(1 + (n - df + 0.5) / (df + 0.5))
score = idf * tf * (k + 1) / (tf + k * (1 - b + b * dl / avgdl))
```

Return zero immediately for `tf == 0`. Otherwise assume valid inputs,
including `avgdl > 0`. Use natural logarithm. Explain how document length
affects score when `b=0` and when `b>0`. Remember the real index counts tokens
from both chunk text and file path in document length.


Implement bm25_term_score below, then run from the repository root:
    .venv/bin/python "him learn/ex05_bm25.py"

Tests intentionally fail until you implement the function.
"""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def bm25_term_score(n, df, tf, dl, avgdl, k=1.5, b=0.5):
    raise NotImplementedError("Task 5")

def span(start, end, path="a.txt"):
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


class TestExercise(unittest.TestCase):
    def test_05_bm25(self):
        self.assertAlmostEqual(bm25_term_score(3, 1, 1, 4, 4), math.log(8 / 3))
        self.assertEqual(bm25_term_score(0, 0, 0, 0, 0), 0)
        short = bm25_term_score(10, 2, 1, 5, 10)
        long = bm25_term_score(10, 2, 1, 20, 10)
        self.assertGreater(short, long)
        self.assertAlmostEqual(bm25_term_score(10, 2, 1, 5, 10, b=0),
                               bm25_term_score(10, 2, 1, 20, 10, b=0))
        self.assertAlmostEqual(bm25_term_score(3, 1, 2, 4, 4, k=2, b=1),
                               1.5 * math.log(8 / 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
