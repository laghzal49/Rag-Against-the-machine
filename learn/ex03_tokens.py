r"""Exercise 3: 3. Tokenize code identifiers — intermediate, 20 minutes

Read `Bm25.tokenize` in `src/retrieval/bm25.py` and `src/retrieval/stop_words.py`.
Implement `tokenize(text)` using the project's exact rules: extract `\b\w+\b`,
keep the lowercase original, split a lowercase letter or digit followed by
an uppercase letter, replace underscores with spaces, and append the split
parts only when there is more than one. Finally remove `STOP_WORD` entries.
Preserve order and duplicates. Example: `getUser user_id` becomes
`['getuser', 'get', 'user', 'user_id', 'user', 'id']`.
Explain why `HTTPServer` is not split by this particular rule.


Implement tokenize below, then run from the repository root:
    .venv/bin/python "him learn/ex03_tokens.py"

Tests intentionally fail until you implement the function.
"""

import math
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def tokenize(text):
    s = re.findall(r"\w+", text)
    token = ""
    for g in s:
        token += re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", g)
    token = token.lower().replace("_", " ")
    print(token.split())
    return token.split()


def span(start, end, path="a.txt"):
    return MinimalSource(
        file_path=path, first_character_index=start, last_character_index=end
    )


class TestExercise(unittest.TestCase):
    def test_03_tokens(self):
        self.assertEqual(
            tokenize("The getUser and user_id"),
            ["getuser", "get", "user", "user_id", "user", "id"],
        )
        self.assertEqual(
            tokenize("HTTPServer café café"), ["httpserver", "café", "café"]
        )
        self.assertEqual(tokenize("the AND !!!"), [])
        self.assertEqual(tokenize("v2Parser"), ["v2parser", "v2", "parser"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
