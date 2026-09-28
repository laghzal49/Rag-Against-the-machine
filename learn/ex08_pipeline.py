"""Exercise 8: 8. Connect the real pipeline — challenge, 30 minutes

Read `src/indexing/indexer.py`, `src/retrieval/search.py`, and `Bm25.search`.
Implement `retrieve_texts(documents, query, top_k)` where documents is a list
of `(file_path, text)` pairs. Assume each text is nonempty and at most 2,000
characters. Create one real `Chunk` per document, build a real `Bm25`, and pass it
to `one_question` with an `UnansweredQuestion` with ID `practice-001`.
Return the resulting `MinimalSearchResults`.
Reuse production code for this task. Preserve input order to break score ties.
The real search method must handle unknown queries, zero limits, and invalid
limits. Explain how this step differs from generating an answer with an LLM.


Implement retrieve_texts below, then run from the repository root:
    .venv/bin/python "him learn/ex08_pipeline.py"

Tests intentionally fail until you implement the function.
"""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MinimalSource


def retrieve_texts(documents, query, top_k):
    raise NotImplementedError("Task 8")

def span(start, end, path="a.txt"):
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


class TestExercise(unittest.TestCase):
    def test_08_pipeline(self):
        docs = [("a.txt", "database connection"), ("b.txt", "banana smoothie")]
        result = retrieve_texts(docs, "database", 1)
        self.assertEqual(result.question_id, "practice-001")
        self.assertEqual(result.question, "database")
        self.assertEqual(result.retrieved_sources, [span(0, 19)])
        self.assertEqual(retrieve_texts(docs, "unknownterm", 5).retrieved_sources, [])
        self.assertEqual(retrieve_texts(docs, "database", 0).retrieved_sources, [])
        self.assertEqual(retrieve_texts([], "database", 5).retrieved_sources, [])
        ties = retrieve_texts([("a.txt", "apple"), ("b.txt", "apple")], "apple", 9)
        self.assertEqual([s.file_path for s in ties.retrieved_sources], ["a.txt", "b.txt"])
        path_match = retrieve_texts([("database.py", "hello")], "database", 1)
        self.assertEqual(path_match.retrieved_sources[0].file_path, "database.py")
        for limit in [-1, True, 1.5]:
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    retrieve_texts(docs, "database", limit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
