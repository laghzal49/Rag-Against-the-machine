"""Regression checks for feature packages and source-context extraction."""

import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.chunking import chunk_python_file
from src.generation import Answer, ContextBuilder
from src.indexing import Indexer
from src.models import MinimalSource, RagDataset, UnansweredQuestion
from src.retrieval import Bm25, multiple_question


# TODO: Add mocked generation-budget tests and persistence tests for
# interrupted saves, malformed index state, and same-length source edits.
class PipelineTests(unittest.TestCase):
    def test_unicode_chunks_preserve_offsets(self) -> None:
        text = '# café\r\ndef hello():\r\n    return "世界"\r\n' * 5
        chunks = chunk_python_file(text, 'sample.py', max_chunk_size=40)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 40)
            self.assertEqual(
                chunk.text,
                text[chunk.first_character_index:chunk.last_character_index],
            )

    def test_round_trip_and_context(self) -> None:
        indexer = Indexer('demo/corpus')
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'index.pkl')
            indexer.save(path)
            engine = Bm25.load_index(path)
        question = UnansweredQuestion(question_id='q1', question='expired')
        results = multiple_question(RagDataset(rag_questions=[question]),
                                    3, engine)
        result = results.search_results[0]
        self.assertEqual(result.question_id, 'q1')
        self.assertTrue(result.retrieved_sources)
        builder = ContextBuilder()
        context = builder.context(question, 3, engine)
        for source in result.retrieved_sources:
            with open(source.file_path, encoding='utf-8', newline='') as f:
                passage = f.read()[source.first_character_index:
                                   source.last_character_index]
            self.assertIn(passage, context)
        answer = Answer.__new__(Answer)
        self.assertEqual(answer.generate(question, ''),
                         'Insufficient retrieved evidence.')

    def test_invalid_context(self) -> None:
        builder = ContextBuilder()
        for path, end in [('/outside.py', 1), ('demo/corpus/cache.py', 2000)]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                builder.context_from_sources([MinimalSource(
                    file_path=path, first_character_index=0,
                    last_character_index=end,
                )])

    def test_empty_index_is_controlled_error(self) -> None:
        with tempfile.NamedTemporaryFile() as file:
            with self.assertRaisesRegex(ValueError, 'rebuild'):
                Bm25.load_index(file.name)

    def test_legacy_pickle_model_path(self) -> None:
        chunk = Indexer('demo/corpus').chunks[0]
        legacy = pickle.dumps(chunk, protocol=0).replace(
            b'csrc.models.schemas\nChunk\n', b'csrc.modele\nChunk\n'
        )
        self.assertIn(b'csrc.modele\nChunk\n', legacy)
        self.assertEqual(pickle.loads(legacy), chunk)

    def test_cli_import_does_not_load_transformers(self) -> None:
        subprocess.run([
            sys.executable, '-c',
            'import src.cli, sys; assert "transformers" not in sys.modules',
        ], check=True)


if __name__ == '__main__':
    unittest.main()
