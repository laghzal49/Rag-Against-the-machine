# Learn your RAG project

Each `ex*.py` file contains its own task explanation, starter function, and
independent test suite. Read the explanation at the top, implement the function,
and run that same file. You do not need to complete another exercise first.

| File | Task |
| --- | --- |
| `ex01_spans.py` | Extract valid source passages |
| `ex02_windows.py` | Split text into overlapping chunks |
| `ex03_tokens.py` | Tokenize code identifiers |
| `ex04_postings.py` | Build an inverted index |
| `ex05_bm25.py` | Calculate BM25 scores |
| `ex06_iou.py` | Measure source overlap |
| `ex07_recall.py` | Calculate retrieval recall |
| `ex08_pipeline.py` | Connect the real retrieval pipeline |

Run from the repository root, for example:

```sh
.venv/bin/python learn/ex01_spans.py
```

Change the filename to run another exercise. Tests deliberately fail with
`NotImplementedError` until you write the solution. They use the existing
project environment and do not download or run a language model.

`solutions.py` contains optional reference solutions. `QUIZ.md` and `ANSWERS.md`
provide a separate written quiz and answer key.
