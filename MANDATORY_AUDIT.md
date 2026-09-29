# Mandatory requirements audit (version 2.0)

This audit covers Chapters V, VI, VII, VIII, and X of the supplied subject.
The foreword and learning advice require personal review rather than a code
change. Chapter IX (bonus) is excluded. Findings refer to the working tree on
2026-09-29; existing uncommitted changes were left untouched.

## Changes needed

The first two rows are confirmed behavior/documentation defects. The recall row
is a required measurement that cannot be completed with the datasets currently
present. The remaining rows are quality risks or small cleanups; they should not
be presented as automatic mandatory failures.

| Priority | Requirement | Evidence | Change to make |
| --- | --- | --- | --- |
| High | Single-query commands must return structured JSON following the supplied models (VI.4-VI.6). | `search` returns a Pydantic object from `src/cli.py:22-49`; Fire prints `search_results=[MinimalSearchResults(...)] k=3`, which is not JSON. `answer` already prints JSON at line 115. | Print `output.model_dump_json(indent=2)` inside `search` and return `None`, as `answer` does. If single-query file output is expected by the examiner, add an optional output path to both commands; the stated minimum CLI does not define one. |
| Medium | README examples and references must actually work (VIII). | `README.md:36-56` uses `demo/corpus/cache.py`, but `demo/corpus/` is deleted in the current working tree. `README.md:151,165,194` points to deleted `SMOKE_TEST.md`. | Restore those files or replace the demo with commands against the available `data/raw/` corpus. Remove dead checklist references and verify each example from a clean checkout. |
| Required verification | Performance analysis must discuss measured recall@k and effects of chunk size (VI.1, VII, VIII). | `README.md:155-165` cites earlier approximate recall figures without the public datasets or a reproducible score report. No dataset JSON is present under `data/datasets/` in this workspace. | Once public answered/unanswered datasets are available, run `search_dataset` and the official evaluator for docs and code at k=5. Record scores, command, corpus, chunk size, and elapsed time in README. Compare at least one smaller chunk size so the required recall effect is reported. Confirm docs >=80% and code >=50%; tune retrieval if either misses. |
| Quality risk | Answer generation should select evidence within the model's token budget (VI.3). | `src/generation/context.py:42-57` concatenates every retrieved passage without a token budget. `src/generation/answer.py:75-80` then truncates the whole prompt from the right. This can silently remove lower-ranked evidence. A real short Qwen answer worked; the long-context case remains untested. | Budget passages before building the final prompt, reserve tokens for the question, instructions, and answer, and verify a long-context case. Preserve the highest-ranked evidence and state when context was omitted. |
| Boundary check | Keep the local recall calculation aligned with the stated IoU bar (VII.1.1). | `src/evaluation/metrics.py:53` uses `> 0.05`; an exact-IoU-0.05 fixture returns zero recall. The subject describes the bar as an IoU of `0.05`, but does not spell out inclusive versus exclusive comparison. | Check the official boundary, then use `>= 0.05` if inclusive. This only affects the local `evaluate` command; official scoring uses the evaluator. |
| Small cleanup | README technical descriptions should match the code (VIII). | `README.md:141` calls the Python chunker an “AST” chunker, while `src/chunking/splitters.py:3-14` uses Tree-sitter syntax nodes. | Replace “Python AST” with “Tree-sitter syntax” in the project-structure table. |
| Optional hygiene | The direct dependency list should include imported libraries. | `src/models/schemas.py:5` imports `typing_extensions`, but `pyproject.toml:6-14` relies on another package to install it transitively. | Add `typing-extensions` as a direct dependency and refresh `uv.lock`, or use `typing.Self` only after raising the Python minimum to 3.11. Keeping Python 3.10 means the direct dependency is the safer choice. |

## Requirement-by-requirement status

| Subject area | Status | Evidence / remaining check |
| --- | --- | --- |
| Python >=3.10; `uv`, root `pyproject.toml` and `uv.lock` (V.1, V.4, VI.7.1, X) | Pass | `pyproject.toml` requires >=3.10; `uv sync --locked --offline` succeeds. |
| `flake8`, mypy, hints, docstrings, resource handling (V.1) | Pass for tested cases | `flake8 .` passes; required mypy flags report no issues in 22 source files. File I/O uses context managers. Forced offline model-loading failure returned a concise error without a traceback. |
| Makefile `install`, `run`, `debug`, `clean`, `lint` (V.2) | Pass for tested rules | `Makefile:3-14` defines all rules. `make lint` passes with the specified commands/flags; `uv sync --locked --offline` verifies installation. `debug` was inspected but not entered interactively; `clean` was not run because it deletes files. |
| Test programs and `.gitignore` (V.3) | Partial | `.gitignore` excludes Python artifacts and data. The tracked `tests/test_pipeline.py` is deleted in this working tree, so no test program is currently present. Restore useful edge-case tests for chunk spans, dataset JSON, and CLI paths. These are recommended, not graded. |
| Pydantic models (V.4, VI.4, VI.5) | Pass | All required models exist in `src/models/schemas.py`; batch output validates and preserves question IDs. |
| Fire CLI and tqdm (V.4, VI.6) | Pass with issue above | All six commands appear in `src/__main__.py`; indexing and dataset loops show progress. Single search is not JSON. |
| Index Python and Markdown/text separately; lexical search; size <=2000 (VI.1) | Pass for tested inputs | Tree-sitter Python and overlapping text windows are distinct; BM25 is implemented. Full corpus run: 1,969 files, 13,717 chunks, maximum span 2,000 characters, 9.30 seconds here. A Python sample with leading and trailing comments was included in a chunk. |
| Exact source paths and ranges; single and batch top-k (VI.2) | Pass for tested examples | All 13,717 tested chunk paths begin `data/raw/`; every chunk equals the source file's exact character slice. Gaps between Python chunks contain only whitespace. Sample batch JSON has exact paths, ranges, k, and preserved IDs. Required recall thresholds remain unverified. |
| Local Qwen/Qwen3-0.6B answer and JSON (VI.3, VI.5) | Pass for short tested inputs | Default model is correct. A real single answer completed, and a two-question `answer_dataset` output validated as `StudentSearchResultsAndAnswer`. Grounding quality across the public dataset remains unverified. |
| Degenerate inputs (VI.6) | Pass for tested cases | Empty or nonsensical queries and k=0 return empty sources or an abstaining answer; negative k, chunk sizes 0 and 2001, missing dataset, malformed JSON, an unknown CLI command, and forced offline model-loading failure produce concise errors without tracebacks. Arbitrary inference failures were not induced. |
| Expected layout, scoped dataset outputs (VI.7) | Pass for code/layout; data unavailable | Required root files and `data/` directories exist. Synthetic batch saved a valid JSON file to the requested directory. Public question datasets are absent. |
| Index <=5 minutes; 200 questions <=90 seconds; recall@5 (VII) | Time passes locally; recall unverified | 9.30-second local full index; 0.94 seconds for a synthetic 200-question batch using five repeated query types. A real 200-question dataset and official recall report are still needed. |
| README required sections, examples, AI use (VIII) | Mostly present; fixes above | First line, Description, Instructions, Resources, architecture, chunking, retrieval, design, challenges, example usage, and AI use are present. Demo/checklist references are stale; recall evidence is not reproducible here. |
| Submission excludes large data, weights, outputs (X) | Pass for tracked files | `.gitignore` excludes `data/`, model artifacts are not tracked, and the large corpus zip is ignored. A stray untracked root file named `]` exists; inspect and remove it if it is accidental before submission. |

## Checks run for this audit

- `uv sync --locked --offline`: passed.
- `flake8 .`: passed; required mypy invocation: passed, 22 source files.
- `make lint`: passed with the Makefile's exact flake8 and mypy commands.
- `uv run --locked python -m src index --max_chunk_size 2000 --index_path /tmp/rag-v2-audit-index.pkl`: passed in 9.30 seconds.
- Inspected the generated index: 13,717 chunks, maximum span 2,000, all paths prefixed `data/raw/`.
- Checked every indexed chunk against `source_text[start:end]`: zero mismatches and zero nonwhitespace gaps in the covered files.
- Single `search` and synthetic two-question `search_dataset`: retrieval worked; batch JSON validated by inspection. The single command printed a Python representation.
- Synthetic 200-question `search_dataset`: 0.94 seconds including CLI startup and index load; five question types were repeated, so this is only a local throughput check.
- Empty and nonsensical queries, k=0, negative k, invalid chunk sizes, missing dataset path, malformed JSON, and an unknown command: handled without tracebacks.
- Real `answer` with Qwen/Qwen3-0.6B and k=1: completed with valid answer JSON. Two-question `answer_dataset`: completed with valid JSON and an abstention for the empty question. An empty answer dataset also completed.
- Forced model-loading failure using an empty offline Hugging Face cache: one concise error, exit code 1, no traceback.
- Local `evaluate` on a synthetic exact-source fixture: 100% at k=1 and 0% at k=0. Exact IoU 0.05 returns 0 under the current strict comparison.
- Did not run the official evaluator because public question datasets are absent from this workspace. Private-set recall and long-context answer quality remain unknown.

The subject's AI/peer-review instructions also require the author to understand the
code and seek peer review; those actions cannot be established by repository inspection.
