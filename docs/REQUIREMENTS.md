# Implementation and remaining validation

This table describes the current code. It does not certify full compliance with
an assignment subject or hidden evaluation datasets.

| Area | Implementation | Remaining validation |
| --- | --- | --- |
| CLI | Six Fire commands in `src/cli.py` | Target-environment checks |
| Style and types | `make lint` runs flake8 and mypy | Re-run after edits |
| Error handling | CLI boundary, input validation, file context managers | Environment-specific failures |
| Data models | Pydantic models in `src/modele.py` | Hidden dataset compatibility |
| Chunking | Tree-sitter Python nodes; overlapping text windows | Python chunks can omit text between nodes |
| Chunk sizes | Integer limit from 1 to 2,000 characters | Corpus-specific retrieval quality |
| Retrieval | BM25 with path and identifier tokenization | Hidden dataset quality |
| Persistence | Trusted pickle at `data/processed/index.pkl` | Direct writes are not atomic |
| Single and batch search | `search` and `search_dataset` | Target runtime and throughput |
| Answers | Lazy Qwen/Qwen3-0.6B, citations, prompt token budget | Real model generation remains unverified |
| Evaluation | Exact paths, character IoU > 0.05, macro recall | Supplied grader remains authoritative |
| Public recall@5 | Docs 85%; code 82.8% on saved results | See the validation record for scope |
| Optional bonuses | Not implemented in the current source tree | No bonus completion claimed |

See [the validation record](../VALIDATION.md) for recorded results and known
limitations, and [testing](TESTING.md) for commands to repeat checks.
