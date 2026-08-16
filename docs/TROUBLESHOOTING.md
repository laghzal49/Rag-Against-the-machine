# Troubleshooting by symptom

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `No module named src` | Wrong working directory | Run from the extracted project root |
| `uv` not found | Package manager not installed/in PATH | Follow uv's official installation instructions |
| Dependency download error | Network/certificate/proxy/cache problem | Check package access; reuse cached dependencies where available |
| Corpus directory missing | Official sources not copied | Place the provided corpus under `data/raw` |
| No indexable text | Empty directory, unsupported extensions, invalid UTF-8 | Inspect warnings and corpus contents; old index is preserved |
| Missing `index.json` | Search before indexing or wrong index directory | Run `index` with matching `--index_directory` |
| Corrupt index JSON | Interrupted external edit or incompatible file | Rebuild with `--incremental=False` |
| Incompatible index version | Tokenizer/schema changed | Rebuild; do not manually change version fields to bypass checks |
| Recall zero despite relevant-looking text | Corpus prefix differs or offsets differ | Compare paths literally and inspect exact original slices |
| Empty BM25 result | No shared indexed terms | Try an exact identifier, then investigate tokenizer/hybrid retrieval |
| Invalid source range | Damaged/imported search JSON | Regenerate search results from the matching index |
| Source changed | Corpus edited after indexing/search | Re-index, repeat search, then generate |
| Question exceeds token budget | Question/instructions leave no room | Shorten question or raise context within model capacity |
| No evidence fits | Output reservation/context too restrictive | Increase context or use smaller chunks; inspect source list |
| Model not available offline | Cache incomplete or model path wrong | Download once or point at a complete local checkpoint |
| CPU answers slow | Model load/inference cost or too many CPU threads | Reuse a batch/service; try bounded OMP threads and output length |
| Semantic extra missing | Only mandatory dependencies installed | `uv sync --extra semantic`; use `uv run --extra semantic ...` |
| Stale semantic index | Lexical corpus/chunks/config changed | Rebuild vectors with `semantic_index` |
| HTTP port occupied | Another local process uses it | Choose another `--port` |
| HTTP requests wait | Serial service or model loading | Wait for current request or use the CLI for a single job |
| Empty evaluation dataset error | No answered rows with sources | Supply the reference `AnsweredQuestions` dataset |
| Duplicate/unknown IDs | Mismatched datasets or edited JSON | Restore original IDs and pair matching dataset files |

## A structured debugging sequence

Check file ingestion first, then chunk spans, then retrieval ranking, then model
prompt content, then final wording. A generation prompt cannot recover a file
that was never indexed. A good BM25 score does not guarantee that the passage
answers the actual question. Print and read the evidence before adjusting code.

## Data and source integrity

Do not normalize line endings in the corpus after indexing. Do not rename the
corpus wrapper directory for convenience. Do not prepend metadata to source
text. Do not use byte positions as character offsets. These changes can leave
human-readable text intact while breaking the evaluator's location matching.

## Recovery

An indexing failure keeps a previous index if no usable chunks were produced.
For a corrupt previous index, explicitly rebuild with `--incremental=False`.
After successful rebuilding, regenerate search results and semantic vectors,
and restart a running HTTP service. Avoid simultaneous writers to the same index.
