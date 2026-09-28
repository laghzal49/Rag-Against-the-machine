# Architecture and implementation map

## Packages

| Path | Responsibility | Main interfaces |
| --- | --- | --- |
| `src/models/` | Data contracts and span validation | `Chunk`, `MinimalSource`, `RagDataset` |
| `src/io/` | Input/output: exact source reads and dataset JSON | `read_source_file`, `read_json_file`, `save_json` |
| `src/chunking/` | Python syntax and overlapping text windows | `chunk_python_file`, `chunk_text` |
| `src/retrieval/` | Tokenization, BM25 ranking, index loading and search | `Bm25`, `one_question`, `multiple_question` |
| `src/indexing/` | Corpus discovery and index construction | `Indexer` |
| `src/generation/` | Source context and local model generation | `ContextBuilder`, `Answer` |
| `src/evaluation/` | Character overlap and macro recall | `evaluate_dataset` |
| `src/cli.py` | Six public commands | `Cli` |
| `src/__main__.py` | Fire routing and error boundary | `main` |

## Data flow

Source files become exact text, bounded chunks, tokens, then a BM25 index.
Questions become ranked source spans, optionally followed by generated answers.
Dataset search and answer commands save JSON with the original question IDs.
Evaluation compares retrieved spans against reference sources and prints recall.

Python chunking converts Tree-sitter byte offsets to Unicode character offsets.
Text reads retain line endings so stored spans remain exact slices.

## Persistence

`Indexer.save` writes a trusted local pickle containing chunks, postings,
document lengths and BM25 parameters. Rebuild after corpus or tokenizer changes.
The index is written directly; saving is not atomic. Unreadable source files are
skipped with a warning. Only load pickle indexes from trusted sources.

`src/modele.py` retains model imports for older pickles referencing that module.
The canonical data contracts now live in `src/models/schemas.py`.

## Retrieval and generation

BM25 scores matching postings and sorts scored chunks by descending score.
Tokenization retains full identifiers and also splits snake_case and CamelCase.
There is no vector index or search cache.

Search does not import Transformers. Constructing `Answer` loads the tokenizer
and model. `ContextBuilder` can validate and format passages independently.
It checks project containment and span bounds, but does not detect edits that
leave the file long enough to satisfy the stored bounds.

Generation reserves output tokens and left-truncates the prompt to fit the
model context window. This can remove evidence or instructions; complete-passage
budgeting remains a possible improvement. Returned sources preserve the original
retrieval results. Empty context or questions produce an abstention message.

## Error boundaries and checks

The CLI catches common input, file and runtime errors and returns a nonzero exit
status. Keyboard interruption returns 130. Library functions raise exceptions.

`tests/test_pipeline.py` covers Unicode offsets, index round trips, source
validation, empty indexes, legacy model pickle paths and lightweight CLI imports.
Actual model output and official dataset recall require separate evaluation.
