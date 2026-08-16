# Code walkthrough

Read this alongside the actual files. The explanation follows function names
rather than brittle line numbers, so small formatting changes do not invalidate it.

## `models.py`: enforce contracts before doing work

`MinimalSource` contains the exact path and two nonnegative offsets.
`check_span()` runs after parsing and rejects a reversed range. It intentionally
does not impose the retrieved-source 2000-character limit: reference sources can
be wider. `Chunk.check_text()` imposes that limit on indexed evidence and checks
that the text length equals the span width.

`UnansweredQuestion` generates a UUID only if the input has no question ID.
`AnsweredQuestion` adds reference sources and an answer. `RagDataset` accepts
either form. The answered type appears first in the union; a full answered row
retains its extra fields through Pydantic validation.

`MinimalSearchResults` and `MinimalAnswer` describe a single query result.
Their two `Student...` envelopes add the requested k around a list of results.
`Chunk.source()` explicitly constructs only source fields, keeping internal
text/context metadata out of the evaluator output.

`IndexedFile` stores one hash and its chunks. `IndexData` stores the version,
chunking configuration, ordered file mapping, lengths and postings.
`QuestionRecall` makes failures inspectable; `EvaluationReport` aggregates them.

## `io_utils.py`: small resource-management helpers

`read_text()` uses a context manager and `newline=""`. Python normally translates
CRLF in text reads; disabling translation preserves the exact original offsets.

`read_model()` passes the JSON string to the requested Pydantic model. It is
generic: a caller specifying `RagDataset` gets a statically typed `RagDataset`.

`write_json()` accepts either a Pydantic model or plain JSON-compatible data.
It creates the destination parent, writes a temporary sibling, closes the file,
and replaces the destination. A `finally` block removes a leftover temporary
file if an error prevented replacement. This protects against publishing a
half-written output, though it is not a multi-file database transaction.

`within_project()` resolves a source path and ensures it remains under the
project root after following any symlinks. `check_k()` rejects negative k and
booleans; k=0 is intentionally valid.

## `chunking.py`: preserve content while choosing boundaries

`_windows()` receives an original text plus a `[start,end)` region. It proposes
a stop at the character limit. For a long region it prefers a newline in the
second half of that window. It records non-whitespace spans, then starts the
next window before the previous end to create overlap. `max(start+1,...)`
guarantees progress, even with a tiny configured limit.

`_python_sections()` builds an array of cumulative line lengths. It parses the
AST and walks declarations. For a decorated declaration it chooses the earliest
decorator line. It adds starts and ends to a set, then sorts the boundaries.
Nested methods are included. Invalid syntax falls back to the full-file region.
The labels provide nearby declaration names for ranking only.

`_text_sections()` walks lines while tracking backtick/tilde fences. A heading
outside a fence starts a new section and gives it a label. A `#` inside a fenced
Python block is not a Markdown heading. Plain text without headings becomes one
region, later split into windows.

`chunk_file()` validates settings, selects a strategy based on `.py`, and runs
`_windows()` for each adjacent boundary pair. Binary search finds the preceding
label. It constructs each `Chunk` from the exact original slice; Pydantic checks
its width immediately. Metadata labels are a ranking heuristic, not an AST scope
resolver: after nested declarations they may refer to the most recent symbol.

## `lexical.py`: tokenizer, inverted index, BM25

`tokenize()` extracts ASCII alphanumeric/underscore identifiers. It lowercases
each full identifier, then creates extra terms for camelCase, acronym boundaries,
and snake_case. English stopwords are omitted. Non-ASCII source text is still
preserved in chunks, although this lexical tokenizer primarily targets English
code/documentation identifiers.

`build_postings()` enumerates chunks in stable order. It tokenizes original
text, file path and context label, counts frequencies with `Counter`, and appends
`(chunk_id, frequency)` to each term's list. A separate array stores each chunk's
resulting token length, including its ranking metadata.

`BM25.__init__()` validates k1 and b and computes average length. Empty/zero
length corpora avoid a zero denominator. `search()` iterates distinct query
terms, skips unknown terms, computes IDF, and adds each posting's saturated,
length-normalized score. `nlargest()` selects top candidates; a negative chunk
ID in the tie key makes lower IDs win equal scores.

## `indexing.py`: incremental ingestion

`list_files()` uses `os.walk`, sorts directory/file names, prunes environment
and cache directories, and avoids symlinks. Only the configured text extensions
are accepted. Generated index output is not recursively ingested.

`all_chunks()` flattens the ordered file mapping in the same order used when
building postings; row IDs therefore retain their meaning after JSON loading.

`create_index()`:

1. Validates character limit/overlap and locates the dedicated corpus.
2. Loads the old index if incremental mode is enabled.
3. Invalidates reuse when corpus prefix, schema/tokenizer version or chunk
   configuration differs.
4. Reads each file as bytes and hashes it.
5. Reuses a hash-identical file's chunks, otherwise decodes/chunks the file.
6. Warns and counts unreadable/nontext files; omits deleted files naturally.
7. Rejects an empty result before replacing a working index.
8. Rebuilds postings and saves the index plus a duration/reuse report.

Hashes cost a file read but detect edits even if timestamps or sizes are unchanged.
The chunk cache is incremental; the postings rebuild is deliberately global.

## `retrieval.py`: reusable search service

`Retriever.__init__()` validates the retrieval mode, loads the Pydantic index,
checks schema/tokenizer versions, verifies posting IDs and initializes BM25.
Semantic loading is optional. A 256-entry `OrderedDict` implements LRU caching.

`_load_semantic()` checks that the saved vector metadata matches the lexical
index hash. It loads non-pickled vectors, checks shape/finiteness, and loads the
CPU encoder once.

`search_chunks()` rejects invalid k, returns early for empty input, caps results
to corpus size, and checks the query cache. On a miss it requests a candidate
pool from BM25 and optionally semantic search. Hybrid mode fuses the two lists.
It caches final IDs, evicts the oldest entry if necessary, and returns deep
copies so callers cannot modify the stored evidence. `search()` converts these
chunks to minimal source locations.

## `generation.py`: evidence-grounded local generation

`Generator.__init__()` validates context/output budgets and device, but keeps
model/tokenizer empty. `_load()` imports the heavyweight stack only when needed,
loads Qwen in CPU float32 by default, enters evaluation mode, and checks the
model's maximum positional capacity.

`_evidence()` deduplicates file reads within one question, verifies each path is
indexed, compares its current SHA-256 with the index, checks the requested range,
and slices the decoded original file. A changed file causes failure before a
model call, preserving the meaning of source locations.

`_prompt()` builds system/user messages and applies the Qwen chat template with
thinking disabled. `_fits()` tokenizes that complete string without adding extra
special tokens and reserves the configured number of output tokens.

`answer()` abstains immediately if query/evidence is empty. Otherwise it verifies
evidence, loads the model, checks the question alone fits, and greedily includes
whole passages that fit. Each included passage gets its own numbered label.
Inference uses `torch.inference_mode()` and deterministic greedy decoding. The
input token prefix is removed before decoding; only generated answer text is
returned. Empty model output is reported explicitly. The final source list
contains only the evidence passed to the model, in citation-label order.

## `evaluation.py`: inspectable local scoring

`span_iou()` immediately returns zero for different file paths. For the same file
it computes intersection and union of half-open character intervals. Touching
intervals with no shared characters have zero overlap.

`evaluate_results()` checks IDs, rejects invalid retrieved spans and unknown
prediction IDs, then processes every reference question. Unanswered/no-source
rows are counted as exclusions. Missing predictions have an empty retrieved
list. For each expected source, `any()` determines whether one top-k result
passes the IoU threshold. The function averages question recalls, giving every
evaluated question equal weight. An entirely unscorable dataset is an error.

## `cli.py` and `__main__.py`: expose the pipeline

Each public `CLI` method corresponds to one required or bonus command. Single
commands create a question ID, save a validated envelope and print it. Batch
commands preserve input IDs/filenames and accept caller-selected directories.
`answer_dataset()` consumes the supplied retrieval results without reranking.
`evaluate()` uses the local metric only; no school executable is called.

`main()` passes a `CLI()` instance to Python Fire. Fire turns typed method
parameters into command options. The outer exception boundary prints a concise
error and exits nonzero. Catching broadly here is intentional; lower-level
functions still raise errors so tests can inspect them.

## `semantic.py`: optional similarity search

`fingerprint()` hashes the index incrementally in blocks. `embedding_model()`
loads Sentence Transformers only when available; otherwise it tells the user
which extra to install. `build_semantic()` encodes ordered chunks on CPU,
normalizes vectors, verifies the index did not change during encoding, and saves
vectors/metadata. Semantic and lexical index creation must not run concurrently.

`SemanticIndex.search()` embeds the query, computes a dot product against all
normalized vectors, and sorts scores. `reciprocal_rank_fusion()` adds a small
rank-based contribution per list; this avoids treating incomparable raw scores
as the same units.

## `http_api.py`: serial local requests

`QueryRequest` validates strict query/k types and bounds. `make_handler()`
returns a handler class closed over the persistent retriever/generator.
`_send()` writes UTF-8 JSON with an explicit byte length and handles disconnected
clients. GET exposes health; POST handles search and answer only. Bad request
bodies become JSON errors. A 64 KiB body limit and read timeout bound parsing.

`serve()` restricts binding to loopback, validates the port, and runs `HTTPServer`
in a context manager. It handles one request at a time, which avoids model/cache
thread races. Restart the service after changing its index snapshot.

## Root configuration

`pyproject.toml` declares Python compatibility, dependencies, optional semantic
packages, the development group and checker settings. `uv.lock` freezes the
resolved graph. `.flake8` excludes generated/corpus/environment content.
`.gitignore` keeps large data, caches and outputs out of Git. The Makefile wraps
installation, execution, debugging, cleanup, and the exact mandatory
lint commands, plus stricter checking.
