# Architecture and implementation map

## Read the project in this order

| File | Responsibility | Key functions/classes |
| --- | --- | --- |
| `src/models.py` | Stage contracts and validation | `Chunk`, all required models |
| `src/io_utils.py` | Exact reads, atomic JSON writes, source confinement | `read_text`, `write_json`, `within_project` |
| `src/chunking.py` | Python AST/text segmentation | `chunk_file`, `_windows` |
| `src/lexical.py` | Tokenization, postings, BM25 arithmetic | `tokenize`, `build_postings`, `BM25` |
| `src/indexing.py` | Scan, fingerprint, reuse, persist | `create_index` |
| `src/retrieval.py` | Load, search, cache | `Retriever` |
| `src/generation.py` | Verify sources, budget context, call Qwen | `Generator` |
| `src/evaluation.py` | Interval matching and macro recall | `span_iou`, `evaluate_results` |
| `src/cli.py` | Public command orchestration | `CLI` |
| `src/__main__.py` | Fire routing and final error boundary | `main` |
| `src/semantic.py` | Optional vectors and rank fusion | `build_semantic`, `SemanticIndex` |
| `src/http_api.py` | Reusable local service | `serve`, `make_handler` |

## Data flow

`data/raw` -> exact text -> `Chunk` -> term postings -> `IndexData` JSON.

`RagDataset` -> BM25 -> `MinimalSearchResults` -> `StudentSearchResults` JSON.

`MinimalSearchResults` -> verified evidence -> Qwen -> `MinimalAnswer` ->
`StudentSearchResultsAndAnswer` JSON.

`StudentSearchResults` + answered `RagDataset` -> `EvaluationReport` JSON.

The default lexical path does not import the language model at retrieval time.
The generation service can consume source spans from a previous search run;
it checks file membership, hash, and bounds before using them.

## Persistence

`index.json` stores the schema/tokenizer versions, corpus prefix, chunk settings,
file hashes, exact chunks, token lengths, and postings. Flattening the ordered
file mapping reconstructs chunk IDs. BM25 statistics derive from this index.
Use a fresh index when changing tokenizer behavior, and bump its version.

A temporary sibling file is fully written before `os.replace`. If indexing finds
no usable chunks it fails before replacing an existing index. Unreadable or
non-UTF-8 source files are skipped with a visible warning and a report count.
The index contains source text, so it should stay out of Git just like the corpus.

Semantic vectors are a NumPy file loaded with `allow_pickle=False`. Metadata
contains the exact index hash and encoder name. Lexical/vector state mismatches
are rejected. Indexing is a single-writer operation; do not run two indexing jobs
against the same output directory simultaneously. Restart long-running services
after rebuilding their index: they deliberately keep an in-memory snapshot.

## Complexity and tradeoffs

For B input characters and T token occurrences, scanning/chunking/tokenization
are approximately O(B+T). AST parsing adds per-Python-file work. Fingerprinting
still reads unchanged files, even in incremental mode. Postings are rebuilt.
JSON repeats source text and has object overhead; readability is preferred over
a binary database here. Measure memory and startup on the official corpus.

BM25 visits the postings for each distinct query term. Selecting the best k
uses a heap, rather than sorting every scored document. Very common terms can
still touch many chunks. Cache hits bypass scoring. Semantic exact search uses
O(Nd) per query for N vectors of dimension d; it is not an approximate-nearest-
neighbor system.

## Generation policy

The question and system instructions must fit before evidence is considered.
Each candidate passage is added only if the whole chat-templated prompt plus
reserved output tokens fits. Oversized candidates are skipped, allowing later,
shorter evidence to fit. Generation returns only included locations; the original
search JSON is the authoritative retrieval result for recall measurement.

The prompt asks for citations and abstention. It reduces hallucination risk but
does not mechanically prove entailment or validate every generated citation.
Greedy decoding favors repeatability; the Qwen card also provides sampled
settings, which can be explored separately. Thinking is disabled to spend the
small model's output budget on its answer.

## Error boundaries

Pydantic errors, bad inputs, missing files, and model failures propagate to one
CLI boundary, which prints an error and exits nonzero without a traceback.
Unexpected exceptions are not silently turned into success. The HTTP boundary
returns JSON errors. Library functions remain directly testable and raise
exceptions normally. `KeyboardInterrupt` exits the CLI with status 130.
