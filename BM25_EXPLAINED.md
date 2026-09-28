# Explanation of `src/bm25.py`

`src/bm25.py` implements a **keyword search engine for source-code chunks**. It builds a searchable index, ranks chunks against a query, and saves or loads the index.

Here, a “document” means **one chunk**, not an entire file.

The overall flow is:

```text
Chunks → tokenize → build index → save
Query  → tokenize → score matching chunks → return best source locations
```

## 1. Imports and the token pattern

```python
import math
import pickle
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
```

Each import serves a specific purpose:

| Import | Purpose |
|---|---|
| `math` | Calculate logarithms and validate that parameters are finite. |
| `pickle` | Save and restore Python objects. |
| `re` | Extract words and split code identifiers. |
| `Path` | Handle the destination path and create directories. |
| `Counter` | Count token occurrences within a chunk. |
| `defaultdict` | Automatically create missing dictionary values. |
| `Any` | Type annotation for a dictionary containing different value types. |

```python
from .modele import Chunk, MinimalSource
from .stop_word import STOP_WORD
```

The leading `.` means these imports come from the same Python package.

- `Chunk` contains the text, file path, and start/end character offsets.
- `MinimalSource` contains only the file path and offsets.
- `STOP_WORD` is a set of common English words excluded from searching.

```python
PATTERN = r"\b\w+\b"
```

This regular expression extracts sequences of word characters:

- `r"..."` is a raw string, which preserves backslashes.
- `\b` marks a word boundary.
- `\w` matches word characters, including letters, digits, and underscores.
- `+` means one or more.

For example:

```text
"cache.get_value(x)" → ["cache", "get_value", "x"]
```

## 2. `__init__`: building the search engine

```python
def __init__(
    self, chunks: list[Chunk], k: float = 1.5, b: float = 0.5
) -> None:
```

This runs when you construct an instance:

```python
engine = Bm25(chunks)
```

`self` is that instance. The type annotations describe expected types; the explicit checks and model validation enforce the relevant constraints.

The parameters are:

| Parameter | Meaning |
|---|---|
| `chunks` | The source passages to search. |
| `k` | Controls how strongly repeated occurrences of a term affect its score. Stored as `k1`. |
| `b` | Controls how much chunk length affects scoring. |

The validation rejects:

- Booleans and values other than integers or floats.
- Infinity and `NaN`.
- `k <= 0`.
- `b` outside `[0, 1]`.

Booleans require an explicit check because Python treats `bool` as a subclass of `int`.

Then the constructor creates its state:

```python
self.chunks = chunks
self.k1 = k
self.b = b
self.N = len(self.chunks)
```

`N` is the total number of chunks.

```python
self.corpus = [
    self.tokenize(doc.text) + self.tokenize(doc.file_path)
    for doc in self.chunks
]
```

`corpus` is a list of token lists, one per chunk.

**Both the chunk text and its file path are searchable.** A query can therefore match a filename even if the term is absent from the chunk’s text. Path tokens contribute to term counts and document length just like text tokens.

```python
self.avgdl = (
    sum(len(chunk) for chunk in self.corpus) / self.N if self.N else 0
)
```

`avgdl` means **average document length**, measured in tokens after processing. The conditional avoids division by zero when there are no chunks.

```python
self.doc_len = {
    i: len(doc) for i, doc in enumerate(self.corpus)
}
```

This maps each chunk’s numerical ID to its token count:

```python
{0: 40, 1: 65, 2: 25}
```

`enumerate` supplies each list position alongside its value. Those positions become chunk IDs throughout the index.

Finally:

```python
self.inverted = self.build_inverted_index()
```

This builds the lookup structure used during search.

## 3. `tokenize`: turning text into searchable terms

The same method processes chunk text, paths, and queries so their tokens follow the same rules.

First:

```python
for token in re.findall(PATTERN, text):
    tokens.append(token.lower())
```

It extracts words and preserves a lowercase version of each complete token.

Next:

```python
expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
```

This inserts a space between a lowercase letter or digit and an uppercase letter:

```text
getValue → get Value
layer2Cache → layer2 Cache
```

Here, `\1` and `\2` refer to the two captured characters.

```python
parts = expanded.replace("_", " ").lower().split()
if len(parts) > 1:
    tokens.extend(parts)
```

This also splits underscores and adds the individual parts when there is more than one:

```text
getValue  → ["getvalue", "get", "value"]
get_value → ["get_value", "get", "value"]
```

Keeping both forms supports searches for complete identifiers and their components.

Finally, it removes stopwords:

```python
for token in tokens:
    if token in STOP_WORD:
        continue
    new.append(token)
```

For example:

```text
"The get_value cache"
    ↓
["get_value", "get", "value", "cache"]
```

Some relevant details:

- Repeated tokens remain repeated; scoring needs their counts.
- It does not reduce words to roots: `running` and `run` remain different.
- Acronym boundaries are not fully split: `HTTPServer` becomes `httpserver`, without separate `http` and `server` tokens.
- Token length includes identifier expansions, not just original words.

## 4. `build_inverted_index`: finding chunks containing a word

A normal document representation maps chunks to words. An **inverted index** maps words to chunks.

```python
invert = defaultdict(dict)
```

A missing word automatically gets an empty dictionary.

```python
for doc_id, doc in enumerate(self.corpus):
    frequencies = Counter(doc)
    for word, f in frequencies.items():
        invert[word][doc_id] = f
```

Suppose the processed corpus is:

```python
[
    ["cache", "cache", "memory"],
    ["cache", "disk"],
]
```

The resulting index is:

```python
{
    "cache":  {0: 2, 1: 1},
    "memory": {0: 1},
    "disk":   {1: 1},
}
```

For `"cache"`, this says:

- Chunk `0` contains it twice.
- Chunk `1` contains it once.

The dictionary for a term is called its **postings**. This lets search visit only chunks that contain a query token.

## 5. `idf`: giving rare terms more weight

```python
df = len(self.inverted.get(term, {}))
return math.log(1 + ((self.N - df + 0.5) / (df + 0.5)))
```

`df` is **document frequency**: the number of chunks containing the term, regardless of how many times it occurs in each.

The formula is:

```text
IDF(term) = ln(1 + (N - df + 0.5) / (df + 0.5))
```

Its effect:

- Terms appearing in many chunks receive less weight.
- Terms appearing in few chunks receive more weight.

The `0.5` values smooth the calculation, and the `1 +` keeps the result positive for valid index statistics.

For an absent term, `.get(term, {})` supplies an empty dictionary. Its IDF is calculated, but it contributes no score because no chunks contain it.

## 6. `_score_terms`: calculating relevance

The leading underscore signals an internal helper.

```python
scores = defaultdict(float)
```

Each unseen chunk starts with score `0.0`.

For every query token:

```python
inverse_document_frequency = self.idf(term)
postings = self.inverted.get(term, {})
```

Then, for each chunk containing that token:

```python
normalization = self.k1 * (
    1 - self.b + self.b * self.doc_len[chunk_id] / self.avgdl
)
```

This calculates the length adjustment.

- An average-length chunk has `doc_len / avgdl = 1`.
- A longer chunk gets a larger adjustment.
- A shorter chunk gets a smaller adjustment.
- With `b = 0`, length has no effect.
- With `b = 1`, the adjustment depends fully on relative length.

Next:

```python
term_score = (
    term_frequency * (self.k1 + 1)
    / (term_frequency + normalization)
) * inverse_document_frequency
```

Combined, the contribution of one query token is:

```text
                      frequency × (k1 + 1)
IDF × ----------------------------------------------------
      frequency + k1 × (1 - b + b × chunk_length / avgdl)
```

This balances three things:

1. **Rarity:** rare terms contribute more.
2. **Frequency:** repeated occurrences increase relevance, with diminishing gains.
3. **Length:** longer chunks need more occurrences to earn the same contribution.

For an average-length chunk with `k1 = 1.5`, the multiplier applied to IDF is:

| Term frequency | Multiplier |
|---:|---:|
| 1 | 1.00 |
| 2 | 1.43 |
| 5 | 1.92 |
| 10 | 2.17 |

Repeating a word ten times therefore does not produce ten times the score.

```python
scores[chunk_id] += term_score
```

Contributions accumulate across query tokens.

**Repeated query tokens also contribute repeatedly.** A query like `"cache cache"` doubles the `"cache"` contribution compared with `"cache"`.

The result looks like:

```python
{0: 2.8, 3: 1.2, 8: 4.1}
```

These are relevance scores, not probabilities.

For a normally built empty index, the inner loop never runs, so the division by `avgdl` is never reached.

## 7. `search`: returning the best source locations

```python
def search(self, query: str, top_k: int) -> list[MinimalSource]:
```

Example:

```python
results = engine.search("cache eviction", top_k=5)
```

`top_k` is the maximum number of results. It is separate from the constructor’s scoring parameter `k`.

The method rejects booleans, non-integers, and negative values. A zero limit immediately returns `[]`.

```python
scores = self._score_terms(self.tokenize(query))
```

This processes the query and calculates scores.

```python
ranked_ids = sorted(
    scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id)
)[:top_k]
```

Python sorts ascending, so the negative score puts higher scores first. The second tuple item breaks ties using the smaller chunk ID—original chunk order.

`[:top_k]` keeps at most the requested number.

For each selected chunk, the method constructs:

```python
MinimalSource(
    file_path=chunk.file_path,
    first_character_index=chunk.first_character_index,
    last_character_index=chunk.last_character_index,
)
```

The returned objects identify where to find the text:

```python
file_text[first_character_index:last_character_index]
```

The start is included; the end is excluded.

The method returns **locations only**, without text or numerical scores. Empty queries, stopword-only queries, and queries with no matching tokens return `[]`.

A chunk can match just one query term; matching every term is not required.

## 8. `index_data`: collecting state for saving

```python
@property
def index_data(self) -> dict[str, Any]:
```

`@property` lets you access this as:

```python
engine.index_data
```

rather than calling `engine.index_data()`.

It returns:

```python
{
    "chunks": self.chunks,
    "inverted": self.inverted,
    "doc_len": self.doc_len,
    "avgdl": self.avgdl,
    "k1": self.k1,
    "b": self.b,
    "N": self.N,
}
```

These values are enough to search without rebuilding the index.

`corpus` is omitted because scoring uses the postings and stored lengths directly. The original chunk text is still saved inside `chunks`.

## 9. `save_index`: writing the index to disk

```python
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
```

This creates missing parent directories. Existing directories are allowed.

```python
with open(target, "wb") as f:
    pickle.dump(self.index_data, f)
```

- `"wb"` means write binary; an existing file is overwritten.
- `pickle.dump` serializes the state.
- `with` closes the file automatically.

This writes directly to the destination, so an interrupted write can leave an incomplete index.

## 10. `load_index`: restoring the search engine

```python
@classmethod
def load_index(cls, path: str) -> "Bm25":
```

A class method receives the class as `cls`. You call it without creating an instance first:

```python
engine = Bm25.load_index("data/index.pkl")
```

It reads the saved state:

```python
with open(path, "rb") as f:
    info = pickle.load(f)
```

`"rb"` means read binary.

The exception handling provides clearer errors:

- Filesystem failures become `OSError` with the path.
- Several corrupt or incompatible pickle failures become `ValueError`, suggesting an index rebuild.
- `raise ... from error` preserves the original exception as the cause.

It then checks the basic structure:

```python
required = {"chunks", "N", "avgdl", "inverted", "doc_len", "k1", "b"}
if not isinstance(info, dict) or not required.issubset(info):
```

The loaded value must be a dictionary containing every required key. Additional keys are allowed.

```python
obj = cls.__new__(cls)
```

This creates an instance **without running `__init__`**, avoiding tokenization and index rebuilding.

The remaining assignments restore the saved attributes, and `return obj` returns the usable engine.

Two implementation details matter here:

- Loading checks required keys, but does not deeply validate their values or rerun constructor validation.
- A loaded instance has no `corpus` attribute. Searching and saving still work, but calling `build_inverted_index()` on it would fail because that method needs `corpus`.

As the method’s docstring states, pickle indexes must be trusted: unpickling can execute code.

## 11. How you use the complete class

```python
# Build from existing Chunk objects.
engine = Bm25(chunks, k=1.5, b=0.5)

# Persist the index.
engine.save_index("data/index.pkl")

# Restore it later.
loaded_engine = Bm25.load_index("data/index.pkl")

# Retrieve up to five source locations.
sources = loaded_engine.search("get_value cache", top_k=5)

for source in sources:
    print(
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )
```

This file handles token-based retrieval. It does not generate answers, use embeddings, expand synonyms, or require query words to appear next to one another.
