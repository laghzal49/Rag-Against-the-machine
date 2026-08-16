# Algorithms with worked examples

## BM25 arithmetic

Suppose the index contains two chunks, with lengths 2 and 8 after tokenization.
Their average length is 5. Both contain `cache`; chunk 0 contains it twice and
chunk 1 once. Use k1=1.5 and b=0.75.

```text
N = 2, df(cache) = 2
IDF = ln(1 + (2 - 2 + 0.5)/(2 + 0.5))
    = ln(1.2) ≈ 0.182322

Chunk 0:
normalization = 0.25 + 0.75*(2/5) = 0.55
TF factor = 2*2.5 / (2 + 1.5*0.55) = 5/2.825 ≈ 1.769912
score ≈ 0.322693

Chunk 1:
normalization = 0.25 + 0.75*(8/5) = 1.45
TF factor = 1*2.5 / (1 + 1.5*1.45) = 2.5/3.175 ≈ 0.787402
score ≈ 0.143560
```

The short chunk with two occurrences ranks first. For a multi-term query, add
each distinct query term's contribution. Repeating the query word does not
multiply its weight in this implementation. IDF uses `log1p` for stable arithmetic.

An inverted index for this example is `{'cache': [(0, 2), (1, 1)]}`. It stores
frequency, not a TF-IDF/BM25 score, because final normalization depends on corpus
statistics and chosen k1/b.

## Character spans

For `"AéB"`, the valid character positions are 0, 1 and 2. `text[1:2]` returns
`"é"`. Its UTF-8 encoding occupies two bytes, but the character span width is one.
The evaluator expects source locations, so the file text cannot be normalized
or reconstructed from tokens. Original raw bytes are decoded once as UTF-8.

Every indexed chunk obeys:

```text
0 <= start < end <= length(original)
end - start <= configured_limit <= 2000
chunk.text == original[start:end]
```

Whitespace-only windows are skipped. Long unbroken lines are hard-split when
no suitable newline exists. This is necessary to enforce the absolute limit.

## Recall@k and interval IoU

Consider a reference source `[100,200)` and a retrieved source `[150,250)` in
the same file. Intersection is 50; union is 150; IoU is 1/3. It counts as a match
because 1/3 exceeds 0.05. A source in another file counts as zero regardless of
identical text. A `[200,300)` span merely touches the reference and does not match.

For one question with M expected sources:

```text
question recall@k = number of expected sources matched by top-k / M
```

Then average these recalls across scored questions. This is macro averaging.
One retrieved span may overlap multiple expected spans; each expected span is
checked independently. Duplicate retrieved spans never give extra credit for
the same expected span. A missing question result contributes zero.

The subject's low IoU threshold makes exact reference boundary matching
unnecessary, but extremely tiny evidence can fail overlap relative to a large
reference. This is another reason to measure chunk size rather than assume
smaller is always better.

## Token budgeting

Characters are not model tokens. A 2000-character snippet can tokenize very
differently depending on language, punctuation and code identifiers. The model
budget must include the system prompt, question, source labels, chat delimiters,
and generated continuation.

For a configured 8192-token context and 384 new tokens:

```text
full encoded prompt length <= 8192 - 384 = 7808
```

The implementation tests the fully rendered chat prompt each time it considers
a passage. It does not estimate using a characters-per-token ratio. If a passage
does not fit, later shorter passages can still be tried. Source labels are
assigned only when included, so `[1]` always corresponds to output source 0.

## Cosine similarity and hybrid fusion

After normalization, two vectors u and v have unit Euclidean norm. Their dot
product equals cosine similarity. The optional encoder embeds chunk text plus
path/context metadata; the same encoder handles queries.

Lexical and semantic scores have different ranges. Reciprocal-rank fusion uses
position rather than raw score:

```text
RRF(chunk) = sum across result lists of 1 / (60 + one_based_rank)
```

A chunk second in both rankings gets `2/62 ≈ 0.032258`. A chunk first in only
one ranking gets `1/61 ≈ 0.016393`. Agreement can outweigh one excellent position.
This is a heuristic to evaluate, not a proof that hybrid recall will increase.

## Caching and invalidation

The query LRU key contains the original query, effective k and retrieval mode.
A hit is promoted to most recently used; the least recently used entry is evicted
after 256 entries. Cached values are chunk IDs. Callers receive copied chunks.

Incremental indexing uses file SHA-256 plus chunking/schema configuration. The
semantic metadata hashes the complete lexical index; changing lexical rows
requires rebuilding vectors. Running services use immutable index snapshots
until restarted. These are three different caches with different invalidation
rules; do not confuse them.
