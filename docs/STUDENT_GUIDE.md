# Student guide and defense preparation

This guide helps you understand the implementation. Do not memorize answers
without running the corresponding functions and changing the code yourself.

## 1. Explain RAG in one minute

A language model does not inherently know a private or newly changed repository.
We first split the repository into searchable evidence. At question time we
retrieve likely evidence and put it in the model's prompt. The model generates
an answer from that supplied context. This is retrieval plus generation, not
fine-tuning: the model weights do not change.

The retriever and generator fail differently. A correct answer is unlikely if
the right evidence was never retrieved. Even good evidence can be misread by a
small model. Inspect search results before changing the prompt.

## 2. Follow a source through the pipeline

Take this string:

```python
text = 'hello\ncafé\n'
assert text[6:10] == 'café'
```

The source is described by a file path and offsets 6, 10. Those are Python string
positions, not UTF-8 byte positions and not line numbers. `é` is one character
but more than one UTF-8 byte. Modifying newlines, stripping text, or prepending
a function name breaks the source contract.

Read `Chunk.check_text`, `read_text`, and `_python_sections`. Notice that AST
line numbers give safe line boundaries. AST column offsets would need byte-to-
character conversion, so this implementation does not use them.

## 3. Understand chunking

Why not index a whole file? It may exceed 2000 characters and contain unrelated
concepts. Why not split every 100 characters? You lose context and spend the
five-result budget on fragments. Chunk size trades local precision against
context and coverage. Overlap helps a long section's boundary not cut away all
relevant surrounding text, but repeats material.

Python: find declarations with AST, keep decorated headers, then split long
regions. Markdown: use headings outside code fences, then split long sections.
Both return exact original slices. Small Python declarations can form short
chunks; this favors precise code locations but must be evaluated for recall.

Exercise: call `chunk_file` on a decorated function containing an accented
string. Print every span and verify `chunk.text == text[start:end]`.

## 4. Understand BM25 without hiding behind a library

- **N:** total number of chunks.
- **df(t):** number of chunks containing term t.
- **tf(t,d):** number of occurrences of t in chunk d.
- **IDF:** rare terms tend to distinguish relevant documents better.
- **k1:** controls how quickly extra term occurrences stop adding much value.
- **b:** controls length normalization; 0 disables it, 1 uses it fully.

If “cache” appears twice, its contribution increases, but not twice as much
forever. A very long chunk is not automatically best because it repeats common
terms. `BM25.search` computes this explicitly, visiting only matching postings.

An inverted index answers “which chunks contain this term?” Its posting list
is the opposite direction to a chunk's token list. This saves scanning the
entire corpus for every question.

Exercise: use lengths `[2, 8]` and postings `{'cache': [(0, 2), (1, 1)]}`.
Compute both scores with a calculator and compare the BM25 implementation.
Change `b` to zero and explain what changes.

## 5. Tokenization choices

A user might write “enable LoRA” while the code contains `enable_lora`.
Keeping only the whole identifier misses natural-language parts; splitting
only into parts loses an exact identifier match. We keep both.

Case is normalized. A small stopword set removes “how”, “the”, etc. It does not
perform a full language understanding step. Stemming and synonym expansion can
help some queries and hurt exact code matching. They are not included by default.
Path and heading tokens add useful context without corrupting the returned span.

## 6. Recall@k: perform a worked example

Suppose a question has two reference sources. The top five retrieved spans match
one of them. That question has recall@5 = 1/2. If another question's three
reference sources are all found, its recall is 1. Macro recall across these two
questions is `(0.5 + 1) / 2 = 0.75`, not `4 / 5`.

To match intervals in the same file:

```text
intersection = max(0, min(end1, end2) - max(start1, start2))
union = (end1-start1) + (end2-start2) - intersection
IoU = intersection / union
```

A reference `[0,100)` and retrieved `[95,100)` have IoU 0.05 and meet the subject's
threshold. A different path never matches. Repeating one retrieved span cannot
count a reference twice. Missing predictions count as zero.

## 7. Explain answer generation

`Generator` does not load Qwen during search. On a real answer request it:

1. Checks every source belongs to the index and its file hash is unchanged.
2. Loads the tokenizer and model once.
3. Creates a system prompt asking for grounded, cited answers.
4. Counts the full templated prompt and reserves generated-output tokens.
5. Adds complete passages that fit; skips ones that do not.
6. Generates only new tokens, decodes them, and creates a Pydantic output.

Why not truncate the whole prompt blindly? You could remove the question or
source labels and silently misrepresent what the model read. Why reserve output
tokens? Input plus output must fit the model's context window. Why not trust
citations automatically? A language model can invent or misapply a citation.

## 8. Understand the bonuses

Embeddings represent text as numerical vectors; normalized-vector dot product
is cosine similarity. The optional index uses a CPU encoder. Hybrid retrieval
combines ranked lists with reciprocal-rank fusion, not by adding unrelated raw
BM25 and cosine scales.

Incremental indexing compares hashes, reuses unchanged chunks, handles deleted
files, and rebuilds global postings. In-process caching holds the index and a
limited LRU query cache. The HTTP server benefits because the process remains
alive. Running a new CLI command does not preserve its previous Python memory.

## 9. Common defense questions

| Question | Answer you should be able to demonstrate |
| --- | --- |
| Why Pydantic? | Validate stage boundaries, IDs, ranges and JSON shape |
| Why 2000 chars? | Subject's hard source-length limit; not the model token limit |
| Why no fabricated answer on a model error? | A valid-looking answer would conceal a failed generation stage |
| Why JSON instead of pickle? | Inspectable, portable, non-executable data |
| Why does a changed index invalidate vectors? | Vector rows refer to exact chunk IDs/order |
| Why does BM25 fail on some paraphrases? | Lexical overlap is required |
| Why is the official recall still unknown? | The reference attachments were not available for measurement |
| What if the corpus changes while serving? | Restart/reload the service after re-indexing; generation rejects changed files |

## 10. Small recode exercises

1. Change the default k from 5 to 3 consistently and test both envelopes.
2. Add a text extension to `EXTENSIONS`; demonstrate it on a temporary corpus.
3. Add a configurable cache limit; explain LRU eviction.
4. Print recall@1, @3 and @5 by reusing `evaluate_results`.
5. Change BM25 b and explain the score changes with the hand example.
6. Add a maximum number of evidence passages without violating token limits.
7. Add a `/stats` endpoint returning only file/chunk counts.

## بالدارجة: شنو خاصك تكون فاهم قبل الديفونس؟

هاد المشروع كيقلب أولاً على المقاطع اللي عندها علاقة بالسؤال، ومن بعد كيعطيها
لـ Qwen باش يجاوب. `BM25` مسؤول على البحث، و`Qwen` مسؤول على صياغة الجواب.
إلا البحث ما جابش المعلومة الصحيحة، تبديل الـ prompt بوحدو ما غاديش يصلح المشكل.

خاصك تقدر تشرح علاش كنحتافظو بالـ `file_path` كيف ما هو، وعلاش الـ offsets
محسوبين بالحروف ماشي بالـ bytes. وخاصك تحسب مثال صغير ديال `BM25` و`recall@5`
بيدك، وتجرب تبدل شي parameter وتشوف النتيجة. قرا الكود ملف بملف، جرّب الأوامر،
ودير مراجعة مع شي peer. النتائج الرسمية خاصها الـ corpus والـ datasets ديال المدرسة.
