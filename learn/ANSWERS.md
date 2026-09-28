# Quiz answer key

Read after completing the quiz.

1. `Indexer.find_files` discovers files; `read_source_file` reads exact text;
   `chunk_text` or `chunk_python_file` creates chunks; `Bm25` builds postings;
   `retrieval.one_question` calls `Bm25.search`; `Answer.context_from_sources`
   reconstructs passages; `Answer.generate` produces an answer.
2. Start is included, end excluded. The slice is `é🙂`. Python string offsets
   count Unicode characters, whereas UTF-8 encodes these characters using
   multiple bytes. Tree-sitter byte offsets need conversion before slicing.
3. `[0,5)`, `[4,9)`, `[8,12)`. Overlap is one character.
4. `['getuser', 'get', 'user', 'user_id', 'user', 'id']`. Full identifiers
   support exact naming matches; parts support queries using individual words.
5. Red: `{0: 2}`. Blue: `{0: 1, 1: 1}`. `N=3`, `avgdl=4/3`, red `df=1`.
   These documents are already tokenized; do not add file-path tokens here.
6. `k1` controls term-frequency saturation; `b` controls document-length
   normalization. `top_k` limits returned sources and is not a scoring parameter.
7. Original chunk order breaks ties. Both no matches and `top_k=0` return `[]`.
8. Intersection is 10, union is 30, IoU is `1/3`. Exactly 0.05 is not a hit:
   the comparison is strict `> 0.05`.
9. `(1 + 0.5 + 0)/3 = 0.5`. Missing predictions count as zero for eligible
   labeled questions. Unlabeled questions and labels with empty sources are
   excluded from the average; no eligible labels raises `ValueError`.
10. Returns `Insufficient retrieved evidence.` without loading the model.
    Examples: a prompt requests source-only answers and numbered citations;
    it tells the model to treat sources as data; context construction rejects
    paths outside the working directory and spans past EOF; generation checks
    the token budget. Prompt instructions encourage grounding but do not
    guarantee that every generated claim is correct.
