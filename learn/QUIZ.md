# RAG understanding test

Time: 30–45 minutes. Total: 20 points, two points per question.
Write answers in `my_answers.md` before opening `ANSWERS.md`.

1. Trace a file through this project from discovery to a generated answer.
   Name at least four actual classes or functions in order.
2. What does `[first_character_index:last_character_index]` mean? For
   `text = 'aé🙂z'`, give the slice `[1:3]` and explain the byte-offset issue.
3. Chunk `'abcdefghijkl'` with maximum size 5. List all start/end offsets.
4. Give the exact tokens for `'The getUser user_id'`. Why retain both the
   original identifier and its parts?
5. Given token documents `[['red','red','blue'], ['blue'], []]`, write
   the postings for red and blue, `N`, `avgdl`, and red's document frequency.
6. Explain the roles of `k1` and `b` in BM25. Is `top_k` the same as `k1`?
7. Two chunks have equal BM25 scores. Which comes first? What happens for
   a query with no matching tokens, or for `top_k=0`?
8. Compute IoU for same-file spans `[0,20)` and `[10,30)`. Does an IoU
   of exactly 0.05 count as a hit in this project?
9. There are three labeled questions with nonempty reference sources.
   Their recalls are 1, 0.5, and missing prediction. What is dataset macro
   recall? What happens to unlabeled questions?
10. What does `Answer.generate` do when context is empty? Give two additional
    mechanisms in `answer.py` that constrain or check answer generation.

Scoring: award one point per correct main part (question 5: one for postings,
one for all statistics). For question 1, award two for a correct sequence of
at least four names and one for a partly correct sequence. For question 10,
award one for the empty-context behavior and one for two mechanisms.

- 17–20: ready to explain the pipeline and attempt the optional investigation.
- 12–16: revisit the source sections behind missed questions.
- 0–11: repeat tasks 1–4, then trace one query by hand.
