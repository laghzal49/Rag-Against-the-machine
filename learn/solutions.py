"""Reference solutions: try the ex*.py files before reading this file."""

import math
import re
from collections import Counter

from src.retrieval.bm25 import Bm25
from src.models import Chunk, UnansweredQuestion
from src.retrieval import one_question
from src.retrieval.tokenizer import tokenize as tokenize_source
from src.stop_word import STOP_WORD


def extract_span(text, start, end):
    if not 0 <= start < end <= len(text) or end - start > 2000:
        raise ValueError("Invalid source span")
    return text[start:end]


def split_windows(text, size):
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 2000:
        raise ValueError("Invalid chunk size")
    windows = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        windows.append((start, end, text[start:end]))
        if end == len(text):
            break
        start = end - min(200, size // 5)
    return windows


def tokenize(text):
    result = []
    for word in re.findall(r"\b\w+\b", text):
        result.append(word.lower())
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
        parts = parts.replace("_", " ").lower().split()
        if len(parts) > 1:
            result.extend(parts)
    return [word for word in result if word not in STOP_WORD]


def build_postings(documents):
    postings = {}
    for doc_id, tokens in enumerate(documents):
        for word, count in Counter(tokens).items():
            postings.setdefault(word, {})[doc_id] = count
    return postings


def bm25_term_score(n, df, tf, dl, avgdl, k=1.5, b=0.5):
    if tf == 0:
        return 0.0
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (k + 1) / (tf + k * (1 - b + b * dl / avgdl))


def span_iou(first, second):
    if first.file_path != second.file_path:
        return 0.0
    intersection = max(0, min(first.last_character_index, second.last_character_index)
                       - max(first.first_character_index, second.first_character_index))
    union = (first.last_character_index - first.first_character_index
             + second.last_character_index - second.first_character_index - intersection)
    return intersection / union


def source_recall(references, predictions):
    if not references or not predictions:
        return 0.0
    hits = sum(any(span_iou(ref, pred) > 0.05 for pred in predictions)
               for ref in references)
    return hits / len(references)


def retrieve_texts(documents, query, top_k):
    chunks = [Chunk(file_path=path, text=text, first_character_index=0,
                    last_character_index=len(text)) for path, text in documents]
    question = UnansweredQuestion(question_id="practice-001", question=query)
    corpus = {i: tokenize_source(chunk.text) + tokenize_source(chunk.file_path)
              for i, chunk in enumerate(chunks)}
    doc_len = {i: len(tokens) for i, tokens in corpus.items()}
    avgdl = sum(doc_len.values()) / len(chunks) if chunks else 0
    bm25 = Bm25(corpus, avgdl, len(chunks), doc_len, chunks)
    return one_question(question, top_k, bm25)
