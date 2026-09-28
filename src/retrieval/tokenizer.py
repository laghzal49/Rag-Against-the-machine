import re
from ..retrieval.stop_words import STOP_WORD

PATTERN = r"\b\w+\b"


def tokenize(text: str) -> list[str]:
    """Extract lowercase words, excluding common English stopwords.

    Args:
        text: Source text, path, or query to analyze.

    Returns:
        Tokens in occurrence order, including repeated words.
    """
    # TODO: Inspect measured retrieval misses before tuning token splitting;
    # compare recall@5/@10 against a fixed, separately evaluated baseline.
    tokens = []
    for token in re.findall(PATTERN, text):
        tokens.append(token.lower())
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
        parts = expanded.replace("_", " ").lower().split(" ")
        if len(parts) > 1:
            tokens.extend(parts)
    new = []
    for token in tokens:
        if token in STOP_WORD:
            continue
        new.append(token)
    return new
