"""Alternative RAG pipeline using Tree-sitter chunks and Python BM25."""

import os
import warnings

# Clean up deprecated TRANSFORMERS_CACHE to prevent warnings.
os.environ.pop("TRANSFORMERS_CACHE", None)
warnings.filterwarnings(
    "ignore",
    message=r".*TRANSFORMERS_CACHE.*",
    category=FutureWarning,
)
