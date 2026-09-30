"""Commands CLi"""

from .search import search
from .answer import answer
from .answer_dataset import answer_dataset
from .index import index
from .search_dataset import search_dataset
from .evaluate import evaluate


__all__ = [
    "search",
    "answer",
    "answer_dataset",
    "index",
    "search_dataset",
    "evaluate",
]
