"""Small corpus used by the repository reproducible RAG demo."""

from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Store a cached value and the monotonic time at which it expires."""

    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """A tiny time-to-live cache with lazy expiration.

    Expired entries are removed when get reads their key. Calling
    purge_expired removes every expired entry eagerly.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._entries: dict[str, CacheEntry[T]] = {}

    def put(self, key: str, value: T, ttl_seconds: float) -> None:
        """Store a value until the requested lifetime has elapsed."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._entries[key] = CacheEntry(
            value=value,
            expires_at=monotonic() + ttl_seconds,
        )

    def get(self, key: str) -> T | None:
        """Return a live value and lazily delete an expired entry."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= monotonic():
            del self._entries[key]
            return None
        return entry.value

    def purge_expired(self) -> int:
        """Eagerly delete all expired entries and return the removal count."""
        now = monotonic()
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for key in expired:
            del self._entries[key]
        return len(expired)
