"""Bounded LRU mapping shared by the middleware's content caches
(warmup.SpanCache, processors.BoundaryCache)."""
from collections import OrderedDict
from typing import Any, Hashable, Optional


class LRUCache:
    """Bounded LRU mapping: ``get`` refreshes recency; ``put`` evicts the
    least-recently-used entry past ``maxsize`` and returns the evicted key."""

    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._entries: "OrderedDict[Hashable, Any]" = OrderedDict()

    def get(self, key: Hashable) -> Optional[Any]:
        """Return the value for key (refreshing its recency), or None."""
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: Hashable, value: Any) -> Optional[Hashable]:
        """Insert key; return the LRU key evicted to make room (or None)."""
        evicted = None
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            evicted, _ = self._entries.popitem(last=False)  # evict least-recently-used
        return evicted

    def discard(self, key: Hashable) -> None:
        """Forget a key."""
        self._entries.pop(key, None)

    def __len__(self) -> int:
        return len(self._entries)
