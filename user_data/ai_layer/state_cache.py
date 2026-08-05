"""
state_cache.py

Provides caching mechanism using hash of discretized feature vectors to avoid redundant LLM calls on similar market states.
"""
import hashlib
import json

class StateCache:
    """
    In-memory state cache for storing and retrieving LLM decisions based on feature states.
    """
    def __init__(self):
        self.cache = {}

    def get_hash(self, indicators: dict) -> str:
        """
        Generates a MD5 hash from discretized indicator key-values.
        """
        # Simple placeholder hashing logic
        serialized = json.dumps(indicators, sort_keys=True)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()

    def lookup(self, state_hash: str) -> dict:
        """
        Checks cache for an existing verdict. Returns dict if hit, else None.
        """
        return self.cache.get(state_hash)

    def update(self, state_hash: str, verdict: dict):
        """
        Stores a verdict in the cache.
        """
        self.cache[state_hash] = verdict
