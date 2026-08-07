"""
state_cache.py

Provides caching mechanism using hash of discretized feature vectors to avoid redundant LLM calls on similar market states.
"""
import hashlib
import json

class StateCache:
    """
    In-memory state cache for storing and retrieving LLM decisions based on discretized feature states.
    """
    def __init__(self):
        self.cache = {}

    def get_hash(self, indicators: dict) -> str:
        """
        Generates a MD5 hash from discretized indicator key-values.
        We round floating-point metrics to discrete bins to optimize caching efficiency.
        """
        discretized = {}
        for k, v in indicators.items():
            if isinstance(v, (int, float)):
                # Discretize continuous floats by multiplying by 10/100 and rounding
                if "rsi" in k:
                    discretized[k] = round(v / 5.0) * 5  # RSI in bins of 5
                elif "natr" in k or "atr" in k:
                    discretized[k] = round(v, 4)
                else:
                    discretized[k] = round(v, 2)
            else:
                discretized[k] = v

        serialized = json.dumps(discretized, sort_keys=True)
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
