"""
state_cache.py

Provides caching mechanism using hash of discretized feature vectors to avoid redundant LLM calls on similar market states.
"""
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

class StateCache:
    """
    In-memory state cache for storing and retrieving LLM decisions based on discretized feature states.
    Ensures that cached verdicts are strictly isolated per trading pair.
    """
    def __init__(self):
        # Maps (pair, state_hash) -> verdict_dict
        self.cache = {}

    def get_hash(self, pair: str, timeframe: str, indicators: dict) -> str:
        """
        Generates an MD5 hash from discretized canonical indicators.
        Floating-point metrics are rounded to discrete bins to optimize caching efficiency.
        """
        # Define canonical keys we care about
        canonical = {
            "pair": str(pair),
            "timeframe": str(timeframe),
            "close": round(float(indicators.get("close", 0.0)), 4),
            "rsi": round(float(indicators.get("rsi-14", 50.0)) / 2.0) * 2,  # RSI in bins of 2
            "natr": round(float(indicators.get("natr-14", 0.0)), 4),
            "relative_volume": round(float(indicators.get("relative-volume", 1.0)), 1),
            "volatility_state": round(float(indicators.get("volatility_state", 0.0)), 1),
            "model_prediction": int(indicators.get("model_prediction", 0)),
            "model_confidence": round(float(indicators.get("model_confidence", 0.0)), 2),
            "do_predict": int(indicators.get("do_predict", 0)),
            "di_value": round(float(indicators.get("di_value", 0.0)), 2),
            "volume_anomaly": bool(indicators.get("volume_anomaly", False)),
            "volatility_anomaly": bool(indicators.get("volatility_anomaly", False)),
            "high_atr": bool(indicators.get("high_atr", False))
        }

        # Deterministic serialization (stable key order)
        serialized = json.dumps(canonical, sort_keys=True)
        state_hash = hashlib.md5(serialized.encode('utf-8')).hexdigest()
        logger.debug(f"Generated state hash {state_hash} for pair {pair} with state: {canonical}")
        return state_hash

    def lookup(self, pair: str, state_hash: str) -> dict:
        """
        Checks cache for an existing verdict for a specific pair.
        Returns dict if hit, else None.
        """
        key = (pair, state_hash)
        if key in self.cache:
            logger.info(f"Cache HIT for pair {pair} with hash {state_hash}")
            return self.cache[key]
        return None

    def update(self, pair: str, state_hash: str, verdict: dict):
        """
        Stores a verdict in the cache under a pair-specific key.
        """
        key = (pair, state_hash)
        self.cache[key] = verdict
        logger.info(f"Cache UPDATED for pair {pair} with hash {state_hash}")
