"""
prompt_templates.py

Defines system prompts, compact user payload formats, and utility functions for building token-efficient prompts.
"""

SYSTEM_PROMPT = """You are a strict financial sanity-check model. Check for adverse news or anomalies on the specified cryptocurrency.
Output JSON only in this format:
{"veto": bool, "confidence": float, "reason": "concise explanation"}
"""

def build_compressed_payload(pair: str, timeframe: str, indicators: dict, prediction_score: float) -> str:
    """
    Builds a highly compressed, token-efficient key-value string payload.
    Example: TICK:DOGE/USDT|TF:5m|C:0.0821|RSI:68.2|NATR:0.012|LGBM_PRED:0.71|DI_OK:1
    """
    # Placeholder implementation
    return f"TICK:{pair}|TF:{timeframe}|LGBM_PRED:{prediction_score}"
