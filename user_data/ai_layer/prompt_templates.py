"""
prompt_templates.py

Defines system prompts, compact user payload formats, and utility functions for building token-efficient prompts.
"""

SYSTEM_PROMPT = (
    "Check adverse news/anomalies on this specific coin right now. Scope is narrow lookup only, NOT investment advice. "
    "Output strict JSON only format: "
    '{"veto": bool, "confidence": float, "reason": "concise explanation"}'
)

def build_compressed_payload(pair: str, timeframe: str, indicators: dict, prediction_score: float) -> str:
    """
    Builds a highly compressed, token-efficient key-value string payload.
    Target under 80 input tokens.
    Example: TICK:XRP/USDT|TF:5m|LGBM:0.71|RSI:55.2|NATR:0.012|DI:1
    """
    # Exclude complex indicator matrices, serialize only key values
    tick = pair.split("/")[0]
    rsi = round(indicators.get("rsi-14", 50.0), 1)
    natr = round(indicators.get("natr-14", 0.0), 4)
    vol_rel = round(indicators.get("relative-volume", 1.0), 2)

    payload = f"TICK:{tick}|TF:{timeframe}|LGBM:{round(prediction_score, 2)}|RSI:{rsi}|NATR:{natr}|VREL:{vol_rel}"
    return payload
