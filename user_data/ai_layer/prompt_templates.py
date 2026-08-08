"""
prompt_templates.py

Defines system prompts, compact user payload formats, and utility functions for building token-efficient prompts.
"""

SYSTEM_PROMPT = (
    "Check for adverse news/anomalies on this specific coin right now. Scope is narrow lookup only, NOT investment advice. "
    "Output strict JSON format only, no markdown, no other text: "
    '{"veto": bool, "confidence": float, "reason": "concise reason"}'
)

def build_compressed_payload(pair: str, timeframe: str, indicators: dict, prediction_score: float) -> str:
    """
    Builds a highly compressed, token-efficient key-value string payload.
    Target under 80 input tokens.
    Example: TICK:XRP/USDT|TF:5m|LGBM:0.71|RSI:55|NATR:0.012|VREL:1.2|ANOM_V:1
    """
    tick = pair.split("/")[0]
    rsi = int(round(indicators.get("rsi-14", 50.0)))
    natr = round(indicators.get("natr-14", 0.0), 4)
    vrel = round(indicators.get("relative-volume", 1.0), 1)

    # Anomaly flags
    vol_anom = 1 if indicators.get("volume_anomaly", False) else 0
    vlt_anom = 1 if indicators.get("volatility_anomaly", False) else 0
    high_atr = 1 if indicators.get("high_atr", False) else 0

    payload = (
        f"TICK:{tick}|TF:{timeframe}|LGBM:{round(prediction_score, 2)}|"
        f"RSI:{rsi}|NATR:{natr}|VREL:{vrel}|V_ANOM:{vol_anom}|VOLT_ANOM:{vlt_anom}|HATR:{high_atr}"
    )
    return payload
