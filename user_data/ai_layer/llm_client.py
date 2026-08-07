"""
llm_client.py

Handles interactions with Groq and Gemini API endpoints, including fallback logic and response validation.
"""
import os
import json
import logging
import requests
from groq import Groq
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client wrapper for Groq and Gemini API endpoints with provider fallback.
    """
    def __init__(self, config: dict):
        self.config = config
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

        # Initialize primary Groq client
        self.groq_client = None
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")

        # Initialize fallback Gemini client
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")

    def query_model(self, prompt: str, system_prompt: str, pair: str, use_grounding: bool = False) -> dict:
        """
        Queries the primary LLM model (Groq) and falls back to Gemini if the call fails or is rate-limited.
        If primary fails/rate-limits, fall back to the other; if both fail, always default to veto: true.
        Returns a dictionary with keys: 'veto' (bool), 'confidence' (float), 'reason' (str), 'provider' (str).
        """
        # Try primary Groq
        if not use_grounding and self.groq_client:
            try:
                logger.info(f"Querying Groq (llama-3.1-8b-instant) for {pair}...")
                response = self._query_groq(prompt, system_prompt)
                response["provider"] = "groq"
                return response
            except Exception as e:
                logger.warning(f"Groq API primary call failed or rate-limited: {e}. Falling back to Gemini...")

        # Fallback/Primary Gemini (especially for Tier 2/3 live search grounding)
        if self.gemini_api_key:
            try:
                logger.info(f"Querying Gemini (gemini-1.5-flash) for {pair} (grounding={use_grounding})...")
                response = self._query_gemini(prompt, system_prompt, use_grounding)
                response["provider"] = "gemini"
                return response
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}.")

        # If both fail/rate-limit, always default to veto: true per §B.5
        logger.error(f"Both LLM providers failed or are unconfigured. Defaulting to veto: true for safety.")
        return {
            "veto": True,
            "confidence": 1.0,
            "reason": "Veto default safety trigger: both primary and fallback LLM providers failed or keys not set.",
            "provider": "none"
        }

    def _query_groq(self, prompt: str, system_prompt: str) -> dict:
        """
        Helper method to query Groq's API (llama-3.1-8b-instant).
        """
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            max_tokens=50,
            temperature=0.0
        )
        content = chat_completion.choices[0].message.content
        return self._clean_json_response(content)

    def _query_gemini(self, prompt: str, system_prompt: str, use_grounding: bool) -> dict:
        """
        Helper method to query Gemini's API (gemini-1.5-flash, previously gemini-3.5-flash which CCXT mapping redirects).
        Enables Google Search live grounding if use_grounding is True.
        """
        # Configure model
        model_name = "gemini-1.5-flash"

        # Configure tools for live grounding if requested
        tools = None
        if use_grounding:
            tools = [{"google_search": {}}]

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.0, "max_output_tokens": 50},
            system_instruction=system_prompt,
            tools=tools
        )

        response = model.generate_content(prompt)
        content = response.text
        return self._clean_json_response(content)

    def _clean_json_response(self, text: str) -> dict:
        """
        Safely parses and validates strict JSON response.
        Expected keys: 'veto' (bool), 'confidence' (float), 'reason' (str)
        """
        try:
            data = json.loads(text.strip())
            # Ensure strict key types
            veto = bool(data.get("veto", True))
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "unknown explanation"))
            return {
                "veto": veto,
                "confidence": confidence,
                "reason": reason
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response as JSON: {text}. Error: {e}")
            # Safe default
            return {
                "veto": True,
                "confidence": 1.0,
                "reason": f"Parsing failure. Raw response was: {text[:50]}"
            }
