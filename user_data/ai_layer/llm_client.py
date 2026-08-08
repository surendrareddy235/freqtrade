"""
llm_client.py

Handles interactions with Groq and Gemini API endpoints, including fallback logic and response validation.
"""
import os
import json
import logging
from groq import Groq

# Attempt to import current official Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client wrapper for Groq and Gemini API endpoints with provider fallback.
    """
    def __init__(self, config: dict):
        self.config = config

        # Read keys from environment
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

        # Read model names from config if present, else use defaults
        self.groq_model = self.config.get("groq_model", "llama-3.1-8b-instant")
        self.gemini_model = self.config.get("gemini_model", "gemini-3.5-flash")

        # Initialize primary Groq client
        self.groq_client = None
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("GROQ_API_KEY environment variable not set. Groq client will not be available.")

        # Initialize fallback Gemini client using official google-genai
        self.gemini_client = None
        if HAS_GENAI and self.gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client with google-genai: {e}")
        elif not HAS_GENAI:
            logger.warning("google-genai SDK not installed. Gemini client will not be available.")
        elif not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY/GEMINI_API_KEY environment variable not set. Gemini client will not be available.")

    def query_model(self, prompt: str, system_prompt: str, pair: str) -> dict:
        """
        Queries the primary LLM model (Groq) and falls back to Gemini if the call fails or is rate-limited.
        If primary fails/rate-limits, fall back to Gemini; if both fail, always default to veto: true.
        Returns a dictionary with keys: 'veto' (bool), 'confidence' (float), 'reason' (str), 'provider' (str).
        """
        # Try primary Groq
        if self.groq_client:
            try:
                logger.info(f"Querying Groq primary ({self.groq_model}) for {pair}...")
                response = self._query_groq(prompt, system_prompt)
                response["provider"] = "groq"
                return response
            except Exception as e:
                logger.warning(f"Groq API primary call failed or rate-limited: {e}. Falling back to Gemini...")

        # Fallback Gemini (especially for Tier 2/3 live search grounding)
        if self.gemini_client:
            try:
                logger.info(f"Querying Gemini fallback ({self.gemini_model}) for {pair} (grounding=True)...")
                response = self._query_gemini(prompt, system_prompt)
                response["provider"] = "gemini"
                return response
            except Exception as e:
                logger.error(f"Gemini API fallback call failed: {e}.")

        # If both fail/rate-limit, always default to veto: True per §B.5 / user instructions
        logger.error("Both LLM providers failed or are unconfigured. Defaulting to veto: true for safety.")
        return {
            "veto": True,
            "confidence": 0.0,
            "reason": "llm_provider_failure",
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
            model=self.groq_model,
            response_format={"type": "json_object"},
            max_tokens=35,
            temperature=0.0
        )
        content = chat_completion.choices[0].message.content
        return self._clean_json_response(content)

    def _query_gemini(self, prompt: str, system_prompt: str) -> dict:
        """
        Helper method to query Gemini's API using google-genai.
        Enables Google Search live grounding config for Tier 2/3 fallback queries.
        """
        # Use full prompt combining system instructions & user payload
        combined_prompt = f"System: {system_prompt}\nUser Payload: {prompt}"

        # Configure search grounding tool
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            max_output_tokens=35,
            response_mime_type="application/json"
        )

        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=combined_prompt,
            config=config
        )

        content = response.text
        return self._clean_json_response(content)

    def _clean_json_response(self, text: str) -> dict:
        """
        Safely parses and validates strict JSON response.
        Expected keys: 'veto' (bool), 'confidence' (float), 'reason' (str)
        """
        try:
            # Strip any markdown backticks if present
            cleaned_text = text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            data = json.loads(cleaned_text)

            # Ensure strict key types and structure
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
                "confidence": 0.0,
                "reason": "malformed_json_response"
            }
