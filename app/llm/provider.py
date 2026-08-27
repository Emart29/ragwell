"""Unified LLM provider wrapper for Groq and Gemini."""
import os
import logging
from typing import Optional
from dotenv import load_dotenv
from app.config import settings
from app.logging_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class LLMProvider:
    """Unified LLM provider that tries Groq first, then Gemini."""
    
    def __init__(self):
        self.groq_client = None
        self.gemini_model = None
        
        # Initialize Groq
        groq_api_key = settings.GROQ_API_KEY
        if groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_api_key)
                logger.info("Groq provider initialized")
            except ImportError:
                logger.warning("groq package not installed, Groq provider unavailable")
            except Exception as e:
                logger.error(f"Failed to initialize Groq provider: {e}")

        # Initialize Gemini if Groq is not available or as fallback
        if not self.groq_client:
            gemini_api_key = settings.GEMINI_API_KEY
            if gemini_api_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    logger.info("Gemini provider initialized (fallback)")
                except ImportError:
                    logger.warning("google-generativeai package not installed, Gemini provider unavailable")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini provider: {e}")

    def generate(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text from prompt using available provider.
        
        Tries Groq first, then Gemini. Returns None if both fail or unavailable.
        """
        # Try Groq
        if self.groq_client:
            try:
                logger.info("Generating with Groq...")
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    timeout=10.0
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq generation failed: {e}")
                # Fall through to Gemini

        # Try Gemini
        if self.gemini_model:
            try:
                logger.info("Generating with Gemini...")
                # Gemini 1.5 Flash doesn't have a direct timeout in generate_content
                # but we can wrap it if needed. For now, simple call.
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": 0.7}
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")

        logger.warning("No LLM provider available or both failed")
        return None

    def is_available(self) -> bool:
        """Return True when at least one provider is configured."""
        return self.groq_client is not None or self.gemini_model is not None

# Singleton instance
_provider = None

def get_provider():
    global _provider
    if _provider is None:
        _provider = LLMProvider()
    return _provider

def generate(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """Helper function to generate text using the global provider instance."""
    return get_provider().generate(prompt, max_tokens)
