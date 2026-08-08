from __future__ import annotations

import keyring

from app.providers.base import ChatMessage, GenerationSettings, LLMProvider, ModelLoadError

KEYRING_SERVICE = "FriendApp"
KEYRING_USERNAME = "gemini_api_key"


def save_api_key(api_key: str) -> None:
    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, api_key)


def has_api_key() -> bool:
    return bool(keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME))


def clear_api_key() -> None:
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except keyring.errors.PasswordDeleteError:
        pass


def masked_key_display() -> str:
    key = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if not key:
        return "(not set)"
    return "*" * max(len(key) - 4, 0) + key[-4:]


class GeminiProvider(LLMProvider):
    """
    Optional cloud fallback. Never called unless app/router.py routes here, and the
    router only routes here for current-info/web/complex requests when the user has
    enabled Gemini. Sends minimum context: current message + summary + relevant
    long-term memory -- never the full raw conversation history (spec section 17).
    """

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        self.model_id = model_id
        self._client = None
        self._loaded = False

    def load(self) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ModelLoadError(
                "google-generativeai is not installed.",
                suggestion="pip install google-generativeai",
            ) from e

        api_key = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        if not api_key:
            raise ModelLoadError(
                "No Gemini API key saved.",
                suggestion="Add your API key in Settings -> Gemini.",
            )

        try:
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model_id)
            self._loaded = True
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize Gemini client: {e}") from e

    def unload(self) -> None:
        self._client = None
        self._loaded = False

    def validate_key(self, api_key: str) -> tuple[bool, str]:
        """Lightweight validation call so Settings can confirm a key works before saving."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_id)
            model.generate_content("ping", generation_config={"max_output_tokens": 5})
            return True, "Key is valid."
        except Exception as e:
            return False, f"Could not validate key: {e}"

    def generate(self, messages: list[ChatMessage], settings: GenerationSettings) -> str:
        if not self._loaded:
            raise ModelLoadError("Gemini not loaded. Call load() first.")

        system = "\n".join(m.content for m in messages if m.role == "system")
        convo = "\n".join(f"{m.role}: {m.content}" for m in messages if m.role != "system")
        prompt = f"{system}\n\n{convo}".strip()

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "max_output_tokens": settings.max_output_tokens,
                },
            )
            return (response.text or "").strip()
        except Exception as e:
            raise ModelLoadError(
                f"Gemini request failed: {e}",
                suggestion="Check your internet connection and API key, or the app will fall back to local.",
            ) from e
