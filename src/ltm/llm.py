"""Minimal Gemini REST client using urllib; no SDK dependency required."""
import json
import os
import urllib.error
import urllib.request
from .config import setting


class GeminiError(RuntimeError):
    pass


def _post(url, payload, api_key):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise GeminiError(f"Gemini API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GeminiError(f"Could not reach Gemini API: {exc}") from exc


def generate(prompt, system="You are a careful, concise assistant.", max_output_tokens=1200):
    """Generate text; callers may set a task-appropriate output token budget."""
    if not isinstance(max_output_tokens, int) or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise GeminiError("GEMINI_API_KEY is not set. Add it to .env or the environment.")
    model = setting("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _post(url, {"system_instruction":{"parts":[{"text":system}]},
        "contents":[{"role":"user","parts":[{"text":prompt}]}],
        "generationConfig":{"temperature":0.3,"maxOutputTokens":max_output_tokens}}, key)
    try:
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise GeminiError("Gemini stopped at the output token limit; increase the budget or request a shorter draft.")
        return "".join(p.get("text","") for p in candidate["content"]["parts"]).strip()
    except GeminiError:
        raise
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected Gemini response: {data}") from exc


def embed(text):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    model = setting("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    data = _post(url, {"model":f"models/{model}","content":{"parts":[{"text":text[:8000]}]}}, key)
    try:
        return data["embedding"]["values"]
    except (KeyError, TypeError) as exc:
        raise GeminiError(f"Unexpected embedding response: {data}") from exc
