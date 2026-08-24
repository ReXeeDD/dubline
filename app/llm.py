"""LLM access for the translation stage.

Two providers share one call surface so translate.py does not care which is in
use: Groq's hosted models (default, much stronger) and any OpenAI-compatible
local server such as LM Studio or Ollama.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from groq import Groq

DEFAULT_LOCAL_URL = "http://localhost:1234/v1"   # LM Studio's default


# --------------------------------------------------------- local shim ---
class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, base_url: str, api_key: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "not-needed"
        self.timeout = timeout

    def create(self, *, model, messages, temperature=0.2, max_tokens=8000,
               response_format=None, **_ignored):
        """_ignored swallows Groq-only kwargs such as reasoning_format."""
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format:
            body["response_format"] = response_format

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach the local LLM at {self.base_url} - is LM Studio "
                f"running with its server started? ({e})") from e

        return _Response(data["choices"][0]["message"]["content"])


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class LocalClient:
    """Minimal stand-in for the Groq client, backed by an OpenAI-style server."""

    def __init__(self, base_url: str = DEFAULT_LOCAL_URL, api_key: str = "",
                 timeout: int = 600):
        self.chat = _Chat(_Completions(base_url, api_key, timeout))


# ------------------------------------------------------------- factory ---
def make_translation_client(cfg: dict):
    if cfg.get("llm_provider") == "local":
        return LocalClient(cfg.get("local_base_url") or DEFAULT_LOCAL_URL)
    key = cfg.get("groq_api_key")
    if not key:
        raise RuntimeError(
            "No Groq API key set. Open Settings and paste your key (free at "
            "https://console.groq.com/keys), or switch the translation provider "
            "to 'local' to use LM Studio.")
    return Groq(api_key=key)


def api_keys(cfg: dict) -> list[str]:
    """Every Groq key to translate with, primary first, duplicates dropped."""
    keys, seen = [], set()
    for k in [cfg.get("groq_api_key", "")] + list(cfg.get("groq_api_keys") or []):
        k = (k or "").strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def make_translation_clients(cfg: dict) -> list:
    """One client per key.

    Rate limits are metered per organization, so a second key from the *same*
    account shares one bucket and adds nothing. A key belonging to someone else
    is a different organization and does bring its own budget - that is the case
    this exists for. Keys that turn out to share a bucket still work: each
    client paces from the rate-limit headers it gets back, and a shared bucket
    simply reports the same falling number to both.
    """
    if cfg.get("llm_provider") == "local":
        return [LocalClient(cfg.get("local_base_url") or DEFAULT_LOCAL_URL)]
    keys = api_keys(cfg)
    if not keys:
        raise RuntimeError(
            "No Groq API key set. Open Settings and paste your key (free at "
            "https://console.groq.com/keys), or switch the translation provider "
            "to 'local' to use LM Studio.")
    return [Groq(api_key=k) for k in keys]


def translation_model(cfg: dict) -> str:
    if cfg.get("llm_provider") == "local":
        return cfg.get("local_model") or "local-model"
    return cfg["llm_model"]


def helper_model(cfg: dict) -> str:
    """A model to run side work on, so the translator keeps its own budget.

    Rate limits are per model, so putting the cast-list request on a helper
    means the primary model starts translating against a full allowance rather
    than one it has already spent.
    """
    helpers = [m for m in (cfg.get("llm_helpers") or []) if m]
    if cfg.get("llm_provider") == "local" or not helpers:
        return translation_model(cfg)
    return helpers[-1]


def list_local_models(base_url: str = DEFAULT_LOCAL_URL, timeout: float = 1.5) -> list[str]:
    """Short timeout on purpose - LM Studio is usually not running, and the
    Settings panel must not stall waiting to find that out."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return sorted(m["id"] for m in data.get("data", []))
    except Exception:
        return []


def verify_key(api_key: str) -> dict:
    """Check a Groq key by listing models. Returns {ok, detail, chat, asr}."""
    if not api_key:
        return {"ok": False, "detail": "No key entered."}
    try:
        Groq(api_key=api_key).models.list()
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg:
            return {"ok": False, "detail": "Groq rejected this key. "
                                           "Check it was copied in full."}
        return {"ok": False, "detail": f"Could not reach Groq: {msg[:200]}"}

    models = list_groq_models(api_key)
    n = len(models["chat"]) + len(models["asr"])
    return {"ok": True, "detail": f"Key works - {n} usable models available.",
            "chat": models["chat"], "asr": models["asr"]}


def list_groq_models(api_key: str) -> dict[str, list[str]]:
    """Live model list, so the UI never offers an id Groq has retired.

    Goes through the Groq SDK rather than a raw urllib call: Groq sits behind
    Cloudflare, which answers the stdlib User-Agent with 403 "error code: 1010".
    """
    if not api_key:
        return {"chat": [], "helpers": [], "asr": []}
    try:
        ids = [m.id for m in Groq(api_key=api_key).models.list().data if m.id]
    except Exception:
        return {"chat": [], "helpers": [], "asr": []}

    # Everything Groq hosts is in this list, including safety classifiers and
    # speech synthesis. Only conversational models can translate.
    skip = ("prompt-guard", "safeguard", "orpheus", "-tts", "guard-")

    chat, speech = [], []
    for mid in ids:
        low = mid.lower()
        if any(k in low for k in skip):
            continue
        (speech if "whisper" in low else chat).append(mid)

    def rank(m: str) -> tuple:
        # surface the strongest translators first
        pref = ["qwen", "kimi", "llama-3.3", "gpt-oss-120", "deepseek", "llama-4"]
        for i, p in enumerate(pref):
            if p in m.lower():
                return (i, m)
        return (len(pref), m)

    chat = sorted(set(chat), key=rank)

    # A parallel helper translates roughly its share of the finished dub, so a
    # weak or specialised model is heard directly in the output. Compound is an
    # agentic tool-using system, not a plain translator; allam is Arabic-first
    # and small. Both stay selectable as the primary model, just not offered as
    # automatic helpers.
    unfit = ("compound", "allam")
    helpers = [m for m in chat if not any(k in m.lower() for k in unfit)]

    return {"chat": chat, "helpers": helpers, "asr": sorted(set(speech))}
