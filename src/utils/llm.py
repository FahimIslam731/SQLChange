"""
    Universal LLM call utility supporting Anthropic, OpenAI, and local (Ollama).
    
    Providers:
      - "local" or "qwen" or "ollama": routes to Ollama at 127.0.0.1:11435
      - "anthropic": uses the Anthropic SDK
      - "openai": uses the OpenAI SDK]
"""

import json
import re

# Ollama-compatible providers
_OLLAMA_PROVIDERS = {"local", "qwen", "ollama"}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from thinking-model responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def llm_universal_call_utility(
    prompt: str,
    provider: str,
    api_key: str = None,
    model: str = None,
    **kwargs,
):
    """
        Universal utility to pick and choose different LLM providers
        for inference. Returns the raw text response string.
    """
    response = None

    if provider in _OLLAMA_PROVIDERS:
        import requests

        payload = {
            "model": model or "qwen2.5-coder:7b",
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        options = {}
        if "num_predict" in kwargs:
            options["num_predict"] = kwargs["num_predict"]
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if options:
            payload["options"] = options

        try:
            resp = requests.post(
                "http://127.0.0.1:11435/api/chat",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            response = resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama at 127.0.0.1:11435. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=kwargs.get("num_predict", 1024),
            messages=[{"role": "user", "content": prompt}],
        )
        response = msg.content[0].text

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        # Filter out non-OpenAI kwargs
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ("num_predict",)}
        msg = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            **openai_kwargs,
        )
        response = msg.choices[0].message.content

    else:
        raise ValueError(
            f"Unknown LLM Provider: '{provider}'. "
            f"Supported: {sorted(_OLLAMA_PROVIDERS | {'anthropic', 'openai'})}"
        )

    return _strip_thinking(response)
