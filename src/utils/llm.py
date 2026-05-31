"""
    Universal LLM call utility supporting Anthropic, OpenAI, and local (Ollama).
    
    Providers:
      - "local" or "qwen" or "ollama": routes to Ollama at localhost:11434
      - "anthropic": uses the Anthropic SDK
      - "openai": uses the OpenAI SDK]
"""

import json

# Ollama-compatible providers 
_OLLAMA_PROVIDERS = {"local", "qwen", "ollama"}


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

        # Build Ollama payload — forward num_predict and temperature if provided
        payload = {
            "model": model or "qwen2.5-coder:7b",
            "prompt": prompt,
            "stream": False,
        }
        # Ollama uses an "options" dict for generation parameters
        options = {}
        if "num_predict" in kwargs:
            options["num_predict"] = kwargs["num_predict"]
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if options:
            payload["options"] = options

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            response = resp.json()["response"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama at localhost:11434. "
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

    return response
