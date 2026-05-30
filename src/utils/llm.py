from langgraph.graph import StateGraph, END

def llm_universal_call_utility(prompt: str, provider: str, api_key: str = None, model: str = None, **kwargs):
    """
        This python funciton is a universal utility to pick and choose different types of llms
        for inferencing and getting response for prompts
    """
    response = None
    if provider == "local":
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model or "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        response = response.json()["response"]
    elif provider == "anthropic":
        import anthropic
        # Initialize the client
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model = model,
            max_tokens = 1024,
            messages=[{"role":"user", "content":prompt}]
        )
        response = response.content[0].text
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model = model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        response = response.choices[0].message.content
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")

    return response