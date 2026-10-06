import os, requests

def gemini_generate(prompt, temperature=0.3, max_tokens=200):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    url = os.getenv("GEMINI_URL")
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    r = requests.post(url, headers=headers, params=params, json=data)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
