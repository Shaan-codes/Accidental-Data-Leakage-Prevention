from src.gemini_api import gemini_generate

def rephrase_safe(prompt, labels, sensitive_words, keep_words):
    instruction = f"""
You are a compliance AI.
Remove sensitive words: {', '.join(sensitive_words)}
Keep words exactly: {', '.join(keep_words)}
Preserve intent. Output only rewritten prompt.

User prompt:
{prompt}
"""
    rewritten = gemini_generate(instruction)

    for w in keep_words:
        if w.lower() not in rewritten.lower():
            rewritten += f" {w}"

    return rewritten.strip()
