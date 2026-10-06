import pandas as pd
from datetime import datetime
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_DIR, DEVICE, LOG_FILE
from src.detector import detect_labels
from src.explainer import explain_sensitive_words
from src.rephraser import rephrase_safe
from src.gemini_api import gemini_generate
from src.similarity import semantic_similarity

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

LABELS = model.config.id2label.values()

def chatbot_terminal():
    print("\n🏦 Banking-Safe Chatbot\n")

    while True:
        prompt = input("> ")
        if prompt.lower() in ["exit", "quit"]:
            break

        probs, labels = detect_labels(prompt, model, tokenizer, LABELS, DEVICE)

        if labels == ["Safe"]:
            print(gemini_generate(prompt))
            continue

        sensitive = [w for w, _ in explain_sensitive_words(prompt, model, tokenizer, LABELS, DEVICE)]
        keep = input("Words to KEEP (comma): ").split(",")

        safe_prompt = rephrase_safe(prompt, labels, sensitive, keep)
        ans1 = gemini_generate(prompt)
        ans2 = gemini_generate(safe_prompt)

        sim = semantic_similarity(ans1, ans2)
        print("\nRephrased:", safe_prompt)
        print("Similarity:", round(sim, 3))

        log = {
            "time": datetime.now().isoformat(),
            "original": prompt,
            "rephrased": safe_prompt,
            "labels": labels,
            "similarity": sim
        }

        pd.DataFrame([log]).to_csv(LOG_FILE, mode="a", header=False, index=False)
