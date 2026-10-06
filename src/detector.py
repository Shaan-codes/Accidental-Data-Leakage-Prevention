import torch

def detect_labels(prompt, model, tokenizer, labels, device):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).flatten().tolist()

    detected = [labels[i] for i, p in enumerate(probs) if p > 0.5]
    return dict(zip(labels, probs)), detected or ["Safe"]
