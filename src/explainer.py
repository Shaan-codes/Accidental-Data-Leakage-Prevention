import torch

def explain_sensitive_words(prompt, model, tokenizer, labels, device, top_k=6):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    embeddings = model.deberta.embeddings.word_embeddings(inputs["input_ids"])
    embeddings.requires_grad_(True)

    outputs = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
    probs = torch.sigmoid(outputs.logits)

    sens_idx = [i for i, l in enumerate(labels) if l != "Safe"]
    loss = probs[0, sens_idx].mean()
    loss.backward()

    grads = embeddings.grad.abs().sum(dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    scores = sorted(
        [(t.replace("▁", ""), float(g)) for t, g in zip(tokens, grads)],
        key=lambda x: -x[1]
    )

    return scores[:top_k]
