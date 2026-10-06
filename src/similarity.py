from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(a, b):
    emb = model.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()
