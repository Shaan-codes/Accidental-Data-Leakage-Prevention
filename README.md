# 🏦 Banking-Safe AI Chatbot

A compliance-aware AI chatbot that:
- Detects sensitive user intent using DeBERTa
- Explains sensitive words via gradients
- Safely rephrases prompts using Gemini
- Preserves semantic intent (measured via cosine similarity)

## 🔍 Features
- Multi-label sensitivity detection (PII, Financial, SensitiveContext,Safe etc.)
- Gradient-based explainability
- Policy-compliant prompt rewriting
- Semantic similarity validation

## 🛠 Tech Stack
- PyTorch, Transformers (DeBERTa)
- SentenceTransformers
- Gemini API
- Scikit-learn

## 🚀 Run
```bash
export GEMINI_API_KEY=your_key
python run.py
