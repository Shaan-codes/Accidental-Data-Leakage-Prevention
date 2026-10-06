import torch

CSV_PATH   = "data/new_prompts.csv"
MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_DIR  = "models/deberta_detector"

SIM_MODEL  = "all-MiniLM-L6-v2"
LOG_FILE   = "chatbot_logs.csv"

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
