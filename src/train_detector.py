import os, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.config import CSV_PATH, MODEL_NAME, MODEL_DIR
from src.data_loader import load_dataset
from src.dataset import PromptDataset

def compute_metrics(pred):
    logits, labels = pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro")
    }

def train_detector():
    texts, y, mlb = load_dataset(CSV_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PromptDataset(X_train, y_train, tokenizer)
    val_ds   = PromptDataset(X_val, y_val, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification"
    )

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        save_strategy="no",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(" Detector trained and saved")
