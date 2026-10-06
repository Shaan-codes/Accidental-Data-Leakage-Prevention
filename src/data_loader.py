import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df["labels"] = df["label"].apply(
        lambda x: [l.strip() for l in str(x).split(",") if l.strip()]
    )

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["labels"])
    texts = df["text"].tolist()

    return texts, y, mlb
