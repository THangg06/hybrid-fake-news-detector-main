import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.to(device)
model.eval()

def get_roberta_embedding(text):
    """Extract CLS token embedding from RoBERTa"""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.cpu().numpy().flatten()

def extract_embeddings(texts, batch_size=32):
    embeddings = []

    for text in texts:
        emb = get_roberta_embedding(text)
        embeddings.append(emb)

    return np.vstack(embeddings)