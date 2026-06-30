import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import unicodedata


#Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load full parquet file
print("Loading full corpus...")
df_golden = pd.read_parquet("riksarkivet_neural_cleaned.parquet")

# Create lightly processed texts
lightly_processed_texts = [
    unicodedata.normalize('NFC', str(text))
    for text in df_golden['clean_text'].fillna("").tolist()
]

print(f"Documents: {len(lightly_processed_texts)}")

# Load Fine-Tuned Historical SBERT model
model_path = "riksarkivet-sbert-historical-swedish-200k"
print(f"\nLoading custom SBERT model from {model_path}...")
model = SentenceTransformer(model_path, device=device)

# Generate LIGHT embeddings
print("\n--- Generating Embeddings: LIGHTLY PROCESSED FULL CORPUS ---")
embeddings_light = model.encode(
    lightly_processed_texts,
    batch_size=32,
    show_progress_bar=True,
    device=device,
    convert_to_numpy=True,
)

print("Embeddings shape:", embeddings_light.shape)
assert embeddings_light.shape[0] == len(lightly_processed_texts)

np.save("riksarkivet_2M_embeddings_LIGHT.npy", embeddings_light.astype(np.float32))

print("\nSuccess!")