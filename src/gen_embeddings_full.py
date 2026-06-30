import os
import re
import string
import unicodedata
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer


# ── Settings ─────

INPUT_FILE = "riksarkivet_neural_cleaned.parquet"
DICTIONARY_FILE = "riksarkivet_2M_bow_dictionary_20.dict"
MODEL_PATH = "riksarkivet-sbert-historical-swedish-200k"

CHUNK_DIR = "riksarkivet_2M_heavy_embedding_chunks"
FINAL_OUTPUT = "riksarkivet_2M_embeddings_HEAVY.npy"

CHUNK_SIZE = 50000
BATCH_SIZE = 32


# ── Setup ──

os.makedirs(CHUNK_DIR, exist_ok=True)

remove_punct = str.maketrans("", "", string.punctuation)
remove_nums = re.compile(r"\d+")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Load data ────

print("Loading full corpus and dictionary...")
df = pd.read_parquet(INPUT_FILE, columns=["clean_text"])
dictionary = Dictionary.load(DICTIONARY_FILE)
valid_words = set(dictionary.token2id.keys())

texts = df["clean_text"].fillna("").tolist()
n_docs = len(texts)

print(f"Documents: {n_docs}")
print(f"Dictionary size: {len(dictionary)}")


# ── Load model 

print(f"Loading SBERT model from {MODEL_PATH}...")
model = SentenceTransformer(MODEL_PATH, device=device)


def make_heavy_text(text):
    clean_text = unicodedata.normalize("NFC", str(text).lower())
    clean_text = clean_text.translate(remove_punct)
    clean_text = remove_nums.sub("", clean_text)

    words = [
        word for word in clean_text.split()
        if word in valid_words
    ]

    return " ".join(words)


# ── Encode in chunks with checkpointing ─────

num_chunks = (n_docs + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"Encoding in {num_chunks} chunks of up to {CHUNK_SIZE} documents...")

for chunk_idx in range(num_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_docs)

    chunk_path = os.path.join(
        CHUNK_DIR,
        f"embeddings_heavy_chunk_{chunk_idx:05d}_{start}_{end}.npy"
    )

    if os.path.exists(chunk_path):
        print(f"Skipping chunk {chunk_idx + 1}/{num_chunks}; already exists: {chunk_path}")
        continue

    print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}: docs {start}–{end}")

    chunk_texts_raw = texts[start:end]

    chunk_texts_heavy = [
        make_heavy_text(text)
        for text in tqdm(chunk_texts_raw, desc="Preprocessing chunk")
    ]

    chunk_embeddings = model.encode(
        chunk_texts_heavy,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True,
    )

    np.save(chunk_path, chunk_embeddings.astype(np.float32))

    print(f"Saved {chunk_path} with shape {chunk_embeddings.shape}")

    del chunk_texts_raw
    del chunk_texts_heavy
    del chunk_embeddings

print("\nAll chunks encoded.")


# ── Merge chunks into one final .npy file ──
print("\nMerging chunks...")

chunk_files = sorted([
    os.path.join(CHUNK_DIR, f)
    for f in os.listdir(CHUNK_DIR)
    if f.startswith("embeddings_heavy_chunk_") and f.endswith(".npy")
])

if len(chunk_files) != num_chunks:
    raise RuntimeError(
        f"Expected {num_chunks} chunks, but found {len(chunk_files)}. "
        "Some chunks are missing."
    )

arrays = []
for path in tqdm(chunk_files, desc="Loading chunks"):
    arrays.append(np.load(path))

embeddings = np.vstack(arrays)

print("Final embeddings shape:", embeddings.shape)
assert embeddings.shape[0] == n_docs

np.save(FINAL_OUTPUT, embeddings.astype(np.float32))
print(f"Saved final embeddings to {FINAL_OUTPUT}")