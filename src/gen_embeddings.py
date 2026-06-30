import pandas as pd
from sentence_transformers import SentenceTransformer
from gensim.corpora import Dictionary
import numpy as np
import torch
from tqdm import tqdm
import unicodedata

import string
import re

remove_punct = str.maketrans('', '', string.punctuation)
remove_nums = re.compile(r'\d+')



# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load SAMPLE
print("Loading Sample and Filtered Dictionary...")
# df_golden = pd.read_parquet("riksarkivet_final_neural_sample_150k.parquet")
# dictionary = Dictionary.load("riksarkivet_final_bow_sample_dictionary_20.dict")
df_golden = pd.read_parquet("riksarkivet_neural_cleaned.parquet")
dictionary = Dictionary.load("riksarkivet_2M_bow_dictionary_20.dict")


valid_words = set(dictionary.token2id.keys())

# Create the two datasets
lightly_processed_texts = df_golden['clean_text'].fillna("").tolist()
heavily_processed_texts =[]

print("Reconstructing 'Heavy' text (keeping only words that survived the Gensim filter)...")
for text in tqdm(lightly_processed_texts):
    # Normalize to NFC and lowercase
    clean_text = unicodedata.normalize('NFC', str(text).lower())

    clean_text = clean_text.translate(remove_punct)   # match dictionary
    clean_text = remove_nums.sub('', clean_text)       # match dictionary
    # 
    filtered_words =[word for word in str(clean_text).split() if word in valid_words]
    heavily_processed_texts.append(" ".join(filtered_words))

# Load your Fine-Tuned Historical SBERT model
model_path = "riksarkivet-sbert-historical-swedish-200k" 
print(f"\nLoading custom SBERT model from {model_path}...")
model = SentenceTransformer(model_path, device=device)

# ==========================================
# Embed Lightly Processed Data (Context Retained)

print("\n--- Generating Embeddings: LIGHTLY PROCESSED (Full Sentences) ---")
embeddings_light = model.encode(
    lightly_processed_texts, 
    batch_size=32, 
    show_progress_bar=True, 
    device=device
)
np.save("riksarkivet_2M_embeddings_LIGHT.npy", embeddings_light)
del embeddings_light # Clear RAM


# Embed Heavily Processed Data

print("\n--- Generating Embeddings: HEAVILY PROCESSED (Filtered Words Only) ---")
embeddings_heavy = model.encode(
    heavily_processed_texts, 
    batch_size=32, 
    show_progress_bar=True, 
    device=device
)
np.save("riksarkivet_2M_embeddings_HEAVY.npy", embeddings_heavy)

print("\nSuccess! ")