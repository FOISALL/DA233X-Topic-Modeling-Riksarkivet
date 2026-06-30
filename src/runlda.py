"""


Dependencies:
    pip install octis gensim
"""

import json
import unicodedata
import string
import os
import numpy as np
import pandas as pd

import shutil

from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

# OCTIS imports
from octis.models.LDA import LDA
from octis.dataset.dataset import Dataset

import time
import os

TIMING_LOG = "final_model_timing_log.csv"

def log_timing(model_name, run_name, K, elapsed_seconds):
    """Appends a timing entry to the shared CSV log."""
    row = pd.DataFrame([{
        "model": model_name,
        "run_name": run_name,
        "K": K,
        "time_seconds": round(elapsed_seconds, 2),
        "time_minutes": round(elapsed_seconds / 60, 2),
    }])
    if os.path.exists(TIMING_LOG):
        row.to_csv(TIMING_LOG, mode='a', header=False, index=False)
    else:
        row.to_csv(TIMING_LOG, mode='w', header=True, index=False)
    print(f"  -> Runtime: {elapsed_seconds/60:.2f} minutes")

# ── Load data & dictionary ─────────────────────────────────────────────────

print("Loading Golden Sample and Dictionary...")
# df_golden = pd.read_parquet("riksarkivet_GOLDEN_neural_150k.parquet")
# dictionary = Dictionary.load("riksarkivet_GOLDEN_bow_dictionary_0stopwords2_20.dict")

df_golden = pd.read_parquet("riksarkivet_final_neural_sample_150k.parquet")
dictionary = Dictionary.load("riksarkivet_final_bow_sample_dictionary_20.dict") # Match your saved filename!
valid_words = set(dictionary.token2id.keys())

# ── Reconstruct heavy_texts (identical to ETM, CTM, BERTopic scripts) ──────

print("Reconstructing heavy_texts...")
remove_punct = str.maketrans('', '', string.punctuation)
light_texts = df_golden['clean_text'].fillna("").tolist()
heavy_texts = []

for text in light_texts:
    clean_str = unicodedata.normalize('NFC', str(text).lower())
    clean_str = clean_str.translate(remove_punct)
    words = [w for w in clean_str.split() if w in valid_words]
    heavy_texts.append(" ".join(words))

# ── Build OCTIS-compatible Dataset ─────────────────────────────────────────


MAX_VOCAB = 2000
DATASET_DIR = "octis_lda_dataset"  # separate folder from ETM to avoid confusion
os.makedirs(DATASET_DIR, exist_ok=True)

print(f"Building vocabulary (max_features={MAX_VOCAB})...")
vectorizer = CountVectorizer(
    token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b",
    max_features=MAX_VOCAB,
    lowercase=True,
)
vectorizer.fit_transform(heavy_texts)
vocab = list(vectorizer.get_feature_names_out())
print(f"  Vocabulary size: {len(vocab)}")

# Tokenize and filter to capped vocabulary
vocab_set = set(vocab)
tokenized_corpus = [
    [w for w in text.split() if w in vocab_set]
    for text in heavy_texts
]

# Write corpus.tsv

print("Writing OCTIS dataset files...")
written = 0
with open(os.path.join(DATASET_DIR, "corpus.tsv"), "w", encoding="utf-8") as f:
    for doc in tokenized_corpus:
        doc_str = " ".join(doc).strip()
        # Must have at least 2 words to be a valid document for OCTIS
        if len(doc) >= 2:
            f.write(f"{doc_str}\ttrain\n")
            written += 1

print(f"  Wrote {written} documents to corpus.tsv")

# Write vocabulary.txt
with open(os.path.join(DATASET_DIR, "vocabulary.txt"), "w", encoding="utf-8") as f:
    for word in vocab:
        clean_word = str(word).strip()
        if clean_word:
            f.write(clean_word + "\n")

# Load via OCTIS
octis_dataset = Dataset()
octis_dataset.load_custom_dataset_from_folder(DATASET_DIR)
print(f"  Dataset loaded. Corpus size: {len(octis_dataset.get_corpus())}")

# ── Main experiment function ───────────────────────────────────────────────

def run_lda(dataset, run_name, K):

    print(f"\n--- Running LDA: {run_name} ---")
    print(f"  Num topics (K): {K}")

    model = LDA(
        num_topics=K,
        alpha='symmetric',
        eta=None,
        passes=10,
        iterations=50,
        chunksize=2000,
        random_state=42,
    )
    model.partitioning(use_partitions=False)  # set after construction

    start = time.time()
    output = model.train_model(dataset)
    log_timing("LDA", run_name, K, time.time() - start)

    # save topic probabilities per document

    topic_doc_matrix = output.get('topic-document-matrix')
    if topic_doc_matrix is not None:
        doc_topics = np.argmax(topic_doc_matrix, axis=0)
    
        topic_labels = {}
        for i, topic_words in enumerate(output['topics']):
            words = [w for w in topic_words[:5] if w and w.strip()]
            topic_labels[i] = "_".join(words)
    
        df_assignments = df_golden[['doc_id']].copy().reset_index(drop=True)
        df_assignments = df_assignments.iloc[:len(doc_topics)].copy()
        df_assignments['topic_id'] = doc_topics
        df_assignments['topic_label'] = df_assignments['topic_id'].map(topic_labels)
        df_assignments.to_csv(f"lda_assignments_{run_name}.csv", index=False)
        print(f"Saved assignments to lda_assignments_{run_name}.csv")

    # ── Export in OCTIS format ─────────────────────────────────────────────────
    octis_topics = []
    for topic_words in output['topics']:
        words = [w for w in topic_words if w and w.strip()]
        if len(words) >= 5:
            octis_topics.append(words[:10])

    out_path = f"lda_topics_{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    print(f"Done. Saved {len(octis_topics)} topics to {out_path}.")

    return octis_topics


# ── Execute experiments ─────────────────────────────────────────────────────


K_VALUES = [10, 20, 30, 40, 50, 60]

for K in K_VALUES:
    run_lda(octis_dataset, f"HEAVY_final_VOCAB2000_K{K}", K)
    run_lda(octis_dataset, f"HEAVY_final2_VOCAB2000_K{K}", K)
    run_lda(octis_dataset, f"HEAVY_final3_VOCAB2000_K{K}", K)