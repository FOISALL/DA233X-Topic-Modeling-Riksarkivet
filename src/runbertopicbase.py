import pandas as pd
import numpy as np
from bertopic import BERTopic
import json
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
from gensim.corpora import Dictionary
import string

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

test_vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b")
analyze = test_vectorizer.build_analyzer()
result = analyze("Här är hofrätten i år")

print(f"Vectorizer Test: {result}") 
# Expected Output: ['här', 'hofrätten'] 
# ('är' and 'år' are dropped because they are only 2 letters long)

# Load Parquet File and Dictionary
print("Loading Golden Sample and Dictionary...")
df_golden = pd.read_parquet("riksarkivet_final_neural_sample_150k.parquet")
dictionary = Dictionary.load("riksarkivet_final_bow_sample_dictionary_20.dict") # Match your saved filename!
valid_words = set(dictionary.token2id.keys())

# Check if the dictionary actually contains Swedish characters
sw_keys = [k for k in dictionary.token2id.keys() if any(c in k for c in "åäö")]
print(f"Words with åäö in dictionary: {len(sw_keys)}")
print(f"Sample: {sw_keys[:10]}")

# Reconstruct heavy_texts (Must match your embedding logic exactly!)
print("Reconstructing heavy_texts...")
remove_punct = str.maketrans('', '', string.punctuation)
light_texts = df_golden['clean_text'].fillna("").tolist()
heavy_texts = []



for text in light_texts:
    # Match the exact normalization used during embedding generation
    clean_str = unicodedata.normalize('NFC', str(text).lower())
    clean_str = clean_str.translate(remove_punct)
    words = [w for w in clean_str.split() if w in valid_words]
    heavy_texts.append(" ".join(words))

# Load the pre-calculated NumPy embeddings
print("Loading Neural Embeddings...")
# embeddings_light = np.load("riksarkivet_GOLDEN_embeddings_LIGHT.npy")
# embeddings_heavy = np.load("riksarkivet_GOLDEN_embeddings_HEAVY3.npy")

# embeddings used in main experiment
embeddings_light = np.load("riksarkivet_final_embeddings_LIGHT.npy")
embeddings_heavy = np.load("riksarkivet_final_embeddings_HEAVY.npy")

# embeddings for final full corpus run
# embeddings_heavy = np.load("riksarkivet_2M_embeddings_HEAVY.npy")

# Minimalist Vectorizer
# We use (?u)\b\w+\b to ensure Swedish characters are safe, 
# but we set stop_words to None because your 'heavy_texts' are already clean.
vectorizer_model = CountVectorizer(
    token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b", 
    stop_words=None,
    lowercase=True,
    min_df=2,
)

sample = [t for t in heavy_texts if any(c in t for c in "åäö")]
print(f"Texts with åäö: {len(sample)} / {len(heavy_texts)}")
print(sample[0][:300])

def run_pure_bertopic(texts, embeddings, run_name, K):
    print(f"\n--- Running BERTopic: {run_name} ---")
    
    # We set top_n_words to 10 to match your OCTIS requirement
    topic_model = BERTopic(
        nr_topics=K,
        language=None, # Very important line, otherwise it removes åäö from all words
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )

    # fit_transform ignores its internal embedding model because you provided 'embeddings'
    start = time.time()
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    log_timing("BERTopic", run_name, K, time.time() - start)

    # Export for OCTIS
    topic_info = topic_model.get_topic_info()
    octis_topics = []
    for topic_id in topic_info.Topic:
        if topic_id != -1:
            words = [word for word, _ in topic_model.get_topic(topic_id)[:10] if word and word.strip()] 
            
            if len(words) >= 5:  # skip degenerate topics, this is probably excessive, as we already remove bad topics ine vcaluation
                octis_topics.append(words)

    with open(f"bertopic_topics_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)
    
    print(f"Done. Saved {len(octis_topics)} topics.")

        # ── Export document-topic assignments ─────────────────────────────────────
    # 'topics' is a list of length n_docs where topics[i] = topic_id for doc i.
    # -1 means the document was classified as noise (no topic assigned).
    df_assignments = df_golden[['doc_id']].copy().reset_index(drop=True)
    df_assignments['topic_id'] = topics

    # Merge in the top words for each topic so the file is human-readable
    topic_labels = {}
    for topic_id in topic_model.get_topic_info().Topic:
        if topic_id != -1:
            words = [w for w, _ in topic_model.get_topic(topic_id)[:5]
                     if w and w.strip()]
            topic_labels[topic_id] = "_".join(words)
        else:
            topic_labels[-1] = "noise"

    df_assignments['topic_label'] = df_assignments['topic_id'].map(topic_labels)

    # Save as CSV for easy inspection
    assignments_path = f"bertopic_assignments_{run_name}.csv"
    df_assignments.to_csv(assignments_path, index=False)
    print(f"Done. Saved {len(octis_topics)} topics and assignments to {assignments_path}.")

    with open(f"bertopic_topics_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)
    
    print(f"Done. Saved {len(octis_topics)} topics.")

        # ── Export document-topic assignments ─────────────────────────────────────
    # 'topics' is a list of length n_docs where topics[i] = topic_id for doc i.
    # -1 means the document was classified as noise (no topic assigned).
    df_assignments = df_golden[['doc_id']].copy().reset_index(drop=True)
    df_assignments['topic_id'] = topics

    # Merge in the top words for each topic so the file is human-readable
    topic_labels = {}
    for topic_id in topic_model.get_topic_info().Topic:
        if topic_id != -1:
            words = [w for w, _ in topic_model.get_topic(topic_id)[:5]
                     if w and w.strip()]
            topic_labels[topic_id] = "_".join(words)
        else:
            topic_labels[-1] = "noise"

    df_assignments['topic_label'] = df_assignments['topic_id'].map(topic_labels)

    # Save as CSV for easy inspection
    assignments_path = f"bertopic_assignments_{run_name}.csv"
    df_assignments.to_csv(assignments_path, index=False)
    print(f"Done. Saved {len(octis_topics)} topics and assignments to {assignments_path}.")

# Execute experiments
K_VALUES = [10, 20, 30, 40, 50, 60]

for K in K_VALUES:

    run_pure_bertopic(heavy_texts, embeddings_light, f"LIGHT_EMBfinal_fullK{K}", K)
    run_pure_bertopic(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal_fullK{K}", K)
    run_pure_bertopic(heavy_texts, embeddings_light, f"LIGHT_EMBfinal2_fullK{K}", K)
    run_pure_bertopic(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal2_fullK{K}", K)
    run_pure_bertopic(heavy_texts, embeddings_light, f"LIGHT_EMBfinal3_fullK{K}", K)
    run_pure_bertopic(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal3_fullK{K}", K)
    
