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

# Load the Parquet File and Dictionary
print("Loading Golden Sample and Dictionary...")
df_golden = pd.read_parquet("riksarkivet_final_neural_sample_150k.parquet")
dictionary = Dictionary.load("riksarkivet_final_bow_sample_dictionary_20.dict") # Match your saved filename!
valid_words = set(dictionary.token2id.keys())

# Check if the dictionary actually contains Swedish characters
sw_keys = [k for k in dictionary.token2id.keys() if any(c in k for c in "åäö")]
print(f"Words with åäö in dictionary: {len(sw_keys)}")
print(f"Sample: {sw_keys[:10]}")

# Reconstruct heavy_texts 
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
embeddings_light = np.load("riksarkivet_final_embeddings_LIGHT.npy")
embeddings_heavy = np.load("riksarkivet_final_embeddings_HEAVY.npy")

# Minimalist Vectorizer
# We use (?u)\b\w+\b to ensure Swedish characters are safe, 
# but we set stop_words to None because your 'heavy_texts' are already clean.
vectorizer_model = CountVectorizer(
    token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b", 
    stop_words=None,
    lowercase=True,
    min_df = 2
)

sample = [t for t in heavy_texts if any(c in t for c in "åäö")]
print(f"Texts with åäö: {len(sample)} / {len(heavy_texts)}")
print(sample[0][:300])

def run_bertopic_with_reassignment(texts, embeddings, run_name, K):
    print(f"\n--- Running BERTopic (Reassignment): {run_name} ---")
    
    topic_model = BERTopic(
        nr_topics=K, 
        vectorizer_model=vectorizer_model,
        verbose=True,
        language = None

    )

    start = time.time()
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    log_timing("BERTopic", run_name, K, time.time() - start)

    # Post-process: Force all noise into nearest cluster
    new_topics = topic_model.reduce_outliers(texts, topics, 
                                            strategy="embeddings", 
                                            embeddings=embeddings)
    
    # Update model to recalculate c-TF-IDF for the new cluster assignments
    topic_model.update_topics(texts, topics=new_topics)
    
    # Export for OCTIS (using updated topics)
    topic_info = topic_model.get_topic_info()
    octis_topics = []
    for topic_id in topic_info.Topic:
        if topic_id != -1: # After reassignment, there should be no -1
            words = [word for word, _ in topic_model.get_topic(topic_id)[:10] if word and word.strip()] 
            if len(words) >= 5:
                octis_topics.append(words)

    with open(f"bertopic_topics_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    # Assignments CSV (using new_topics)
    df_assignments = df_golden[['doc_id']].copy().reset_index(drop=True)
    df_assignments['topic_id'] = new_topics
    
    topic_labels = {tid: "_".join([w for w, _ in topic_model.get_topic(tid)[:5]]) 
                    for tid in topic_info.Topic if tid != -1}
    topic_labels[-1] = "noise" # Should be empty now
    
    df_assignments['topic_label'] = df_assignments['topic_id'].map(topic_labels)
    df_assignments.to_csv(f"bertopic_assignments_{run_name}.csv", index=False)
    print(f"Done. Reassigned noise and saved {len(octis_topics)} topics.")

# Execute experiments

K_VALUES = [10, 20, 30, 40, 50, 60]

for K in K_VALUES:
    run_bertopic_with_reassignment(heavy_texts, embeddings_light, f"LIGHT_EMBfinal_reassignment_fullK{K}", K)
    run_bertopic_with_reassignment(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal_reassignment_fullK{K}", K)

    run_bertopic_with_reassignment(heavy_texts, embeddings_light, f"LIGHT_EMBfinal2_reassignment_fullK{K}", K)
    run_bertopic_with_reassignment(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal2_reassignment_fullK{K}", K)
    run_bertopic_with_reassignment(heavy_texts, embeddings_light, f"LIGHT_EMBfinal3_reassignment_fullK{K}", K)
    run_bertopic_with_reassignment(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal3_reassignment_fullK{K}", K)


# run_bertopic_with_reassignment(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal_reassignment_fullK{60}", 60)