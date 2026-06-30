"""
Dependencies:
    pip install contextualized-topic-models octis
"""

import pandas as pd
import numpy as np
import json
import unicodedata
import string

from gensim.corpora import Dictionary
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from sklearn.feature_extraction.text import CountVectorizer

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


# ── Load data & dictionary ───

print("Loading Golden Sample and Dictionary...")
df_golden = pd.read_parquet("riksarkivet_final_neural_sample_150k.parquet")
dictionary = Dictionary.load("riksarkivet_final_bow_sample_dictionary_20.dict") # Match your saved filename!
valid_words = set(dictionary.token2id.keys())

# ── Reconstruct heavy_texts (identical to BERTopic script) ────

print("Reconstructing heavy_texts...")
remove_punct = str.maketrans('', '', string.punctuation)
light_texts = df_golden['clean_text'].fillna("").tolist()
heavy_texts = []

for text in light_texts:
    clean_str = unicodedata.normalize('NFC', str(text).lower())
    clean_str = clean_str.translate(remove_punct)
    words = [w for w in clean_str.split() if w in valid_words]
    heavy_texts.append(" ".join(words))

# ── Load pre-computed embeddings ──────────────────────────────────────────
# CTM's TopicModelDataPreparation normally calls a SBERT model to produce
# embeddings. We bypass this by injecting pre-computed numpy arrays
# directly, keeping the experiment fully controlled.

print("Loading Neural Embeddings...")
embeddings_light = np.load("riksarkivet_final_embeddings_LIGHT.npy")
embeddings_heavy = np.load("riksarkivet_final_embeddings_HEAVY.npy")

# ── build CTM dataset with injected embeddings ────────────────────


def build_ctm_dataset(heavy_texts, embeddings, max_vocab=2000):
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b",
        max_features=max_vocab,
        lowercase=True,
    )
    bow_matrix = vectorizer.fit_transform(heavy_texts)

    vocab = vectorizer.get_feature_names_out()
    id2token = {i: w for i, w in enumerate(vocab)}

    print(f"  Capped BoW vocab size: {len(vocab)}")

    tp = TopicModelDataPreparation()
    dataset = tp.load(
        contextualized_embeddings=embeddings.astype(np.float32),
        bow_embeddings=bow_matrix,
        id2token=id2token,
    )

    # Return vocab size directly from vectorizer, not tp.vocab
    return dataset, len(vocab)


def run_ctm(heavy_texts, embeddings, run_name, K, n_epochs=100):
    print(f"\n--- Running CTM: {run_name} ---")

    dataset, bow_size = build_ctm_dataset(heavy_texts, embeddings)  # unpack bow_size here

    contextual_size = embeddings.shape[1]

    print(f"  BoW vocab size:    {bow_size}")
    print(f"  Embedding dim:     {contextual_size}")
    print(f"  Num topics (K):    {K}")

    model = CombinedTM(
        bow_size=bow_size,         
        contextual_size=contextual_size,
        n_components=K,
        num_epochs=n_epochs,
        hidden_sizes=(100, 100),
        activation='softplus',
        dropout=0.2,
        learn_priors=True,
        batch_size=200,
        lr=2e-3,
        momentum=0.99,
        solver='adam',
        reduce_on_plateau=False,
        num_data_loader_workers=0,
    )

    start = time.time()
    model.fit(dataset)
    log_timing("CTM", run_name, K, time.time() - start)

    # ── Export in OCTIS format ───────────────
    # get_topic_lists returns a list of lists of (word, score) tuples.
    # We take the top 10 words per topic, matching BERTopic export logic.

    octis_topics = []
    topic_lists = model.get_topic_lists(10)   # top-10 words per topic

    for topic_words in topic_lists:
        # topic_words is already a list of strings in recent CTM versions.
        if topic_words and isinstance(topic_words[0], tuple):
            words = [w for w, _ in topic_words]
        else:
            words = list(topic_words)

        words = [w for w in words if w and w.strip()]  # drop empty strings

        if len(words) >= 5:   # skip degenerate topics (matches BERTopic filter) (but later I realised that this is unnecessary, as evaluation already removes degenerate topics)
            octis_topics.append(words)

    out_path = f"ctm_topics_{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    # Get document-topic distributions (shape: n_docs x n_topics)
    topic_distributions = model.get_thetas(dataset)  # returns numpy array
    doc_topics = np.argmax(topic_distributions, axis=1)  # most probable topic per doc
    
    df_assignments = df_golden[['doc_id']].copy().reset_index(drop=True)
    df_assignments['topic_id'] = doc_topics
    
    topic_labels = {}
    for i, topic_words in enumerate(topic_lists):
        if isinstance(topic_words[0], tuple):
            words = [w for w, _ in topic_words[:5]]
        else:
            words = list(topic_words[:5])
        topic_labels[i] = "_".join(words)
    
    df_assignments['topic_label'] = df_assignments['topic_id'].map(topic_labels)
    df_assignments.to_csv(f"ctm_assignments_{run_name}.csv", index=False)
    print(f"Saved assignments to ctm_assignments_{run_name}.csv")

    print(f"Done. Saved {len(octis_topics)} topics to {out_path}.")

    # save the trained model 
    model.save(models_dir=f"ctm_model_{run_name}")

    return octis_topics


# ──  Execute experiments ──────────

K_VALUES = [10, 20, 30, 40, 50, 60]

for K in K_VALUES:

    run_ctm(heavy_texts, embeddings_light, f"LIGHT_EMBfinal_K{K}", K)
    run_ctm(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal_K{K}", K)
    run_ctm(heavy_texts, embeddings_light, f"LIGHT_EMBfinal2_K{K}", K)
    run_ctm(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal2_K{K}", K)
    run_ctm(heavy_texts, embeddings_light, f"LIGHT_EMBfinal3_K{K}", K)
    run_ctm(heavy_texts, embeddings_heavy, f"HEAVY_EMBfinal3_K{K}", K)



