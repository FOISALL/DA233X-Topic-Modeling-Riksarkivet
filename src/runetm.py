"""
Dependencies:
    pip install octis
"""

import json
import unicodedata
import string
import numpy as np
import pandas as pd

from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

# OCTIS imports
from octis.models.ETM import ETM
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
dictionary = Dictionary.load("riksarkivet_final_bow_sample_dictionary_20.dict")  # Match your saved filename!
valid_words = set(dictionary.token2id.keys())


# ──Reconstruct heavy_texts (identical to BERTopic and CTM scripts) ────────

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
# OCTIS's Dataset expects a specific internal format.
# We build it manually from heavy_texts using a capped vocabulary.
#
# ETM can handle larger vocabularies than CTM, but we run both 2000 and 6000
# to test vocabulary sensitivity.


def build_octis_etm_dataset(heavy_texts, max_vocab, dataset_dir):
    print(f"\nBuilding vocabulary (max_features={max_vocab})...")

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b",
        max_features=max_vocab,
        lowercase=True,
    )

    bow_matrix = vectorizer.fit_transform(heavy_texts)
    vocab = list(vectorizer.get_feature_names_out())

    print(f"  Vocabulary size: {len(vocab)}")

    # Convert to OCTIS Dataset format.
    # OCTIS Dataset.train_corpus expects a list of tokenized documents.
    tokenized_corpus = [text.split() for text in heavy_texts]

    # Filter tokens to only keep words in the capped vocabulary set.
    vocab_set = set(vocab)
    tokenized_corpus = [
        [w for w in doc if w in vocab_set]
        for doc in tokenized_corpus
    ]

    os.makedirs(dataset_dir, exist_ok=True)

    # Write corpus.tsv — one document per line, partition column = "train"
    print("Writing OCTIS dataset files...")
    written = 0
    with open(os.path.join(dataset_dir, "corpus.tsv"), "w", encoding="utf-8") as f:
        for doc in tokenized_corpus:
            doc_str = " ".join(doc).strip()

            # We use len(doc) to ensure we have at least 2 tokens.
            # This prevents the float/NaN error in OCTIS.
            if len(doc) >= 2:
                f.write(f"{doc_str}\ttrain\n")
                written += 1

    print(f"  Wrote {written} valid documents to corpus.tsv")

    # Write vocabulary.txt — one word per line
    with open(os.path.join(dataset_dir, "vocabulary.txt"), "w", encoding="utf-8") as f:
        for word in vocab:
            clean_word = str(word).strip()
            if clean_word:
                f.write(clean_word + "\n")

    # Now load via the proper OCTIS method.
    octis_dataset = Dataset()
    octis_dataset.load_custom_dataset_from_folder(dataset_dir)

    print(f"  Dataset loaded. Corpus size: {len(octis_dataset.get_corpus())}")

    return octis_dataset


# Build both ETM datasets once.
octis_dataset_2k = build_octis_etm_dataset(
    heavy_texts=heavy_texts,
    max_vocab=2000,
    dataset_dir="octis_etm_dataset_2000",
)

octis_dataset_6k = build_octis_etm_dataset(
    heavy_texts=heavy_texts,
    max_vocab=6000,
    dataset_dir="octis_etm_dataset_6000",
)


# ── Main experiment function ───────────────────────────────────────────────

def run_etm(dataset, run_name, K, n_epochs=100):

    print(f"\n--- Running ETM: {run_name} ---")
    print(f"  Num topics (K):    {K}")
    print(f"  Num epochs:        {n_epochs}")
    print(f"  Vocab size:        {len(dataset.get_vocabulary())}")

    model = ETM(
        num_topics=K,
        num_epochs=n_epochs,
        t_hidden_size=800,        # inference network hidden layer size
        rho_size=300,             # word embedding dimensionality
        embedding_size=300,       # topic embedding dimensionality
        activation='relu',
        dropout=0.5,
        lr=0.005,
        optimizer='adam',
        batch_size=1000,
        train_embeddings=True,
        # num_samples=20,         # MC samples for ELBO estimation; caused crash earlier
        use_partitions=False,     # use all data for training
    )

    # Train the model
    start = time.time()
    output = model.train_model(dataset)
    log_timing("ETM", run_name, K, time.time() - start)

    # Save best topic for each document.
    # OCTIS returns topic-document-matrix with shape (n_topics, n_docs).
    topic_doc_matrix = output.get('topic-document-matrix')
    if topic_doc_matrix is not None:
        doc_topics = np.argmax(topic_doc_matrix, axis=0)  # axis=0 because topics are rows

        # Build topic labels from the topics we already extracted.
        topic_labels = {}
        for i, topic_words in enumerate(output['topics']):
            words = [w for w in topic_words[:5] if w and w.strip()]
            topic_labels[i] = "_".join(words)

        df_assignments = df_golden[['doc_id']].copy().reset_index(drop=True)

        # ETM dataset may have fewer docs than df_golden if empty docs were filtered,
        # so we only assign to the docs that made it into the dataset.
        df_assignments = df_assignments.iloc[:len(doc_topics)].copy()
        df_assignments['topic_id'] = doc_topics
        df_assignments['topic_label'] = df_assignments['topic_id'].map(topic_labels)
        df_assignments.to_csv(f"etm_assignments_{run_name}.csv", index=False)
        print(f"Saved assignments to etm_assignments_{run_name}.csv")
    else:
        print("Warning: topic-document-matrix not available in output.")

    # ── Export in OCTIS format ─────────────────────────────────────────────────
    octis_topics = []
    for topic_words in output['topics']:
        words = [w for w in topic_words if w and w.strip()]
        if len(words) >= 5:
            octis_topics.append(words[:10])  # ensure max 10 words

    out_path = f"etm_topics_{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    print(f"Done. Saved {len(octis_topics)} topics to {out_path}.")

    # save full OCTIS output.
    np.save(
        f"etm_topic_word_matrix_{run_name}.npy",
        output.get('topic-word-matrix', np.array([]))
    )

    return octis_topics


# ── Execute experiments ────────────────────────────────────────────────────
# ETM has no embedding variants like BERTopic and CTM since it learns
# its own embeddings. We run both 2000 and 6000 vocabulary versions.
#

K_VALUES = [10, 20, 30, 40, 50, 60]

for K in K_VALUES:
    # --- ETM with 2000 vocabulary ---
    run_etm(
        octis_dataset_2k,
        f"HEAVY_final2_VOCAB2000_K{K}",
        K,
        n_epochs=100,
    )

    run_etm(
        octis_dataset_2k,
        f"HEAVY_final3_VOCAB2000_K{K}",
        K,
        n_epochs=100,
    )

    # --- ETM with 6000 vocabulary ---
    run_etm(
        octis_dataset_6k,
        f"HEAVY_final2_VOCAB6000_K{K}",
        K,
        n_epochs=100,
    )

    run_etm(
        octis_dataset_6k,
        f"HEAVY_final3_VOCAB6000_K{K}",
        K,
        n_epochs=100,
    )