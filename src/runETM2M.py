
import os
import json
import time
import string
import unicodedata
import re

import numpy as np
import pandas as pd

from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

from octis.models.ETM import ETM
from octis.dataset.dataset import Dataset

"""
I was unable to get this code to fully run on the entire dataset,
but it might still be possible if more work is putt into optimizing the code
"""


# ── Settings ────────────────────────────────────────────────────────────────

TIMING_LOG = "2Mrun_model_timing_log.csv"

INPUT_PARQUET = "riksarkivet_neural_cleaned.parquet"
DICTIONARY_PATH = "riksarkivet_2M_bow_dictionary_20.dict"

MAX_VOCAB = 2000
K = 20
N_EPOCHS = 100

DATASET_DIR = f"octis_etm_2M_dataset_VOCAB{MAX_VOCAB}"
RUN_NAME = f"ETM_2M_VOCAB{MAX_VOCAB}_K{K}"


# ── Timing helper ───────────────────────────────────────────────────────────

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
        row.to_csv(TIMING_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(TIMING_LOG, mode="w", header=True, index=False)

    print(f"  -> Runtime: {elapsed_seconds / 60:.2f} minutes")


# ── Load full corpus and dictionary ──────────────────────────────────────

print("Loading full corpus and dictionary...")

df_golden = pd.read_parquet(INPUT_PARQUET)
dictionary = Dictionary.load(DICTIONARY_PATH)
valid_words = set(dictionary.token2id.keys())

print(f"Documents in parquet: {len(df_golden)}")
print(f"Dictionary size: {len(dictionary)}")


# ── Reconstruct heavy_texts ──────────────────────────────────────────────

print("Reconstructing heavy_texts...")

remove_punct = str.maketrans("", "", string.punctuation)
remove_nums = re.compile(r"\d+")

light_texts = df_golden["clean_text"].fillna("").tolist()
heavy_texts = []

for text in light_texts:
    clean_str = unicodedata.normalize("NFC", str(text).lower())
    clean_str = clean_str.translate(remove_punct)
    clean_str = remove_nums.sub("", clean_str)

    words = [w for w in clean_str.split() if w in valid_words]
    heavy_texts.append(" ".join(words))

print(f"Constructed heavy_texts: {len(heavy_texts)}")


# ──Build OCTIS-compatible ETM dataset ───────────────────────────────────

def build_octis_etm_dataset_full(heavy_texts, max_vocab, dataset_dir):
    """
    Builds an OCTIS dataset for ETM from the full corpus.

    This avoids storing a second huge tokenized_corpus list in memory.
    It fits a CountVectorizer vocabulary, then writes corpus.tsv line by line.
    """

    print(f"\nBuilding vocabulary with max_features={max_vocab}...")

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b",
        max_features=max_vocab,
        lowercase=True,
    )

    vectorizer.fit(heavy_texts)

    vocab = list(vectorizer.get_feature_names_out())
    vocab_set = set(vocab)

    print(f"  Vocabulary size: {len(vocab)}")

    os.makedirs(dataset_dir, exist_ok=True)

    corpus_path = os.path.join(dataset_dir, "corpus.tsv")
    vocab_path = os.path.join(dataset_dir, "vocabulary.txt")
    kept_doc_ids_path = os.path.join(dataset_dir, "kept_doc_ids.txt")

    print("Writing OCTIS corpus.tsv...")
    written = 0
    skipped = 0

    with open(corpus_path, "w", encoding="utf-8") as corpus_file, \
         open(kept_doc_ids_path, "w", encoding="utf-8") as kept_file:

        for idx, text in enumerate(heavy_texts):
            doc = [w for w in text.split() if w in vocab_set]

            # OCTIS can behave badly with empty/very short docs.
            # Keep only docs with at least 2 tokens.
            if len(doc) >= 2:
                corpus_file.write(f"{' '.join(doc)}\ttrain\n")
                kept_file.write(str(idx) + "\n")
                written += 1
            else:
                skipped += 1

            if idx > 0 and idx % 100000 == 0:
                print(f"  Processed {idx:,} docs | written={written:,} | skipped={skipped:,}")

    print(f"  Wrote {written} valid documents to {corpus_path}")
    print(f"  Skipped {skipped} documents with fewer than 2 valid tokens")

    print("Writing vocabulary.txt...")
    with open(vocab_path, "w", encoding="utf-8") as vocab_file:
        for word in vocab:
            clean_word = str(word).strip()
            if clean_word:
                vocab_file.write(clean_word + "\n")

    print("Loading OCTIS dataset...")
    octis_dataset = Dataset()
    octis_dataset.load_custom_dataset_from_folder(dataset_dir)

    print(f"  Dataset loaded. Corpus size: {len(octis_dataset.get_corpus())}")
    print(f"  Vocabulary size: {len(octis_dataset.get_vocabulary())}")

    return octis_dataset, written, skipped, kept_doc_ids_path


octis_dataset, written_docs, skipped_docs, kept_doc_ids_path = build_octis_etm_dataset_full(
    heavy_texts=heavy_texts,
    max_vocab=MAX_VOCAB,
    dataset_dir=DATASET_DIR,
)


# ── Run ETM ──────────────────────────────────────────────────────────────

def run_etm_full(dataset, run_name, K, n_epochs=100):
    print(f"\n--- Running ETM full corpus: {run_name} ---")
    print(f"  Num topics (K):    {K}")
    print(f"  Num epochs:        {n_epochs}")
    print(f"  Vocab size:        {len(dataset.get_vocabulary())}")
    print(f"  Corpus size:       {len(dataset.get_corpus())}")

    model = ETM(
        num_topics=K,
        num_epochs=n_epochs,
        t_hidden_size=800,
        rho_size=300,
        embedding_size=300,
        activation="relu",
        dropout=0.5,
        lr=0.005,
        optimizer="adam",
        batch_size=1000,
        train_embeddings=True,
        use_partitions=False,
    )

    start = time.time()
    output = model.train_model(dataset)
    elapsed = time.time() - start

    log_timing("ETM", run_name, K, elapsed)

    # ── Save topics ─────────────────────────────────────────────────────────

    octis_topics = []

    for topic_words in output["topics"]:
        words = [w for w in topic_words if w and str(w).strip()]
        if len(words) >= 5:
            octis_topics.append(words[:10])

    topics_path = f"etm_topics_{run_name}.json"

    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(octis_topics)} topics to {topics_path}")

    # ── Save topic-word matrix ──────────────────────────────────────────────

    topic_word_matrix = output.get("topic-word-matrix", np.array([]))
    np.save(f"etm_topic_word_matrix_{run_name}.npy", topic_word_matrix)

    # ── Save document-topic assignments ────────────────────────────────────

    topic_doc_matrix = output.get("topic-document-matrix")

    if topic_doc_matrix is None:
        print("Warning: topic-document-matrix not available in ETM output.")
        return octis_topics

    # OCTIS returns shape: n_topics x n_docs
    doc_topics = np.argmax(topic_doc_matrix, axis=0)

    print(f"Topic-document matrix shape: {topic_doc_matrix.shape}")
    print(f"Document assignments: {len(doc_topics)}")

    # Load indices of documents that survived ETM dataset filtering
    with open(kept_doc_ids_path, "r", encoding="utf-8") as f:
        kept_indices = [int(line.strip()) for line in f if line.strip()]

    if len(kept_indices) != len(doc_topics):
        print("Warning: kept_indices and doc_topics length mismatch.")
        print(f"  kept_indices: {len(kept_indices)}")
        print(f"  doc_topics:   {len(doc_topics)}")

        min_len = min(len(kept_indices), len(doc_topics))
        kept_indices = kept_indices[:min_len]
        doc_topics = doc_topics[:min_len]

    topic_labels = {}

    for i, topic_words in enumerate(output["topics"]):
        words = [w for w in topic_words[:5] if w and str(w).strip()]
        topic_labels[i] = "_".join(words)

    if "doc_id" in df_golden.columns:
        df_assignments = df_golden.iloc[kept_indices][["doc_id"]].copy().reset_index(drop=True)
    else:
        df_assignments = pd.DataFrame({
            "doc_id": kept_indices
        })

    df_assignments["topic_id"] = doc_topics
    df_assignments["topic_label"] = df_assignments["topic_id"].map(topic_labels)

    assignments_path = f"etm_assignments_{run_name}.csv"
    df_assignments.to_csv(assignments_path, index=False)

    print(f"Saved assignments to {assignments_path}")

    # ── Assignment summary ─────────────────────────────────────────────────

    counts = pd.Series(doc_topics).value_counts().sort_index()

    print("\n--- Full-corpus ETM assignment summary ---")
    print(f"Original docs:   {len(df_golden)}")
    print(f"Used docs:       {len(doc_topics)}")
    print(f"Skipped docs:    {len(df_golden) - len(doc_topics)}")
    print(f"Topics:          {K}")
    print("\nDocuments per topic:")
    print(counts.to_string())

    return octis_topics


octis_topics = run_etm_full(
    octis_dataset,
    RUN_NAME,
    K,
    n_epochs=N_EPOCHS,
)