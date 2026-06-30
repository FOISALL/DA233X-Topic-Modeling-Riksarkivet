

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

from octis.models.LDA import LDA
from octis.dataset.dataset import Dataset

"""
I had some trouble with getting this to work so I had to use some AI for optimizations

hence, some of the weird error or logging texts
"""


# ── Settings ────────────────────────────────────────────────────────────────

TIMING_LOG = "2Mrun_model_timing_log.csv"

INPUT_PARQUET = "riksarkivet_neural_cleaned.parquet"
DICTIONARY_PATH = "riksarkivet_2M_bow_dictionary_20.dict"

MAX_VOCAB = 2000
DATASET_DIR = f"octis_lda_2M_dataset_VOCAB{MAX_VOCAB}"

# Run all values like before.
# If you want to test quickly first, change this to: K_VALUES = [30]
K_VALUES = [10, 20, 30, 40, 50, 60]

RUN_PREFIX = f"LDA_2M_VOCAB{MAX_VOCAB}"


# ── Safety helpers ──────────────────────────────────────────────────────────

def refuse_overwrite(path):
    if os.path.exists(path):
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}\n"
            "Rename RUN_PREFIX or delete the old file if you really want to replace it."
        )


def log_timing(model_name, run_name, K, elapsed_seconds):
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


# ── Build or load OCTIS-compatible LDA dataset ───────────────────────────

def build_octis_lda_dataset_full(heavy_texts, max_vocab, dataset_dir):


    corpus_path = os.path.join(dataset_dir, "corpus.tsv")
    vocab_path = os.path.join(dataset_dir, "vocabulary.txt")
    kept_doc_ids_path = os.path.join(dataset_dir, "kept_doc_ids.txt")

    # If dataset already exists, reuse it instead of overwriting.
    if os.path.exists(corpus_path) and os.path.exists(vocab_path) and os.path.exists(kept_doc_ids_path):
        print(f"\nFound existing OCTIS dataset folder: {dataset_dir}")
        print("Reusing existing corpus.tsv, vocabulary.txt, and kept_doc_ids.txt.")

        octis_dataset = Dataset()
        octis_dataset.load_custom_dataset_from_folder(dataset_dir)

        print(f"  Dataset loaded. Corpus size: {len(octis_dataset.get_corpus())}")
        print(f"  Vocabulary size: {len(octis_dataset.get_vocabulary())}")

        return octis_dataset, kept_doc_ids_path

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

    print("Writing OCTIS corpus.tsv...")
    written = 0
    skipped = 0

    with open(corpus_path, "w", encoding="utf-8") as corpus_file, \
         open(kept_doc_ids_path, "w", encoding="utf-8") as kept_file:

        for idx, text in enumerate(heavy_texts):
            doc = [w for w in text.split() if w in vocab_set]

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

    return octis_dataset, kept_doc_ids_path


octis_dataset, kept_doc_ids_path = build_octis_lda_dataset_full(
    heavy_texts=heavy_texts,
    max_vocab=MAX_VOCAB,
    dataset_dir=DATASET_DIR,
)


# ── Run LDA ──────────────────────────────────────────────────────────────

def run_lda_full(dataset, run_name, K):
    print(f"\n--- Running LDA full corpus: {run_name} ---")
    print(f"  Num topics (K): {K}")
    print(f"  Vocab size:     {len(dataset.get_vocabulary())}")
    print(f"  Corpus size:    {len(dataset.get_corpus())}")

    topics_path = f"lda_topics_{run_name}.json"
    assignments_path = f"lda_assignments_{run_name}.csv"
    topic_word_path = f"lda_topic_word_matrix_{run_name}.npy"

    refuse_overwrite(topics_path)
    refuse_overwrite(assignments_path)
    refuse_overwrite(topic_word_path)

    model = LDA(
        num_topics=K,
        alpha="symmetric",
        eta=None,
        passes=10,
        iterations=50,
        chunksize=2000,
        random_state=42,
    )

    model.partitioning(use_partitions=False)

    start = time.time()
    output = model.train_model(dataset)
    elapsed = time.time() - start

    log_timing("LDA", run_name, K, elapsed)

    # ── Save topics ─────────────────────────────────────────────────────────

    octis_topics = []

    for topic_words in output["topics"]:
        words = [w for w in topic_words if w and str(w).strip()]
        if len(words) >= 5:
            octis_topics.append(words[:10])

    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(octis_topics)} topics to {topics_path}")

    # ── Save topic-word matrix if available ─────────────────────────────────

    topic_word_matrix = output.get("topic-word-matrix", np.array([]))
    np.save(topic_word_path, topic_word_matrix)
    print(f"Saved topic-word matrix to {topic_word_path}")

    # ── Save document-topic assignments ────────────────────────────────────

    topic_doc_matrix = output.get("topic-document-matrix")

    if topic_doc_matrix is None:
        print("Warning: topic-document-matrix not available in LDA output.")
        return octis_topics

    # OCTIS returns shape: n_topics x n_docs
    doc_topics = np.argmax(topic_doc_matrix, axis=0)

    print(f"Topic-document matrix shape: {topic_doc_matrix.shape}")
    print(f"Document assignments: {len(doc_topics)}")

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

    df_assignments.to_csv(assignments_path, index=False)
    print(f"Saved assignments to {assignments_path}")

    # ── Assignment summary ─────────────────────────────────────────────────

    counts = pd.Series(doc_topics).value_counts().sort_index()

    print("\n--- Full-corpus LDA assignment summary ---")
    print(f"Original docs:   {len(df_golden)}")
    print(f"Used docs:       {len(doc_topics)}")
    print(f"Skipped docs:    {len(df_golden) - len(doc_topics)}")
    print(f"Topics:          {K}")
    print("\nDocuments per topic:")
    print(counts.to_string())

    return octis_topics


# ── Execute experiments ──────────────────────────────────────────────────

for K in K_VALUES:
    run_name = f"{RUN_PREFIX}_K{K}_fullcorpus"

    run_lda_full(
        octis_dataset,
        run_name,
        K,
    )

print("\nAll full-corpus LDA runs finished.")