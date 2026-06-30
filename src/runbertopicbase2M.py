import pandas as pd
import numpy as np
from bertopic import BERTopic
import json
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
from gensim.corpora import Dictionary
import string
import re
import time
import os

from umap import UMAP
from hdbscan import HDBSCAN


# ── Settings ─────

TIMING_LOG = "2Mrun_model_timing_log.csv"

INPUT_PARQUET = "riksarkivet_neural_cleaned.parquet"
DICTIONARY_PATH = "riksarkivet_2M_bow_dictionary_20.dict"
EMBEDDINGS_PATH = "riksarkivet_2M_embeddings_HEAVY.npy"

FIT_SIZE = 150_000
TRANSFORM_CHUNK_SIZE = 50_000

RUN_NAME = "Best2_2M_K30_fit150k_transform_all"
K = 30


# ── Timing helper ───

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


# ── Vectorizer test ────

test_vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b")
analyze = test_vectorizer.build_analyzer()
result = analyze("Här är hofrätten i år")

print(f"Vectorizer Test: {result}")
# Expected: ['här', 'hofrätten']


# ──  Load full corpus and dictionary ────
print("Loading full corpus and dictionary...")

df_golden = pd.read_parquet(INPUT_PARQUET)
dictionary = Dictionary.load(DICTIONARY_PATH)
valid_words = set(dictionary.token2id.keys())

print(f"Documents in parquet: {len(df_golden)}")
print(f"Dictionary size: {len(dictionary)}")

sw_keys = [k for k in dictionary.token2id.keys() if any(c in k for c in "åäö")]
print(f"Words with åäö in dictionary: {len(sw_keys)}")
print(f"Sample: {sw_keys[:10]}")


# ──  Reconstruct heavy_texts ───

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

sample = [t for t in heavy_texts if any(c in t for c in "åäö")]
print(f"Texts with åäö: {len(sample)} / {len(heavy_texts)}")
if sample:
    print(sample[0][:300])


# ──  Load full embeddings ─────

print("Loading full neural embeddings...")

# mmap_mode avoids loading the whole 5+ GB array into RAM at once.
embeddings_heavy = np.load(EMBEDDINGS_PATH, mmap_mode="r")

print(f"Embeddings shape: {embeddings_heavy.shape}")

assert len(heavy_texts) == embeddings_heavy.shape[0], (
    f"Mismatch: {len(heavy_texts)} texts but {embeddings_heavy.shape[0]} embeddings"
)


# ── Vectorizer, UMAP, HDBSCAN ──

vectorizer_model = CountVectorizer(
    token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b",
    stop_words=None,
    lowercase=True,
    min_df=2,
)
"""
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    low_memory=True,
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
"""

# ──Main BERTopic function ───────────────────────────────────────────────

def run_bertopic_fit_sample_transform_all(texts, embeddings, run_name, K):
    print(f"\n--- Running BERTopic: {run_name} ---")
    print(f"Fitting on first {FIT_SIZE} documents.")
    print(f"Transforming all {len(texts)} documents.")

    fit_texts = texts[:FIT_SIZE]
    fit_embeddings = np.asarray(embeddings[:FIT_SIZE], dtype=np.float32)
    fit_embeddings = np.nan_to_num(fit_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    topic_model = BERTopic(
        nr_topics=K,
        language=None, # very important to avoid removing åäö from all words
        vectorizer_model=vectorizer_model,
        # umap_model=umap_model,
        # hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )

    start_total = time.time()

    # ── Fit on sample ──────────
    print("\nFitting BERTopic on sample...")
    topics_fit, _ = topic_model.fit_transform(
        fit_texts,
        embeddings=fit_embeddings,
    )

    print(f"Finished fitting on sample. Fit topics length: {len(topics_fit)}")

    # ── Transform full corpus in chunks ──────────────────────────
    print("\nTransforming full corpus in chunks...")

    all_topics = []
    n_docs = len(texts)

    for start in range(0, n_docs, TRANSFORM_CHUNK_SIZE):
        end = min(start + TRANSFORM_CHUNK_SIZE, n_docs)

        print(f"Transforming docs {start}–{end}")

        chunk_texts = texts[start:end]
        chunk_embeddings = np.asarray(embeddings[start:end], dtype=np.float32)

        chunk_embeddings = np.nan_to_num(chunk_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        chunk_topics, _ = topic_model.transform(
            chunk_texts,
            embeddings=chunk_embeddings,
        )

        all_topics.extend(chunk_topics)

        print(f"  -> Done chunk. Total assigned so far: {len(all_topics)}")

    assert len(all_topics) == n_docs, (
        f"Mismatch after transform: {len(all_topics)} topics for {n_docs} docs"
    )

    elapsed = time.time() - start_total
    log_timing("BERTopic", run_name, K, elapsed)

    # ── Export topics for OCTIS / inspection ────────
    topic_info = topic_model.get_topic_info()

    octis_topics = []
    for topic_id in topic_info.Topic:
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                words = [
                    word for word, _ in topic_words[:10]
                    if word and str(word).strip()
                ]

                if len(words) >= 5:
                    octis_topics.append(words)

    topics_path = f"bertopic_topics_{run_name}.json"
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(octis_topics)} topics to {topics_path}")

    # ── Export document-topic assignments ──────────────────────────────────
    if "doc_id" in df_golden.columns:
        df_assignments = df_golden[["doc_id"]].copy().reset_index(drop=True)
    else:
        df_assignments = pd.DataFrame({
            "doc_id": np.arange(len(df_golden))
        })

    df_assignments["topic_id"] = all_topics

    topic_labels = {}
    for topic_id in topic_info.Topic:
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                words = [
                    w for w, _ in topic_words[:5]
                    if w and str(w).strip()
                ]
                topic_labels[topic_id] = "_".join(words)
        else:
            topic_labels[-1] = "noise"

    df_assignments["topic_label"] = df_assignments["topic_id"].map(topic_labels)

    assignments_path = f"bertopic_assignments_{run_name}.csv"
    df_assignments.to_csv(assignments_path, index=False)

    print(f"Saved assignments to {assignments_path}")

    # ── Save model ──────────────────────────────────────────────────────────
    model_path = f"bertopic_model_{run_name}"
    topic_model.save(model_path, serialization="pickle")

    print(f"Saved BERTopic model to {model_path}")

    # ── Basic noise summary ─────────────────────────────────────────────────
    noise_docs = sum(1 for t in all_topics if t == -1)
    noise_pct = 100 * noise_docs / len(all_topics)

    print("\n--- Full-corpus assignment summary ---")
    print(f"Total docs:  {len(all_topics)}")
    print(f"Noise docs:  {noise_docs}")
    print(f"Noise %:     {noise_pct:.2f}%")
    print(f"Topics saved: {len(octis_topics)}")

    return topic_model, all_topics, octis_topics


# ── Run the full assignment experiment ───────────────────────────────────

topic_model, all_topics, octis_topics = run_bertopic_fit_sample_transform_all(
    heavy_texts,
    embeddings_heavy,
    RUN_NAME,
    K,
)