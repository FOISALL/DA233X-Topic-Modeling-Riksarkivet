

import os
import re
import time
import json
import string
import unicodedata

import pandas as pd
import numpy as np

from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation


# ── Settings ───────────────────

TIMING_LOG = "2Mrun_model_timing_log.csv"

INPUT_PARQUET = "riksarkivet_neural_cleaned.parquet"
DICTIONARY_PATH = "riksarkivet_2M_bow_dictionary_20.dict"

# Choose ONE of these depending on which CTM version you want to run.
# Because of the longer runtimes, I did not want to combine them, so I ran the seperately for the different embeddings

# For CTM D1 / Light:
EMBEDDINGS_PATH = "riksarkivet_2M_embeddings_LIGHT.npy"
EMBEDDING_LABEL = "LIGHT"

# For CTM D2 / Heavy, comment the two lines above and uncomment these:
# EMBEDDINGS_PATH = "riksarkivet_2M_embeddings_HEAVY.npy"
# EMBEDDING_LABEL = "HEAVY"

FIT_SIZE = 150_000
INFER_CHUNK_SIZE = 50_000

K = 20 # I only ran it for the best K
RUN_NAME = f"CTM_{EMBEDDING_LABEL}_2M_K{K}_fit150k_transform_all"

MAX_VOCAB = 2000
N_EPOCHS = 100
BATCH_SIZE = 64


# ── Timing helper ───────────────────────────────────────────────────────────

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


# ── Load full corpus and dictionary ─────────────────────────────────────────

print("Loading full corpus and dictionary...")

df_golden = pd.read_parquet(INPUT_PARQUET)
dictionary = Dictionary.load(DICTIONARY_PATH)
valid_words = set(dictionary.token2id.keys())

print(f"Documents in parquet: {len(df_golden)}")
print(f"Dictionary size: {len(dictionary)}")


# ── Reconstruct heavy_texts ─────────────────────────────────────────────────

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


# ── Load full embeddings ────────────────────────────────────────────────────

print(f"Loading full {EMBEDDING_LABEL} embeddings...")

embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

print(f"Embeddings shape: {embeddings.shape}")

assert len(heavy_texts) == embeddings.shape[0], (
    f"Mismatch: {len(heavy_texts)} texts but {embeddings.shape[0]} embeddings"
)


# ── Helper functions ────────────────────────────────────────────────────────

def clean_embeddings(x):
    """
    This prevents crashes during inference.
    """
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fit_ctm_vectorizer(fit_texts, max_vocab=2000):
    """
    Fit CountVectorizer only once on the training subset.
    """
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b[a-zåäöA-ZÅÄÖ]{3,}\b",
        max_features=max_vocab,
        lowercase=True,
    )

    bow_matrix = vectorizer.fit_transform(fit_texts)

    vocab = vectorizer.get_feature_names_out()
    id2token = {i: w for i, w in enumerate(vocab)}

    print(f"  Capped BoW vocab size: {len(vocab)}")

    return vectorizer, bow_matrix, id2token, len(vocab)


def build_ctm_dataset_from_bow(bow_matrix, contextual_embeddings, id2token):
    """
    Build a CTM dataset using already-created BoW matrix and precomputed embeddings.
    """
    tp = TopicModelDataPreparation()

    dataset = tp.load(
        contextualized_embeddings=clean_embeddings(contextual_embeddings),
        bow_embeddings=bow_matrix,
        id2token=id2token,
    )

    return dataset


def export_topics(model, run_name):
    """
    Save CTM topics in OCTIS-compatible JSON format.
    """
    topic_lists = model.get_topic_lists(10)

    octis_topics = []

    for topic_words in topic_lists:
        if topic_words and isinstance(topic_words[0], tuple):
            words = [w for w, _ in topic_words]
        else:
            words = list(topic_words)

        words = [w for w in words if w and str(w).strip()]

        if len(words) >= 5:
            octis_topics.append(words)

    out_path = f"ctm_topics_{run_name}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(octis_topics, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(octis_topics)} topics to {out_path}")

    return topic_lists, octis_topics


def save_full_assignments(df_golden, all_doc_topics, topic_lists, run_name):
    """
    Save full-corpus document-topic assignments.
    """
    if "doc_id" in df_golden.columns:
        df_assignments = df_golden[["doc_id"]].copy().reset_index(drop=True)
    else:
        df_assignments = pd.DataFrame({
            "doc_id": np.arange(len(df_golden))
        })

    df_assignments["topic_id"] = all_doc_topics

    topic_labels = {}

    for i, topic_words in enumerate(topic_lists):
        if topic_words and isinstance(topic_words[0], tuple):
            words = [w for w, _ in topic_words[:5]]
        else:
            words = list(topic_words[:5])

        topic_labels[i] = "_".join([w for w in words if w and str(w).strip()])

    df_assignments["topic_label"] = df_assignments["topic_id"].map(topic_labels)

    out_path = f"ctm_assignments_{run_name}.csv"
    df_assignments.to_csv(out_path, index=False)

    print(f"Saved full-corpus assignments to {out_path}")


# ── Main CTM full-corpus function ───────────────────────────────────────────

def run_ctm_fit_sample_transform_all(texts, embeddings, run_name, K):
    print(f"\n--- Running CTM: {run_name} ---")
    print(f"Fitting on first {FIT_SIZE} documents.")
    print(f"Inferring topics for all {len(texts)} documents.")

    fit_texts = texts[:FIT_SIZE]
    fit_embeddings = embeddings[:FIT_SIZE]

    # Fit vectorizer on only the training subset.
    vectorizer, fit_bow, id2token, bow_size = fit_ctm_vectorizer(
        fit_texts,
        max_vocab=MAX_VOCAB,
    )

    fit_dataset = build_ctm_dataset_from_bow(
        fit_bow,
        fit_embeddings,
        id2token,
    )

    contextual_size = embeddings.shape[1]

    print(f"  BoW vocab size:    {bow_size}")
    print(f"  Embedding dim:     {contextual_size}")
    print(f"  Num topics (K):    {K}")
    print(f"  Epochs:            {N_EPOCHS}")
    print(f"  Batch size:        {BATCH_SIZE}")

    model = CombinedTM(
        bow_size=bow_size,
        contextual_size=contextual_size,
        n_components=K,
        num_epochs=N_EPOCHS,
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        batch_size=BATCH_SIZE,
        lr=2e-3,
        momentum=0.99,
        solver="adam",
        reduce_on_plateau=False,
        num_data_loader_workers=0,
    )

    # ── Fit on 150k subset ──────────────────────────────────────────────────

    start_total = time.time()

    print("\nFitting CTM on sample...")
    start_fit = time.time()
    model.fit(fit_dataset)
    fit_elapsed = time.time() - start_fit

    log_timing("CTM", run_name + "_fit150k", K, fit_elapsed)

    # ── Export topics after fitting ─────────────────────────────────────────

    topic_lists, octis_topics = export_topics(model, run_name)

    # ── Infer topic assignments for full corpus ─────────────────────────────

    print("\nInferring full-corpus topic assignments in chunks...")

    all_doc_topics = []
    n_docs = len(texts)

    start_infer = time.time()

    for start_idx in range(0, n_docs, INFER_CHUNK_SIZE):
        end_idx = min(start_idx + INFER_CHUNK_SIZE, n_docs)

        print(f"Inferring docs {start_idx}–{end_idx}")

        chunk_texts = texts[start_idx:end_idx]
        chunk_embeddings = embeddings[start_idx:end_idx]

        chunk_bow = vectorizer.transform(chunk_texts)

        chunk_dataset = build_ctm_dataset_from_bow(
            chunk_bow,
            chunk_embeddings,
            id2token,
        )

        theta = model.get_thetas(chunk_dataset)
        doc_topics = np.argmax(theta, axis=1)

        all_doc_topics.extend(doc_topics.tolist())

        print(f"  -> Done chunk. Total assigned so far: {len(all_doc_topics)}")

    infer_elapsed = time.time() - start_infer
    total_elapsed = time.time() - start_total

    assert len(all_doc_topics) == n_docs, (
        f"Mismatch: {len(all_doc_topics)} assignments for {n_docs} documents"
    )

    log_timing("CTM", run_name + "_infer_all", K, infer_elapsed)
    log_timing("CTM", run_name + "_total", K, total_elapsed)

    # ── Save assignments and model ──────────────────────────────────────────

    save_full_assignments(df_golden, all_doc_topics, topic_lists, run_name)

    model_dir = f"ctm_model_{run_name}"
    model.save(models_dir=model_dir)
    print(f"Saved CTM model to {model_dir}")

    # ── Summary ─────────────────────────────────────────────────────────────

    counts = pd.Series(all_doc_topics).value_counts().sort_index()

    print("\n--- Full-corpus CTM assignment summary ---")
    print(f"Total docs: {len(all_doc_topics)}")
    print(f"Topics:     {K}")
    print("\nDocuments per topic:")
    print(counts.to_string())

    return model, all_doc_topics, octis_topics


# ── Run one full-corpus CTM experiment ──────────────────────────────────────

model, all_doc_topics, octis_topics = run_ctm_fit_sample_transform_all(
    heavy_texts,
    embeddings,
    RUN_NAME,
    K,
)