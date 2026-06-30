import os
import json
import sys
import string
import unicodedata
import re

import pandas as pd
from tqdm import tqdm
from gensim.corpora import Dictionary

from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity


# ── Settings ────────────────────────────────────────────────────────────────

INPUT_PARQUET = "riksarkivet_neural_cleaned.parquet"
DICTIONARY_PATH = "riksarkivet_2M_bow_dictionary_20.dict"

LOG_FILE = "evaluation_results_log_2M_bertopic_only.txt"
OUTPUT_CSV = "evaluation_metrics_2M_bertopic_only.csv"

TOPK = 10


# ── Logging setup ───────────────────────────────────────────────────────────

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)
results_data = []


# ── Evaluation helper ───────────────────────────────────────────────────────

def evaluate_topic_file(json_file_path, run_name, model, variant, K):
    print(f"\n--- Evaluating {run_name} ---")

    if not os.path.exists(json_file_path):
        print(f"  -> SKIPPED: File not found: {json_file_path}")
        results_data.append({
            "Model": model,
            "Variant": variant,
            "K": K,
            "Experiment": run_name,
            "Topic_File": json_file_path,
            "NPMI": None,
            "Diversity": None,
            "Valid_Topics": 0,
            "Dropped_Topics": None,
            "Status": "missing_file",
        })
        return

    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_topics = json.load(f)

    clean_topics = []
    dropped_topics = 0

    for topic in raw_topics:
        words = [w for w in topic if w and str(w).strip()]

        if len(words) >= TOPK:
            clean_topics.append(words[:TOPK])
        else:
            dropped_topics += 1

    if len(clean_topics) == 0:
        print("  -> ERROR: No valid topics.")
        results_data.append({
            "Model": model,
            "Variant": variant,
            "K": K,
            "Experiment": run_name,
            "Topic_File": json_file_path,
            "NPMI": None,
            "Diversity": None,
            "Valid_Topics": 0,
            "Dropped_Topics": dropped_topics,
            "Status": "no_valid_topics",
        })
        return

    model_output = {"topics": clean_topics}

    npmi_score = npmi_metric.score(model_output)
    diversity_score = diversity_metric.score(model_output)

    print(f"  -> Topic Coherence (NPMI): {npmi_score:.4f}")
    print(f"  -> Topic Diversity:        {diversity_score:.4f}")
    print(f"  -> Valid topics:           {len(clean_topics)}")

    results_data.append({
        "Model": model,
        "Variant": variant,
        "K": K,
        "Experiment": run_name,
        "Topic_File": json_file_path,
        "NPMI": npmi_score,
        "Diversity": diversity_score,
        "Valid_Topics": len(clean_topics),
        "Dropped_Topics": dropped_topics,
        "Status": "ok",
    })


# ── Reconstruct full reference corpus ───

print("=== 1. Reconstructing the 2M Reference Corpus ===")

df_golden = pd.read_parquet(INPUT_PARQUET)
dictionary = Dictionary.load(DICTIONARY_PATH)

valid_words = set(dictionary.token2id.keys())
remove_punct = str.maketrans("", "", string.punctuation)
remove_nums = re.compile(r"\d+")

light_texts = df_golden["clean_text"].fillna("").tolist()
tokenized_reference_corpus = []

for text in tqdm(light_texts, desc="Tokenizing reference corpus"):
    clean_str = unicodedata.normalize("NFC", str(text).lower())
    clean_str = clean_str.translate(remove_punct)
    clean_str = remove_nums.sub("", clean_str)

    words = [w for w in clean_str.split() if w in valid_words]
    tokenized_reference_corpus.append(words)

print(f"Constructed reference corpus with {len(tokenized_reference_corpus)} documents.")
print(f"Dictionary size: {len(dictionary)}")


# ──Initialize metrics ──────────────────────────────────────────────

print("\n=== 2. Initializing OCTIS Metrics ===")

npmi_metric = Coherence(
    texts=tokenized_reference_corpus,
    topk=TOPK,
    measure="c_npmi",
)

diversity_metric = TopicDiversity(topk=TOPK)


# ── Evaluate only BERTopic 2M runs ──────────────

print("\n=== 3. Evaluating BERTopic 2M runs only ===")

bertopic_experiments = [
    {
        "model": "BERTopic",
        "variant": "Best1 D2 fit150k transform all",
        "K": 30,
        "run_name": "BERTopic Best1 D2 K30 fit150k transform all",
        "file": "bertopic_topics_Best1_2M_K30_fit150k_transform_all.json",
    },
    {
        "model": "BERTopic",
        "variant": "Best2 D2 fit150k transform all",
        "K": 30,
        "run_name": "BERTopic Best2 D2 K30 fit150k transform all",
        "file": "bertopic_topics_Best2_2M_K30_fit150k_transform_all.json",
    },
]

for exp in bertopic_experiments:
    evaluate_topic_file(
        json_file_path=exp["file"],
        run_name=exp["run_name"],
        model=exp["model"],
        variant=exp["variant"],
        K=exp["K"],
    )


# ── Save output ──────────────────────────────────────────────────────────

print("\n=== BERTopic-only 2M Evaluation Summary ===")

df_final = pd.DataFrame(results_data)
print(df_final.to_string(index=False))

df_final.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved BERTopic-only 2M evaluation summary to: {OUTPUT_CSV}")
print(f"Saved BERTopic-only evaluation log to: {LOG_FILE}")
print("\nEvaluation complete!")