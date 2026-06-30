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

LOG_FILE = "evaluation_results_log_2M.txt"
OUTPUT_CSV = "evaluation_metrics_2M_runs.csv"

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


# ── Helper functions ────────────────────────────────────────────────────────

def file_exists(path):
    return os.path.exists(path)


def evaluate_topic_file(json_file_path, run_name, model, variant, K):
    print(f"\n--- Evaluating {run_name} ---")

    if not file_exists(json_file_path):
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

    if dropped_topics > 0:
        print(f"  -> Dropped {dropped_topics} degenerate topics fewer than {TOPK} words.")

    if len(clean_topics) == 0:
        print("  -> ERROR: No valid topics. Score is NaN.")
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


# ──Reconstruct full reference corpus ────────────────────────────────────

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


# ──Initialize OCTIS metrics ─────────────────────────────────────────────

print("\n=== 2. Initializing OCTIS Metrics ===")

npmi_metric = Coherence(
    texts=tokenized_reference_corpus,
    topk=TOPK,
    measure="c_npmi",
)

diversity_metric = TopicDiversity(topk=TOPK)


# ── Define 2M experiments ────────────────────────────────────────────────

experiments = []

# BERTopic full-corpus assignment experiment
experiments.append({
    "model": "BERTopic",
    "variant": "Base D2 fit150k transform all",
    "K": 30,
    "run_name": "BERTopic Base D2 K30 fit150k transform all",
    "file": "bertopic_topics_Best2_2M_K30_fit150k_transform_all.json",
})

# CTM full-corpus assignment experiment
experiments.append({
    "model": "CTM",
    "variant": "Light fit150k transform all",
    "K": 20,
    "run_name": "CTM Light K20 fit150k transform all",
    "file": "ctm_topics_CTM_LIGHT_2M_K20_fit150k_transform_all.json",
})

# If CTM Heavy exists
experiments.append({
    "model": "CTM",
    "variant": "Heavy fit150k transform all",
    "K": 20,
    "run_name": "CTM Heavy K20 fit150k transform all",
    "file": "ctm_topics_CTM_HEAVY_2M_K20_fit150k_transform_all.json",
})

# LDA full-corpus runs
for K in [10, 20, 30, 40, 50, 60]:
    experiments.append({
        "model": "LDA",
        "variant": "Vocab2000 full corpus",
        "K": K,
        "run_name": f"LDA Vocab2000 K{K} full corpus",
        "file": f"lda_topics_LDA_2M_VOCAB2000_K{K}_fullcorpus.json",
    })

# ETM full-corpus run.
experiments.append({
    "model": "ETM",
    "variant": "Vocab2000 full corpus",
    "K": 20,
    "run_name": "ETM Vocab2000 K20 full corpus",
    "file": "etm_topics_ETM_2M_VOCAB2000_K20.json",
})


# ── Check files before evaluation ────────────────────────────────────────

print("\n=== 3. Checking topic files ===")

missing_files = []
existing_files = []

for exp in experiments:
    if file_exists(exp["file"]):
        existing_files.append(exp["file"])
        print(f"OK:      {exp['file']}")
    else:
        missing_files.append(exp["file"])
        print(f"MISSING: {exp['file']}")

print(f"\nExisting files: {len(existing_files)}")
print(f"Missing files:  {len(missing_files)}")

print("\nThe script will evaluate existing files and skip missing files.")


# ── Run evaluations ──────────────────────────────────────────────────────

print("\n=== 4. Running 2M Evaluations ===")

for exp in experiments:
    evaluate_topic_file(
        json_file_path=exp["file"],
        run_name=exp["run_name"],
        model=exp["model"],
        variant=exp["variant"],
        K=exp["K"],
    )


# ── Save summary ─────────────────────────────────────────────────────────

print("\n=== Final 2M Evaluation Summary Table ===")

df_final = pd.DataFrame(results_data)
print(df_final.to_string(index=False))

df_final.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved 2M evaluation summary to: {OUTPUT_CSV}")
print(f"Saved evaluation log to: {LOG_FILE}")

print("\nEvaluation Complete!")